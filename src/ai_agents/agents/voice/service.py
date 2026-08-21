from __future__ import annotations

import base64
import io
import logging
import uuid
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ai_agents.agents.coding.model_factory import caption_model
from ai_agents.agents.voice.graph import build_voice_agent_graph
from ai_agents.agents.voice.provider_clients import build_audio_client
from ai_agents.config.settings import settings

logger = logging.getLogger(__name__)

# Groq Orpheus currently accepts at most 200 characters per speech request.
GROQ_ORPHEUS_MAX_INPUT_CHARS = 200
DEFAULT_TTS_MAX_CHUNKS = 20
MAX_VOICE_IMAGE_CAPTION_CHARS = 10_000


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part.strip() for part in parts if part.strip()).strip()

    return str(content).strip()


def _caption_uploaded_image(*, name: str, mime_type: str, data_url: str) -> str:
    model = caption_model()
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You convert a user-attached image into concise text context for a voice "
                    "intake agent that is preparing work for a coding agent. Focus on visible UI, "
                    "screenshots, diagrams, error messages, labels, tables, layout, code snippets, "
                    "technology names, and implementation-relevant details. Do not guess hidden data."
                )
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Describe this image for downstream requirement synthesis. "
                            f"File name: {name}. MIME type: {mime_type}."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ]
            ),
        ]
    )

    caption = _message_content_to_text(response.content)
    if not caption:
        raise RuntimeError("vision model returned an empty image caption.")

    return caption[:MAX_VOICE_IMAGE_CAPTION_CHARS]


class VoiceAgentService:
    def __init__(self) -> None:
        # Keep STT and TTS independently routable. The runtime configuration store
        # clears the cached voice service after settings change, so new turns rebuild
        # these clients with the selected providers and current credentials.
        self.stt_provider = getattr(settings, "voice_stt_provider", "groq")
        self.tts_provider = getattr(settings, "voice_tts_provider", "groq")
        self.stt_client = build_audio_client(self.stt_provider)
        self.tts_client = build_audio_client(self.tts_provider)
        self.graph = build_voice_agent_graph()

    def transcribe_audio(
        self,
        *,
        filename: str,
        audio_bytes: bytes,
        content_type: str | None,
    ) -> str:
        file_obj = io.BytesIO(audio_bytes)
        file_obj.name = filename or "voice-input.webm"

        result = self.stt_client.audio.transcriptions.create(
            model=settings.voice_stt_model,
            file=(file_obj.name, file_obj, content_type or "audio/webm"),
            response_format="json",
        )

        text = getattr(result, "text", None)
        return text.strip() if isinstance(text, str) else str(result).strip()

    @staticmethod
    def _tts_chunk_size() -> int:
        configured = getattr(
            settings,
            "voice_tts_max_chars",
            GROQ_ORPHEUS_MAX_INPUT_CHARS,
        )

        try:
            configured_value = max(1, int(configured))
        except (TypeError, ValueError):
            configured_value = GROQ_ORPHEUS_MAX_INPUT_CHARS

        # The 200-character request ceiling belongs to Groq's Orpheus models.
        # Other providers use the administrator-configured limit without this cap.
        if getattr(settings, "voice_tts_provider", "groq") == "groq":
            return min(GROQ_ORPHEUS_MAX_INPUT_CHARS, configured_value)

        return configured_value

    @classmethod
    def _tts_inputs(cls, text: str) -> list[str]:
        """Split a reply into complete TTS requests instead of truncating it."""
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return []

        max_chars = cls._tts_chunk_size()
        chunks: list[str] = []
        remaining = normalized

        while remaining:
            if len(remaining) <= max_chars:
                chunks.append(remaining)
                break

            window = remaining[: max_chars + 1]

            # Prefer a sentence boundary so each generated clip ends naturally.
            sentence_cut = max(
                window.rfind(". "),
                window.rfind("! "),
                window.rfind("? "),
            )
            cut = sentence_cut + 1 if sentence_cut >= max_chars // 2 else -1

            # Otherwise split at the last available word boundary.
            if cut <= 0:
                cut = window.rfind(" ", 0, max_chars + 1)

            # A single unusually long token still needs a hard split.
            if cut <= 0:
                cut = max_chars

            chunk = remaining[:cut].strip()
            if not chunk:
                chunk = remaining[:max_chars]
                cut = max_chars

            chunks.append(chunk)
            remaining = remaining[cut:].lstrip()

        return chunks

    @staticmethod
    def _read_audio_bytes(response: object) -> bytes:
        if isinstance(response, bytes):
            return response

        read = getattr(response, "read", None)
        if callable(read):
            value = read()
            if isinstance(value, (bytes, bytearray, memoryview)):
                return bytes(value)

        content = getattr(response, "content", None)
        if isinstance(content, (bytes, bytearray, memoryview)):
            return bytes(content)

        write_to_file = getattr(response, "write_to_file", None)
        if callable(write_to_file):
            with TemporaryDirectory() as directory:
                output_path = Path(directory) / "voice-reply.wav"
                write_to_file(output_path)
                return output_path.read_bytes()

        raise TypeError("TTS response did not contain readable audio bytes.")

    @staticmethod
    def _merge_wav_chunks(chunks: list[bytes]) -> bytes:
        """Combine WAV frame data under one valid header.

        Raw WAV byte concatenation is invalid because every chunk has its own header;
        many browsers stop at the first header, recreating the original cutoff symptom.
        """
        if not chunks:
            raise ValueError("No WAV chunks were provided.")
        if len(chunks) == 1:
            return chunks[0]

        expected_format: tuple[int, int, int, str] | None = None
        first_params: Any | None = None
        frame_parts: list[bytes] = []

        for index, audio_bytes in enumerate(chunks, start=1):
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
                    params = reader.getparams()
                    current_format = (
                        params.nchannels,
                        params.sampwidth,
                        params.framerate,
                        params.comptype,
                    )

                    if expected_format is None:
                        expected_format = current_format
                        first_params = params
                    elif current_format != expected_format:
                        raise ValueError(
                            "TTS WAV chunk formats do not match: "
                            f"chunk 1={expected_format}, chunk {index}={current_format}."
                        )

                    frame_parts.append(reader.readframes(params.nframes))
            except wave.Error as exc:
                raise ValueError(f"TTS chunk {index} is not a valid WAV file: {exc}") from exc

        if first_params is None:
            raise ValueError("TTS WAV chunks did not contain readable parameters.")

        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(first_params.nchannels)
            writer.setsampwidth(first_params.sampwidth)
            writer.setframerate(first_params.framerate)
            writer.setcomptype(first_params.comptype, first_params.compname)
            writer.writeframes(b"".join(frame_parts))

        return output.getvalue()

    def prepare_voice_attachments(
        self,
        attached_files: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Caption uploaded images, then strip raw image data before LangGraph state."""
        prepared: list[dict[str, Any]] = []
        errors: list[str] = []

        for attached in attached_files:
            item = dict(attached)
            data_url = item.pop("data_url", None)
            image_data_error = item.pop("image_data_error", None)

            if isinstance(image_data_error, str) and image_data_error.strip():
                errors.append(image_data_error.strip())

            if isinstance(data_url, str) and data_url:
                name = str(item.get("name") or "image")
                mime_type = str(item.get("mime_type") or "image/png")

                try:
                    caption = _caption_uploaded_image(
                        name=name,
                        mime_type=mime_type,
                        data_url=data_url,
                    )
                    item["image_caption"] = caption
                    item["content"] = (
                        f"Vision-generated caption for {name}:\n{caption}"
                    )
                    item["has_image_data"] = True
                except Exception as exc:
                    errors.append(
                        f"Could not caption image attachment {name}: {exc}"
                    )

            prepared.append(item)

        return prepared, errors

    def synthesize_reply(
        self,
        text: str,
    ) -> tuple[str | None, str | None, str | None]:
        if not settings.voice_tts_enabled:
            return None, None, None

        tts_inputs = self._tts_inputs(text)
        if not tts_inputs:
            return None, None, "TTS was skipped because the reply text was empty."

        max_chunks_value = getattr(settings, "voice_tts_max_chunks", DEFAULT_TTS_MAX_CHUNKS)
        try:
            max_chunks = max(1, int(max_chunks_value))
        except (TypeError, ValueError):
            max_chunks = DEFAULT_TTS_MAX_CHUNKS

        if len(tts_inputs) > max_chunks:
            return (
                None,
                None,
                "TTS was skipped because the reply requires "
                f"{len(tts_inputs)} chunks, exceeding the configured limit of {max_chunks}.",
            )

        try:
            audio_chunks: list[bytes] = []

            for index, tts_input in enumerate(tts_inputs, start=1):
                logger.debug(
                    "Generating voice TTS chunk %s/%s (%s characters)",
                    index,
                    len(tts_inputs),
                    len(tts_input),
                )

                response = self.tts_client.audio.speech.create(
                    model=settings.voice_tts_model,
                    voice=settings.voice_tts_voice,
                    input=tts_input,
                    response_format="wav",
                )

                audio_bytes = self._read_audio_bytes(response)
                if len(audio_bytes) < 44:
                    raise ValueError(
                        f"TTS chunk {index} returned an empty or invalid WAV payload."
                    )

                audio_chunks.append(audio_bytes)

            merged_audio = self._merge_wav_chunks(audio_chunks)

            logger.info(
                "Generated complete voice reply: chars=%s chunks=%s wav_bytes=%s",
                len(" ".join(text.split()).strip()),
                len(tts_inputs),
                len(merged_audio),
            )

            return (
                "audio/wav",
                base64.b64encode(merged_audio).decode("ascii"),
                None,
            )

        except Exception as exc:
            logger.exception("Voice TTS generation failed")

            error_text = str(exc)

            if self.tts_provider == "groq" and "model_terms_required" in error_text:
                return (
                    None,
                    None,
                    "TTS is unavailable because the Groq organization that "
                    f"owns GROQ_API_KEY has not accepted the terms for "
                    f"{settings.voice_tts_model}. An organization administrator "
                    "must accept the model terms in Groq Console.",
                )

            return None, None, f"TTS failed: {error_text}"

    def run_turn(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None,
        session_id: str | None,
        history: list[dict[str, str]],
        prompt_text: str,
        attached_files: list[dict[str, Any]],
        repo_root: str | None,
        workspace_root: str | None,
        active_path: str | None,
        allow_write: bool,
    ) -> dict[str, Any]:
        resolved_session_id = session_id or str(uuid.uuid4())

        transcript = self.transcribe_audio(
            filename=filename,
            audio_bytes=audio_bytes,
            content_type=content_type,
        )

        prepared_attached_files, attachment_errors = self.prepare_voice_attachments(
            attached_files
        )

        state = self.graph.invoke(
            {
                "session_id": resolved_session_id,
                "transcript": transcript,
                "prompt_text": prompt_text,
                "history": history,
                "repo_root": repo_root,
                "workspace_root": workspace_root,
                "active_path": active_path,
                "allow_write": allow_write,
                "attached_files": prepared_attached_files,
                "errors": attachment_errors,
            }
        )

        reply_text = state.get("reply_text") or "I heard you, but I need you to repeat that."
        audio_mime_type, audio_base64, tts_error = self.synthesize_reply(reply_text)

        errors = list(state.get("errors", []))
        if tts_error:
            errors.append(tts_error)

        return {
            "session_id": resolved_session_id,
            "transcript": transcript,
            "reply_text": reply_text,
            "status": state.get("status", "clarifying"),
            "coding_request": state.get("coding_request"),
            "audio_mime_type": audio_mime_type,
            "audio_base64": audio_base64,
            "errors": errors,
        }
