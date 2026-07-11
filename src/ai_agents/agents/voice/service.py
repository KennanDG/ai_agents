from __future__ import annotations

import base64
import io
import logging
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from groq import Groq

from ai_agents.agents.voice.graph import build_voice_agent_graph
from ai_agents.config.settings import settings

logger = logging.getLogger(__name__)


class VoiceAgentService:
    def __init__(self) -> None:
        api_key = settings.resolved_groq_api_key()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for the voice agent.")

        self.client = Groq(api_key=api_key)
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

        result = self.client.audio.transcriptions.create(
            model=settings.voice_stt_model,
            file=(file_obj.name, file_obj, content_type or "audio/webm"),
            response_format="json",
        )

        text = getattr(result, "text", None)
        return text.strip() if isinstance(text, str) else str(result).strip()

    
    
    @staticmethod
    def _tts_input(text: str) -> str:
        normalized = " ".join(text.split()).strip()
        max_chars = max(1, settings.voice_tts_max_chars)

        if len(normalized) <= max_chars:
            return normalized

        shortened = normalized[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")

        return shortened or normalized[:max_chars]

    
    
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

    
    
    
    def synthesize_reply(
        self,
        text: str,
    ) -> tuple[str | None, str | None, str | None]:
        if not settings.voice_tts_enabled:
            return None, None, None

        tts_input = self._tts_input(text)
        
        if not tts_input:
            return None, None, "TTS was skipped because the reply text was empty."

        try:
            response = self.client.audio.speech.create(
                model=settings.voice_tts_model,
                voice=settings.voice_tts_voice,
                input=tts_input,
                response_format="wav",
            )

            audio_bytes = self._read_audio_bytes(response)
            if len(audio_bytes) < 44:
                return None, None, "TTS returned an empty or invalid WAV payload."

            return (
                "audio/wav",
                base64.b64encode(audio_bytes).decode("ascii"),
                None,
            )

        except Exception as exc:
            logger.exception("Voice TTS generation failed")

            error_text = str(exc)

            if "model_terms_required" in error_text:
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
        repo_root: str | None,
        workspace_root: str | None,
        active_path: str | None,
        allow_write: bool,
    ) -> dict:
        resolved_session_id = session_id or str(uuid.uuid4())

        transcript = self.transcribe_audio(
            filename=filename,
            audio_bytes=audio_bytes,
            content_type=content_type,
        )

        state = self.graph.invoke(
            {
                "session_id": resolved_session_id,
                "transcript": transcript,
                "history": history,
                "repo_root": repo_root,
                "workspace_root": workspace_root,
                "active_path": active_path,
                "allow_write": allow_write,
                "errors": [],
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
