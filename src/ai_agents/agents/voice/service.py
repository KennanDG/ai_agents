from __future__ import annotations

import base64
import io
import uuid

from groq import Groq

from ai_agents.agents.voice.graph import build_voice_agent_graph
from ai_agents.config.settings import settings


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



    def synthesize_reply(self, text: str) -> tuple[str | None, str | None]:
        if not settings.voice_tts_enabled:
            return None, None, None

        try:
            response = self.client.audio.speech.create(
                model=settings.voice_tts_model,
                voice=settings.voice_tts_voice,
                input=text,
                response_format="wav",
            )

            if hasattr(response, "read"):
                audio_bytes = response.read()
            elif hasattr(response, "content"):
                audio_bytes = response.content
            elif isinstance(response, bytes):
                audio_bytes = response
            else:
                return None, None, "TTS response did not contain readable audio bytes."

            return "audio/wav", base64.b64encode(audio_bytes).decode("ascii"), None

        except Exception as exc:
            return None, None, f"TTS failed: {exc}"



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