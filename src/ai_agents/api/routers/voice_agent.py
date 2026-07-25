from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ai_agents.agents.voice.service import VoiceAgentService
from ai_agents.api.schemas import VoiceAgentTurnResponse


router = APIRouter(prefix="/voice-agent", tags=["voice-agent"])

MAX_VOICE_ATTACHMENTS = 5
MAX_VOICE_ATTACHMENT_CONTENT_CHARS = 20_000
MAX_TOTAL_VOICE_ATTACHMENT_CONTENT_CHARS = 60_000


@lru_cache(maxsize=1)
def get_voice_service() -> VoiceAgentService:
    return VoiceAgentService()


def _parse_history(history_json: str | None) -> list[dict[str, str]]:
    if not history_json:
        return []

    try:
        parsed = json.loads(history_json)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    history: list[dict[str, str]] = []

    for item in parsed[-12:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant", "system"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        history.append({"role": role, "content": content[:4_000]})

    return history


def _optional_text(value: Any, *, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    if not normalized:
        return None

    return normalized[:max_chars]


def _parse_attached_files(attached_files_json: str | None) -> list[dict[str, Any]]:
    """Parse a bounded, voice-safe attachment summary.

    The browser intentionally does not send image base64 data to this endpoint. The
    original attachment objects remain in the UI and are forwarded directly to the
    coding-agent WebSocket after voice intake is ready.
    """
    if not attached_files_json:
        return []

    try:
        parsed = json.loads(attached_files_json)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    attachments: list[dict[str, Any]] = []
    total_content_chars = 0

    for index, item in enumerate(parsed[:MAX_VOICE_ATTACHMENTS]):
        if not isinstance(item, dict):
            continue

        name = _optional_text(item.get("name"), max_chars=255) or f"attachment-{index + 1}"
        source = item.get("source") if item.get("source") in {"upload", "repo"} else "upload"
        path = _optional_text(item.get("path"), max_chars=1_000)
        mime_type = _optional_text(item.get("mime_type"), max_chars=255)
        raw_content = item.get("content") if isinstance(item.get("content"), str) else ""

        remaining = MAX_TOTAL_VOICE_ATTACHMENT_CONTENT_CHARS - total_content_chars
        content = raw_content[: min(MAX_VOICE_ATTACHMENT_CONTENT_CHARS, max(0, remaining))]
        total_content_chars += len(content)

        raw_size = item.get("size")
        size = raw_size if isinstance(raw_size, int) and raw_size >= 0 else None

        attachments.append(
            {
                "name": name,
                "source": source,
                "path": path,
                "mime_type": mime_type,
                "size": size,
                "content": content or None,
                "content_truncated": bool(raw_content and len(content) < len(raw_content)),
                "has_image_data": bool(item.get("has_image_data")),
            }
        )

    return attachments


@router.post("/turn", response_model=VoiceAgentTurnResponse)
async def voice_turn(
    audio: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    history_json: str | None = Form(default=None),
    prompt_text: str | None = Form(default=None),
    attached_files_json: str | None = Form(default=None),
    repo_root: str | None = Form(default=None),
    workspace_root: str | None = Form(default=None),
    active_path: str | None = Form(default=None),
    allow_write: bool = Form(default=False),
) -> VoiceAgentTurnResponse:
    content = await audio.read()

    if not content:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file is too large.")

    try:
        result = get_voice_service().run_turn(
            audio_bytes=content,
            filename=audio.filename or "voice-input.webm",
            content_type=audio.content_type,
            session_id=session_id,
            history=_parse_history(history_json),
            prompt_text=(prompt_text or "").strip()[:20_000],
            attached_files=_parse_attached_files(attached_files_json),
            repo_root=repo_root,
            workspace_root=workspace_root,
            active_path=active_path,
            allow_write=allow_write,
        )

        return VoiceAgentTurnResponse(**result)

    except Exception as exc:
        return VoiceAgentTurnResponse(
            session_id=session_id or "",
            transcript="",
            reply_text=(
                "The voice agent hit a backend error before it could finish. "
                "Check the backend terminal for the full traceback."
            ),
            status="error",
            coding_request=None,
            audio_mime_type=None,
            audio_base64=None,
            errors=[f"Voice agent request failed: {exc}"],
        )
