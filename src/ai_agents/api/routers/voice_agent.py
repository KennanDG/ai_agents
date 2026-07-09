from __future__ import annotations

import json
from functools import lru_cache

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ai_agents.agents.voice.service import VoiceAgentService
from ai_agents.api.schemas import VoiceAgentTurnResponse


router = APIRouter(prefix="/voice-agent", tags=["voice-agent"])


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

        history.append({"role": role, "content": content})

    return history



@router.post("/turn", response_model=VoiceAgentTurnResponse)
async def voice_turn(
    audio: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    history_json: str | None = Form(default=None),
    repo_root: str | None = Form(default=None),
    workspace_root: str | None = Form(default=None),
    active_path: str | None = Form(default=None),
    allow_write: bool = Form(default=False),
) -> VoiceAgentTurnResponse:
    content = await audio.read()

    if not content:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    # Keep this conservative. Browser recordings should usually be tiny.
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file is too large.")


    result = get_voice_service().run_turn(
        audio_bytes=content,
        filename=audio.filename or "voice-input.webm",
        content_type=audio.content_type,
        session_id=session_id,
        history=_parse_history(history_json),
        repo_root=repo_root,
        workspace_root=workspace_root,
        active_path=active_path,
        allow_write=allow_write,
    )

    return VoiceAgentTurnResponse(**result)