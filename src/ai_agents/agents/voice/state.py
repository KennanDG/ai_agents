from __future__ import annotations

from typing import Any, Literal, TypedDict


class VoiceAgentState(TypedDict, total=False):
    session_id: str
    transcript: str
    history: list[dict[str, str]]

    repo_root: str | None
    workspace_root: str | None
    active_path: str | None
    allow_write: bool

    status: Literal["clarifying", "ready", "error"]
    reply_text: str
    coding_request: str | None

    collected_facts: list[str]
    errors: list[str]
    raw: dict[str, Any]