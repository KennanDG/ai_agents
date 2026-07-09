from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VoiceIntakeDecision(BaseModel):
    status: Literal["clarifying", "ready"] = "clarifying"
    reply_text: str = Field(min_length=1)
    coding_request: str | None = None
    collected_facts: list[str] = Field(default_factory=list)