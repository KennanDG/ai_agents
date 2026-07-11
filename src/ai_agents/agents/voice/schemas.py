from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class VoiceIntakeDecision(BaseModel):
    status: Literal["clarifying", "ready"] = "clarifying"
    reply_text: str = Field(min_length=1)
    coding_request: str | None = None
    collected_facts: list[str] = Field(default_factory=list)

    @field_validator("collected_facts", mode="before")
    @classmethod
    def normalize_collected_facts(cls, value: Any) -> list[str]:
        if value is None:
            return []

        items = value if isinstance(value, list) else [value]
        normalized: list[str] = []

        for item in items:
            if item is None:
                continue

            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = json.dumps(
                    item,
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                )
            elif isinstance(item, list):
                text = json.dumps(
                    item,
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                )
            else:
                text = str(item).strip()

            if text:
                normalized.append(text)

        return normalized