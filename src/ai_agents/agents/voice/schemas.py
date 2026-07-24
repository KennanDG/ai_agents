from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


MAX_CODING_REQUEST_CHARS = 4_000


class VoiceIntakeDecision(BaseModel):
    status: Literal["clarifying", "ready"] = "clarifying"
    reply_text: str = Field(min_length=1)
    coding_request: str | None = None
    collected_facts: list[str] = Field(default_factory=list)
    selected_skills: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    target_files: list[str] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)

    @field_validator("coding_request", mode="before")
    @classmethod
    def normalize_coding_request(cls, value: Any) -> str | None:
        """Accept a model-produced object without failing the whole intake turn.

        The prompt requires a string, but smaller JSON-mode models sometimes emit a
        nested object. Prefer its objective and otherwise serialize a bounded value.
        """
        if value is None:
            return None

        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, dict):
            objective = value.get("objective")
            if isinstance(objective, str) and objective.strip():
                text = objective.strip()
            else:
                text = json.dumps(
                    value,
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                )
        elif isinstance(value, list):
            text = "\n".join(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value).strip()

        return text[:MAX_CODING_REQUEST_CHARS] or None

    @field_validator(
        "collected_facts",
        "selected_skills",
        "tools_used",
        "target_files",
        "plan",
        mode="before",
    )
    @classmethod
    def normalize_string_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []

        items = value if isinstance(value, list) else [value]
        normalized: list[str] = []

        for item in items:
            if item is None:
                continue

            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, (dict, list)):
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
