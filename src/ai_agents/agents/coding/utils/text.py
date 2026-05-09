from __future__ import annotations

from typing import Any


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None"


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        normalized = item.strip()

        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


def truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value

    return value[:max_chars] + "\n...<truncated>"


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)
