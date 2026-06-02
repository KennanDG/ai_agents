from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_agents.agents.coding.utils.constants import VALIDATION_OUTPUT_MAX_CHARS
from ai_agents.agents.coding.state import CodingAgentState
from ai_agents.agents.coding.utils.text import bullets, truncate


def apply_exact_replace(original: str, old: str, new: str, *, path: str) -> str:
    if not old:
        raise ValueError(f"Empty old text for {path}")

    count = original.count(old)

    if count == 0:
        raise ValueError(f"Could not find exact old text in {path}")

    if count > 1:
        raise ValueError(f"Old text matched more than once in {path}; refusing ambiguous edit")

    return original.replace(old, new, 1)


def is_forbidden_write_path(path: str) -> bool:
    parts = set(Path(path).parts)
    forbidden_parts = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
    forbidden_names = {
        ".env",
        ".env.local",
        ".env.production",
        ".env.docker",
        "uv.lock",
        "poetry.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
    }

    return bool(parts & forbidden_parts) or Path(path).name in forbidden_names


def build_patch_context(state: CodingAgentState) -> str:
    sections = list(state.get("context", []))

    validation_feedback = format_failed_validation_results(
        state.get("validation_results", [])
    )
    if validation_feedback:
        sections.append(
            "Previous validation failures. Fix these in the next patch attempt:\n"
            f"{validation_feedback}"
        )

    recent_errors = state.get("errors", [])[-10:]
    if recent_errors:
        sections.append("Prior graph errors:\n" + bullets(recent_errors))

    return "\n\n".join(sections)


def format_failed_validation_results(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    for result in results:
        try:
            returncode = int(result.get("returncode", 0))
        except (TypeError, ValueError):
            returncode = 1

        if returncode == 0:
            continue

        command = result.get("command", "unknown command")
        stdout = truncate(str(result.get("stdout", "")), VALIDATION_OUTPUT_MAX_CHARS)
        stderr = truncate(str(result.get("stderr", "")), VALIDATION_OUTPUT_MAX_CHARS)
        lines.append(
            f"Command: {command}\n"
            f"Exit code: {returncode}\n"
            f"STDOUT:\n{stdout or '(empty)'}\n"
            f"STDERR:\n{stderr or '(empty)'}"
        )

    return "\n\n".join(lines)
