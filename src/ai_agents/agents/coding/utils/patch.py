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

    loop_feedback = _format_loop_feedback_for_patch(state)
    if loop_feedback:
        sections.append(loop_feedback)

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



def _format_loop_feedback_for_patch(state: CodingAgentState) -> str:
    
    lines: list[str] = []

    loop_context_focus = str(state.get("loop_context_focus", "")).strip()
    progress_reason = str(state.get("progress_reason", "")).strip()
    remaining_tasks = list(state.get("remaining_tasks") or [])
    loop_notes = list(state.get("loop_notes") or [])[-5:]

    if loop_context_focus:
        lines.append("Focus for this retry/context-refresh loop:")
        lines.append(loop_context_focus)

    if progress_reason:
        lines.append(f"Progress assessment: {progress_reason}")

    if remaining_tasks:
        lines.append("Remaining tasks:")
        lines.append(bullets([str(item) for item in remaining_tasks]))

    if loop_notes:
        lines.append("Prior loop notes:")
        lines.append(bullets([str(item) for item in loop_notes]))

    if not lines:
        return ""

    return "# Retry/context-refresh guidance\n" + "\n".join(lines)



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
