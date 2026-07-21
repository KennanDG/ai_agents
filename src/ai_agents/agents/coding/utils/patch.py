from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ai_agents.agents.coding.utils.constants import (
    PATCH_CONTEXT_MAX_CHARS,
    VALIDATION_OUTPUT_MAX_CHARS,
)
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


_FILE_CONTEXT_RE = re.compile(r"^File:\s*(?P<path>[^\n]+)\n", re.MULTILINE)


def _repo_attachment_paths(state: CodingAgentState) -> list[str]:
    paths: list[str] = []

    for item in state.get("attached_files") or []:
        if str(item.get("source", "upload")).strip() != "repo":
            continue

        path = str(item.get("path", "") or "").strip()

        if path and path not in paths:
            paths.append(path)

    return paths


def _navigation_paths(state: CodingAgentState) -> list[str]:
    paths: list[str] = []

    for item in state.get("repo_navigation_files") or []:
        path = str(item.get("path", "") or "").strip()

        if path and path not in paths:
            paths.append(path)

    return paths


def _file_context_path(section: str) -> str | None:
    match = _FILE_CONTEXT_RE.match(section.strip())
    return match.group("path").strip() if match else None


def _bounded_sections(
    sections: list[str],
    *,
    max_chars: int,
    preserve_whole: bool = False,
) -> tuple[list[str], list[str]]:
    
    included: list[str] = []
    omitted: list[str] = []
    used = 0

    for section in sections:
        section = section.strip()
        if not section or section in included:
            continue

        separator_chars = 2 if included else 0
        remaining = max_chars - used - separator_chars

        if remaining <= 0:
            omitted.append(_file_context_path(section) or section.splitlines()[0][:120])
            continue

        if len(section) <= remaining:
            included.append(section)
            used += separator_chars + len(section)
            continue

        if preserve_whole:
            omitted.append(_file_context_path(section) or section.splitlines()[0][:120])
            continue

        truncation_marker = "\n...[context section truncated to fit patch budget]"
        
        if remaining > len(truncation_marker):
            included.append(section[: remaining - len(truncation_marker)] + truncation_marker)
            used += separator_chars + remaining
        else:
            omitted.append(_file_context_path(section) or section.splitlines()[0][:120])

    return included, omitted


def build_patch_context(state: CodingAgentState) -> str:
    """Build a bounded patch context with exact repository files first."""

    raw_sections = [str(section) for section in state.get("context", []) if str(section).strip()]
    file_sections: dict[str, str] = {}
    non_file_sections: list[str] = []

    for section in raw_sections:
        path = _file_context_path(section)

        if path:
            file_sections.setdefault(path, section)
        else:
            non_file_sections.append(section)

    preferred_paths = [
        *_repo_attachment_paths(state),
        *_navigation_paths(state),
        *[str(path) for path in state.get("files_inspected", [])],
    ]

    ordered_paths: list[str] = []

    for path in preferred_paths:
        if path in file_sections and path not in ordered_paths:
            ordered_paths.append(path)

    for path in file_sections:
        if path not in ordered_paths:
            ordered_paths.append(path)

    exact_file_sections = [file_sections[path] for path in ordered_paths]

    navigation_sections: list[str] = []
    external_attachment_sections: list[str] = []
    supporting_sections: list[str] = []

    for section in non_file_sections:
        stripped = section.lstrip()

        if stripped.startswith(("# Repo navigator", "Ranked repository search results:")):
            navigation_sections.append(section)

        elif stripped.startswith("# External user-attached files"):
            external_attachment_sections.append(section)

        else:
            supporting_sections.append(section)

    critical_sections: list[str] = []
    loop_feedback = _format_loop_feedback_for_patch(state)

    if loop_feedback:
        critical_sections.append(loop_feedback)

    validation_feedback = format_failed_validation_results(
        state.get("validation_results", [])
    )
    if validation_feedback:
        critical_sections.append(
            "Previous validation failures. Fix these in the next patch attempt:\n"
            f"{validation_feedback}"
        )

    recent_errors = state.get("errors", [])[-10:]
    if recent_errors:
        critical_sections.append("Prior graph errors:\n" + bullets(recent_errors))

    # Reserve most of the budget for complete repository file blocks. Metadata and
    # uploads are still included afterward when capacity remains.
    critical_budget = min(12_000, PATCH_CONTEXT_MAX_CHARS // 6)

    included_critical, omitted_critical = _bounded_sections(
        critical_sections,
        max_chars=critical_budget,
    )

    critical_used = len("\n\n".join(included_critical))
    separator_after_critical = 2 if included_critical and exact_file_sections else 0

    file_budget = min(
        PATCH_CONTEXT_MAX_CHARS - critical_budget,
        max(PATCH_CONTEXT_MAX_CHARS - critical_used - separator_after_critical, 0),
    )

    included_files, omitted_files = _bounded_sections(
        exact_file_sections,
        max_chars=file_budget,
        preserve_whole=True,
    )

    used = len("\n\n".join([*included_critical, *included_files]))

    has_tail_sections = bool(
        navigation_sections or external_attachment_sections or supporting_sections
    )

    separator_before_tail = 2 if used and has_tail_sections else 0
    remaining = max(PATCH_CONTEXT_MAX_CHARS - used - separator_before_tail, 0)
    
    included_tail, omitted_tail = _bounded_sections(
        [*navigation_sections, *external_attachment_sections, *supporting_sections],
        max_chars=remaining,
    )

    omitted = [*omitted_critical, *omitted_files, *omitted_tail]
    sections = [*included_critical, *included_files, *included_tail]

    if omitted:
        omission_summary = (
            "# Context omitted because of the patch-context budget\n"
            + bullets(omitted)
        )

        current = len("\n\n".join(sections))
        available = PATCH_CONTEXT_MAX_CHARS - current - (2 if sections else 0)
        
        if available > 0:
            sections.append(omission_summary[:available])

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
