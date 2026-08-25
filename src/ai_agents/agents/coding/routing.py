from __future__ import annotations

from typing import Literal

from ai_agents.agents.coding.coding_agent_settings import settings as default_settings
from ai_agents.agents.coding.state import CodingAgentState
from ai_agents.agents.coding.utils.constants import MAX_PATCH_ATTEMPTS
from ai_agents.agents.coding.utils.helpers import _implementation_complete


def _selected_skills(state: CodingAgentState) -> set[str]:
    selected = {
        str(name).strip()
        for name in state.get("selected_skills", [])
        if str(name).strip()
    }

    primary = str(state.get("selected_skill", "")).strip()
    if primary:
        selected.add(primary)

    return selected


def route_after_plan(state: CodingAgentState) -> str:
    if (state.get("web_search_query") or "").strip():
        return "web_search"

    selected = _selected_skills(state)
    if "web_search" in selected:
        return "web_search"

    if "gmail_access" in selected:
        return "gmail_access"

    return "repo_navigator"


def route_after_web_search(
    state: CodingAgentState,
) -> Literal["gmail_access", "repo_navigator"]:
    return (
        "gmail_access"
        if "gmail_access" in _selected_skills(state)
        else "repo_navigator"
    )


def route_after_context(state: CodingAgentState) -> Literal["patch", "report"]:
    """Legacy compatibility for old checkpoints/graphs."""
    if state.get("status") == "context_failed" or not state.get("context"):
        return "report"
    return "patch"


def route_after_patch(
    state: CodingAgentState,
) -> Literal["assess_progress", "validate", "report"]:
    # The deterministic completion ledger, not one global patch attempt counter, is
    # the source of truth for whether another implementation batch is needed.
    if not _implementation_complete(state):
        return "assess_progress"

    if not state.get("file_changes"):
        return "report"

    # A dry-run validates the unchanged repository rather than the proposed patch.
    if not bool(state.get("allow_write", default_settings.allow_write)):
        return "report"

    return "validate"


def route_after_validate(
    state: CodingAgentState,
) -> Literal["assess_progress", "report"]:
    return (
        "assess_progress"
        if bool(state.get("blocking_validation_failed"))
        else "report"
    )


def route_after_assess(
    state: CodingAgentState,
) -> Literal["repo_navigator", "report"]:
    return "repo_navigator" if bool(state.get("continue_loop")) else "report"


def patch_attempts_remaining(state: CodingAgentState) -> bool:
    """Legacy helper retained for old checkpoints; active workers track retries per unit."""
    patch_attempts = int(state.get("patch_attempts", 0))
    max_patch_attempts = int(state.get("max_patch_attempts", MAX_PATCH_ATTEMPTS))
    return patch_attempts < max_patch_attempts
