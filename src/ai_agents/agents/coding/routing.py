from __future__ import annotations

from typing import Literal

from ai_agents.agents.coding.utils.constants import MAX_PATCH_ATTEMPTS
from ai_agents.agents.coding.coding_agent_settings import settings as default_settings
from ai_agents.agents.coding.state import CodingAgentState


def route_after_plan(state: CodingAgentState) -> str:
    selected_skill = state.get("selected_skill")

    if selected_skill == "web_search":
        return "web_search"
    if selected_skill == "gmail_access":
        return "gmail_access"
    return "repo_navigator"


def route_after_context(state: CodingAgentState) -> Literal["patch", "report"]:
    if state.get("status") == "context_failed" or not state.get("context"):
        return "report"
    return "patch"


def route_after_patch(
    state: CodingAgentState,
) -> Literal["repo_navigator", "validate", "report"]:
    if state.get("status") in {"patch_failed", "patch_skipped"}:
        should_refresh_context = bool(state.get("continue_loop"))
        return (
            "repo_navigator"
            if should_refresh_context and patch_attempts_remaining(state)
            else "report"
        )

    if not state.get("file_changes"):
        return "report"

    # A dry-run validates the unchanged repository, not the proposed patch.
    if not bool(state.get("allow_write", default_settings.allow_write)):
        return "report"

    return "validate"


def route_after_validate(
    state: CodingAgentState,
) -> Literal["assess_progress", "report"]:
    # Successful validation needs no evaluator LLM call. Enter the repair loop only
    # when validation produced a blocking failure.
    if bool(state.get("blocking_validation_failed")):
        return "assess_progress"
    return "report"


def route_after_assess(
    state: CodingAgentState,
) -> Literal["repo_navigator", "report"]:
    return "repo_navigator" if bool(state.get("continue_loop")) else "report"


def patch_attempts_remaining(state: CodingAgentState) -> bool:
    patch_attempts = int(state.get("patch_attempts", 0))
    max_patch_attempts = int(state.get("max_patch_attempts", MAX_PATCH_ATTEMPTS))
    return patch_attempts < max_patch_attempts
