from __future__ import annotations

from typing import Any, Literal, TypedDict
from ai_agents.api.schemas import CodingAgentAttachedFile


CodingAgentStatus = Literal[
    "planned",
    "routed",
    "repo_navigated",
    "repo_navigation_failed",
    "web_search_skipped",
    "web_search_completed",
    "web_search_failed",
    "gmail_access_skipped",
    "gmail_access_completed",
    "context_gathered",
    "context_failed",
    "patched",
    "patch_failed",
    "patch_skipped",
    "validated",
    "validation_failed",
    "assessed",
    "loop_limit_reached",
    "reported",
]


class CodingAgentState(TypedDict, total=False):
    user_request: str
    repo_root: str          # target root for searching/patching
    workspace_root: str     # project root for validation
    allow_write: bool

    attached_files: list[dict[str, Any]]
    attached_files_used: list[str]
    attachment_errors: list[str]
    
    selected_skill: str
    skill_instructions: str
    route_confidence: float
    route_reason: str
    route_alternatives: list[dict[str, str]]

    plan: list[str]
    search_requests: list[dict[str, Any]]
    search_queries: list[str]  # legacy fallback while migrating to structured search
    search_results: list[dict[str, Any]]

    long_term_memories: list[str]
    memory_enabled: bool
    memory_namespace: str
    memory_saved: bool
    memory_errors: list[str]

    repo_navigation_summary: str
    repo_navigation_files: list[dict[str, str]]
    repo_navigation_confidence: float
    repo_navigation_missing_context: list[str]
    repo_navigation_search_requests: list[dict[str, Any]]

    context: list[str]
    files_inspected: list[str]
    file_changes: list[dict[str, str]]
    diffs: list[str]
    patch_summary: str
    validation_commands: list[str]
    validation_results: list[dict[str, Any]]

    report: str
    status: CodingAgentStatus
    errors: list[str]
    patch_attempts: int
    max_patch_attempts: int


    iteration: int
    max_iterations: int
    continue_loop: bool
    remaining_tasks: list[str]
    loop_notes: list[str]
    progress_reason: str



    