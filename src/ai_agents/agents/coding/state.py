from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


CodingAgentStatus = Literal[
    "planned",
    "routed",
    "route_failed",
    "repo_navigated",
    "repo_navigation_failed",
    "context_workers_completed",
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
    "approval_pending",
    "applied",
    "rejected",
]


class CodingAgentState(TypedDict, total=False):
    user_request: str
    repo_root: str              # target root for searching/patching
    original_repo_root: str     # Stable source repository. Use this for persistence namespaces.
    workspace_root: str         # project root for validation
    sandbox_root: str
    sandbox_enabled: bool
    allow_write: bool
    runtime_settings: dict[str, int]

    attached_files: list[dict[str, Any]]
    attached_files_used: list[str]
    attachment_errors: list[str]

    # Execution strategy. Simple tasks use a deterministic fast path; parallel
    # tasks fan out read-only context workers with isolated inputs.
    task_mode: Literal["simple", "standard", "parallel"]
    subtasks: list[dict[str, Any]]
    active_subtask: dict[str, Any]
    context_generation: int
    context_worker_results: Annotated[list[dict[str, Any]], operator.add]
    requested_context: list[dict[str, Any]]

    selected_skill: str
    skill_instructions: str
    route_confidence: float
    route_reason: str
    route_alternatives: list[dict[str, str]]

    plan: list[str]
    search_requests: list[dict[str, Any]]
    search_queries: list[str]  # legacy fallback while migrating to structured search    
    web_search_query: str
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

    blocking_validation_failed: bool
    advisory_validation_failed: bool

    approval_required: bool
    approval_status: Literal["not_required", "pending", "applied", "rejected"]
    applied_files: list[str]

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
    loop_context_focus: str
    progress_reason: str
