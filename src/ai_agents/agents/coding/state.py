from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


CodingAgentStatus = Literal[
    "planned",
    "routed",
    "route_failed",
    "repo_navigated",
    "repo_navigation_failed",
    "subtask_workers_completed",
    "subtask_workers_failed",
    "custom_tools_skipped",
    "custom_tools_completed",
    "custom_tools_failed",
    "web_search_skipped",
    "web_search_completed",
    "web_search_failed",
    "gmail_access_skipped",
    "gmail_access_completed",
    "patches_reconciled",
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
    run_id: str
    user_request: str
    repo_root: str
    original_repo_root: str
    workspace_root: str
    sandbox_root: str
    sandbox_enabled: bool
    allow_write: bool
    runtime_settings: dict[str, int]

    attached_files: list[dict[str, Any]]
    attached_files_used: list[str]
    attachment_errors: list[str]

    task_mode: Literal["simple", "standard", "parallel"]

    # Durable work decomposition. Total implementation units are independent from
    # worker concurrency, so a 3-worker run may process more than three units over
    # multiple deterministic batches.
    implementation_units: list[dict[str, Any]]
    implementation_run_id: str
    active_implementation_unit: dict[str, Any]
    implementation_generation: int
    implementation_iteration: int
    max_implementation_iterations: int
    subtask_worker_results: Annotated[list[dict[str, Any]], operator.add]
    completion_ledger: dict[str, dict[str, Any]]

    # Legacy loop/context fields remain in state so older checkpoints and API
    # payloads can still deserialize during the architecture migration.
    subtasks: list[dict[str, Any]]
    active_subtask: dict[str, Any]
    context_generation: int
    context_worker_results: Annotated[list[dict[str, Any]], operator.add]
    requested_context: list[dict[str, Any]]
    iteration: int
    max_iterations: int
    continue_loop: bool
    remaining_tasks: list[str]
    loop_notes: list[str]
    loop_context_focus: str
    progress_reason: str

    selected_skill: str
    selected_skills: list[str]
    selected_skill_tools: list[str]
    custom_tool_calls: list[dict[str, Any]]
    custom_tool_results: list[dict[str, Any]]
    skill_instructions: str
    route_confidence: float
    route_reason: str
    route_alternatives: list[dict[str, str]]

    plan: list[str]
    search_requests: list[dict[str, Any]]
    search_queries: list[str]
    web_search_query: str
    web_search_results: str
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

    # Aggregated observability. Workers own prompt context locally, so `context` is
    # retained only for backward compatibility and reporting.
    context: list[str]
    files_inspected: list[str]
    file_changes: list[dict[str, Any]]
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

    # Legacy global patch counters. New runs use per-unit patch_retries stored in
    # completion_ledger/subtask_worker_results instead.
    patch_attempts: int
    max_patch_attempts: int
