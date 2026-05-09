from __future__ import annotations

from typing import Any, Literal, TypedDict


CodingAgentStatus = Literal[
    "planned",
    "routed",
    "context_gathered",
    "context_failed",
    "patched",
    "patch_failed",
    "patch_skipped",
    "validated",
    "validation_failed",
    "reported",
]


class CodingAgentState(TypedDict, total=False):
    user_request: str
    repo_root: str          # target root for searching/patching
    workspace_root: str     # project root for validation
    allow_write: bool
    selected_skill: str
    skill_instructions: str
    plan: list[str]
    search_queries: list[str]
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
