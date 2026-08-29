from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


#################################### Search Service Classes ####################################

SearchMode = Literal["all", "any", "exact", "symbol"]


@dataclass(frozen=True)
class RepoSearchRequest:
    """Structured repository search request used by the deterministic search service."""

    terms: list[str] = field(default_factory=list)
    path_includes: list[str] = field(default_factory=list)
    path_excludes: list[str] = field(default_factory=list)
    file_extensions: list[str] = field(default_factory=list)
    mode: SearchMode = "all"
    max_results: int | None = None


@dataclass(frozen=True)
class PythonSymbol:
    path: str
    name: str
    kind: str
    line_no: int


class SearchRequest(BaseModel):
    terms: list[str] = Field(
        default_factory=list,
        description=(
            "Code, symbol, filename, or domain terms to search for. Keep terms short; "
            "do not include unsupported operators such as in:path:."
        ),
    )
    path_includes: list[str] = Field(
        default_factory=list,
        description="Repo-relative path fragments that results should be under or contain.",
    )
    path_excludes: list[str] = Field(
        default_factory=list,
        description="Repo-relative path fragments to exclude from search.",
    )
    file_extensions: list[str] = Field(
        default_factory=list,
        description="Optional file extensions to include, such as .py, .md, .tsx, or .sql.",
    )
    mode: SearchMode = Field(
        default="all",
        description=(
            "Matching mode. Use all for focused searches, any for broad fallback/path-only "
            "discovery, exact for exact phrase searches, and symbol for Python "
            "functions/classes/imports."
        ),
    )
    max_results: int | None = Field(
        default=None,
        description="Optional per-request result cap.",
        ge=1,
        le=50,
    )


#################################### Routing / Planning ####################################


class SkillRouteAlternative(BaseModel):
    skill_name: str = Field(description="Name of another plausible available skill.")
    reason: str = Field(default="", description="Why this alternative might fit.")


class SkillRouteDecision(BaseModel):
    selected_skills: list[str] = Field(
        min_length=1,
        max_length=3,
        description=(
            "One to three exact available skill names in priority order. The first "
            "skill is primary; later skills are supplemental and should add distinct "
            "guidance rather than duplicate the primary skill."
        ),
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence that the selected skill set is the best route.",
        ge=0.0,
        le=1.0,
    )
    reason: str = Field(
        default="",
        description="Brief routing rationale based on the request and skill catalog.",
    )
    alternatives: list[SkillRouteAlternative] = Field(
        default_factory=list,
        description="Other plausible skills not selected, ranked from most to least plausible.",
    )


TaskMode = Literal["simple", "standard", "parallel"]


class ImplementationUnitDecision(BaseModel):
    id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
        description="Stable short identifier for this independently implementable unit.",
    )
    objective: str = Field(
        min_length=1,
        description="One independently implementable portion of the user request.",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Concrete conditions that indicate this unit is complete.",
    )
    search_requests: list[SearchRequest] = Field(
        default_factory=list,
        description="Focused repository searches for this unit.",
    )
    candidate_paths: list[str] = Field(
        default_factory=list,
        description="Known repo-relative files likely relevant to this unit.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "Earlier implementation-unit IDs that must be successfully reconciled "
            "before this unit runs."
        ),
    )
    validation_commands: list[str] = Field(
        default_factory=list,
        description="Optional targeted validation commands for this unit.",
    )


class CustomToolCallDecision(BaseModel):
    tool_name: str = Field(description="Exact approved custom tool name to invoke.")
    arguments: dict[str, object] = Field(
        default_factory=dict,
        description=(
            "JSON-compatible keyword arguments for the tool. Never include repo_root; "
            "the runtime injects that value."
        ),
    )
    reason: str = Field(
        default="",
        description="Why this tool call will improve implementation context.",
    )


class PlanDecision(BaseModel):
    task_mode: TaskMode = Field(
        default="standard",
        description="Use simple for one localized edit, parallel for independent concerns.",
    )
    plan: list[str] = Field(
        default_factory=list,
        description="Short implementation plan steps.",
    )
    implementation_units: list[ImplementationUnitDecision] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Independently implementable units. Workers may inspect context and generate "
            "patch proposals for these units in parallel. Worker concurrency is controlled "
            "separately by runtime settings."
        ),
    )
    search_requests: list[SearchRequest] = Field(
        default_factory=list,
        description="Run-level structured repository search requests.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Legacy repository search terms. Prefer search_requests.",
    )
    validation_commands: list[str] = Field(
        default_factory=list,
        description="Safe validation commands to run after reconciled edits.",
    )
    web_search_query: str = Field(
        default="",
        description=(
            "Optional web search query when repository context is insufficient and "
            "current external information is needed."
        ),
    )
    custom_tool_calls: list[CustomToolCallDecision] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Optional calls to approved custom read-only tools exposed by selected skills."
        ),
    )


class FileToInspect(BaseModel):
    path: str
    reason: str = ""


class RepoNavigationDecision(BaseModel):
    """Read-only repo navigator output used before implementation workers run."""

    task_summary: str = Field(
        default="",
        description="Brief interpretation of the repository task.",
    )
    files_to_inspect: list[FileToInspect] = Field(
        default_factory=list,
        description="Small, ranked set of repo-relative files to inspect.",
    )
    additional_search_requests: list[SearchRequest] = Field(
        default_factory=list,
        description=(
            "Optional follow-up structured searches when initial ranked results are "
            "insufficient."
        ),
    )
    missing_context: list[str] = Field(
        default_factory=list,
        description="Specific missing information needed before safe editing.",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence that selected files are sufficient for the task.",
        ge=0.0,
        le=1.0,
    )


class ContextDecision(BaseModel):
    """Legacy context-selector output kept for backward compatibility."""

    files_to_inspect: list[FileToInspect] = Field(default_factory=list)


#################################### Patch Decisions ####################################


class FileEdit(BaseModel):
    operation: Literal[
        "replace",
        "create",
        "full_file_replace",
        "insert_after",
        "insert_before",
        "append",
    ] = Field(
        default="replace",
        description=(
            "Patch operation. Prefer targeted exact replacements for existing files and "
            "create only for genuinely new files."
        ),
    )
    path: str = Field(description="Repository-relative path to edit.")
    old: str = Field(
        default="",
        description=(
            "Exact existing text or anchor. Required for replace/insert operations and "
            "empty for create."
        ),
    )
    new: str = Field(
        description="Replacement, inserted, appended, or complete new-file text."
    )
    reason: str = Field(default="", description="Why this edit is needed.")


class ContextRequest(BaseModel):
    path: str = Field(description="Repo-relative file needing more exact context.")
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    terms: list[str] = Field(default_factory=list)
    reason: str = Field(default="")


class PatchDecision(BaseModel):
    summary: str = ""
    edits: list[FileEdit] = Field(default_factory=list)
    context_requests: list[ContextRequest] = Field(
        default_factory=list,
        description=(
            "Exact file ranges or search terms needed before this implementation unit can "
            "produce a safe patch."
        ),
    )
    validation_commands: list[str] = Field(default_factory=list)
    no_change_needed: bool = Field(
        default=False,
        description=(
            "True only when the implementation unit is already satisfied by the current "
            "repository and no edit is required."
        ),
    )
    blocking_reason: str = Field(
        default="",
        description=(
            "Concrete reason this implementation unit cannot be completed when neither "
            "safe edits nor actionable context requests are possible."
        ),
    )


class ReconciliationDecision(BaseModel):
    """One bounded reasoning pass used only to merge conflicting worker proposals."""

    summary: str = ""
    unit_ids: list[str] = Field(
        default_factory=list,
        description="Implementation unit ids covered by the reconciled patch.",
    )
    edits: list[FileEdit] = Field(
        default_factory=list,
        description="Merged edits that satisfy the covered implementation units.",
    )
    validation_commands: list[str] = Field(default_factory=list)
    blocking_reason: str = Field(
        default="",
        description="Why the conflicting proposals could not be safely reconciled.",
    )


class ReportDecision(BaseModel):
    report: str


class ProgressDecision(BaseModel):
    """Legacy progress-evaluator schema kept for persisted-run compatibility."""

    is_complete: bool = False
    should_continue: bool = False
    reason: str = ""
    remaining_tasks: list[str] = Field(default_factory=list)
    additional_search_requests: list[SearchRequest] = Field(default_factory=list)
    next_iteration_notes: str = ""
