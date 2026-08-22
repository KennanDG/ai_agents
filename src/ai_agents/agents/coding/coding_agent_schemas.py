from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from dataclasses import dataclass, field




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


class SubtaskDecision(BaseModel):
    id: str = Field(description="Stable short identifier for the context worker.")
    objective: str = Field(description="Focused implementation concern for this worker.")

    search_requests: list[SearchRequest] = Field(
        default_factory=list,
        description="Narrow structured searches for this subtask.",
    )

    candidate_paths: list[str] = Field(
        default_factory=list,
        description="Known repo-relative files that this worker should inspect.",
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
    reason: str = Field(default="", description="Why this tool call will improve implementation context.")


class PlanDecision(BaseModel):
    task_mode: TaskMode = Field(
        default="standard",
        description="Use simple for one localized edit, parallel for independent concerns.",
    )
    plan: list[str] = Field(
        default_factory=list,
        description="Short implementation plan steps.",
    )
    search_requests: list[SearchRequest] = Field(
        default_factory=list,
        description="Structured repository search requests to gather context.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Repository search terms to gather context.",
    )
    validation_commands: list[str] = Field(
        default_factory=list,
        description="Safe validation commands to run after edits.",
    )
    web_search_query: str = Field(
        default="",
        description="Optional web search query when current context is insufficient and external information is needed.",
    )
    custom_tool_calls: list[CustomToolCallDecision] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Optional calls to approved custom read-only tools exposed by selected skills. "
            "Leave empty unless a listed tool materially improves repository context."
        ),
    )
    subtasks: list[SubtaskDecision] = Field(
        default_factory=list,
        description=(
            "Independent read-only context work. Keep the count modest; the runtime "
            "applies the configured worker cap (maximum six). Do not split changes "
            "that must be reasoned about atomically."
        ),
    )


class FileToInspect(BaseModel):
    path: str
    reason: str = ""




class RepoNavigationDecision(BaseModel):
    """Read-only repo navigator output used before context loading."""

    task_summary: str = Field(
        default="",
        description="Brief interpretation of the repository task.",
    )
    files_to_inspect: list[FileToInspect] = Field(
        default_factory=list,
        description="Small, ranked set of repo-relative files to read before editing.",
    )
    additional_search_requests: list[SearchRequest] = Field(
        default_factory=list,
        description=(
            "Optional follow-up structured searches when the initial ranked results are "
            "insufficient. Leave empty when enough files were found."
        ),
    )
    missing_context: list[str] = Field(
        default_factory=list,
        description="Specific missing information needed before safe editing.",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence that the selected files are sufficient for the task.",
        ge=0.0,
        le=1.0,
    )




class ContextDecision(BaseModel):
    files_to_inspect: list[FileToInspect] = Field(default_factory=list)


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
                "Patch operation type. Use 'replace' for exact text replacement in an "
                "existing file. Use 'create' only for creating a brand-new file."
            ),
        )

    path: str = Field(description="Repository-relative path to edit.")

    old: str = Field(
        default="",
        description=(
            "Exact existing text to replace. Required for replace operations. "
            "Must be empty for create operations."
        ),
    )

    new: str = Field(
        description=(
            "Replacement text for replace operations, or full file contents for "
            "create operations."
        )
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
            "Exact file ranges or search terms needed before patching. Use this instead "
            "of guessing when a large file was represented by selected chunks."
        ),
    )
    
    validation_commands: list[str] = Field(default_factory=list)


class ReportDecision(BaseModel):
    report: str


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
            "discovery, exact for exact phrase searches, and symbol for Python functions/classes/imports."
        ),
    )
    max_results: int | None = Field(
        default=None,
        description="Optional per-request result cap.",
        ge=1,
        le=50,
    )






class ProgressDecision(BaseModel):
    is_complete: bool = Field(
        default=False,
        description="Whether the user request is fully complete.",
    )
    should_continue: bool = Field(
        default=False,
        description="Whether another implementation loop is needed.",
    )
    reason: str = Field(
        default="",
        description="Short explanation of the progress decision.",
    )
    remaining_tasks: list[str] = Field(
        default_factory=list,
        description="Specific remaining tasks for the next loop.",
    )
    additional_search_requests: list[SearchRequest] = Field(
        default_factory=list,
        description="Additional repo searches needed for the next loop.",
    )
    next_iteration_notes: str = Field(
        default="",
        description="Useful notes to carry into the next patch loop.",
    )


    