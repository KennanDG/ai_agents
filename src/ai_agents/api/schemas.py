from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator

from ai_agents.config.constants import ChatProvider, AgentKind



NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_MAX_SKILL_CHARS = 50_000
_MAX_TOOL_CHARS = 100_000


class HealthResponse(BaseModel):
    status: str = "ok"



############################## CODING AGENT ##############################
class CodingAgentAttachedFile(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    content: str | None = None
    data_url: str | None = Field(default=None, max_length=8_000_000)

    # Repo-relative path when the file already exists in the repo.
    # For local uploads, this can stay null.
    path: str | None = None

    source: Literal["upload", "repo"] = "upload"
    mime_type: str | None = None
    size: int | None = Field(default=None, ge=0)
    truncated: bool | None = False

    
class CodingAgentRunRequest(BaseModel):
    request: str
    repo_root: str
    workspace_root: str | None = None
    allow_write: bool = False
    thread_id: str | None = None
    memory_user_id: str | None = None
    memory_namespace: str | None = None
    memory_enabled: bool | None = None
    setup_memory: bool = False
    max_iterations: int | None = Field(default=3, ge=1, le=8)

    # Optional per-run overrides. Defaults come from the saved admin profile.
    subagent_count: int | None = Field(default=None, ge=1, le=6)
    route_max_tokens: int | None = Field(default=None, ge=256, le=2_000)
    planner_max_tokens: int | None = Field(default=None, ge=512, le=6_000)
    repo_navigation_max_tokens: int | None = Field(default=None, ge=512, le=4_000)
    simple_patch_max_tokens: int | None = Field(default=None, ge=2_000, le=16_000)
    patch_max_tokens: int | None = Field(default=None, ge=4_000, le=32_000)
    progress_max_tokens: int | None = Field(default=None, ge=512, le=4_000)

    attached_files: list[CodingAgentAttachedFile] = Field(default_factory=list, max_length=20)


class CodingAgentRunResult(BaseModel):
    thread_id: str
    status: str = "unknown"

    report: str | None = None
    selected_skill: str | None = None
    task_mode: Literal["simple", "standard", "parallel"] | None = None
    subtasks: List[Dict[str, Any]] = Field(default_factory=list)
    context_worker_count: int = 0
    route_confidence: float | None = None
    route_reason: str | None = None

    plan: List[str] = Field(default_factory=list)
    files_inspected: List[str] = Field(default_factory=list)
    patch_summary: str | None = None
    file_changes: List[Dict[str, Any]] = Field(default_factory=list)
    diffs: List[str] = Field(default_factory=list)

    validation_commands: List[str] = Field(default_factory=list)
    validation_results: List[Dict[str, Any]] = Field(default_factory=list)

    approval_required: bool = False
    approval_status: str = "not_required"
    blocking_validation_failed: bool = False
    advisory_validation_failed: bool = False
    applied_files: list[str] = Field(default_factory=list)

    memory_enabled: bool = False
    memory_namespace: str | None = None
    long_term_memories: List[str] = Field(default_factory=list)
    memory_errors: List[str] = Field(default_factory=list)

    errors: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class CodingAgentClientMessage(BaseModel):
    type: Literal[
        "ping",
        "run.request",
        "run.apply.request",
        "run.reject.request",
    ]
    
    payload: dict[str, Any] = Field(default_factory=dict)


class CodingAgentServerEvent(BaseModel):
    type: Literal[
        "session.ready",
        "run.started",
        "node.completed",
        "run.completed",
        "run.failed",
        "run.approval_required",
        "run.applied",
        "run.rejected",
        "pong",
    ]

    run_id: str | None = None
    thread_id: str | None = None
    node: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)






############################## VOICE AGENT ##############################

class VoiceMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class VoiceTextTurnRequest(BaseModel):
    text: str = Field(min_length=1)
    session_id: str | None = None
    history: list[VoiceMessage] = Field(default_factory=list)
    repo_root: str | None = None
    workspace_root: str | None = None
    active_path: str | None = None
    allow_write: bool = False


class VoiceAgentTurnResponse(BaseModel):
    session_id: str
    transcript: str
    reply_text: str
    status: Literal["clarifying", "ready", "error"] = "clarifying"

    # When ready, frontend should hand this to the existing coding agent.
    coding_request: str | None = None

    audio_mime_type: str | None = None
    audio_base64: str | None = None

    errors: list[str] = Field(default_factory=list)





############################## REPOSITORY API ##############################
class RepositoryTreeEntry(BaseModel):
    path: str
    name: str
    kind: Literal["file", "directory"]
    depth: int = 0
    size: int | None = None


class RepositoryTreeResponse(BaseModel):
    repo_root: str
    entries: List[RepositoryTreeEntry] = Field(default_factory=list)


class RepositoryFileResponse(BaseModel):
    repo_root: str
    path: str
    language: str = "plaintext"
    content: str
    size: int






############################## GITHUB ##############################
class GitHubRepositoryImportRequest(BaseModel):
    full_name: str = Field(..., examples=["owner/repository"])
    ref: str | None = Field(default=None, description="Branch to check out.")
    refresh: bool = Field(
        default=False,
        description="Fetch and fast-forward an existing managed checkout.",
    )


class GitHubRepositorySummary(BaseModel):
    id: int
    full_name: str
    name: str
    owner: str
    private: bool
    default_branch: str
    clone_url: str
    html_url: str
    updated_at: str | None = None
    permissions: dict[str, bool] = Field(default_factory=dict)


class GitHubRepositoryImportResponse(BaseModel):
    full_name: str
    ref: str
    repo_root: str
    reused_existing_checkout: bool
    previous_ref: str | None = None
    saved_previous_changes: bool = False
    restored_target_changes: bool = False


class GitHubBranchSummary(BaseModel):
    name: str
    sha: str


class GitHubConnectionTestResponse(BaseModel):
    connected: bool
    api_connected: bool
    git_available: bool
    git_transport_connected: bool
    workspace_writable: bool
    token_kind: Literal["user", "installation"]
    account: str | None = None
    full_name: str | None = None
    default_branch: str | None = None
    permissions: dict[str, bool] = Field(default_factory=dict)
    message: str


class GitHubRepositoryStatus(BaseModel):
    full_name: str
    repo_root: str
    branch: str
    default_branch: str
    head_sha: str
    upstream: str | None = None
    ahead: int = 0
    behind: int = 0
    dirty: bool
    staged_files: list[str] = Field(default_factory=list)
    unstaged_files: list[str] = Field(default_factory=list)
    untracked_files: list[str] = Field(default_factory=list)


class GitHubCreateBranchRequest(BaseModel):
    full_name: str
    branch: str
    base: str | None = None


class GitHubCreateBranchResponse(BaseModel):
    full_name: str
    branch: str
    sha: str


class GitHubPullRequest(BaseModel):
    full_name: str


class GitHubPullResponse(BaseModel):
    full_name: str
    branch: str
    head_sha: str
    changed: bool


class GitHubCommitRequest(BaseModel):
    full_name: str
    message: str = Field(min_length=3, max_length=200)
    paths: list[str] = Field(min_length=1)


class GitHubCommitResponse(BaseModel):
    full_name: str
    branch: str
    commit_sha: str
    committed_files: list[str]


class GitHubPushRequest(BaseModel):
    full_name: str


class GitHubPushResponse(BaseModel):
    full_name: str
    branch: str
    commit_sha: str
    pushed: bool


class GitHubPullRequestCreateRequest(BaseModel):
    full_name: str
    title: str = Field(min_length=3, max_length=256)
    body: str = Field(default="", max_length=65_536)
    base: str | None = None
    head: str | None = None
    draft: bool = True
    maintainer_can_modify: bool = True


class GitHubPullRequestResponse(BaseModel):
    full_name: str
    number: int
    title: str
    html_url: str
    base: str
    head: str
    draft: bool
    created: bool









############################## ADMIN ##############################

class AgentConfigurationUpdate(BaseModel):
    coding_provider: ChatProvider
    coding_model: str = Field(min_length=1, max_length=255)
    reasoning_provider: ChatProvider
    reasoning_model: str = Field(min_length=1, max_length=255)
    caption_provider: ChatProvider
    caption_model: str = Field(min_length=1, max_length=255)
    voice_chat_provider: ChatProvider
    voice_chat_model: str = Field(min_length=1, max_length=255)
    voice_stt_provider: ChatProvider
    voice_stt_model: str = Field(min_length=1, max_length=255)
    voice_tts_provider: ChatProvider
    voice_tts_model: str = Field(min_length=1, max_length=255)
    voice_tts_voice: str = Field(min_length=1, max_length=100)
    voice_tts_enabled: bool = True

    coding_subagent_count: int | None = Field(default=None, ge=1, le=6)
    coding_route_max_tokens: int | None = Field(default=None, ge=256, le=2_000)
    coding_planner_max_tokens: int | None = Field(default=None, ge=512, le=6_000)
    coding_repo_navigation_max_tokens: int | None = Field(default=None, ge=512, le=4_000)
    coding_simple_patch_max_tokens: int | None = Field(default=None, ge=2_000, le=16_000)
    coding_patch_max_tokens: int | None = Field(default=None, ge=4_000, le=32_000)
    coding_progress_max_tokens: int | None = Field(default=None, ge=512, le=4_000)

    secrets: dict[ChatProvider, str] = Field(default_factory=dict)

    @field_validator(
        "coding_model",
        "reasoning_model",
        "caption_model",
        "voice_chat_model",
        "voice_stt_model",
        "voice_tts_model",
        "voice_tts_voice",
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class SkillWriteRequest(BaseModel):
    agent: AgentKind
    name: str
    content: str = Field(min_length=1, max_length=_MAX_SKILL_CHARS)
    overwrite: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not NAME_RE.fullmatch(normalized):
            raise ValueError(
                "Skill names must start with a letter and contain only lowercase "
                "letters, numbers, underscores, or hyphens."
            )
        return normalized


class ToolQuarantineRequest(BaseModel):
    agent: AgentKind
    name: str
    purpose: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=_MAX_TOOL_CHARS)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not NAME_RE.fullmatch(normalized):
            raise ValueError("Tool names must use lowercase snake_case.")
        return normalized

    @field_validator("purpose")
    @classmethod
    def normalize_purpose(cls, value: str) -> str:
        return value.strip()


class SkillSummary(BaseModel):
    agent: AgentKind
    name: str
    purpose: str
    allowed_tools: list[str] = Field(default_factory=list)
    content: str
    custom: bool


class ToolSummary(BaseModel):
    agent: AgentKind
    name: str
    module: str
    purpose: str
    status: Literal["builtin", "pending_review"]











############################## RAG ##############################
class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    
    # Configuration overrides
    k: Optional[int] = None
    namespace: Optional[str] = None
    collection_name: Optional[str] = None
    preferred_collections: Optional[List[str]] = None
    enable_query_expansion: Optional[bool] = None
    enable_parallel_collection_retrieval: Optional[bool] = None


class RagQueryResponse(BaseModel):
    answer: Union[str, Dict]
    meta: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    # Accept files/dirs/globs
    paths: List[str] = Field(..., min_length=1)
    namespace: Optional[str] = None
    collection_name: Optional[str] = None


class IngestResponse(BaseModel):
    ingested_chunks: int
    meta: Dict[str, Any] = Field(default_factory=dict)


class SourceRow(BaseModel):
    # id: int
    source_uri: str
    content_hash: str
    collection_name: str
    namespace: str
    chunk_size: int
    chunk_overlap: int


class SourcesListResponse(BaseModel):
    sources: List[SourceRow]












