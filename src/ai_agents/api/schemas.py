from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"



############################## CODING AGENT ##############################
class CodingAgentRunRequest(BaseModel):
    request: str = Field(..., min_length=1)

    repo_root: str = "."
    workspace_root: str | None = "."

    allow_write: bool = False

    thread_id: str | None = None
    memory_user_id: str | None = None
    memory_namespace: str | None = None
    memory_enabled: bool | None = None
    setup_memory: bool = False


class CodingAgentRunResult(BaseModel):
    thread_id: str
    status: str = "unknown"

    report: str | None = None
    selected_skill: str | None = None
    route_confidence: float | None = None
    route_reason: str | None = None

    plan: List[str] = Field(default_factory=list)
    files_inspected: List[str] = Field(default_factory=list)
    patch_summary: str | None = None
    file_changes: List[Dict[str, Any]] = Field(default_factory=list)
    diffs: List[str] = Field(default_factory=list)

    validation_commands: List[str] = Field(default_factory=list)
    validation_results: List[Dict[str, Any]] = Field(default_factory=list)

    memory_enabled: bool = False
    memory_namespace: str | None = None
    long_term_memories: List[str] = Field(default_factory=list)
    memory_errors: List[str] = Field(default_factory=list)

    errors: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class CodingAgentClientMessage(BaseModel):
    type: Literal["run.request", "ping"]
    payload: Dict[str, Any] = Field(default_factory=dict)


class CodingAgentServerEvent(BaseModel):
    type: Literal[
        "session.ready",
        "run.started",
        "node.completed",
        "run.completed",
        "run.failed",
        "pong",
    ]
    run_id: str | None = None
    thread_id: str | None = None
    node: str | None = None
    payload: Dict[str, Any] = Field(default_factory=dict)







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
