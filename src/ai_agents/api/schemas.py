from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


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
    id: int
    source_uri: str
    content_hash: str
    collection_name: str
    namespace: str
    chunk_size: int
    chunk_overlap: int


class SourcesListResponse(BaseModel):
    sources: List[SourceRow]
