from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from ai_agents.api.dependency import build_rag_settings, db_session
from ai_agents.api.schemas import (
    IngestRequest,
    IngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    SourcesListResponse,
    SourceRow,
)

from ai_agents.rag.ingest import ingest_files
from ai_agents.rag.query import answer_langgraph  
from ai_agents.db.models import RagSource
from sqlalchemy import select


router = APIRouter(prefix="/v1/rag", tags=["rag"])


@router.post("/query", response_model=RagQueryResponse)
def rag_query(request: RagQueryRequest) -> RagQueryResponse:
    
    settings = build_rag_settings(
        namespace=request.namespace,
        collection_name=request.collection_name,
        preferred_collections=request.preferred_collections,
        enable_query_expansion=request.enable_query_expansion,
    )

    result = answer_langgraph(request.question, settings)

    # Standardize outward shape for clients
    meta: Dict[str, Any] = {
        "question": result.get("question"),
        "queries": result.get("queries", []),
        "verification": result.get("verification", {}),
        "retrieval": result.get("retrieval", {}),
        "attempt": result.get("attempt"),
        "max_attempts": result.get("max_attempts"),
        "error": result.get("error"),
        "question_original": result.get("question_original"),
        "rewritten_question": result.get("rewritten_question"),
    }

    return RagQueryResponse(answer=result.get("answer", ""), meta=meta)




@router.post("/ingest", response_model=IngestResponse)
def rag_ingest(request: IngestRequest, background: bool = Query(False)) -> IngestResponse:
    
    settings = build_rag_settings(namespace=request.namespace)

    # If you want non-blocking ingestion, push it to background tasks.
    # NOTE: BackgroundTasks still runs in-process. For real jobs, swap this to Celery/RQ/SQS.
    if background:
        # We can’t return the chunk count in this mode without a job store.
        # So return a “queued” response and add a job system later.
        from fastapi import BackgroundTasks

        tasks = BackgroundTasks()
        tasks.add_task(ingest_files, request.paths, settings)
        return IngestResponse(
            ingested_chunks=0,
            meta={"queued": True, "paths": request.paths, "namespace": settings.namespace},
        )

    try:
        count = ingest_files(request.paths, settings)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest failed: {type(e).__name__}: {e}") from e

    return IngestResponse(
        ingested_chunks=count,
        meta={"paths": request.paths, "namespace": settings.namespace},
    )




@router.get("/sources", response_model=SourcesListResponse)
def list_sources(
    namespace: str | None = None,
    collection_name: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> SourcesListResponse:
    with db_session() as db:
        stmt = select(RagSource)

        if namespace:
            stmt = stmt.where(RagSource.namespace == namespace)
        if collection_name:
            stmt = stmt.where(RagSource.collection_name == collection_name)

        stmt = stmt.order_by(RagSource.updated_at.desc()).limit(limit).offset(offset)

        rows: List[RagSource] = list(db.scalars(stmt).all())

    return SourcesListResponse(
        sources=[
            SourceRow(
                id=row.id,
                source_uri=row.source_uri,
                content_hash=row.content_hash,
                collection_name=row.collection_name,
                namespace=row.namespace,
                chunk_size=row.chunk_size,
                chunk_overlap=row.chunk_overlap,
            )
            for row in rows
        ]
    )