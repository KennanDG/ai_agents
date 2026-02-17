from __future__ import annotations

from typing import Any, Dict, List
import json

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from ai_agents.api.dependency import build_retrieval_settings, build_ingestion_settings, db_session
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
from ai_agents.rag.dynamodb import create_job, get_job, list_sources
from ai_agents.db.models import RagSource, RagIngestJob
from ai_agents.jobs.tasks import ingest_job
from sqlalchemy import select


router = APIRouter(prefix="/v1/rag", tags=["rag"])


@router.post("/query", response_model=RagQueryResponse)
def rag_query(request: RagQueryRequest) -> RagQueryResponse:
    
    settings = build_retrieval_settings(
        k=request.k,
        namespace=request.namespace,
        collection_name=request.collection_name,
        preferred_collections=request.preferred_collections,
        enable_query_expansion=request.enable_query_expansion,
        enable_parallel_collection_retrieval=request.enable_parallel_collection_retrieval
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
    
    settings = build_ingestion_settings(
        namespace=request.namespace,
        collection_name=request.collection_name
    )

    if background:
        # Enqueue Celery job
        async_result = ingest_job.delay(
            paths=request.paths,
            namespace=settings.namespace,
            collection_name=settings.collection_name,
        )

        job_id = async_result.id

        create_job(
            job_id=job_id,
            namespace=settings.namespace,
            collection_name=settings.collection_name,
            paths=request.paths,
        )

        # # Create/record a job row so clients can poll status
        # with db_session() as db:

        #     db.add(
        #         RagIngestJob(
        #             job_id=job_id,
        #             status="QUEUED",
        #             namespace=settings.namespace,
        #             collection_name=settings.collection_name,
        #             paths_json=json.dumps(request.paths),
        #             ingested_chunks=0,
        #             error=None,
        #         )
        #     )

        #     db.commit()

        return IngestResponse(
            ingested_chunks=0,
            meta={
                "queued": True,
                "job_id": job_id,
                "paths": request.paths,
                "namespace": settings.namespace,
                "collection_name": settings.collection_name,
                "status_url": f"/v1/rag/ingest/jobs/{job_id}",
            },
        )

    
    # in-sync path (dev only)
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





@router.get("/ingest/jobs/{job_id}")
def ingest_job_status(job_id: str) -> dict:

    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return row
    
    # with db_session() as db:
    #     row = db.query(RagIngestJob).filter(RagIngestJob.job_id == job_id).one_or_none()
        
    #     if not row:
    #         raise HTTPException(status_code=404, detail="job not found")

    #     return {
    #         "job_id": row.job_id,
    #         "status": row.status,
    #         "namespace": row.namespace,
    #         "collection_name": row.collection_name,
    #         "ingested_chunks": row.ingested_chunks,
    #         "error": row.error,
    #         "created_at": row.created_at,
    #         "updated_at": row.updated_at,
    #     }






@router.get("/sources", response_model=SourcesListResponse)
def list_sources(
    namespace: str | None = None,
    collection_name: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> SourcesListResponse:
    ns = namespace or "default"  # or enforce required
    
    items = list_sources(
        namespace=ns, 
        collection_name=collection_name, 
        limit=limit, 
        offset=offset
    )

    return SourcesListResponse(
    sources=[
        SourceRow(
        # id=0, 
        source_uri=item["source_uri"],
        content_hash=item["content_hash"],
        collection_name=item["collection_name"],
        namespace=item["namespace"],
        chunk_size=int(item["chunk_size"]),
        chunk_overlap=int(item["chunk_overlap"]),
        )
        for item in items
    ]
    )
    # with db_session() as db:
    #     stmt = select(RagSource)

    #     if namespace:
    #         stmt = stmt.where(RagSource.namespace == namespace)
            
    #     if collection_name:
    #         stmt = stmt.where(RagSource.collection_name == collection_name)

    #     stmt = stmt.order_by(RagSource.updated_at.desc()).limit(limit).offset(offset)

    #     rows: List[RagSource] = list(db.scalars(stmt).all())

    # return SourcesListResponse(
    #     sources=[
    #         SourceRow(
    #             id=row.id,
    #             source_uri=row.source_uri,
    #             content_hash=row.content_hash,
    #             collection_name=row.collection_name,
    #             namespace=row.namespace,
    #             chunk_size=row.chunk_size,
    #             chunk_overlap=row.chunk_overlap,
    #         )
    #         for row in rows
    #     ]
    # )