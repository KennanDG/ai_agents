from __future__ import annotations

import json
from typing import Any, Dict, List

from celery.utils.log import get_task_logger

from ai_agents.jobs.celery import celery_app
from ai_agents.rag.ingest import ingest_files
from ai_agents.api.dependency import db_session, build_ingestion_settings
from ai_agents.db.models import RagIngestJob

logger = get_task_logger(__name__)


@celery_app.task(
    bind=True,
    name="ai_agents.rag.ingest_job",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=2,
)
def ingest_job(
    self,
    *,
    paths: List[str],
    namespace: str,
    collection_name: str,
) -> Dict[str, Any]:
    """
    Celery job to ingest files.
    """
    job_id = self.request.id

    # Mark STARTED
    with db_session() as db:
        row = db.query(RagIngestJob).filter(RagIngestJob.job_id == job_id).one_or_none()

        if row:
            row.status = "STARTED"

        db.commit()

    settings = build_ingestion_settings(namespace=namespace, collection_name=collection_name)

    try:
        count = ingest_files(paths, settings)

        with db_session() as db:
            row = db.query(RagIngestJob).filter(RagIngestJob.job_id == job_id).one_or_none()
            
            if row:
                row.status = "SUCCEEDED"
                row.ingested_chunks = int(count or 0)
                row.error = None

            db.commit()

        return {"job_id": job_id, "status": "SUCCEEDED", "ingested_chunks": int(count or 0)}

    except Exception as e:
        logger.exception("Ingest job failed: %s", e)

        with db_session() as db:
            row = db.query(RagIngestJob).filter(RagIngestJob.job_id == job_id).one_or_none()
            
            if row:
                row.status = "FAILED"
                row.error = f"{type(e).__name__}: {e}"

            db.commit()

        raise
