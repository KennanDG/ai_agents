from __future__ import annotations

import os
from celery import Celery


def make_celery() -> Celery:
    """
    Celery app configured for SQS broker.

    Expected env vars:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_REGION
      - CELERY_QUEUE (optional, default 'rag-ingest')
      - CELERY_VISIBILITY_TIMEOUT (optional)
    """
    queue = os.getenv("CELERY_QUEUE", "rag-ingest")
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION", "us-east-1")


    # Celery SQS broker URL format is usually: sqs://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}@
    broker_url = os.getenv("CELERY_BROKER_URL", "sqs://")

    app = Celery("ai_agents", broker=broker_url, include=["ai_agents.jobs.tasks"])

    app.conf.update(
        task_default_queue=queue,
        broker_transport_options={
            "region": region,
            "visibility_timeout": int(os.getenv("CELERY_VISIBILITY_TIMEOUT", "3600")),
            "polling_interval": float(os.getenv("CELERY_SQS_POLLING_INTERVAL", "1")),
            "queue_name_prefix": os.getenv("CELERY_QUEUE_PREFIX", "ai-agents-"),
        },
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,  # ingestion jobs can be heavy; avoid over-prefetch
    )

    return app


celery_app = make_celery()
