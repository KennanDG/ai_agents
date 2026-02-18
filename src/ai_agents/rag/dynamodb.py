from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

def _dynamodb():
    return boto3.resource("dynamodb")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())



SOURCES_TABLE = os.environ.get("SOURCES_TABLE", "rag_sources")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "rag_ingest_jobs")



def sources_table():
    return _dynamodb().Table(SOURCES_TABLE)


def jobs_table():
    return _dynamodb().Table(JOBS_TABLE)


def source_pk(namespace: str) -> str:
    return f"NS#{namespace}"


def source_sk(source_uri: str) -> str:
    return f"SRC#{source_uri}"


def job_pk(job_id: str) -> str:
    return f"JOB#{job_id}"


def upsert_source_if_changed(
    *,
    namespace: str,
    source_uri: str,
    content_hash: str,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> bool:
    """
    Returns True if unchanged, False if updated/created.
    """
    tbl = sources_table()
    pk = source_pk(namespace)
    sk = source_sk(source_uri)

    # Get existing item
    res = tbl.get_item(Key={"pk": pk, "sk": sk})
    item = res.get("Item")

    if item:
        if (
            item.get("content_hash") == content_hash
            and item.get("collection_name") == collection_name
            and int(item.get("chunk_size", 0)) == int(chunk_size)
            and int(item.get("chunk_overlap", 0)) == int(chunk_overlap)
        ):
            return True  # unchanged

    # write/update
    tbl.put_item(
        Item={
            "pk": pk,
            "sk": sk,
            "namespace": namespace,
            "source_uri": source_uri,
            "content_hash": content_hash,
            "collection_name": collection_name,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "updated_at": _now_iso(),
        }
    )
    return False


def list_sources(
    *,
    namespace: str,
    collection_name: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    I'll page and then slice.
    Good enough for now; later add a cursor (LastEvaluatedKey).
    """
    table = sources_table()
    pk = source_pk(namespace)

    res = table.query(
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": pk},
        Limit=min(1000, max(limit + offset, 1)),
    )

    items = res.get("Items", [])

    if collection_name:
        items = [x for x in items if x.get("collection_name") == collection_name]

    # mimic offset/limit
    return items[offset : offset + limit]


def create_job(
    *,
    job_id: str,
    namespace: str,
    collection_name: str,
    paths: List[str],
) -> None:
    table = jobs_table()

    table.put_item(
        Item={
            "pk": job_pk(job_id),
            "job_id": job_id,
            "status": "QUEUED",
            "namespace": namespace,
            "collection_name": collection_name,
            "paths": paths,
            "ingested_chunks": 0,
            "error": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
    )


def update_job(
    *,
    job_id: str,
    status: str,
    ingested_chunks: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    tbl = jobs_table()
    UpdateExpression = "SET #st = :st, #err = :err"
    ExpressionAttributeNames = {"#st": "status", "#err": "error"}
    ExpressionAttributeValues: Dict[str, Any] = {":st": status, ":err": error}

    if ingested_chunks is not None:
        UpdateExpression += ", ingested_chunks = :c"
        ExpressionAttributeValues[":c"] = int(ingested_chunks)

    # if error is not None:
    #     UpdateExpression += ", error = :e"
    #     vals[":e"] = error

    tbl.update_item(
        Key={"pk": job_pk(job_id)},
        UpdateExpression=UpdateExpression,
        ExpressionAttributeNames=ExpressionAttributeNames,
        ExpressionAttributeValues=ExpressionAttributeValues,
    )



def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    table = jobs_table()
    res = table.get_item(Key={"pk": job_pk(job_id)})
    return res.get("Item")
