from __future__ import annotations

from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from ai_agents.db.models import RagSource, RagChunk


def get_source(db: Session, source_uri: str) -> RagSource | None:
    return db.scalar(select(RagSource).where(RagSource.source_uri == source_uri))


def upsert_source(
    db: Session,
    *,
    source_uri: str,
    content_hash: str,
    collection_name: str,
    namespace: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[RagSource, bool]:
    """
    Returns (source_row, unchanged).

    unchanged=True means:
    - same content_hash
    - same collection_name/namespace
    - same chunk params
    """
    src = get_source(db, source_uri)

    if src:
        unchanged = (
            src.content_hash == content_hash
            and src.collection_name == collection_name
            and src.namespace == namespace
            and src.chunk_size == chunk_size
            and src.chunk_overlap == chunk_overlap
        )

        if unchanged:
            return src, True

        src.content_hash = content_hash
        src.collection_name = collection_name
        src.namespace = namespace
        src.chunk_size = chunk_size
        src.chunk_overlap = chunk_overlap
        db.add(src)
        db.flush()

        return src, False

    src = RagSource(
        source_uri=source_uri,
        content_hash=content_hash,
        collection_name=collection_name,
        namespace=namespace,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    db.add(src)
    db.flush()

    return src, False


def replace_chunks(
    db: Session,
    *,
    source_id: int,
    chunks: list[dict],
) -> list[RagChunk]:
    """
    chunks: list of dicts with keys: chunk_index, chunk_hash, qdrant_point_id
    """
    db.execute(delete(RagChunk).where(RagChunk.source_id == source_id))

    rows = [
        RagChunk(
            source_id=source_id,
            chunk_index=c["chunk_index"],
            chunk_hash=c["chunk_hash"],
            qdrant_point_id=c["qdrant_point_id"],
        )
        for c in chunks
    ]
    db.add_all(rows)
    db.flush()
    
    return rows
