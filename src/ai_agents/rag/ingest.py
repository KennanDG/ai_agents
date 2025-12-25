from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib
from langsmith import traceable

from .embeddings import build_ollama_embeddings
from .loaders import load_text_files
from .settings import RagSettings
from .splitter import split_docs
from .vectorstore import build_qdrant, delete_source, upsert_documents
from ai_agents.db.session import SessionLocal
from ai_agents.rag.repo import upsert_source, replace_chunks
from sqlalchemy import select
from ai_agents.db.models import RagChunk
import uuid
from ai_agents.config.constants import QDRANT_ID_NAMESPACE


# ---------------- Create IDs for document embeddings ----------------
def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_point_id(source_uri: str, content_hash: str, chunk_index: int, chunk_hash: str) -> str:

    raw = f"{source_uri}|{content_hash}|{chunk_index}|{chunk_hash}"

    return str(uuid.uuid5(QDRANT_ID_NAMESPACE, raw))




# ---------------- Ingest docs in vector database ----------------


@traceable
def ingest_files(paths: Iterable[str | Path], settings: RagSettings) -> int:
    paths = [Path(p) for p in paths]

    docs = load_text_files(paths)
    splits = split_docs(docs, settings.chunk_size, settings.chunk_overlap)

    embeddings = build_ollama_embeddings(settings.embedding_model)
    vs = build_qdrant(settings=settings, embedding_fn=embeddings)


    # Pre-compute file-level hashes once per source_uri
    file_hashes: dict[str, str] = {}

    for doc in docs:
        relative_path = doc.metadata["path_rel"]
        file_hashes[relative_path] = _sha256_file(Path(doc.metadata["path"]))


    # Group splits by source_uri so chunk_index is per file
    by_source: dict[str, list] = {}

    for doc in splits:
        source_uri = doc.metadata["source_uri"]  # from loaders.py
        by_source.setdefault(source_uri, []).append(doc)

    
    total = 0

    with SessionLocal() as db:
        for source_uri, chunk_docs in by_source.items():
            relative_path = chunk_docs[0].metadata["path_rel"]
            content_hash = file_hashes[relative_path]

            # 1) Upsert source & check unchanged
            src_row, unchanged = upsert_source(
                db,
                source_uri=source_uri,
                content_hash=content_hash,
                collection_name=settings.collection_name,
                namespace=settings.namespace,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )


            # nothing changed, do not touch qdrant
            if unchanged:
                continue 

            # 2) Build chunk rows (stable point ids)
            chunks_payload: list[dict] = []
            ids: list[str] = []

            for idx, doc in enumerate(chunk_docs):
                chunk_hash = _sha256_text(doc.page_content)
                point_id = _make_point_id(source_uri, content_hash, idx, chunk_hash)

                # metadata for qdrant payload
                doc.metadata.update({
                    "source_uri": source_uri,
                    "content_hash": content_hash,
                    "chunk_index": idx,
                    "chunk_hash": chunk_hash,
                    "pg_source_id": src_row.id,
                })

                chunks_payload.append({
                    "chunk_index": idx,
                    "chunk_hash": chunk_hash,
                    "qdrant_point_id": point_id,
                })
                ids.append(point_id)

            # 3) Replace chunks in Postgres
            chunk_rows = replace_chunks(db, source_id=src_row.id, chunks=chunks_payload)

            # attach pg_chunk_id for traceability/debugging
            for doc, row in zip(chunk_docs, chunk_rows):
                doc.metadata["pg_chunk_id"] = row.id

            # 4) Qdrant index update (index follows Postgres)
            try:
                delete_source(vs, source_uri)
                upsert_documents(vs, chunk_docs, ids)
                db.commit()
                
            except Exception:
                db.rollback()
                raise

            total += len(chunk_docs)

    return total






# @traceable
# def ingest_files(paths: Iterable[str | Path], settings: RagSettings) -> int:

#     paths = [Path(p) for p in paths] # Ensures parameter is Path data type
    
#     # Split docs
#     docs = load_text_files(paths)
#     splits = split_docs(docs, settings.chunk_size, settings.chunk_overlap)

#     # Embed
#     embeddings = build_ollama_embeddings(settings.embedding_model)
#     vs = build_qdrant(settings=settings, embedding_fn=embeddings)

#     # Pre-compute file-level hashes (once per file)
#     file_hashes: dict[str, str] = {}
    
#     for doc in docs:
#         relative_path = doc.metadata["path_rel"]
#         file_hashes[relative_path] = _sha256_file(Path(doc.metadata["path"]))

    
#     ids: list[str] = [] # attach metadata + build stable IDs

#     chunk_counters: dict[str, int] = {} # chunk_index must be per source

#     for doc in splits:
#         relative_path = doc.metadata["path_rel"]
#         source_uri = f"file:{relative_path}"

#         chunk_index = chunk_counters.get(source_uri, 0)
#         chunk_counters[source_uri] = chunk_index + 1

#         content_hash = file_hashes[relative_path]
#         chunk_hash = _sha256_text(doc.page_content)

#         point_id = _make_point_id(
#             source_uri=source_uri,
#             content_hash=content_hash,
#             chunk_index=chunk_index,
#             chunk_hash=chunk_hash,
#         )

#         doc.metadata.update({
#             "source_uri": source_uri,
#             "content_hash": content_hash,
#             "chunk_index": chunk_index,
#             "chunk_hash": chunk_hash,
#         })

#         ids.append(point_id)

#     # Delete existing vectors per source (true idempotency)
#     for src in sorted(chunk_counters.keys()):
#         delete_source(vs, src)
    

#     upsert_documents(vs, splits, ids)
    
#     return len(splits)


#     # for i, doc in enumerate(splits):
#     #     # assuming loaders set metadata["path"] already
#     #     file_path = doc.metadata.get("path")
#     #     source_uri = f"file:{file_path}" if file_path else "unknown"

#     #     # compute full file hash if we can
#     #     content_hash = _sha256_file(Path(file_path)) if file_path else _sha256_text(doc.page_content)

#     #     chunk_hash = _sha256_text(doc.page_content)
#     #     point_id = _make_point_id(source_uri, content_hash, i, chunk_hash)

#     #     doc.metadata.update({
#     #         "source_uri": source_uri,
#     #         "content_hash": content_hash,
#     #         "chunk_index": i,
#     #         "chunk_hash": chunk_hash,
#     #     })
#     #     ids.append(point_id)

#     # # easiest true-idempotent behavior: delete all existing points for each source_uri
#     # # (dedupe sources)
#     # for src in sorted({doc.metadata["source_uri"] for doc in splits if "source_uri" in doc.metadata}):
#     #     delete_source(vs, src)

#     # upsert_documents(vs, splits, ids)
    
#     # return len(splits)