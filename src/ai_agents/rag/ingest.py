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


# ---------------- Create IDs for document embeddings ----------------
def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_point_id(source_uri: str, content_hash: str, chunk_index: int, chunk_hash: str) -> str:

    raw = f"{source_uri}|{content_hash}|{chunk_index}|{chunk_hash}"

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()




# ---------------- Ingest docs in vector database ----------------

@traceable
def ingest_files(paths: Iterable[str | Path], settings: RagSettings) -> int:

    paths = [Path(p) for p in paths] # Ensures parameter is Path data type
    
    # Split docs
    docs = load_text_files(paths)
    splits = split_docs(docs, settings.chunk_size, settings.chunk_overlap)

    # Embed
    embeddings = build_ollama_embeddings(settings.embedding_model)
    vs = build_qdrant(settings=settings, embedding_fn=embeddings)

    # Pre-compute file-level hashes (once per file)
    file_hashes: dict[str, str] = {}
    
    for doc in docs:
        relative_path = doc.metadata["path_rel"]
        file_hashes[relative_path] = _sha256_file(Path(doc.metadata["path"]))

    
    ids: list[str] = [] # attach metadata + build stable IDs

    chunk_counters: dict[str, int] = {} # chunk_index must be per source

    for doc in splits:
        relative_path = doc.metadata["path_rel"]
        source_uri = f"file:{relative_path}"

        chunk_index = chunk_counters.get(source_uri, 0)
        chunk_counters[source_uri] = chunk_index + 1

        content_hash = file_hashes[relative_path]
        chunk_hash = _sha256_text(doc.page_content)

        point_id = _make_point_id(
            source_uri=source_uri,
            content_hash=content_hash,
            chunk_index=chunk_index,
            chunk_hash=chunk_hash,
        )

        doc.metadata.update({
            "source_uri": source_uri,
            "content_hash": content_hash,
            "chunk_index": chunk_index,
            "chunk_hash": chunk_hash,
        })

        ids.append(point_id)

    # Delete existing vectors per source (true idempotency)
    for src in sorted(chunk_counters.keys()):
        delete_source(vs, src)
    

    upsert_documents(vs, splits, ids)
    
    return len(splits)


    # for i, doc in enumerate(splits):
    #     # assuming loaders set metadata["path"] already
    #     file_path = doc.metadata.get("path")
    #     source_uri = f"file:{file_path}" if file_path else "unknown"

    #     # compute full file hash if we can
    #     content_hash = _sha256_file(Path(file_path)) if file_path else _sha256_text(doc.page_content)

    #     chunk_hash = _sha256_text(doc.page_content)
    #     point_id = _make_point_id(source_uri, content_hash, i, chunk_hash)

    #     doc.metadata.update({
    #         "source_uri": source_uri,
    #         "content_hash": content_hash,
    #         "chunk_index": i,
    #         "chunk_hash": chunk_hash,
    #     })
    #     ids.append(point_id)

    # # easiest true-idempotent behavior: delete all existing points for each source_uri
    # # (dedupe sources)
    # for src in sorted({doc.metadata["source_uri"] for doc in splits if "source_uri" in doc.metadata}):
    #     delete_source(vs, src)

    # upsert_documents(vs, splits, ids)
    
    # return len(splits)