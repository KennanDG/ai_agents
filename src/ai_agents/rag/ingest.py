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

    # attach metadata + build stable IDs
    ids: list[str] = []
    for i, doc in enumerate(splits):
        # assuming loaders set metadata["path"] already
        file_path = doc.metadata.get("path")
        source_uri = f"file:{file_path}" if file_path else "unknown"

        # compute full file hash if we can
        content_hash = _sha256_file(Path(file_path)) if file_path else _sha256_text(doc.page_content)

        chunk_hash = _sha256_text(doc.page_content)
        point_id = _make_point_id(source_uri, content_hash, i, chunk_hash)

        doc.metadata.update({
            "source_uri": source_uri,
            "content_hash": content_hash,
            "chunk_index": i,
            "chunk_hash": chunk_hash,
        })
        ids.append(point_id)

    # easiest true-idempotent behavior: delete all existing points for each source_uri
    # (dedupe sources)
    for src in sorted({doc.metadata["source_uri"] for doc in splits if "source_uri" in doc.metadata}):
        delete_source(vs, src)

    # Basic upsert (no ids). Next step: pass stable ids.
    upsert_documents(vs, splits)
    
    return len(splits)