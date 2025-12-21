from __future__ import annotations
from pathlib import Path
from typing import Iterable

from .settings import RagSettings
from .loaders import load_text_files
from .splitter import split_docs
from .embeddings import build_ollama_embeddings
from .vectorstore import build_chroma, upsert_documents

def ingest_files(paths: Iterable[str | Path], settings: RagSettings) -> int:
    
    docs = load_text_files(paths)
    splits = split_docs(docs, settings.chunk_size, settings.chunk_overlap)

    embeddings = build_ollama_embeddings(settings.embedding_model)

    vs = build_chroma(
        embedding_fn=embeddings,
        persist_dir=settings.persist_dir,
        collection_name=f"{settings.collection_name}-{settings.namespace}",
    )

    # Basic upsert (no ids). Next step: pass stable ids.
    upsert_documents(vs, splits)
    
    return len(splits)