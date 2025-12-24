from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .embeddings import build_ollama_embeddings
from .loaders import load_text_files
from .settings import RagSettings
from .splitter import split_docs
from .vectorstore import build_qdrant, upsert_documents


def ingest_files(paths: Iterable[str | Path], settings: RagSettings) -> int:
    
    docs = load_text_files(paths)
    splits = split_docs(docs, settings.chunk_size, settings.chunk_overlap)

    embeddings = build_ollama_embeddings(settings.embedding_model)

    vs = build_qdrant(settings=settings, embedding_fn=embeddings)

    # Basic upsert (no ids). Next step: pass stable ids.
    upsert_documents(vs, splits)
    
    return len(splits)