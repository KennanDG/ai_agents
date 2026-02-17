from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, List

from ai_agents.rag.settings import RagSettings
# from ai_agents.db.session import SessionLocal


def build_retrieval_settings(
    *,
    k: Optional[int] = None,
    namespace: Optional[str] = None,
    collection_name: Optional[str] = None,
    preferred_collections: Optional[List[str]] = None,
    enable_query_expansion: Optional[bool] = None,
    enable_parallel_collection_retrieval: Optional[bool] = None
) -> RagSettings:
    
    settings = RagSettings()

    # -------- RAG overrides --------
    if k is not None:
        settings.k = k

    if namespace is not None:
        settings.namespace = namespace

    if collection_name is not None:
        settings.collection_name = collection_name

    if preferred_collections is not None:
        settings.preferred_collections = preferred_collections

    if enable_query_expansion is not None:
        settings.enable_query_expansion = enable_query_expansion

    if enable_parallel_collection_retrieval is not None:
        settings.enable_parallel_collection_retrieval = enable_parallel_collection_retrieval

    return settings



def build_ingestion_settings(
    *,
    namespace: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> RagSettings:
    
    settings = RagSettings()

    if namespace is not None:
        settings.namespace = namespace

    if collection_name is not None:
        settings.collection_name = collection_name


# @contextmanager
# def db_session() -> Iterator:
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
