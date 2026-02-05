from __future__ import annotations
from functools import lru_cache
from typing import Optional

from .settings import RagSettings
from .embeddings import build_fastembed_embeddings
from .vectorstore import build_qdrant



# ---------------------------
# Embeddings
# ---------------------------

@lru_cache(maxsize=1)
def get_embeddings(model_name: str, chunk_size: int):
    return build_fastembed_embeddings(model_name, chunk_size)


# ---------------------------
# Vectorstore / Retriever
# ---------------------------

# Cannot use RagSettings as a parameter because it is not hashable for lru_cache
@lru_cache(maxsize=32)
def _get_vectorstore_cached( 
    qdrant_url: str,
    embedding_model: str,
    chunk_size: int,
    collection_name: str,
    namespace: str,
):
    """
    Cache vectorstores by (url, model, chunk_size, collection, namespace).
    """
    # Avoid instantiating a new RagSettings in case of required values
    class _Settings:
        pass

    settings = _Settings()
    settings.qdrant_url = qdrant_url
    settings.embedding_model = embedding_model
    settings.chunk_size = chunk_size
    settings.collection_name = collection_name
    settings.namespace = namespace

    emb = build_fastembed_embeddings(embedding_model, chunk_size)

    return build_qdrant(settings=settings, embedding_fn=emb)


def get_vectorstore(settings: RagSettings, *, collection_name_override: Optional[str] = None):
    collection_name = collection_name_override or settings.collection_name
    qdrant_url = settings.qdrant_url
    embedding_model = settings.embedding_model
    chunk_size = settings.chunk_size
    namespace = settings.namespace

    return _get_vectorstore_cached(
        qdrant_url, 
        embedding_model, 
        chunk_size, 
        collection_name, 
        namespace
    )


def get_retriever(settings: RagSettings, *, collection_name_override: Optional[str] = None):
    """Return retriever for a specific collection."""
    vs = get_vectorstore(settings, collection_name_override=collection_name_override)
    return vs.as_retriever(search_kwargs={"k": settings.k_per_query})


# ---------------------------
# RAG Chain
# ---------------------------

@lru_cache(maxsize=1)
def get_rag_chain(chat_model: str):
    # retriever is injected dynamically later
    return chat_model


# ---------------------------
# LangGraph
# ---------------------------

@lru_cache(maxsize=1)
def get_rag_graph():
    from .graph import build_rag_graph
    return build_rag_graph()