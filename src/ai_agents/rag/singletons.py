from __future__ import annotations
from functools import lru_cache

from .settings import RagSettings
from .embeddings import build_fastembed_embeddings
from .vectorstore import build_qdrant
from .chain import build_rag_chain

_VECTORSTORE = None
_RETRIEVER = None



# ---------------------------
# Embeddings
# ---------------------------

@lru_cache(maxsize=1)
def get_embeddings(model_name: str, chunk_size: int):
    return build_fastembed_embeddings(model_name, chunk_size)


# ---------------------------
# Vectorstore / Retriever
# ---------------------------

def get_vectorstore(settings: RagSettings):
    global _VECTORSTORE
    if _VECTORSTORE is None:
        emb = build_fastembed_embeddings(settings.embedding_model, settings.chunk_size)
        _VECTORSTORE = build_qdrant(settings=settings, embedding_fn=emb)
    return _VECTORSTORE


def get_retriever(settings: RagSettings):
    global _RETRIEVER
    if _RETRIEVER is None:
        vs = get_vectorstore(settings)
        _RETRIEVER = vs.as_retriever(search_kwargs={"k": settings.k_per_query})
    return _RETRIEVER


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