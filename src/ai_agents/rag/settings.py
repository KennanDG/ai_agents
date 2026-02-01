import os
from pydantic import BaseModel
from ai_agents.config.settings import settings


class RagSettings(BaseModel):

    # Inherit from global settings
    ollama_host: str = settings.ollama_host
    ollama_url: str = settings.ollama_url
    qdrant_url: str = settings.qdrant_url
    embedding_model: str = settings.embedding_model
    chat_model: str = settings.chat_model
    query_model: str = settings.query_model
    rerank_model: str = settings.rerank_model
    rerank_device: str = settings.rerank_device
    caption_model: str = settings.caption_model
    k: int = settings.k
    candidate_k: int = settings.candidate_k
    k_per_query: int = settings.k_per_query
    rrf_k: int = settings.rrf_k
    n_query_expansions: int = settings.n_query_expansions

    # RAG-specific behavior
    chunk_size: int = 500
    chunk_overlap: int = 50
    collection_name: str = "rag-default"
    namespace: str = "default"