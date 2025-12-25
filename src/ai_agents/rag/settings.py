import os
from pydantic import BaseModel
from ai_agents.config.settings import settings


class RagSettings(BaseModel):
    # Inherit infra from global settings
    ollama_host: str = settings.ollama_host
    qdrant_url: str = settings.qdrant_url
    embedding_model: str = settings.embedding_model
    chat_model: str = settings.chat_model
    k: int = settings.k

    # RAG-specific behavior
    chunk_size: int = 500
    chunk_overlap: int = 50
    collection_name: str = "rag-default"
    namespace: str = "default"