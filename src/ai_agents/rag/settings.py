import os
from pydantic import BaseModel
from ai_agents.config.settings import settings


class RagSettings(BaseModel):
    """Settings for the LangGraph RAG pipeline.

    This wrapper exists so the LangGraph workflow can rely on a single typed
    settings object, while still inheriting defaults from the global
    ai_agents.config.settings.settings.

    Use getattr(...) when pulling from the global settings so older configs
    don't crash on missing fields.
    """


    # -------------------------
    # Infrastructure defaults
    # -------------------------
    # ollama_host: str = getattr(settings, "ollama_host", "http://localhost:11434")
    # ollama_url: str = getattr(settings, "ollama_url", getattr(settings, "ollama_host", "http://localhost:11434"))
    qdrant_url: str = getattr(settings, "qdrant_url", "http://localhost:6333")
    groq_api_key: str = getattr(settings, "groq_api_key", os.environ.get("GROQ_API_KEY"))
    groq_api_url: str = getattr(settings, "groq_api_url", os.environ.get("GROQ_API_URL"))

    # -------------------------
    # Model selection
    # -------------------------
    embedding_model: str = getattr(settings, "embedding_model", "nomic-ai/nomic-embed-text-v1.5")
    chat_model: str = getattr(settings, "chat_model", "llama-3.1-8b-instant")
    query_model: str = getattr(settings, "query_model", getattr(settings, "chat_model", "llama-3.1-8b-instant"))
    verify_model: str = getattr(settings, "verify_model", getattr(settings, "chat_model", "llama-3.1-8b-instant"))
    caption_model: str = getattr(settings, "caption_model", "meta-llama/llama-4-scout-17b-16e-instruct")
    rerank_model: str = getattr(settings, "rerank_model", "BAAI/bge-reranker-base")
    rerank_device: str = getattr(settings, "rerank_device", "cpu") 
    

    # -------------------------
    # Retrieval configuration
    # -------------------------
    k: int = int(getattr(settings, "k", 8))
    candidate_k: int = int(getattr(settings, "candidate_k", 100))
    k_per_query: int = int(getattr(settings, "k_per_query", 10))
    rrf_k: int = int(getattr(settings, "rrf_k", 60))

    # Query expansion
    enable_query_expansion: bool = bool(getattr(settings, "enable_query_expansion", True))
    n_query_expansions: int = int(getattr(settings, "n_query_expansions", 2))
    min_question_chars_for_expansion: int = int(getattr(settings, "min_question_chars_for_expansion", 25))

    # -------------------------
    # Retry & safety controls
    # -------------------------
    retrieve_attempts: int = int(getattr(settings, "retrieve_attempts", 2))
    generate_attempts: int = int(getattr(settings, "generate_attempts", 2))
    verify_attempts: int = int(getattr(settings, "verify_attempts", 2))

    # max number of verification failures allowed before we stop retrying
    max_rag_attempts: int = int(getattr(settings, "max_rag_attempts", 2))


    # -------------------------
    # Verification prompt context shaping
    # -------------------------
    verify_max_chars: int = int(getattr(settings, "verify_max_chars", 6000))


    # -------------------------
    # Chunking
    # -------------------------
    chunk_size: int = 512
    chunk_overlap: int = 50
    collection_name: str = "rag-default"
    namespace: str = "default"