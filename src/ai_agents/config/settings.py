import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        extra="ignore"
        )

    # Ollama
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_url: str = Field(default="http://host.docker.internal:11434", alias="OLLAMA_URL")
    chat_model: str = Field(default="llama3.1:8b", alias="CHAT_MODEL")                   # Main LLM
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")    # Doc embedding
    query_model: str = Field(default="qwen2.5:3b-instruct", alias="QUERY_MODEL")         # Query translation
    caption_model: str = Field(default="moondream:1.8b", alias="CAPTION_MODEL")                # VLM

    # Cross-encoder reranker (sentence-transformers)
    rerank_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANK_MODEL")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="rag-default", alias="QDRANT_COLLECTION")

    # DB
    database_url: str = Field(
        default="postgresql+psycopg://ai_agents:ai_agents@localhost:5432/ai_agents",
        alias="DATABASE_URL",
    )

    # Retrieval
    k: int = Field(default=20, alias="K")
    candidate_k: int = Field(default=50, alias="CANDIDATE_K")   # docs kept after RRF before rerank
    k_per_query: int = Field(default=10, alias="K_PER_QUERY")    # docs retrieved per expanded query
    rrf_k: int = Field(default=60, alias="RRF_K")               # RRF constant
    n_query_expansions: int = Field(default=5, alias="N_QUERY_EXPANSIONS")



settings = Settings()