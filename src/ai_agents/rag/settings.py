import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        extra="ignore",
    )

    # Endpoints
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")

    # Models
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    chat_model: str = Field(default="llama3.1:8b", alias="CHAT_MODEL")

    # Retrieval
    k: int = Field(default=8, alias="K")

    # Chunking
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")

    # Storage
    collection_name: str = Field(default="rag-default", alias="QDRANT_COLLECTION")
    namespace: str = Field(default="default", alias="RAG_NAMESPACE")