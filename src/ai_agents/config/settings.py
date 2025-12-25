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
    chat_model: str = Field(default="llama3.1:8b", alias="CHAT_MODEL")
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="rag-default", alias="QDRANT_COLLECTION")

    # DB
    database_url: str = Field(
        default="postgresql+psycopg://ai_agents:ai_agents@localhost:5432/ai_agents",
        alias="DATABASE_URL",
    )

    # Retrieval
    k: int = Field(default=8, alias="K")


settings = Settings()