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


    # Groq
    chat_model: str = Field(default="llama-3.1-8b-instant", alias="CHAT_MODEL")                   # Main LLM
    query_model: str = Field(default="llama-3.1-8b-instant", alias="QUERY_MODEL")         # Query translation   "qwen2.5:3b-instruct"
    caption_model: str = Field(default="meta-llama/llama-4-scout-17b-16e-instruct", alias="CAPTION_MODEL")                # VLM
    verify_model: str = Field(default="llama-3.1-8b-instant", alias="VERIFY_MODEL")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_api_url: str | None = Field(default="https://api.groq.com/openai/v1", alias="GROQ_API_URL")
    

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="rag-default", alias="QDRANT_COLLECTION")

    # FastEmbed
    embedding_model: str = Field(default="nomic-ai/nomic-embed-text-v1.5", alias="EMBEDDING_MODEL")    # Doc embedding
    rerank_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANK_MODEL")
    rerank_device: str = Field(default="cpu", alias="RERANK_DEVICE") 


    # DB
    database_url: str = Field(
        default="postgresql+psycopg://ai_agents:ai_agents@localhost:5432/ai_agents",
        alias="DATABASE_URL",
    )

    # Retrieval
    k: int = Field(default=8, alias="K")
    candidate_k: int = Field(default=30, alias="CANDIDATE_K")   # docs kept after RRF before rerank
    k_per_query: int = Field(default=8, alias="K_PER_QUERY")    # docs retrieved per expanded query
    rrf_k: int = Field(default=60, alias="RRF_K")               # RRF constant
    n_query_expansions: int = Field(default=2, alias="N_QUERY_EXPANSIONS")
    enable_query_expansion: bool = Field(default=True, alias="ENABLE_QUERY_EXPANSION")
    min_question_chars_for_expansion: int = Field(default=25, alias="MIN_QUESTION_CHARS_FOR_EXPANSION")

    # Generation
    max_rag_attempts: int = Field(default=2, alias="MAX_RAG_ATTEMPTS")
    retrieve_attempts: int = Field(default=2, alias="RETRIEVE_ATTEMPTS")
    generate_attempts: int = Field(default=2, alias="GENERATE_ATTEMPTS")
    verify_attempts: int = Field(default=2, alias="VERIFY_ATTEMPTS")
    verify_max_chars: int = Field(default=6_000, alias="VERIFY_MAX_CHARS")



settings = Settings()