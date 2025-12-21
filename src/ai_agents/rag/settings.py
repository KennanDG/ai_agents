from pydantic import BaseModel, Field

class RagSettings(BaseModel):
    # Models
    embedding_model: str = Field(default="nomic-embed-text")
    chat_model: str = Field(default="llama3.1:8b")

    # Chunking
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)

    # Retrieval
    k: int = Field(default=3)

    # Storage
    collection_name: str = Field(default="rag")
    persist_dir: str = Field(default=".chroma")  # persistent local folder

    # Optional: namespace/versioning
    namespace: str = Field(default="default")