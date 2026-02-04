from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langsmith import traceable

# Deprecated
@traceable
def build_ollama_embeddings(model: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model)

@traceable
def build_fastembed_embeddings(model: str = "nomic-ai/nomic-embed-text-v1.5", chunk_size: int = 512) -> FastEmbedEmbeddings:
    # This runs locally in your Python process using ONNX
    return FastEmbedEmbeddings(model_name=model, max_length=chunk_size)