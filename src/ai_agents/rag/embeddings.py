from langchain_ollama import OllamaEmbeddings
from langsmith import traceable

@traceable
def build_ollama_embeddings(model: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model)