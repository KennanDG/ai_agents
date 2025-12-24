from langchain_ollama import OllamaEmbeddings


def build_ollama_embeddings(model: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model)