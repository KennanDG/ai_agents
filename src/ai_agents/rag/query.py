from .chain import build_rag_chain
from .embeddings import build_ollama_embeddings
from .settings import RagSettings
from .vectorstore import build_qdrant, build_retriever


def answer(question: str, settings: RagSettings) -> str:
    
    embeddings = build_ollama_embeddings(settings.embedding_model) # Embedding model

    # initialize the vector database
    vs = build_qdrant(settings=settings, embedding_fn=embeddings)

    retriever = build_retriever(vs, settings.k) # fetch documents

    docs = retriever.invoke(question)

    print("\n--- RETRIEVED DOCS ---")

    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        print(f"[{i}] source={src}")
        print(d.page_content[:300].replace("\n", " "))
        print()

    chain = build_rag_chain(retriever, settings.chat_model) # RAG pipeline

    return chain.invoke(question) # Returns LLM response