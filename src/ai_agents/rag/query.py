from .settings import RagSettings
from .embeddings import build_ollama_embeddings
from .vectorstore import build_chroma, build_retriever
from .chain import build_rag_chain

def answer(question: str, settings: RagSettings) -> str:
    
    embeddings = build_ollama_embeddings(settings.embedding_model) # Embedding model

    # initialize the vector database
    vs = build_chroma(
        embedding_fn=embeddings,
        persist_dir=settings.persist_dir,
        collection_name=f"{settings.collection_name}-{settings.namespace}",
    )

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