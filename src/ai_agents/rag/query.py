from langsmith import traceable
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from .chain import build_rag_chain
from .embeddings import build_ollama_embeddings
from .settings import RagSettings
from .vectorstore import build_qdrant, build_retriever
from .query_translations.query_expansion import expand_queries
from .query_translations.re_rank import rerank_docs


@traceable(name="rag_answer", tags=["rag", "query-expansion", "rerank"])
def answer(question: str, settings: RagSettings) -> str:
    
    embeddings = build_ollama_embeddings(settings.embedding_model) # Embedding model

    # initialize the vector database
    vs = build_qdrant(settings=settings, embedding_fn=embeddings)

    base_retriever = build_retriever(vs, settings.k) # fetch documents

    # 1) multi-query expansion
    queries = expand_queries(question, chat_model=settings.chat_model, n=5)

    # 2) retrieve for each query (smaller k each)
    k_per_query = max(2, settings.k // 2)
    all_docs: list[Document] = []
    
    for q in queries:
        all_docs.extend(vs.as_retriever(search_kwargs={"k": k_per_query}).invoke(q))

    # 3) Remove duplicate chunks 
    seen = set()
    deduped: list[Document] = []

    for doc in all_docs:
        key = (
            doc.metadata.get("source_uri"),
            doc.metadata.get("chunk_index"),
            doc.metadata.get("chunk_hash"),
        )

        if key not in seen:
            seen.add(key)
            deduped.append(doc)

    # 4) rerank down to final k
    final_docs = rerank_docs(question, deduped, chat_model=settings.chat_model, top_k=settings.k)

    # 5) Make a “static retriever” that always returns final_docs
    static_retriever = RunnableLambda(lambda _: final_docs)

    # debug print (optional)
    print("\n--- RETRIEVED DOCS (after expansion + rerank) ---")
    for i, doc in enumerate(final_docs, 1):
        src = doc.metadata.get("source_uri", doc.metadata.get("source", "unknown"))
        print(f"[{i}] {src}")
        print(doc.page_content[:300].replace("\n", " "))
        print()

    chain = build_rag_chain(static_retriever, settings.chat_model) # RAG pipeline

    return chain.invoke(question) # Returns LLM response