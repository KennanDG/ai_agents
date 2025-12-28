from langsmith import traceable
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from .chain import build_rag_chain
from .embeddings import build_ollama_embeddings
from .settings import RagSettings
from .vectorstore import build_qdrant, build_retriever
from .query_translations.query_expansion import expand_queries
from .query_translations.rag_fusion import rrf_fuse
from .query_translations.cross_encoder import cross_encoder_rerank


@traceable(name="rag_answer", tags=["rag", "query-expansion", "rrf-fusion", "cross-encoder-rerank"])
def answer(question: str, settings: RagSettings) -> str:
    
    embeddings = build_ollama_embeddings(settings.embedding_model) # Embedding model

    # initialize the vector database
    vs = build_qdrant(settings=settings, embedding_fn=embeddings)

    # base_retriever = build_retriever(vs, settings.k) # fetch documents

    # 1) multi-query expansion
    queries = expand_queries(
        question, 
        chat_model=settings.query_model, 
        n=settings.n_query_expansions
    )
    

    # 2) retrieve for each query (smaller k each)
    retriever = vs.as_retriever(search_kwargs={"k": settings.k_per_query})
    results_by_query: list[list[Document]] = []
    
    for q in queries:
        results_by_query.append(retriever.invoke(q))


    # 3) RAG-Fusion: fuse ranked lists across queries (RRF)
    fused_docs = rrf_fuse(
        results_by_query=results_by_query,
        k=settings.candidate_k,
        rrf_k=settings.rrf_k
    )


    # 4) Cross-encoder rerank down to final k
    final_docs = cross_encoder_rerank(
        question=question,
        docs=fused_docs,
        model_name=settings.rerank_model,
        top_k=settings.k
    )

    # 5) Make a “static retriever” that always returns final_docs
    static_retriever = RunnableLambda(lambda _: final_docs)

    # debug print
    print("\n--- RETRIEVED DOCS (after expansion + RRF + cross-encoder rerank) ---")
    for i, doc in enumerate(final_docs, 1):
        src = doc.metadata.get("source_uri", doc.metadata.get("source", "unknown"))
        print(f"[{i}] {src}")
        print(doc.page_content[:300].replace("\n", " "))
        print()

    chain = build_rag_chain(static_retriever, settings.chat_model) # RAG pipeline

    return chain.invoke(question) # Returns LLM response