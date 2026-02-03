from langsmith import traceable
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from .chain import build_rag_chain
from .embeddings import build_ollama_embeddings
from .settings import RagSettings
from .vectorstore import build_qdrant, build_retriever
from .singletons import get_rag_graph, get_vectorstore
from .retrieval import parallel_retrieve

from .query_translations.query_expansion import expand_queries
from .query_translations.rag_fusion import rrf_fuse
from .query_translations.cross_encoder import cross_encoder_rerank
from .query_translations.plan import plan_queries


@traceable(name="rag_answer", tags=["rag", "query-expansion", "rrf-fusion", "cross-encoder-rerank"])
def answer(question: str, settings: RagSettings) -> str:
    
    embeddings = build_ollama_embeddings(settings.embedding_model) # Embedding model

    # initialize the vector database
    vs = get_vectorstore(settings)
    retriever = vs.as_retriever(search_kwargs={"k": settings.k_per_query})

    # base_retriever = build_retriever(vs, settings.k) # fetch documents

    # 1) multi-query expansion if enabled, otherwise just the user query is returned
    plan = plan_queries(
        question,
        chat_model=settings.query_model,
        n=settings.n_query_expansions,
        enabled=getattr(settings, "enable_query_expansion", True),
        min_question_chars=getattr(settings, "min_question_chars_for_expansion", 25),
    )

    queries = plan.queries
    

    # 2) PARALLEL retrieval
    results_by_query = parallel_retrieve(
        retriever=retriever,
        queries=queries,
        max_workers=8,
    )


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
        top_k=settings.k,
        max_chars=512,
        device=settings.rerank_device
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




@traceable(name="rag_answer_langgraph", tags=["langgraph", "rag"])
def answer_langgraph(question: str, settings: RagSettings) -> dict:
    graph = get_rag_graph() # Fetches singleton graph
    final_state = graph.invoke({"question": question, "settings": settings})
    return final_state["result"]