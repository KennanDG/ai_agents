from __future__ import annotations

from typing import TypedDict, List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langsmith import traceable
from langgraph.graph import StateGraph, END

from .settings import RagSettings
from .embeddings import build_ollama_embeddings
from .vectorstore import build_qdrant
from .chain import build_rag_chain
from .singletons import get_vectorstore
from .retrieval import parallel_retrieve

from .query_translations.query_expansion import expand_queries
from .query_translations.rag_fusion import rrf_fuse
from .query_translations.cross_encoder import cross_encoder_rerank


class RagGraphState(TypedDict, total=False):
    # inputs
    question: str
    settings: RagSettings

    # retrieval artifacts
    queries: List[str]
    retrieved_docs: List[Document]

    # generation artifacts
    answer: str

    # debugging / control
    error: Optional[str]



@traceable(name="lg_retrieve", tags=["langgraph", "rag", "retrieve"])
def retrieve_node(state: RagGraphState) -> RagGraphState:
    question = state["question"]
    settings = state["settings"]

    # --- singleton objects ---
    vs = get_vectorstore(settings)
    retriever = vs.as_retriever(search_kwargs={"k": settings.k_per_query})

    # 1) multi-query expansion
    queries = expand_queries(
        question,
        chat_model=settings.query_model,
        n=settings.n_query_expansions,
    )

    # 2) PARALLEL retrieval
    results_by_query = parallel_retrieve(
        retriever=retriever,
        queries=queries,
        max_workers=8,
    )

    # 3) RAG-Fusion (RRF)
    fused_docs = rrf_fuse(
        results_by_query=results_by_query,
        k=settings.candidate_k,
        rrf_k=settings.rrf_k,
    )

    # 4) Cross-encoder rerank down to final k
    final_docs = cross_encoder_rerank(
        question=question,
        docs=fused_docs,
        model_name=settings.rerank_model,
        top_k=settings.k,
    )

    # Debug print (same as your current flow)
    print("\n--- RETRIEVED DOCS (LangGraph; after expansion + RRF + cross-encoder rerank) ---")
    for i, doc in enumerate(final_docs, 1):
        src = doc.metadata.get("source_uri", doc.metadata.get("source", "unknown"))
        print(f"[{i}] {src}")
        print(doc.page_content[:300].replace("\n", " "))
        print()

    return {
        "queries": queries,
        "retrieved_docs": final_docs,
    }



@traceable(name="lg_generate", tags=["langgraph", "rag", "generate"])
def generate_node(state: RagGraphState) -> RagGraphState:
    question = state["question"]
    settings = state["settings"]
    docs = state.get("retrieved_docs", [])

    # Make a static retriever that always returns docs
    static_retriever = RunnableLambda(lambda _: docs)

    chain = build_rag_chain(static_retriever, settings.chat_model)
    answer = chain.invoke(question)

    return {"answer": answer}



def build_rag_graph():
    graph = StateGraph(RagGraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()



# @traceable(name="rag_answer_langgraph", tags=["langgraph", "rag"])
# def answer_langgraph(question: str, settings: RagSettings) -> str:
#     graph = get_rag_graph() # Fetches singleton graph
#     final_state = graph.invoke({"question": question, "settings": settings})
#     return final_state["answer"]