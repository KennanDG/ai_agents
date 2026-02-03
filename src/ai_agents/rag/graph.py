from __future__ import annotations

from typing import TypedDict, List, Optional, Literal
import json

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from langsmith import traceable
from langgraph.graph import StateGraph, START, END

from ai_agents.core.retry import retry

from .settings import RagSettings
from .chain import build_rag_chain
from .singletons import get_vectorstore
from .retrieval import parallel_retrieve
from .prompts import build_grader_prompt

from .query_translations.rag_fusion import rrf_fuse
from .query_translations.cross_encoder import cross_encoder_rerank
from .query_translations.plan import plan_queries


class RagGraphState(TypedDict, total=False):
    """
    Represents the state of the RAG pipeline as it moves through the graph.
    
    Attributes:
        question (str): The user's original question.
        settings (RagSettings): Configuration settings for the pipeline.
        queries (List[str]): Generated search queries (from expansion or original).
        retrieved_docs (List[Document]): The final list of docs after retrieval/fusion/reranking.
        answer (str): The generated answer from the LLM.
        attempt (int): The current attempt count for the retry loop.
        max_attempts (int): Max number of re-tries
        verification_score (int): 0 for failure/hallucination, 1 for success.
        verification_reason (str): The grader's explanation for the score.
        error (Optional[str]): Debugging output
        result (dict): Final output of the logic graph
    """


    # inputs
    question: str
    settings: RagSettings

    # --- Internal State ---
    queries: List[str]            # Search queries (original or expanded)
    retrieved_docs: List[Document]
    answer: str

    # --- Control Flow ---
    attempt: int
    max_attempts: int
    verification_score: int       # 1 = Pass, 0 = Fail
    verification_reason: str

    # --- debugging / control ---
    error: Optional[str]

    # --- Output ---
    result: dict





# ==========================================
# NODE 0: Initialization
# ==========================================
@traceable(name="lg_init", tags=["langgraph", "init"])
def init_state_node(state: RagGraphState) -> RagGraphState:
    """Initialize/normalize control-flow fields for the RAG LangGraph.

    Why this exists:
        LangGraph state is incremental. If the caller doesn't provide
        control fields (attempt counters, max attempts), routing logic can
        become inconsistent and accidentally create unbounded loops.

    Inputs (from state):
        - settings: RagSettings

    Outputs (merged into state):
        - attempt: int
            The number of verification failures encountered so far.
            Starts at 0.
        - max_attempts: int
            Maximum number of verification failures to tolerate before
            finalizing. Pulled from settings.max_rag_attempts.
        - error: Optional[str]
            Cleared to None at the start of each run.

    Side effects:
        None.
    """
    settings = state["settings"]
    max_attempts = int(getattr(settings, "max_rag_attempts", 2))

    return {
        "attempt": int(state.get("attempt", 0)),
        "max_attempts": max_attempts,
        "error": None,
    }




# ==========================================
# NODE 1: Query Expansion (Conditional)
# ==========================================
@traceable(name="lg_expand_queries", tags=["langgraph", "expansion"])
def expand_queries_node(state: RagGraphState) -> RagGraphState:
    """
    Generates multiple search queries based on the user's question to improve retrieval coverage.
    
    Logic:
        Uses `plan_queries` which handles the logic of whether to expand based on 
        settings and question length.
        
    Inputs (from state):
        - question: str
        - settings: RagSettings

    Outputs (merged into state):
        - queries: List[str]
    """
    question = state["question"]
    settings = state["settings"]

    
    plan = retry(
        lambda: plan_queries(
            question,
            chat_model=settings.query_model,
            n=settings.n_query_expansions,
            enabled=getattr(settings, "enable_query_expansion", True),
            min_question_chars=getattr(settings, "min_question_chars_for_expansion", 25),
        ),
        attempts=getattr(settings, "retrieve_attempts", 2),
    )
    
    return {"queries": plan.queries}





# ==========================================
# NODE 2: Retrieval (Fetch + Fuse + Rerank)
# ==========================================
@traceable(name="lg_retrieve", tags=["langgraph", "rag", "retrieve"])
def retrieve_node(state: RagGraphState) -> RagGraphState:
    """
    Orchestrates the retrieval pipeline:
    1. Parallel Retrieval: Fetches docs for all queries in state['queries'].
    2. RRF Fusion: Merges results using Reciprocal Rank Fusion.
    3. Cross-Encoder Reranking: Re-scores top candidates for semantic relevance.
    
    Inputs (from state):
        - question: str
        - settings: RagSettings
        - queries: Optional[List[str]]
            If missing/empty, this node falls back to [question].

    Outputs (merged into state):
        - queries: List[str]
            The queries actually used for retrieval (after fallback).
        - retrieved_docs: List[Document]

    Side effects:
        Prints a short preview of retrieved docs for debugging.
    """

    question = state["question"]
    settings = state["settings"]

    # Fallback if expansion didn't happen
    queries = state.get("queries", [question])
    if not queries: 
        queries = [question]

    # --- singleton objects ---
    vs = get_vectorstore(settings)
    retriever = vs.as_retriever(search_kwargs={"k": settings.k_per_query})


    # 1) PARALLEL retrieval
    results_by_query = retry(
        lambda: parallel_retrieve(
            retriever=retriever, 
            queries=queries, 
            max_workers=8
        ),
        attempts=getattr(settings, "retrieve_attempts", 2),
    )

    # 2) RAG-Fusion (RRF)
    fused_docs = retry(
        lambda: rrf_fuse(
            results_by_query=results_by_query, 
            k=settings.candidate_k, 
            rrf_k=settings.rrf_k
        ),
        attempts=getattr(settings, "retrieve_attempts", 2),
    )

    # 3) Cross-encoder rerank down to final k
    final_docs = retry(
        lambda: cross_encoder_rerank(
            question=question,
            docs=fused_docs,
            model_name=settings.rerank_model,
            top_k=settings.k,
            max_chars=512,
            device=settings.rerank_device,
        ),
        attempts=getattr(settings, "retrieve_attempts", 2),
    )

    # Debug print (same as your current flow)
    print("\n--- RETRIEVED DOCS (LangGraph; after expansion + RRF + cross-encoder rerank) ---")
    for i, doc in enumerate(final_docs, 1):
        src = doc.metadata.get("source_uri", doc.metadata.get("source", "unknown"))
        print(f"[{i}] {src}")
        print()
        print(f"Length of chunk: {len(doc.page_content)}")
        print(doc.page_content[:300].replace("\n", " "))
        print()
        print()

    return {
        "queries": queries,
        "retrieved_docs": final_docs,
    }




# ==========================================
# NODE 3: Generation
# ==========================================
@traceable(name="lg_generate", tags=["langgraph", "rag", "generate"])
def generate_node(state: RagGraphState) -> RagGraphState:
    """
    Generates an answer using the LLM and the retrieved documents.
    
    Inputs (from state):
        - question: str
        - settings: RagSettings
        - retrieved_docs: Optional[List[Document]]

    Outputs (merged into state):
        - answer: str
    """
    question = state["question"]
    settings = state["settings"]
    docs = state.get("retrieved_docs", [])

    # Make a static retriever that always returns docs
    static_retriever = RunnableLambda(lambda _: docs)

    chain = build_rag_chain(static_retriever, settings.chat_model)

    answer = retry(
        lambda: chain.invoke(question),
        attempts=getattr(settings, "generate_attempts", 2),
    )

    return {"answer": answer}




# ==========================================
# NODE 4: Verification
# ==========================================
@traceable(name="lg_verify", tags=["langgraph", "verify"])
def verify_answer_node(state: RagGraphState) -> RagGraphState:
    """
    Grades the generated answer for groundedness and relevance.
    
    Strategy:
    - We use a "Fail Open" strategy. If the grader LLM fails or throws an exception,
      we default to score=1 (Pass). This ensures the user still gets an answer 
      rather than a system error, assuming the generation was likely okay.
      
    State Updates:
    - Increments 'attempt' counter here to ensure the retry loop advances.
    
    Inputs (from state):
        - question: str
        - answer: str
        - retrieved_docs: List[Document]
        - settings: RagSettings

    Outputs (merged into state):
        - verification_score: int
        - verification_reason: str
        - attempt: int
            Incremented by 1 *only when verification fails or errors*.
        - error: Optional[str]
            Populated when verification errors occur.
    """

    question = state["question"]
    answer = state["answer"]
    docs = state.get("retrieved_docs", [])
    settings = state["settings"]
    current_attempt = state.get("attempt", 0) # Get current attempt count, defaulting to 0 if not set
    error: Optional[str] = None

    # Combine context for the grader
    context_text = "\n\n".join([d.page_content for d in docs])
    max_chars = int(getattr(settings, "verify_max_chars", 6_000))

    if max_chars > 0 and len(context_text) > max_chars:
        context_text = context_text[:max_chars]

    # Build chain: Prompt -> LLM -> JSON Parser
    grader = settings.verify_model 
    grader_prompt = build_grader_prompt()
    

    llm = ChatOllama(model=grader, format="json", temperature=0)
    chain = grader_prompt | llm | JsonOutputParser()


    def _invoke() -> dict:
        return chain.invoke(
            {
                "context": context_text,
                "question": question,
                "answer": answer,
            }
        )

    try:
        result = retry(
            _invoke,
            attempts=int(getattr(settings, "verify_attempts", 2)),
        )
        score = result.get("score", 0)
        reason = result.get("reason", "No reason provided")

    except Exception as e:
        print(f"Verification failed: {e}")
        # FAIL OPEN: Assume the answer is good if the grader crashes
        score = 1 
        reason = f"Verification skipped due to error: {str(e)}"
        error = f"verify_answer_node: {type(e).__name__}: {e}"

    return {
        "verification_score": score,
        "verification_reason": reason,
        "attempt": current_attempt + 1,
        "error": error
    }




# ==========================================
# NODE 5: Finalize (Optional)
# ==========================================
@traceable(name="lg_finalize", tags=["langgraph", "finalize"])
def finalize_node(state: RagGraphState) -> RagGraphState:
    """Produce a single, stable output payload for callers.

    Why this exists:
        In LangGraph, the *final state* is whatever accumulated fields exist
        at END. Callers often prefer a single predictable key like
        state['result'] rather than knowing which intermediate keys to read.

    Inputs (from state):
        - question, answer
        - verification_score, verification_reason
        - attempt, max_attempts
        - queries
        - error

    Outputs (merged into state):
        - result: dict
            A stable payload suitable for APIs/CLI.

    Notes:
        We intentionally do NOT include full retrieved docs by default
        (to keep responses small).
    """
    return {
        "result": {
            "question": state.get("question"),
            "answer": state.get("answer"),
            "queries": state.get("queries", []),
            "verification": {
                "score": state.get("verification_score", 0),
                "reason": state.get("verification_reason", ""),
            },
            "attempt": int(state.get("attempt", 0)),
            "max_attempts": int(state.get("max_attempts", 0)),
            "error": state.get("error"),
        }
    }




# ==========================================
# CONDITIONAL EDGES
# ==========================================

def route_query_expansion(state: RagGraphState) -> Literal["expand_queries", "retrieve"]:
    """Route: decide whether to run the query expansion node.

    Notes:
        - This router should be lightweight and deterministic.
        - The *planner* inside expand_queries_node owns the decision about
          whether expansion actually happens.

    Returns:
        "expand_queries" if expansion is enabled, else "retrieve".
    """

    settings = state["settings"]
    
    # Check settings AND question length
    if getattr(settings, "enable_query_expansion", True):
        if len(state["question"]) >= getattr(settings, "min_question_chars_for_expansion", 25):
            return "expand_queries"
            
    return "retrieve"



def route_verification(state: RagGraphState) -> Literal["generate", "finalize"]:
    """
    Routing Logic:
    1. If score == 1 (Pass) -> Finalize.
    2. If score == 0 (Fail) AND attempts < max_attempts -> Retry.
    3. Else -> Finalize.
    """
    score = state.get("verification_score", 0)
    attempt = state.get("attempt", 1) # Note: 'verify' node has already incremented this to 1
    max_attempts = getattr(state["settings"], "max_rag_attempts", 2) 

    if score == 1:
        return "finalize"
    
    if attempt < max_attempts:
        print(f"--- RETRYING ({attempt+1}/{max_attempts}): {state.get('verification_reason')} ---")
        return "expand_queries" # or loop back to generate
    

    print(f"--- MAX RETRIES REACHED ({attempt}/{max_attempts}) ---")
    return "finalize"




# ==========================================
# GRAPH BUILDER
# ==========================================
def build_rag_graph():
    """Build and compile the LangGraph RAG workflow.

    High-level flow:
        START -> init -> (expand_queries?) -> retrieve -> generate -> verify
                                              ^                     |
                                              |---------------------|

    Retry behavior:
        Verification failures loop back to retrieval (and then regeneration)
        until max_attempts is reached.
    """

    graph = StateGraph(RagGraphState)

    # Add Nodes
    graph.add_node("init", init_state_node)
    graph.add_node("expand_queries", expand_queries_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_answer_node)
    graph.add_node("finalize", finalize_node)

    # 1. Entry Point
    graph.add_edge(START, "init")

    graph.add_conditional_edges(
        "init",
        route_query_expansion,
        {
            "expand_queries": "expand_queries",
            "retrieve": "retrieve"
        }
    )

    # 2. Linear flow
    graph.add_edge("expand_queries", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")

    # 3. Retry Loop: Verification
    graph.add_conditional_edges(     # Logic: If verification fails -> Retry expansion (if enabled). If Pass -> End.
        "verify",
        route_verification,
        {
            "expand_queries": "expand_queries", # Retry loop
            "finalize": "finalize"
        }
    )

    graph.add_edge("finalize", END)

    return graph.compile()

