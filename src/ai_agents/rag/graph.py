from __future__ import annotations

from typing import TypedDict, List, Optional, Literal, Tuple, Dict
import json

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langsmith import traceable
from langgraph.graph import StateGraph, START, END

from qdrant_client import QdrantClient

from ai_agents.core.retry import retry

from .settings import RagSettings
from .chain import build_rag_chain
from .singletons import get_vectorstore
from .retrieval import retrieve_collection, parallel_retrieve_collections, available_base_collections
from .prompts import build_grader_prompt, build_collection_router_prompt, build_docs_grader_prompt

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
    question_original: str 
    rewritten_question: str 
    settings: RagSettings

    # --- Internal State ---
    queries: List[str]            # Search queries (original or expanded)
    retrieved_docs: List[Document]
    retrieval_candidates_docs : Dict[str, List[Document]]
    answer: str

    # --- Control Flow ---
    attempt: int
    max_attempts: int
    docs_should_retry: bool
    docs_retry_reason: str

    # --- Retrieval control ---
    collection_index: int          # which preferred_collections index we're trying next

    # Verification states
    verification_score: int       # 1 = Pass, 0 = Fail
    verification_reason: str

    doc_verification_score: int   # Document verification (1=Pass, 0=Fail)
    doc_verification_reason: str
    
    # --- Collection Routing ---
    selected_collection: str
    attempted_collections: List[str]
    preferred_collections: List[str]
    collection_router_reason: Optional[str]

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
        - question_original
            Original user query
        - rewritten_question
            Modifies user's original query if generate node fails verification
        - collection_index
            The current index of preferred_collections
        - docs_verification_score
            'pass' or 'fail' score received from the verify_documents node
        - docs_verification_reason
            LLM's reasoning for the verification score
        - docs_should_retry
            whether or not the next collection in preferred_collections
            should do retrieval
        - docs_retry_reason
            Reasoning for retrieval on a different collection
        - error: Optional[str]
            Cleared to None at the start of each run.

    Side effects:
        None.
    """
    settings = state["settings"]
    max_attempts = int(getattr(settings, "max_rag_attempts", 2))

    # preserve the very first user question
    question = state["question"]
    question_original = state.get("question_original", question)

    return {
        "attempt": int(state.get("attempt", 0)),
        "max_attempts": max_attempts,
        "question_original": question_original,
        "rewritten_question": state.get("rewritten_question", ""),
        "collection_index": int(state.get("collection_index", 0)),
        "docs_verification_score": int(state.get("docs_verification_score", 0)),
        "docs_verification_reason": state.get("docs_verification_reason", ""),
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
# NODE 2: Collections Routing
# ==========================================
@traceable(name="lg_route_collections", tags=["langgraph", "rag", "collections"])
def route_collections_node(state: RagGraphState) -> RagGraphState:
    """
    Infer the preferred Qdrant collections to search for retrieval using an LLM-based router.

    Purpose:
        This node dynamically determines which domain-specific vector collections should be
        queried for a given question. Instead of relying on static heuristics or a single
        default collection, it uses an LLM to infer intent and rank candidate collections
        by likelihood of relevance.

        The output of this node is consumed by the retrieval node, which performs robust
        retrieval with retry and fallback logic across the inferred collection order.

    High-level behavior:
        1. Discover available base collections from Qdrant for the active namespace.
        2. If only one (or zero) collections exist, short-circuit and select it directly.
        3. Otherwise, invoke an LLM router with:
            - the user question
            - the list of available collections
            - the default fallback collection
            - the maximum number of allowed fallbacks
        4. Sanitize and validate the LLM output:
            - remove unknown collections
            - ensure the default collection is included as a fallback
            - guarantee at least one valid collection is returned
        5. Return the ordered collection preferences along with a human-readable reason.

    Design notes:
        - This node is intentionally non-authoritative: downstream retrieval still applies
          its own retry, thresholding, and fallback logic.
        - The router LLM is run with temperature=0 to ensure deterministic routing decisions.
        - Routing decisions are surfaced in the graph state for observability and debugging.

    Inputs (from state):
        - question (str):
            The current user question (may be rewritten by earlier verification steps).
        - settings (RagSettings):
            Configuration containing Qdrant connection info, routing limits,
            model selection, and API credentials.

    Outputs (merged into state):
        - preferred_collections (List[str]):
            Ordered list of base collection names to try during retrieval
            (most likely → least likely).
        - collection_router_reason (str):
            Short explanation produced by the router LLM describing its decision.

    Failure & fallback behavior:
        - If the router returns invalid or empty output, the node falls back to using
          the default collection (if available) or the first discovered collection.
        - If only one collection exists, no LLM call is made.

    Side effects:
        - Makes a lightweight LLM call for routing (no document retrieval).
        - Emits LangSmith traces for routing decisions.
    """
    
    question = state["question"]
    settings = state["settings"]

    # Discover available base collections
    available = available_base_collections(qdrant_url=settings.qdrant_url, namespace=settings.namespace)

    # If only one exists, just use it
    if len(available) <= 1:
        return {
            "preferred_collections": available or [settings.preferred_collections],
            "collection_router_reason": "Only one (or zero) collections available.",
        }

    router_prompt = build_collection_router_prompt()

    llm = ChatGroq(
        model=settings.query_model,  
        api_key=settings.groq_api_key,
        temperature=0.0,
    ).bind(response_format={"type": "json_object"})

    parser = JsonOutputParser()

    chain = router_prompt | llm | parser

    def _invoke():
        return chain.invoke({
            "available": available,
            "question": question,
            "default_collection": settings.collection_name,
            "max_fallbacks": settings.max_collection_fallbacks,
        })

    result = retry(
        _invoke, 
        attempts=int(getattr(settings, "retrieve_attempts", 2))
    )

    raw = result.get("preferred_collections", []) or []
    reason = result.get("reason", "")

    # sanitize + enforce constraints
    ordered = [collection for collection in raw if collection in available]

    # Ensure default is somewhere in the list if the LLM missed it completely
    if settings.collection_name in available and settings.collection_name not in ordered:
        ordered.append(settings.collection_name)

    if not ordered:
        ordered = [settings.collection_name] if settings.collection_name in available else available[:1]

    return {
        "preferred_collections": ordered,
        "collection_router_reason": reason,
    }





# ==========================================
# NODE 3: Retrieval (Fetch + Fuse + Rerank)
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
    queries = state.get("queries", [question]) or [question]

    preferred = state.get("preferred_collections", []) or [settings.collection_name]
    index = int(state.get("collection_index", 0)) 

    enable_parallel = bool(getattr(settings, "enable_parallel_collection_retrieval", True))
    parallel_max = int(getattr(settings, "parallel_collection_workers", 3))


    # ------ PARALLEL ------
    if enable_parallel:
        
        # take up to first 3 by router order
        target_collections = preferred[: max(1, parallel_max)]
        results = parallel_retrieve_collections(
            question=question,
            queries=queries,
            settings=settings,
            collections=target_collections,
            max_workers=parallel_max,
        )

        # Store all attempted in router order for observability
        attempted_collections = target_collections

        # keep the router order preference.
        docs_by_col = {collection: docs for (collection, docs) in results}

        selected_collection = next(
            (collection for collection in target_collections if docs_by_col.get(collection)), 
            target_collections[0]
        )
        
        final_docs = docs_by_col.get(selected_collection, [])

        return {
            "queries": queries,
            "retrieved_docs": final_docs,
            "selected_collection": selected_collection,
            "attempted_collections": attempted_collections,
            "retrieval_candidates": [{"collection": collection, "doc_count": len(docs_by_col.get(collection, []))} for collection in target_collections],
            "retrieval_candidates_docs": {collection: docs_by_col.get(collection, []) for collection in target_collections},
        }
    
    
    # ------ SEQUENTIAL ------
    selected_collection = preferred[min(index, len(preferred) - 1)]

    final_docs = retrieve_collection(
        question=question,
        queries=queries,
        settings=settings,
        base_collection=selected_collection,
    )

    attempted_collections = preferred[: min(index + 1, len(preferred))]


    # # Debug print
    # print("\n--- RETRIEVED DOCS (LangGraph; after expansion + RRF + cross-encoder rerank) ---")
    # for i, doc in enumerate(final_docs, 1):
    #     src = doc.metadata.get("source_uri", doc.metadata.get("source", "unknown"))
    #     print(f"[{i}] {src}")
    #     print()
    #     print(f"Length of chunk: {len(doc.page_content)}")
    #     print(doc.page_content[:300].replace("\n", " "))
    #     print()
    #     print()

    return {
        "queries": queries,
        "retrieved_docs": final_docs,
        "selected_collection": selected_collection,
        "attempted_collections": attempted_collections,
    }





# ==========================================
# NODE 4: Verify documents
# ==========================================
@traceable(name="lg_verify_documents", tags=["langgraph", "rag", "verify_documents"])
def verify_documents_node(state: RagGraphState) -> RagGraphState:
    question = state["question"]
    settings = state["settings"]

    preferred_collections: List[str] = state.get("preferred_collections", []) or [settings.collection_name]
    attempted_collections: List[str] = state.get("attempted_collections", []) or []
    index: int = int(state.get("collection_index", 0))

    # If parallel mode stashed candidates, we can verify the best option(s)
    candidates_docs = state.get("retrieval_candidates_docs")  # dict[str, List[Document]] | None

    def _next_collection_index(preferred: List[str], attempted: List[str], current_index: int) -> Tuple[bool, int, str]:
        """
        Returns (has_next, next_index, reason).
        Chooses the first preferred collection not yet attempted.
        """
        remaining_collections = [collection for collection in preferred if collection not in attempted]

        if not remaining_collections:
            return False, current_index, "No remaining collections to try."

        next_collection = remaining_collections[0]

        try:
            next_idx = preferred.index(next_collection)
        except ValueError:
            next_idx = min(current_index + 1, max(0, len(preferred) - 1))

        return True, next_idx, f"Retrying retrieval with next collection: {next_collection}"


    def _grade_docs(docs: List[Document]) -> Tuple[int, str]:
        context_text = "\n\n".join([d.page_content for d in (docs or [])])
        max_chars = int(getattr(settings, "verify_docs_max_chars", 6_000))
        
        if max_chars > 0 and len(context_text) > max_chars:
            context_text = context_text[:max_chars]

        llm = ChatGroq(
            model=getattr(settings, "verify_docs_model", settings.query_model),
            api_key=settings.groq_api_key,
            temperature=0.0,
        ).bind(response_format={"type": "json_object"})

        chain = build_docs_grader_prompt() | llm | JsonOutputParser()

        result = retry(
            lambda: chain.invoke({"question": question, "context": context_text}),
            attempts=int(getattr(settings, "verify_docs_attempts", 2)),
        )

        score = int(result.get("score", 0))
        reason = result.get("reason", "No reason provided")

        return score, reason



    # -------- PARALLEL CANDIDATES PATH --------
    if isinstance(candidates_docs, dict) and candidates_docs:
        # keep router-order preference when grading
        router_order = attempted_collections or list(candidates_docs.keys())

        last_reason = "No candidate collection produced sufficient/relevant documents."
        
        for collection in router_order:
            docs = candidates_docs.get(collection, [])
            try:
                score, reason = _grade_docs(docs)
            except Exception as e:
                score, reason = 0, f"verify_documents_node error on {collection}: {type(e).__name__}: {e}"

            if score == 1:
                return {
                    "docs_verification_score": 1,
                    "docs_verification_reason": reason,
                    "docs_should_retry": False,
                    "docs_retry_reason": "",
                    "selected_collection": collection,
                    "retrieved_docs": docs,
                }
            
            last_reason = reason


        # none passed → decide whether to retry with another (untried) collection
        has_next, next_index, retry_reason = _next_collection_index(preferred_collections, attempted_collections, index)

        return {
            "docs_verification_score": 0,
            "docs_verification_reason": last_reason,
            "docs_should_retry": has_next,
            "docs_retry_reason": retry_reason if has_next else "No more collections; proceeding to generate.",
            "collection_index": next_index if has_next else index,
        }

    # -------- SEQUENTIAL PATH --------
    docs = state.get("retrieved_docs", [])
    try:
        score, reason = _grade_docs(docs)
    except Exception as e:
        # guarding generation quality. If it errors, we should try next collection.
        score, reason = 0, f"verify_documents_node error: {type(e).__name__}: {e}"

    if score == 1:
        return {
            "docs_verification_score": 1,
            "docs_verification_reason": reason,
            "docs_should_retry": False,
            "docs_retry_reason": "",
        }

    # failed → try next collection if available
    has_next, next_index, retry_reason = _next_collection_index(preferred_collections, attempted_collections, index)
    return {
        "docs_verification_score": 0,
        "docs_verification_reason": reason,
        "docs_should_retry": has_next,
        "docs_retry_reason": retry_reason if has_next else "No more collections; proceeding to generate.",
        "collection_index": next_index if has_next else index,
    }





# ==========================================
# NODE 5: Generation
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
    candidates_docs = state.get("retrieval_candidates_docs", {})  

    if candidates_docs:
        static_retriever = RunnableLambda(lambda _: candidates_docs)
    else:
        static_retriever = RunnableLambda(lambda _: docs)

    chain = build_rag_chain(static_retriever, settings.chat_model)

    answer = retry(
        lambda: chain.invoke(question),
        attempts=getattr(settings, "generate_attempts", 2),
    )

    return {"answer": answer}




# ==========================================
# NODE 6: Verification
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
    original_question = state.get("question_original", question)
    answer = state["answer"]
    docs = state.get("retrieved_docs", [])
    candidates_docs = state.get("retrieval_candidates_docs", {})
    settings = state["settings"]
    current_attempt = state.get("attempt", 0) # Get current attempt count, defaulting to 0 if not set
    error: Optional[str] = None
    context_text = ""

    # Combine context for the grader
    if candidates_docs:
        for _, documents in candidates_docs.items():
            context_text += "\n\n".join([d.page_content for d in documents])
    else:
        context_text = "\n\n".join([d.page_content for d in docs])

    max_chars = int(getattr(settings, "verify_max_chars", 6_000))

    if max_chars > 0 and len(context_text) > max_chars:
        context_text = context_text[:max_chars]

    # Build chain: Prompt -> LLM -> JSON Parser
    grader = settings.verify_model 
    grader_prompt = build_grader_prompt()
    

    llm = ChatGroq(
        model=grader,
        api_key=settings.groq_api_key,
        temperature=0.0
    ).bind(
        response_format={"type": "json_object"}
    )

    chain = grader_prompt | llm | JsonOutputParser()


    def _invoke() -> dict:
        return chain.invoke(
            {
                "context": context_text,
                "original_question": original_question,
                "question": question,
                "answer": answer,
            }
        )

    try:
        result = retry(
            _invoke,
            attempts=int(getattr(settings, "verify_attempts", 2)),
        )
        score = int(result.get("score", 0))
        reason = result.get("reason", "No reason provided")
        rewritten_question = result.get("rewritten_question", question)

    except Exception as e:
        print(f"Verification failed: {e}")
        # FAIL OPEN: Assume the answer is good if the grader crashes
        score = 1 
        reason = f"Verification skipped due to error: {str(e)}"
        rewritten_question = question
        error = f"verify_answer_node: {type(e).__name__}: {e}"

    return {
        "verification_score": score,
        "verification_reason": reason,
        "rewritten_question": rewritten_question,
        "question": rewritten_question if score == 0 and rewritten_question else question,
        "attempt": current_attempt + 1,
        "error": error
    }




# ==========================================
# NODE 7: Finalize (Optional)
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
                "answer_score": state.get("verification_score", 0),
                "answer_reason": state.get("verification_reason", ""),
                "docs_score": state.get("doc_verification_score", 0),
                "docs_reason": state.get("doc_verification_reason", ""),
                "docs_retry": state.get("docs_should_retry", False),
                "docs_retry_reason": state.get("docs_retry_reason", ""),
            },
            "retrieval": {
                "preferred_collections": state.get("preferred_collections", []),
                "router_reason": state.get("collection_router_reason"),
                "selected_collection": state.get("selected_collection"),
                "attempted_collections": state.get("attempted_collections", []),
            },
            "attempt": int(state.get("attempt", 0)),
            "max_attempts": int(state.get("max_attempts", 0)),
            "error": state.get("error"),
            "question_original": state.get("question_original", ""), 
            "rewritten_question": state.get("rewritten_question", ""), 
        }
    }




# ==========================================
# CONDITIONAL EDGES
# ==========================================

def route_query_expansion(state: RagGraphState) -> Literal["expand_queries", "route_collections"]:
    """Route: decide whether to run the query expansion node.

    Notes:
        - This router should be lightweight and deterministic.
        - The *planner* inside expand_queries_node owns the decision about
          whether expansion actually happens.

    Returns:
        "expand_queries" if expansion is enabled, else "route_collections".
    """

    settings = state["settings"]
    
    # Check settings AND question length
    if getattr(settings, "enable_query_expansion", True):
        if len(state["question"]) >= getattr(settings, "min_question_chars_for_expansion", 25):
            return "expand_queries"
            
    return "route_collections"



def route_docs_verification(state: RagGraphState) -> Literal["retrieve", "generate"]:
    """Loops back to retrieve or passes forward to generate"""
    return "retrieve" if state.get("docs_should_retry", False) else "generate"




def route_answer_verification(state: RagGraphState) -> Literal["generate", "finalize"]:
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
    graph.add_node("route_collections", route_collections_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("verify_documents", verify_documents_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_answer_node)
    graph.add_node("finalize", finalize_node)

    # 1. Entry Point
    graph.add_edge(START, "init")

    # 2. Determine if expansion is needed
    graph.add_conditional_edges(
        "init",
        route_query_expansion,
        {
            "expand_queries": "expand_queries",
            "route_collections": "route_collections"
        }
    )

    # 3. Retrieval
    graph.add_edge("expand_queries", "route_collections")
    graph.add_edge("route_collections", "retrieve")
    graph.add_edge("retrieve", "verify_documents")

    # 4. Determine if documents are good
    graph.add_conditional_edges(
        "verify_documents",
        route_docs_verification,
        {
            "retrieve": "retrieve", 
            "generate": "generate"
        }
    )

    # 4. Generate answer & verify
    graph.add_edge("generate", "verify")

    # 5. Retry Loop
    graph.add_conditional_edges(     # Logic: If verification fails -> Retry expansion (if enabled). If Pass -> End.
        "verify",
        route_answer_verification,
        {
            "expand_queries": "expand_queries", # Retry loop
            "finalize": "finalize"
        }
    )

    graph.add_edge("finalize", END)

    return graph.compile()

