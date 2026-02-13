from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langsmith import traceable
from qdrant_client import QdrantClient

from ai_agents.core.retry import retry

from .singletons import get_retriever
from .settings import RagSettings
from .query_translations.rag_fusion import rrf_fuse
from .query_translations.cross_encoder import cross_encoder_rerank



def parallel_retrieve(
    retriever,
    queries: List[str],
    max_workers: int = 8,
) -> List[List[Document]]:
    """
    Run retriever.invoke(query) in parallel for each query.
    """
    results: List[List[Document]] = []

    if not queries:
        return results

    with ThreadPoolExecutor(max_workers=min(max_workers, len(queries))) as executor:
        future_map = {
            executor.submit(retriever.invoke, q): q for q in queries
        }

        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception:
                # fail soft — empty result for this query
                results.append([])

    return results



@traceable(name="retrieve_collection", tags=["rag", "retrieve", "single_collection"])
def retrieve_collection(
    *,
    question: str,
    queries: List[str],
    settings: RagSettings,
    base_collection: str,
) -> List[Document]:
    """Retrieve docs from a single base collection.

    Pipeline:
      1) parallel_retrieve over queries
      2) rrf_fuse
      3) cross_encoder_rerank -> top_k=settings.k
    """
    min_docs = int(getattr(settings, "min_docs_for_success", 2))

    retriever = get_retriever(settings, collection_name_override=base_collection)

    results_by_query = retry(
        lambda: parallel_retrieve(
            retriever=retriever,
            queries=queries,
            max_workers=int(getattr(settings, "retrieve_workers", 8)),
        ),
        attempts=int(getattr(settings, "retrieve_attempts", 2)),
    )

    # quick signal check (pre-fusion)
    flat = [d for bucket in results_by_query for d in (bucket or [])]
    if len(flat) < min_docs:
        return []

    fused_docs = retry(
        lambda: rrf_fuse(
            results_by_query=results_by_query,
            k=settings.candidate_k,
            rrf_k=settings.rrf_k,
        ),
        attempts=int(getattr(settings, "retrieve_attempts", 2)),
    )

    final_docs = retry(
        lambda: cross_encoder_rerank(
            question=question,
            docs=fused_docs,
            model_name=settings.rerank_model,
            top_k=settings.k,
            max_chars=512,
            device=getattr(settings, "rerank_device", None),
        ),
        attempts=int(getattr(settings, "retrieve_attempts", 2)),
    )

    if len(final_docs) < min_docs:
        return []

    return final_docs



@traceable(name="parallel_retrieve_collections", tags=["rag", "retrieve", "collections", "parallel"])
def parallel_retrieve_collections(
    *,
    question: str,
    queries: List[str],
    settings: RagSettings,
    collections: List[str],
    max_workers: int = 3,
) -> List[Tuple[str, List[Document]]]:
    """Retrieve from multiple collections in parallel.

    Returns list of (collection_name, docs). Order is completion-order.
    Caller should re-order / select using the router order if desired.
    """
    if not collections:
        return []

    out: List[Tuple[str, List[Document]]] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(collections))) as ex:
        futures = {
            ex.submit(
                retrieve_collection,
                question=question,
                queries=queries,
                settings=settings,
                base_collection=collection,
            ): collection
            for collection in collections
        }

        for future in as_completed(futures):
            collection = futures[future]
            
            try:
                docs = future.result()
            except Exception:
                docs = []

            out.append((collection, docs))

    return out









# -----------------------------------------------------------------------------
# Collection discovery + routing
# -----------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _list_qdrant_collections(qdrant_url: str) -> List[str]:
    """Return raw Qdrant collection names."""
    client = QdrantClient(url=qdrant_url)
    cols = client.get_collections().collections
    return [c.name for c in cols]


def available_base_collections(*, qdrant_url: str, namespace: str) -> List[str]:
    """Return base collection names (without '-{namespace}').

    Example:
        'rag-engineering-default' -> 'rag-engineering'
    """
    suffix = f"-{namespace}"
    out: List[str] = []

    for name in _list_qdrant_collections(qdrant_url):
        if name.endswith(suffix):
            out.append(name[: -len(suffix)])

    return sorted(set(out))





#TODO: Remove
def route_collections(
    *,
    question: str,
    qdrant_url: str,
    namespace: str,
    default_collection: str,
    preferred: Optional[Iterable[str]] = None,
) -> List[str]:
    """Choose an ordered list of base collections to search.

    Policy:
        1) If caller provides `preferred`, use that order (filtered to existing).
        2) Else use heuristic routing among known collections.
        3) Always include default as final fallback.
        4) If nothing matches, fallback to all known collections.
    """
    available = set(available_base_collections(qdrant_url=qdrant_url, namespace=namespace))

    def _keep_existing(collections: Iterable[str]) -> List[str]:
        return [c for c in collections if c in available]

    if preferred:
        ordered = _keep_existing(preferred)
    else:
        # Fallback heuristic if no router output provided
        ordered = [default_collection] if default_collection in available else list(available)


    if not ordered:
        ordered = sorted(available)

    # dedupe while preserving order
    seen = set()
    final: List[str] = []

    for collection in ordered:
        if collection not in seen:
            seen.add(collection)
            final.append(collection)

    return final




# -----------------------------------------------------------------------------
# Dynamic retrieval with retry + fallback
# -----------------------------------------------------------------------------


@traceable(name="dynamic_retrieve", tags=["rag", "retrieve", "collections"])
def dynamic_retrieve(
    *,
    question: str,
    queries: List[str],
    settings: RagSettings,
    preferred_collections: Optional[List[str]] = None,
) -> Tuple[List[Document], str, List[str]]:
    """Retrieve docs using best-matching collection with fallback.

    Returns:
        (final_docs, selected_collection, attempted_collections)

    Fallback rules:
        - If a collection returns < min_docs, try next collection.
        - If retrieval errors, try next collection.
        - Always attempt settings.collection_name as the final fallback (if it exists).
    """
    min_docs = int(getattr(settings, "min_docs_for_success", 2))
    max_collections = int(getattr(settings, "max_collection_fallbacks", 3))

    preferred = preferred_collections or getattr(settings, "preferred_collections", None)

    ordered = route_collections(
        question=question,
        qdrant_url=settings.qdrant_url,
        namespace=settings.namespace,
        default_collection=settings.collection_name,
        preferred=preferred,
    )

    ordered = ordered[: max(1, max_collections)]

    attempted: List[str] = []
    last_error: Optional[str] = None

    for base_collection in ordered:
        attempted.append(base_collection)

        try:
            retriever = get_retriever(settings, collection_name_override=base_collection)

            results_by_query = retry(
                lambda: parallel_retrieve(
                    retriever=retriever,
                    queries=queries,
                    max_workers=int(getattr(settings, "retrieve_workers", 8)),
                ),
                attempts=int(getattr(settings, "retrieve_attempts", 2)),
            )

            # quick signal check (pre-fusion)
            flat = [d for bucket in results_by_query for d in (bucket or [])]
            if len(flat) < min_docs:
                continue

            fused_docs = retry(
                lambda: rrf_fuse(
                    results_by_query=results_by_query,
                    k=settings.candidate_k,
                    rrf_k=settings.rrf_k,
                ),
                attempts=int(getattr(settings, "retrieve_attempts", 2)),
            )

            final_docs = retry(
                lambda: cross_encoder_rerank(
                    question=question,
                    docs=fused_docs,
                    model_name=settings.rerank_model,
                    top_k=settings.k,
                    max_chars=512,
                    device=getattr(settings, "rerank_device", None),
                ),
                attempts=int(getattr(settings, "retrieve_attempts", 2)),
            )

            if len(final_docs) < min_docs:
                continue

            return final_docs, base_collection, attempted

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            continue

    if last_error:
        setattr(settings, "_last_retrieve_error", last_error)

    return [], (ordered[-1] if ordered else settings.collection_name), attempted
