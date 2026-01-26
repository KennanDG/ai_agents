from collections import defaultdict
from langchain_core.documents import Document
from langsmith import traceable


@traceable
def rrf_fuse(results_by_query: list[list[Document]], k: int = 60, rrf_k: int = 60) -> list[Document]:
    """
    results_by_query: list of ranked doc lists (one list per query variant)
    k: how many fused docs to return
    rrf_k: RRF constant (larger reduces impact of rank position); common 60
    """
    scores = defaultdict(float)
    doc_by_key = {}

    for docs in results_by_query:
        for rank, doc in enumerate(docs, start=1):
            key = (
                doc.metadata.get("source_uri"),
                doc.metadata.get("chunk_index"),
                doc.metadata.get("chunk_hash"),
            )
            
            doc_by_key[key] = doc
            scores[key] += 1.0 / (rrf_k + rank)

    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [doc_by_key[key] for key in ranked_keys[:k]]
