from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain_core.documents import Document


def parallel_retrieve(
    retriever,
    queries: List[str],
    max_workers: int = 8,
) -> List[List[Document]]:
    """
    Run retriever.invoke(query) in parallel for each query.
    """
    results: List[List[Document]] = []

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
