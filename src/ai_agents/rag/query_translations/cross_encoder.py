from __future__ import annotations
from typing import List, Tuple
import os
import requests

from langchain_core.documents import Document
from langsmith import traceable
# from fastembed.rerank.cross_encoder import TextCrossEncoder
from ai_agents.rag.settings import RagSettings as settings

# _MODEL_CACHE: dict[tuple[str, str], TextCrossEncoder] = {}

# def _get_model(model_name: str, device: str) -> TextCrossEncoder:
#     key = (model_name, device)
#     if key not in _MODEL_CACHE:
#         _MODEL_CACHE[key] = TextCrossEncoder(model_name=model_name)
#     return _MODEL_CACHE[key]

@traceable
def cross_encoder_rerank(
    question: str,
    docs: List[Document],
    model_name: str,
    top_k: int,
    max_chars: int = 512,
    device: str = None # Not used currently, only used for sentencetransformers module
) -> List[Document]:
    """
    Re-rank documents using FastEmbed's cross-encoder reranker.

    - `question` is the query text (original or rewritten).
    - `docs` are candidate chunks after retrieval + RRF fusion.
    - Returns docs sorted by relevance, top_k only.

    """
    if not docs:
        return []
    

    url = settings.jina_api_url
    api_key = settings.jina_api_key

    documents = [(d.page_content or "")[:max_chars] for d in docs]

    payload = {
        "model": model_name,
        "query": question,
        "documents": documents,
        "top_n": min(top_k, len(documents)),
        "return_documents": False,
        "truncation": True,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


    try:
        res = requests.post(url, json=payload, headers=headers, timeout=20)
        res.raise_for_status()
        data = res.json()
    except Exception:
        # Fail soft: keep pipeline running
        return docs[:top_k]

    results = data.get("results") or []

    if not results:
        return docs[:top_k]
    
    return results[:top_k]

    # # Jina returns entries with `index` pointing into the input documents list
    # ranked = []
    # seen = set()

    # for r in results:
    #     idx = r.get("index")
    #     if isinstance(idx, int) and 0 <= idx < len(docs) and idx not in seen:
    #         ranked.append(docs[idx])
    #         seen.add(idx)

    # # If API returned fewer than top_k, pad with remaining docs in original order
    # if len(ranked) < top_k:
    #     for i, d in enumerate(docs):
    #         if i not in seen:
    #             ranked.append(d)
    #         if len(ranked) >= top_k:
    #             break

    # return ranked[:top_k]



    # model = _get_model(model_name, device)
    # documents = [(d.page_content or "")[:max_chars] for d in docs]

    # scores = list(model.rerank(question, documents))  # higher score means the chunk is more relevant

    # scored: List[Tuple[float, Document]] = list(zip(scores, docs))
    # scored.sort(key=lambda x: x[0], reverse=True)

    # return [d for _, d in scored[:top_k]]
