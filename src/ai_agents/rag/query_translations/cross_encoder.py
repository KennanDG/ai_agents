from __future__ import annotations
from typing import List, Tuple

from langchain_core.documents import Document
from langsmith import traceable
from fastembed.rerank.cross_encoder import TextCrossEncoder

_MODEL_CACHE: dict[tuple[str, str], TextCrossEncoder] = {}

def _get_model(model_name: str, device: str) -> TextCrossEncoder:
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = TextCrossEncoder(model_name=model_name)
    return _MODEL_CACHE[key]

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

    model = _get_model(model_name, device)
    documents = [(d.page_content or "")[:max_chars] for d in docs]

    scores = list(model.rerank(question, documents))  # higher score means the chunk is more relevant

    scored: List[Tuple[float, Document]] = list(zip(scores, docs))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [d for _, d in scored[:top_k]]
