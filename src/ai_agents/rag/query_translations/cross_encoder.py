from __future__ import annotations
from typing import List, Tuple

from langchain_core.documents import Document
from langsmith import traceable
from sentence_transformers import CrossEncoder

_MODEL_CACHE: dict[tuple[str, str], CrossEncoder] = {}

def _get_model(model_name: str, device: str) -> CrossEncoder:
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = CrossEncoder(model_name, device=device)
    return _MODEL_CACHE[key]

@traceable
def cross_encoder_rerank(
    question: str,
    docs: List[Document],
    model_name: str,
    top_k: int,
    max_chars: int = 512,
    device: str = "cuda"
) -> List[Document]:
    """
    Uses a CrossEncoder model to score the relevance of each document returned
    By the RAG Fusion. 

    Returns a list of docs orderd by highest score to lowest.
    """
    if not docs:
        return []

    model = _get_model(model_name, device)

    pairs = [(question, (d.page_content or "")[:max_chars]) for d in docs]
    scores = model.predict(pairs)  # higher score means the chunk is more relevant

    scored: List[Tuple[float, Document]] = list(zip(scores, docs))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [d for _, d in scored[:top_k]]
