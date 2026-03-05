from __future__ import annotations
from typing import List, Tuple
import json

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langsmith import traceable

from ai_agents.rag.prompts import RERANK_PROMPT
from ai_agents.config.settings import settings


def _format_passages(docs: List[Document], max_chars: int = 600) -> str:
    lines = []

    for i, doc in enumerate(docs):
        text = (doc.page_content or "")[:max_chars].replace("\n", " ")
        src = (
            doc.metadata.get("source_uri") or 
            doc.metadata.get("path_rel") or 
            doc.metadata.get("file_name") or "unknown"
        )

        lines.append(f"[{i}] ({src}) {text}")

    return "\n".join(lines)



# ---- Deprecated ----
@traceable
def rerank_docs(question: str, docs: List[Document], chat_model: str, top_k: int) -> List[Document]:
    if not docs:
        return []

    llm = ChatGroq(
        model=chat_model,
        api_key=settings.resolved_groq_api_key(),
        temperature=0.0,
    )

    passages = _format_passages(docs)

    msg = RERANK_PROMPT.format_messages(
        question=question, 
        passages=passages
    )

    raw = llm.invoke(msg).content

    try:
        scores = json.loads(raw)

        # build list of (index, score)
        pairs: List[Tuple[int, float]] = []

        for item in scores:
            idx = int(item["index"])
            score = float(item["score"])
            pairs.append((idx, score))

        # stable sort by score desc
        pairs.sort(key=lambda x: x[1], reverse=True)
        ranked = [docs[i] for i, _ in pairs if 0 <= i < len(docs)]
        
    except Exception:
        # fallback: no rerank
        ranked = docs

    return ranked[:top_k]