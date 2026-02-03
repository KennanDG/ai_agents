from __future__ import annotations
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
import json
from ai_agents.rag.prompts import QUERY_EXPANSION_PROMPT


@traceable
def expand_queries(question: str, chat_model: str, n: int = 5) -> List[str]:
    
    llm = ChatOllama(model=chat_model, temperature=0.3)

    msg = QUERY_EXPANSION_PROMPT.format_messages(
        question=question,
        n=n,
    )

    raw = llm.invoke(msg).content

    # minimal, safe parsing
    try:
        queries = json.loads(raw)

        if not isinstance(queries, list):
            raise ValueError("Expected a JSON list")

        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

    except Exception:
        # fallback: treat as single query
        queries = [question]

    # Always include original question first (good anchor)
    out = [question]
    
    for q in queries:
        if q.lower() not in {x.lower() for x in out}:
            out.append(q)

    return out[: max(1, n + 1)]
