from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from .query_expansion import expand_queries

@dataclass(frozen=True)
class QueryPlan:
    do_expand: bool
    queries: List[str]
    reason: str



def plan_queries(
    question: str,
    *,
    chat_model: str,
    n: int,
    enabled: bool = True,
    min_question_chars: int = 25,
    force: bool = False,
) -> QueryPlan:
    """
    Central policy:
      - If !enabled -> no expansion
      - If force -> expand
      - If question is short -> expand 
      - Else -> only expand when n > 1
    """
    q = (question or "").strip()

    if not enabled:
        return QueryPlan(False, [q], "query expansion disabled")

    if force:
        queries = expand_queries(q, chat_model=chat_model, n=n)
        return QueryPlan(True, queries, "forced expansion")

    # heuristic: short questions benefit most from expansion
    if len(q) < min_question_chars and n > 1:
        queries = expand_queries(q, chat_model=chat_model, n=n)
        return QueryPlan(True, queries, "short question -> expanded")

    # default: only expand if caller requested multiple expansions
    if n and n > 1:
        queries = expand_queries(q, chat_model=chat_model, n=n)
        return QueryPlan(True, queries, "n_query_expansions > 1")

    return QueryPlan(False, [q], "n_query_expansions <= 1")
