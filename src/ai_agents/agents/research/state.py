from __future__ import annotations

from typing import Any, Literal, TypedDict


ResearchAgentStatus = Literal[
    "planned",
    "searched",
    "synthesized",
    "reported",
    "failed",
]


class ResearchAgentState(TypedDict, total=False):
    user_request: str
    plan: list[str]
    search_queries: list[str]
    search_results: list[dict[str, Any]]
    research_summary: str
    report: str
    status: ResearchAgentStatus
    errors: list[str]
