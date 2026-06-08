from __future__ import annotations

from ai_agents.agents.research.state import ResearchAgentState


def route_after_plan(state: ResearchAgentState) -> str:
    if state.get("search_queries"):
        return "web_search"
    return "report"
