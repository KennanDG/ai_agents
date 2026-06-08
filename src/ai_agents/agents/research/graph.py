from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from ai_agents.agents.research.nodes import plan_node, web_search_node, synthesize_node, report_node
from ai_agents.agents.research.routing import route_after_plan
from ai_agents.agents.research.state import ResearchAgentState


def build_research_agent_graph():
    builder = StateGraph(ResearchAgentState)
    transient_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=8.0,
    )

    builder.add_node("plan", plan_node, retry_policy=transient_retry)
    builder.add_node("web_search", web_search_node, retry_policy=transient_retry)
    builder.add_node("synthesize", synthesize_node, retry_policy=transient_retry)
    builder.add_node("report", report_node, retry_policy=transient_retry)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "web_search": "web_search",
            "report": "report",
        },
    )
    builder.add_edge("web_search", "synthesize")
    builder.add_edge("synthesize", "report")
    builder.add_edge("report", END)

    return builder.compile()
