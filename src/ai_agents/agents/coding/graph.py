from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


from ai_agents.agents.coding.nodes import (
    gather_context_node,
    patch_node,
    plan_node,
    report_node,
    route_node,
    validate_node,
)
from ai_agents.agents.coding.routing import (
    route_after_context,
    route_after_patch,
    route_after_validate,
)
from ai_agents.agents.coding.state import CodingAgentState



def build_coding_agent_graph():
    builder = StateGraph(CodingAgentState)
    transient_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=8.0,
    )

    builder.add_node("route", route_node, retry_policy=transient_retry)
    builder.add_node("plan", plan_node, retry_policy=transient_retry)
    builder.add_node("gather_context", gather_context_node, retry_policy=transient_retry)
    builder.add_node("patch", patch_node, retry_policy=transient_retry)
    builder.add_node("validate", validate_node, retry_policy=transient_retry)
    builder.add_node("report", report_node, retry_policy=transient_retry)

    builder.add_edge(START, "route")
    builder.add_edge("route", "plan")
    builder.add_edge("plan", "gather_context")
    builder.add_conditional_edges(
        "gather_context",
        route_after_context,
        {
            "patch": "patch",
            "report": "report",
        },
    )
    builder.add_conditional_edges(
        "patch",
        route_after_patch,
        {
            "gather_context": "gather_context",
            "validate": "validate",
            "report": "report",
        },
    )
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "gather_context": "gather_context",
            "report": "report",
        },
    )
    builder.add_edge("report", END)

    return builder.compile()
