from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from ai_agents.agents.coding.memory import CodingAgentRuntimeContext
from ai_agents.agents.coding.nodes import (
    gather_context_node,
    gmail_access_node,
    patch_node,
    plan_node,
    recall_memory_node,
    remember_run_node,
    repo_navigator_node,
    report_node,
    route_node,
    validate_node,
    web_search_node,
)

from ai_agents.agents.coding.routing import (
    route_after_plan,
    route_after_context,
    route_after_patch,
    route_after_validate,
)
from ai_agents.agents.coding.state import CodingAgentState



def build_coding_agent_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    builder = StateGraph(CodingAgentState, context_schema=CodingAgentRuntimeContext)
    transient_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=8.0,
    )

    builder.add_node("route", route_node, retry_policy=transient_retry)
    builder.add_node("recall_memory", recall_memory_node, retry_policy=transient_retry)
    builder.add_node("plan", plan_node, retry_policy=transient_retry)
    builder.add_node("repo_navigator", repo_navigator_node, retry_policy=transient_retry)
    builder.add_node("gather_context", gather_context_node, retry_policy=transient_retry)
    builder.add_node("patch", patch_node, retry_policy=transient_retry)
    builder.add_node("validate", validate_node, retry_policy=transient_retry)
    builder.add_node("report", report_node, retry_policy=transient_retry)
    builder.add_node("remember_run", remember_run_node, retry_policy=transient_retry)
    builder.add_node("web_search", web_search_node, retry_policy=transient_retry)
    builder.add_node("gmail_access", gmail_access_node, retry_policy=transient_retry)

    builder.add_edge(START, "route")
    builder.add_edge("route", "recall_memory")
    builder.add_edge("recall_memory", "plan")
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "web_search": "web_search",
            "gmail_access": "gmail_access",
            "repo_navigator": "repo_navigator",
        },
    )
    builder.add_edge("web_search", "repo_navigator")
    builder.add_edge("gmail_access", "repo_navigator")
    builder.add_edge("repo_navigator", "gather_context")
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
    builder.add_edge("report", "remember_run")
    builder.add_edge("remember_run", END)

    return builder.compile(checkpointer=checkpointer, store=store)
