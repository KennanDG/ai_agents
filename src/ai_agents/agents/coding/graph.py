from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

from ai_agents.agents.coding.implementation import (
    assess_progress_node,
    assign_subtask_workers,
    gather_subtask_results_node,
    reconcile_subtask_patches_node,
    subtask_worker_node,
)
from ai_agents.agents.coding.memory import CodingAgentRuntimeContext
from ai_agents.agents.coding.nodes import (
    custom_tools_node,
    gmail_access_node,
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
    route_after_assess,
    route_after_patch,
    route_after_plan,
    route_after_validate,
    route_after_web_search,
)
from ai_agents.agents.coding.state import CodingAgentState


def build_coding_agent_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    builder = StateGraph(CodingAgentState, context_schema=CodingAgentRuntimeContext)

    # Provider/parser retry behavior is controlled inside the LLM wrapper and inside
    # each implementation worker. Graph-level retries are reserved for external
    # persistence/connectors so retry counts remain deterministic.
    no_retry = RetryPolicy(max_attempts=1)
    transient_retry = RetryPolicy(
        max_attempts=2,
        initial_interval=0.5,
        backoff_factor=2.0,
        max_interval=2.0,
    )

    builder.add_node("route", route_node, retry_policy=no_retry)
    builder.add_node("recall_memory", recall_memory_node, retry_policy=transient_retry)
    builder.add_node("plan", plan_node, retry_policy=no_retry)
    builder.add_node("custom_tools", custom_tools_node, retry_policy=no_retry)
    builder.add_node("repo_navigator", repo_navigator_node, retry_policy=no_retry)

    # Divide-and-conquer implementation path. Workers may read/search/call an LLM
    # but never mutate repository files. Only the deterministic reconciler writes.
    builder.add_node("subtask_worker", subtask_worker_node, retry_policy=no_retry)
    builder.add_node(
        "gather_subtask_results",
        gather_subtask_results_node,
        retry_policy=no_retry,
    )
    builder.add_node(
        "patch",
        reconcile_subtask_patches_node,
        retry_policy=no_retry,
    )
    builder.add_node("validate", validate_node, retry_policy=no_retry)
    builder.add_node("assess_progress", assess_progress_node, retry_policy=no_retry)

    builder.add_node("report", report_node, retry_policy=no_retry)
    builder.add_node("remember_run", remember_run_node, retry_policy=transient_retry)
    builder.add_node("web_search", web_search_node, retry_policy=transient_retry)
    builder.add_node("gmail_access", gmail_access_node, retry_policy=transient_retry)

    # Routing and memory recall are independent.
    builder.add_edge(START, "route")
    builder.add_edge(START, "recall_memory")
    builder.add_edge(["route", "recall_memory"], "plan")

    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "web_search": "web_search",
            "gmail_access": "gmail_access",
            "repo_navigator": "custom_tools",
        },
    )
    builder.add_conditional_edges(
        "web_search",
        route_after_web_search,
        {
            "gmail_access": "gmail_access",
            "repo_navigator": "custom_tools",
        },
    )
    builder.add_edge("gmail_access", "custom_tools")
    builder.add_edge("custom_tools", "repo_navigator")

    # Every navigation pass advances implementation_generation. Only the next
    # dependency-ready batch is fanned out; additional units are scheduled by the
    # deterministic completion assessor in later implementation iterations.
    builder.add_conditional_edges(
        "repo_navigator",
        assign_subtask_workers,
        ["subtask_worker"],
    )
    builder.add_edge("subtask_worker", "gather_subtask_results")
    builder.add_edge("gather_subtask_results", "patch")

    builder.add_conditional_edges(
        "patch",
        route_after_patch,
        {
            "assess_progress": "assess_progress",
            "validate": "validate",
            "report": "report",
        },
    )
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "assess_progress": "assess_progress",
            "report": "report",
        },
    )
    builder.add_conditional_edges(
        "assess_progress",
        route_after_assess,
        {
            "repo_navigator": "repo_navigator",
            "report": "report",
        },
    )

    builder.add_edge("report", "remember_run")
    builder.add_edge("remember_run", END)

    return builder.compile(checkpointer=checkpointer, store=store)
