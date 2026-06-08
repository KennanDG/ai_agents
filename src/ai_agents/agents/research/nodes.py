from __future__ import annotations

from ai_agents.agents.research.llm import invoke_parsed_decision, model, reasoning_model
from ai_agents.agents.research.prompts import (
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    build_planner_user_prompt,
    build_synthesizer_user_prompt,
)
from ai_agents.agents.research.schemas import PlanDecision, SynthesizeDecision
from ai_agents.agents.research.state import ResearchAgentState
from ai_agents.agents.coding.tools.web_search import web_search


def plan_node(state: ResearchAgentState) -> ResearchAgentState:
    request = state["user_request"]
    user_prompt = build_planner_user_prompt(request)
    try:
        decision: PlanDecision = invoke_parsed_decision(
            model=reasoning_model,
            schema=PlanDecision,
            node_name="plan",
            state=state,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return {
            "plan": decision.plan,
            "search_queries": decision.search_queries,
            "status": "planned",
        }
    except Exception as exc:
        return {
            "plan": ["Research the topic."],
            "search_queries": [request],
            "errors": [str(exc)],
            "status": "planned",
        }


def web_search_node(state: ResearchAgentState) -> ResearchAgentState:
    queries = state.get("search_queries", [])
    results: list[dict[str, object]] = []
    errors = list(state.get("errors", []))
    for query in queries:
        try:
            result = web_search(query)
            results.append({"query": query, "result": result})
        except Exception as exc:
            errors.append(f"Web search failed for '{query}': {exc}")
    return {
        "search_results": results,
        "errors": errors,
        "status": "searched",
    }


def synthesize_node(state: ResearchAgentState) -> ResearchAgentState:
    search_results = state.get("search_results", [])
    results_text = "\n\n---\n".join(str(r) for r in search_results)
    user_prompt = build_synthesizer_user_prompt(state["user_request"], results_text)
    try:
        decision: SynthesizeDecision = invoke_parsed_decision(
            model=model,
            schema=SynthesizeDecision,
            node_name="synthesize",
            state=state,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return {
            "research_summary": decision.research_summary,
            "report": decision.report,
            "status": "synthesized",
        }
    except Exception as exc:
        errors = list(state.get("errors", [])) + [f"Synthesize failed: {exc}"]
        return {
            "errors": errors,
            "report": "Research incomplete due to error.",
            "status": "failed",
        }


def report_node(state: ResearchAgentState) -> ResearchAgentState:
    # In a full agent this node could format the final output or log results.
    return {"status": "reported"}
