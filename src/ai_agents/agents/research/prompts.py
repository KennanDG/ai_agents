from __future__ import annotations


PLANNER_SYSTEM_PROMPT = """You are a research planner. Break down the user request into a concrete plan and generate effective search queries to gather necessary information."""


def build_planner_user_prompt(user_request: str) -> str:
    return f"User request: {user_request}\n\nProvide the plan as a list of steps and a list of search queries."


SYNTHESIZER_SYSTEM_PROMPT = """You are a research synthesizer. Review the search results and produce a concise summary and a final comprehensive report that directly answers the user's original request."""


def build_synthesizer_user_prompt(user_request: str, search_results: str) -> str:
    return (
        f"User request: {user_request}\n\n"
        f"Search results:\n{search_results}\n\n"
        "Provide the research summary and final report."
    )
