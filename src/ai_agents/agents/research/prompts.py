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


REPORT_SYSTEM_PROMPT = """You are a report editor. Given the research summary and the user's original request, produce the final polished report. Output only the final report text."""


def build_report_user_prompt(user_request: str, research_summary: str, report: str) -> str:
    return (
        f"User request: {user_request}\n\n"
        f"Research summary:\n{research_summary}\n\n"
        f"Draft report:\n{report}\n\n"
        "Produce the final polished report."
    )
