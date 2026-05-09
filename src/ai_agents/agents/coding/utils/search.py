from __future__ import annotations

from ai_agents.agents.coding.utils.text import dedupe


def derive_search_queries(user_request: str) -> list[str]:
    base = user_request.lower()
    queries: list[str] = []

    for term in [
        "fastapi",
        "router",
        "endpoint",
        "langgraph",
        "state",
        "pytest",
        "test",
        "settings",
        "registry",
        "tool",
        "skill",
    ]:
        if term in base:
            queries.append(term)

    return queries or ["TODO", "def ", "class "]


def paths_from_search_results(search_blocks: list[str]) -> list[str]:
    paths: list[str] = []

    for block in search_blocks:
        for line in block.splitlines():
            if ":" not in line:
                continue

            candidate = line.split(":", 1)[0].strip()

            if candidate:
                paths.append(candidate)

    return dedupe(paths)
