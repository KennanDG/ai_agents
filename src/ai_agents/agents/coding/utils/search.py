from __future__ import annotations

from ai_agents.agents.coding.utils.text import dedupe


IGNORED_CONTEXT_PATH_PARTS = (
    ("logs", "runs"),
)


def is_ignored_context_path(path: str) -> bool:
    """Return True when a repo-relative path should never enter LLM context."""

    normalized = path.strip().replace("\\", "/")
    parts = tuple(part for part in normalized.split("/") if part and part != ".")

    for ignored_parts in IGNORED_CONTEXT_PATH_PARTS:
        ignored_len = len(ignored_parts)

        for index in range(0, len(parts) - ignored_len + 1):
            if parts[index : index + ignored_len] == ignored_parts:
                return True

    return False



def filter_context_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if not is_ignored_context_path(path)]




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

            if candidate and not is_ignored_context_path(candidate):
                paths.append(candidate)

    return dedupe(paths)
