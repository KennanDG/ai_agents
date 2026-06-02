from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ai_agents.agents.coding.utils.text import dedupe
from ai_agents.agents.coding.utils.constants import IGNORED_CONTEXT_PATH_PARTS, DEFAULT_EXCLUDED_PATH_HINTS, STOPWORDS




#################################### Helpers ####################################
def _extract_path_hints(text: str) -> list[str]:
    normalized = text.replace("\\", "/")
    hints = re.findall(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.*-]+", normalized)

    clean_hints: list[str] = []

    for hint in hints:
        hint = hint.strip("`'\".,:;()[]{}")
        if not hint or hint.startswith(("http://", "https://")):
            continue
        clean_hints.append(hint.rstrip("/*"))

    return dedupe(clean_hints)


def _extract_extensions(text: str) -> list[str]:
    extensions = re.findall(r"(?<![A-Za-z0-9_])\.[A-Za-z0-9]+", text)

    allowed = {
        ".py",
        ".md",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".sql",
        ".sh",
        ".tf",
    }

    return dedupe([extension.lower() for extension in extensions if extension.lower() in allowed])



def _extract_search_terms(text: str) -> list[str]:
    quoted = re.findall(r"`([^`]+)`|'([^']+)'|\"([^\"]+)\"", text)
    terms: list[str] = []

    for groups in quoted:
        value = next((item for item in groups if item), "").strip()

        if value and "/" not in value and not value.startswith("."):
            terms.append(value)

    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)

    for identifier in identifiers:
        lowered = identifier.lower()

        if lowered in STOPWORDS:
            continue
        if lowered in {"file", "files", "folder", "directory"}:
            continue

        terms.append(identifier)

    return dedupe(terms)


def _looks_like_symbol(term: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", term))

########################################################################





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




def derive_search_requests(user_request: str) -> list[dict[str, Any]]:
    """
    These requests are intentionally conservative. They prioritize path hints and
    code-like terms that appear in the user's request instead of scanning the whole
    repository for generic Python syntax.
    """

    path_hints = _extract_path_hints(user_request)
    extensions = _extract_extensions(user_request)
    terms = _extract_search_terms(user_request)
    requests: list[dict[str, Any]] = []

    if path_hints:
        requests.append(
            {
                "terms": terms[:6],
                "path_includes": path_hints,
                "path_excludes": DEFAULT_EXCLUDED_PATH_HINTS,
                "file_extensions": extensions,
                "mode": "all" if terms else "any",
            }
        )

    if terms:
        requests.append(
            {
                "terms": terms[:8],
                "path_includes": [],
                "path_excludes": DEFAULT_EXCLUDED_PATH_HINTS,
                "file_extensions": extensions,
                "mode": "all",
            }
        )

        symbol_terms = [term for term in terms if _looks_like_symbol(term)]

        if symbol_terms:
            requests.append(
                {
                    "terms": symbol_terms[:6],
                    "path_includes": [],
                    "path_excludes": DEFAULT_EXCLUDED_PATH_HINTS,
                    "file_extensions": [".py"],
                    "mode": "symbol",
                }
            )

    if extensions and not requests:
        requests.append(
            {
                "terms": [],
                "path_includes": path_hints,
                "path_excludes": DEFAULT_EXCLUDED_PATH_HINTS,
                "file_extensions": extensions,
                "mode": "any",
            }
        )

    return requests




def derive_search_queries(user_request: str) -> list[str]:
    """Legacy query derivation kept for callers not yet migrated to search_requests."""

    queries: list[str] = []

    for request in derive_search_requests(user_request):
        terms = request.get("terms") or []
        if terms:
            queries.append(" ".join(terms))

    return queries


def legacy_queries_to_search_requests(queries: Sequence[str]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    for query in queries:
        terms = _extract_search_terms(query) or [term for term in query.split() if term]
        if not terms:
            continue

        requests.append(
            {
                "terms": terms,
                "path_includes": _extract_path_hints(query),
                "path_excludes": DEFAULT_EXCLUDED_PATH_HINTS,
                "file_extensions": _extract_extensions(query),
                "mode": "all",
            }
        )

    return requests


def paths_from_search_results(search_blocks: Sequence[str]) -> list[str]:
    paths: list[str] = []

    for block in search_blocks:
        for line in block.splitlines():
            if ":" not in line:
                continue

            candidate = line.split(":", 1)[0].strip()

            if candidate and not is_ignored_context_path(candidate):
                paths.append(candidate)

    return dedupe(paths)


def paths_from_ranked_results(results: Sequence[Mapping[str, Any]]) -> list[str]:
    paths: list[str] = []

    for result in results:
        path = str(result.get("path", "")).strip()
        if path and not is_ignored_context_path(path):
            paths.append(path)

    return dedupe(paths)
