# ruff: noqa: I001
from __future__ import annotations

import re
from pathlib import Path


DEFAULT_IGNORES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
}


def _tokenize(query: str) -> list[str]:
    return [token for token in re.split(r"\s+", query.strip()) if token]


def _best_snippet(text: str, terms: list[str]) -> tuple[int | None, str]:
    lines = text.splitlines()
    for line_no, line in enumerate(lines, start=1):
        if any(term in line for term in terms):
            return line_no, line.strip()[:300]
    for line_no, line in enumerate(lines, start=1):
        if line.strip():
            return line_no, line.strip()[:300]
    return None, ""


def _score(rel: str, text: str, terms: list[str]) -> float:
    score = 0.0
    for term in terms:
        score += text.count(term) * 2.0
    for term in terms:
        if term in rel.lower():
            score += 10.0
    return score


def _skip_path(parts: tuple[str, ...]) -> bool:
    return any(part in DEFAULT_IGNORES for part in parts)


def search_repo(repo_root: Path, query: str, max_results: int = 25) -> list[str]:
    """Search repository files for a simple query string.

    Returns a list of formatted result lines: "path:line: snippet".
    """
    terms = [t.lower() for t in _tokenize(query)]
    if not terms:
        return []
    scored: list[tuple[float, str]] = []
    for item in sorted(repo_root.rglob("*")):
        if not item.is_file():
            continue
        if _skip_path(item.relative_to(repo_root).parts):
            continue
        rel = item.relative_to(repo_root).as_posix()
        try:
            text = item.read_text(encoding="utf-8", errors="replace")[:50_000].lower()
        except Exception:
            continue
        line_no, snippet = _best_snippet(text, terms)
        if line_no is None:
            continue
        score = _score(rel, text, terms)
        scored.append((score, f"{rel}:{line_no}: {snippet}"))
    scored.sort(key=lambda x: -x[0])
    return [entry for _, entry in scored[:max_results]]


def robust_search(repo_root: Path, queries: list[str] | str, max_results: int = 25) -> list[str]:
    """Search repository using one or more queries and return combined results."""
    query_list = [queries] if isinstance(queries, str) else queries
    scored: list[tuple[float, str]] = []
    for query in query_list:
        terms = [t.lower() for t in _tokenize(query)]
        if not terms:
            continue
        for item in sorted(repo_root.rglob("*")):
            if not item.is_file():
                continue
            if _skip_path(item.relative_to(repo_root).parts):
                continue
            rel = item.relative_to(repo_root).as_posix()
            try:
                text = item.read_text(encoding="utf-8", errors="replace")[:50_000].lower()
            except Exception:
                continue
            line_no, snippet = _best_snippet(text, terms)
            if line_no is None:
                continue
            score = _score(rel, text, terms)
            scored.append((score, f"{rel}:{line_no}: {snippet}"))
    scored.sort(key=lambda x: -x[0])
    return [entry for _, entry in scored[:max_results]]


def format_search_results(results: list[str]) -> list[str]:
    """Pass-through formatter; returns the list unchanged."""
    return results
