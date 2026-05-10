from __future__ import annotations

from pathlib import Path

from ai_agents.agents.coding.tools.filesystem import DEFAULT_IGNORES, read_file
from ai_agents.agents.coding.utils.search import is_ignored_context_path
import difflib
import re


TEXT_SUFFIXES = {
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
    ".dockerfile",
}


def search_repo(repo_root: Path, query: str, max_results: int = 25) -> list[str]:
    """Simple substring search across text-like files."""
    
    query_lower = query.lower().strip()
    
    if not query_lower:
        return []

    matches: list[str] = []
    
    for path in sorted(repo_root.rglob("*")):
        if len(matches) >= max_results:
            break

        if any(part in DEFAULT_IGNORES for part in path.parts):
            continue

        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue

        rel = path.relative_to(repo_root)

        if is_ignored_context_path(str(rel)):
            continue
        
        try:
            text = read_file(repo_root, rel, max_chars=50_000)
        except Exception:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):

            line_lower = line.lower()
            if query_lower in line_lower:
                matches.append(f"{rel}:{line_no}: {line.strip()[:300]}")
                break
            # Simple fuzzy matching: consider a line a match if similarity > 0.8
            similarity = difflib.SequenceMatcher(None, query_lower, line_lower).ratio()
            if similarity > 0.8:
                matches.append(f"{rel}:{line_no}: {line.strip()[:300]}")
                break

    return matches


def robust_search(repo_root: Path, queries: list[str] | str, max_results: int = 25) -> list[str]:
    """Enhanced search supporting multiple terms, case-insensitive matching, simple wildcards, and hidden file exclusion.

    Args:
        repo_root: Root directory of the repository.
        queries: A single query string or a list of query strings. Wildcards using "*" are supported.
        max_results: Maximum number of result strings to return.
    """
    if isinstance(queries, str):
        # Split on whitespace to allow multiple terms in a single string
        queries = [q for q in queries.split() if q]
    
    # Prepare regex patterns for each query (case‑insensitive, wildcard support)
    patterns = []

    for q in queries:
        q = q.strip()
        if not q:
            continue
        # Escape regex special chars except '*'
        escaped = re.escape(q).replace(r"\*", ".*")
        patterns.append(re.compile(escaped, re.IGNORECASE))

    if not patterns:
        return []
    
    matches: list[str] = []

    for path in sorted(repo_root.rglob("*")):
        if len(matches) >= max_results:
            break

        # Skip ignored parts and hidden files/directories
        if any(part in DEFAULT_IGNORES for part in path.parts):
            continue

        if any(part.startswith('.') for part in path.parts):
            continue

        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue

        rel = path.relative_to(repo_root)

        if is_ignored_context_path(str(rel)):
            continue

        try:
            text = read_file(repo_root, rel, max_chars=50_000)
        except Exception:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            for pat in patterns:
                if pat.search(line):
                    matches.append(f"{rel}:{line_no}: {line.strip()[:300]}")
                    break

            if len(matches) >= max_results:
                break

    return matches
