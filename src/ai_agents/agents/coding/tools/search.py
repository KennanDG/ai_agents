from __future__ import annotations

from pathlib import Path

from ai_agents.agents.coding.tools.filesystem import DEFAULT_IGNORES, read_file

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
        
        try:
            text = read_file(repo_root, rel, max_chars=50_000)
        except Exception:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):

            if query_lower in line.lower():
                matches.append(f"{rel}:{line_no}: {line.strip()[:300]}")
                break

    return matches