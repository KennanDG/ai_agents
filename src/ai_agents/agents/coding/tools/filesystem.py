from __future__ import annotations

import os
from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field



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
    

def resolve_in_repo(repo_root: Path, path: str | Path) -> Path:
    root = repo_root.resolve()
    target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()

    if root != target and root not in target.parents:
        raise ValueError(f"Path escapes repository root: {path}")
    
    return target


    
def list_files(repo_root: Path, path: str | Path = ".", max_depth: int = 2) -> list[str]:
    start = resolve_in_repo(repo_root, path)

    if not start.exists():
        return []

    results: list[str] = []
    root_depth = len(start.parts)

    for item in sorted(start.rglob("*")):
        if any(part in DEFAULT_IGNORES for part in item.parts):
            continue

        if len(item.parts) - root_depth > max_depth:
            continue

        if item.is_file():
            results.append(str(item.relative_to(repo_root)))

    return results




def read_file(repo_root: Path, path: str | Path, max_chars: int = 25_000) -> str:
    target = resolve_in_repo(repo_root, path)

    if not target.exists():
        raise FileNotFoundError(str(path))
    
    if not target.is_file():
        raise IsADirectoryError(str(path))
    
    text = target.read_text(encoding="utf-8", errors="replace")

    return text[:max_chars]


def write_file(repo_root: Path, path: str | Path, content: str, *, allow_write: bool = False) -> str:
    if not allow_write:
        return f"DRY_RUN: would write {path} ({len(content)} chars)"

    target = resolve_in_repo(repo_root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    
    return f"WROTE: {target.relative_to(repo_root)} ({len(content)} chars)"


@tool
def file_size(repo_root: Path, path: str | Path) -> int:
    """Return the size in bytes of a file within the repository.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path points to a directory.
    """
    target = resolve_in_repo(repo_root, path)

    if not target.exists():
        raise FileNotFoundError(str(path))

    if not target.is_file():
        raise IsADirectoryError(str(path))

    return target.stat().st_size

