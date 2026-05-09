from __future__ import annotations

from pathlib import Path
from typing import Any


def default_validation_commands(repo_root: Path) -> list[str]:
    if (repo_root / Path("pyproject.toml")).exists():
        return ["uv run pytest", "uv run ruff check ."]

    return ["python -m compileall ."]


def validation_failed_results(results: list[dict[str, Any]]) -> bool:
    for result in results:
        try:
            if int(result.get("returncode", 0)) != 0:
                return True
        except (TypeError, ValueError):
            return True

    return False
