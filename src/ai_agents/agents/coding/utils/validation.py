from __future__ import annotations

from pathlib import Path
from typing import Any
import shlex


LINT_COMMAND_PREFIXES = (
    "ruff check",
    "uv run ruff check",
    "ruff format --check",
    "uv run ruff format --check",
)



def default_validation_commands(repo_root: Path) -> list[str]:
    if (repo_root / Path("pyproject.toml")).exists():
        return ["uv run pytest", "uv run ruff check ."]

    return ["python -m compileall ."]



def is_lint_command(command: str) -> bool:
    """
    If the validation is a lint command, 
    failure should be treated as advisory should be treated as advisory.
    """

    try:
        normalized = " ".join(shlex.split(command))
    except ValueError:
        normalized = command.strip()

    return any(normalized.startswith(prefix) for prefix in LINT_COMMAND_PREFIXES)





def validation_failed_results(results: list[dict[str, Any]]) -> bool:
    """Return True only when a blocking validation command failed.

    Lint commands are intentionally advisory. They should still be reported,
    but they should not cause the coding-agent run to fail by themselves.
    """

    for result in results:
        command = str(result.get("command", ""))

        try:
            returncode = int(result.get("returncode", 0))
        except (TypeError, ValueError):
            return True

        if returncode == 0:
            continue

        if is_lint_command(command):
            continue

        return True

    return False
