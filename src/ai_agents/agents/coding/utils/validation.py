from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any


LINT_COMMAND_PREFIXES = (
    "ruff check",
    "uv run ruff check",
    "ruff format --check",
    "uv run ruff format --check",
)

INFRASTRUCTURE_ERROR_SNIPPETS = (
    "Executable not found",
    "Command blocked by coding-agent allowlist",
    "No such file or directory (os error 2)",
    "No such file or directory",
    "ModuleNotFoundError: No module named",
    "command not found",
    "not recognized as an internal or external command",
)


def _normalize_command(command: str) -> str:
    try:
        return " ".join(shlex.split(command))
    except ValueError:
        return command.strip()


def _returncode(result: dict[str, Any]) -> int | None:
    try:
        return int(result.get("returncode", 0))
    except (TypeError, ValueError):
        return None


def default_validation_commands(repo_root: Path) -> list[str]:
    if (repo_root / "pyproject.toml").exists():
        return ["uv run pytest", "uv run ruff check ."]

    return ["python -m compileall ."]


def is_lint_command(command: str) -> bool:
    """
    If the validation is a lint command, 
    failure should be treated as advisory should be treated as advisory.
    """
    normalized = _normalize_command(command)
    return any(normalized.startswith(prefix) for prefix in LINT_COMMAND_PREFIXES)


def validation_result_failed(result: dict[str, Any]) -> bool:
    returncode = _returncode(result)
    return returncode is None or returncode != 0


def is_infrastructure_validation_failure(result: dict[str, Any]) -> bool:
    command = str(result.get("command", ""))
    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))
    returncode = _returncode(result)

    combined_output = f"{stdout}\n{stderr}"

    return (
        command == "validation_suite"
        or returncode in {126, 127}
        or any(snippet in combined_output for snippet in INFRASTRUCTURE_ERROR_SNIPPETS)
    )


def is_advisory_validation_failure(result: dict[str, Any]) -> bool:
    if not validation_result_failed(result):
        return False

    command = str(result.get("command", ""))

    return is_lint_command(command) or is_infrastructure_validation_failure(result)


def advisory_validation_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        result
        for result in results
        if validation_result_failed(result) and is_advisory_validation_failure(result)
    ]


def blocking_validation_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        result
        for result in results
        if validation_result_failed(result) and not is_advisory_validation_failure(result)
    ]


def validation_failed_results(results: list[dict[str, Any]]) -> bool:
    """Return True only when validation has real blocking failures."""
    return bool(blocking_validation_failures(results))