from __future__ import annotations

from pathlib import Path

from ai_agents.agents.coding.tests.results import ValidationCommand


def has_pyproject(repo_root: Path) -> bool:
    return (repo_root / "pyproject.toml").exists()



def default_validation_profile(repo_root: Path) -> list[ValidationCommand]:

    if has_pyproject(repo_root):
        return [
            ValidationCommand("uv run pytest", "Run the Python test suite."),
            ValidationCommand("uv run ruff check .", "Run Python lint checks."),
        ]

    return [
        ValidationCommand("python -m compileall .", "Verify Python files compile."),
    ]



def quick_python_profile(repo_root: Path) -> list[ValidationCommand]:

    if has_pyproject(repo_root):
        return [
            ValidationCommand("uv run pytest", "Run tests."),
        ]

    return [
        ValidationCommand("python -m compileall .", "Verify Python files compile."),
    ]



def lint_profile(repo_root: Path) -> list[ValidationCommand]:
    if has_pyproject(repo_root):
        return [
            ValidationCommand("uv run ruff check .", "Run lint checks."),
        ]

    return []