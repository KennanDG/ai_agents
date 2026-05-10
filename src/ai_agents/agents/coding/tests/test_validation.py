from ai_agents.agents.coding.utils.validation import (
    is_lint_command,
    validation_failed_results,
)


def test_validation_passes_when_only_ruff_fails() -> None:
    results = [
        {
            "command": "uv run pytest",
            "returncode": 0,
        },
        {
            "command": "uv run ruff check .",
            "returncode": 1,
        },
    ]

    assert validation_failed_results(results) is False


def test_validation_fails_when_pytest_fails() -> None:
    results = [
        {
            "command": "uv run pytest",
            "returncode": 1,
        },
        {
            "command": "uv run ruff check .",
            "returncode": 1,
        },
    ]

    assert validation_failed_results(results) is True


def test_validation_fails_when_compileall_fails() -> None:
    results = [
        {
            "command": "python -m compileall .",
            "returncode": 1,
        },
    ]

    assert validation_failed_results(results) is True


def test_lint_command_detection() -> None:
    assert is_lint_command("uv run ruff check .") is True
    assert is_lint_command("ruff check src/ai_agents") is True
    assert is_lint_command("uv run pytest") is False
    assert is_lint_command("python -m compileall .") is False