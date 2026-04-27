from __future__ import annotations

from pathlib import Path

from ai_agents.agents.coding.tests.results import ValidationCommand


def discover_targeted_tests(repo_root: Path, changed_files: list[str]) -> list[ValidationCommand]:
    commands: list[ValidationCommand] = []

    for changed_file in changed_files:
        path = Path(changed_file)

        if path.suffix != ".py":
            continue

        if "tests" in path.parts and path.name.startswith("test_"):
            commands.append(
                ValidationCommand(
                    f"uv run pytest {changed_file}",
                    f"Run changed test file {changed_file}.",
                )
            )

            continue

        candidate_tests = _candidate_test_paths(path)


        for candidate in candidate_tests:

            if (repo_root / candidate).exists():
                commands.append(
                    ValidationCommand(
                        f"uv run pytest {candidate.as_posix()}",
                        f"Run likely related test file for {changed_file}.",
                    )
                )

    return _dedupe_commands(commands)


def _candidate_test_paths(source_path: Path) -> list[Path]:
    """
    Example:
    ai_agents/agents/coding/graph.py
    -> tests/test_graph.py
    -> ai_agents/agents/coding/tests/test_graph.py
    """
    filename = f"test_{source_path.stem}.py"

    candidates = [
        Path("tests") / filename,
    ]

    if len(source_path.parts) > 1:
        candidates.append(Path(*source_path.parts[:-1]) / "tests" / filename)

    return candidates


def _dedupe_commands(commands: list[ValidationCommand]) -> list[ValidationCommand]:
    seen: set[str] = set()
    result: list[ValidationCommand] = []

    for command in commands:
        if command.command in seen:
            continue
        
        seen.add(command.command)
        result.append(command)

    return result