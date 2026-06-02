from __future__ import annotations

from pathlib import Path

from ai_agents.agents.coding.tests.discovery import discover_targeted_tests
from ai_agents.agents.coding.tests.profiles import default_validation_profile
from ai_agents.agents.coding.tests.results import (
    ValidationCommand,
    ValidationResult,
    ValidationSuiteResult,
)
from ai_agents.agents.coding.tools.shell import run_command




def run_validation_suite(
    repo_root: Path,
    *,
    changed_files: list[str] | None = None,
    requested_commands: list[str] | None = None,
    allow_shell: bool = True,
    timeout_seconds: int = 60,
    profile_name: str = "default",
) -> ValidationSuiteResult:
    

    commands = build_validation_commands(
        repo_root,
        changed_files=changed_files or [],
        requested_commands=requested_commands or [],
    )

    suite = ValidationSuiteResult(profile=profile_name)

    if not allow_shell:
        suite.results = [
            ValidationResult(
                command=command.command,
                returncode=126,
                stderr="Shell disabled.",
                reason=command.reason,
            )

            for command in commands
        ]

        return suite

    for command in commands:
        raw_result = run_command(
            repo_root,
            command.command,
            timeout_seconds=timeout_seconds,
        )

        suite.results.append(
            ValidationResult(
                command=str(raw_result.get("command", command.command)),
                returncode=int(raw_result.get("returncode", 1)),
                stdout=str(raw_result.get("stdout", "")),
                stderr=str(raw_result.get("stderr", "")),
                reason=command.reason,
            )
        )

    return suite




def build_validation_commands(
    repo_root: Path,
    *,
    changed_files: list[str],
    requested_commands: list[str],
) -> list[ValidationCommand]:
    
    commands: list[ValidationCommand] = []

    commands.extend(discover_targeted_tests(repo_root, changed_files))

    for command in requested_commands:
        commands.append(
            ValidationCommand(
                command=command,
                reason="Requested by planner or patcher.",
            )
        )

    if not commands:
        commands.extend(default_validation_profile(repo_root))

    return _dedupe_commands(commands)




def _dedupe_commands(commands: list[ValidationCommand]) -> list[ValidationCommand]:
    seen: set[str] = set()
    result: list[ValidationCommand] = []

    for command in commands:
        normalized = command.command.strip()

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        result.append(command)

    return result