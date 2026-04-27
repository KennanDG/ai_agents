from pathlib import Path

from ai_agents.agents.coding.tests.runner import build_validation_commands


def test_build_validation_commands_uses_requested_commands(tmp_path: Path) -> None:

    commands = build_validation_commands(
        tmp_path,
        changed_files=[],
        requested_commands=["python -m compileall ."],
    )

    assert [command.command for command in commands] == ["python -m compileall ."]


def test_build_validation_commands_falls_back_to_default(tmp_path: Path) -> None:

    commands = build_validation_commands(
        tmp_path,
        changed_files=[],
        requested_commands=[],
    )

    assert commands
    assert commands[0].command == "python -m compileall ."


def test_build_validation_commands_discovers_changed_test_file(tmp_path: Path) -> None:
    
    test_file = tmp_path / "tests" / "test_example.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_example():\n    assert True\n", encoding="utf-8")

    commands = build_validation_commands(
        tmp_path,
        changed_files=["tests/test_example.py"],
        requested_commands=[],
    )

    assert commands[0].command == "uv run pytest tests/test_example.py"