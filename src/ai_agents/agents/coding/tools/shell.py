from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

ALLOWED_COMMAND_PREFIXES = (
    ("pytest",),
    ("uv", "run", "pytest"),
    ("ruff", "check"),
    ("uv", "run", "ruff", "check"),
    ("ruff", "format", "--check"),
    ("uv", "run", "ruff", "format", "--check"),
    ("python", "-m", "pytest"),
    ("uv", "run", "python", "-m", "pytest"),
    ("python", "-m", "compileall"),
    ("uv", "run", "python", "-m", "compileall"),
    ("python", "-m", "py_compile"),
    ("python", "-c"),
    ("uv", "run", "python", "-m", "py_compile"),
    ("npx", "tsc"),
    ("npm", "run", "typecheck"),
    ("npm", "run", "build"),
    ("npx", "tailwindcss"),
)

BLOCKED_COMMANDS = {
    "sudo",
    "rm",
    "rmdir",
    "del",
    "format",
    "shutdown",
    "reboot",
    "mkfs",
    "chmod",
    "chown",
}


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _starts_with_tokens(tokens: list[str], prefix: tuple[str, ...]) -> bool:
    return len(tokens) >= len(prefix) and tuple(tokens[: len(prefix)]) == prefix


def is_allowed_command(command: str) -> bool:
    tokens = _command_tokens(command)

    if not tokens:
        return False

    if tokens[0] in BLOCKED_COMMANDS:
        return False

    return any(_starts_with_tokens(tokens, prefix) for prefix in ALLOWED_COMMAND_PREFIXES)



def run_command(repo_root: Path, command: str, timeout_seconds: int = 60) -> dict[str, object]:
    tokens = _command_tokens(command)

    if not tokens or not is_allowed_command(command):
        return {
            "command": command,
            "returncode": 126,
            "stdout": "",
            "stderr": "Command blocked by coding-agent allowlist.",
        }

    try:
        completed = subprocess.run(
            tokens,
            cwd=repo_root,
            shell=False,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )

    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": f"Executable not found: {exc}",
        }

    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-10_000:],
        "stderr": completed.stderr[-10_000:],
    }
