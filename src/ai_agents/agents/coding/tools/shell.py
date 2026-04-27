from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

ALLOWED_COMMAND_PREFIXES = (
    "pytest",
    "uv run pytest",
    "ruff check",
    "uv run ruff check",
    "ruff format --check",
    "uv run ruff format --check",
    "python -m pytest",
    "python -m compileall",
)

BLOCKED_COMMANDS = {"sudo", "rm", "rmdir", "del", "format", "shutdown", "reboot", "mkfs", "chmod", "chown"}


def is_allowed_command(command: str) -> bool:
    normalized = " ".join(shlex.split(command))

    if not normalized:
        return False
    
    first = shlex.split(normalized)[0]

    if first in BLOCKED_COMMANDS:
        return False
    
    return any(normalized.startswith(prefix) for prefix in ALLOWED_COMMAND_PREFIXES)



def run_command(repo_root: Path, command: str, timeout_seconds: int = 60) -> dict[str, object]:
    
    if not is_allowed_command(command):
        return {
            "command": command,
            "returncode": 126,
            "stdout": "",
            "stderr": "Command blocked by coding-agent allowlist.",
        }

    try:
        completed = subprocess.run(
            shlex.split(command),
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
