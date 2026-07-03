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

SHELL_CONTROL_TOKENS = {
    "&&",
    "||",
    ";",
    "|",
    ">",
    ">>",
    "<",
}


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _starts_with_tokens(tokens: list[str], prefix: tuple[str, ...]) -> bool:
    return len(tokens) >= len(prefix) and tuple(tokens[: len(prefix)]) == prefix


def _strip_safe_cd_prefix(tokens: list[str]) -> tuple[list[str], str | None]:
    """Support commands like: cd agents/frontend && npx tsc --noEmit.

    This keeps shell=False while still allowing a narrowly-scoped cwd change.
    """
    if not tokens or tokens[0] != "cd":
        return tokens, None

    if len(tokens) < 4 or tokens[2] != "&&":
        return [], None

    cd_target = tokens[1]
    remaining = tokens[3:]

    if not remaining:
        return [], None

    return remaining, cd_target


def _is_allowed_tokens(tokens: list[str]) -> bool:
    if not tokens:
        return False

    if tokens[0] in BLOCKED_COMMANDS:
        return False

    if any(token in SHELL_CONTROL_TOKENS for token in tokens):
        return False

    return any(_starts_with_tokens(tokens, prefix) for prefix in ALLOWED_COMMAND_PREFIXES)


def is_allowed_command(command: str) -> bool:
    tokens = _command_tokens(command)
    tokens, _cwd = _strip_safe_cd_prefix(tokens)
    return _is_allowed_tokens(tokens)


def _resolve_working_directory(repo_root: Path, cwd_fragment: str | None) -> Path:
    root = repo_root.resolve()

    if not cwd_fragment:
        return root

    cwd_path = Path(cwd_fragment)

    if cwd_path.is_absolute() or ".." in cwd_path.parts:
        raise ValueError(f"Unsafe command working directory: {cwd_fragment}")

    resolved = (root / cwd_path).resolve()

    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Command working directory escapes repository root: {cwd_fragment}")

    if not resolved.is_dir():
        raise FileNotFoundError(f"Command working directory does not exist: {cwd_fragment}")

    return resolved


def run_command(repo_root: Path, command: str, timeout_seconds: int = 60) -> dict[str, object]:
    original_command = command
    tokens = _command_tokens(command)
    tokens, cwd_fragment = _strip_safe_cd_prefix(tokens)

    if not tokens or not _is_allowed_tokens(tokens):
        return {
            "command": original_command,
            "returncode": 126,
            "stdout": "",
            "stderr": "Command blocked by coding-agent allowlist.",
        }

    try:
        cwd = _resolve_working_directory(repo_root, cwd_fragment)

        completed = subprocess.run(
            tokens,
            cwd=cwd,
            shell=False,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )

    except FileNotFoundError as exc:
        return {
            "command": original_command,
            "returncode": 127,
            "stdout": "",
            "stderr": f"Executable not found: {exc}",
        }

    except Exception as exc:
        return {
            "command": original_command,
            "returncode": 1,
            "stdout": "",
            "stderr": str(exc),
        }

    return {
        "command": original_command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-10_000:],
        "stderr": completed.stderr[-10_000:],
    }