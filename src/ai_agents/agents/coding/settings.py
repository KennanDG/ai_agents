from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodingAgentSettings:
    """Runtime guardrails for the coding agent."""

    repo_root: Path = Path.cwd()
    max_search_results: int = 25
    max_file_chars: int = 25_000
    dry_run: bool = True
    allow_write: bool = False
    allow_shell: bool = True
    shell_timeout_seconds: int = 60


settings = CodingAgentSettings()