from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


_MEMORY_DB_URI = os.getenv("CODING_AGENT_MEMORY_DB_URI") or os.getenv("DATABASE_URL")


@dataclass(frozen=True)
class CodingAgentSettings:
    """Runtime guardrails for the coding agent."""

    repo_root: Path = Path.cwd()
    max_search_results: int = 50
    max_file_chars: int = 25_000
    dry_run: bool = True
    allow_write: bool = False
    allow_shell: bool = True
    shell_timeout_seconds: int = 60

    # Persistent LangGraph memory.
    # Checkpoints are thread-scoped; store items are long-term/cross-thread.
    memory_db_uri: str | None = _MEMORY_DB_URI
    memory_enabled: bool = _env_bool("CODING_AGENT_MEMORY_ENABLED", bool(_MEMORY_DB_URI))
    memory_setup: bool = _env_bool("CODING_AGENT_MEMORY_SETUP", False)
    memory_user_id: str = os.getenv("CODING_AGENT_MEMORY_USER_ID", "default")
    memory_namespace: str = os.getenv("CODING_AGENT_MEMORY_NAMESPACE", "default")
    memory_search_limit: int = _env_int("CODING_AGENT_MEMORY_SEARCH_LIMIT", 5)
    memory_semantic_enabled: bool = _env_bool(
        "CODING_AGENT_MEMORY_SEMANTIC",
        bool(os.getenv("JINA_API_KEY")),
    )
    memory_embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "google_genai:gemini-embedding-2",
    )
    memory_embedding_dims: int = _env_int("CODING_AGENT_MEMORY_EMBEDDING_DIMS", 768)
    memory_index_fields: tuple[str, ...] = _env_csv(
        "CODING_AGENT_MEMORY_INDEX_FIELDS",
        ("text", "request", "summary"),
    )


settings = CodingAgentSettings()