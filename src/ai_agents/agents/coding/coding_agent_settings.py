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


# _MEMORY_DB_URI = os.getenv("CODING_AGENT_MEMORY_DB_URI") or os.getenv("DATABASE_URL")

_MEMORY_DIR = Path(
    os.getenv(
        "CODING_AGENT_MEMORY_DIR",
        ".ai-agents/memory",
    )
).expanduser()

_MEMORY_CHECKPOINT_DB = Path(
    os.getenv(
        "CODING_AGENT_MEMORY_CHECKPOINT_DB",
        str(_MEMORY_DIR / "checkpoints.sqlite3"),
    )
).expanduser()

_MEMORY_STORE_DB = Path(
    os.getenv(
        "CODING_AGENT_MEMORY_STORE_DB",
        str(_MEMORY_DIR / "store.sqlite3"),
    )
).expanduser()

_MEMORY_EMBEDDING_CACHE = Path(
    os.getenv(
        "CODING_AGENT_MEMORY_EMBEDDING_CACHE_DIR",
        str(_MEMORY_DIR / "fastembed-cache"),
    )
).expanduser()


@dataclass(frozen=True)
class CodingAgentSettings:
    """Runtime guardrails for the coding agent."""

    repo_root: Path = Path.cwd()
    max_search_results: int = _env_int("CODING_AGENT_MAX_SEARCH_RESULTS", 50)

    # Context engineering. Large files are retained at intake, then reduced to
    # relevant line windows before they are placed in an LLM prompt.
    max_file_chars: int = _env_int("CODING_AGENT_MAX_FILE_CHARS", 1_000_000)
    max_full_file_chars: int = _env_int("CODING_AGENT_MAX_FULL_FILE_CHARS", 60_000)
    context_chunk_chars: int = _env_int("CODING_AGENT_CONTEXT_CHUNK_CHARS", 12_000)
    context_chunk_overlap_chars: int = _env_int(
        "CODING_AGENT_CONTEXT_CHUNK_OVERLAP_CHARS", 1_500
    )
    # LLM prompt budgets are expressed in tokens. Character limits above remain
    # storage/read safeguards only. Each worker clamps this configured budget to the
    # selected model slot's context window after reserving output and safety tokens.
    context_prompt_base_tokens: int = _env_int(
        "CODING_AGENT_CONTEXT_PROMPT_BASE_TOKENS", 48_000
    )
    max_context_prompt_tokens: int = _env_int(
        "CODING_AGENT_MAX_CONTEXT_PROMPT_TOKENS", 96_000
    )
    context_prompt_reserve_tokens: int = _env_int(
        "CODING_AGENT_CONTEXT_PROMPT_RESERVE_TOKENS", 10_000
    )
    context_window_safety_tokens: int = _env_int(
        "CODING_AGENT_CONTEXT_WINDOW_SAFETY_TOKENS", 6_000
    )

    # Context-window fallbacks are slot-specific because the coding and reasoning
    # models may use different providers/models. Override these when the selected
    # provider advertises a different window.
    coding_model_context_window_tokens: int = _env_int(
        "CODING_AGENT_CODING_CONTEXT_WINDOW_TOKENS", 131_072
    )
    reasoning_model_context_window_tokens: int = _env_int(
        "CODING_AGENT_REASONING_CONTEXT_WINDOW_TOKENS", 131_072
    )

    coding_model_max_output_tokens: int = _env_int(
        "CODING_AGENT_CODING_MAX_OUTPUT_TOKENS", 32_000
    )
    reasoning_model_max_output_tokens: int = _env_int(
        "CODING_AGENT_REASONING_MAX_OUTPUT_TOKENS", 32_000
    )

    # Optional exact provider/model overrides, e.g.
    # {"groq:openai/gpt-oss-120b": 131072, "deepseek:deepseek-v4-pro": 163840}
    # This lets the desktop runtime stay model-aware without hardcoding a provider
    # catalog that can become stale.
    model_context_window_overrides_json: str = os.getenv(
        "CODING_AGENT_MODEL_CONTEXT_WINDOW_OVERRIDES_JSON", "{}"
    )
    model_max_output_overrides_json: str = os.getenv(
        "CODING_AGENT_MODEL_MAX_OUTPUT_OVERRIDES_JSON", "{}"
    )

    # Worker count controls concurrency, not total work decomposition. A plan may
    # contain more implementation units than active workers; unfinished units are
    # scheduled in later deterministic batches.
    max_context_workers: int = _env_int("CODING_AGENT_MAX_CONTEXT_WORKERS", 3)
    max_worker_files: int = _env_int("CODING_AGENT_MAX_WORKER_FILES", 6)
    max_implementation_units: int = _env_int(
        "CODING_AGENT_MAX_IMPLEMENTATION_UNITS", 12
    )
    max_patch_retries_per_unit: int = _env_int(
        "CODING_AGENT_MAX_PATCH_RETRIES_PER_UNIT", 2
    )
    max_implementation_iterations: int = _env_int(
        "CODING_AGENT_MAX_IMPLEMENTATION_ITERATIONS", 6
    )
    max_attached_files: int = _env_int("CODING_AGENT_MAX_ATTACHED_FILES", 20)
    max_attachment_storage_chars: int = _env_int(
        "CODING_AGENT_MAX_ATTACHMENT_STORAGE_CHARS", 1_000_000
    )
    max_total_attachment_storage_chars: int = _env_int(
        "CODING_AGENT_MAX_TOTAL_ATTACHMENT_STORAGE_CHARS", 3_000_000
    )

    # Latency controls. Deterministic routing/navigation remove unnecessary LLM
    # calls; implementation workers use the fast coding model only for simple first
    # attempts and otherwise use the reasoning model for isolated unit proposals.
    fast_path_enabled: bool = _env_bool("CODING_AGENT_FAST_PATH_ENABLED", True)
    llm_skill_routing_enabled: bool = _env_bool(
        "CODING_AGENT_LLM_SKILL_ROUTING_ENABLED", False
    )
    llm_navigation_enabled: bool = _env_bool(
        "CODING_AGENT_LLM_NAVIGATION_ENABLED", False
    )
    model_timeout_seconds: int = _env_int("CODING_AGENT_MODEL_TIMEOUT_SECONDS", 120)

    
    prompt_caching_enabled: bool = _env_bool("CODING_AGENT_PROMPT_CACHING_ENABLED", True)
    prompt_cache_version: str = os.getenv("CODING_AGENT_PROMPT_CACHE_VERSION", "v1")
    anthropic_prompt_cache_ttl: str = os.getenv(
        "CODING_AGENT_ANTHROPIC_PROMPT_CACHE_TTL",
        "5m",
    )

    # Output-token budgets. These are run-time tunable, but the API enforces
    # conservative lower/upper bounds so one user cannot create an unbounded run.
    route_max_tokens: int = _env_int("CODING_AGENT_ROUTE_MAX_TOKENS", 900)
    planner_max_tokens: int = _env_int("CODING_AGENT_PLANNER_MAX_TOKENS", 3_000)
    repo_navigation_max_tokens: int = _env_int(
        "CODING_AGENT_REPO_NAVIGATION_MAX_TOKENS", 1_600
    )
    simple_patch_max_tokens: int = _env_int(
        "CODING_AGENT_SIMPLE_PATCH_MAX_TOKENS", 8_000
    )
    patch_max_tokens: int = _env_int("CODING_AGENT_PATCH_MAX_TOKENS", 20_000)
    progress_max_tokens: int = _env_int("CODING_AGENT_PROGRESS_MAX_TOKENS", 1_200)

    dry_run: bool = True
    allow_write: bool = False
    allow_shell: bool = True
    shell_timeout_seconds: int = 60

    # Persistent LangGraph memory.
    # Checkpoints are thread-scoped; store items are long-term/cross-thread.
    memory_checkpoint_db_path: Path = _MEMORY_CHECKPOINT_DB
    memory_store_db_path: Path = _MEMORY_STORE_DB

    # Local desktop builds should have persistence enabled by default.
    memory_enabled: bool = _env_bool(
        "CODING_AGENT_MEMORY_ENABLED",
        True,
    )

    # SQLite setup is cheap/idempotent, so initialize automatically.
    memory_setup: bool = _env_bool(
        "CODING_AGENT_MEMORY_SETUP",
        True,
    )

    # Single-user local installation defaults.
    memory_user_id: str = os.getenv(
        "CODING_AGENT_MEMORY_USER_ID",
        "local",
    )

    memory_namespace: str = os.getenv(
        "CODING_AGENT_MEMORY_NAMESPACE",
        "default",
    )

    memory_search_limit: int = _env_int(
        "CODING_AGENT_MEMORY_SEARCH_LIMIT",
        3,
    )

    # Semantic memory is fully local.
    memory_semantic_enabled: bool = _env_bool(
        "CODING_AGENT_MEMORY_SEMANTIC",
        True,
    )

    # Do NOT reuse the application's generic EMBEDDING_MODEL setting.
    # Coding-agent memory now has its own local embedding configuration.
    memory_embedding_model: str = os.getenv(
        "CODING_AGENT_MEMORY_EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5",
    )

    # bge-small-en-v1.5 outputs 384-dimensional vectors.
    memory_embedding_dims: int = _env_int(
        "CODING_AGENT_MEMORY_EMBEDDING_DIMS",
        384,
    )

    memory_embedding_cache_dir: Path = _MEMORY_EMBEDDING_CACHE

    memory_index_fields: tuple[str, ...] = _env_csv(
        "CODING_AGENT_MEMORY_INDEX_FIELDS",
        ("text", "request", "summary"),
    )


settings = CodingAgentSettings()








# memory_db_uri: str | None = _MEMORY_DB_URI
#     memory_enabled: bool = _env_bool("CODING_AGENT_MEMORY_ENABLED", bool(_MEMORY_DB_URI))
#     memory_setup: bool = _env_bool("CODING_AGENT_MEMORY_SETUP", False)
#     memory_user_id: str = os.getenv("CODING_AGENT_MEMORY_USER_ID", "default")
#     memory_namespace: str = os.getenv("CODING_AGENT_MEMORY_NAMESPACE", "default")
#     memory_search_limit: int = _env_int("CODING_AGENT_MEMORY_SEARCH_LIMIT", 5)
#     memory_semantic_enabled: bool = _env_bool(
#         "CODING_AGENT_MEMORY_SEMANTIC",
#         bool(os.getenv("JINA_API_KEY")),
#     )
#     memory_embedding_model: str = os.getenv(
#         "EMBEDDING_MODEL",
#         "BAAI/bge-small-en-v1.5",
#     )
#     memory_embedding_dims: int = _env_int("CODING_AGENT_MEMORY_EMBEDDING_DIMS", 768)
#     memory_index_fields: tuple[str, ...] = _env_csv(
#         "CODING_AGENT_MEMORY_INDEX_FIELDS",
#         ("text", "request", "summary"),
#     )