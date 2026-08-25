from __future__ import annotations

import hashlib
import re

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from ai_agents.agents.coding.coding_agent_settings import (
    CodingAgentSettings,
    settings as default_settings,
)
from ai_agents.agents.coding.state import CodingAgentState
from ai_agents.agents.coding.utils.text import dedupe, truncate


try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.sqlite import SqliteStore
except ImportError:
    SqliteSaver = None
    SqliteStore = None


try:
    from langchain_community.embeddings import FastEmbedEmbeddings
except ImportError:
    FastEmbedEmbeddings = None


@dataclass(frozen=True)
class CodingAgentRuntimeContext:
    """Invocation-scoped context used by LangGraph persistence and memory."""

    user_id: str = "local"
    memory_namespace: str = "default"



@dataclass(frozen=True)
class CodingAgentPersistence:
    checkpointer: BaseCheckpointSaver | None = None
    store: BaseStore | None = None



# def _repo_memory_id(repo_root: str | None) -> str:
#     if not repo_root:
#         return "repo:unknown"

#     resolved = str(Path(repo_root).expanduser().resolve())
#     digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:10]
#     return f"repo:{Path(resolved).name}:{digest}"




# def memory_namespace(
#     state: CodingAgentState,
#     context: CodingAgentRuntimeContext | None,
#     cfg: CodingAgentSettings = default_settings,
# ) -> tuple[str, ...]:
#     user_id = (context.user_id if context else cfg.memory_user_id) or "kennan"
#     namespace = (context.memory_namespace if context else cfg.memory_namespace) or "memory"
#     repo_id = _repo_memory_id(state.get("repo_root"))
#     return ("coding_agent", namespace, user_id, repo_id)




_NAMESPACE_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _namespace_segment(
    value: str | None,
    *,
    fallback: str,
) -> str:
    """
    Produce a SQLite/LangGraph-safe namespace segment.

    SqliteStore internally serializes hierarchical namespaces, so keeping
    each segment simple prevents values containing '.', '/', ':', etc.
    from producing confusing namespace layouts.
    """

    normalized = (value or "").strip().casefold()

    normalized = _NAMESPACE_SEGMENT_RE.sub(
        "-",
        normalized,
    ).strip("-_")

    return normalized[:96] or fallback


def _repo_memory_id(state: CodingAgentState) -> str:
    """
    Stable identity for the actual repository.

    IMPORTANT:
    Do not hash state["repo_root"] when sandboxing is enabled because
    repo_root points at the temporary per-run sandbox.
    """

    repo_root = (
        state.get("original_repo_root")
        or state.get("repo_root")
    )

    if not repo_root:
        return "unknown"

    resolved = str(
        Path(repo_root)
        .expanduser()
        .resolve()
    )

    repo_name = _namespace_segment(
        Path(resolved).name,
        fallback="repo",
    )

    digest = hashlib.sha256(
        resolved.encode("utf-8")
    ).hexdigest()[:12]

    return f"{repo_name}-{digest}"


def memory_namespace(
    state: CodingAgentState,
    context: CodingAgentRuntimeContext | None,
    cfg: CodingAgentSettings = default_settings,
) -> tuple[str, ...]:

    raw_user_id = (
        context.user_id
        if context
        else cfg.memory_user_id
    )

    raw_scope = (
        context.memory_namespace
        if context
        else cfg.memory_namespace
    )

    user_id = _namespace_segment(
        raw_user_id,
        fallback="local",
    )

    scope = _namespace_segment(
        raw_scope,
        fallback="default",
    )

    repo_id = _repo_memory_id(state)

    return (
        "coding_agent",
        user_id,
        scope,
        "repo",
        repo_id,
    )




def _memory_index_config(
    cfg: CodingAgentSettings,
) -> dict[str, Any] | None:

    if not cfg.memory_semantic_enabled:
        return None

    if FastEmbedEmbeddings is None:
        raise RuntimeError(
            "Local semantic memory requires FastEmbed. "
            "Run: uv add fastembed"
        )

    cache_dir = (
        Path(cfg.memory_embedding_cache_dir)
        .expanduser()
        .resolve()
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings = FastEmbedEmbeddings(
        model_name=cfg.memory_embedding_model,
        cache_dir=str(cache_dir),

        # FastEmbed will use the appropriate query embedding path for
        # queries, while stored memories are passages/documents.
        doc_embed_type="passage",
    )

    return {
        "embed": embeddings,
        "dims": cfg.memory_embedding_dims,
        "fields": list(cfg.memory_index_fields),
    }



@contextmanager
def coding_agent_persistence(
    cfg: CodingAgentSettings = default_settings,
    *,
    setup: bool | None = None,
) -> Iterator[CodingAgentPersistence]:
    """
    Open local SQLite-backed LangGraph persistence.

    Checkpoints and Store use different database files to minimize SQLite
    lock contention between graph checkpointing and long-term-memory access.
    """

    if not cfg.memory_enabled:
        yield CodingAgentPersistence()
        return

    if SqliteSaver is None or SqliteStore is None:
        raise RuntimeError(
            "Persistent coding-agent memory requires "
            "langgraph-checkpoint-sqlite. "
            "Run: uv add langgraph-checkpoint-sqlite"
        )

    checkpoint_path = (
        Path(cfg.memory_checkpoint_db_path)
        .expanduser()
        .resolve()
    )

    store_path = (
        Path(cfg.memory_store_db_path)
        .expanduser()
        .resolve()
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    store_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    setup_resources = (
        cfg.memory_setup
        if setup is None
        else setup
    )

    index_config = _memory_index_config(cfg)

    with ExitStack() as stack:

        checkpointer = stack.enter_context(
            SqliteSaver.from_conn_string(
                str(checkpoint_path)
            )
        )

        if index_config is None:
            store_context = SqliteStore.from_conn_string(
                str(store_path)
            )
        else:
            store_context = SqliteStore.from_conn_string(
                str(store_path),
                index=index_config,
            )

        store = stack.enter_context(store_context)

        if setup_resources:
            checkpointer.setup()
            store.setup()

        yield CodingAgentPersistence(
            checkpointer=checkpointer,
            store=store,
        )

        


def _runtime_store(runtime: Any) -> BaseStore | None:
    return getattr(runtime, "store", None) if runtime is not None else None


def _runtime_context(runtime: Any) -> CodingAgentRuntimeContext | None:

    context = getattr(runtime, "context", None) if runtime is not None else None

    if isinstance(context, CodingAgentRuntimeContext):
        return context
    return None




def _memory_value(item: Any) -> dict[str, Any]:
    value = getattr(item, "value", {}) or {}
    return value if isinstance(value, dict) else {}


def _format_memory_item(item: Any) -> str:
    value = _memory_value(item)
    text = value.get("text") or value.get("summary") or value.get("request") or ""
    score = getattr(item, "score", None)
    score_text = f" relevance={score:.2f}" if isinstance(score, float) else ""
    created_at = value.get("created_at", "")
    prefix = f"- Durable outcome{score_text}"
    if created_at:
        prefix += f" ({created_at})"
    return f"{prefix}: {truncate(str(text), 1_000)}"


def recall_coding_memories(
    state: CodingAgentState,
    runtime: Any,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Search only durable successful outcomes relevant to the current request.

    Semantic retrieval fails closed. An embedding/index problem must not silently
    inject unrelated recent runs into the next plan.
    """

    store = _runtime_store(runtime)
    if store is None:
        return {"long_term_memories": [], "memory_enabled": False}

    namespace = memory_namespace(state, _runtime_context(runtime), cfg)
    query = state.get("user_request", "")

    try:
        items = store.search(
            namespace,
            query=query,
            limit=max(cfg.memory_search_limit * 2, cfg.memory_search_limit),
        )
    except Exception as exc:
        return {
            "long_term_memories": [],
            "memory_enabled": True,
            "memory_namespace": "/".join(namespace),
            "memory_errors": [
                *state.get("memory_errors", []),
                f"Semantic memory search failed closed: {exc}",
            ],
        }

    durable_items = [
        item
        for item in items
        if _memory_value(item).get("type") == "coding_agent_outcome"
        and _memory_value(item).get("validation_passed", True) is not False
    ][: cfg.memory_search_limit]

    return {
        "long_term_memories": [_format_memory_item(item) for item in durable_items],
        "memory_enabled": True,
        "memory_namespace": "/".join(namespace),
    }


def _validation_summary(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No validation commands were run."

    return "; ".join(
        f"{result.get('command', 'unknown command')} -> exit code "
        f"{result.get('returncode', 'unknown')}"
        for result in results
    )


def _successful_unit_ids(state: CodingAgentState) -> list[str]:
    successful = {"proposed", "implemented", "completed"}
    return [
        str(unit_id)
        for unit_id, entry in (state.get("completion_ledger") or {}).items()
        if str(entry.get("status", "")) in successful
    ]


def _outcome_is_durable(state: CodingAgentState) -> bool:
    ledger = state.get("completion_ledger") or {}
    if not ledger:
        return False

    durable = {"implemented", "completed"}
    # A dry-run "proposed" unit is useful for the current report but does not
    # describe repository state, so never feed it into future long-term memory.
    if any(str(entry.get("status", "")) not in durable for entry in ledger.values()):
        return False

    return not bool(state.get("blocking_validation_failed"))


def _build_run_memory_text(state: CodingAgentState) -> str:
    changed_files = dedupe(
        [
            str(item.get("path", ""))
            for item in state.get("file_changes", [])
            if item.get("path")
        ]
    )
    completed_units = _successful_unit_ids(state)

    parts = [
        f"Request: {state.get('user_request', '')}",
        (
            "Completed implementation units: "
            + (", ".join(completed_units) if completed_units else "none")
        ),
        (
            "Files changed/proposed: "
            + (", ".join(changed_files) if changed_files else "none")
        ),
        f"Validation: {_validation_summary(state.get('validation_results', []))}",
    ]

    patch_summary = str(state.get("patch_summary", "")).strip()
    if patch_summary:
        parts.append(f"Outcome summary: {truncate(patch_summary, 1_500)}")

    return "\n".join(parts)


def _stable_task_memory_key(request: str) -> str:
    normalized = " ".join(request.casefold().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return f"task:{digest}"


def remember_coding_run(
    state: CodingAgentState,
    runtime: Any,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Persist one compact durable outcome per normalized task."""

    store = _runtime_store(runtime)
    if store is None:
        return {"memory_saved": False, "memory_enabled": False}

    namespace = memory_namespace(state, _runtime_context(runtime), cfg)

    if not _outcome_is_durable(state):
        return {
            "memory_saved": False,
            "memory_enabled": True,
            "memory_namespace": "/".join(namespace),
        }

    created_at = datetime.now(timezone.utc).isoformat()
    request = state.get("user_request", "")
    changed_files = dedupe(
        [
            str(item.get("path", ""))
            for item in state.get("file_changes", [])
            if item.get("path")
        ]
    )
    completed_units = _successful_unit_ids(state)
    validation_results = state.get("validation_results", [])
    validation_passed = all(
        int(result.get("returncode", 0) or 0) == 0
        for result in validation_results
    )

    value = {
        "type": "coding_agent_outcome",
        "text": _build_run_memory_text(state),
        "summary": truncate(str(state.get("patch_summary", "")), 1_500),
        "request": request,
        "changed_files": changed_files,
        "completed_units": completed_units,
        "validation_passed": validation_passed,
        "created_at": created_at,
    }
    key = _stable_task_memory_key(request)

    try:
        put_kwargs = (
            {"index": list(cfg.memory_index_fields)}
            if cfg.memory_semantic_enabled
            else {}
        )
        store.put(namespace, key, value, **put_kwargs)
    except Exception as exc:
        return {
            "memory_saved": False,
            "memory_enabled": True,
            "memory_namespace": "/".join(namespace),
            "memory_errors": [
                *state.get("memory_errors", []),
                f"Memory write failed: {exc}",
            ],
        }

    return {
        "memory_saved": True,
        "memory_enabled": True,
        "memory_namespace": "/".join(namespace),
    }
