from __future__ import annotations

import hashlib
import re

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from ai_agents.agents.coding.coding_agent_settings import (
    CodingAgentSettings,
    settings as default_settings,
)
from ai_agents.agents.coding.state import CodingAgentState
from ai_agents.agents.coding.utils.text import bullets, truncate


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




def _format_memory_item(item: Any) -> str:

    value = getattr(item, "value", {}) or {}

    if not isinstance(value, dict):
        return truncate(str(value), 1_000)

    text = value.get("text") or value.get("summary") or value.get("request") or ""
    score = getattr(item, "score", None)
    score_text = f" relevance={score:.2f}" if isinstance(score, float) else ""
    created_at = value.get("created_at", "")
    prefix = f"- Memory{score_text}"

    if created_at:
        prefix += f" ({created_at})"
    return f"{prefix}: {truncate(str(text), 1_000)}"






def recall_coding_memories(
    state: CodingAgentState,
    runtime: Any,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Search cross-thread coding memories relevant to the current request."""

    store = _runtime_store(runtime)

    if store is None:
        return {"long_term_memories": [], "memory_enabled": False}

    namespace = memory_namespace(state, _runtime_context(runtime), cfg)
    query = state.get("user_request", "")

    try:
        items = store.search(namespace, query=query, limit=cfg.memory_search_limit)
    except Exception as exc:
        try:
            items = store.search(namespace, limit=cfg.memory_search_limit)
        except Exception as fallback_exc:
            return {
                "long_term_memories": [],
                "memory_enabled": True,
                "memory_namespace": "/".join(namespace),
                "memory_errors": [
                    *state.get("memory_errors", []),
                    f"Memory search failed: {exc}; fallback search failed: {fallback_exc}",
                ],
            }

    return {
        "long_term_memories": [_format_memory_item(item) for item in items],
        "memory_enabled": True,
        "memory_namespace": "/".join(namespace),
    }






def _validation_summary(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No validation commands were run."

    lines = []

    for result in results:
        lines.append(
            f"{result.get('command', 'unknown command')} -> exit code "
            f"{result.get('returncode', 'unknown')}"
        )

    return "; ".join(lines)





def _build_run_memory_text(state: CodingAgentState) -> str:

    changed_files = [
        item.get("path", "")
        for item in state.get("file_changes", [])
        if item.get("path")
    ]

    inspected = state.get("files_inspected", [])
    errors = state.get("errors", [])[-5:]

    parts = [
        f"Request: {state.get('user_request', '')}",
        f"Selected skill: {state.get('selected_skill', 'none')}",
        f"Final status: {state.get('status', 'unknown')}",
        f"Files inspected: {', '.join(inspected) if inspected else 'none'}",
        f"Files changed/proposed: {', '.join(changed_files) if changed_files else 'none'}",
        f"Validation: {_validation_summary(state.get('validation_results', []))}",
    ]

    patch_summary = state.get("patch_summary")
    if patch_summary:
        parts.append(f"Patch summary: {truncate(patch_summary, 1_500)}")

    if errors:
        parts.append("Errors: " + bullets(errors))

    return "\n".join(parts)





def remember_coding_run(
    state: CodingAgentState,
    runtime: Any,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Persist a compact cross-thread memory after the run report is produced."""

    store = _runtime_store(runtime)
    
    if store is None:
        return {"memory_saved": False, "memory_enabled": False}

    namespace = memory_namespace(state, _runtime_context(runtime), cfg)
    created_at = datetime.now(timezone.utc).isoformat()
    text = _build_run_memory_text(state)
    key = f"run:{created_at}:{uuid4().hex}"

    value = {
        "type": "coding_agent_run",
        "text": text,
        "summary": state.get("patch_summary", ""),
        "request": state.get("user_request", ""),
        "selected_skill": state.get("selected_skill"),
        "status": state.get("status"),
        "files_inspected": state.get("files_inspected", []),
        "file_changes": state.get("file_changes", []),
        "validation_results": state.get("validation_results", []),
        "created_at": created_at,
    }

    try:
        put_kwargs = {"index": list(cfg.memory_index_fields)} if cfg.memory_semantic_enabled else {}
        store.put(namespace, key, value, **put_kwargs)
    except Exception as exc:
        return {
            "memory_saved": False,
            "memory_enabled": True,
            "memory_namespace": "/".join(namespace),
            "memory_errors": [*state.get("memory_errors", []), f"Memory write failed: {exc}"],
        }

    return {
        "memory_saved": True,
        "memory_enabled": True,
        "memory_namespace": "/".join(namespace),
    }

