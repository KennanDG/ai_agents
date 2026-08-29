from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


logger = logging.getLogger(__name__)


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


@dataclass
class MemoryMaintenanceStats:
    checkpoint_threads_deleted: int = 0
    store_items_deleted: int = 0
    duplicate_items_deleted: int = 0
    namespaces_scanned: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    @property
    def total_deleted(self) -> int:
        return (
            self.checkpoint_threads_deleted
            + self.store_items_deleted
            + self.duplicate_items_deleted
        )


_NAMESPACE_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_REQUEST_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")
_REQUEST_STOP_WORDS = {
    "and",
    "for",
    "from",
    "help",
    "implement",
    "please",
    "that",
    "the",
    "this",
    "update",
    "with",
}


def _namespace_segment(
    value: str | None,
    *,
    fallback: str,
) -> str:
    """Produce a SQLite/LangGraph-safe namespace segment."""

    normalized = (value or "").strip().casefold()
    normalized = _NAMESPACE_SEGMENT_RE.sub("-", normalized).strip("-_")
    return normalized[:96] or fallback


def _repo_memory_id(state: CodingAgentState) -> str:
    """Stable identity for the actual repository, not a per-run sandbox."""

    repo_root = state.get("original_repo_root") or state.get("repo_root")
    if not repo_root:
        return "unknown"

    resolved = str(Path(repo_root).expanduser().resolve())
    repo_name = _namespace_segment(Path(resolved).name, fallback="repo")
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    return f"{repo_name}-{digest}"


def memory_namespace(
    state: CodingAgentState,
    context: CodingAgentRuntimeContext | None,
    cfg: CodingAgentSettings = default_settings,
) -> tuple[str, ...]:
    raw_user_id = context.user_id if context else cfg.memory_user_id
    raw_scope = context.memory_namespace if context else cfg.memory_namespace

    user_id = _namespace_segment(raw_user_id, fallback="local")
    scope = _namespace_segment(raw_scope, fallback="default")
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

    cache_dir = Path(cfg.memory_embedding_cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    embeddings = FastEmbedEmbeddings(
        model_name=cfg.memory_embedding_model,
        cache_dir=str(cache_dir),
        doc_embed_type="passage",
    )

    return {
        "embed": embeddings,
        "dims": cfg.memory_embedding_dims,
        "fields": list(cfg.memory_index_fields),
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_maintenance_state(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_maintenance_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _interval_due(
    last_value: Any,
    *,
    interval: timedelta,
    now: datetime,
) -> bool:
    last = _coerce_datetime(last_value)
    return last is None or now - last >= interval


def _latest_checkpoint_activity(
    checkpointer: BaseCheckpointSaver,
) -> dict[str, datetime]:
    """Read only the latest checkpoint per thread when using SqliteSaver.

    The direct grouped query avoids deserializing every historical checkpoint. If a
    different saver is supplied, fall back to the public list API.
    """

    conn = getattr(checkpointer, "conn", None)
    serde = getattr(checkpointer, "serde", None)
    lock = getattr(checkpointer, "lock", None)

    if conn is not None and serde is not None:
        query = """
            WITH latest AS (
                SELECT thread_id, MAX(checkpoint_id) AS checkpoint_id
                FROM checkpoints
                GROUP BY thread_id
            )
            SELECT c.thread_id, c.type, c.checkpoint
            FROM checkpoints AS c
            JOIN latest AS l
              ON l.thread_id = c.thread_id
             AND l.checkpoint_id = c.checkpoint_id
        """

        def read_rows() -> list[tuple[Any, Any, Any]]:
            cursor = conn.execute(query)
            try:
                return list(cursor.fetchall())
            finally:
                cursor.close()

        rows = read_rows() if lock is None else _with_lock(lock, read_rows)
        result: dict[str, datetime] = {}

        for thread_id, type_tag, blob in rows:
            try:
                checkpoint = serde.loads_typed((type_tag, blob))
                ts = _coerce_datetime(
                    checkpoint.get("ts") if isinstance(checkpoint, dict) else None
                )
                if ts is not None:
                    result[str(thread_id)] = ts
            except Exception:
                continue
        return result

    result: dict[str, datetime] = {}

    for item in checkpointer.list(None):
        thread_id = str(item.config.get("configurable", {}).get("thread_id", ""))
        if not thread_id or thread_id in result:
            continue

        checkpoint = getattr(item, "checkpoint", {}) or {}
        ts = _coerce_datetime(checkpoint.get("ts")) if isinstance(checkpoint, dict) else None

        if ts is not None:
            result[thread_id] = ts
    return result


def _with_lock(lock: Any, func):
    with lock:
        return func()


def _prune_checkpoint_threads(
    checkpointer: BaseCheckpointSaver,
    cfg: CodingAgentSettings,
    *,
    now: datetime,
) -> int:
    activity = _latest_checkpoint_activity(checkpointer)
    if not activity:
        return 0

    retention_days = max(1, int(cfg.memory_checkpoint_retention_days))
    max_threads = max(1, int(cfg.memory_checkpoint_max_threads))
    cutoff = now - timedelta(days=retention_days)

    ordered = sorted(activity.items(), key=lambda item: item[1], reverse=True)
    delete_ids: set[str] = set()

    for index, (thread_id, last_seen) in enumerate(ordered):
        if last_seen < cutoff or index >= max_threads:
            delete_ids.add(thread_id)

    deleted = 0

    for thread_id in sorted(delete_ids):
        try:
            checkpointer.delete_thread(thread_id)
            deleted += 1
        except Exception as exc:
            logger.warning("Failed to delete checkpoint thread %s: %s", thread_id, exc)
    return deleted


def _memory_value(item: Any) -> dict[str, Any]:
    value = getattr(item, "value", {}) or {}
    return value if isinstance(value, dict) else {}


def _item_namespace(item: Any) -> tuple[str, ...]:
    value = getattr(item, "namespace", ()) or ()
    return tuple(str(part) for part in value)


def _item_updated_at(item: Any) -> datetime:
    value = _coerce_datetime(getattr(item, "updated_at", None))
    if value is not None:
        return value
    
    memory = _memory_value(item)

    return (
        _coerce_datetime(memory.get("updated_at"))
        or _coerce_datetime(memory.get("created_at"))
        or datetime.min.replace(tzinfo=timezone.utc)
    )


def _list_coding_memory_namespaces(
    store: BaseStore,
    *,
    page_size: int = 100,
) -> list[tuple[str, ...]]:
    namespaces: list[tuple[str, ...]] = []
    offset = 0
    while True:
        page = store.list_namespaces(
            prefix=("coding_agent",),
            max_depth=5,
            limit=page_size,
            offset=offset,
        )
        if not page:
            break
        namespaces.extend(tuple(item) for item in page)
        if len(page) < page_size:
            break
        offset += len(page)

    filtered = [
        namespace
        for namespace in namespaces
        if len(namespace) == 5
        and namespace[0] == "coding_agent"
        and namespace[3] == "repo"
    ]
    return list(dict.fromkeys(filtered))


def _list_namespace_items(
    store: BaseStore,
    namespace: tuple[str, ...],
    *,
    scan_limit: int,
    page_size: int = 100,
) -> list[Any]:
    items: list[Any] = []
    offset = 0
    cap = max(page_size, scan_limit)

    while len(items) < cap:
        limit = min(page_size, cap - len(items))
        page = store.search(namespace, limit=limit, offset=offset)
        if not page:
            break
        items.extend(item for item in page if _item_namespace(item) == namespace)
        if len(page) < limit:
            break
        offset += len(page)

    return items


def _normalize_request(request: str) -> str:
    normalized = request.casefold()
    normalized = re.sub(r"[^a-z0-9_]+", " ", normalized)
    return " ".join(normalized.split())


def _request_tokens(request: str) -> set[str]:
    return {
        token
        for token in _REQUEST_TOKEN_RE.findall(_normalize_request(request))
        if token not in _REQUEST_STOP_WORDS
    }


def _set_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _memory_identity(value: dict[str, Any]) -> str:
    request = str(value.get("request", ""))
    normalized = _normalize_request(request)
    if not normalized:
        return ""
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return f"request:{digest}"


def _prune_store_namespace(
    store: BaseStore,
    namespace: tuple[str, ...],
    cfg: CodingAgentSettings,
    *,
    now: datetime,
) -> tuple[int, int]:
    items = _list_namespace_items(
        store,
        namespace,
        scan_limit=max(100, int(cfg.memory_store_scan_limit)),
    )
    if not items:
        return 0, 0

    ordered = sorted(items, key=_item_updated_at, reverse=True)

    # Conservative exact dedup for historical rows created before stable task keys
    # were introduced. Keep the newest representation of each normalized request.
    duplicate_keys: set[str] = set()
    seen_identity: set[str] = set()

    for item in ordered:
        value = _memory_value(item)
        if value.get("type") != "coding_agent_outcome":
            continue
        identity = _memory_identity(value)
        if not identity:
            continue
        if identity in seen_identity:
            key = str(getattr(item, "key", ""))
            if key and not value.get("pinned", False):
                duplicate_keys.add(key)
        else:
            seen_identity.add(identity)

    for key in duplicate_keys:
        store.delete(namespace, key)

    remaining = [
        item
        for item in ordered
        if str(getattr(item, "key", "")) not in duplicate_keys
    ]

    retention_days = max(1, int(cfg.memory_store_retention_days))
    cutoff = now - timedelta(days=retention_days)
    max_items = max(1, int(cfg.memory_store_max_items_per_namespace))
    min_items = min(
        max_items,
        max(0, int(cfg.memory_store_min_items_per_namespace)),
    )

    delete_keys: set[str] = set()
    unpinned_position = 0
    for item in remaining:
        value = _memory_value(item)
        key = str(getattr(item, "key", ""))
        if not key or value.get("pinned", False):
            continue

        should_keep_minimum = unpinned_position < min_items
        over_cap = unpinned_position >= max_items
        stale = _item_updated_at(item) < cutoff

        if not should_keep_minimum and (over_cap or stale):
            delete_keys.add(key)
        unpinned_position += 1

    for key in delete_keys:
        store.delete(namespace, key)

    return len(delete_keys), len(duplicate_keys)


def _prune_store_memories(
    store: BaseStore,
    cfg: CodingAgentSettings,
    *,
    now: datetime,
    stats: MemoryMaintenanceStats,
) -> None:
    namespaces = _list_coding_memory_namespaces(store)
    stats.namespaces_scanned = len(namespaces)

    for namespace in namespaces:
        try:
            deleted, duplicates = _prune_store_namespace(
                store,
                namespace,
                cfg,
                now=now,
            )
            stats.store_items_deleted += deleted
            stats.duplicate_items_deleted += duplicates
        except Exception as exc:
            stats.errors.append(
                f"Store maintenance failed for {'/'.join(namespace)}: {exc}"
            )


def _run_memory_maintenance(
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    cfg: CodingAgentSettings,
    *,
    now: datetime,
) -> MemoryMaintenanceStats:
    stats = MemoryMaintenanceStats()

    try:
        stats.checkpoint_threads_deleted = _prune_checkpoint_threads(
            checkpointer,
            cfg,
            now=now,
        )
    except Exception as exc:
        stats.errors.append(f"Checkpoint maintenance failed: {exc}")

    _prune_store_memories(store, cfg, now=now, stats=stats)
    return stats


def _compact_sqlite_file(path: Path) -> None:
    if not path.exists():
        return

    with sqlite3.connect(str(path), timeout=30) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA optimize")
        conn.execute("VACUUM")


def _maybe_compact_databases(
    cfg: CodingAgentSettings,
    *,
    now: datetime,
    maintenance_state: dict[str, Any],
    stats: MemoryMaintenanceStats,
) -> bool:
    if not cfg.memory_vacuum_enabled:
        return False

    due = _interval_due(
        maintenance_state.get("last_vacuum_at"),
        interval=timedelta(days=max(1, int(cfg.memory_vacuum_interval_days))),
        now=now,
    )
    if not due:
        return False

    paths = [
        Path(cfg.memory_checkpoint_db_path).expanduser().resolve(),
        Path(cfg.memory_store_db_path).expanduser().resolve(),
    ]
    min_bytes = max(0, int(cfg.memory_vacuum_min_db_bytes))
    should_run = stats.total_deleted > 0 or any(
        path.exists() and path.stat().st_size >= min_bytes for path in paths
    )
    if not should_run:
        return False

    ran = False
    for path in paths:
        try:
            _compact_sqlite_file(path)
            ran = True
        except Exception as exc:
            logger.warning("SQLite compaction failed for %s: %s", path, exc)
    return ran


@contextmanager
def coding_agent_persistence(
    cfg: CodingAgentSettings = default_settings,
    *,
    setup: bool | None = None,
) -> Iterator[CodingAgentPersistence]:
    """Open local SQLite-backed LangGraph persistence with bounded growth.

    Checkpoints and Store use separate database files. Lifecycle maintenance runs
    opportunistically before the graph is yielded, while VACUUM/WAL compaction runs
    only after both LangGraph connections have closed.
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

    checkpoint_path = Path(cfg.memory_checkpoint_db_path).expanduser().resolve()
    store_path = Path(cfg.memory_store_db_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    setup_resources = cfg.memory_setup if setup is None else setup
    index_config = _memory_index_config(cfg)

    now = _utc_now()
    maintenance_state_path = Path(cfg.memory_maintenance_state_path).expanduser().resolve()
    maintenance_state = _load_maintenance_state(maintenance_state_path)
    maintenance_due = bool(cfg.memory_maintenance_enabled) and _interval_due(
        maintenance_state.get("last_maintenance_at"),
        interval=timedelta(
            hours=max(1, int(cfg.memory_maintenance_interval_hours))
        ),
        now=now,
    )
    maintenance_stats = MemoryMaintenanceStats()

    try:
        with ExitStack() as stack:
            checkpointer = stack.enter_context(
                SqliteSaver.from_conn_string(str(checkpoint_path))
            )

            if index_config is None:
                store_context = SqliteStore.from_conn_string(str(store_path))
            else:
                store_context = SqliteStore.from_conn_string(
                    str(store_path),
                    index=index_config,
                )

            store = stack.enter_context(store_context)

            if setup_resources:
                checkpointer.setup()
                store.setup()

            if maintenance_due:
                try:
                    maintenance_stats = _run_memory_maintenance(
                        checkpointer,
                        store,
                        cfg,
                        now=now,
                    )
                    if maintenance_stats.errors:
                        logger.warning(
                            "Coding-agent memory maintenance completed with warnings: %s",
                            "; ".join(maintenance_stats.errors),
                        )
                except Exception as exc:
                    # Memory cleanup must never prevent the coding agent from running.
                    logger.warning("Coding-agent memory maintenance failed: %s", exc)

            yield CodingAgentPersistence(
                checkpointer=checkpointer,
                store=store,
            )
    finally:
        if maintenance_due:
            completed_at = _utc_now()
            vacuumed = _maybe_compact_databases(
                cfg,
                now=completed_at,
                maintenance_state=maintenance_state,
                stats=maintenance_stats,
            )
            new_state = {
                **maintenance_state,
                "last_maintenance_at": completed_at.isoformat(),
                "last_checkpoint_threads_deleted": maintenance_stats.checkpoint_threads_deleted,
                "last_store_items_deleted": maintenance_stats.store_items_deleted,
                "last_duplicate_items_deleted": maintenance_stats.duplicate_items_deleted,
                "last_namespaces_scanned": maintenance_stats.namespaces_scanned,
                "last_errors": maintenance_stats.errors[-10:],
            }
            if vacuumed:
                new_state["last_vacuum_at"] = completed_at.isoformat()
            try:
                _write_maintenance_state(maintenance_state_path, new_state)
            except OSError as exc:
                logger.warning("Could not persist memory maintenance state: %s", exc)


def _runtime_store(runtime: Any) -> BaseStore | None:
    return getattr(runtime, "store", None) if runtime is not None else None


def _runtime_context(runtime: Any) -> CodingAgentRuntimeContext | None:
    context = getattr(runtime, "context", None) if runtime is not None else None
    if isinstance(context, CodingAgentRuntimeContext):
        return context
    return None


def _format_memory_item(item: Any) -> str:
    value = _memory_value(item)
    text = value.get("text") or value.get("summary") or value.get("request") or ""
    score = getattr(item, "score", None)
    score_text = f" relevance={score:.2f}" if isinstance(score, float) else ""
    timestamp = value.get("updated_at") or value.get("created_at", "")
    prefix = f"- Durable outcome{score_text}"
    if timestamp:
        prefix += f" ({timestamp})"
    return f"{prefix}: {truncate(str(text), 1_000)}"


def recall_coding_memories(
    state: CodingAgentState,
    runtime: Any,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Search only durable successful outcomes relevant to the current request.

    Semantic retrieval fails closed. Duplicate historical outcomes with the same
    normalized request are collapsed before they reach the planner.
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
            limit=max(cfg.memory_search_limit * 3, cfg.memory_search_limit),
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

    durable_items: list[Any] = []
    seen_identity: set[str] = set()
    
    for item in items:
        value = _memory_value(item)
        if value.get("type") != "coding_agent_outcome":
            continue
        if value.get("validation_passed", True) is False:
            continue

        identity = _memory_identity(value)
        if identity and identity in seen_identity:
            continue
        if identity:
            seen_identity.add(identity)

        durable_items.append(item)
        if len(durable_items) >= cfg.memory_search_limit:
            break

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
    # Preserve the original key algorithm so existing store rows continue to be
    # overwritten in place after this lifecycle upgrade.
    normalized = " ".join(request.casefold().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return f"task:{digest}"


def _file_overlap(left: list[str], right: list[str]) -> float:
    return _set_overlap(
        {str(item).casefold() for item in left if str(item).strip()},
        {str(item).casefold() for item in right if str(item).strip()},
    )


def _request_overlap(left: str, right: str) -> float:
    return _set_overlap(_request_tokens(left), _request_tokens(right))


def _find_consolidation_target(
    store: BaseStore,
    namespace: tuple[str, ...],
    *,
    request: str,
    changed_files: list[str],
    cfg: CodingAgentSettings,
) -> Any | None:
    if not cfg.memory_consolidation_enabled or not cfg.memory_semantic_enabled:
        return None
    if not request.strip():
        return None

    try:
        candidates = store.search(
            namespace,
            query=request,
            limit=max(1, int(cfg.memory_consolidation_candidate_limit)),
        )
    except Exception:
        return None

    for item in candidates:
        value = _memory_value(item)
        if value.get("type") != "coding_agent_outcome":
            continue
        if value.get("validation_passed", True) is False:
            continue

        score = getattr(item, "score", None)
        if not isinstance(score, (int, float)):
            continue
        if float(score) < float(cfg.memory_consolidation_similarity_threshold):
            continue

        request_overlap = _request_overlap(request, str(value.get("request", "")))
        changed_overlap = _file_overlap(
            changed_files,
            [str(path) for path in value.get("changed_files", [])],
        )
        if (
            request_overlap >= float(cfg.memory_consolidation_min_request_overlap)
            or changed_overlap >= float(cfg.memory_consolidation_min_file_overlap)
        ):
            return item

    return None


def _merge_memory_values(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, Any]:
    previous = previous or {}
    first_created_at = (
        previous.get("first_created_at")
        or previous.get("created_at")
        or current.get("created_at")
        or now.isoformat()
    )

    variants = dedupe(
        [
            *[str(item) for item in previous.get("request_variants", [])],
            str(previous.get("request", "")),
            str(current.get("request", "")),
        ]
    )
    variants = [item for item in variants if item.strip()][-3:]

    return {
        **current,
        "created_at": first_created_at,
        "first_created_at": first_created_at,
        "updated_at": now.isoformat(),
        "observation_count": int(previous.get("observation_count", 0) or 0) + 1,
        "request_variants": variants,
    }


def remember_coding_run(
    state: CodingAgentState,
    runtime: Any,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Persist one compact, consolidated durable outcome for the repository.

    Exact repetitions overwrite the same stable task key. Near-duplicate successful
    outcomes are conservatively merged into an existing semantic neighbor so minor
    wording changes do not create a new vectorized memory row every run.
    """

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

    now = _utc_now()
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

    current_value = {
        "type": "coding_agent_outcome",
        "text": _build_run_memory_text(state),
        "summary": truncate(str(state.get("patch_summary", "")), 1_500),
        "request": request,
        "changed_files": changed_files,
        "completed_units": completed_units,
        "validation_passed": validation_passed,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "observation_count": 1,
    }

    stable_key = _stable_task_memory_key(request)
    key = stable_key
    previous_value: dict[str, Any] | None = None

    try:
        exact = store.get(namespace, stable_key)
        if exact is not None:
            previous_value = _memory_value(exact)
        else:
            target = _find_consolidation_target(
                store,
                namespace,
                request=request,
                changed_files=changed_files,
                cfg=cfg,
            )
            if target is not None:
                target_key = str(getattr(target, "key", "")).strip()
                if target_key:
                    key = target_key
                    previous_value = _memory_value(target)

        value = _merge_memory_values(previous_value, current_value, now=now)
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
