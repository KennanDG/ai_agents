from __future__ import annotations

import os
import asyncio
import base64
import binascii
import mimetypes
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from ai_agents.agents.coding.graph import build_coding_agent_graph
from ai_agents.agents.coding.model_factory import caption_model
from ai_agents.agents.coding.memory import (
    CodingAgentRuntimeContext,
    coding_agent_persistence,
)
from ai_agents.agents.coding.coding_agent_settings import settings as default_coding_settings
from ai_agents.api.auth import authorize_websocket, generate_websocket_token
from ai_agents.api.api_schemas import (
    CodingAgentClientMessage,
    CodingAgentRunRequest,
    CodingAgentRunResult,
    CodingAgentServerEvent,
    RepositoryFileResponse,
    RepositoryTreeEntry,
    RepositoryTreeResponse,
)
from ai_agents.agents.coding.sandbox import (
    CodingSandbox,
    apply_sandbox_files_to_repo,
    cleanup_coding_sandbox,
    create_coding_sandbox,
)
from ai_agents.agents.coding.utils.validation import validation_failed_results
from ai_agents.config.settings import settings as config_settings
from ai_agents.config.constants import (
    IGNORED_REPOSITORY_FILES, 
    IGNORED_REPOSITORY_DIRS,
    MAX_REPOSITORY_FILE_BYTES,
    MAX_ATTACHED_FILES,
    MAX_ATTACHMENT_CHARS,
    MAX_TOTAL_ATTACHMENT_CHARS,
    MAX_ATTACHED_IMAGE_BYTES,
    ALLOWED_IMAGE_MIME_TYPES,
    IMAGE_DATA_URL_RE,
    LANGUAGE_BY_EXTENSION,
)




router = APIRouter(prefix="/coding-agent", tags=["coding-agent"])





@dataclass
class PendingCodingAgentRun:
    run_id: str
    thread_id: str
    sandbox: CodingSandbox
    final_state: dict[str, Any]
    changed_paths: list[str]


def _changed_paths_from_state(state: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()

    for item in state.get("file_changes", []):
        if not isinstance(item, dict):
            continue

        path = str(item.get("path", "")).strip()

        if not path or path in seen:
            continue

        paths.append(path)
        seen.add(path)

    return paths







#############################################################################
############################## Repository Tree ##############################
#############################################################################
def _resolve_repo_root(repo_root: str) -> Path:
    root = Path(repo_root).expanduser().resolve()

    if not root.exists():
        raise HTTPException(status_code=404, detail="Repository root does not exist.")

    if not root.is_dir():
        raise HTTPException(status_code=400, detail="Repository root must be a directory.")

    return root



def _resolve_repo_file(root: Path, relative_path: str) -> Path:
    target = (root / relative_path).resolve()

    if target != root and root not in target.parents:
        raise HTTPException(status_code=400, detail="File path escapes repository root.")

    if not target.exists():
        raise HTTPException(status_code=404, detail="Repository file does not exist.")

    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file.")

    return target



def _repository_language(path: Path) -> str:
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "plaintext")


def _is_ignored_repository_dir(name: str) -> bool:
    return name in IGNORED_REPOSITORY_DIRS or name.endswith(".egg-info")





@router.get("/repository/tree", response_model=RepositoryTreeResponse)
def repository_tree(
    repo_root: str = Query(".", description="Absolute or service-relative repository root."),
    max_depth: int = Query(8, ge=1, le=32),
    max_entries: int = Query(1500, ge=1, le=5000),
) -> RepositoryTreeResponse:
    
    root = _resolve_repo_root(repo_root)
    entries: list[RepositoryTreeEntry] = []

    for current_dir, dir_names, file_names in os.walk(root):
        current_path = Path(current_dir)
        relative_dir = current_path.relative_to(root)
        current_depth = len(relative_dir.parts)

        dir_names[:] = [
            name for name in sorted(dir_names) if not _is_ignored_repository_dir(name)
        ]

        if current_depth >= max_depth:
            dir_names[:] = []

        for directory_name in dir_names:
            directory_path = current_path / directory_name
            relative_path = directory_path.relative_to(root).as_posix()

            entries.append(
                RepositoryTreeEntry(
                    path=relative_path,
                    name=directory_name,
                    kind="directory",
                    depth=current_depth,
                )
            )

            if len(entries) >= max_entries:
                return RepositoryTreeResponse(repo_root=str(root), entries=entries)

        for file_name in sorted(file_names):
            if file_name in IGNORED_REPOSITORY_FILES:
                continue

            file_path = current_path / file_name

            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            relative_path = file_path.relative_to(root).as_posix()

            entries.append(
                RepositoryTreeEntry(
                    path=relative_path,
                    name=file_name,
                    kind="file",
                    depth=current_depth,
                    size=size,
                )
            )

            if len(entries) >= max_entries:
                return RepositoryTreeResponse(repo_root=str(root), entries=entries)

    return RepositoryTreeResponse(repo_root=str(root), entries=entries)



@router.get("/repository/file", response_model=RepositoryFileResponse)
def repository_file(
    path: str = Query(..., min_length=1),
    repo_root: str = Query(".", description="Absolute or service-relative repository root."),
) -> RepositoryFileResponse:
    
    root = _resolve_repo_root(repo_root)
    file_path = _resolve_repo_file(root, path)
    size = file_path.stat().st_size

    if size > MAX_REPOSITORY_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large to preview. Limit is {MAX_REPOSITORY_FILE_BYTES} bytes.",
        )

    raw = file_path.read_bytes()
    if b"\0" in raw[:4096]:
        raise HTTPException(status_code=415, detail="Binary files cannot be previewed.")

    return RepositoryFileResponse(
        repo_root=str(root),
        path=path,
        language=_repository_language(file_path),
        content=raw.decode("utf-8", errors="replace"),
        size=size,
    )





#######################################################################
############################## WebSocket ##############################
#######################################################################




############################## File attachment helpers ##############################
def _coerce_message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part.strip() for part in parts if part.strip()).strip()

    return str(content).strip()


def _is_supported_image_mime(mime_type: str | None) -> bool:
    if not mime_type:
        return False

    normalized = mime_type.lower().replace("image/jpg", "image/jpeg")
    return normalized in ALLOWED_IMAGE_MIME_TYPES


def _guess_mime_type(path: Path) -> str | None:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed


def _image_data_url(mime_type: str, raw: bytes) -> str:
    normalized = mime_type.lower().replace("image/jpg", "image/jpeg")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{normalized};base64,{encoded}"


def _parse_image_data_url(data_url: str) -> tuple[str, bytes]:
    match = IMAGE_DATA_URL_RE.match(data_url.strip())

    if not match:
        raise ValueError("expected a base64 data URL for a PNG, JPEG, or WebP image.")

    mime_type = match.group("mime").lower().replace("image/jpg", "image/jpeg")
    if not _is_supported_image_mime(mime_type):
        raise ValueError(f"unsupported image MIME type: {mime_type}")

    try:
        raw = base64.b64decode(match.group("data"), validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid base64 image data.") from exc

    if len(raw) > MAX_ATTACHED_IMAGE_BYTES:
        raise ValueError(
            f"image is too large ({len(raw)} bytes). Limit is {MAX_ATTACHED_IMAGE_BYTES} bytes."
        )

    return mime_type, raw


def _describe_image_attachment(
    *,
    name: str,
    mime_type: str,
    data_url: str,
) -> str:
    vision_model = caption_model()

    response = vision_model.invoke(
        [
            SystemMessage(
                content=(
                    "You convert user-attached images into concise, useful text context "
                    "for a coding agent. Focus on visible UI, screenshots, diagrams, "
                    "error messages, labels, tables, layout, code snippets, and other "
                    "implementation-relevant details. Do not guess hidden data."
                )
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Describe this uploaded image for downstream coding work. "
                            f"File name: {name}. MIME type: {mime_type}."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ]
            ),
        ]
    )

    description = _coerce_message_content_to_text(response.content)
    if not description:
        raise RuntimeError("vision model returned an empty image description.")

    return (
        f"Image attachment: {name}\n"
        f"MIME type: {mime_type}\n"
        "Vision-generated context:\n"
        f"{description}"
    )



def _truncate_text(value: str, max_chars: int) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True



def _find_matching_repo_text_file(
    *,
    repo_root: Path,
    name: str,
    content: str,
) -> str | None:
    
    target_name = Path(name).name
    candidates: list[tuple[str, str]] = []

    for current_dir, dir_names, file_names in os.walk(repo_root):
        dir_names[:] = [
            directory_name
            for directory_name in dir_names
            if not _is_ignored_repository_dir(directory_name)
        ]

        if target_name not in file_names:
            continue

        file_path = Path(current_dir) / target_name

        try:
            size = file_path.stat().st_size
            if size > MAX_REPOSITORY_FILE_BYTES:
                continue

            raw = file_path.read_bytes()
            if b"\0" in raw[:4096]:
                continue

            text = raw.decode("utf-8", errors="replace")
            relative_path = file_path.relative_to(repo_root).as_posix()
            candidates.append((relative_path, text))
        except OSError:
            continue

    exact_matches = [
        relative_path
        for relative_path, text in candidates
        if text == content or text.rstrip("\n") == content.rstrip("\n")
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(candidates) == 1:
        return candidates[0][0]

    return None




def _normalize_attached_files(
    *,
    request: CodingAgentRunRequest,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:

    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    total_chars = 0

    for index, attached in enumerate(request.attached_files[:MAX_ATTACHED_FILES]):
        name = Path(attached.name).name.strip() or f"attachment-{index + 1}.txt"
        source = attached.source
        path = attached.path.strip() if attached.path else None
        content = attached.content or ""
        mime_type = attached.mime_type

        if source == "repo":
            if not path:
                errors.append(f"Skipped repo attachment {name}: missing repo-relative path.")
                continue

            try:
                file_path = _resolve_repo_file(repo_root, path)
                size = file_path.stat().st_size

                if size > MAX_REPOSITORY_FILE_BYTES:
                    errors.append(
                        f"Skipped repo attachment {path}: file is too large "
                        f"({size} bytes)."
                    )
                    continue

                raw = file_path.read_bytes()
                mime_type = mime_type or _guess_mime_type(file_path)
                name = file_path.name

                if _is_supported_image_mime(mime_type):
                    if size > MAX_ATTACHED_IMAGE_BYTES:
                        errors.append(
                            f"Skipped image attachment {path}: image is too large "
                            f"({size} bytes)."
                        )
                        continue

                    data_url = _image_data_url(mime_type or "image/png", raw)
                    content = _describe_image_attachment(
                        name=name,
                        mime_type=mime_type or "image/png",
                        data_url=data_url,
                    )

                else:
                    if b"\0" in raw[:4096]:
                        errors.append(f"Skipped repo attachment {path}: binary file.")
                        continue

                    normalized.append(
                        {
                            "name": name,
                            "path": path,
                            "source": "repo",
                            "mime_type": mime_type,
                            "size": size,
                            "content": "",
                            "truncated": False,
                        }
                    )
                    continue

            except HTTPException as exc:
                errors.append(f"Skipped repo attachment {path}: {exc.detail}")
                continue
            except Exception as exc:
                errors.append(f"Skipped repo attachment {path}: {exc}")
                continue

        else: # source == "upload"
            data_url = attached.data_url

            # If attachment is an image
            if data_url:
                try:
                    parsed_mime_type, raw = _parse_image_data_url(data_url)
                    mime_type = mime_type or parsed_mime_type
                    content = _describe_image_attachment(
                        name=name,
                        mime_type=parsed_mime_type,
                        data_url=data_url,
                    )
                except Exception as exc:
                    errors.append(f"Skipped image attachment {name}: {exc}")
                    continue

            elif _is_supported_image_mime(mime_type):
                errors.append(
                    f"Skipped image attachment {name}: missing base64 data_url payload."
                )
                continue

            # Checks if uploaded file is an existing repo file
            if content.strip():
                matched_repo_path = _find_matching_repo_text_file(
                    repo_root=repo_root,
                    name=name,
                    content=content,
                )

                if matched_repo_path:
                    matched_file = _resolve_repo_file(repo_root, matched_repo_path)

                    normalized.append(
                        {
                            "name": matched_file.name,
                            "path": matched_repo_path,
                            "source": "repo",
                            "mime_type": mime_type or _guess_mime_type(matched_file),
                            "size": matched_file.stat().st_size,
                            "content": "",
                            "truncated": False,
                        }
                    )
                    continue



        if not content.strip():
            errors.append(f"Skipped attachment {name}: empty content.")
            continue

        # Preserve complete text at intake. Prompt-time context selection happens
        # later in parallel workers, so silently truncating here only guarantees that
        # the patcher can never recover the omitted section.
        if len(content) > MAX_ATTACHMENT_CHARS:
            errors.append(
                f"Skipped attachment {name}: {len(content)} characters exceeds the "
                f"{MAX_ATTACHMENT_CHARS}-character intake ceiling. Attach it as a "
                "repository file/path or raise CODING_AGENT_MAX_ATTACHMENT_STORAGE_CHARS."
            )
            continue

        if total_chars + len(content) > MAX_TOTAL_ATTACHMENT_CHARS:
            errors.append(
                f"Skipped attachment {name}: total attachment storage would exceed "
                f"{MAX_TOTAL_ATTACHMENT_CHARS} characters."
            )
            continue

        total_chars += len(content)
        normalized.append(
            {
                "name": name,
                "path": path,
                "source": source,
                "mime_type": mime_type,
                "size": attached.size,
                "content": content,
                "truncated": bool(attached.truncated),
            }
        )

    if len(request.attached_files) > MAX_ATTACHED_FILES:
        errors.append(
            f"Only the first {MAX_ATTACHED_FILES} attached files were included."
        )

    return normalized, errors









############################## Stream helpers ##############################
def _new_thread_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"coding-run-{timestamp}-{uuid4().hex[:8]}"


def _generation_result_count(
    items: list[dict[str, Any]],
    generation: int,
) -> int:
    """Count worker results for the active generation without hiding legacy data."""
    matching = 0

    for item in items:
        try:
            item_generation = int(item.get("generation", -1))
        except (TypeError, ValueError):
            continue

        if item_generation == generation:
            matching += 1

    return matching if matching else len(items)


def _public_result(state: dict[str, Any], thread_id: str) -> CodingAgentRunResult:
    selected_skill = state.get("selected_skill")
    selected_skills = list(state.get("selected_skills") or [])

    if selected_skill and not selected_skills:
        selected_skills = [str(selected_skill)]

    context_worker_results = [
        item
        for item in state.get("context_worker_results", [])
        if isinstance(item, dict)
    ]

    context_generation = int(state.get("context_generation", 0) or 0)

    subtask_worker_results = [
        item
        for item in state.get("subtask_worker_results", [])
        if isinstance(item, dict)
    ]
    
    implementation_generation = int(state.get("implementation_generation", 0) or 0)
    runtime_settings = dict(state.get("runtime_settings") or {})
    configured_subtask_workers = int(
        runtime_settings.get("max_subtask_workers")
        or runtime_settings.get("max_context_workers")
        or _generation_result_count(subtask_worker_results, implementation_generation)
        or 0
    )

    return CodingAgentRunResult(
        thread_id=thread_id,
        status=str(state.get("status", "unknown")),
        report=state.get("report"),
        selected_skill=selected_skill,
        selected_skills=selected_skills,
        task_mode=state.get("task_mode"),

        # Legacy aliases remain populated while clients migrate.
        subtasks=list(state.get("subtasks") or []),
        context_worker_count=_generation_result_count(
            context_worker_results,
            context_generation,
        ),

        # Divide-and-conquer execution state.
        implementation_units=list(state.get("implementation_units") or []),
        completion_ledger=dict(state.get("completion_ledger") or {}),
        implementation_generation=implementation_generation,
        implementation_iteration=int(state.get("implementation_iteration", 0) or 0),
        max_implementation_iterations=int(
            state.get("max_implementation_iterations", state.get("max_iterations", 0)) or 0
        ),
        subtask_worker_count=configured_subtask_workers,
        subtask_worker_results=subtask_worker_results,
        runtime_settings=runtime_settings,

        route_confidence=state.get("route_confidence"),
        route_reason=state.get("route_reason"),
        plan=list(state.get("plan") or []),
        files_inspected=list(state.get("files_inspected") or []),
        patch_summary=state.get("patch_summary"),
        file_changes=list(state.get("file_changes") or []),
        diffs=list(state.get("diffs") or []),
        validation_commands=list(state.get("validation_commands") or []),
        validation_results=list(state.get("validation_results") or []),
        memory_enabled=bool(state.get("memory_enabled", False)),
        memory_namespace=state.get("memory_namespace"),
        long_term_memories=list(state.get("long_term_memories") or []),
        memory_errors=list(state.get("memory_errors") or []),
        errors=list(state.get("errors") or []),
        approval_required=bool(state.get("approval_required", False)),
        approval_status=state.get("approval_status", "not_required"),
        blocking_validation_failed=bool(state.get("blocking_validation_failed", False)),
        advisory_validation_failed=bool(state.get("advisory_validation_failed", False)),
        applied_files=list(state.get("applied_files") or []),
        raw=state,
    )


def _send_threadsafe(
    *,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any] | None],
    event: CodingAgentServerEvent,
) -> None:
    asyncio.run_coroutine_threadsafe(
        queue.put(jsonable_encoder(event.model_dump())),
        loop,
    )


def _stream_coding_agent_worker(
    *,
    request: CodingAgentRunRequest,
    run_id: str,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any] | None],
) -> None:
    
    thread_id = request.thread_id or _new_thread_id()

    # Prefer divide-and-conquer names, while accepting the pre-migration aliases.
    resolved_subtask_worker_count = (
        request.subtask_worker_count
        or request.subagent_count
        or config_settings.coding_subagent_count
    )
    resolved_max_implementation_iterations = (
        request.max_implementation_iterations
        or request.max_iterations
        or 3
    )

    try:
        cfg = replace(
            default_coding_settings,
            # The settings dataclass still exposes this as max_context_workers;
            # semantically these are now implementation-unit subtask workers.
            max_context_workers=resolved_subtask_worker_count,
            route_max_tokens=(
                request.route_max_tokens or config_settings.coding_route_max_tokens
            ),
            planner_max_tokens=(
                request.planner_max_tokens or config_settings.coding_planner_max_tokens
            ),
            repo_navigation_max_tokens=(
                request.repo_navigation_max_tokens
                or config_settings.coding_repo_navigation_max_tokens
            ),
            simple_patch_max_tokens=(
                request.simple_patch_max_tokens
                or config_settings.coding_simple_patch_max_tokens
            ),
            patch_max_tokens=(
                request.patch_max_tokens or config_settings.coding_patch_max_tokens
            ),
            progress_max_tokens=(
                request.progress_max_tokens or config_settings.coding_progress_max_tokens
            ),
        )

        if request.memory_enabled is not None:
            cfg = replace(cfg, memory_enabled=request.memory_enabled) # set up memory


        # Resolve roots
        repo_root_path = Path(request.repo_root).expanduser().resolve()
        repo_root = str(repo_root_path)

        workspace_root = (
            str(Path(request.workspace_root).expanduser().resolve())
            if request.workspace_root
            else None
        )


        original_repo_root_path = Path(request.repo_root).expanduser().resolve()
        original_workspace_root_path = (
            Path(request.workspace_root).expanduser().resolve()
            if request.workspace_root
            else original_repo_root_path
        )

        # initialize sandbox
        sandbox = create_coding_sandbox(
            repo_root=original_repo_root_path,
            workspace_root=original_workspace_root_path,
            run_id=run_id,
        )

        repo_root = str(sandbox.repo_root)
        workspace_root = str(sandbox.workspace_root)

        _send_threadsafe(
            loop=loop,
            queue=queue,
            event=CodingAgentServerEvent(
                type="run.started",
                run_id=run_id,
                thread_id=thread_id,
                payload={
                    "repo_root": repo_root,
                    "workspace_root": workspace_root,
                    "allow_write": request.allow_write,
                    "subtask_worker_count": cfg.max_context_workers,
                    # Compatibility alias for older frontends.
                    "subagent_count": cfg.max_context_workers,
                    "max_implementation_iterations": resolved_max_implementation_iterations,
                    "token_budgets": {
                        "route": cfg.route_max_tokens,
                        "planner": cfg.planner_max_tokens,
                        "repo_navigation": cfg.repo_navigation_max_tokens,
                        "simple_patch": cfg.simple_patch_max_tokens,
                        "patch": cfg.patch_max_tokens,
                        "progress": cfg.progress_max_tokens,
                    },
                    "runtime_settings": {
                        "max_subtask_workers": cfg.max_context_workers,
                        "max_context_workers": cfg.max_context_workers,
                        "max_implementation_iterations": resolved_max_implementation_iterations,
                    },
                },
            ),
        )

        try:
            attached_files, attachment_errors = _normalize_attached_files(
                request=request,
                repo_root=repo_root_path,
            )
        except Exception as exc:
            attached_files = []
            attachment_errors = [f"Attachment normalization failed: {exc}"]

        initial_state: dict[str, Any] = {
            "user_request": request.request,
            "repo_root": repo_root,
            "workspace_root": workspace_root,
            "original_repo_root": str(sandbox.original_repo_root),
            "sandbox_root": str(sandbox.sandbox_root),
            "sandbox_enabled": True,
            "allow_write": request.allow_write,
            "runtime_settings": {
                "max_subtask_workers": cfg.max_context_workers,
                "max_context_workers": cfg.max_context_workers,
                "max_implementation_iterations": resolved_max_implementation_iterations,
                "route_max_tokens": cfg.route_max_tokens,
                "planner_max_tokens": cfg.planner_max_tokens,
                "repo_navigation_max_tokens": cfg.repo_navigation_max_tokens,
                "simple_patch_max_tokens": cfg.simple_patch_max_tokens,
                "patch_max_tokens": cfg.patch_max_tokens,
                "progress_max_tokens": cfg.progress_max_tokens,
            },
            "attached_files": attached_files,
            "attached_files_used": [],
            "attachment_errors": attachment_errors,
            "errors": [*attachment_errors],
            "memory_errors": [],
            # Divide-and-conquer implementation lifecycle.
            "implementation_units": [],
            "active_implementation_unit": {},
            "implementation_generation": 0,
            "implementation_iteration": 0,
            "max_implementation_iterations": resolved_max_implementation_iterations,
            "subtask_worker_results": [],
            "completion_ledger": {},

            # Legacy loop/context fields remain initialized so older graph nodes and
            # persisted threads can still resume during the migration.
            "iteration": 0,
            "max_iterations": resolved_max_implementation_iterations,
            "continue_loop": False,
            "remaining_tasks": [],
            "loop_notes": [],
            "task_mode": "standard",
            "subtasks": [],
            "active_subtask": {},
            "context_generation": 0,
            "context_worker_results": [],
            "requested_context": [],
            "patch_attempts": 0,
        }

        final_state = dict(initial_state)

        runtime_context = CodingAgentRuntimeContext(
            user_id=request.memory_user_id or cfg.memory_user_id,
            memory_namespace=request.memory_namespace or cfg.memory_namespace,
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        with coding_agent_persistence(cfg, setup=request.setup_memory) as persistence:
            graph = build_coding_agent_graph(
                checkpointer=persistence.checkpointer,
                store=persistence.store,
            )

            for update in graph.stream(
                initial_state,
                config=config,
                context=runtime_context,
                stream_mode="updates",
            ):
                if not isinstance(update, dict):
                    continue

                for node_name, node_delta in update.items():
                    if isinstance(node_delta, dict):
                        final_state.update(node_delta)
                        payload = node_delta
                    else:
                        payload = {"value": node_delta}

                    _send_threadsafe(
                        loop=loop,
                        queue=queue,
                        event=CodingAgentServerEvent(
                            type="node.completed",
                            run_id=run_id,
                            thread_id=thread_id,
                            node=str(node_name),
                            payload=payload,
                        ),
                    )

        
        final_state["thread_id"] = thread_id

        changed_paths = _changed_paths_from_state(final_state)
        blocking_validation_failed = validation_failed_results(
            final_state.get("validation_results", [])
        )

        approval_required = bool(request.allow_write and changed_paths)

        final_state["blocking_validation_failed"] = blocking_validation_failed
        final_state["approval_required"] = approval_required
        final_state["approval_status"] = "pending" if approval_required else "not_required"
        final_state["applied_files"] = []

        result = _public_result(final_state, thread_id)

        _send_threadsafe(
            loop=loop,
            queue=queue,
            event=CodingAgentServerEvent(
                type="run.completed",
                run_id=run_id,
                thread_id=thread_id,
                payload=result.model_dump(),
            ),
        )

        if approval_required:
            _send_threadsafe(
                loop=loop,
                queue=queue,
                event=CodingAgentServerEvent(
                    type="run.approval_required",
                    run_id=run_id,
                    thread_id=thread_id,
                    payload={
                        "thread_id": thread_id,
                        "changed_paths": changed_paths,
                        "blocking_validation_failed": blocking_validation_failed,
                        "advisory_validation_failed": bool(
                            final_state.get("advisory_validation_failed", False)
                        ),
                    },
                ),
            )

            return PendingCodingAgentRun(
                run_id=run_id,
                thread_id=thread_id,
                sandbox=sandbox,
                final_state=final_state,
                changed_paths=changed_paths,
            )

        cleanup_coding_sandbox(sandbox, keep=False)
        return None

    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)





async def _run_and_forward_events(
    *,
    websocket: WebSocket,
    request: CodingAgentRunRequest,
) -> PendingCodingAgentRun | None:
    
    run_id = uuid4().hex
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    worker_task = asyncio.create_task(
        asyncio.to_thread(
            _stream_coding_agent_worker,
            request=request,
            run_id=run_id,
            loop=loop,
            queue=queue,
        )
    )

    try:
        while True:
            event = await queue.get()

            if event is None:
                break

            await websocket.send_json(event)

    finally:
        pending_run = await worker_task
    
    return pending_run





@router.get("/token")
async def websocket_token() -> dict:
    token = generate_websocket_token()
    return {"token": token}




@router.websocket("/ws")
async def coding_agent_ws(websocket: WebSocket) -> None:
   
    if not await authorize_websocket(websocket):
        return

    await websocket.accept()

    await websocket.send_json(
        CodingAgentServerEvent(
            type="session.ready",
            payload={
                "message": "Coding agent WebSocket is ready.",
                "protocol_version": "0.2.0",
            },
        ).model_dump()
    )

    pending_runs: dict[str, PendingCodingAgentRun] = {}

    try:
        while True:
            raw_message = await websocket.receive_json()

            try:
                message = CodingAgentClientMessage.model_validate(raw_message)
            except ValidationError as exc:
                await websocket.send_json(
                    CodingAgentServerEvent(
                        type="run.failed",
                        payload={
                            "error": "Invalid WebSocket message.",
                            "details": exc.errors(),
                        },
                    ).model_dump()
                )
                continue

            if message.type == "ping":
                await websocket.send_json(
                    CodingAgentServerEvent(type="pong").model_dump()
                )
                continue



            if message.type == "run.request":
                try:
                    run_request = CodingAgentRunRequest.model_validate(message.payload)
                except ValidationError as exc:
                    await websocket.send_json(
                        CodingAgentServerEvent(
                            type="run.failed",
                            payload={
                                "error": "Invalid coding-agent run request.",
                                "details": exc.errors(),
                            },
                        ).model_dump()
                    )
                    continue

                pending_run = await _run_and_forward_events(
                    websocket=websocket,
                    request=run_request,
                )

                if pending_run:
                    pending_runs[pending_run.thread_id] = pending_run

                continue
            

            if message.type == "run.apply.request":
                payload = message.payload or {}
                thread_id = str(payload.get("thread_id", "")).strip()
                requested_paths = payload.get("paths")

                pending_run = pending_runs.get(thread_id)

                if not pending_run:
                    await websocket.send_json(
                        CodingAgentServerEvent(
                            type="run.failed",
                            thread_id=thread_id or None,
                            payload={"error": "No pending approval run found for this thread."},
                        ).model_dump()
                    )
                    continue

                paths = (
                    [str(path) for path in requested_paths]
                    if isinstance(requested_paths, list) and requested_paths
                    else pending_run.changed_paths
                )

                allowed = set(pending_run.changed_paths)
                invalid = [path for path in paths if path not in allowed]

                if invalid:
                    await websocket.send_json(
                        CodingAgentServerEvent(
                            type="run.failed",
                            run_id=pending_run.run_id,
                            thread_id=thread_id,
                            payload={
                                "error": "Approval request included paths that are not pending.",
                                "details": invalid,
                            },
                        ).model_dump()
                    )
                    continue

                try:
                    applied_paths = apply_sandbox_files_to_repo(
                        sandbox=pending_run.sandbox,
                        changed_paths=paths,
                    )

                    remaining_paths = [
                        path for path in pending_run.changed_paths
                        if path not in set(applied_paths)
                    ]

                    pending_run.final_state["applied_files"] = [
                        *pending_run.final_state.get("applied_files", []),
                        *applied_paths,
                    ]

                    if remaining_paths:
                        pending_run.changed_paths = remaining_paths
                        approval_status = "pending"
                    else:
                        approval_status = "applied"
                        pending_run.final_state["approval_status"] = "applied"
                        pending_run.final_state["status"] = "applied"
                        cleanup_coding_sandbox(pending_run.sandbox, keep=False)
                        pending_runs.pop(thread_id, None)

                    await websocket.send_json(
                        CodingAgentServerEvent(
                            type="run.applied",
                            run_id=pending_run.run_id,
                            thread_id=thread_id,
                            payload={
                                "thread_id": thread_id,
                                "applied_files": applied_paths,
                                "remaining_paths": remaining_paths,
                                "approval_status": approval_status,
                            },
                        ).model_dump()
                    )

                except Exception as exc:
                    await websocket.send_json(
                        CodingAgentServerEvent(
                            type="run.failed",
                            run_id=pending_run.run_id,
                            thread_id=thread_id,
                            payload={
                                "error": f"Failed to apply approved files: {exc}",
                                "error_type": type(exc).__name__,
                            },
                        ).model_dump()
                    )

                continue



            if message.type == "run.reject.request":
                payload = message.payload or {}
                thread_id = str(payload.get("thread_id", "")).strip()
                pending_run = pending_runs.pop(thread_id, None)

                if pending_run:
                    pending_run.final_state["approval_status"] = "rejected"
                    pending_run.final_state["status"] = "rejected"
                    cleanup_coding_sandbox(pending_run.sandbox, keep=False)

                await websocket.send_json(
                    CodingAgentServerEvent(
                        type="run.rejected",
                        thread_id=thread_id or None,
                        payload={
                            "thread_id": thread_id,
                            "approval_status": "rejected",
                        },
                    ).model_dump()
                )

                continue


    except WebSocketDisconnect:
        return
    



    