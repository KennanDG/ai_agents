from __future__ import annotations

import os
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from ai_agents.agents.coding.graph import build_coding_agent_graph
from ai_agents.agents.coding.memory import (
    CodingAgentRuntimeContext,
    coding_agent_persistence,
)
from ai_agents.agents.coding.settings import settings as default_coding_settings
from ai_agents.api.auth import authorize_websocket
from ai_agents.api.schemas import (
    CodingAgentClientMessage,
    CodingAgentRunRequest,
    CodingAgentRunResult,
    CodingAgentServerEvent,
    RepositoryFileResponse,
    RepositoryTreeEntry,
    RepositoryTreeResponse,
)


IGNORED_REPOSITORY_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

MAX_REPOSITORY_FILE_BYTES = 1_000_000

#TODO: Add c++, Rust, and java files
LANGUAGE_BY_EXTENSION = {
    ".css": "css",
    ".html": "html",
    ".js": "javascript",
    ".jsx": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".sql": "sql",
    ".ts": "typescript",
    ".tsx": "typescript",
    # ".txt": "plaintext",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
}



router = APIRouter(prefix="/coding-agent", tags=["coding-agent"])




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
def _new_thread_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"coding-run-{timestamp}-{uuid4().hex[:8]}"


def _public_result(state: dict[str, Any], thread_id: str) -> CodingAgentRunResult:
    return CodingAgentRunResult(
        thread_id=thread_id,
        status=str(state.get("status", "unknown")),
        report=state.get("report"),
        selected_skill=state.get("selected_skill"),
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

    cfg = default_coding_settings
    if request.memory_enabled is not None:
        cfg = replace(cfg, memory_enabled=request.memory_enabled)

    repo_root = str(Path(request.repo_root).expanduser().resolve())
    workspace_root = (
        str(Path(request.workspace_root).expanduser().resolve())
        if request.workspace_root
        else None
    )

    initial_state: dict[str, Any] = {
        "user_request": request.request,
        "repo_root": repo_root,
        "workspace_root": workspace_root,
        "allow_write": request.allow_write,
        "errors": [],
        "memory_errors": [],
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

    try:
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
                },
            ),
        )

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
                    payload: dict[str, Any]

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

    except Exception as exc:
        _send_threadsafe(
            loop=loop,
            queue=queue,
            event=CodingAgentServerEvent(
                type="run.failed",
                run_id=run_id,
                thread_id=thread_id,
                payload={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            ),
        )

    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


async def _run_and_forward_events(
    *,
    websocket: WebSocket,
    request: CodingAgentRunRequest,
) -> None:
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
        await worker_task


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
                "protocol_version": "0.1.0",
            },
        ).model_dump()
    )

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

                await _run_and_forward_events(
                    websocket=websocket,
                    request=run_request,
                )

    except WebSocketDisconnect:
        return
    


    