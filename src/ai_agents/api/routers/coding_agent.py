from __future__ import annotations

import os
import asyncio
import base64
import binascii
import mimetypes
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
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
from ai_agents.agents.coding.sandbox import (
    apply_sandbox_files_to_repo,
    cleanup_coding_sandbox,
    create_coding_sandbox,
)
from ai_agents.agents.coding.utils.validation import validation_failed_results



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
MAX_ATTACHED_FILES = 5
MAX_ATTACHMENT_CHARS = 50_000
MAX_TOTAL_ATTACHMENT_CHARS = 150_000
MAX_ATTACHED_IMAGE_BYTES = 5_000_000

GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
VISION_MODEL = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}

IMAGE_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>image/(?:png|jpeg|jpg|webp));base64,(?P<data>[A-Za-z0-9+/=\r\n]+)$",
    re.IGNORECASE,
)


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
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".java": "java",
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
    api_key = os.getenv("GROQ_API_KEY")
    vision_model_name = VISION_MODEL

    if not api_key or not vision_model_name:
        raise RuntimeError(
            f"image attachments require GROQ_API_URL and VISION_MODEL."
        )

    vision_model = ChatOpenAI(
        model=vision_model_name,
        api_key=api_key,
        base_url=GROQ_API_URL,
        max_retries=2,
    )

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

                    content = raw.decode("utf-8", errors="replace")

            except HTTPException as exc:
                errors.append(f"Skipped repo attachment {path}: {exc.detail}")
                continue
            except Exception as exc:
                errors.append(f"Skipped repo attachment {path}: {exc}")
                continue

        else:
            data_url = attached.data_url

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

        if not content.strip():
            errors.append(f"Skipped attachment {name}: empty content.")
            continue

        remaining = MAX_TOTAL_ATTACHMENT_CHARS - total_chars

        if remaining <= 0:
            errors.append("Skipped remaining attachments: total attachment context limit reached.")
            break

        content, truncated_by_file_limit = _truncate_text(
            content,
            min(MAX_ATTACHMENT_CHARS, remaining),
        )

        total_chars += len(content)

        normalized.append(
            {
                "name": name,
                "path": path,
                "source": source,
                "mime_type": mime_type,
                "size": attached.size,
                "content": content,
                "truncated": (
                    bool(attached.truncated)
                    or truncated_by_file_limit
                    or total_chars >= MAX_TOTAL_ATTACHMENT_CHARS
                ),
            }
        )

    if len(request.attached_files) > MAX_ATTACHED_FILES:
        errors.append(
            f"Only the first {MAX_ATTACHED_FILES} attached files were included."
        )

    return normalized, errors


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

    try:
        cfg = default_coding_settings

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
            "attached_files": attached_files,
            "attached_files_used": [],
            "attachment_errors": attachment_errors,
            "errors": [*attachment_errors],
            "memory_errors": [],
            "iteration": 0,
            "max_iterations": request.max_iterations or 3,
            "continue_loop": False,
            "remaining_tasks": [],
            "loop_notes": [],
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
        result = _public_result(final_state, thread_id)


        # Apply changes to sandbox if validated & allow write on
        changed_paths = [
            item.get("path", "")
            for item in final_state.get("file_changes", [])
            if item.get("path")
        ]

        validation_failed = validation_failed_results(final_state.get("validation_results", []))

        if request.allow_write and changed_paths and not validation_failed:
            try:
                applied_paths = apply_sandbox_files_to_repo(
                    sandbox=sandbox,
                    changed_paths=changed_paths,
                )
                final_state["applied_files"] = applied_paths
            except Exception as exc:
                final_state["errors"] = [
                    *final_state.get("errors", []),
                    f"Failed to apply sandbox changes to repository: {exc}",
                ]



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
        if "sandbox" in locals():
            cleanup_coding_sandbox(sandbox, keep=False)
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
    



    