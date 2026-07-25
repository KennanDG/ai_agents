from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

from groq import Groq

from ai_agents.agents.voice.prompts import VOICE_INTAKE_SYSTEM_PROMPT
from ai_agents.agents.voice.schemas import VoiceIntakeDecision
from ai_agents.agents.voice.state import VoiceAgentState
from ai_agents.config.settings import settings

from ai_agents.agents.voice.utils.constants import (
    MAX_REPO_FILES,
    MAX_TREE_FILES,
    MAX_SEARCH_MATCHES,
    MAX_FILE_BYTES,
    MAX_EXPLICIT_FILE_CHARS,
    MAX_ATTACHMENT_CONTENT_CHARS,
    MAX_TOTAL_ATTACHMENT_CONTENT_CHARS,
    MAX_CONTEXT_JSON_CHARS,
    MAX_LLM_TREE_PATHS,
    MAX_LLM_EXPLICIT_FILE_CHARS,
    MAX_LLM_SEARCH_EXCERPT_CHARS,
    MAX_LLM_ATTACHMENT_EXCERPT_CHARS
)


logger = logging.getLogger(__name__)


IGNORED_DIRS = {
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

TEXT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".css",
    ".csv",
    ".h",
    ".hh",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

STOP_WORDS = {
    "about",
    "agent",
    "attached",
    "coding",
    "could",
    "files",
    "from",
    "have",
    "into",
    "please",
    "should",
    "that",
    "their",
    "this",
    "update",
    "voice",
    "with",
    "would",
}




def _client() -> Groq:
    api_key = settings.resolved_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for the voice agent.")
    return Groq(api_key=api_key)


def _safe_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    safe: list[dict[str, str]] = []

    for item in history[-12:]:
        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant", "system"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        safe.append({"role": role, "content": content[:4_000]})

    return safe


def _count_prior_clarifications(history: list[dict[str, str]]) -> int:
    """Count prior voice-agent replies in the isolated voice conversation."""
    return sum(1 for item in history if item.get("role") == "assistant")


def _strip_voice_prefix(text: str) -> str:
    return text.strip().removeprefix("🎙️").strip()


def _resolve_repo_root(repo_root: str | None) -> Path | None:
    if not repo_root:
        return None

    try:
        root = Path(repo_root).expanduser().resolve()
    except OSError:
        return None

    return root if root.exists() and root.is_dir() else None


def _iter_repository_files(root: Path) -> Iterable[Path]:
    yielded = 0

    for current_dir, dir_names, file_names in os.walk(root):
        dir_names[:] = [
            name
            for name in sorted(dir_names)
            if name not in IGNORED_DIRS and not name.endswith(".egg-info")
        ]

        for file_name in sorted(file_names):
            path = Path(current_dir) / file_name
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue

            yield path
            yielded += 1
            if yielded >= MAX_REPO_FILES:
                return


def _safe_repo_path(root: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None

    try:
        target = (root / relative_path).resolve()
    except OSError:
        return None

    if target != root and root not in target.parents:
        return None

    if not target.exists() or not target.is_file():
        return None

    return target


def _read_text_excerpt(path: Path, *, max_chars: int) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        raw = path.read_bytes()
    except OSError:
        return None

    if b"\0" in raw[:4_096]:
        return None

    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[truncated]"


def _keywords(text: str) -> list[str]:
    candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", text.lower())
    ordered: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        value = candidate.strip("._-")
        if len(value) < 3 or value in STOP_WORDS or value in seen:
            continue
        seen.add(value)
        ordered.append(value)

    return ordered[:18]


def _matching_excerpt(text: str, terms: list[str], *, radius: int = 260) -> str:
    lowered = text.lower()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    if not positions:
        return text[: radius * 2].strip()

    center = min(positions)
    start = max(0, center - radius)
    end = min(len(text), center + radius)
    excerpt = text[start:end].strip()
    return ("..." if start else "") + excerpt + ("..." if end < len(text) else "")


def _select_skills(combined_request: str, attachments: list[dict[str, Any]]) -> list[str]:
    lowered = combined_request.lower()
    skills = [
        "requirement_synthesis",
        "repository_reconnaissance",
        "implementation_planning",
        "validation_planning",
    ]

    if attachments:
        skills.append("attachment_analysis")
    if any(token in lowered for token in ("react", "tsx", "frontend", "textarea", "button", "ui")):
        skills.append("frontend_flow_tracing")
    if any(token in lowered for token in ("api", "fastapi", "backend", "endpoint", "websocket")):
        skills.append("backend_api_tracing")
    if any(token in lowered for token in ("schema", "state", "payload", "request", "response")):
        skills.append("data_contract_analysis")
    if any(token in lowered for token in ("langgraph", "graph", "node", "tool", "skill")):
        skills.append("agent_graph_design")

    return list(dict.fromkeys(skills))


def _attachment_context(
    attachments: list[dict[str, Any]],
    *,
    root: Path | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    remaining_content = MAX_TOTAL_ATTACHMENT_CONTENT_CHARS

    for attached in attachments[:5]:
        name = str(attached.get("name") or "attachment")
        source = str(attached.get("source") or "upload")
        path = attached.get("path") if isinstance(attached.get("path"), str) else None
        content = attached.get("content") if isinstance(attached.get("content"), str) else ""

        if source == "repo" and root and path:
            repo_path = _safe_repo_path(root, path)
            repo_excerpt = (
                _read_text_excerpt(repo_path, max_chars=MAX_ATTACHMENT_CONTENT_CHARS)
                if repo_path
                else None
            )
            if repo_excerpt:
                content = repo_excerpt

        max_chars = min(MAX_ATTACHMENT_CONTENT_CHARS, max(0, remaining_content))
        excerpt = content[:max_chars] if content and max_chars > 0 else None
        remaining_content -= len(excerpt or "")

        results.append(
            {
                "name": name,
                "source": source,
                "path": path,
                "mime_type": attached.get("mime_type"),
                "size": attached.get("size"),
                "has_image_data": bool(attached.get("has_image_data")),
                "content_excerpt": excerpt,
                "content_truncated": bool(attached.get("content_truncated"))
                or bool(content and excerpt is not None and len(excerpt) < len(content)),
            }
        )

    return results


def _compact_repo_context(repo_context: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded, structured context object for the intake model.

    Do not pass the full context JSON as a nested string. Double-encoding makes it
    easy for a model to echo thousands of escaped characters and hit its output cap.
    """
    explicit_files = [
        {
            "path": item.get("path"),
            "content_excerpt": str(item.get("content_excerpt") or "")[
                :MAX_LLM_EXPLICIT_FILE_CHARS
            ],
        }
        for item in repo_context.get("explicit_files", [])[:5]
        if isinstance(item, dict) and item.get("path")
    ]

    search_matches = [
        {
            "path": item.get("path"),
            "score": item.get("score"),
            "matched_terms": item.get("matched_terms", [])[:8],
            "content_excerpt": str(item.get("content_excerpt") or "")[
                :MAX_LLM_SEARCH_EXCERPT_CHARS
            ],
        }
        for item in repo_context.get("search_matches", [])[:MAX_SEARCH_MATCHES]
        if isinstance(item, dict) and item.get("path")
    ]

    attachment_context = [
        {
            "name": item.get("name"),
            "source": item.get("source"),
            "path": item.get("path"),
            "mime_type": item.get("mime_type"),
            "size": item.get("size"),
            "has_image_data": bool(item.get("has_image_data")),
            "content_excerpt": str(item.get("content_excerpt") or "")[
                :MAX_LLM_ATTACHMENT_EXCERPT_CHARS
            ],
            "content_truncated": bool(item.get("content_truncated")),
        }
        for item in repo_context.get("attachment_context", [])[:5]
        if isinstance(item, dict)
    ]

    relevant_paths = list(
        dict.fromkeys(
            [item["path"] for item in explicit_files]
            + [item["path"] for item in search_matches]
        )
    )

    return {
        "repo_root": repo_context.get("repo_root"),
        "active_path": repo_context.get("active_path"),
        "relevant_paths": relevant_paths[:25],
        "tree_sample": repo_context.get("repository_tree", [])[:MAX_LLM_TREE_PATHS],
        "explicit_files": explicit_files,
        "search_matches": search_matches,
        "attachment_context": attachment_context,
    }


def _is_json_generation_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "json_validate_failed" in text or "failed to generate json" in text


def _request_intake_decision(
    *,
    messages: list[dict[str, str]],
    temperature: float,
) -> VoiceIntakeDecision:
    
    completion = _client().chat.completions.create(
        model=settings.voice_chat_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max(512, settings.voice_chat_max_tokens),
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content or "{}"
    return VoiceIntakeDecision.model_validate_json(content)


def gather_context_node(state: VoiceAgentState) -> VoiceAgentState:
    transcript = state.get("transcript", "").strip()
    prompt_text = state.get("prompt_text", "").strip()
    attachments = state.get("attached_files", [])
    root = _resolve_repo_root(state.get("repo_root"))
    combined_request = "\n".join(part for part in (prompt_text, transcript) if part)
    search_hints = [combined_request, state.get("active_path") or ""]
    for attachment in attachments:
        search_hints.append(str(attachment.get("name") or ""))
        search_hints.append(str(attachment.get("path") or ""))
    search_terms = _keywords("\n".join(search_hints))

    tools_used = ["inspect_attached_files"]
    repo_context: dict[str, Any] = {
        "repo_root": str(root) if root else state.get("repo_root"),
        "active_path": state.get("active_path"),
        "attachment_context": _attachment_context(attachments, root=root),
        "repository_tree": [],
        "explicit_files": [],
        "search_matches": [],
    }
    errors = list(state.get("errors", []))

    if root:
        repository_files = list(_iter_repository_files(root))
        tools_used.append("list_repository_tree")
        repo_context["repository_tree"] = [
            path.relative_to(root).as_posix() for path in repository_files[:MAX_TREE_FILES]
        ]

        explicit_paths: list[str] = []
        active_path = state.get("active_path")
        if active_path:
            explicit_paths.append(active_path)

        for attachment in attachments:
            if attachment.get("source") == "repo" and isinstance(attachment.get("path"), str):
                explicit_paths.append(attachment["path"])

        for relative_path in dict.fromkeys(explicit_paths):
            path = _safe_repo_path(root, relative_path)
            if not path:
                continue
            excerpt = _read_text_excerpt(path, max_chars=MAX_EXPLICIT_FILE_CHARS)
            if not excerpt:
                continue
            repo_context["explicit_files"].append(
                {"path": path.relative_to(root).as_posix(), "content_excerpt": excerpt}
            )

        if repo_context["explicit_files"]:
            tools_used.append("read_repository_file")

        if search_terms:
            matches: list[dict[str, Any]] = []
            for path in repository_files:
                relative_path = path.relative_to(root).as_posix()
                path_lower = relative_path.lower()
                path_terms = [term for term in search_terms if term in path_lower]
                text = _read_text_excerpt(path, max_chars=24_000)
                if not text:
                    continue

                text_lower = text.lower()
                content_terms = [term for term in search_terms if term in text_lower]
                matched_terms = list(dict.fromkeys([*path_terms, *content_terms]))
                if not matched_terms:
                    continue

                score = len(path_terms) * 3 + len(content_terms)
                matches.append(
                    {
                        "path": relative_path,
                        "score": score,
                        "matched_terms": matched_terms[:8],
                        "content_excerpt": _matching_excerpt(text, matched_terms),
                    }
                )

            matches.sort(key=lambda item: (-int(item["score"]), str(item["path"])))
            repo_context["search_matches"] = matches[:MAX_SEARCH_MATCHES]
            if matches:
                tools_used.append("search_repository")
    else:
        errors.append("Voice context gathering could not resolve the repository root.")

    recommended_skills = _select_skills(combined_request, attachments)
    compact_context = _compact_repo_context(repo_context)
    context_summary = json.dumps(compact_context, ensure_ascii=False, default=str)
    if len(context_summary) > MAX_CONTEXT_JSON_CHARS:
        context_summary = context_summary[:MAX_CONTEXT_JSON_CHARS] + "...[context truncated]"

    return {
        "repo_context": repo_context,
        "context_summary": context_summary,
        "recommended_skills": recommended_skills,
        "tools_used": list(dict.fromkeys(tools_used)),
        "errors": errors,
    }


def _default_plan(state: VoiceAgentState) -> list[str]:
    repo_context = state.get("repo_context", {})
    explicit_paths = [
        str(item.get("path"))
        for item in repo_context.get("explicit_files", [])
        if isinstance(item, dict) and item.get("path")
    ]
    search_paths = [
        str(item.get("path"))
        for item in repo_context.get("search_matches", [])[:8]
        if isinstance(item, dict) and item.get("path")
    ]
    relevant_paths = list(dict.fromkeys([*explicit_paths, *search_paths]))

    plan = [
        "Review the resolved voice conversation and confirm the exact requested outcome.",
    ]

    if relevant_paths:
        plan.append(
            "Inspect the most relevant repository files before editing: "
            + ", ".join(relevant_paths[:6])
            + "."
        )
    else:
        plan.append(
            "Search the repository for the files and existing patterns that implement the requested behavior."
        )

    attachments = state.get("attached_files", [])
    if attachments:
        names = ", ".join(str(item.get("name") or "attachment") for item in attachments[:5])
        plan.append(f"Inspect the attached context passed separately to the coding agent: {names}.")

    plan.extend(
        [
            "Implement the smallest safe change using the repository's existing architecture and style.",
            "Run focused validation for the changed files and report any failures or remaining assumptions.",
        ]
    )

    if state.get("allow_write"):
        plan.append("Prepare reviewable changes through the normal human approval flow.")
    else:
        plan.append("Remain read-only and report the exact proposed changes.")

    return plan


def _fallback_coding_request(
    *,
    state: VoiceAgentState,
    history: list[dict[str, str]],
    transcript: str,
) -> str:
    user_turns = [
        _strip_voice_prefix(item["content"])
        for item in history
        if item.get("role") == "user" and isinstance(item.get("content"), str)
    ]
    prompt_text = state.get("prompt_text", "").strip()
    if prompt_text:
        user_turns.append(f"Typed draft: {prompt_text}")
    user_turns.append(_strip_voice_prefix(transcript))
    conversation = "\n".join(f"- {turn}" for turn in user_turns if turn)

    attachments = state.get("attached_files", [])
    attachment_lines = [
        f"- {item.get('name')} ({item.get('source')}, path={item.get('path') or 'n/a'})"
        for item in attachments
    ]
    repo_context = state.get("repo_context", {})
    
    target_files = list(
        dict.fromkeys(
            [
                str(item.get("path"))
                for item in repo_context.get("explicit_files", [])
                if isinstance(item, dict) and item.get("path")
            ]
            + [
                str(item.get("path"))
                for item in repo_context.get("search_matches", [])[:8]
                if isinstance(item, dict) and item.get("path")
            ]
        )
    )
    target_text = "\n".join(f"- {path}" for path in target_files) or "- Verify the correct files from the repository tree and search matches."
    plan = _default_plan(state)
    write_mode = (
        "Prepare the patch and use the normal human approval flow before repository writes."
        if state.get("allow_write")
        else "Remain read-only and report the exact proposed changes."
    )

    return (
        "Objective\n"
        "Implement the resolved request from this voice conversation.\n\n"
        "Resolved requirements\n"
        f"{conversation}\n\n"
        "Repository and attachment context\n"
        + ("\n".join(attachment_lines) if attachment_lines else "- No files were attached.")
        + "\n\nTarget files or areas\n"
        + target_text
        + "\n\nDetailed plan of action\n"
        + "\n".join(f"{index}. {step}" for index, step in enumerate(plan, start=1))
        + "\n\nValidation and acceptance criteria\n"
        "- Confirm the resolved user requirements are implemented without unrelated behavior changes.\n"
        "- Run the smallest relevant tests, type checks, lint checks, or build commands available in the repository.\n"
        "- Report validation failures and any assumptions that still need verification.\n\n"
        "Constraints and assumptions\n"
        "- Inspect repository evidence before making assumptions.\n"
        "- Preserve existing attachment limits and approval behavior.\n"
        f"- {write_mode}"
    )


def _ensure_detailed_coding_request(
    *,
    decision: VoiceIntakeDecision,
    state: VoiceAgentState,
) -> str:
    request = (decision.coding_request or "").strip()
    plan = decision.plan or _default_plan(state)
    attachment_names = [
        str(item.get("name") or "attachment") for item in state.get("attached_files", [])
    ]

    required_markers = (
        "Objective",
        "Detailed plan",
        "Validation",
        "Constraints",
    )
    if all(marker.lower() in request.lower() for marker in required_markers):
        return request

    repo_context = state.get("repo_context", {})
    target_files = decision.target_files or list(
        dict.fromkeys(
            [
                str(item.get("path"))
                for item in repo_context.get("explicit_files", [])
                if isinstance(item, dict) and item.get("path")
            ]
            + [
                str(item.get("path"))
                for item in repo_context.get("search_matches", [])[:8]
                if isinstance(item, dict) and item.get("path")
            ]
        )
    )

    return (
        "Objective\n"
        f"{request or 'Implement the resolved voice request using the gathered repository context.'}\n\n"
        "Repository and attachment context\n"
        + (
            "- Attached files passed separately: " + ", ".join(attachment_names)
            if attachment_names
            else "- No files were attached."
        )
        + "\n- Context tools used: "
        + ", ".join(decision.tools_used or state.get("tools_used", []))
        + "\n\nTarget files or areas\n"
        + ("\n".join(f"- {path}" for path in target_files) or "- Verify targets from repository context before editing.")
        + "\n\nDetailed plan of action\n"
        + "\n".join(f"{index}. {step}" for index, step in enumerate(plan, start=1))
        + "\n\nValidation and acceptance criteria\n"
        "- Confirm the resolved requirements are implemented without unrelated behavior changes.\n"
        "- Run focused validation for the files and technologies actually changed.\n"
        "- Report failures and unresolved assumptions clearly.\n\n"
        "Constraints and assumptions\n"
        "- Use gathered context as evidence and verify uncertain targets.\n"
        "- Preserve existing approval and attachment-limit behavior."
    )


def intake_node(state: VoiceAgentState) -> VoiceAgentState:
    transcript = state.get("transcript", "").strip()
    prompt_text = state.get("prompt_text", "").strip()
    history = _safe_history(state.get("history", []))

    if not transcript and not prompt_text:
        return {
            "status": "error",
            "reply_text": "I could not hear anything clearly. Please try again.",
            "coding_request": None,
            "errors": [*state.get("errors", []), "Empty transcript and typed draft."],
        }

    repository_context = _compact_repo_context(state.get("repo_context", {}))
    context: dict[str, object] = {
        "repo_root": state.get("repo_root"),
        "workspace_root": state.get("workspace_root"),
        "active_path": state.get("active_path"),
        "allow_write": state.get("allow_write", False),
        "attached_files": [
            {
                "name": item.get("name"),
                "source": item.get("source"),
                "path": item.get("path"),
                "mime_type": item.get("mime_type"),
                "size": item.get("size"),
                "has_image_data": item.get("has_image_data"),
            }
            for item in state.get("attached_files", [])
        ],
        "recommended_skills": state.get("recommended_skills", []),
        "tools_used": state.get("tools_used", []),
        "repository_context": repository_context,
    }

    clarification_count = _count_prior_clarifications(history)
    max_clarifications = max(1, settings.voice_max_clarifications)
    clarification_limit_reached = clarification_count >= max_clarifications

    user_content = (
        f"Latest user transcript:\n{transcript or '[none]'}\n\n"
        f"Current typed draft in the text area:\n{prompt_text or '[none]'}\n\n"
        f"Current UI, attachment, and repository context:\n{json.dumps(context, indent=2, default=str)}\n\n"
        f"Clarifying questions already asked: {clarification_count}\n"
        f"Maximum clarifying questions allowed: {max_clarifications}\n"
        f"Clarification limit reached: {clarification_limit_reached}\n\n"
        "Use the supplied skills and tool results. If ready, return a concise coding_request string, "
        "with implementation steps in plan and paths in target_files. Do not copy raw repository context. "
        "If the clarification limit is reached, return status=ready with the best repository-grounded plan."
    )

    messages = [
        {"role": "system", "content": VOICE_INTAKE_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_content},
    ]

    try:
        try:
            decision = _request_intake_decision(messages=messages, temperature=0.2)
        except Exception as first_exc:
            if not _is_json_generation_error(first_exc):
                raise

            logger.warning(
                "Voice intake JSON generation failed; retrying with a minimal response instruction: %s",
                first_exc,
            )

            retry_content = (
                "Return the smallest valid JSON object matching the system schema. "
                "coding_request must be a short plain string, never an object. "
                "Do not repeat repository context, trees, excerpts, or raw JSON.\n\n"
                f"Latest transcript: {transcript or '[none]'}\n"
                f"Typed draft: {prompt_text or '[none]'}\n"
                f"Relevant paths: {repository_context.get('relevant_paths', [])}\n"
                f"Clarification limit reached: {clarification_limit_reached}"
            )

            retry_messages = [
                {"role": "system", "content": VOICE_INTAKE_SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": retry_content},
            ]

            try:
                decision = _request_intake_decision(
                    messages=retry_messages,
                    temperature=0.0,
                )
            except Exception as retry_exc:
                raise RuntimeError(
                    f"Initial JSON generation failed: {first_exc}; retry failed: {retry_exc}"
                ) from retry_exc

        if clarification_limit_reached and (
            decision.status != "ready" or not decision.coding_request
        ):
            return {
                "status": "ready",
                "reply_text": "I have enough context. I am handing a detailed plan to the coding agent now.",
                "coding_request": _fallback_coding_request(
                    state=state,
                    history=history,
                    transcript=transcript,
                ),
                "collected_facts": decision.collected_facts,
            }

        if decision.status == "ready":
            return {
                "status": "ready",
                "reply_text": decision.reply_text,
                "coding_request": _ensure_detailed_coding_request(
                    decision=decision,
                    state=state,
                ),
                "collected_facts": decision.collected_facts,
            }

        return {
            "status": "clarifying",
            "reply_text": decision.reply_text,
            "coding_request": None,
            "collected_facts": decision.collected_facts,
        }

    except Exception as exc:
        fallback_request = _fallback_coding_request(
            state=state,
            history=history,
            transcript=transcript,
        )
        return {
            "status": "ready",
            "reply_text": "I gathered the available context and prepared a fallback plan for the coding agent.",
            "coding_request": fallback_request,
            "errors": [*state.get("errors", []), f"Voice intake model failed: {exc}"],
        }
