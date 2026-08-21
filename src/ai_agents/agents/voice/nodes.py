from __future__ import annotations

import json
import logging
from typing import Any

from ai_agents.agents.voice.prompts import VOICE_INTAKE_SYSTEM_PROMPT
from ai_agents.agents.voice.schemas import VoiceIntakeDecision
from ai_agents.config.settings import settings
from ai_agents.agents.voice.state import VoiceAgentState
from ai_agents.agents.voice.utils.constants import (
    MAX_CONTEXT_JSON_CHARS,
    MAX_EXPLICIT_FILE_CHARS,
    MAX_TREE_FILES,
    MAX_SEARCH_MATCHES
)
from .utils.helpers import (
    _safe_history,
    _questions_from_history,
    _count_prior_clarifications,
    _used_clarification_topics,
    _next_novel_clarification,
    _decision_has_novel_question,
    _resolve_repo_root,
    _iter_repository_files,
    _safe_repo_path,
    _read_text_excerpt,
    _keywords,
    _matching_excerpt,
    _select_skills,
    _attachment_context,
    _compact_repo_context,
    _is_json_generation_error,
    _request_intake_decision,
    _fallback_coding_request,
    _ensure_detailed_coding_request,
)


logger = logging.getLogger(__name__)


def _is_retryable_intake_generation_error(exc: Exception) -> bool:
    """Return True for malformed structured output and unwanted model tool calls."""
    if _is_json_generation_error(exc):
        return True

    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "tool_use_failed",
            "tool choice is none",
            "model called a tool",
            "failed_generation",
        )
    )






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
                "caption_available": bool(item.get("image_caption")),
            }
            for item in state.get("attached_files", [])
        ],
        "recommended_skills": state.get("recommended_skills", []),
        "context_sources_used": state.get("tools_used", []),
        "repository_context": repository_context,
    }

    clarification_count = _count_prior_clarifications(history)
    max_clarifications = max(1, settings.voice_max_clarifications)
    clarification_limit_reached = clarification_count >= max_clarifications

    previous_questions = _questions_from_history(history)
    previous_topics = _used_clarification_topics(previous_questions)
    previous_questions_list = "\n".join(
        f"- {question}" for question in previous_questions[-8:]
    ) or "- None"
    previous_topics_list = ", ".join(sorted(previous_topics)) or "none"

    user_content = (
        f"Latest user transcript:\n{transcript or '[none]'}\n\n"
        f"Current typed draft in the text area:\n{prompt_text or '[none]'}\n\n"
        f"Current UI, attachment, and repository context:\n"
        f"{json.dumps(context, indent=2, default=str)}\n\n"
        f"Clarifying questions already asked: {clarification_count}\n"
        f"Maximum clarifying questions allowed: {max_clarifications}\n"
        f"Clarification limit reached: {clarification_limit_reached}\n\n"
        f"Previously asked questions (do not repeat or rephrase any of these—"
        f"ask something genuinely different):\n{previous_questions_list}\n\n"
        f"Previously used clarification topics: {previous_topics_list}\n"
        "Choose a different, unanswered clarification topic or return status=ready.\n\n"
        "All repository and attachment inspection has already been performed by the backend. "
        "You have no callable tools or functions in this step. Treat context_sources_used as audit metadata, "
        "and never attempt to call those names. Use only the supplied pre-gathered evidence. "
        "If ready, return a concise coding_request string, with implementation steps in plan and paths "
        "in target_files. Do not copy raw repository context. If the clarification limit is reached, "
        "return status=ready with the best repository-grounded plan."
    )

    messages = [
        {"role": "system", "content": VOICE_INTAKE_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_content},
    ]

    try:
        try:
            decision = _request_intake_decision(messages=messages, temperature=0.0)
        except Exception as first_exc:
            if not _is_retryable_intake_generation_error(first_exc):
                raise

            logger.warning(
                "Voice intake JSON generation failed; retrying with a minimal response instruction: %s",
                first_exc,
            )

            retry_content = (
                "Return the smallest valid JSON object matching the system schema. "
                "You have no callable tools or functions. Do not emit a tool call, function call, "
                "or tool-shaped response. coding_request must be a short plain string, never an object. "
                "Do not repeat repository context, trees, excerpts, or raw JSON.\n\n"
                f"Latest transcript: {transcript or '[none]'}\n"
                f"Typed draft: {prompt_text or '[none]'}\n"
                f"Relevant paths: {repository_context.get('relevant_paths', [])}\n"
                f"Previously asked questions: {previous_questions[-8:]}\n"
                f"Previously used topics: {sorted(previous_topics)}\n"
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

        if decision.status == "clarifying" and not clarification_limit_reached:
            if not _decision_has_novel_question(
                decision,
                previous_questions=previous_questions,
                previous_topics=previous_topics,
            ):
                fallback_topic, fallback_question = _next_novel_clarification(
                    history=history,
                    transcript=transcript,
                    prompt_text=prompt_text,
                    previous_questions=previous_questions,
                )

                if fallback_topic is None or fallback_question is None:
                    decision = decision.model_copy(
                        update={
                            "status": "ready",
                            "reply_text": (
                                "I have enough context to prepare the implementation handoff."
                            ),
                            "clarification_topic": None,
                            "coding_request": (
                                decision.coding_request
                                or "Implement the resolved voice request using the gathered context."
                            ),
                        }
                    )
                else:
                    repair_content = (
                        "Your proposed clarifying question repeated an earlier "
                        "question or topic. Return one corrected JSON decision. "
                        "Ask exactly one concise question using the unused topic "
                        f"'{fallback_topic}', or return status=ready if that "
                        "topic is already answered. Do not reuse any blocked "
                        "wording.\n\n"
                        f"Blocked questions: {previous_questions[-8:]}\n"
                        f"Blocked topics: {sorted(previous_topics)}\n"
                        f"Deterministic fallback question for this topic: "
                        f"{fallback_question}"
                    )
                    repair_messages = [
                        *messages,
                        {
                            "role": "assistant",
                            "content": decision.model_dump_json(),
                        },
                        {"role": "user", "content": repair_content},
                    ]

                    try:
                        repaired_decision = _request_intake_decision(
                            messages=repair_messages,
                            temperature=0.0,
                        )
                    except Exception as repair_exc:
                        logger.warning(
                            "Voice intake duplicate-question repair failed; "
                            "using deterministic fallback: %s",
                            repair_exc,
                        )
                        repaired_decision = decision

                    if repaired_decision.status == "ready":
                        decision = repaired_decision
                    elif _decision_has_novel_question(
                        repaired_decision,
                        previous_questions=previous_questions,
                        previous_topics=previous_topics,
                    ):
                        decision = repaired_decision
                    else:
                        decision = decision.model_copy(
                            update={
                                "status": "clarifying",
                                "reply_text": fallback_question,
                                "clarification_topic": fallback_topic,
                                "coding_request": None,
                            }
                        )

        if clarification_limit_reached and (
            decision.status != "ready" or not decision.coding_request
        ):
            return {
                "status": "ready",
                "reply_text": (
                    "I have enough context. I am handing a detailed "
                    "plan to the coding agent now."
                ),
                "coding_request": _fallback_coding_request(
                    state=state,
                    history=history,
                    transcript=transcript,
                ),
                "collected_facts": decision.collected_facts,
            }

        if decision.status == "ready":
            decision = decision.model_copy(update={"clarification_topic": None})
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
            "asked_questions": [*previous_questions, decision.reply_text],
        }

    except Exception as exc:
        fallback_request = _fallback_coding_request(
            state=state,
            history=history,
            transcript=transcript,
        )
        return {
            "status": "ready",
            "reply_text": (
                "I gathered the available context and prepared a "
                "fallback plan for the coding agent."
            ),
            "coding_request": fallback_request,
            "errors": [*state.get("errors", []), f"Voice intake model failed: {exc}"],
        }
