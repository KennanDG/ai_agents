from __future__ import annotations

import json

from groq import Groq

from ai_agents.agents.voice.prompts import VOICE_INTAKE_SYSTEM_PROMPT
from ai_agents.agents.voice.schemas import VoiceIntakeDecision
from ai_agents.agents.voice.state import VoiceAgentState
from ai_agents.config.settings import settings


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

        safe.append({"role": role, "content": content[:4000]})

    return safe


def _count_prior_clarifications(history: list[dict[str, str]]) -> int:
    """Count prior voice-agent replies in the isolated voice conversation."""
    return sum(1 for item in history if item.get("role") == "assistant")


def _strip_voice_prefix(text: str) -> str:
    value = text.strip()
    return value.removeprefix("🎙️").strip()


def _fallback_coding_request(
    *,
    history: list[dict[str, str]],
    transcript: str,
    context: dict[str, object],
) -> str:
    user_turns = [
        _strip_voice_prefix(item["content"])
        for item in history
        if item.get("role") == "user" and isinstance(item.get("content"), str)
    ]
    user_turns.append(_strip_voice_prefix(transcript))

    resolved_turns = [turn for turn in user_turns if turn]
    conversation = "\n".join(f"- {turn}" for turn in resolved_turns)

    active_path = context.get("active_path")
    target_hint = (
        f"Start by inspecting the active file `{active_path}` and related files."
        if active_path
        else "Inspect the repository to identify the correct files before editing."
    )

    write_mode = (
        "The user has enabled write mode, so prepare the patch and follow the normal approval flow."
        if context.get("allow_write")
        else "Keep the run read-only and report the proposed changes."
    )

    return (
        "Implement the user's resolved request from this voice conversation:\n"
        f"{conversation}\n\n"
        f"{target_hint}\n"
        "Infer minor missing details from the repository and preserve existing behavior unless the request requires changing it.\n"
        f"{write_mode}"
    )


def intake_node(state: VoiceAgentState) -> VoiceAgentState:
    transcript = state.get("transcript", "").strip()
    history = _safe_history(state.get("history", []))

    if not transcript:
        return {
            "status": "error",
            "reply_text": "I could not hear anything clearly. Please try again.",
            "coding_request": None,
            "errors": [*state.get("errors", []), "Empty transcript."],
        }

    context: dict[str, object] = {
        "repo_root": state.get("repo_root"),
        "workspace_root": state.get("workspace_root"),
        "active_path": state.get("active_path"),
        "allow_write": state.get("allow_write", False),
    }

    clarification_count = _count_prior_clarifications(history)
    max_clarifications = max(1, settings.voice_max_clarifications)
    clarification_limit_reached = clarification_count >= max_clarifications

    user_content = (
        f"Latest user transcript:\n{transcript}\n\n"
        f"Current UI/repo context:\n{json.dumps(context, indent=2)}\n\n"
        f"Clarifying questions already asked: {clarification_count}\n"
        f"Maximum clarifying questions allowed: {max_clarifications}\n"
        f"Clarification limit reached: {clarification_limit_reached}\n\n"
        "If the clarification limit is reached, return status=ready and create the best implementation-ready coding_request from the full conversation."
    )

    try:
        completion = _client().chat.completions.create(
            model=settings.voice_chat_model,
            messages=[
                {"role": "system", "content": VOICE_INTAKE_SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content or "{}"
        decision = VoiceIntakeDecision.model_validate_json(content)

        # Deterministic guard: the model is not allowed to keep asking forever.
        if clarification_limit_reached and (
            decision.status != "ready" or not decision.coding_request
        ):
            return {
                "status": "ready",
                "reply_text": "I have enough to proceed. I am handing this to the coding agent now.",
                "coding_request": _fallback_coding_request(
                    history=history,
                    transcript=transcript,
                    context=context,
                ),
                "collected_facts": decision.collected_facts,
            }

        if decision.status == "ready" and not decision.coding_request:
            return {
                "status": "clarifying",
                "reply_text": "What file or feature should the coding agent focus on?",
                "coding_request": None,
                "collected_facts": decision.collected_facts,
            }

        return {
            "status": decision.status,
            "reply_text": decision.reply_text,
            "coding_request": decision.coding_request,
            "collected_facts": decision.collected_facts,
        }

    except Exception as exc:
        return {
            "status": "error",
            "reply_text": "I had trouble preparing that request. Please try again or type the task.",
            "coding_request": None,
            "errors": [*state.get("errors", []), f"Voice intake failed: {exc}"],
        }
