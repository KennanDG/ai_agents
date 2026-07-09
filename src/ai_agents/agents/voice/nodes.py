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

    context = {
        "repo_root": state.get("repo_root"),
        "workspace_root": state.get("workspace_root"),
        "active_path": state.get("active_path"),
        "allow_write": state.get("allow_write", False),
    }

    user_content = (
        f"Latest user transcript:\n{transcript}\n\n"
        f"Current UI/repo context:\n{json.dumps(context, indent=2)}"
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

        if decision.status == "ready" and not decision.coding_request:
            return {
                "status": "clarifying",
                "reply_text": "I understand the general request, but I need one more detail before handing it to the coding agent. What file or feature should it focus on?",
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