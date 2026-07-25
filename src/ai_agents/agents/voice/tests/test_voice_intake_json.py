from __future__ import annotations

from ai_agents.agents.voice.nodes import _compact_repo_context, _fallback_coding_request
from ai_agents.agents.voice.schemas import VoiceIntakeDecision


def test_coding_request_object_is_normalized_to_objective() -> None:
    decision = VoiceIntakeDecision.model_validate(
        {
            "status": "ready",
            "reply_text": "Ready.",
            "coding_request": {
                "objective": "Add a clarity-check voice skill.",
                "repository_and_attachment_context": {"repository_tree": ["many/files"]},
            },
        }
    )

    assert decision.coding_request == "Add a clarity-check voice skill."


def test_compact_repo_context_does_not_embed_full_context_json() -> None:
    compact = _compact_repo_context(
        {
            "repo_root": "/repo",
            "active_path": "agents/voice/nodes.py",
            "repository_tree": [f"file-{index}.py" for index in range(200)],
            "explicit_files": [
                {"path": "agents/voice/nodes.py", "content_excerpt": "x" * 5_000}
            ],
            "search_matches": [
                {
                    "path": "agents/voice/prompts.py",
                    "score": 10,
                    "matched_terms": ["voice", "prompt"],
                    "content_excerpt": "y" * 5_000,
                }
            ],
            "attachment_context": [],
        }
    )

    assert len(compact["tree_sample"]) == 60
    assert len(compact["explicit_files"][0]["content_excerpt"]) == 1_500
    assert len(compact["search_matches"][0]["content_excerpt"]) == 900
    assert "repository_context_summary" not in compact


def test_fallback_plan_is_request_agnostic() -> None:
    request = _fallback_coding_request(
        state={
            "allow_write": True,
            "repo_context": {
                "explicit_files": [],
                "search_matches": [
                    {"path": "agents/voice/skills/implement_change.md"},
                    {"path": "agents/voice/nodes.py"},
                ],
            },
            "attached_files": [],
        },
        history=[
            {
                "role": "user",
                "content": "Add a new markdown skill under agents/voice/skills.",
            }
        ],
        transcript="Choose the skill name based on the repository.",
    )

    assert "agents/voice/skills/implement_change.md" in request
    assert "text area and voice recorder" not in request
    assert "attached_files array" not in request
