from __future__ import annotations

from ai_agents.agents.voice.nodes import (
    _extract_questions,
    _is_repeated_question,
    _next_novel_clarification,
    _question_similarity,
)


def test_extract_questions_drops_non_question_preamble() -> None:
    reply = (
        "I understand that validation is the focus. "
        "What concrete validation failure are you seeing now?"
    )

    assert _extract_questions(reply) == [
        "What concrete validation failure are you seeing now?"
    ]


def test_rephrased_generic_question_is_detected_as_duplicate() -> None:
    first = (
        "Can you please clarify what specific changes or improvements you would "
        "like to see in nodes.py?"
    )
    repeated = (
        "Can you please clarify what specific changes or improvements you would "
        "like to see in the validation logic?"
    )

    assert _question_similarity(first, repeated) >= 0.64
    assert _is_repeated_question(repeated, [first]) is True


def test_fallback_rotates_to_an_unanswered_topic() -> None:
    history = [
        {
            "role": "user",
            "content": "Could you suggest how coding nodes.py can be improved?",
        },
        {
            "role": "assistant",
            "content": (
                "Can you please clarify what specific changes or improvements you "
                "would like to see in nodes.py?"
            ),
        },
        {
            "role": "user",
            "content": (
                "Optimize performance and focus on validation logic where useful."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Can you please clarify what specific changes or improvements you "
                "would like to see in the validation logic?"
            ),
        },
    ]
    previous_questions = [
        history[1]["content"],
        history[3]["content"],
    ]

    topic, question = _next_novel_clarification(
        history=history,
        transcript="It should align with the project's environments.",
        prompt_text="",
        previous_questions=previous_questions,
    )

    assert topic == "current_behavior"
    assert question == (
        "What concrete failure, slowdown, or false validation result are you seeing now?"
    )
