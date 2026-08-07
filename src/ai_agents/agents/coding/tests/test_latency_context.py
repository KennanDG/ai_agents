from __future__ import annotations

from ai_agents.agents.coding.coding_agent_settings import CodingAgentSettings
from ai_agents.agents.coding.nodes import _chunk_ranges_for_text, _classify_task_mode


def test_localized_request_uses_simple_fast_path() -> None:
    state = {
        "user_request": "Make the text size bigger in src/App.tsx.",
        "attached_files": [
            {"source": "repo", "path": "src/App.tsx", "name": "App.tsx"}
        ],
    }

    assert _classify_task_mode(state, CodingAgentSettings()) == "simple"


def test_multi_concern_request_uses_parallel_mode() -> None:
    state = {
        "user_request": """
        Please update the coding agent:
        - add parallel context workers
        - fix attachment truncation
        - reduce model calls
        - add a simple-task fast path
        """,
        "attached_files": [],
    }

    assert _classify_task_mode(state, CodingAgentSettings()) == "parallel"


def test_large_file_chunks_include_matching_symbol() -> None:
    cfg = CodingAgentSettings(
        max_full_file_chars=1_000,
        context_chunk_chars=600,
        context_chunk_overlap_chars=50,
    )
    text = "header\n" + ("x = 1\n" * 1_000) + "def target_symbol():\n    return 42\n" + ("y = 2\n" * 1_000)

    ranges = _chunk_ranges_for_text(
        text,
        terms=["target_symbol"],
        requested_ranges=[],
        cfg=cfg,
    )

    assert ranges
    assert any("target_symbol" in text[start:end] for start, end, _, _ in ranges)


def test_requested_line_range_is_included_without_full_file() -> None:
    cfg = CodingAgentSettings(
        max_full_file_chars=1_000,
        context_chunk_chars=300,
        context_chunk_overlap_chars=0,
    )
    lines = [f"line_{index}\n" for index in range(1, 501)]
    text = "".join(lines)

    ranges = _chunk_ranges_for_text(
        text,
        terms=[],
        requested_ranges=[(250, 260)],
        cfg=cfg,
    )

    selected = "".join(text[start:end] for start, end, _, _ in ranges)
    assert "line_250" in selected
    assert "line_260" in selected
