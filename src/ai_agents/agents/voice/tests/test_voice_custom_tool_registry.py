from __future__ import annotations

from pathlib import Path

import pytest

from ai_agents.agents.coding.tool_registry import CustomToolValidationError
from ai_agents.agents.voice.tool_registry import (
    ApprovedVoiceToolRegistry,
    validate_voice_custom_tool_source,
)


def test_voice_tool_rejects_unknown_required_argument() -> None:
    source = '''
"""Invalid voice tool."""

def invalid_voice_tool(query: str):
    return {"query": query}
'''

    with pytest.raises(CustomToolValidationError, match="Unsupported required argument"):
        validate_voice_custom_tool_source("invalid_voice_tool", source)


def test_voice_registry_injects_backend_context(tmp_path: Path) -> None:
    source = '''
"""Summarize backend-owned voice context."""

from pathlib import Path


def voice_context_summary(repo_root: str, transcript: str, active_path: str | None = None, limit: int = 3):
    root = Path(repo_root)
    python_files = sorted(path.name for path in root.rglob("*.py"))[:limit]
    return {
        "repo_name": root.name,
        "transcript": transcript,
        "active_path": active_path,
        "python_files": python_files,
    }
'''
    tool_path = tmp_path / "voice_context_summary.py"
    tool_path.write_text(source, encoding="utf-8")

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")

    registry = ApprovedVoiceToolRegistry(tmp_path).load()
    assert registry.has("voice_context_summary")

    result = registry.invoke_from_state(
        "voice_context_summary",
        {
            "repo_root": str(repo),
            "transcript": "Please inspect the Python project.",
            "active_path": "a.py",
            "attached_files": [],
            "history": [],
            "repo_context": {},
        },
    )

    assert result["success"] is True
    assert '"repo_name": "repo"' in result["output"]
    assert '"active_path": "a.py"' in result["output"]
    assert '"a.py"' in result["output"]
