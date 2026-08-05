from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from ai_agents.agents.coding.coding_agent_settings import CodingAgentSettings
from ai_agents.agents.coding.state import CodingAgentState


def repo_root(state: CodingAgentState, cfg: CodingAgentSettings) -> Path:
    return Path(state.get("repo_root") or cfg.repo_root).resolve()


def allow_write(state: CodingAgentState, cfg: CodingAgentSettings) -> bool:
    return bool(state.get("allow_write", cfg.allow_write))


def node_config(
    node_name: str,
    state: CodingAgentState,
    extra_metadata: dict[str, Any] | None = None,
) -> RunnableConfig:
    return {
        "run_name": f"coding_agent_{node_name}",
        "tags": [
            "coding-agent",
            node_name,
            state.get("selected_skill", "no-skill"),
            state.get("task_mode", "standard"),
        ],
        "metadata": {
            "node": node_name,
            "selected_skill": state.get("selected_skill"),
            "task_mode": state.get("task_mode", "standard"),
            "context_generation": state.get("context_generation", 0),
            "patch_attempts": state.get("patch_attempts", 0),
            "allow_write": state.get("allow_write", False),
            **(extra_metadata or {}),
        },
    }
