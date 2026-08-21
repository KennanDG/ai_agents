from __future__ import annotations

from ai_agents.agents.coding.memory import memory_namespace, CodingAgentRuntimeContext





def test_memory_namespace_uses_original_repo_root():
    context = CodingAgentRuntimeContext(
        user_id="local",
        memory_namespace="default",
    )

    first_run = {
        "repo_root": "/tmp/coding-sandboxes/run-1/repo",
        "original_repo_root": "/home/user/projects/ai_agents",
    }

    second_run = {
        "repo_root": "/tmp/coding-sandboxes/run-2/repo",
        "original_repo_root": "/home/user/projects/ai_agents",
    }

    first_namespace = memory_namespace(
        first_run,
        context,
    )

    second_namespace = memory_namespace(
        second_run,
        context,
    )

    assert first_namespace == second_namespace