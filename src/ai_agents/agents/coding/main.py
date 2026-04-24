from __future__ import annotations

import argparse
from pathlib import Path

from ai_agents.agents.coding.graph import build_coding_agent_graph


def run_coding_agent(
    user_request: str,
    repo_root: str | Path | None = None,
    *,
    allow_write: bool = False,
) -> dict:
    graph = build_coding_agent_graph()

    initial_state = {
        "user_request": user_request,
        "repo_root": str(repo_root or Path.cwd()),
        "allow_write": allow_write,
        "errors": [],
    }

    return graph.invoke(initial_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight coding agent.")
    parser.add_argument("request", help="Coding task or bug report")
    parser.add_argument("--repo-root", default=".", help="Repository root to inspect")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Allow the agent to write generated file changes. Without this, it runs in dry-run mode.",
    )

    args = parser.parse_args()

    result = run_coding_agent(args.request, args.repo_root, allow_write=args.write)
    
    print(result.get("report", result))


if __name__ == "__main__":
    main()