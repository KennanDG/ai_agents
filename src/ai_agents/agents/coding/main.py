from __future__ import annotations

import os
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from langchain_core.runnables import RunnableConfig
from langsmith import traceable
from dotenv import load_dotenv

from ai_agents.agents.coding.graph import build_coding_agent_graph, _bullets

load_dotenv()

LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2", True)
LANGCHAIN_API_KEY=os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT", "ai-agents-dev")
LANGSMITH_ENDPOINT=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")


def _fence(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        text = "(empty)"
    return f"```text\n{text[-8000:]}\n```"




def render_markdown_report(result: dict[str, Any]) -> str:
    """Render the final graph state as a Markdown report."""

    validation_results = result.get("validation_results", [])
    validation_md = []

    for item in validation_results:
        command = item.get("command", "")
        returncode = item.get("returncode", "")
        stdout = _fence(item.get("stdout", ""))
        stderr = _fence(item.get("stderr", ""))

        validation_md.append(
            f"### `{command}`\n\n"
            f"- Exit code: `{returncode}`\n\n"
            f"**stdout**\n\n{stdout}\n\n"
            f"**stderr**\n\n{stderr}"
        )

    diff_blocks = []
    for idx, diff in enumerate(result.get("diffs", []), start=1):
        diff_blocks.append(f"### Diff {idx}\n\n```diff\n{diff.strip()}\n```")

    file_change_lines = []

    for item in result.get("file_changes", []):
        path = item.get("path", "")
        reason = item.get("reason", "")
        write_result = item.get("write_result", "")
        file_change_lines.append(f"- `{path}` — {write_result}" + (f"\n  - Reason: {reason}" if reason else ""))

    return f"""# Coding Agent Run Report

## Request

{result.get("user_request", "")}

## Status

- Final status: `{result.get("status", "unknown")}`
- Selected skill: `{result.get("selected_skill", "none")}`
- Write mode: `{"enabled" if result.get("allow_write") else "dry-run"}`

## Plan

{_bullets(result.get("plan", []))}

## Search Queries

{_bullets(result.get("search_queries", []))}

## Files Inspected

{_bullets(result.get("files_inspected", []))}

## Patch Summary

{result.get("patch_summary", "No patch summary generated.")}

## File Changes

{chr(10).join(file_change_lines) if file_change_lines else "- None"}

## Diffs

{chr(10).join(diff_blocks) if diff_blocks else "- None"}

## Validation

{chr(10).join(validation_md) if validation_md else "- No validation commands were run."}

## Errors

{_bullets(result.get("errors", []))}
""".strip() + "\n"




def write_markdown_report(
        result: dict, 
        repo_root: str | Path, 
        output_path: str | Path | None = None
    ) -> Path:
    """Write a readable coding-agent run report to Markdown."""

    root = Path(repo_root).resolve()

    if output_path:
        report_path = Path(output_path)

        if not report_path.is_absolute():
            report_path = root / report_path

    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = root / "agents" / "coding" / "logs" / "runs"
        report_path = report_dir / f"coding_agent_run_{timestamp}.md"


    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown_report(result), encoding="utf-8")

    return report_path



@traceable(name="run_coding_agent", tags=["coding", "agent"])
def run_coding_agent(
    user_request: str,
    repo_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
    *,
    allow_write: bool = False,
) -> dict:
    
    graph = build_coding_agent_graph()

    initial_state = {
        "user_request": user_request,
        "repo_root": str(repo_root or Path.cwd()),
        "workspace_root": str(workspace_root) if workspace_root else None,
        "allow_write": allow_write,
        "errors": [],
    }

    return graph.invoke(initial_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight coding agent.")

    parser.add_argument("--repo-root", default=".", help="Repository root to inspect")

    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Project root used for validation commands. Defaults to nearest parent containing pyproject.toml.",
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Allow the agent to write generated file changes. Without this, it runs in dry-run mode.",
    )

    parser.add_argument(
        "--markdown-report",
        action="store_true",
        help="Write a detailed Markdown report under logs/runs.",
    )

    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional Markdown report path. Implies --markdown-report.",
    )

    parser.add_argument("request", help="Coding task or bug report")

    args = parser.parse_args()

    result = run_coding_agent(args.request, args.repo_root, args.workspace_root, allow_write=args.write)

    if args.markdown_report or args.report_path:
        report_path = write_markdown_report(result, args.repo_root, args.report_path)
        print(f"Markdown report written to: {report_path}")
        print()

    print(result.get("report", result))


if __name__ == "__main__":
    main()