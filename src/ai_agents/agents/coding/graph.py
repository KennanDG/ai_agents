from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from ai_agents.config.settings import settings as config_settings

from ai_agents.agents.coding.prompts import PATCHER_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT, REPORTER_SYSTEM_PROMPT
from ai_agents.agents.coding.registry import SkillRegistry, route_skill
from ai_agents.agents.coding.settings import CodingAgentSettings, settings as default_settings
from ai_agents.agents.coding.tools.filesystem import list_files, read_file, write_file
from ai_agents.agents.coding.tools.search import search_repo
from ai_agents.agents.coding.tools.shell import run_command
from ai_agents.agents.coding.tools.patch import unified_diff
from ai_agents.agents.coding.tests.runner import run_validation_suite



# Initialize the Groq model
llm = ChatGroq(
    model=config_settings.coding_model,
    api_key=config_settings.resolved_groq_api_key()
)


class CodingAgentState(TypedDict, total=False):
    user_request: str
    repo_root: str          # target root for searching/patching
    workspace_root: str     # project root for validation
    allow_write: bool
    selected_skill: str
    skill_instructions: str
    plan: list[str]
    search_queries: list[str]
    context: list[str]
    files_inspected: list[str]
    file_changes: list[dict[str, str]]
    diffs: list[str]
    patch_summary: str
    validation_commands: list[str]
    validation_results: list[dict[str, Any]]
    report: str
    status: Literal["planned", "routed", "context_gathered", "patched", "validated", "reported"]
    errors: list[str]



class PlanDecision(BaseModel):
    plan: list[str] = Field(description="Short implementation plan steps.")
    search_queries: list[str] = Field(description="Repository search terms to gather context.")
    validation_commands: list[str] = Field(description="Safe validation commands to run after edits.")


class FileToInspect(BaseModel):
    path: str
    reason: str = ""


class ContextDecision(BaseModel):
    files_to_inspect: list[FileToInspect] = Field(default_factory=list)


class FileChange(BaseModel):
    path: str = Field(description="Repository-relative path to create or overwrite.")
    content: str = Field(description="Full final content for the file.")
    reason: str = Field(default="", description="Why this file needs to change.")


class PatchDecision(BaseModel):
    summary: str = ""
    file_changes: list[FileChange] = Field(default_factory=list)
    validation_commands: list[str] = Field(default_factory=list)


class ReportDecision(BaseModel):
    report: str




def _repo_root(state: CodingAgentState, config: CodingAgentSettings) -> Path:
    return Path(state.get("repo_root") or config.repo_root).resolve()


def _allow_write(state: CodingAgentState, config: CodingAgentSettings) -> bool:
    return bool(state.get("allow_write", config.allow_write))




def plan_node(state: CodingAgentState) -> CodingAgentState:

    request = state["user_request"]

    try:
        planner = llm.with_structured_output(PlanDecision)
        decision: PlanDecision = planner.invoke(
            [
                ("system", PLANNER_SYSTEM_PROMPT),
                (
                    "human",
                    f"""Create a minimal coding-agent plan for this request.
                    \n\nRequest:\n{request}\n\nRules:
                    \n- Search queries should be short terms likely to appear in file names or code.
                    \n- Validation commands must be safe, like `uv run pytest`, `uv run ruff check .`, or `python -m compileall .`.
                    \n- Do not invent specific files unless the request clearly names them.""",
                ),
            ]
        )
        return {
            "plan": decision.plan,
            "search_queries": decision.search_queries or _derive_search_queries(request),
            "validation_commands": decision.validation_commands,
            "status": "planned",
        }
    
    except Exception as exc:
        return {
            "plan": [
                "Select the most relevant coding skill.",
                "Gather repository context before editing.",
                "Create a minimal patch.",
                "Run targeted validation.",
                "Report inspected files, changed files, validation, and errors.",
            ],
            "search_queries": _derive_search_queries(request),
            "errors": [*state.get("errors", []), f"LLM planning failed; used fallback plan: {exc}"],
            "status": "planned",
        }




def route_node(state: CodingAgentState) -> CodingAgentState:
    selected_skill = route_skill(state["user_request"])
    registry = SkillRegistry().load()
    skill = registry.get(selected_skill)

    return {
        "selected_skill": skill.name,
        "skill_instructions": skill.instructions,
        "status": "routed",
    }




def gather_context_node(state: CodingAgentState, config: CodingAgentSettings = default_settings) -> CodingAgentState:
    repo_root = _repo_root(state, config)
    context: list[str] = []
    errors = list(state.get("errors", []))
    files_inspected: list[str] = []

    try:
        root_files = list_files(repo_root, ".", max_depth=2)[: config.max_search_results]
        context.append("Repository files:\n" + "\n".join(root_files))

    except Exception as exc:
        return {"errors": [*errors, f"Context gathering failed while listing files: {exc}"]}

    
    search_blocks: list[str] = []

    for query in state.get("search_queries", []):
        matches = search_repo(repo_root, query, max_results=config.max_search_results)
        if matches:
            search_blocks.append(f"Search results for '{query}':\n" + "\n".join(matches))
    context.extend(search_blocks)

    try:
        selector = llm.with_structured_output(ContextDecision)

        decision: ContextDecision = selector.invoke(
            [
                (
                    "system",
                    "You select repository files that must be read before editing. Return only repo-relative paths that appear in the repository map or search results.",
                ),
                (
                    "human",
                    f"""Request:
                    \n{state['user_request']}
                    \n\nSelected skill:
                    \n{state.get('selected_skill')}
                    \n\nSkill instructions:
                    \n{state.get('skill_instructions', '')[:4000]}
                    \n\nAvailable context:\n{chr(10).join(context)[:12000]}""",
                ),
            ]
        )

        candidate_paths = [item.path for item in decision.files_to_inspect]

    except Exception as exc:
        errors.append(f"LLM context selection failed; using search-derived context only: {exc}")
        candidate_paths = _paths_from_search_results(search_blocks)


    for path in _dedupe(candidate_paths)[:10]:
        try:
            content = read_file(repo_root, path, max_chars=config.max_file_chars)
            files_inspected.append(path)
            context.append(f"File: {path}\n```\n{content}\n```")

        except Exception as exc:
            errors.append(f"Could not read {path}: {exc}")

    return {
        "context": context,
        "files_inspected": files_inspected,
        "errors": errors,
        "status": "context_gathered",
    }




def patch_node(state: CodingAgentState, config: CodingAgentSettings = default_settings) -> CodingAgentState:
    """Ask the LLM for full-file changes and apply them when writes are enabled."""

    repo_root = _repo_root(state, config)
    errors = list(state.get("errors", []))
    allow_write = _allow_write(state, config)

    file_changes: list[dict[str, str]] = []
    diffs: list[str] = []
    write_results: list[str] = []

    try:
        patcher = llm.with_structured_output(PatchDecision)
        decision: PatchDecision = patcher.invoke(
            [
                ("system", PATCHER_SYSTEM_PROMPT),
                (
                    "human",
                    f"""You are modifying a real repository. Produce the smallest safe full-file edits needed for the request.
                    \n\nRequest:\n{state['user_request']}\n\nSelected skill:\n{state.get('selected_skill')}
                    \n\nSkill instructions:\n{state.get('skill_instructions', '')[:6000]}
                    \n\nPlan:
                    \n{_bullets(state.get('plan', []))}
                    \n\nContext:
                    \n{chr(10).join(state.get('context', []))[:30000]}
                    \n\nRules:
                    \n- Return full final file contents for every changed file.
                    \n- Only change files supported by the provided context.
                    \n- Prefer small, focused edits.
                    \n- Do not modify secrets, .env files, lock files, generated caches, or unrelated files.
                    \n- Include validation commands relevant to the changed files.
                    \n- If there is not enough context, return no file changes and explain what is missing in the summary.""",
                ),
            ]
        )

    except Exception as exc:
        return {
            "patch_summary": f"LLM patching failed: {exc}",
            "errors": [*errors, f"LLM patching failed: {exc}"],
            "status": "patched",
        }

    for change in decision.file_changes:
        path = change.path.strip()

        if not path or _is_forbidden_write_path(path):
            errors.append(f"Skipped forbidden or empty write path: {path}")
            continue

        try:
            try:
                before = read_file(repo_root, path, max_chars=1_000_000)

            except FileNotFoundError:
                before = ""

            after = change.content
            diffs.append(unified_diff(path, before, after))
            result = write_file(repo_root, path, after, allow_write=allow_write)
            write_results.append(result)
            file_changes.append({"path": path, "reason": change.reason, "write_result": result})

        except Exception as exc:
            errors.append(f"Failed to process change for {path}: {exc}")



    validation_commands = decision.validation_commands or state.get("validation_commands") or _default_validation_commands(repo_root)


    mode = "WRITE MODE" if allow_write else "DRY RUN"

    patch_summary = (
        f"{mode}: {decision.summary}\n\n"
        f"Files changed/proposed: {len(file_changes)}\n"
        f"Write results:\n{_bullets(write_results)}"
    )


    return {
        "file_changes": file_changes,
        "diffs": diffs,
        "patch_summary": patch_summary,
        "validation_commands": validation_commands,
        "errors": errors,
        "status": "patched",
    }




def validate_node(state: CodingAgentState, config: CodingAgentSettings = default_settings) -> CodingAgentState:
    
    workspace_root = state.get("workspace_root", "")

    print("\n\nCWD:", workspace_root, "\n\n")

    commands = state.get("validation_commands") or _default_validation_commands(workspace_root)

    changed_files = [
        item.get("path", "")
        for item in state.get("file_changes", [])
        if item.get("path")
    ]

    suite = run_validation_suite(
        workspace_root,
        changed_files=changed_files,
        requested_commands=commands,
        allow_shell=config.allow_shell,
        timeout_seconds=config.shell_timeout_seconds,
        profile_name="coding-agent-default",
    )

    return {
        "validation_results": suite.to_dicts(),
        "status": "validated",
    }




def report_node(state: CodingAgentState) -> CodingAgentState:
    validation_results = state.get("validation_results", [])
    validation_lines = []

    for result in validation_results:
        command = result.get("command")
        returncode = result.get("returncode")
        validation_lines.append(f"- `{command}` -> exit code {returncode}")

    try:
        reporter = llm.with_structured_output(ReportDecision)
        decision: ReportDecision = reporter.invoke(
            [
                ("system", REPORTER_SYSTEM_PROMPT),
                (
                    "human",
                    f"""Create a concise coding-agent run report.
                    \n\nRequest:\n{state.get('user_request', '')}
                    \n\nSelected skill:
                    \n{state.get('selected_skill', 'none')}
                    \n\nFiles inspected:
                    \n{_bullets(state.get('files_inspected', []))}
                    \n\nFile changes:
                    \n{_bullets([item.get('path', '') + ' - ' + item.get('write_result', '') for item in state.get('file_changes', [])])}
                    \n\nPatch summary:\n{state.get('patch_summary', '')}
                    \n\nValidation:
                    \n{chr(10).join(validation_lines) if validation_lines else 'No validation commands were run.'}
                    \n\nErrors:
                    \n{_bullets(state.get('errors', [])) if state.get('errors') else 'None'}""",
                ),
            ]
        )

        return {"report": decision.report, "status": "reported"}
    
    except Exception:   
        report = f"""Coding agent run summary

Request:
{state.get("user_request", "")}

Selected skill:
{state.get("selected_skill", "none")}

Plan:
{_bullets(state.get("plan", []))}

Files inspected:
{_bullets(state.get("files_inspected", []))}

Patch:
{state.get("patch_summary", "No patch summary generated.")}

Validation:
{chr(10).join(validation_lines) if validation_lines else "No validation commands were run."}

Errors:
{_bullets(state.get("errors", [])) if state.get("errors") else "None"}
""".strip()
        return {"report": report, "status": "reported"}





def build_coding_agent_graph():
    
    builder = StateGraph(CodingAgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("route", route_node)
    builder.add_node("gather_context", gather_context_node)
    builder.add_node("patch", patch_node)
    builder.add_node("validate", validate_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "route")
    builder.add_edge("route", "gather_context")
    builder.add_edge("gather_context", "patch")
    builder.add_edge("patch", "validate")
    builder.add_edge("validate", "report")
    builder.add_edge("report", END)


    return builder.compile()





def _derive_search_queries(user_request: str) -> list[str]:
    base = user_request.lower()
    queries: list[str] = []

    for term in [
        "fastapi",
        "router",
        "endpoint",
        "langgraph",
        "state",
        "pytest",
        "test",
        "settings",
        "registry",
        "tool",
        "skill",
    ]:
        if term in base:
            queries.append(term)
    return queries or ["TODO", "def ", "class "]


def _default_validation_commands(repo_root: Path) -> list[str]:
    if (repo_root / "pyproject.toml").exists():
        return ["uv run pytest", "uv run ruff check ."]
    
    return ["python -m compileall ."]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        normalized = item.strip()

        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


def _paths_from_search_results(search_blocks: list[str]) -> list[str]:
    paths: list[str] = []

    for block in search_blocks:
        for line in block.splitlines():
            if ":" not in line:
                continue

            candidate = line.split(":", 1)[0].strip()

            if candidate:
                paths.append(candidate)
                
    return _dedupe(paths)


def _is_forbidden_write_path(path: str) -> bool:
    parts = set(Path(path).parts)
    forbidden_parts = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
    forbidden_names = {".env", ".env.local", ".env.production"}

    return bool(parts & forbidden_parts) or Path(path).name in forbidden_names