from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypeVar, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ai_agents.config.settings import settings as config_settings

from ai_agents.agents.coding.prompts import (
    CONTEXT_SELECTOR_SYSTEM_PROMPT,
    PATCHER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REPORTER_SYSTEM_PROMPT,
    build_context_selector_user_prompt,
    build_patcher_user_prompt,
    build_planner_user_prompt,
    build_reporter_user_prompt,
)
from ai_agents.agents.coding.registry import SkillRegistry, route_skill
from ai_agents.agents.coding.settings import CodingAgentSettings, settings as default_settings
from ai_agents.agents.coding.tools.filesystem import list_files, read_file, write_file
from ai_agents.agents.coding.tools.search import search_repo
from ai_agents.agents.coding.tools.shell import run_command
from ai_agents.agents.coding.tools.patch import unified_diff
from ai_agents.agents.coding.tests.runner import run_validation_suite


# Initialize the Groq model
llm = ChatGroq(
    model=config_settings.reasoning_model,
    api_key=config_settings.resolved_groq_api_key()
)

model = ChatGroq(
    model=config_settings.coding_model,
    api_key=config_settings.resolved_groq_api_key()
)

print("llm:", config_settings.coding_model)
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
    plan: list[str] = Field(
        default_factory=list,
        description="Short implementation plan steps.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Repository search terms to gather context.",
    )
    validation_commands: list[str] = Field(
        default_factory=list,
        description="Safe validation commands to run after edits.",
    )


class FileToInspect(BaseModel):
    path: str
    reason: str = ""


class ContextDecision(BaseModel):
    files_to_inspect: list[FileToInspect] = Field(default_factory=list)


class FileEdit(BaseModel):
    path: str = Field(description="Repository-relative path to edit.")
    old: str = Field(description="Exact existing text to replace.")
    new: str = Field(description="Replacement text.")
    reason: str = Field(default="", description="Why this edit is needed.")


class PatchDecision(BaseModel):
    summary: str = ""
    edits: list[FileEdit] = Field(default_factory=list)
    validation_commands: list[str] = Field(default_factory=list)


class ReportDecision(BaseModel):
    report: str




def _repo_root(state: CodingAgentState, cfg: CodingAgentSettings) -> Path:
    return Path(state.get("repo_root") or cfg.repo_root).resolve()


def _allow_write(state: CodingAgentState, cfg: CodingAgentSettings) -> bool:
    return bool(state.get("allow_write", cfg.allow_write))




DecisionT = TypeVar("DecisionT", bound=BaseModel)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def _invoke_parsed_decision(
    *,
    schema: type[DecisionT],
    node_name: str,
    state: CodingAgentState,
    system_prompt: str,
    user_prompt: str,
) -> DecisionT:
    """
    Calls the model as a normal chat completion and parses the response locally.

    This intentionally avoids llm.with_structured_output(...), because Groq may
    convert that into provider tool-calling. The graph runner owns all repository
    operations; the model only returns a structured decision object.
    """
    parser = PydanticOutputParser(pydantic_object=schema)

    response = llm.invoke(
        [
            (
                "system",
                f"{system_prompt}\n\n"
                "Do not call tools or functions. The LangGraph runner executes "
                "all repository operations after your response is parsed.\n"
                "Return only the structured object requested below.\n\n"
                f"{parser.get_format_instructions()}",
            ),
            ("human", user_prompt),
        ],
        config=_node_config(node_name, state),
    )

    return parser.parse(_message_content_to_text(response.content))



def _apply_exact_replace(original: str, old: str, new: str, *, path: str) -> str:
    if not old:
        raise ValueError(f"Empty old text for {path}")

    count = original.count(old)

    if count == 0:
        raise ValueError(f"Could not find exact old text in {path}")

    if count > 1:
        raise ValueError(f"Old text matched more than once in {path}; refusing ambiguous edit")

    return original.replace(old, new, 1)



def _skill_instructions_for_llm(skill_instructions: str) -> str:
    """
    Skill files list allowed tools for the graph runner.

    The model itself is not a tool-calling agent in this workflow. If we pass
    the raw "Allowed tools" section to the LLM, some models try to call those
    tool names directly, which fails because they are not bound in request.tools.
    """
    if not skill_instructions:
        return ""

    lines = skill_instructions.splitlines()
    cleaned: list[str] = []
    skipping_allowed_tools = False

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("allowed tools"):
            skipping_allowed_tools = True
            cleaned.append(
                "Repository operations are executed by the LangGraph runner. "
                "Do not call tools directly. Return structured output only."
            )
            continue

        if skipping_allowed_tools:
            if not stripped:
                skipping_allowed_tools = False
                continue

            if stripped.startswith("-"):
                continue

            skipping_allowed_tools = False

        cleaned.append(line)

    return "\n".join(cleaned).strip()



def _node_config(
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
        ],
        "metadata": {
            "node": node_name,
            "selected_skill": state.get("selected_skill"),
            "allow_write": state.get("allow_write", False),
            **(extra_metadata or {}),
        },
    }







def plan_node(state: CodingAgentState) -> CodingAgentState:

    request = state["user_request"]

    try:
        planner = llm.with_structured_output(PlanDecision)

        # planner = llm.with_structured_output(PlanDecision, method="json_mode")

        decision: PlanDecision = planner.invoke(
            [
                ("system", PLANNER_SYSTEM_PROMPT),
                ("human", build_planner_user_prompt(request)),
            ],
            config=_node_config("plan", state),
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




def gather_context_node(state: CodingAgentState, cfg: CodingAgentSettings = default_settings) -> CodingAgentState:
    repo_root = _repo_root(state, cfg)
    context: list[str] = []
    errors = list(state.get("errors", []))
    files_inspected: list[str] = []

    try:
        root_files = list_files(repo_root, ".", max_depth=2)[: cfg.max_search_results]
        context.append("Repository files:\n" + "\n".join(root_files))

    except Exception as exc:
        return {"errors": [*errors, f"Context gathering failed while listing files: {exc}"]}

    
    search_blocks: list[str] = []

    for query in state.get("search_queries", []):
        matches = search_repo(repo_root, query, max_results=cfg.max_search_results)
        if matches:
            search_blocks.append(f"Search results for '{query}':\n" + "\n".join(matches))
    context.extend(search_blocks)

    try:
        selector = llm.with_structured_output(ContextDecision)

        # selector = llm.with_structured_output(ContextDecision, method="json_mode")

        decision: ContextDecision = selector.invoke(
            [
                ("system", CONTEXT_SELECTOR_SYSTEM_PROMPT),
                (
                    "human",
                    build_context_selector_user_prompt(
                        request=state["user_request"],
                        selected_skill=state.get("selected_skill"),
                        skill_instructions=_skill_instructions_for_llm(
                            state.get("skill_instructions", "")
                        ),
                        available_context="\n".join(context),
                    ),
                ),
            ],
            config=_node_config("context_selector", state),
        )

        candidate_paths = [item.path for item in decision.files_to_inspect]

    except Exception as exc:
        errors.append(f"LLM context selection failed; using search-derived context only: {exc}")
        candidate_paths = _paths_from_search_results(search_blocks)


    for path in _dedupe(candidate_paths)[:10]:
        try:
            content = read_file(repo_root, path, max_chars=cfg.max_file_chars)
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




def patch_node(state: CodingAgentState, cfg: CodingAgentSettings = default_settings) -> CodingAgentState:
    """Ask the LLM for full-file changes and apply them when writes are enabled."""

    repo_root = _repo_root(state, cfg)
    errors = list(state.get("errors", []))
    allow_write = _allow_write(state, cfg)

    file_changes: list[dict[str, str]] = []
    diffs: list[str] = []
    write_results: list[str] = []

    try:
        patcher = llm.with_structured_output(PatchDecision)
        
        # patcher = llm.with_structured_output(PatchDecision, method="json_mode")

        decision: PatchDecision = patcher.invoke(
            [
                ("system", PATCHER_SYSTEM_PROMPT),
                (
                    "human",
                    build_patcher_user_prompt(
                        request=state["user_request"],
                        selected_skill=state.get("selected_skill"),
                        skill_instructions=_skill_instructions_for_llm(
                            state.get("skill_instructions", "")
                        ),
                        plan=_bullets(state.get("plan", [])),
                        context="\n".join(state.get("context", [])),
                    ),
                ),
            ],
            config=_node_config("patch", state),
        )


        # decision: PatchDecision = _invoke_parsed_decision(
        #     schema=PatchDecision,
        #     node_name="patch",
        #     state=state,
        #     system_prompt=PATCHER_SYSTEM_PROMPT,
        #     user_prompt=build_patcher_user_prompt(
        #         request=state["user_request"],
        #         selected_skill=state.get("selected_skill"),
        #         skill_instructions=_skill_instructions_for_llm(
        #             state.get("skill_instructions", "")
        #         ),
        #         plan=_bullets(state.get("plan", [])),
        #         context="\n".join(state.get("context", [])),
        #     ),
        # )


    except Exception as exc:
        return {
            "patch_summary": f"LLM patching failed: {exc}",
            "errors": [*errors, f"LLM patching failed: {exc}"],
            "status": "patched",
        }
    
    print("patch_decision:\n", decision.model_dump())
    # print("type:\n", type(json.loads(decision.model_dump()['content'])))

    # decision_dict = json.loads(decision.model_dump()['content'])

    for edit in decision.edits:
        path = edit.path.strip()

        if not path or _is_forbidden_write_path(path):
            errors.append(f"Skipped forbidden or empty write path: {path}")
            continue

        try:
            before = read_file(repo_root, path, max_chars=1_000_000)
            after = _apply_exact_replace(before, edit.old, edit.new, path=path)

            diffs.append(unified_diff(path, before, after))
            result = write_file(repo_root, path, after, allow_write=allow_write)
            write_results.append(result)
            file_changes.append({"path": path, "reason": edit.reason, "write_result": result})

        except Exception as exc:
            errors.append(f"Failed to process edit for {path}: {exc}")

    # for change in decision.file_changes:
    #     path = change.path.strip()

    #     if not path or _is_forbidden_write_path(path):
    #         errors.append(f"Skipped forbidden or empty write path: {path}")
    #         continue

    #     try:
    #         try:
    #             before = read_file(repo_root, path, max_chars=1_000_000)

    #         except FileNotFoundError:
    #             before = ""

    #         after = change.content
    #         diffs.append(unified_diff(path, before, after))
    #         result = write_file(repo_root, path, after, allow_write=allow_write)
    #         write_results.append(result)
    #         file_changes.append({"path": path, "reason": change.reason, "write_result": result})

    #     except Exception as exc:
    #         errors.append(f"Failed to process change for {path}: {exc}")



    validation_commands = decision.validation_commands or state.get("validation_commands") or []


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




def validate_node(state: CodingAgentState, cfg: CodingAgentSettings = default_settings) -> CodingAgentState:
    
    workspace_root = Path(state.get("workspace_root") or ".").resolve()

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
        allow_shell=cfg.allow_shell,
        timeout_seconds=cfg.shell_timeout_seconds,
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

        # reporter = llm.with_structured_output(ReportDecision, method="json_mode")

        decision: ReportDecision = reporter.invoke(
            [
                ("system", REPORTER_SYSTEM_PROMPT),
                (
                    "human",
                    build_reporter_user_prompt(
                        request=state.get("user_request", ""),
                        selected_skill=state.get("selected_skill"),
                        files_inspected=_bullets(state.get("files_inspected", [])),
                        file_changes=_bullets(
                            [
                                item.get("path", "") + " - " + item.get("write_result", "")
                                for item in state.get("file_changes", [])
                            ]
                        ),
                        patch_summary=state.get("patch_summary", ""),
                        validation=chr(10).join(validation_lines)
                        if validation_lines
                        else "No validation commands were run.",
                        errors=_bullets(state.get("errors", []))
                        if state.get("errors")
                        else "None",
                    ),
                ),
            ],
            config=_node_config("report", state),
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
    if (repo_root / Path("pyproject.toml")).exists():
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
    forbidden_parts = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
    forbidden_names = {
        ".env",
        ".env.local",
        ".env.production",
        ".env.docker",
        "uv.lock",
        "poetry.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
    }

    return bool(parts & forbidden_parts) or Path(path).name in forbidden_names