from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from ai_agents.agents.coding.utils.constants import (
    MAX_FILES_TO_INSPECT,
    MAX_PATCH_ATTEMPTS,
    VALIDATION_PROFILE_NAME,
)

from ai_agents.agents.coding.llm import invoke_parsed_decision
from ai_agents.agents.coding.utils.patch import (
    apply_exact_replace,
    build_patch_context,
    is_forbidden_write_path,
)

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
from ai_agents.agents.coding.runtime import allow_write as resolve_allow_write
from ai_agents.agents.coding.runtime import repo_root as resolve_repo_root
from ai_agents.agents.coding.schemas import (
    ContextDecision,
    PatchDecision,
    PlanDecision,
    ReportDecision,
)

from ai_agents.agents.coding.utils.search import (
    derive_search_requests,
    filter_context_paths,
    legacy_queries_to_search_requests,
    paths_from_ranked_results,
    paths_from_search_results,
)

from ai_agents.agents.coding.settings import CodingAgentSettings, settings as default_settings
from ai_agents.agents.coding.utils.skills import skill_instructions_for_llm
from ai_agents.agents.coding.state import CodingAgentState
from ai_agents.agents.coding.tests.runner import run_validation_suite
from ai_agents.agents.coding.utils.text import bullets, dedupe
from ai_agents.agents.coding.tools.filesystem import list_files, read_file, write_file
from ai_agents.agents.coding.tools.patch import unified_diff
from ai_agents.agents.coding.tools.web_search import web_search
from ai_agents.agents.coding.tools.search import format_search_results, search_repository

from ai_agents.agents.coding.utils.validation import (
    default_validation_commands,
    validation_failed_results,
)

from ai_agents.config.settings import settings as config_settings


load_dotenv()



OPEN_ROUTER_API_KEY=os.environ["OPEN_ROUTER_API_KEY"]
OPEN_ROUTER_URL="https://openrouter.ai/api/v1"

DOCKER_API_KEY=os.environ["DOCKER_API_KEY"]
DOCKER_URL="https://api.deepseek.com"


model = ChatGroq(
    model=config_settings.coding_model,
    api_key=config_settings.resolved_groq_api_key(),
)


reasoning_model = ChatOpenAI(
    model=config_settings.reasoning_model, # e.g., "deepseek/deepseek-v4-pro"
    api_key=OPEN_ROUTER_API_KEY, 
    base_url=OPEN_ROUTER_URL,
    max_retries=2
)






def _dump_search_request(search_request: object) -> dict[str, object]:
    """Serialize Pydantic search request models into plain state dicts."""

    if hasattr(search_request, "model_dump"):
        return search_request.model_dump(exclude_none=True)  # type: ignore[attr-defined]

    if hasattr(search_request, "dict"):
        return search_request.dict(exclude_none=True)  # type: ignore[attr-defined]

    if isinstance(search_request, dict):
        return dict(search_request)

    return {}


def _planned_search_requests(decision: PlanDecision, request: str) -> list[dict[str, object]]:
    search_requests = [
        item
        for item in (_dump_search_request(item) for item in decision.search_requests)
        if item
    ]

    if search_requests:
        return search_requests

    if decision.search_queries:
        return legacy_queries_to_search_requests(decision.search_queries)

    return derive_search_requests(request)





def plan_node(state: CodingAgentState) -> CodingAgentState:
    request = state["user_request"]
    planner_prompt = build_planner_user_prompt(request)

    if state.get("selected_skill"):
        planner_prompt += (
            "\n\nSelected skill:\n"
            f"{state.get('selected_skill')}\n\n"
            "Skill guidance for planning:\n"
            f"{skill_instructions_for_llm(state.get('skill_instructions', ''))}"
        )

    try:
        decision: PlanDecision = invoke_parsed_decision(
            model=reasoning_model,
            schema=PlanDecision,
            node_name="plan",
            state=state,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=planner_prompt,
        )

        search_requests = _planned_search_requests(decision, request)

        return {
            "plan": decision.plan,
            "search_requests": search_requests,
            "search_queries": decision.search_queries,
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
            "search_queries": derive_search_requests(request),
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




def gather_context_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    repo_root = resolve_repo_root(state, cfg)
    context: list[str] = []
    errors = list(state.get("errors", []))
    files_inspected: list[str] = []

    try:
        root_files = filter_context_paths(list_files(repo_root, ".", max_depth=2))[: cfg.max_search_results]
        context.append("Repository files:\n" + "\n".join(root_files))

    except Exception as exc:
        return {
            "errors": [*errors, f"Context gathering failed while listing files: {exc}"],
            "status": "context_failed",
        }

    search_blocks: list[str] = []
    search_result_dicts: list[dict[str, object]] = []
    search_requests = list(state.get("search_requests") or [])

    if not search_requests:
        search_requests = legacy_queries_to_search_requests(state.get("search_queries", []))

    if not search_requests:
        search_requests = derive_search_requests(state["user_request"])

    if search_requests:
        try:
            search_results = search_repository(
                repo_root,
                search_requests,
                max_results=cfg.max_search_results,
            )
            search_result_dicts = [result.to_dict() for result in search_results]

            if search_results:
                search_blocks.append(
                    "Ranked repository search results:\n"
                    + "\n".join(format_search_results(search_results))
                )

        except Exception as exc:
            errors.append(f"Structured repository search failed: {exc}")

    context.extend(search_blocks)

    web_results = state.get("web_search_results")

    if web_results:
        context.append(f"\n\n# Web search results:\n{web_results}")

    try:
        decision: ContextDecision = invoke_parsed_decision(
            model=model,
            schema=ContextDecision,
            node_name="context_selector",
            state=state,
            system_prompt=CONTEXT_SELECTOR_SYSTEM_PROMPT,
            user_prompt=build_context_selector_user_prompt(
                request=state["user_request"],
                selected_skill=state.get("selected_skill"),
                skill_instructions=skill_instructions_for_llm(
                    state.get("skill_instructions", "")
                ),
                available_context="\n".join(context),
            ),
        )

        candidate_paths = filter_context_paths([item.path for item in decision.files_to_inspect])

        if not candidate_paths:
            candidate_paths = paths_from_ranked_results(search_result_dicts)


    except Exception as exc:
        errors.append(f"LLM context selection failed; using search-derived context only: {exc}")
        candidate_paths = paths_from_ranked_results(search_result_dicts) or paths_from_search_results(search_blocks)


    for path in dedupe(filter_context_paths(candidate_paths))[:MAX_FILES_TO_INSPECT]:
        try:
            content = read_file(repo_root, path, max_chars=cfg.max_file_chars)
            files_inspected.append(path)
            context.append(f"File: {path}\n```\n{content}\n```")

        except Exception as exc:
            errors.append(f"Could not read {path}: {exc}")

    return {
        "context": context,
        "files_inspected": files_inspected,
        "search_requests": search_requests,
        "search_results": search_result_dicts,
        "errors": errors,
        "status": "context_gathered" if context else "context_failed",
    }






def _same_file_content(existing: str, requested: str) -> bool:
    """Return True when an attempted create is effectively already applied."""
    return existing == requested or existing.rstrip("\n") == requested.rstrip("\n")


def patch_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Ask the LLM for exact edits and apply them when writes are enabled."""
    
    repo_root = resolve_repo_root(state, cfg)
    errors = list(state.get("errors", []))
    allow_write = resolve_allow_write(state, cfg)
    patch_attempts = int(state.get("patch_attempts", 0)) + 1
    max_patch_attempts = int(state.get("max_patch_attempts", MAX_PATCH_ATTEMPTS))

    previous_file_changes = list(state.get("file_changes", []))
    previous_diffs = list(state.get("diffs", []))
    
    known_changed_paths = {
        item.get("path", "")
        for item in previous_file_changes
        if item.get("path")
    }

    file_changes: list[dict[str, str]] = [*previous_file_changes]
    diffs: list[str] = [*previous_diffs]

    attempt_file_changes: list[dict[str, str]] = []
    attempt_write_results: list[str] = []
    idempotent_noops = 0

    try:
        decision: PatchDecision = invoke_parsed_decision(
            model=reasoning_model,
            schema=PatchDecision,
            node_name="patch",
            state={**state, "patch_attempts": patch_attempts},
            system_prompt=PATCHER_SYSTEM_PROMPT,
            user_prompt=build_patcher_user_prompt(
                request=state["user_request"],
                selected_skill=state.get("selected_skill"),
                skill_instructions=skill_instructions_for_llm(
                    state.get("skill_instructions", "")
                ),
                plan=bullets(state.get("plan", [])),
                context=build_patch_context(state),
            ),
        )

    except Exception as exc:
        return {
            "patch_attempts": patch_attempts,
            "max_patch_attempts": max_patch_attempts,
            "patch_summary": f"LLM patching failed: {exc}",
            "errors": [*errors, f"LLM patching failed on attempt {patch_attempts}: {exc}"],
            "status": "patch_failed",
        }

    for edit in decision.edits:
        path = edit.path.strip()

        if not path or is_forbidden_write_path(path):
            errors.append(f"Skipped forbidden or empty write path: {path}")
            continue

        try:
            if edit.operation == "create":
                if edit.old.strip():
                    raise ValueError(
                        f"Create operation for {path} must use an empty old value."
                    )

                try:
                    existing = read_file(repo_root, path, max_chars=1_000_000)
                except FileNotFoundError:
                    before = ""
                    after = edit.new
                else:
                    if _same_file_content(existing, edit.new):
                        result = (
                            f"No-op: {path} already exists with the requested content."
                        )
                        attempt_write_results.append(result)
                        idempotent_noops += 1

                        if path not in known_changed_paths:
                            change = {
                                "path": path,
                                "operation": edit.operation,
                                "reason": edit.reason,
                                "write_result": result,
                            }
                            attempt_file_changes.append(change)
                            file_changes.append(change)
                            known_changed_paths.add(path)

                        continue

                    raise FileExistsError(
                        f"Cannot create {path}; file already exists with different content. "
                        "Use operation='replace'."
                    )

            elif edit.operation == "replace":
                if not edit.old:
                    raise ValueError(
                        f"Replace operation for {path} requires non-empty old text."
                    )

                before = read_file(repo_root, path, max_chars=1_000_000)
                after = apply_exact_replace(before, edit.old, edit.new, path=path)

            else:
                raise ValueError(f"Unsupported edit operation for {path}: {edit.operation}")

            
            diffs.append(unified_diff(path, before, after))
            result = write_file(repo_root, path, after, allow_write=allow_write)
            attempt_write_results.append(result)

            change = {
                "path": path,
                "operation": edit.operation,
                "reason": edit.reason,
                "write_result": result,
            }

            attempt_file_changes.append(change)
            file_changes.append(change)
            known_changed_paths.add(path)

        except Exception as exc:
            errors.append(f"Failed to process edit for {path}: {exc}")

    validation_commands = decision.validation_commands or state.get("validation_commands") or []
    mode = "WRITE MODE" if allow_write else "DRY RUN"

    successful_attempt_items = len(attempt_file_changes) + idempotent_noops

    patch_summary = (
        f"{mode}: {decision.summary}\n\n"
        f"Patch attempt: {patch_attempts}/{max_patch_attempts}\n"
        f"Files changed/proposed this attempt: {len(attempt_file_changes)}\n"
        f"Idempotent create no-ops this attempt: {idempotent_noops}\n"
        f"Total files changed/proposed: {len(file_changes)}\n"
        f"Write results:\n{bullets(attempt_write_results)}"
    )


    if decision.edits and successful_attempt_items == 0:
        status: Literal["patched", "patch_failed", "patch_skipped"] = "patch_failed"
    elif not decision.edits:
        status = "patch_skipped"
    else:
        status = "patched"
        

    return {
        "patch_attempts": patch_attempts,
        "max_patch_attempts": max_patch_attempts,
        "file_changes": file_changes,
        "diffs": diffs,
        "patch_summary": patch_summary,
        "validation_commands": validation_commands,
        "errors": errors,
        "status": status,
    }




def validate_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    workspace_root = Path(state.get("workspace_root") or ".").resolve()
    commands = state.get("validation_commands") or default_validation_commands(workspace_root)

    changed_files = [
        item.get("path", "")
        for item in state.get("file_changes", [])
        if item.get("path")
    ]

    try:
        suite = run_validation_suite(
            workspace_root,
            changed_files=changed_files,
            requested_commands=commands,
            allow_shell=cfg.allow_shell,
            timeout_seconds=cfg.shell_timeout_seconds,
            profile_name=VALIDATION_PROFILE_NAME,
        )
        results = suite.to_dicts()

    except Exception as exc:
        results = [
            {
                "command": "validation_suite",
                "returncode": 1,
                "stdout": "",
                "stderr": str(exc),
            }
        ]
        return {
            "validation_results": results,
            "errors": [*state.get("errors", []), f"Validation suite failed: {exc}"],
            "status": "validation_failed",
        }

    return {
        "validation_results": results,
        "status": "validation_failed" if validation_failed_results(results) else "validated",
    }




def report_node(state: CodingAgentState) -> CodingAgentState:
    validation_results = state.get("validation_results", [])
    validation_lines = []

    for result in validation_results:
        command = result.get("command")
        returncode = result.get("returncode")
        validation_lines.append(f"- `{command}` -> exit code {returncode}")

    try:
        decision: ReportDecision = invoke_parsed_decision(
            model=model,
            schema=ReportDecision,
            node_name="report",
            state=state,
            system_prompt=REPORTER_SYSTEM_PROMPT,
            user_prompt=build_reporter_user_prompt(
                request=state.get("user_request", ""),
                selected_skill=state.get("selected_skill"),
                files_inspected=bullets(state.get("files_inspected", [])),
                file_changes=bullets(
                    [
                        item.get("path", "") + " - " + item.get("write_result", "")
                        for item in state.get("file_changes", [])
                    ]
                ),
                patch_summary=state.get("patch_summary", ""),
                validation=chr(10).join(validation_lines)
                if validation_lines
                else "No validation commands were run.",
                errors=bullets(state.get("errors", []))
                if state.get("errors")
                else "None",
            ),
        )

        return {"report": decision.report, "status": "reported"}

    except Exception:
        report = f"""Coding agent run summary

Request:
{state.get("user_request", "")}

Selected skill:
{state.get("selected_skill", "none")}

Plan:
{bullets(state.get("plan", []))}

Files inspected:
{bullets(state.get("files_inspected", []))}

Patch:
{state.get("patch_summary", "No patch summary generated.")}

Validation:
{chr(10).join(validation_lines) if validation_lines else "No validation commands were run."}

Errors:
{bullets(state.get("errors", [])) if state.get("errors") else "None"}
""".strip()
        return {"report": report, "status": "reported"}


def web_search_node(state: CodingAgentState) -> CodingAgentState:
    """
    Perform web search if the selected skill is web_search.
    Uses the user request as the search query.
    """
    if state.get("selected_skill") != "web_search":
        return {"status": "web_search_skipped"}

    query = state.get("user_request", "")
    if not query:
        return {
            "web_search_results": "",
            "status": "web_search_skipped",
        }

    try:
        results = web_search(query, num_results=5)
        return {
            "web_search_results": results,
            "status": "web_search_completed",
        }
    except Exception as exc:
        return {
            "web_search_results": f"Web search failed: {exc}",
            "errors": [*state.get("errors", []), f"Web search failed: {exc}"],
            "status": "web_search_failed",
        }


def gmail_access_node(state: CodingAgentState) -> CodingAgentState:
    """
    Perform Gmail access if the selected skill is gmail_access.
    Currently a placeholder.
    """
    if state.get("selected_skill") != "gmail_access":
        return {"status": "gmail_access_skipped"}
    # Placeholder: log that gmail access was triggered
    # In future, invoke gmail API here.
    return {"status": "gmail_access_completed"}
