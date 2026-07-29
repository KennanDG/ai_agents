from __future__ import annotations


from typing import Literal
from textwrap import dedent


from ai_agents.agents.coding.utils.constants import (
    MAX_FILES_TO_INSPECT,
    MAX_PATCH_ATTEMPTS,
    MAX_REPO_NAVIGATION_FILES,
    MAX_REPO_NAVIGATION_FOLLOWUP_RESULTS,
    VALIDATION_PROFILE_NAME,
    VALIDATION_PROFILE_NAME,
)

from ai_agents.agents.coding.llm import invoke_parsed_decision
from ai_agents.agents.coding.memory import recall_coding_memories, remember_coding_run
from ai_agents.agents.coding.utils.patch import (
    apply_exact_replace,
    build_patch_context,
    is_forbidden_write_path,
)

from ai_agents.agents.coding.prompts import (
    CONTEXT_SELECTOR_SYSTEM_PROMPT,
    PATCHER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REPO_NAVIGATOR_SYSTEM_PROMPT,
    REPORTER_SYSTEM_PROMPT,
    SKILL_ROUTER_SYSTEM_PROMPT,
    build_context_selector_user_prompt,
    build_patcher_user_prompt,
    build_planner_user_prompt,
    build_repo_navigator_user_prompt,
    build_reporter_user_prompt,
    build_skill_router_user_prompt
)


from ai_agents.agents.coding.registry import SkillRegistry, route_skill
from ai_agents.agents.coding.model_factory import coding_model, reasoning_model
from ai_agents.agents.coding.routing import patch_attempts_remaining
from ai_agents.agents.coding.runtime import allow_write as resolve_allow_write
from ai_agents.agents.coding.runtime import repo_root as resolve_repo_root
from ai_agents.agents.coding.schemas import (
    ContextDecision,
    PatchDecision,
    PlanDecision,
    ProgressDecision,
    RepoNavigationDecision,
    ReportDecision,
    SkillRouteDecision,
)

from ai_agents.agents.coding.utils.search import (
    derive_search_requests,
    filter_context_paths,
    paths_from_ranked_results,
    paths_from_search_results,
)

from ai_agents.agents.coding.coding_agent_settings import CodingAgentSettings, settings as default_settings
from ai_agents.agents.coding.utils.skills import skill_instructions_for_llm
from ai_agents.agents.coding.state import CodingAgentState
from ai_agents.agents.coding.tests.runner import run_validation_suite
from ai_agents.agents.coding.utils.text import bullets, dedupe
from ai_agents.agents.coding.tools.filesystem import list_files, read_file, write_file
from ai_agents.agents.coding.tools.patch import unified_diff
from ai_agents.agents.coding.tools.web_search import web_search
from ai_agents.agents.coding.tools.search import format_search_results, search_repository

from ai_agents.agents.coding.utils.validation import (
    advisory_validation_failures,
    blocking_validation_failures,
    default_validation_commands,
    validation_failed_results,
)


from ai_agents.agents.coding.utils.helpers import(
    _dump_search_request,
    _planned_search_requests,
    _search_requests_from_state, 
    _format_search_result_dicts,
    _repo_navigation_path_reasons,
    _route_with_fallback,
    _repo_attachment_paths,
    _format_attached_files_for_context,
    _format_attached_files_for_context,
    _attached_file_summary,
    _resolve_existing_repo_path,
    _resolve_context_candidate_paths,
    _format_loop_context_focus,
    _derive_loop_search_requests,
    _context_file_read_limit,
    _same_file_content,
)




#################################### Nodes ####################################

def recall_memory_node(state: CodingAgentState, runtime) -> CodingAgentState:
    return recall_coding_memories(state, runtime)


def remember_run_node(state: CodingAgentState, runtime) -> CodingAgentState:
    return remember_coding_run(state, runtime)



def route_node(state: CodingAgentState) -> CodingAgentState:
    registry = SkillRegistry().load()

    try:
        default_skill = registry.default_skill_name()
    except ValueError as exc:
        return {
            "selected_skill": "",
            "skill_instructions": "",
            "route_confidence": 0.0,
            "route_reason": str(exc),
            "route_alternatives": [],
            "errors": [*state.get("errors", []), str(exc)],
            "status": "route_failed",
        }

    try:
        decision: SkillRouteDecision = invoke_parsed_decision(
            model=coding_model(),
            schema=SkillRouteDecision,
            node_name="route",
            state=state,
            system_prompt=SKILL_ROUTER_SYSTEM_PROMPT,
            user_prompt=build_skill_router_user_prompt(
                request=state["user_request"],
                skill_catalog=registry.router_catalog(),
            ),
        )

    except Exception as exc:
        return _route_with_fallback(
            state=state,
            registry=registry,
            error=f"LLM skill routing failed; used fallback router: {exc}",
        )

    selected_skill = decision.selected_skill.strip()

    if not registry.has(selected_skill):
        selected_skill = route_skill(
            state["user_request"],
            registry.list_names(),
            default_skill=default_skill,
        )
        route_reason = (
            f"LLM selected an unknown skill; used fallback route. "
            f"LLM reason: {decision.reason}"
        )
        confidence = 0.0
    else:
        route_reason = decision.reason
        confidence = decision.confidence

    skill = registry.get(selected_skill)
    alternatives = [
        {
            "skill_name": item.skill_name,
            "reason": item.reason,
        }
        for item in decision.alternatives
        if registry.has(item.skill_name) and item.skill_name != selected_skill
    ][:3]

    return {
        "selected_skill": skill.name,
        "skill_instructions": skill.instructions,
        "route_confidence": confidence,
        "route_reason": route_reason,
        "route_alternatives": alternatives,
        "status": "routed",
    }




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

    long_term_memories = state.get("long_term_memories", [])

    if long_term_memories:
        planner_prompt += (
            "\n\nRelevant long-term coding memories from previous runs:\n"
            f"{bullets(long_term_memories)}"
        )
    

    attachment_summary = _attached_file_summary(state)

    if attachment_summary:
        planner_prompt += (
            "\n\nUser-attached files available as additional read-only context:\n"
            f"{attachment_summary}"
        )


    try:
        decision: PlanDecision = invoke_parsed_decision(
            model=reasoning_model(),
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




def repo_navigator_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Read-only sub-agent that chooses the most useful repo files to inspect."""

    repo_root = resolve_repo_root(state, cfg)
    errors = list(state.get("errors", []))
    search_requests = _search_requests_from_state(state)
    search_result_dicts = list(state.get("search_results") or [])

    try:
        root_files = filter_context_paths(
            list_files(repo_root, ".", max_depth=6)
        )[: cfg.max_search_results]
    except Exception as exc:
        return {
            "errors": [*errors, f"Repo navigation failed while listing files: {exc}"],
            "status": "repo_navigation_failed",
        }

    if not search_result_dicts and search_requests:
        try:
            search_results = search_repository(
                repo_root,
                search_requests,
                max_results=cfg.max_search_results,
            )
            search_result_dicts = [result.to_dict() for result in search_results]
        except Exception as exc:
            errors.append(f"Repo navigation search failed: {exc}")

    try:
        decision: RepoNavigationDecision = invoke_parsed_decision(
            model=reasoning_model(),
            schema=RepoNavigationDecision,
            node_name="repo_navigator",
            state=state,
            system_prompt=REPO_NAVIGATOR_SYSTEM_PROMPT,
            user_prompt=build_repo_navigator_user_prompt(
                request=state["user_request"],
                selected_skill=state.get("selected_skill"),
                skill_instructions=skill_instructions_for_llm(
                    state.get("skill_instructions", "")
                ),
                plan=bullets(state.get("plan", [])),
                repository_files="\n".join(root_files),
                search_requests=bullets([str(item) for item in search_requests]),
                ranked_search_results=_format_search_result_dicts(search_result_dicts),
                web_results=str(state.get("web_search_results", "")),
                long_term_memories=bullets(state.get("long_term_memories", [])),
                attached_file_summary=_attached_file_summary(state),
                loop_context_focus=_format_loop_context_focus(state),
            ),
        )

    except Exception as exc:
        errors.append(
            "Repo navigator LLM failed; using ranked search results as navigation fallback: "
            f"{exc}"
        )
        fallback_paths = paths_from_ranked_results(search_result_dicts)[:MAX_REPO_NAVIGATION_FILES]
        fallback_files = [
            {"path": path, "reason": "Ranked search fallback after repo navigator failure."}
            for path in fallback_paths
        ]
        return {
            "search_requests": search_requests,
            "search_results": search_result_dicts,
            "repo_navigation_summary": "Repo navigator failed; using ranked search fallback.",
            "repo_navigation_files": fallback_files,
            "repo_navigation_confidence": 0.0,
            "repo_navigation_missing_context": ["Repo navigator LLM decision failed."],
            "repo_navigation_search_requests": [],
            "errors": errors,
            "status": "repo_navigated" if fallback_files else "repo_navigation_failed",
        }

    additional_search_requests = [
        item
        for item in (_dump_search_request(item) for item in decision.additional_search_requests)
        if item
    ]

    if additional_search_requests:
        try:
            followup_results = search_repository(
                repo_root,
                additional_search_requests,
                max_results=MAX_REPO_NAVIGATION_FOLLOWUP_RESULTS,
            )
            search_result_dicts.extend(result.to_dict() for result in followup_results)
        except Exception as exc:
            errors.append(f"Repo navigator follow-up search failed: {exc}")

    path_reasons = _repo_navigation_path_reasons(decision, search_result_dicts)
    selected_paths = [
        item.path.strip()
        for item in decision.files_to_inspect
        if item.path.strip()
    ]

    if additional_search_requests:
        selected_paths.extend(paths_from_ranked_results(search_result_dicts))

    if not selected_paths:
        selected_paths = paths_from_ranked_results(search_result_dicts)

    selected_paths = dedupe(filter_context_paths(selected_paths))[:MAX_REPO_NAVIGATION_FILES]
    navigation_files = [
        {
            "path": path,
            "reason": path_reasons.get(path, "Selected by repo navigator."),
        }
        for path in selected_paths
    ]

    return {
        "search_requests": search_requests,
        "search_results": search_result_dicts,
        "repo_navigation_summary": decision.task_summary,
        "repo_navigation_files": navigation_files,
        "repo_navigation_confidence": decision.confidence,
        "repo_navigation_missing_context": decision.missing_context,
        "repo_navigation_search_requests": additional_search_requests,
        "errors": errors,
        "status": "repo_navigated" if navigation_files else "repo_navigation_failed",
    }





def gather_context_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    
    repo_root = resolve_repo_root(state, cfg)
    context: list[str] = []
    errors = list(state.get("errors", []))
    files_inspected: list[str] = []

    ################## Workspace repo files ##################
    try:
        all_repo_files = filter_context_paths(list_files(repo_root, ".", max_depth=12))
        root_files = all_repo_files[: cfg.max_search_results]
        context.append("Repository files:\n" + "\n".join(root_files))

    except Exception as exc:
        return {
            "errors": [*errors, f"Context gathering failed while listing files: {exc}"],
            "status": "context_failed",
        }


    ################## Repo search results ##################
    search_blocks: list[str] = []
    search_result_dicts: list[dict[str, object]] = list(state.get("search_results") or [])
    search_requests = _search_requests_from_state(state)

    if search_result_dicts:
        search_blocks.append(
            "Ranked repository search results:\n"
            + _format_search_result_dicts(search_result_dicts)
        )

    elif search_requests:
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


    ################## Long-term memory ##################
    long_term_memories = state.get("long_term_memories", [])
    
    if long_term_memories:
        context.append(
            "# Relevant long-term coding memories from previous runs\n"
            + bullets(long_term_memories)
        )
    

    ################## User Attached files ##################
    attached_context, attached_files_used = _format_attached_files_for_context(state)

    if attached_context:
        context.append(attached_context)



    ################## Repo navigation ##################
    repo_navigation_summary = state.get("repo_navigation_summary")
    repo_navigation_files = list(state.get("repo_navigation_files") or [])
    repo_navigation_missing = list(state.get("repo_navigation_missing_context") or [])

    if repo_navigation_summary or repo_navigation_files:
        nav_lines = ["# Repo navigator"]

        if repo_navigation_summary:
            nav_lines.append(f"Summary: {repo_navigation_summary}")

        nav_lines.append(
            f"Confidence: {state.get('repo_navigation_confidence', 0.0)}"
        )

        if repo_navigation_files:
            nav_lines.append("Files selected:")
            for item in repo_navigation_files:
                path = str(item.get("path", "")).strip()
                reason = str(item.get("reason", "")).strip()
                if path:
                    nav_lines.append(f"- {path}: {reason}")

        if repo_navigation_missing:
            nav_lines.append("Missing context:")
            nav_lines.extend(f"- {item}" for item in repo_navigation_missing)

        context.append("\n".join(nav_lines))



    ################## Web Search Results ##################
    web_results = state.get("web_search_results")

    if web_results:
        context.append(f"\n\n# Web search results:\n{web_results}")


    ################## Context to focus on for re-try attempt ##################
    loop_context_focus = _format_loop_context_focus(state)

    if loop_context_focus:
        context.append("# Retry/context-refresh focus\n" + loop_context_focus)


    repo_attachment_paths = _repo_attachment_paths(state)
    navigator_paths = filter_context_paths(
        [str(item.get("path", "")).strip() for item in repo_navigation_files]
    )
    candidate_paths = dedupe([*repo_attachment_paths, *navigator_paths])

    
    if not navigator_paths:
        selected_paths: list[str] = []

        try:
            decision: ContextDecision = invoke_parsed_decision(
                model=coding_model(),
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

            selected_paths = filter_context_paths(
                [item.path for item in decision.files_to_inspect]
            )

            if not selected_paths:
                selected_paths = paths_from_ranked_results(search_result_dicts)

        except Exception as exc:
            errors.append(
                f"LLM context selection failed; using search-derived context only: {exc}"
            )
            selected_paths = (
                paths_from_ranked_results(search_result_dicts)
                or paths_from_search_results(search_blocks)
            )

        candidate_paths = dedupe([*repo_attachment_paths, *selected_paths])


    ################## Resolve paths ##################
    resolved_paths, unresolved_paths = _resolve_context_candidate_paths(
        repo_root=repo_root,
        candidate_paths=filter_context_paths(candidate_paths),
        repo_files=all_repo_files,
    )

    for path in unresolved_paths:
        errors.append(
            f"Skipped non-repo or ambiguous context path selected by navigator: {path}"
        )

    explicit_repo_paths = set(repo_attachment_paths)
    explicit_repo_file_count = len(explicit_repo_paths)

    for path in resolved_paths[:MAX_FILES_TO_INSPECT]:
        try:
            read_limit = _context_file_read_limit(
                path=path,
                explicit_repo_paths=explicit_repo_paths,
                explicit_repo_file_count=explicit_repo_file_count,
                cfg=cfg,
            )

            # Read one extra character so truncation can be detected instead of
            # silently passing a partial file as complete.
            probe = read_file(
                repo_root,
                path,
                max_chars=read_limit + 1,
            )

            content_truncated = len(probe) > read_limit
            content = probe[:read_limit]

            files_inspected.append(path)
            context.append(
                f"File: {path}\n"
                f"Content-Status: "
                f"{'truncated' if content_truncated else 'complete'}\n"
                f"Content-Limit: {read_limit}\n"
                "```\n"
                f"{content}\n"
                "```"
            )

            if content_truncated:
                errors.append(
                    f"Repository context for {path} was truncated at "
                    f"{read_limit} characters."
                )

        except Exception as exc:
            errors.append(f"Could not read {path}: {exc}")


    return {
        "context": context,
        "files_inspected": files_inspected,
        "attached_files_used": attached_files_used,
        "search_requests": search_requests,
        "search_results": search_result_dicts,
        "errors": errors,
        "status": "context_gathered" if context else "context_failed",
    }





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
    converted_creates = 0

    try:
        decision: PatchDecision = invoke_parsed_decision(
            model=reasoning_model(),
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
    

    repo_files = filter_context_paths(list_files(repo_root, ".", max_depth=12))

    for edit in decision.edits:

        path = edit.path.strip()

        resolved_path = _resolve_existing_repo_path(
            repo_root=repo_root,
            candidate=path,
            repo_files=repo_files,
        )

        if resolved_path and resolved_path != path:
            errors.append(f"Resolved patch path {path} to {resolved_path}.")
            path = resolved_path

        if not path or is_forbidden_write_path(path):
            errors.append(f"Skipped forbidden or empty write path: {path}")
            continue

        try:
            effective_operation = edit.operation
            converted_create_to_replace = False

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
                    before = existing
                    after = edit.new

                    if _same_file_content(existing, edit.new):
                        result = (
                            f"No-op: {path} already exists with the requested content."
                        )
                        attempt_write_results.append(result)
                        idempotent_noops += 1

                        if path not in known_changed_paths:
                            change = {
                                "path": path,
                                "operation": "create",
                                "status": "unchanged",
                                "reason": edit.reason,
                                "write_result": result,
                                "original": before,
                                "modified": after,
                            }
                            attempt_file_changes.append(change)
                            file_changes.append(change)
                            known_changed_paths.add(path)

                        continue

                    # The model selected create for a file that already exists. Treat
                    # the proposed full file contents as a safe full-file replacement
                    # instead of failing the whole run.
                    effective_operation = "replace"
                    converted_create_to_replace = True
                    converted_creates += 1

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

            if converted_create_to_replace:
                result = (
                    f"{result} Converted requested create operation to replace because "
                    f"{path} already existed."
                )

            attempt_write_results.append(result)

            change = {
                "path": path,
                "operation": effective_operation,
                "requested_operation": edit.operation,
                "status": "modified" if before else "added",
                "reason": edit.reason,
                "write_result": result,
                "original": before,
                "modified": after,
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
        f"Create operations converted to replace this attempt: {converted_creates}\n"
        f"Total files changed/proposed: {len(file_changes)}\n"
        f"Write results:\n{bullets(attempt_write_results)}"
    )


    if decision.edits and successful_attempt_items == 0:
        status: Literal["patched", "patch_failed", "patch_skipped"] = "patch_failed"
    elif not decision.edits:
        status = "patch_skipped"
    else:
        status = "patched"

    patch_retry_fields: dict[str, object] = {}

    if status in {"patch_failed", "patch_skipped"} and patch_attempts < max_patch_attempts:
        if status == "patch_failed":
            retry_reason = "failed while processing the proposed edits"
            retry_detail = (
                "Emphasize the files involved in the failed edits, exact current file "
                "contents, and any missing surrounding symbols/imports needed for a "
                "valid replacement."
            )
        else:
            retry_reason = "returned no edits even though repository context was gathered"
            retry_detail = (
                "Treat the patch summary as a missing-context signal. Re-select the "
                "direct implementation files and related schemas/transport files, then "
                "load their exact repository contents before retrying."
            )

        patch_retry_focus = (
            f"Patch attempt {patch_attempts} {retry_reason}. {retry_detail} "
            f"Patcher summary: {decision.summary or '(none)'}"
        )

        patch_retry_fields = {
            "continue_loop": True,
            "loop_context_focus": patch_retry_focus,
            "loop_notes": [
                *state.get("loop_notes", []),
                (
                    f"Patch attempt {patch_attempts}: {status}; refresh prioritized "
                    "repository context before retry."
                ),
            ][-8:],
            "search_results": [],
            "repo_navigation_files": [],
            "repo_navigation_missing_context": [],
            "context": [],
            "files_inspected": [],
        }

    return {
        **patch_retry_fields,
        "continue_loop": bool(patch_retry_fields),
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
    repo_root = resolve_repo_root(state, cfg)
    commands = state.get("validation_commands") or default_validation_commands(repo_root)

    changed_files = [
        item.get("path", "")
        for item in state.get("file_changes", [])
        if item.get("path")
    ]

    try:
        suite = run_validation_suite(
            repo_root,
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
                "reason": "Validation harness failed. Treating this as advisory so the user can review the patch.",
                "passed": False,
            }
        ]

    blocking_failures = blocking_validation_failures(results)
    advisory_failures = advisory_validation_failures(results)

    errors = list(state.get("errors", []))

    if blocking_failures:
        errors.append(
            "Blocking validation failed. Patch is still available for human review."
        )

    if advisory_failures:
        errors.append(
            "Advisory validation warnings were reported. Patch is still available for human approval."
        )

    return {
        "validation_results": results,
        "blocking_validation_failed": bool(blocking_failures),
        "advisory_validation_failed": bool(advisory_failures),
        "errors": errors,
        # Important: do not return validation_failed here.
        # The graph should continue to report/approval.
        "status": "validated",
    }


def assess_progress_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    
    iteration = int(state.get("iteration", 0)) + 1
    max_iterations = int(state.get("max_iterations", 3))
    errors = list(state.get("errors", []))
    loop_notes = list(state.get("loop_notes", []))

    validation_results = state.get("validation_results", [])
    validation_failed = validation_failed_results(validation_results)

    if iteration >= max_iterations:
        return {
            "iteration": iteration,
            "max_iterations": max_iterations,
            "continue_loop": False,
            "progress_reason": (
                f"Loop limit reached at {iteration}/{max_iterations}. "
                "Reporting current work instead of continuing."
            ),
            "loop_notes": [
                *loop_notes,
                f"Iteration {iteration}: loop limit reached.",
            ][-8:],
            "status": "loop_limit_reached",
        }

    validation_summary = "\n".join(
        f"- {item.get('command', 'unknown')} -> exit code {item.get('returncode', 'unknown')}"
        for item in validation_results
    ) or "No validation results."

    try:
        decision: ProgressDecision = invoke_parsed_decision(
            model=reasoning_model(),
            schema=ProgressDecision,
            node_name="assess_progress",
            state=state,
            system_prompt=(
                "You are the progress assessment node for a coding agent. "
                "Decide if the user's request is complete or if another implementation "
                "loop is needed. Continue only when there is concrete remaining work, "
                "failed validation that can likely be fixed, or missing context that can "
                "be gathered. Do not loop just to polish."
            ),
            user_prompt=dedent(
                f"""
                Assess whether this coding task is complete.

                # User request
                {state.get("user_request", "")}

                # Plan
                {bullets(state.get("plan", []))}

                # Files inspected
                {bullets(state.get("files_inspected", []))}

                # File changes
                {bullets([
                    item.get("path", "") + " - " + item.get("write_result", "")
                    for item in state.get("file_changes", [])
                ])}

                # Patch summary
                {state.get("patch_summary", "")}

                # Validation
                {validation_summary}

                # Existing errors
                {bullets(errors) if errors else "None"}

                # Prior loop notes
                {bullets(loop_notes) if loop_notes else "None"}

                # Iteration
                {iteration}/{max_iterations}

                Return whether the task is complete, whether to continue, and what
                the next loop should focus on.
                """
            ).strip(),
        )
    except Exception as exc:
        if validation_failed and patch_attempts_remaining(state):
            return {
                "iteration": iteration,
                "max_iterations": max_iterations,
                "continue_loop": True,
                "progress_reason": f"Progress assessment failed, but validation failed: {exc}",
                "remaining_tasks": ["Fix failing validation."],
                "loop_context_focus": (
                    "Progress assessment failed, but validation failed. "
                    "Refresh context around changed files and validation errors before patching again."
                ),
                "loop_notes": [
                    *loop_notes,
                    f"Iteration {iteration}: validation failed; continue with repair loop.",
                ][-8:],
                "search_requests": _derive_loop_search_requests(
                    state=state,
                    remaining_tasks=["Fix failing validation."],
                    next_iteration_notes="Refresh context around changed files and validation errors.",
                    reason=str(exc),
                ),
                "search_results": [],
                "repo_navigation_files": [],
                "repo_navigation_missing_context": [],
                "context": [],
                "files_inspected": [],
                "errors": [*errors, f"Progress assessment failed: {exc}"],
                "status": "assessed",
            }

        return {
            "iteration": iteration,
            "max_iterations": max_iterations,
            "continue_loop": False,
            "progress_reason": f"Progress assessment failed; reporting current state: {exc}",
            "errors": [*errors, f"Progress assessment failed: {exc}"],
            "status": "assessed",
        }

    additional_search_requests = [
        item
        for item in (_dump_search_request(item) for item in decision.additional_search_requests)
        if item
    ]

    should_continue = (
        decision.should_continue
        and not decision.is_complete
        and iteration < max_iterations
    )

    next_loop_focus = "\n".join(
        item
        for item in [
            f"Iteration {iteration} assessment: {decision.reason}",
            f"Next iteration focus: {decision.next_iteration_notes}"
            if decision.next_iteration_notes
            else "",
            "Remaining tasks:\n" + bullets(decision.remaining_tasks)
            if decision.remaining_tasks
            else "",
        ]
        if item
    )

    next_loop_search_requests = additional_search_requests or _derive_loop_search_requests(
        state=state,
        remaining_tasks=decision.remaining_tasks,
        next_iteration_notes=decision.next_iteration_notes,
        reason=decision.reason,
    )

    return {
        "iteration": iteration,
        "max_iterations": max_iterations,
        "continue_loop": should_continue,
        "remaining_tasks": decision.remaining_tasks,
        "progress_reason": decision.reason,
        "loop_context_focus": next_loop_focus if should_continue else state.get("loop_context_focus", ""),
        "loop_notes": [
            *loop_notes,
            f"Iteration {iteration}: {decision.reason}\nNext: {decision.next_iteration_notes}",
        ][-8:],
        # Force fresh navigation/search/context on the next loop.
        "search_requests": next_loop_search_requests if should_continue else state.get("search_requests", []),
        "search_results": [] if should_continue else state.get("search_results", []),
        "repo_navigation_summary": "" if should_continue else state.get("repo_navigation_summary", ""),
        "repo_navigation_files": [] if should_continue else state.get("repo_navigation_files", []),
        "repo_navigation_missing_context": [],
        "context": [] if should_continue else state.get("context", []),
        "files_inspected": [] if should_continue else state.get("files_inspected", []),
        "status": "assessed",
    }





def report_node(state: CodingAgentState) -> CodingAgentState:
    validation_results = state.get("validation_results", [])
    validation_lines = []

    for result in validation_results:
        command = result.get("command")
        returncode = result.get("returncode")
        validation_lines.append(f"- `{command}` -> exit code {returncode}")

    all_errors = [*state.get("errors", []), *state.get("memory_errors", [])]
    errors_text = bullets(all_errors) if all_errors else "None"

    try:
        decision: ReportDecision = invoke_parsed_decision(
            model=coding_model(),
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
                errors=errors_text,
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
{errors_text}
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
