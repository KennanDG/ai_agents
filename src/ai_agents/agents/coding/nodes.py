from __future__ import annotations


from dataclasses import replace
from pathlib import Path
import re
from typing import Any, Iterable, Literal
from textwrap import dedent

from langgraph.types import Send


from ai_agents.agents.coding.utils.constants import (
    MAX_PATCH_ATTEMPTS,
    MAX_REPO_NAVIGATION_FILES,
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
    PATCHER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REPO_NAVIGATOR_SYSTEM_PROMPT,
    SKILL_ROUTER_SYSTEM_PROMPT,
    build_patcher_user_prompt,
    build_planner_user_prompt,
    build_repo_navigator_user_prompt,
    build_skill_router_user_prompt,
)


from ai_agents.agents.coding.skill_registry import MAX_SELECTED_SKILLS, SkillRegistry
from ai_agents.agents.coding.model_factory import build_chat_model
from ai_agents.agents.coding.tool_registry import (
    ApprovedCustomToolRegistry,
    MAX_CUSTOM_TOOL_CALLS,
)
from ai_agents.config.settings import settings as app_settings
from ai_agents.agents.coding.routing import patch_attempts_remaining
from ai_agents.agents.coding.runtime import allow_write as resolve_allow_write
from ai_agents.agents.coding.runtime import repo_root as resolve_repo_root
from ai_agents.agents.coding.coding_agent_schemas import (
    PatchDecision,
    PlanDecision,
    ProgressDecision,
    RepoNavigationDecision,
    SkillRouteDecision,
)

from ai_agents.agents.coding.utils.search import (
    filter_context_paths,
    paths_from_ranked_results,
)

from ai_agents.agents.coding.coding_agent_settings import CodingAgentSettings, settings as default_settings
from ai_agents.agents.coding.utils.skills import skill_instructions_for_llm
from ai_agents.agents.coding.state import CodingAgentState
from ai_agents.agents.coding.tests.runner import run_validation_suite
from ai_agents.agents.coding.utils.text import bullets, dedupe
from ai_agents.agents.coding.tools.filesystem import list_files, read_file, write_file
from ai_agents.agents.coding.tools.patch import unified_diff
from ai_agents.agents.coding.tools.web_search import web_search
from ai_agents.agents.coding.tools.search import search_repository

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
    _route_with_fallback,
    _repo_attachment_paths,
    _attached_file_summary,
    _resolve_existing_repo_path,
    _format_loop_context_focus,
    _derive_loop_search_requests,
    _same_file_content,
)


_RUNTIME_SETTING_FIELDS = {
    "max_context_workers",
    "route_max_tokens",
    "planner_max_tokens",
    "repo_navigation_max_tokens",
    "simple_patch_max_tokens",
    "patch_max_tokens",
    "progress_max_tokens",
}


def _settings_for_state(
    state: CodingAgentState | dict[str, Any],
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentSettings:
    raw = state.get("runtime_settings") or {}
    overrides = {
        key: int(value)
        for key, value in raw.items()
        if key in _RUNTIME_SETTING_FIELDS and isinstance(value, int)
    }
    return replace(cfg, **overrides) if overrides else cfg


def _coding_node_model(max_tokens: int, *, cache_namespace: str):
    return build_chat_model(
        provider=app_settings.coding_provider,
        model_name=app_settings.coding_model,
        max_tokens=max_tokens,
        prompt_cache_namespace=cache_namespace,
    )


def _reasoning_node_model(max_tokens: int, *, cache_namespace: str):
    return build_chat_model(
        provider=app_settings.reasoning_provider,
        model_name=app_settings.reasoning_model,
        max_tokens=max_tokens,
        prompt_cache_namespace=cache_namespace,
    )


#################################### Nodes ####################################

def recall_memory_node(state: CodingAgentState, runtime) -> CodingAgentState:
    # A localized fast-path edit does not benefit from a cross-thread memory query.
    if _classify_task_mode(state) == "simple":
        return {"long_term_memories": [], "memory_enabled": False}
    return recall_coding_memories(state, runtime)


def remember_run_node(state: CodingAgentState, runtime) -> CodingAgentState:
    return remember_coding_run(state, runtime)




_FILE_REFERENCE_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./()\-]+\.(?:py|tsx?|jsx?|css|html|json|md|sql|pls|fex|toml|ya?ml))",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _classify_task_mode(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> Literal["simple", "standard", "parallel"]:
    cfg = _settings_for_state(state, cfg)
    request = state.get("user_request", "")
    lowered = request.lower()
    attached = state.get("attached_files", [])
    file_refs = set(match.group("path") for match in _FILE_REFERENCE_RE.finditer(request))
    repo_attachments = {
        str(item.get("path", "")).strip()
        for item in attached
        if item.get("source") == "repo" and str(item.get("path", "")).strip()
    }

    bullet_count = sum(
        1 for line in request.splitlines() if line.lstrip().startswith(("-", "*", "•"))
    )
    parallel_markers = (
        "frontend and backend",
        "backend and frontend",
        "in parallel",
        "parallelization",
        "subagents",
        "sub-agents",
        "multiple components",
        "across the project",
        "configure sub agents",
    )
    if (
        bullet_count >= 3
        or len(file_refs | repo_attachments) >= 4
        or len(attached) >= 4
        or any(marker in lowered for marker in parallel_markers)
    ):
        return "parallel"

    multi_action_markers = (
        " and also ",
        " as well as ",
        "refactor",
        "architecture",
        "migration",
        "redesign",
        "multiple files",
    )
    localized_target_count = len(file_refs | repo_attachments)
    if (
        cfg.fast_path_enabled
        and len(request) <= 360
        and localized_target_count <= 1
        and bullet_count <= 1
        and not any(marker in lowered for marker in multi_action_markers)
    ):
        return "simple"

    return "standard"


def _explicit_request_paths(state: CodingAgentState) -> list[str]:
    request_paths = [
        match.group("path").replace("\\", "/")
        for match in _FILE_REFERENCE_RE.finditer(state.get("user_request", ""))
    ]
    return dedupe(filter_context_paths([*request_paths, *_repo_attachment_paths(state)]))


def _deterministic_search_requests(state: CodingAgentState) -> list[dict[str, Any]]:
    request = state.get("user_request", "")
    paths = _explicit_request_paths(state)

    stop_words = {
        "about", "after", "agent", "before", "change", "coding", "create",
        "file", "fix", "from", "help", "implement", "improve", "make",
        "please", "request", "task", "that", "this", "through", "update",
        "using", "want", "with",
    }
    terms: list[str] = []
    for token in _TOKEN_RE.findall(request):
        lowered = token.lower()
        if lowered in stop_words or lowered.isdigit():
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= 8:
            break

    path_includes = []
    extensions = []
    for raw_path in paths:
        normalized = raw_path.replace("\\", "/").strip("/")
        if "/" in normalized:
            parent = normalized.rsplit("/", 1)[0]
            if parent and parent not in path_includes:
                path_includes.append(parent)
        suffix = Path(normalized).suffix.lower()
        if suffix and suffix not in extensions:
            extensions.append(suffix)

    return [
        {
            "terms": terms,
            "path_includes": path_includes[:4],
            "path_excludes": ["agents/coding/logs/runs"],
            "file_extensions": extensions[:6],
            "mode": "any",
            "max_results": 20,
        }
    ]


def route_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    cfg = _settings_for_state(state, cfg)
    registry = SkillRegistry().load()
    task_mode = _classify_task_mode(state, cfg)

    try:
        default_skill = registry.default_skill_name()
    except ValueError as exc:
        return {
            "task_mode": task_mode,
            "selected_skill": "",
            "selected_skills": [],
            "selected_skill_tools": [],
            "skill_instructions": "",
            "route_confidence": 0.0,
            "route_reason": str(exc),
            "route_alternatives": [],
            "errors": [*state.get("errors", []), str(exc)],
            "status": "route_failed",
        }

    deterministic_skills = registry.rank_for_request(
        state["user_request"],
        max_skills=MAX_SELECTED_SKILLS,
    )

    def build_payload(
        selected_names: list[str],
        *,
        confidence: float,
        reason: str,
        alternatives: list[dict[str, str]] | None = None,
    ) -> CodingAgentState:

        valid_names = [
            name for name in dict.fromkeys(selected_names)
            if registry.has(name)
        ][:MAX_SELECTED_SKILLS]

        if not valid_names:
            valid_names = [default_skill]
        return {
            "task_mode": task_mode,
            # Preserve the old scalar field so existing API/UI code does not break.
            "selected_skill": valid_names[0],
            "selected_skills": valid_names,
            "selected_skill_tools": registry.allowed_tools_for(valid_names),
            "skill_instructions": registry.combined_instructions(valid_names),
            "route_confidence": confidence,
            "route_reason": reason,
            "route_alternatives": alternatives or [],
            "status": "routed",
        }

    # The deterministic router can now return complementary skills and rank custom
    # skills by name/purpose overlap. Keep the LLM route opt-in for installations
    # that want stronger semantic matching without making every run pay that latency.
    if not cfg.llm_skill_routing_enabled or task_mode == "simple":
        return build_payload(
            deterministic_skills,
            confidence=0.92 if len(deterministic_skills) == 1 else 0.86,
            reason=(
                "Deterministic low-latency multi-skill route."
                if len(deterministic_skills) > 1
                else "Deterministic low-latency skill route."
            ),
        )

    try:
        decision: SkillRouteDecision = invoke_parsed_decision(
            model=_coding_node_model(cfg.route_max_tokens, cache_namespace="route"),
            schema=SkillRouteDecision,
            node_name="route",
            state=state,
            system_prompt=SKILL_ROUTER_SYSTEM_PROMPT,
            user_prompt=build_skill_router_user_prompt(
                request=state["user_request"],
                skill_catalog=registry.router_catalog(),
            ),
            max_attempts=1,
        )
    except Exception as exc:
        return build_payload(
            deterministic_skills,
            confidence=0.5,
            reason=f"LLM skill routing failed; used deterministic route: {exc}",
        )

    selected_names = [
        name.strip()
        for name in decision.selected_skills
        if name.strip() and registry.has(name.strip())
    ]
    selected_names = list(dict.fromkeys(selected_names))[:MAX_SELECTED_SKILLS]

    if not selected_names:
        selected_names = deterministic_skills
        route_reason = f"LLM selected no known skills; used deterministic route. {decision.reason}"
        confidence = 0.0
    else:
        route_reason = decision.reason
        confidence = decision.confidence

    alternatives = [
        {"skill_name": item.skill_name, "reason": item.reason}
        for item in decision.alternatives
        if registry.has(item.skill_name) and item.skill_name not in selected_names
    ][:3]
    return build_payload(
        selected_names,
        confidence=confidence,
        reason=route_reason,
        alternatives=alternatives,
    )



def _fallback_subtasks(
    state: CodingAgentState,
    search_requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": "primary",
            "objective": state.get("user_request", "Gather the implementation context."),
            "search_requests": search_requests,
            "candidate_paths": _explicit_request_paths(state),
        }
    ]


def plan_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    cfg = _settings_for_state(state, cfg)
    request = state["user_request"]
    task_mode = state.get("task_mode") or _classify_task_mode(state, cfg)
    deterministic_search = _deterministic_search_requests(state)
    approved_custom_tool_registry = ApprovedCustomToolRegistry().load()
    selected_tool_names = list(state.get("selected_skill_tools") or [])
    has_selected_custom_tools = any(
        approved_custom_tool_registry.has(name) for name in selected_tool_names
    )

    # The simple fast path skips the planner. Promote the run to standard when a
    # selected skill exposes an approved custom tool so the planner gets a chance
    # to decide whether and how to call it.
    if task_mode == "simple" and has_selected_custom_tools:
        task_mode = "standard"

    if task_mode == "simple":
        return {
            "task_mode": "simple",
            "plan": [
                "Load the named or attached implementation file and only the nearest required context.",
                "Produce the smallest exact patch.",
                "Run targeted validation when write mode is enabled.",
            ],
            "search_requests": deterministic_search,
            "search_queries": [],
            "subtasks": _fallback_subtasks(state, deterministic_search),
            "custom_tool_calls": [],
            "validation_commands": state.get("validation_commands", []),
            "status": "planned",
        }

    planner_prompt = build_planner_user_prompt(request)
    selected_skills = list(state.get("selected_skills") or [])

    if not selected_skills and state.get("selected_skill"):
        selected_skills = [str(state.get("selected_skill"))]

    if selected_skills:
        planner_prompt += (
            "\n\nSelected skills in priority order:\n"
            f"{bullets(selected_skills)}\n\n"
            "Combined skill guidance for planning:\n"
            f"{skill_instructions_for_llm(state.get('skill_instructions', ''))}"
        )

    approved_custom_tool_catalog = approved_custom_tool_registry.prompt_catalog(selected_tool_names)
    if approved_custom_tool_catalog:
        planner_prompt += (
            "\n\nApproved custom tools available for this run:\n"
            f"{approved_custom_tool_catalog}\n\n"
            "Use custom_tool_calls only when one of these tools materially improves "
            "repository inspection. Tool arguments must be JSON-compatible and must "
            "not include repo_root."
        )

    memories = state.get("long_term_memories", [])
    if memories:
        planner_prompt += (
            "\n\nRelevant long-term coding memories:\n" + bullets(memories[:5])
        )
    attachment_summary = _attached_file_summary(state)
    if attachment_summary:
        planner_prompt += (
            "\n\nUser-attached files available as read-only context:\n"
            f"{attachment_summary}"
        )

    try:
        decision: PlanDecision = invoke_parsed_decision(
            model=_coding_node_model(cfg.planner_max_tokens, cache_namespace="plan"),
            schema=PlanDecision,
            node_name="plan",
            state=state,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=planner_prompt,
            max_attempts=1,
        )
        search_requests = _planned_search_requests(decision, request) or deterministic_search
        subtasks = [item.model_dump() for item in decision.subtasks[: cfg.max_context_workers]]
        if not subtasks:
            subtasks = _fallback_subtasks(state, search_requests)

        effective_mode = task_mode
        if task_mode != "parallel" and decision.task_mode == "parallel" and len(subtasks) > 1:
            effective_mode = "parallel"

        return {
            "task_mode": effective_mode,
            "plan": decision.plan,
            "search_requests": search_requests,
            "search_queries": decision.search_queries,
            "subtasks": subtasks,
            "custom_tool_calls": [
                item.model_dump() for item in decision.custom_tool_calls[:MAX_CUSTOM_TOOL_CALLS]
            ],
            "validation_commands": decision.validation_commands,
            "web_search_query": decision.web_search_query or "",
            "status": "planned",
        }
    except Exception as exc:
        return {
            "task_mode": task_mode,
            "plan": [
                "Search only the files and symbols directly related to the request.",
                "Gather bounded exact context with parallel read-only workers when useful.",
                "Create a minimal patch and run targeted validation.",
            ],
            "search_requests": deterministic_search,
            "search_queries": [],
            "subtasks": _fallback_subtasks(state, deterministic_search),
            "custom_tool_calls": [],
            "web_search_query": "",
            "errors": [
                *state.get("errors", []),
                f"LLM planning failed; used deterministic fallback plan: {exc}",
            ],
            "status": "planned",
        }



def custom_tools_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Execute approved custom inspection tools requested by the planner.

    Only tools explicitly exposed by the selected skills may run. Repository scope
    is injected by the runtime and cannot be overridden by model-generated args.
    """

    calls = list(state.get("custom_tool_calls") or [])[:MAX_CUSTOM_TOOL_CALLS]
    if not calls:
        return {"custom_tool_results": [], "status": "custom_tools_skipped"}

    cfg = _settings_for_state(state, cfg)
    repo_root = resolve_repo_root(state, cfg)
    allowed = set(state.get("selected_skill_tools") or [])
    custom_tool_registry = ApprovedCustomToolRegistry().load()
    results: list[dict[str, Any]] = []
    errors = list(state.get("errors", []))

    for raw_call in calls:
        tool_name = str(raw_call.get("tool_name", "")).strip()
        arguments = raw_call.get("arguments") or {}
        reason = str(raw_call.get("reason", "")).strip()

        if not tool_name:
            continue
        if tool_name not in allowed:
            errors.append(
                f"Skipped custom tool '{tool_name}': it is not allowed by the selected skills."
            )
            continue
        if not custom_tool_registry.has(tool_name):
            errors.append(
                f"Skipped custom tool '{tool_name}': it is not approved/available at runtime."
            )
            continue
        if not isinstance(arguments, dict):
            errors.append(f"Skipped custom tool '{tool_name}': arguments must be an object.")
            continue

        try:
            result = custom_tool_registry.invoke(
                tool_name,
                repo_root=repo_root,
                arguments=arguments,
            )
            result["reason"] = reason
            results.append(result)

        except Exception as exc:
            errors.append(f"Approved custom tool '{tool_name}' failed: {exc}")
            results.append(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "reason": reason,
                    "output": str(exc),
                    "truncated": False,
                    "success": False,
                }
            )

    if not results:
        status = "custom_tools_failed" if calls else "custom_tools_skipped"
        
    elif any(bool(item.get("success")) for item in results):
        status = "custom_tools_completed"
    else:
        status = "custom_tools_failed"

    return {
        "custom_tool_results": results,
        "errors": errors,
        "status": status,
    }


def repo_navigator_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Select files deterministically; use an LLM navigator only as an opt-in fallback."""

    cfg = _settings_for_state(state, cfg)
    repo_root = resolve_repo_root(state, cfg)
    errors = list(state.get("errors", []))
    search_requests = _search_requests_from_state(state) or _deterministic_search_requests(state)
    search_result_dicts = list(state.get("search_results") or [])

    requested_paths = [
        str(item.get("path", "")).strip()
        for item in state.get("requested_context", [])
        if str(item.get("path", "")).strip()
    ]
    explicit_paths = _explicit_request_paths(state)
    should_search = not (
        state.get("task_mode") == "simple" and bool(explicit_paths or requested_paths)
    )
    if not search_result_dicts and search_requests and should_search:
        try:
            results = search_repository(
                repo_root,
                search_requests,
                max_results=cfg.max_search_results,
            )
            search_result_dicts = [result.to_dict() for result in results]
        except Exception as exc:
            errors.append(f"Repo navigation search failed: {exc}")

    selected_paths = dedupe(
        filter_context_paths(
            [
                *explicit_paths,
                *requested_paths,
                *paths_from_ranked_results(search_result_dicts),
            ]
        )
    )[:MAX_REPO_NAVIGATION_FILES]

    navigation_summary = (
        "Selected files from explicit attachments, patcher context requests, and "
        "ranked structured search results."
    )
    confidence = 0.9 if selected_paths else 0.45
    missing_context: list[str] = []

    if not selected_paths and cfg.llm_navigation_enabled:
        try:
            root_files = filter_context_paths(list_files(repo_root, ".", max_depth=6))[
                : cfg.max_search_results
            ]
            decision: RepoNavigationDecision = invoke_parsed_decision(
                model=_coding_node_model(
                    cfg.repo_navigation_max_tokens,
                    cache_namespace="repo-navigation",
                ),
                schema=RepoNavigationDecision,
                node_name="repo_navigator",
                state=state,
                system_prompt=REPO_NAVIGATOR_SYSTEM_PROMPT,
                user_prompt=build_repo_navigator_user_prompt(
                    request=state["user_request"],
                    selected_skill=", ".join(state.get("selected_skills") or [state.get("selected_skill", "")]),
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
                max_attempts=1,
            )
            selected_paths = dedupe(
                filter_context_paths([item.path for item in decision.files_to_inspect])
            )[:MAX_REPO_NAVIGATION_FILES]
            navigation_summary = decision.task_summary or navigation_summary
            confidence = decision.confidence
            missing_context = decision.missing_context
        except Exception as exc:
            errors.append(f"Optional LLM repo navigator failed: {exc}")

    navigation_files = [
        {
            "path": path,
            "reason": (
                "Explicit file/context request."
                if path in set(_explicit_request_paths(state) + requested_paths)
                else "Ranked structured-search result."
            ),
        }
        for path in selected_paths
    ]

    return {
        "context_generation": int(state.get("context_generation", 0)) + 1,
        "search_requests": search_requests,
        "search_results": search_result_dicts,
        "repo_navigation_summary": navigation_summary,
        "repo_navigation_files": navigation_files,
        "repo_navigation_confidence": confidence,
        "repo_navigation_missing_context": missing_context,
        "repo_navigation_search_requests": [],
        "errors": errors,
        # Upload-only tasks can still proceed even when no repository path was found.
        "status": "repo_navigated",
    }


def assign_context_workers(state: CodingAgentState) -> list[Send]:
    """Fan out isolated read-only context workers with the LangGraph Send API."""

    subtasks = list(state.get("subtasks") or [])
    if not subtasks:
        subtasks = _fallback_subtasks(
            state,
            list(state.get("search_requests") or _deterministic_search_requests(state)),
        )

    cfg = _settings_for_state(state)
    max_workers = max(1, cfg.max_context_workers)
    subtasks = subtasks[:max_workers]
    nav_paths = [
        str(item.get("path", "")).strip()
        for item in state.get("repo_navigation_files", [])
        if str(item.get("path", "")).strip()
    ]

    worker_payloads: list[Send] = []
    for index, raw_subtask in enumerate(subtasks):
        subtask = dict(raw_subtask)
        assigned = [
            path for path_index, path in enumerate(nav_paths)
            if path_index % len(subtasks) == index
        ]
        subtask["candidate_paths"] = dedupe(
            [*subtask.get("candidate_paths", []), *assigned]
        )
        worker_payloads.append(
            Send(
                "context_worker",
                {
                    "repo_root": state.get("repo_root", ""),
                    "user_request": state.get("user_request", ""),
                    "task_mode": state.get("task_mode", "standard"),
                    "runtime_settings": state.get("runtime_settings", {}),
                    "active_subtask": subtask,
                    "search_requests": state.get("search_requests", []),
                    "requested_context": (
                        state.get("requested_context", []) if index == 0 else []
                    ),
                    "context_generation": state.get("context_generation", 0),
                },
            )
        )

    return worker_payloads


def _search_terms_from_worker_state(state: dict[str, Any]) -> list[str]:
    subtask = state.get("active_subtask") or {}
    values = [str(subtask.get("objective", "")), str(state.get("user_request", ""))]
    for request in subtask.get("search_requests", []) or state.get("search_requests", []):
        if isinstance(request, dict):
            values.extend(str(term) for term in request.get("terms", []))
    terms: list[str] = []
    for value in values:
        for token in _TOKEN_RE.findall(value):
            if len(token) < 3 or token.lower() in {"the", "and", "with", "from", "this"}:
                continue
            if token.lower() not in {item.lower() for item in terms}:
                terms.append(token)
            if len(terms) >= 12:
                return terms
    return terms


def _line_offsets(lines: list[str]) -> list[int]:
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    return offsets


def _chunk_ranges_for_text(
    text: str,
    *,
    terms: Iterable[str],
    requested_ranges: list[tuple[int, int]],
    cfg: CodingAgentSettings,
) -> list[tuple[int, int, int, int]]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    offsets = _line_offsets(lines)
    ranges: list[tuple[int, int]] = []

    for start_line, end_line in requested_ranges:
        start = max(1, start_line)
        end = min(len(lines), max(start, end_line))
        ranges.append((offsets[start - 1], offsets[end]))

    lowered = text.lower()
    for term in terms:
        needle = term.strip().lower()
        if len(needle) < 3:
            continue
        start_at = 0
        while len(ranges) < 8:
            hit = lowered.find(needle, start_at)
            if hit < 0:
                break
            half = cfg.context_chunk_chars // 2
            ranges.append((max(0, hit - half), min(len(text), hit + half)))
            start_at = hit + len(needle)

    if not ranges:
        head = min(len(text), cfg.context_chunk_chars)
        ranges.append((0, head))

    # Expand to line boundaries, merge overlap, and keep the most useful first windows.
    normalized: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        start = max(0, start - cfg.context_chunk_overlap_chars)
        end = min(len(text), end + cfg.context_chunk_overlap_chars)
        while start > 0 and text[start - 1] != "\n":
            start -= 1
        while end < len(text) and text[end - 1 : end] != "\n":
            end += 1
        if normalized and start <= normalized[-1][1]:
            normalized[-1] = (normalized[-1][0], max(normalized[-1][1], end))
        else:
            normalized.append((start, end))

    output: list[tuple[int, int, int, int]] = []
    used = 0
    per_file_budget = max(cfg.max_full_file_chars, cfg.context_chunk_chars)
    for start, end in normalized:
        if used >= per_file_budget:
            break
        end = min(end, start + (per_file_budget - used))
        start_line = text.count("\n", 0, start) + 1
        end_line = text.count("\n", 0, end) + 1
        output.append((start, end, start_line, end_line))
        used += end - start
    return output


def _requested_ranges_for_path(
    requested_context: list[dict[str, Any]],
    path: str,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for item in requested_context:
        if str(item.get("path", "")).strip() != path:
            continue
        start = item.get("start_line")
        end = item.get("end_line")
        if isinstance(start, int):
            ranges.append((start, end if isinstance(end, int) else start + 120))
    return ranges


def _format_file_context(
    *,
    path: str,
    text: str,
    terms: list[str],
    requested_context: list[dict[str, Any]],
    cfg: CodingAgentSettings,
) -> str:
    if len(text) <= cfg.max_full_file_chars:
        return (
            f"File: {path}\nContent-Status: complete\n"
            f"```\n{text}\n```"
        )

    ranges = _chunk_ranges_for_text(
        text,
        terms=terms,
        requested_ranges=_requested_ranges_for_path(requested_context, path),
        cfg=cfg,
    )
    blocks = [
        f"File: {path}\nContent-Status: selected-chunks ({len(text)} total characters)"
    ]
    for start, end, start_line, end_line in ranges:
        blocks.append(
            f"Chunk-Lines: {start_line}-{end_line}\n```\n{text[start:end]}\n```"
        )
    return "\n".join(blocks)


def context_worker_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Search and read one focused context slice. Workers never call an LLM or edit."""

    cfg = _settings_for_state(state, cfg)
    repo_root = resolve_repo_root(state, cfg)
    subtask = dict(state.get("active_subtask") or {})
    worker_id = str(subtask.get("id") or "context")
    search_requests = subtask.get("search_requests") or state.get("search_requests", [])
    errors: list[str] = []
    paths = filter_context_paths(list(subtask.get("candidate_paths") or []))

    should_search = not (state.get("task_mode") == "simple" and bool(paths))
    if search_requests and should_search:
        try:
            results = search_repository(
                repo_root,
                search_requests,
                max_results=max(10, cfg.max_worker_files * 4),
            )
            paths.extend(paths_from_ranked_results([item.to_dict() for item in results]))
        except Exception as exc:
            errors.append(f"Worker {worker_id} search failed: {exc}")

    for item in state.get("requested_context", []):
        path = str(item.get("path", "")).strip()
        if path:
            paths.append(path)

    paths = dedupe(filter_context_paths(paths))[: cfg.max_worker_files]
    terms = _search_terms_from_worker_state(state)
    context_blocks: list[str] = []
    inspected: list[str] = []

    for path in paths:
        try:
            probe = read_file(repo_root, path, max_chars=cfg.max_file_chars + 1)
            if len(probe) > cfg.max_file_chars:
                errors.append(
                    f"Skipped {path}: file exceeds the configured read ceiling of "
                    f"{cfg.max_file_chars} characters."
                )
                continue
            context_blocks.append(
                _format_file_context(
                    path=path,
                    text=probe,
                    terms=terms,
                    requested_context=state.get("requested_context", []),
                    cfg=cfg,
                )
            )
            inspected.append(path)
        except Exception as exc:
            errors.append(f"Worker {worker_id} could not read {path}: {exc}")

    result = {
        "generation": int(state.get("context_generation", 0)),
        "worker_id": worker_id,
        "objective": str(subtask.get("objective", "")),
        "context_blocks": context_blocks,
        "files_inspected": inspected,
        "errors": errors,
    }
    # This is the only state key written by fan-out workers; it has an append reducer.
    return {"context_worker_results": [result]}


def _compact_upload_context(
    state: CodingAgentState,
    cfg: CodingAgentSettings,
) -> tuple[list[str], list[str], list[str]]:
    blocks: list[str] = []
    used: list[str] = []
    errors: list[str] = []
    terms = _search_terms_from_worker_state(state)

    for item in state.get("attached_files", []):
        if item.get("source") == "repo":
            continue
        name = str(item.get("name", "attachment")).strip() or "attachment"
        content = str(item.get("content", ""))
        if not content:
            continue
        if item.get("truncated"):
            errors.append(
                f"Attachment {name} arrived already truncated by the client or an older API."
            )
        if len(content) <= cfg.max_full_file_chars:
            block = f"Attachment: {name}\nContent-Status: complete\n```\n{content}\n```"
        else:
            ranges = _chunk_ranges_for_text(
                content,
                terms=terms,
                requested_ranges=[],
                cfg=cfg,
            )
            parts = [
                f"Attachment: {name}\nContent-Status: selected-chunks "
                f"({len(content)} total characters)"
            ]
            for start, end, start_line, end_line in ranges:
                parts.append(
                    f"Chunk-Lines: {start_line}-{end_line}\n```\n{content[start:end]}\n```"
                )
            block = "\n".join(parts)
        blocks.append(block)
        used.append(name)
    return blocks, used, errors


def gather_context_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Fan-in worker results and build a bounded, high-signal patch prompt."""

    cfg = _settings_for_state(state, cfg)
    generation = int(state.get("context_generation", 0))
    worker_results = [
        item
        for item in state.get("context_worker_results", [])
        if int(item.get("generation", -1)) == generation
    ]
    errors = list(state.get("errors", []))
    files_inspected: list[str] = []
    candidate_blocks: list[str] = []

    selected_paths = [
        str(item.get("path", "")).strip()
        for item in state.get("repo_navigation_files", [])
        if str(item.get("path", "")).strip()
    ]
    candidate_blocks.append(
        "# Execution context\n"
        f"Task mode: {state.get('task_mode', 'standard')}\n"
        f"Selected skills: {', '.join(state.get('selected_skills') or [state.get('selected_skill', 'none')])}\n"
        f"Plan:\n{bullets(state.get('plan', []))}\n"
        f"Selected repository paths:\n{bullets(selected_paths)}"
    )

    for tool_result in state.get("custom_tool_results", []):
        tool_name = str(tool_result.get("tool_name", "custom_tool"))
        if tool_result.get("success"):
            candidate_blocks.append(
                f"# Approved custom tool result: {tool_name}\n"
                f"Reason: {tool_result.get('reason', '')}\n"
                f"{tool_result.get('output', '')}"
            )
        else:
            errors.append(
                f"Custom tool {tool_name} did not produce usable context: "
                f"{tool_result.get('output', '')}"
            )

    for result in worker_results:
        worker_errors = [str(item) for item in result.get("errors", [])]
        errors.extend(worker_errors)
        files_inspected.extend(str(path) for path in result.get("files_inspected", []))
        blocks = result.get("context_blocks", [])
        if blocks:
            candidate_blocks.append(
                f"# Context worker: {result.get('worker_id', 'worker')}\n"
                f"Objective: {result.get('objective', '')}"
            )
            candidate_blocks.extend(str(block) for block in blocks)

    upload_blocks, attached_files_used, upload_errors = _compact_upload_context(state, cfg)
    candidate_blocks.extend(upload_blocks)
    errors.extend(upload_errors)

    memories = state.get("long_term_memories", [])
    if memories:
        candidate_blocks.append(
            "# Relevant prior coding memories\n" + bullets(memories[:5])
        )
    if state.get("web_search_results"):
        candidate_blocks.append(
            "# Web search results\n" + str(state.get("web_search_results"))[:8_000]
        )
    loop_focus = _format_loop_context_focus(state)
    if loop_focus:
        candidate_blocks.append("# Retry focus\n" + loop_focus)

    # Never silently truncate a file block. Include whole selected blocks until the
    # prompt budget is reached, then report what was omitted so the patcher can ask
    # for a narrower range on its next pass.
    context: list[str] = []
    used_chars = 0
    for block in candidate_blocks:
        if used_chars + len(block) > cfg.max_context_prompt_chars:
            errors.append(
                "Context prompt budget reached; omitted lower-priority blocks. "
                "The patcher can request exact file ranges if needed."
            )
            continue
        context.append(block)
        used_chars += len(block)

    files_inspected = dedupe(files_inspected)
    has_usable_context = bool(context and (files_inspected or attached_files_used))
    return {
        "context": context,
        "files_inspected": files_inspected,
        "attached_files_used": attached_files_used,
        "errors": errors,
        "status": "context_gathered" if has_usable_context else "context_failed",
    }


def patch_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Ask the LLM for exact edits and apply them when writes are enabled."""
    
    cfg = _settings_for_state(state, cfg)
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

    use_fast_patch_model = (
        cfg.fast_path_enabled
        and state.get("task_mode") == "simple"
        and patch_attempts == 1
    )
    patch_prompt = build_patcher_user_prompt(
        request=state["user_request"],
        selected_skill=", ".join(state.get("selected_skills") or [state.get("selected_skill", "")]),
        skill_instructions=skill_instructions_for_llm(
            state.get("skill_instructions", "")
        ),
        plan=bullets(state.get("plan", [])),
        context=build_patch_context(state),
    )
    patch_state = {**state, "patch_attempts": patch_attempts}
    patch_model = (
        _coding_node_model(cfg.simple_patch_max_tokens, cache_namespace="patch")
        if use_fast_patch_model
        else _reasoning_node_model(cfg.patch_max_tokens, cache_namespace="patch")
    )

    try:
        decision: PatchDecision = invoke_parsed_decision(
            model=patch_model,
            schema=PatchDecision,
            node_name="patch_fast" if use_fast_patch_model else "patch",
            state=patch_state,
            system_prompt=PATCHER_SYSTEM_PROMPT,
            user_prompt=patch_prompt,
            max_attempts=1 if use_fast_patch_model else 2,
        )
    except Exception as exc:
        if use_fast_patch_model:
            errors.append(
                f"Fast patch model failed; escalated to reasoning model: {exc}"
            )
            try:
                decision = invoke_parsed_decision(
                    model=_reasoning_node_model(
                        cfg.patch_max_tokens,
                        cache_namespace="patch",
                    ),
                    schema=PatchDecision,
                    node_name="patch_escalated",
                    state=patch_state,
                    system_prompt=PATCHER_SYSTEM_PROMPT,
                    user_prompt=patch_prompt,
                    max_attempts=2,
                )
                use_fast_patch_model = False
            except Exception as escalated_exc:
                return {
                    "patch_attempts": patch_attempts,
                    "max_patch_attempts": max_patch_attempts,
                    "patch_summary": f"LLM patching failed: {escalated_exc}",
                    "errors": [
                        *errors,
                        f"LLM patching failed on attempt {patch_attempts}: {escalated_exc}",
                    ],
                    "status": "patch_failed",
                }
        else:
            return {
                "patch_attempts": patch_attempts,
                "max_patch_attempts": max_patch_attempts,
                "patch_summary": f"LLM patching failed: {exc}",
                "errors": [*errors, f"LLM patching failed on attempt {patch_attempts}: {exc}"],
                "status": "patch_failed",
            }
    

    requested_context = [item.model_dump() for item in decision.context_requests]

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
                    existing = read_file(repo_root, path, max_chars=cfg.max_file_chars)
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
                before = read_file(repo_root, path, max_chars=cfg.max_file_chars)
                after = apply_exact_replace(before, edit.old, edit.new, path=path)

            elif edit.operation == "full_file_replace":
                before = read_file(repo_root, path, max_chars=cfg.max_file_chars)
                after = edit.new

            elif edit.operation in {"insert_after", "insert_before"}:
                if not edit.old:
                    raise ValueError(
                        f"{edit.operation} for {path} requires a non-empty anchor in old."
                    )
                before = read_file(repo_root, path, max_chars=cfg.max_file_chars)
                replacement = (
                    edit.old + edit.new
                    if edit.operation == "insert_after"
                    else edit.new + edit.old
                )
                after = apply_exact_replace(before, edit.old, replacement, path=path)

            elif edit.operation == "append":
                before = read_file(repo_root, path, max_chars=cfg.max_file_chars)
                after = before + edit.new

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

    should_retry_patch = (
        status == "patch_failed"
        or (status == "patch_skipped" and bool(requested_context))
    )
    if should_retry_patch and patch_attempts < max_patch_attempts:
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

        context_request_detail = ""
        if requested_context:
            context_request_detail = (
                " Exact context requested: "
                + "; ".join(
                    f"{item.get('path')} lines {item.get('start_line') or '?'}-"
                    f"{item.get('end_line') or '?'} ({item.get('reason') or 'no reason'})"
                    for item in requested_context
                )
            )
        patch_retry_focus = (
            f"Patch attempt {patch_attempts} {retry_reason}. {retry_detail} "
            f"Patcher summary: {decision.summary or '(none)'}.{context_request_detail}"
        )

        patch_retry_fields = {
            "requested_context": requested_context,
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
        "requested_context": (
            requested_context
            if patch_retry_fields
            else []
        ),
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
    cfg = _settings_for_state(state, cfg)
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
    
    cfg = _settings_for_state(state, cfg)
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
            model=_reasoning_node_model(cfg.progress_max_tokens, cache_namespace="progress"),
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
                "subtasks": [],
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
        "subtasks": [] if should_continue else state.get("subtasks", []),
        "repo_navigation_summary": "" if should_continue else state.get("repo_navigation_summary", ""),
        "repo_navigation_files": [] if should_continue else state.get("repo_navigation_files", []),
        "repo_navigation_missing_context": [],
        "context": [] if should_continue else state.get("context", []),
        "files_inspected": [] if should_continue else state.get("files_inspected", []),
        "status": "assessed",
    }






def report_node(state: CodingAgentState) -> CodingAgentState:
    validation_results = state.get("validation_results", [])
    validation_lines = [
        f"- `{item.get('command', 'unknown')}` -> exit code "
        f"{item.get('returncode', 'unknown')}"
        for item in validation_results
    ]
    all_errors = [*state.get("errors", []), *state.get("memory_errors", [])]
    changed = [
        item.get("path", "") + " - " + item.get("write_result", "")
        for item in state.get("file_changes", [])
    ]
    current_generation = int(state.get("context_generation", 0))
    worker_count = sum(
        1
        for item in state.get("context_worker_results", [])
        if int(item.get("generation", -1)) == current_generation
    )

    report = f"""Coding agent run summary

Request:
{state.get('user_request', '')}

Execution mode:
{state.get('task_mode', 'standard')} ({worker_count} context worker(s))

Execution profile:
{state.get('runtime_settings', {})}

Selected skills:
{bullets(state.get('selected_skills') or [state.get('selected_skill', 'none')])}

Plan:
{bullets(state.get('plan', []))}

Approved custom tools used:
{bullets([
    item.get('tool_name', '') + (' (ok)' if item.get('success') else ' (failed)')
    for item in state.get('custom_tool_results', [])
]) if state.get('custom_tool_results') else 'None'}

Files inspected:
{bullets(state.get('files_inspected', []))}

Files changed/proposed:
{bullets(changed)}

Patch:
{state.get('patch_summary', 'No patch summary generated.')}

Validation:
{chr(10).join(validation_lines) if validation_lines else 'No validation commands were run.'}

Errors:
{bullets(all_errors) if all_errors else 'None'}
""".strip()
    return {"report": report, "status": "reported"}


def web_search_node(state: CodingAgentState) -> CodingAgentState:
    """
    Perform web search when a dynamic query is present or when the web_search
    skill is selected. Clears the dynamic query after search.
    """
    dynamic_query = (state.get("web_search_query") or "").strip()

    if dynamic_query:
        query = dynamic_query
    elif "web_search" in set(state.get("selected_skills") or [state.get("selected_skill", "")]):
        query = state.get("user_request", "")
    else:
        return {"status": "web_search_skipped"}

    if not query:
        return {
            "web_search_results": "",
            "status": "web_search_skipped",
        }

    try:
        results = web_search(query, num_results=5)
        return {
            "web_search_results": results,
            "web_search_query": "",
            "status": "web_search_completed",
        }
    except Exception as exc:
        return {
            "web_search_results": f"Web search failed: {exc}",
            "web_search_query": "",
            "errors": [*state.get("errors", []), f"Web search failed: {exc}"],
            "status": "web_search_failed",
        }




def gmail_access_node(state: CodingAgentState) -> CodingAgentState:
    """
    Perform Gmail access if the selected skill is gmail_access.
    Currently a placeholder.
    """
    if "gmail_access" not in set(state.get("selected_skills") or [state.get("selected_skill", "")]):
        return {"status": "gmail_access_skipped"}
    # Placeholder: log that gmail access was triggered
    # In future, invoke gmail API here.
    return {"status": "gmail_access_completed"}
