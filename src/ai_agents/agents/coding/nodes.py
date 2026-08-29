from __future__ import annotations


from dataclasses import replace
import math
from pathlib import Path
from uuid import uuid4
import re
from typing import Any, Iterable, Literal
from ai_agents.agents.coding.utils.constants import (
    MAX_REPO_NAVIGATION_FILES,
    VALIDATION_PROFILE_NAME,
)

from ai_agents.agents.coding.llm import invoke_parsed_decision
from ai_agents.agents.coding.memory import recall_coding_memories, remember_coding_run
from ai_agents.agents.coding.prompts import (
    PLANNER_SYSTEM_PROMPT,
    REPO_NAVIGATOR_SYSTEM_PROMPT,
    SKILL_ROUTER_SYSTEM_PROMPT,
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
from ai_agents.agents.coding.runtime import repo_root as resolve_repo_root
from ai_agents.agents.coding.coding_agent_schemas import (
    PlanDecision,
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
from ai_agents.agents.coding.tools.filesystem import list_files, read_file
from ai_agents.agents.coding.tools.web_search import web_search
from ai_agents.agents.coding.tools.search import search_repository

from ai_agents.agents.coding.utils.validation import (
    advisory_validation_failures,
    blocking_validation_failures,
    default_validation_commands,
)


from ai_agents.agents.coding.utils.helpers import(
    _dump_search_request,
    _planned_search_requests,
    _search_requests_from_state, 
    _format_search_result_dicts,
    _repo_attachment_paths,
    _attached_file_summary,
    _resolve_existing_repo_path,
    _format_loop_context_focus,
)


_RUNTIME_SETTING_FIELDS = {
    "max_subtask_workers",
    "max_context_workers",
    "route_max_tokens",
    "planner_max_tokens",
    "repo_navigation_max_tokens",
    "simple_patch_max_tokens",
    "patch_max_tokens",
    "progress_max_tokens",
    "max_implementation_units",
    "max_patch_retries_per_unit",
    "max_implementation_iterations",
    "context_prompt_base_tokens",
    "max_context_prompt_tokens",
    "context_prompt_reserve_tokens",
    "context_window_safety_tokens",
    "coding_model_context_window_tokens",
    "reasoning_model_context_window_tokens",
    "coding_model_max_output_tokens",
    "reasoning_model_max_output_tokens",
    "reconciliation_max_tokens",
    "reconciliation_context_max_tokens",
    "max_reasoning_reconciliations",
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
    r"(?P<path>[A-Za-z0-9_][A-Za-z0-9_./()\-]*\.(?:py|tsx?|jsx?|css|html|json|md|sql|pls|fex|toml|ya?ml))",
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


def _clean_repo_path_reference(value: str) -> str:
    """Normalize path text emitted by users/models without changing repository scope."""

    candidate = str(value or "").strip().replace("\\", "/")
    candidate = candidate.strip("`\"'")
    candidate = candidate.lstrip("([{<,;:").rstrip(")]}>,;:")
    candidate = re.sub(r"/+", "/", candidate)

    while candidate.startswith("./"):
        candidate = candidate[2:]
    return candidate.strip()


def _repo_path_variants(candidate: str, repo_files: list[str]) -> list[str]:
    """Return safe repo-relative variants for common duplicated-root handoffs.

    The coding API already runs with repo_root at ``.../src/ai_agents``. Voice/model
    handoffs sometimes repeat ``src/ai_agents`` or ``ai_agents`` in front of an
    otherwise valid repository path. Only strip leading segments when the remaining
    path starts with a top-level directory that actually exists in the repository.
    """

    normalized = _clean_repo_path_reference(candidate)
    if not normalized:
        return []

    path = Path(normalized)
    if path.is_absolute() or normalized.startswith("/") or ".." in path.parts:
        return []

    normalized = normalized.lstrip("/")

    variants = [normalized]
    top_levels = {item.split("/", 1)[0] for item in repo_files if item}

    parts = [part for part in normalized.split("/") if part]
    for index in range(1, len(parts)):
        if parts[index] not in top_levels:
            continue
        variant = "/".join(parts[index:])
        if variant and variant not in variants:
            variants.append(variant)

    return variants


def _resolve_repo_file_reference(
    *,
    repo_root: Path,
    candidate: str,
    repo_files: list[str],
) -> str | None:
    """Resolve a loose reference to one canonical repository file."""

    for variant in _repo_path_variants(candidate, repo_files):
        resolved = _resolve_existing_repo_path(
            repo_root=repo_root,
            candidate=variant,
            repo_files=repo_files,
        )
        if resolved:
            return resolved
    return None


def _resolve_repo_directory_reference(
    *,
    repo_root: Path,
    candidate: str,
    repo_files: list[str],
) -> str | None:
    """Resolve a loose reference to a canonical repository directory."""

    root = repo_root.resolve()
    known_directories: set[str] = set()
    for repo_file in repo_files:
        parts = Path(repo_file).parts[:-1]
        for index in range(1, len(parts) + 1):
            known_directories.add(Path(*parts[:index]).as_posix())

    for variant in _repo_path_variants(candidate, repo_files):
        target = (root / variant).resolve()
        if root not in target.parents and target != root:
            continue
        if target.is_dir():
            return target.relative_to(root).as_posix() or "."
        if variant in known_directories:
            return variant
    return None


def _rank_directory_context_files(
    *,
    directory: str,
    repo_files: list[str],
    terms: Iterable[str] = (),
    limit: int = 4,
) -> list[str]:
    """Choose a small useful file sample when a model requests a directory."""

    prefix = directory.rstrip("/") + "/"
    candidates = [path for path in repo_files if path.startswith(prefix)]
    if not candidates:
        return []

    needles = [
        token.lower()
        for value in terms
        for token in _TOKEN_RE.findall(str(value))
        if len(token) >= 3
    ]

    def score(path: str) -> tuple[int, int, int, str]:
        lowered = path.lower()
        term_hits = sum(1 for term in needles if term in lowered)
        direct_depth = path[len(prefix):].count("/")
        test_bonus = 1 if ("test" in lowered or "spec" in lowered) else 0
        return (-term_hits, -test_bonus, direct_depth, path)

    return sorted(candidates, key=score)[:limit]


def _explicit_request_paths(state: CodingAgentState) -> list[str]:
    attachment_paths = dedupe(filter_context_paths(_repo_attachment_paths(state)))
    attachment_basenames = {Path(path).name for path in attachment_paths}

    request_paths: list[str] = []
    for match in _FILE_REFERENCE_RE.finditer(state.get("user_request", "")):
        candidate = _clean_repo_path_reference(match.group("path"))
        if not candidate:
            continue

        # A bare filename repeated in prose is weaker evidence than an exact repo
        # attachment path. This also prevents ambiguous names such as tool_registry.py
        # from generating a false resolution error when both coding/voice versions
        # were already attached canonically.
        if "/" not in candidate and Path(candidate).name in attachment_basenames:
            continue
        request_paths.append(candidate)

    return dedupe(filter_context_paths([*attachment_paths, *request_paths]))


def _resolve_context_paths(
    *,
    repo_root: Path,
    candidate_paths: Iterable[str],
    repo_files: list[str] | None = None,
    max_depth: int = 12,
    directory_terms: Iterable[str] = (),
    directory_file_limit: int = 4,
) -> tuple[list[str], list[str]]:
    """Resolve loose file or directory references to canonical repo-relative files."""

    if repo_files is None:
        repo_files = filter_context_paths(list_files(repo_root, ".", max_depth=max_depth))

    resolved: list[str] = []
    unresolved: list[str] = []

    for raw_candidate in candidate_paths:
        candidate = _clean_repo_path_reference(str(raw_candidate))
        if not candidate:
            continue

        resolved_path = _resolve_repo_file_reference(
            repo_root=repo_root,
            candidate=candidate,
            repo_files=repo_files,
        )
        if resolved_path:
            resolved.append(resolved_path)
            continue

        directory = _resolve_repo_directory_reference(
            repo_root=repo_root,
            candidate=candidate,
            repo_files=repo_files,
        )
        if directory:
            resolved.extend(
                _rank_directory_context_files(
                    directory=directory,
                    repo_files=repo_files,
                    terms=directory_terms,
                    limit=directory_file_limit,
                )
            )
            continue

        unresolved.append(candidate)

    return dedupe(resolved), dedupe(unresolved)


def _canonicalize_context_requests(
    *,
    repo_root: Path,
    requests: list[dict[str, Any]],
    repo_files: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Canonicalize patcher context requests before they enter another graph loop."""

    canonical: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for item in requests:
        raw_path = _clean_repo_path_reference(str(item.get("path", "")))
        if not raw_path:
            continue

        requested_terms = [
            str(term).strip()
            for term in item.get("terms", [])
            if str(term).strip()
        ]
        reason = str(item.get("reason", "")).strip()
        directory_terms = [*requested_terms, reason]

        file_path = _resolve_repo_file_reference(
            repo_root=repo_root,
            candidate=raw_path,
            repo_files=repo_files,
        )
        if file_path:
            canonical.append({**item, "path": file_path})
            continue

        directory = _resolve_repo_directory_reference(
            repo_root=repo_root,
            candidate=raw_path,
            repo_files=repo_files,
        )
        if directory:
            for path in _rank_directory_context_files(
                directory=directory,
                repo_files=repo_files,
                terms=directory_terms,
                limit=4,
            ):
                canonical.append(
                    {
                        **item,
                        "path": path,
                        # Directory ranges are not meaningful for the discovered files.
                        "start_line": None,
                        "end_line": None,
                    }
                )
            continue

        unresolved.append(raw_path)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for item in canonical:
        key = (
            item.get("path"),
            item.get("start_line"),
            item.get("end_line"),
            tuple(item.get("terms") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, dedupe(unresolved)


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



def _fallback_implementation_units(
    state: CodingAgentState,
    search_requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": "primary",
            "objective": state.get("user_request", "Implement the requested repository change."),
            "acceptance_criteria": ["The requested behavior is implemented without unrelated changes."],
            "search_requests": search_requests,
            "candidate_paths": _explicit_request_paths(state),
            "depends_on": [],
            "validation_commands": list(state.get("validation_commands") or []),
        }
    ]


def _is_non_implementation_unit(unit: dict[str, Any]) -> bool:
    """Filter planner work that belongs to navigation or final validation instead.

    Be conservative here: a product feature can legitimately be called "validate input".
    We only remove clearly meta-level inspection/integration-validation units.
    """

    unit_id = str(unit.get("id", "")).strip().lower().replace("-", "_")
    objective = str(unit.get("objective", "")).strip().lower()

    if unit_id.startswith((
        "inspect_", "inspection_", "discover_", "understand_", "explore_",
        "context_", "gather_context", "repo_context", "repository_context",
    )):
        return True
    if objective.startswith((
        "inspect and understand ", "inspect the current ", "inspect attached ",
        "discover the current ", "explore the repository ", "gather context ",
        "understand the current ",
    )):
        return True

    validation_meta_markers = (
        "integration", "all changes", "work together", "no regressions", "build",
        "typecheck", "type-check", "run tests", "visual test", "responsive on",
    )
    looks_like_validation_unit = (
        unit_id.startswith(("validate_", "validation_", "verify_integration"))
        or objective.startswith(("validate integration", "validate that all", "verify integration"))
    )
    return looks_like_validation_unit and any(
        marker in objective or marker in unit_id
        for marker in validation_meta_markers
    )


def _merge_unit_into(target: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(target)
    objectives = [
        str(value).strip()
        for value in (target.get("objective", ""), incoming.get("objective", ""))
        if str(value).strip()
    ]
    merged["objective"] = " / ".join(dict.fromkeys(objectives))
    merged["acceptance_criteria"] = dedupe(
        [
            *[str(value) for value in target.get("acceptance_criteria", [])],
            *[str(value) for value in incoming.get("acceptance_criteria", [])],
        ]
    )[:8]
    merged["search_requests"] = [
        *list(target.get("search_requests") or []),
        *list(incoming.get("search_requests") or []),
    ]
    merged["candidate_paths"] = dedupe(
        [
            *[str(path) for path in target.get("candidate_paths", [])],
            *[str(path) for path in incoming.get("candidate_paths", [])],
        ]
    )
    merged["depends_on"] = dedupe(
        [
            *[str(dep) for dep in target.get("depends_on", [])],
            *[str(dep) for dep in incoming.get("depends_on", [])],
        ]
    )
    merged["validation_commands"] = dedupe(
        [
            *[str(command) for command in target.get("validation_commands", [])],
            *[str(command) for command in incoming.get("validation_commands", [])],
        ]
    )
    return merged


def _normalize_implementation_units(
    raw_units: list[dict[str, Any]],
    *,
    state: CodingAgentState,
    search_requests: list[dict[str, Any]],
    cfg: CodingAgentSettings,
) -> list[dict[str, Any]]:
    """Normalize, remove non-implementation work, and coalesce shared-file units."""

    units = [
        dict(item)
        for item in raw_units[: max(1, cfg.max_implementation_units)]
        if not _is_non_implementation_unit(dict(item))
    ]
    if not units:
        units = _fallback_implementation_units(state, search_requests)

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    alias: dict[str, str] = {}

    for index, raw in enumerate(units, start=1):
        item = dict(raw)
        raw_id = re.sub(r"[^A-Za-z0-9_-]+", "-", str(item.get("id", "")).strip()).strip("-")
        unit_id = raw_id or f"unit-{index}"
        if unit_id in seen_ids:
            unit_id = f"{unit_id}-{index}"

        candidate_paths = dedupe(
            filter_context_paths(
                [
                    str(path).strip()
                    for path in item.get("candidate_paths", [])
                    if str(path).strip()
                ]
            )
        )
        item.update(
            {
                "id": unit_id,
                "objective": str(item.get("objective", "")).strip()
                or state.get("user_request", "Implement the requested change."),
                "acceptance_criteria": [
                    str(value).strip()
                    for value in item.get("acceptance_criteria", [])
                    if str(value).strip()
                ][:8],
                "search_requests": list(item.get("search_requests") or search_requests),
                "candidate_paths": candidate_paths,
                "depends_on": [str(dep).strip() for dep in item.get("depends_on", []) if str(dep).strip()],
                "validation_commands": [
                    str(command).strip()
                    for command in item.get("validation_commands", [])
                    if str(command).strip()
                ],
            }
        )

        # File ownership is exclusive. Merge units that advertise any common target
        # file rather than allowing parallel workers to generate stale overlapping edits.
        overlap_index = next(
            (
                existing_index
                for existing_index, existing in enumerate(normalized)
                if candidate_paths
                and set(candidate_paths) & set(existing.get("candidate_paths", []))
            ),
            None,
        )
        if overlap_index is not None:
            owner_id = str(normalized[overlap_index]["id"])
            alias[unit_id] = owner_id
            normalized[overlap_index] = _merge_unit_into(normalized[overlap_index], item)
            continue

        normalized.append(item)
        seen_ids.add(unit_id)
        alias[unit_id] = unit_id

    valid_ids: set[str] = set()
    for item in normalized:
        remapped: list[str] = []
        for dep in item.get("depends_on", []):
            resolved = alias.get(str(dep), str(dep))
            if resolved in valid_ids and resolved != item["id"]:
                remapped.append(resolved)
        item["depends_on"] = dedupe(remapped)
        valid_ids.add(str(item["id"]))

    return normalized


def _initial_completion_ledger(units: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(unit["id"]): {
            "status": "pending",
            "implementation_attempts": 0,
            "patch_retries": 0,
            "files_inspected": [],
            "files_changed": [],
            "last_error": "",
            "last_generation": 0,
            "last_model": "",
            "last_provider": "",
            "last_context_tokens": 0,
            "last_context_budget_tokens": 0,
        }
        for unit in units
    }


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

    if task_mode == "simple" and has_selected_custom_tools:
        task_mode = "standard"

    decision: PlanDecision | None = None
    planning_errors = list(state.get("errors", []))

    if task_mode != "simple":
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

        approved_custom_tool_names = {
            name
            for name in selected_tool_names
            if approved_custom_tool_registry.has(name)
        }
        approved_custom_tool_catalog = approved_custom_tool_registry.prompt_catalog(
            selected_tool_names
        )
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
                "\n\nRelevant durable coding outcomes:\n" + bullets(memories[:3])
            )

        attachment_summary = _attached_file_summary(state)
        if attachment_summary:
            planner_prompt += (
                "\n\nUser-attached files available as read-only context:\n"
                f"{attachment_summary}"
            )

        try:
            decision = invoke_parsed_decision(
                model=_coding_node_model(cfg.planner_max_tokens, cache_namespace="plan"),
                schema=PlanDecision,
                node_name="plan",
                state=state,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=planner_prompt,
                max_attempts=1,
            )
        except Exception as exc:
            planning_errors.append(
                f"LLM planning failed; used deterministic implementation unit: {exc}"
            )

    if decision is None:
        search_requests = deterministic_search
        raw_units = _fallback_implementation_units(state, search_requests)
        plan = [
            "Inspect the smallest repository surface needed for the request.",
            "Implement each unit independently and reconcile patch proposals deterministically.",
            "Run targeted validation after all units are reconciled.",
        ]
        search_queries: list[str] = []
        validation_commands = list(state.get("validation_commands") or [])
        web_search_query = ""
        custom_tool_calls: list[dict[str, Any]] = []
        effective_mode = task_mode
    else:
        search_requests = _planned_search_requests(decision, request) or deterministic_search
        raw_units = [item.model_dump() for item in decision.implementation_units]
        plan = decision.plan
        search_queries = decision.search_queries
        validation_commands = decision.validation_commands
        web_search_query = decision.web_search_query or ""
        approved_names = {
            name
            for name in selected_tool_names
            if approved_custom_tool_registry.has(name)
        }
        custom_tool_calls = [
            {**item.model_dump(), "tool_name": item.tool_name.strip()}
            for item in decision.custom_tool_calls[:MAX_CUSTOM_TOOL_CALLS]
            if item.tool_name.strip() in approved_names
        ]
        effective_mode = decision.task_mode if len(raw_units) > 1 else task_mode

    units = _normalize_implementation_units(
        raw_units,
        state=state,
        search_requests=search_requests,
        cfg=cfg,
    )
    if len(units) > 1:
        effective_mode = "parallel"

    # This is a repair-round ceiling, not a scheduling-batch ceiling. Fresh pending
    # units may run in later generations without consuming another repair iteration.
    max_iterations = max(
        1,
        int(state.get("max_implementation_iterations", 0) or cfg.max_implementation_iterations),
    )

    implementation_run_id = str(state.get("run_id", "")).strip() or uuid4().hex

    return {
        "task_mode": effective_mode,
        "plan": plan,
        "implementation_run_id": implementation_run_id,
        "search_requests": search_requests,
        "search_queries": search_queries,
        "implementation_units": units,
        "completion_ledger": _initial_completion_ledger(units),
        "implementation_generation": 0,
        "implementation_iteration": 1,
        "max_implementation_iterations": max_iterations,
        "subtask_worker_results": [],
        "reasoning_reconciliations_used": 0,
        "custom_tool_calls": custom_tool_calls,
        "validation_commands": validation_commands,
        "web_search_query": web_search_query,
        "errors": planning_errors,
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
    """Select shared canonical repository hints before implementation workers run."""

    cfg = _settings_for_state(state, cfg)
    repo_root = resolve_repo_root(state, cfg)
    repo_files = filter_context_paths(list_files(repo_root, ".", max_depth=12))
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

    ranked_paths = paths_from_ranked_results(search_result_dicts)
    ledger = state.get("completion_ledger") or {}
    unit_candidate_paths = [
        str(path).strip()
        for unit in state.get("implementation_units", [])
        if str((ledger.get(str(unit.get("id", ""))) or {}).get("status", "pending"))
        in {"pending", "retryable"}
        for path in unit.get("candidate_paths", [])
        if str(path).strip()
    ]
    # Direct request paths and candidate paths from unfinished implementation units
    # are higher priority than general ranked search results.
    raw_candidates = [
        *requested_paths,
        *explicit_paths,
        *unit_candidate_paths,
        *ranked_paths,
    ]
    resolved_paths, unresolved_paths = _resolve_context_paths(
        repo_root=repo_root,
        candidate_paths=raw_candidates,
        repo_files=repo_files,
    )
    selected_paths = resolved_paths[:MAX_REPO_NAVIGATION_FILES]

    # Unresolved explicit/request paths are useful diagnostics, but do not let them
    # consume navigation slots or fan out into repeated worker read failures.
    important_unresolved = set(requested_paths + explicit_paths)
    for path in unresolved_paths:
        if path in important_unresolved:
            message = f"Could not resolve repository path reference: {path}"
            if message not in errors:
                errors.append(message)

    navigation_summary = (
        "Selected canonical files from patcher context requests, repository "
        "attachments/request paths, and ranked structured search results."
    )
    confidence = 0.9 if selected_paths else 0.45
    missing_context: list[str] = []

    if not selected_paths and cfg.llm_navigation_enabled:
        try:
            root_files = repo_files[: cfg.max_search_results]
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
                    selected_skill=", ".join(
                        state.get("selected_skills") or [state.get("selected_skill", "")]
                    ),
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

            selected_paths, unresolved_llm_paths = _resolve_context_paths(
                repo_root=repo_root,
                candidate_paths=[item.path for item in decision.files_to_inspect],
                repo_files=repo_files,
            )

            selected_paths = selected_paths[:MAX_REPO_NAVIGATION_FILES]
            if unresolved_llm_paths:
                errors.append(
                    "Optional LLM repo navigator returned unresolved paths: "
                    + ", ".join(unresolved_llm_paths[:5])
                )

            navigation_summary = decision.task_summary or navigation_summary
            confidence = decision.confidence
            missing_context = decision.missing_context
        except Exception as exc:
            errors.append(f"Optional LLM repo navigator failed: {exc}")

    requested_resolved, _ = _resolve_context_paths(
        repo_root=repo_root,
        candidate_paths=requested_paths,
        repo_files=repo_files,
    )

    explicit_resolved, _ = _resolve_context_paths(
        repo_root=repo_root,
        candidate_paths=explicit_paths,
        repo_files=repo_files,
    )

    unit_candidate_resolved, _ = _resolve_context_paths(
        repo_root=repo_root,
        candidate_paths=unit_candidate_paths,
        repo_files=repo_files,
    )

    high_priority = set(
        requested_resolved + explicit_resolved + unit_candidate_resolved
    )

    navigation_files = [
        {
            "path": path,
            "reason": (
                "Explicit file/context request."
                if path in high_priority
                else "Ranked structured-search result."
            ),
        }
        for path in selected_paths
    ]

    return {
        "implementation_generation": int(
            state.get("implementation_generation", 0)
        ) + 1,
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


_VALIDATION_INFRA_MARKERS = (
    "not found",
    "no such file or directory",
    "command blocked",
    "blocked by coding-agent allowlist",
    "timed out",
    "timeout",
    "validation harness failed",
    "executable file not found",
    "is not recognized as an internal or external command",
)


def _validation_is_infrastructure_failure(result: dict[str, Any]) -> bool:
    try:
        returncode = int(result.get("returncode", 0))
    except (TypeError, ValueError):
        returncode = 1
    if returncode == 0:
        return False
    haystack = "\n".join(
        str(result.get(key, ""))
        for key in ("stderr", "stdout", "reason")
    ).lower()
    return any(marker in haystack for marker in _VALIDATION_INFRA_MARKERS)


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

    infrastructure_failures = [
        item for item in results if _validation_is_infrastructure_failure(item)
    ]
    code_results = [
        item for item in results if not _validation_is_infrastructure_failure(item)
    ]
    for item in infrastructure_failures:
        item["failure_kind"] = "infrastructure"
        existing_reason = str(item.get("reason", "")).strip()
        item["reason"] = (
            "Validation infrastructure unavailable; this does not reopen implementation work."
            + (f" {existing_reason}" if existing_reason else "")
        )

    blocking_failures = blocking_validation_failures(code_results)
    advisory_failures = [
        *advisory_validation_failures(code_results),
        *infrastructure_failures,
    ]

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


def report_node(state: CodingAgentState) -> CodingAgentState:
    validation_results = state.get("validation_results", [])
    validation_lines = [
        f"- `{item.get('command', 'unknown')}` -> exit code "
        f"{item.get('returncode', 'unknown')}"
        for item in validation_results
    ]
    all_errors = [*state.get("errors", []), *state.get("memory_errors", [])]
    changed = [
        str(item.get("path", "")) + " - " + str(item.get("write_result", ""))
        for item in state.get("file_changes", [])
    ]
    current_run_id = str(state.get("implementation_run_id", ""))
    current_generation = int(state.get("implementation_generation", 0))
    worker_count = sum(
        1
        for item in state.get("subtask_worker_results", [])
        if str(item.get("run_id", "")) == current_run_id
        and int(item.get("generation", -1)) == current_generation
    )
    ledger_lines = []
    for unit_id, entry in (state.get("completion_ledger") or {}).items():
        model = str(entry.get("last_model", "")).strip()
        context_usage = ""
        if entry.get("last_context_budget_tokens"):
            context_usage = (
                f"; context={entry.get('last_context_tokens', 0)}/"
                f"{entry.get('last_context_budget_tokens', 0)} tokens"
            )
        ledger_lines.append(
            f"- {unit_id}: {entry.get('status', 'unknown')}; "
            f"implementation_attempts={entry.get('implementation_attempts', 0)}; "
            f"patch_retries={entry.get('patch_retries', 0)}"
            + (f"; model={model}" if model else "")
            + context_usage
            + (
                f"; last_error={entry.get('last_error')}"
                if entry.get("last_error")
                else ""
            )
        )

    report = f"""Coding agent run summary

Request:
{state.get('user_request', '')}

Execution mode:
{state.get('task_mode', 'standard')} ({worker_count} worker(s) in final generation)

Repair rounds:
{state.get('implementation_iteration', 1)}/{state.get('max_implementation_iterations', 1)}

Reasoning reconciliations used:
{state.get('reasoning_reconciliations_used', 0)}

Execution profile:
{state.get('runtime_settings', {})}

Selected skills:
{bullets(state.get('selected_skills') or [state.get('selected_skill', 'none')])}

Plan:
{bullets(state.get('plan', []))}

Completion ledger:
{chr(10).join(ledger_lines) if ledger_lines else 'No implementation ledger was produced.'}

Approved custom tools used:
{bullets([
    item.get('tool_name', '') + (' (ok)' if item.get('success') else ' (failed)')
    for item in state.get('custom_tool_results', [])
]) if state.get('custom_tool_results') else 'None'}

Files inspected:
{bullets(state.get('files_inspected', []))}

Files changed/proposed:
{bullets(changed)}

Patch reconciliation:
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
