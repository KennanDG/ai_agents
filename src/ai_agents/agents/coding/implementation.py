from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Any, Iterable

from langgraph.types import Send

from ai_agents.agents.coding.coding_agent_schemas import (
    FileEdit,
    PatchDecision,
    ReconciliationDecision,
)
from ai_agents.agents.coding.coding_agent_settings import (
    CodingAgentSettings,
    settings as default_settings,
)
from ai_agents.agents.coding.llm import invoke_parsed_decision
from ai_agents.agents.coding.model_factory import build_chat_model
from ai_agents.agents.coding.prompts import (
    PATCHER_SYSTEM_PROMPT,
    RECONCILER_SYSTEM_PROMPT,
    build_patcher_user_prompt,
    build_reconciler_user_prompt,
)
from ai_agents.agents.coding.runtime import allow_write as resolve_allow_write
from ai_agents.agents.coding.runtime import repo_root as resolve_repo_root
from ai_agents.agents.coding.state import CodingAgentState
from ai_agents.agents.coding.tools.filesystem import list_files, read_file, write_file
from ai_agents.agents.coding.tools.patch import unified_diff
from ai_agents.agents.coding.tools.search import search_repository
from ai_agents.agents.coding.utils.constants import (
    RETRYABLE_UNIT_STATUSES,
    SUCCESSFUL_UNIT_STATUSES,
)
from ai_agents.agents.coding.utils.helpers import (
    _implementation_complete,
    _resolve_existing_repo_path,
)
from ai_agents.agents.coding.utils.model_budget import (
    configured_context_window,
    configured_max_output_tokens,
    estimate_tokens,
    fit_blocks_to_token_budget,
    resolve_model_profile,
)
from ai_agents.agents.coding.utils.patch import (
    apply_exact_replace,
    format_failed_validation_results,
    is_forbidden_write_path,
)
from ai_agents.agents.coding.utils.search import (
    filter_context_paths,
    paths_from_ranked_results,
)
from ai_agents.agents.coding.utils.skills import skill_instructions_for_llm
from ai_agents.agents.coding.utils.text import bullets, dedupe
from ai_agents.config.settings import settings as app_settings


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

_PATCH_ANCHOR_SIMILARITY_THRESHOLD = 0.97
_PATCH_ANCHOR_SIMILARITY_MARGIN = 0.04

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


def _ledger_entry(state: CodingAgentState, unit_id: str) -> dict[str, Any]:
    return dict((state.get("completion_ledger") or {}).get(unit_id) or {})


def _unit_by_id(state: CodingAgentState) -> dict[str, dict[str, Any]]:
    return {
        str(unit.get("id", "")).strip(): dict(unit)
        for unit in state.get("implementation_units", [])
        if str(unit.get("id", "")).strip()
    }


def _dependencies_satisfied(
    unit: dict[str, Any],
    ledger: dict[str, dict[str, Any]],
) -> bool:
    return all(
        str((ledger.get(dep) or {}).get("status", "pending"))
        in SUCCESSFUL_UNIT_STATUSES
        for dep in unit.get("depends_on", [])
    )


def _eligible_units(state: CodingAgentState) -> list[dict[str, Any]]:
    ledger = state.get("completion_ledger") or {}
    eligible: list[dict[str, Any]] = []

    for raw_unit in state.get("implementation_units", []):
        unit = dict(raw_unit)
        unit_id = str(unit.get("id", "")).strip()

        if not unit_id:
            continue

        status = str((ledger.get(unit_id) or {}).get("status", "pending"))
        if status not in RETRYABLE_UNIT_STATUSES:
            continue

        if _dependencies_satisfied(unit, ledger):
            eligible.append(unit)

    # Schedule never-attempted work before repairs so worker concurrency does not
    # consume the repair-round budget merely because there are more units than slots.
    eligible.sort(
        key=lambda item: (
            0
            if str((ledger.get(str(item.get("id", ""))) or {}).get("status", "pending")) == "pending"
            else 1
        )
    )
    return eligible


def assign_subtask_workers(state: CodingAgentState) -> list[Send]:
    """Schedule the next bounded batch of implementation-unit workers.

    Total implementation units are independent from concurrency. Only dependency-ready
    pending/retryable units are dispatched, up to ``max_subtask_workers``.
    """

    cfg = _settings_for_state(state)
    units = _eligible_units(state)[: max(1, cfg.max_subtask_workers)]
    nav_paths = [
        str(item.get("path", "")).strip()
        for item in state.get("repo_navigation_files", [])
        if str(item.get("path", "")).strip()
    ]

    sends: list[Send] = []
    for unit in units:
        owned_paths = {
            str(path).strip()
            for path in unit.get("candidate_paths", [])
            if str(path).strip()
        }
        # Do not round-robin unrelated navigation files into worker prompts. A worker
        # gets its owned candidate files; navigator hints are fallback context only when
        # the planner could not identify any candidate path.
        assigned_nav = (
            [path for path in nav_paths if path in owned_paths]
            if owned_paths
            else nav_paths[: max(1, cfg.max_worker_files)]
        )
        payload = {
            "repo_root": state.get("repo_root", ""),
            "original_repo_root": state.get("original_repo_root", ""),
            "sandbox_root": state.get("sandbox_root", ""),
            "sandbox_enabled": state.get("sandbox_enabled", False),
            "allow_write": state.get("allow_write", False),
            "user_request": state.get("user_request", ""),
            "task_mode": state.get("task_mode", "standard"),
            "runtime_settings": state.get("runtime_settings", {}),
            "active_implementation_unit": unit,
            "implementation_run_id": state.get("implementation_run_id", ""),
            "implementation_generation": state.get("implementation_generation", 0),
            "implementation_iteration": state.get("implementation_iteration", 1),
            "max_implementation_iterations": state.get(
                "max_implementation_iterations",
                cfg.max_implementation_iterations,
            ),
            "search_requests": state.get("search_requests", []),
            "repo_navigation_files": [
                {"path": path, "reason": "Assigned navigator context."}
                for path in assigned_nav
            ],
            "selected_skill": state.get("selected_skill", ""),
            "selected_skills": state.get("selected_skills", []),
            "skill_instructions": state.get("skill_instructions", ""),
            "plan": state.get("plan", []),
            "custom_tool_results": state.get("custom_tool_results", []),
            "attached_files": state.get("attached_files", []),
            "validation_results": state.get("validation_results", []),
            "loop_context_focus": state.get("loop_context_focus", ""),
        }
        
        sends.append(Send("subtask_worker", payload))

    return sends


def _worker_terms(state: dict[str, Any], unit: dict[str, Any]) -> list[str]:
    values = [
        str(unit.get("objective", "")),
        *[str(item) for item in unit.get("acceptance_criteria", [])],
        str(state.get("user_request", "")),
    ]
    for request in unit.get("search_requests", []) or state.get("search_requests", []):
        if isinstance(request, dict):
            values.extend(str(term) for term in request.get("terms", []))

    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in _TOKEN_RE.findall(value):
            lowered = token.casefold()
            if len(token) < 3 or lowered in {
                "the", "and", "for", "with", "from", "this", "that", "into",
                "implementation", "unit", "change", "update",
            }:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            terms.append(token)
            if len(terms) >= 16:
                return terms
    return terms


def _normalize_repo_path(value: str) -> str:
    candidate = str(value or "").strip().replace("\\", "/")
    candidate = candidate.strip("`\"'")
    while candidate.startswith("./"):
        candidate = candidate[2:]
    candidate = candidate.lstrip("/")
    parts = Path(candidate).parts
    if not candidate or Path(candidate).is_absolute() or ".." in parts:
        return ""
    return candidate


def _resolve_candidate_paths(
    *,
    repo_root: Path,
    candidates: Iterable[str],
    repo_files: list[str],
    terms: list[str],
    max_files: int,
) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    unresolved: list[str] = []

    for raw in candidates:
        candidate = _normalize_repo_path(str(raw))
        if not candidate:
            continue

        path = _resolve_existing_repo_path(
            repo_root=repo_root,
            candidate=candidate,
            repo_files=repo_files,
        )
        if path:
            resolved.append(path)
            continue

        # Be tolerant of a stale or over-qualified path when the repository contains
        # one obvious file with the same basename. This is especially useful for
        # handoffs where another model guessed one extra directory segment.
        basename = Path(candidate).name
        basename_matches = [
            repo_path for repo_path in repo_files if Path(repo_path).name == basename
        ]
        if basename_matches:
            needles = [term.casefold() for term in terms]

            def basename_score(path_value: str) -> tuple[int, int, str]:
                lowered = path_value.casefold()
                hits = sum(1 for term in needles if term in lowered)
                depth = path_value.count("/")
                return (-hits, depth, path_value)

            ranked = sorted(basename_matches, key=basename_score)
            if len(ranked) == 1 or (
                len(ranked) > 1
                and basename_score(ranked[0]) < basename_score(ranked[1])
            ):
                resolved.append(ranked[0])
                continue

        # A planner may reasonably name a directory. Expand it deterministically
        # using filename/term relevance instead of passing the directory to read_file.
        prefix = candidate.rstrip("/") + "/"
        directory_matches = [path for path in repo_files if path.startswith(prefix)]
        if directory_matches:
            needles = [term.casefold() for term in terms]

            def score(path_value: str) -> tuple[int, int, str]:
                lowered = path_value.casefold()
                hits = sum(1 for term in needles if term in lowered)
                depth = path_value[len(prefix):].count("/")
                return (-hits, depth, path_value)

            resolved.extend(sorted(directory_matches, key=score)[:2])
            continue

        unresolved.append(candidate)

    return dedupe(filter_context_paths(resolved))[:max_files], dedupe(unresolved)


def _requested_context_for_path(
    requests: list[dict[str, Any]],
    path: str,
) -> tuple[list[tuple[int, int]], list[str], bool]:
    ranges: list[tuple[int, int]] = []
    terms: list[str] = []
    matched = False

    for item in requests:
        if _normalize_repo_path(str(item.get("path", ""))) != path:
            continue
        matched = True
        start = item.get("start_line")
        end = item.get("end_line")
        if isinstance(start, int) and isinstance(end, int):
            ranges.append((max(1, start), max(start, end)))
        terms.extend(
            str(term).strip()
            for term in item.get("terms", [])
            if str(term).strip()
        )

    return ranges, dedupe(terms), matched


def _char_windows(
    text: str,
    *,
    terms: list[str],
    chunk_chars: int,
    overlap_chars: int,
    max_windows: int = 6,
) -> list[tuple[int, int]]:
    lowered = text.casefold()
    ranges: list[tuple[int, int]] = []
    half = max(1, chunk_chars // 2)

    for term in terms:
        needle = term.casefold().strip()
        if len(needle) < 3:
            continue
        start_at = 0
        while len(ranges) < max_windows:
            hit = lowered.find(needle, start_at)
            if hit < 0:
                break
            ranges.append((max(0, hit - half), min(len(text), hit + half)))
            start_at = hit + len(needle)

    if not ranges:
        ranges.append((0, min(len(text), chunk_chars)))

    normalized: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        start = max(0, start - overlap_chars)
        end = min(len(text), end + overlap_chars)
        while start > 0 and text[start - 1] != "\n":
            start -= 1
        while end < len(text) and text[end - 1 : end] != "\n":
            end += 1
        if normalized and start <= normalized[-1][1]:
            normalized[-1] = (normalized[-1][0], max(normalized[-1][1], end))
        else:
            normalized.append((start, end))
    return normalized[:max_windows]


def _format_worker_file_context(
    *,
    path: str,
    text: str,
    terms: list[str],
    context_requests: list[dict[str, Any]],
    cfg: CodingAgentSettings,
) -> str:
    ranges, requested_terms, explicitly_requested = _requested_context_for_path(
        context_requests,
        path,
    )

    if ranges:
        lines = text.splitlines(keepends=True)
        blocks = [
            f"File: {path}\nContent-Status: selected-lines "
            f"({len(text)} total characters)"
        ]
        for start, end in ranges[:6]:
            bounded_start = min(max(1, start), max(1, len(lines)))
            bounded_end = min(max(bounded_start, end), max(1, len(lines)))
            excerpt = "".join(lines[bounded_start - 1 : bounded_end])
            blocks.append(
                f"Chunk-Lines: {bounded_start}-{bounded_end}\n```\n{excerpt}\n```"
            )
        return "\n".join(blocks)

    if len(text) <= cfg.max_full_file_chars:
        return f"File: {path}\nContent-Status: complete\n```\n{text}\n```"

    effective_terms = requested_terms or terms
    if explicitly_requested and not effective_terms:
        effective_terms = [Path(path).stem, *terms]

    windows = _char_windows(
        text,
        terms=effective_terms,
        chunk_chars=cfg.context_chunk_chars,
        overlap_chars=cfg.context_chunk_overlap_chars,
    )
    blocks = [
        f"File: {path}\nContent-Status: selected-chunks "
        f"({len(text)} total characters)"
    ]
    for start, end in windows:
        start_line = text.count("\n", 0, start) + 1
        end_line = text.count("\n", 0, end) + 1
        blocks.append(
            f"Chunk-Lines: {start_line}-{end_line}\n```\n{text[start:end]}\n```"
        )
    return "\n".join(blocks)


def _format_upload_blocks(
    state: dict[str, Any],
    *,
    terms: list[str],
    cfg: CodingAgentSettings,
) -> list[str]:
    blocks: list[str] = []
    for item in state.get("attached_files", []):
        if item.get("source") == "repo":
            continue
        content = str(item.get("content", ""))
        if not content:
            continue
        name = str(item.get("name", "attachment")).strip() or "attachment"
        if len(content) <= cfg.max_full_file_chars:
            excerpt = content
            status = "complete"
        else:
            windows = _char_windows(
                content,
                terms=terms,
                chunk_chars=cfg.context_chunk_chars,
                overlap_chars=cfg.context_chunk_overlap_chars,
                max_windows=3,
            )
            excerpt = "\n\n".join(content[start:end] for start, end in windows)
            status = f"selected-chunks ({len(content)} total characters)"
        blocks.append(
            f"Attachment: {name}\nContent-Status: {status}\n```\n{excerpt}\n```"
        )
    return blocks


def _canonicalize_context_requests(
    *,
    repo_root: Path,
    requests: list[dict[str, Any]],
    repo_files: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    canonical: list[dict[str, Any]] = []
    unresolved: list[str] = []
    seen: set[tuple[Any, ...]] = set()

    for item in requests:
        raw = _normalize_repo_path(str(item.get("path", "")))
        if not raw:
            continue
        resolved = _resolve_existing_repo_path(
            repo_root=repo_root,
            candidate=raw,
            repo_files=repo_files,
        )
        if not resolved:
            unresolved.append(raw)
            continue
        normalized = {**item, "path": resolved}
        key = (
            resolved,
            normalized.get("start_line"),
            normalized.get("end_line"),
            tuple(normalized.get("terms") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        canonical.append(normalized)

    return canonical, dedupe(unresolved)


def _model_profile(
    *,
    role: str,
    cfg: CodingAgentSettings,
):
    if role == "coding":
        provider = app_settings.coding_provider
        model_name = app_settings.coding_model
        requested_output_tokens = cfg.simple_patch_max_tokens
        fallback_window = cfg.coding_model_context_window_tokens
        fallback_output = cfg.coding_model_max_output_tokens
        configured_input = cfg.max_context_prompt_tokens
    elif role == "reasoning":
        provider = app_settings.reasoning_provider
        model_name = app_settings.reasoning_model
        requested_output_tokens = cfg.reconciliation_max_tokens
        fallback_window = cfg.reasoning_model_context_window_tokens
        fallback_output = cfg.reasoning_model_max_output_tokens
        configured_input = min(
            cfg.max_context_prompt_tokens,
            cfg.reconciliation_context_max_tokens,
        )
    else:
        raise ValueError(f"Unknown model role: {role}")

    window = configured_context_window(
        provider=provider,
        model_name=model_name,
        fallback_tokens=fallback_window,
        overrides_json=cfg.model_context_window_overrides_json,
    )
    model_output_cap = configured_max_output_tokens(
        provider=provider,
        model_name=model_name,
        fallback_tokens=fallback_output,
        overrides_json=cfg.model_max_output_overrides_json,
    )
    output_tokens = min(requested_output_tokens, model_output_cap)

    return resolve_model_profile(
        provider=provider,
        model_name=model_name,
        context_window_tokens=window,
        requested_output_tokens=output_tokens,
        configured_max_input_tokens=configured_input,
        reserve_tokens=cfg.context_prompt_reserve_tokens,
        safety_tokens=cfg.context_window_safety_tokens,
    )


def _worker_model(profile, *, cache_namespace: str):
    return build_chat_model(
        provider=profile.provider,
        model_name=profile.model_name,
        max_tokens=profile.requested_output_tokens,
        prompt_cache_namespace=cache_namespace,
    )


def subtask_worker_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Implement one unit in isolation and return a patch proposal, never direct writes."""

    cfg = _settings_for_state(state, cfg)
    repo_root = resolve_repo_root(state, cfg)
    unit = dict(state.get("active_implementation_unit") or {})
    unit_id = str(unit.get("id", "")).strip() or "unit"
    terms = _worker_terms(state, unit)
    repo_files = filter_context_paths(list_files(repo_root, ".", max_depth=12))
    errors: list[str] = []
    inspected: list[str] = []
    context_requests: list[dict[str, Any]] = []

    search_requests = unit.get("search_requests") or state.get("search_requests", [])
    candidates = [
        *[
            str(item.get("path", "")).strip()
            for item in state.get("repo_navigation_files", [])
            if str(item.get("path", "")).strip()
        ],
        *[str(path) for path in unit.get("candidate_paths", [])],
    ]

    # Candidate files are authoritative and cheap. Only broaden into repository
    # search when planning/navigation did not identify a concrete file for this unit.
    if search_requests and not [path for path in candidates if str(path).strip()]:
        try:
            results = search_repository(
                repo_root,
                search_requests,
                max_results=max(8, cfg.max_worker_files * 3),
            )
            candidates.extend(
                paths_from_ranked_results([item.to_dict() for item in results])
            )
        except Exception as exc:
            errors.append(f"Unit {unit_id} repository search failed: {exc}")

    max_model_attempts = 1 + max(0, cfg.max_patch_retries_per_unit)
    last_summary = ""
    last_blocking_reason = ""

    for model_attempt in range(1, max_model_attempts + 1):
        requested_paths = [
            str(item.get("path", "")).strip()
            for item in context_requests
            if str(item.get("path", "")).strip()
        ]
        paths, unresolved = _resolve_candidate_paths(
            repo_root=repo_root,
            candidates=[*requested_paths, *candidates],
            repo_files=repo_files,
            terms=terms,
            max_files=cfg.max_worker_files,
        )

        # Do not let one bad candidate path suppress normal repository discovery. If
        # any requested/candidate path is unresolved, use the unit's structured
        # searches as a fallback and merge the highest-ranked real files into context.
        if unresolved and search_requests:
            try:
                fallback_results = search_repository(
                    repo_root,
                    search_requests,
                    max_results=max(8, cfg.max_worker_files * 3),
                )
                fallback_paths = paths_from_ranked_results(
                    [item.to_dict() for item in fallback_results]
                )
                resolved_fallback, _ = _resolve_candidate_paths(
                    repo_root=repo_root,
                    candidates=fallback_paths,
                    repo_files=repo_files,
                    terms=terms,
                    max_files=cfg.max_worker_files,
                )
                paths = dedupe([*paths, *resolved_fallback])[: cfg.max_worker_files]
            except Exception as exc:
                errors.append(
                    f"Unit {unit_id} fallback repository search failed: {exc}"
                )

        for path in unresolved:
            if path in requested_paths:
                errors.append(
                    f"Unit {unit_id} could not resolve requested context path: {path}"
                )

        file_blocks: list[str] = []
        for path in paths:
            try:
                probe = read_file(repo_root, path, max_chars=cfg.max_file_chars + 1)
                if len(probe) > cfg.max_file_chars:
                    errors.append(
                        f"Unit {unit_id} skipped {path}: file exceeds raw read ceiling "
                        f"of {cfg.max_file_chars} characters."
                    )
                    continue
                file_blocks.append(
                    _format_worker_file_context(
                        path=path,
                        text=probe,
                        terms=terms,
                        context_requests=context_requests,
                        cfg=cfg,
                    )
                )
                inspected.append(path)
            except Exception as exc:
                errors.append(f"Unit {unit_id} could not read {path}: {exc}")

        unit_block = (
            "# Implementation unit\n"
            f"ID: {unit_id}\n"
            f"Objective: {unit.get('objective', '')}\n"
            "Acceptance criteria:\n"
            f"{bullets([str(item) for item in unit.get('acceptance_criteria', [])])}\n"
            f"Implementation iteration: {state.get('implementation_iteration', 1)}\n"
            f"Patch-model attempt for this unit: {model_attempt}/{max_model_attempts}"
        )
        priority_blocks = [unit_block]

        validation_feedback = (
            format_failed_validation_results(state.get("validation_results", []))
            if bool(state.get("blocking_validation_failed"))
            else ""
        )
        if validation_feedback:
            priority_blocks.append(
                "# Validation failures from the reconciled implementation\n"
                + validation_feedback
            )

        loop_focus = str(state.get("loop_context_focus", "")).strip()
        if loop_focus:
            priority_blocks.append("# Current retry focus\n" + loop_focus)

        if model_attempt == max_model_attempts:
            priority_blocks.append(
                "# Best-effort patching policy\n"
                "This is the final patch-model attempt for the unit. If at least one direct "
                "target file or a strong repository implementation pattern is visible, prefer "
                "a conservative reviewable patch with clearly stated assumptions over another "
                "request for merely helpful context. Request more context only for a material "
                "contract/security/correctness blocker."
            )

        tool_blocks = [
            (
                f"# Approved custom tool result: {item.get('tool_name', 'custom_tool')}\n"
                f"{item.get('output', '')}"
            )
            for item in state.get("custom_tool_results", [])
            if item.get("success") and str(item.get("output", "")).strip()
        ]
        upload_blocks = _format_upload_blocks(state, terms=terms, cfg=cfg)

        # Every implementation worker uses the coding slot. The reasoning slot is
        # reserved for one conditional reconciliation pass after fan-in.
        profile = _model_profile(role="coding", cfg=cfg)

        all_blocks = [
            *priority_blocks,
            *file_blocks,
            *upload_blocks,
            *tool_blocks,
        ]
        requested_context_tokens = sum(estimate_tokens(block) for block in all_blocks)
        adaptive_budget = min(
            profile.max_input_tokens,
            max(cfg.context_prompt_base_tokens, requested_context_tokens),
        )
        included, omitted, used_tokens = fit_blocks_to_token_budget(
            all_blocks,
            max_tokens=adaptive_budget,
        )

        context = "\n\n".join(included)
        if omitted:
            context += (
                "\n\n# Context omitted by token budget\n"
                + bullets(omitted)
                + "\nRequest narrower exact context if one of these blocks is required."
            )

        prompt = build_patcher_user_prompt(
            request=state.get("user_request", ""),
            selected_skill=", ".join(
                state.get("selected_skills")
                or [state.get("selected_skill", "")]
            ),
            skill_instructions=skill_instructions_for_llm(
                state.get("skill_instructions", "")
            ),
            plan=bullets(state.get("plan", [])),
            context=context,
            implementation_unit=unit.get("objective", ""),
            acceptance_criteria=bullets(
                [str(item) for item in unit.get("acceptance_criteria", [])]
            ),
        )

        try:
            decision: PatchDecision = invoke_parsed_decision(
                model=_worker_model(
                    profile,
                    cache_namespace=f"unit-{unit_id}",
                ),
                schema=PatchDecision,
                node_name=f"subtask_worker:{unit_id}",
                state={
                    **state,
                    "active_implementation_unit": unit,
                    "unit_patch_attempt": model_attempt,
                },
                system_prompt=PATCHER_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_attempts=1,
            )
        except Exception as exc:
            last_blocking_reason = f"Model invocation failed: {exc}"
            errors.append(
                f"Unit {unit_id} model attempt {model_attempt} failed: {exc}"
            )
            if model_attempt < max_model_attempts:
                continue
            break

        last_summary = decision.summary
        last_blocking_reason = decision.blocking_reason

        canonical_requests, unresolved_requests = _canonicalize_context_requests(
            repo_root=repo_root,
            requests=[item.model_dump() for item in decision.context_requests],
            repo_files=repo_files,
        )
        for path in unresolved_requests:
            errors.append(
                f"Unit {unit_id} requested unresolved repository context: {path}"
            )

        if decision.edits:
            return {
                "subtask_worker_results": [
                    {
                        "run_id": str(state.get("implementation_run_id", "")),
                        "generation": int(state.get("implementation_generation", 0)),
                        "implementation_iteration": int(
                            state.get("implementation_iteration", 1)
                        ),
                        "unit_id": unit_id,
                        "objective": str(unit.get("objective", "")),
                        "status": "proposed",
                        "summary": decision.summary,
                        "edits": [item.model_dump() for item in decision.edits],
                        "context_requests": canonical_requests,
                        "validation_commands": decision.validation_commands,
                        "files_inspected": dedupe(inspected),
                        "errors": errors,
                        "patch_retries": model_attempt - 1,
                        "model": profile.model_name,
                        "provider": profile.provider,
                        "context_window_tokens": profile.context_window_tokens,
                        "context_budget_tokens": adaptive_budget,
                        "context_tokens": used_tokens,
                    }
                ]
            }

        if decision.no_change_needed:
            return {
                "subtask_worker_results": [
                    {
                        "run_id": str(state.get("implementation_run_id", "")),
                        "generation": int(state.get("implementation_generation", 0)),
                        "implementation_iteration": int(
                            state.get("implementation_iteration", 1)
                        ),
                        "unit_id": unit_id,
                        "objective": str(unit.get("objective", "")),
                        "status": "completed",
                        "summary": decision.summary or "Unit already satisfied.",
                        "edits": [],
                        "context_requests": canonical_requests,
                        "validation_commands": decision.validation_commands,
                        "files_inspected": dedupe(inspected),
                        "errors": errors,
                        "patch_retries": model_attempt - 1,
                        "model": profile.model_name,
                        "provider": profile.provider,
                        "context_window_tokens": profile.context_window_tokens,
                        "context_budget_tokens": adaptive_budget,
                        "context_tokens": used_tokens,
                    }
                ]
            }

        if canonical_requests and model_attempt < max_model_attempts:
            context_requests = canonical_requests
            candidates = [
                *[str(item.get("path", "")) for item in canonical_requests],
                *candidates,
            ]
            continue

        last_blocking_reason = (
            decision.blocking_reason
            or decision.summary
            or "Worker returned no edits and no actionable context request."
        )
        if model_attempt < max_model_attempts:
            continue

    return {
        "subtask_worker_results": [
            {
                "run_id": str(state.get("implementation_run_id", "")),
                "generation": int(state.get("implementation_generation", 0)),
                "implementation_iteration": int(
                    state.get("implementation_iteration", 1)
                ),
                "unit_id": unit_id,
                "objective": str(unit.get("objective", "")),
                "status": "retryable",
                "summary": last_summary,
                "blocking_reason": last_blocking_reason,
                "edits": [],
                "context_requests": context_requests,
                "validation_commands": [],
                "files_inspected": dedupe(inspected),
                "errors": errors,
                "patch_retries": max(0, max_model_attempts - 1),
            }
        ]
    }


def gather_subtask_results_node(state: CodingAgentState) -> CodingAgentState:
    """Fan in current-generation worker outputs without building one giant prompt."""

    run_id = str(state.get("implementation_run_id", ""))
    generation = int(state.get("implementation_generation", 0))
    results = [
        item
        for item in state.get("subtask_worker_results", [])
        if str(item.get("run_id", "")) == run_id
        and int(item.get("generation", -1)) == generation
    ]
    errors = list(state.get("errors", []))
    files_inspected = list(state.get("files_inspected", []))

    for result in results:
        errors.extend(str(item) for item in result.get("errors", []))
        files_inspected.extend(
            str(path) for path in result.get("files_inspected", [])
        )

    return {
        "files_inspected": dedupe(files_inspected),
        "errors": dedupe(errors),
        "status": (
            "subtask_workers_completed"
            if results
            else "subtask_workers_failed"
        ),
    }


def _intended_edit_paths(
    *,
    repo_root: Path,
    repo_files: list[str],
    raw_edits: list[dict[str, Any]],
) -> list[str]:
    """Normalize a proposal's target paths without reading or staging file contents."""

    paths: list[str] = []
    for raw_edit in raw_edits:
        edit = FileEdit.model_validate(raw_edit)
        candidate = _normalize_repo_path(edit.path)
        if not candidate or is_forbidden_write_path(candidate):
            raise ValueError(f"Forbidden or invalid edit path: {edit.path}")

        resolved = _resolve_existing_repo_path(
            repo_root=repo_root,
            candidate=candidate,
            repo_files=repo_files,
        )
        paths.append(resolved or candidate)

    return dedupe(paths)



def _normalize_anchor_line(value: str) -> str:
    """Normalize only formatting noise for conservative patch-anchor matching."""

    return re.sub(r"\s+", " ", value.strip())


def _normalize_anchor_block(value: str) -> str:
    lines = [_normalize_anchor_line(line) for line in value.replace("\r\n", "\n").split("\n")]
    # Leading/trailing blank lines are common model formatting drift and are not
    # semantically meaningful for an edit anchor.
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _find_rebased_anchor(
    current: str,
    old: str,
    *,
    threshold: float,
    margin: float,
) -> tuple[str, float] | None:
    """Find one uniquely strong current-source match for a stale patch anchor.

    This intentionally tolerates only small formatting/source drift. It is not a
    general fuzzy patcher: ambiguous matches are rejected and sent back through the
    normal repair loop.
    """

    normalized_old = _normalize_anchor_block(old)
    if not normalized_old:
        return None

    current_lines = current.splitlines(keepends=True)
    old_lines = old.splitlines(keepends=True)
    old_normalized_lines = [_normalize_anchor_line(line) for line in old_lines]
    significant = [index for index, line in enumerate(old_normalized_lines) if line]
    if not significant:
        return None

    first_sig = significant[0]
    first_line = old_normalized_lines[first_sig]
    current_normalized_lines = [_normalize_anchor_line(line) for line in current_lines]

    # Start with exact normalized-line matches. If source drift touched that line,
    # allow a small bounded fuzzy shortlist rather than scanning every possible block.
    start_positions = [
        index for index, line in enumerate(current_normalized_lines) if line == first_line
    ]
    if not start_positions and len(first_line) >= 24:
        fuzzy_starts: list[tuple[float, int]] = []
        for index, line in enumerate(current_normalized_lines):
            if not line:
                continue
            score = SequenceMatcher(None, first_line, line).ratio()
            if score >= 0.92:
                fuzzy_starts.append((score, index))
        start_positions = [index for _, index in sorted(fuzzy_starts, reverse=True)[:24]]

    if not start_positions:
        return None

    old_line_count = max(1, len(old_lines))
    candidates: list[tuple[float, int, int, str]] = []
    seen_ranges: set[tuple[int, int]] = set()

    for position in start_positions[:48]:
        estimated_start = max(0, position - first_sig)
        for start_delta in (-2, -1, 0, 1, 2):
            start = estimated_start + start_delta
            if start < 0 or start >= len(current_lines):
                continue
            for line_delta in (-2, -1, 0, 1, 2):
                length = max(1, old_line_count + line_delta)
                end = min(len(current_lines), start + length)
                if end <= start or (start, end) in seen_ranges:
                    continue
                seen_ranges.add((start, end))
                candidate = "".join(current_lines[start:end])
                normalized_candidate = _normalize_anchor_block(candidate)
                if not normalized_candidate:
                    continue
                score = SequenceMatcher(
                    None,
                    normalized_old,
                    normalized_candidate,
                ).ratio()
                candidates.append((score, start, end, candidate))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_start, best_end, best_text = candidates[0]

    # Several candidate windows can describe the same source neighborhood because we
    # intentionally vary start/length by a couple of lines. Treat those as one match
    # and compare the best score only against materially different locations.
    neighborhood = max(2, old_line_count // 2)
    competing_scores = [
        score
        for score, start, end, _ in candidates[1:]
        if abs(start - best_start) >= neighborhood
        and (end <= best_start or start >= best_end)
    ]
    second_score = max(competing_scores, default=0.0)

    # Short anchors are easy to match accidentally, so only permit normalized-exact
    # rebasing for them. Longer blocks may tolerate a tiny amount of source drift.
    effective_threshold = 1.0 if len(normalized_old) < 80 else max(0.0, min(1.0, threshold))
    effective_margin = max(0.0, min(1.0, margin))
    if best_score < effective_threshold:
        return None
    if competing_scores and (best_score - second_score) < effective_margin:
        return None

    return best_text, best_score


def _resolve_patch_anchor(
    current: str,
    old: str,
    *,
    path: str,
    cfg: CodingAgentSettings,
) -> tuple[str, bool, float]:
    if not old:
        raise ValueError(f"Patch operation for {path} requires a non-empty anchor.")

    exact_count = current.count(old)
    if exact_count == 1:
        return old, False, 1.0
    if exact_count > 1:
        raise ValueError(
            f"Patch anchor occurs {exact_count} times in {path}; refusing ambiguous edit."
        )

    rebased = _find_rebased_anchor(
        current,
        old,
        threshold=_PATCH_ANCHOR_SIMILARITY_THRESHOLD,
        margin=_PATCH_ANCHOR_SIMILARITY_MARGIN,
    )
    if rebased is None:
        raise ValueError(
            f"Could not find exact old text or a unique high-confidence rebase in {path}"
        )

    matched, score = rebased
    return matched, True, score


def _stage_unit_edits(
    *,
    repo_root: Path,
    repo_files: list[str],
    raw_edits: list[dict[str, Any]],
    cfg: CodingAgentSettings,
) -> tuple[
    dict[str, str],
    dict[str, bool],
    dict[str, str],
    dict[str, dict[str, str]],
    list[str],
]:
    """Apply one unit's edits to memory first so the unit is atomic.

    ``original_exists`` intentionally distinguishes a missing path from a real,
    zero-byte text file. That distinction is needed for safe create/append/full-file
    replacement semantics and for rollback.
    """

    originals: dict[str, str] = {}
    original_exists: dict[str, bool] = {}
    staged: dict[str, str] = {}
    metadata: dict[str, dict[str, str]] = {}
    intended_paths: list[str] = []
    known_exists: set[str] = set(repo_files)

    for raw_edit in raw_edits:
        edit = FileEdit.model_validate(raw_edit)
        candidate = _normalize_repo_path(edit.path)
        if not candidate or is_forbidden_write_path(candidate):
            raise ValueError(f"Forbidden or invalid edit path: {edit.path}")

        resolved = _resolve_existing_repo_path(
            repo_root=repo_root,
            candidate=candidate,
            repo_files=repo_files,
        )
        path = resolved or candidate
        intended_paths.append(path)

        if path not in originals:
            target = (repo_root / path).resolve()
            root = repo_root.resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Path escapes repository root: {path}")

            exists = target.is_file()
            original_exists[path] = exists
            if exists:
                originals[path] = read_file(
                    repo_root,
                    path,
                    max_chars=cfg.max_file_chars + 1,
                )
                if len(originals[path]) > cfg.max_file_chars:
                    raise ValueError(
                        f"{path} exceeds raw edit ceiling of {cfg.max_file_chars} characters."
                    )
                known_exists.add(path)
            else:
                originals[path] = ""

            staged[path] = originals[path]

        before = staged[path]
        effective_operation = edit.operation
        path_exists = path in known_exists

        anchor_rebased = False
        anchor_score = 1.0

        if edit.operation == "create":
            if edit.old.strip():
                raise ValueError(
                    f"Create operation for {path} must use an empty old value."
                )
            if path_exists:
                if before.rstrip("\n") == edit.new.rstrip("\n"):
                    after = before
                    effective_operation = "create"
                else:
                    # Preserve the previous forgiving behavior, but record what the
                    # reconciler actually did.
                    after = edit.new
                    effective_operation = "full_file_replace"
            else:
                after = edit.new
                known_exists.add(path)

        elif edit.operation == "replace":
            if not path_exists:
                raise FileNotFoundError(path)
            anchor, anchor_rebased, anchor_score = _resolve_patch_anchor(
                before,
                edit.old,
                path=path,
                cfg=cfg,
            )
            after = apply_exact_replace(before, anchor, edit.new, path=path)

        elif edit.operation == "full_file_replace":
            anchor_rebased = False
            anchor_score = 1.0
            if not path_exists:
                raise FileNotFoundError(path)
            after = edit.new

        elif edit.operation in {"insert_after", "insert_before"}:
            if not path_exists:
                raise FileNotFoundError(path)
            anchor, anchor_rebased, anchor_score = _resolve_patch_anchor(
                before,
                edit.old,
                path=path,
                cfg=cfg,
            )
            replacement = (
                anchor + edit.new
                if edit.operation == "insert_after"
                else edit.new + anchor
            )
            after = apply_exact_replace(before, anchor, replacement, path=path)

        elif edit.operation == "append":
            anchor_rebased = False
            anchor_score = 1.0
            if not path_exists:
                raise FileNotFoundError(path)
            after = before + edit.new

        else:
            raise ValueError(
                f"Unsupported edit operation for {path}: {edit.operation}"
            )

        staged[path] = after
        metadata[path] = {
            "operation": effective_operation,
            "requested_operation": edit.operation,
            "reason": edit.reason,
            "anchor_rebased": "true" if anchor_rebased else "false",
            "anchor_similarity": f"{anchor_score:.4f}",
        }

    return originals, original_exists, staged, metadata, dedupe(intended_paths)


def _rollback_unit(
    *,
    repo_root: Path,
    originals: dict[str, str],
    original_exists: dict[str, bool],
    written_paths: list[str],
) -> None:
    """Best-effort rollback for a partially committed implementation unit."""

    for path in reversed(written_paths):
        original = originals[path]
        target = (repo_root / path).resolve()
        root = repo_root.resolve()
        if target != root and root not in target.parents:
            continue
        try:
            if original_exists.get(path, False):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(original, encoding="utf-8")
            elif target.exists():
                target.unlink()
        except OSError:
            # The original write error is more useful than a secondary rollback error.
            pass


def _upsert_file_change(
    changes: list[dict[str, Any]],
    change: dict[str, Any],
) -> None:
    """Keep one user-visible change record per path across repair iterations."""

    path = str(change.get("path", ""))
    for index, existing in enumerate(changes):
        if str(existing.get("path", "")) != path:
            continue
        merged = dict(existing)
        merged.update(change)
        # Approval/review should diff from the repository state at the start of the
        # coding run, not from the previous repair iteration.
        merged["original"] = existing.get("original", change.get("original", ""))
        merged["status"] = "modified" if merged.get("original") else change.get(
            "status", "modified"
        )
        changes[index] = merged
        return
    changes.append(change)


def _upsert_diff(
    diffs: list[str],
    *,
    path: str,
    original: str,
    modified: str,
) -> None:
    """Keep one final unified diff per path instead of retry-history noise."""

    prefix_a = f"--- {path}"
    prefix_b = f"+++ {path}"
    replacement = unified_diff(path, original, modified)
    for index, existing in enumerate(diffs):
        if existing.startswith(prefix_a) or (
            prefix_a in existing[: max(200, len(prefix_a) + 20)]
            and prefix_b in existing[: max(400, len(prefix_b) + 40)]
        ):
            diffs[index] = replacement
            return
    diffs.append(replacement)


def _proposal_paths(
    *,
    repo_root: Path,
    repo_files: list[str],
    result: dict[str, Any],
) -> list[str]:
    edits = [item for item in result.get("edits", []) if isinstance(item, dict)]
    if not edits:
        return []
    return _intended_edit_paths(
        repo_root=repo_root,
        repo_files=repo_files,
        raw_edits=edits,
    )


def _overlapping_proposal_ids(
    *,
    repo_root: Path,
    repo_files: list[str],
    results_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    owners: dict[str, list[str]] = {}
    for unit_id, result in results_by_id.items():
        if str(result.get("status", "")) != "proposed":
            continue
        try:
            paths = _proposal_paths(
                repo_root=repo_root,
                repo_files=repo_files,
                result=result,
            )
        except Exception:
            continue
        for path in paths:
            owners.setdefault(path, []).append(unit_id)

    conflict_paths = sorted(path for path, unit_ids in owners.items() if len(unit_ids) > 1)
    conflict_ids = dedupe(
        unit_id
        for path in conflict_paths
        for unit_id in owners.get(path, [])
    )
    return conflict_ids, conflict_paths


def _reasoning_reconcile_conflicts(
    *,
    state: CodingAgentState,
    cfg: CodingAgentSettings,
    repo_root: Path,
    repo_files: list[str],
    results_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int, str]:
    conflict_ids, conflict_paths = _overlapping_proposal_ids(
        repo_root=repo_root,
        repo_files=repo_files,
        results_by_id=results_by_id,
    )
    used = int(state.get("reasoning_reconciliations_used", 0))
    if (
        len(conflict_ids) < 2
        or used >= max(0, cfg.max_reasoning_reconciliations)
        or not conflict_paths
    ):
        return results_by_id, used, ""

    units_by_id = _unit_by_id(state)
    source_blocks: list[str] = []
    for path in conflict_paths:
        try:
            text = read_file(repo_root, path, max_chars=cfg.max_file_chars + 1)
        except Exception:
            continue
        if len(text) > cfg.max_file_chars:
            continue
        if len(text) <= cfg.max_full_file_chars:
            source_blocks.append(f"File: {path}\nContent-Status: complete\n```\n{text}\n```")
        else:
            # For large files, center chunks around exact old anchors from the worker proposals.
            anchors: list[str] = []
            for unit_id in conflict_ids:
                for edit in results_by_id[unit_id].get("edits", []):
                    if str(edit.get("path", "")).strip().replace("\\", "/") != path:
                        continue
                    old = str(edit.get("old", ""))
                    if old:
                        anchors.append(old[:120])
            windows = _char_windows(
                text,
                terms=anchors or [Path(path).stem],
                chunk_chars=cfg.context_chunk_chars,
                overlap_chars=cfg.context_chunk_overlap_chars,
                max_windows=4,
            )
            chunks = "\n\n".join(text[start:end] for start, end in windows)
            source_blocks.append(
                f"File: {path}\nContent-Status: selected-conflict-chunks\n```\n{chunks}\n```"
            )

    proposal_blocks = [
        (
            f"Unit {unit_id}: {results_by_id[unit_id].get('objective', '')}\n"
            f"Summary: {results_by_id[unit_id].get('summary', '')}\n"
            f"Edits: {results_by_id[unit_id].get('edits', [])}"
        )
        for unit_id in conflict_ids
    ]
    unit_blocks = [
        (
            f"{unit_id}: {units_by_id.get(unit_id, {}).get('objective', '')}\n"
            f"Acceptance: {units_by_id.get(unit_id, {}).get('acceptance_criteria', [])}"
        )
        for unit_id in conflict_ids
    ]

    profile = _model_profile(role="reasoning", cfg=cfg)
    included_source, _, _ = fit_blocks_to_token_budget(
        source_blocks,
        max_tokens=max(1, profile.max_input_tokens // 2),
    )
    prompt = build_reconciler_user_prompt(
        request=state.get("user_request", ""),
        units="\n\n".join(unit_blocks),
        proposals="\n\n".join(proposal_blocks),
        current_source="\n\n".join(included_source),
    )

    try:
        decision: ReconciliationDecision = invoke_parsed_decision(
            model=_worker_model(profile, cache_namespace="reasoning-reconciler"),
            schema=ReconciliationDecision,
            node_name="reasoning_reconciler",
            state={**state, "reconciliation_unit_ids": conflict_ids},
            system_prompt=RECONCILER_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_attempts=1,
        )
    except Exception as exc:
        return results_by_id, used + 1, f"Reasoning reconciliation failed: {exc}"

    requested_covered = {unit_id for unit_id in decision.unit_ids if unit_id in conflict_ids}
    covered = [
        str(unit.get("id", ""))
        for unit in state.get("implementation_units", [])
        if str(unit.get("id", "")) in requested_covered
    ]
    if not decision.edits or len(covered) < 2:
        reason = decision.blocking_reason or "Reasoning reconciler returned no merged patch."
        return results_by_id, used + 1, reason

    primary_id = covered[0]
    merged_results = dict(results_by_id)
    merged_results[primary_id] = {
        **merged_results[primary_id],
        "status": "proposed",
        "summary": decision.summary or "Reasoning reconciler merged overlapping proposals.",
        "edits": [item.model_dump() for item in decision.edits],
        "validation_commands": dedupe(
            [
                *merged_results[primary_id].get("validation_commands", []),
                *decision.validation_commands,
            ]
        ),
        "reconciled_unit_ids": covered,
        "reconciliation_model": profile.model_name,
        "reconciliation_provider": profile.provider,
    }
    for unit_id in covered[1:]:
        merged_results[unit_id] = {
            **merged_results[unit_id],
            "status": "reconciled_pending",
            "reconciliation_owner": primary_id,
            "summary": f"Pending merged reconciliation owned by {primary_id}.",
            "edits": [],
            "validation_commands": [],
        }

    return merged_results, used + 1, decision.summary


def reconcile_subtask_patches_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Deterministically reconcile worker patch proposals and update the ledger."""

    cfg = _settings_for_state(state, cfg)
    repo_root = resolve_repo_root(state, cfg)
    allow_write = resolve_allow_write(state, cfg)
    run_id = str(state.get("implementation_run_id", ""))
    generation = int(state.get("implementation_generation", 0))
    results_by_id = {
        str(item.get("unit_id", "")): item
        for item in state.get("subtask_worker_results", [])
        if str(item.get("run_id", "")) == run_id
        and int(item.get("generation", -1)) == generation
        and str(item.get("unit_id", "")).strip()
    }
    ledger = deepcopy(state.get("completion_ledger") or {})
    file_changes = list(state.get("file_changes", []))
    diffs = list(state.get("diffs", []))
    errors = list(state.get("errors", []))
    validation_commands = list(state.get("validation_commands", []))
    claimed_paths: set[str] = set()
    repo_files = filter_context_paths(list_files(repo_root, ".", max_depth=12))
    summary_lines: list[str] = []

    results_by_id, reconciliations_used, reconciliation_note = _reasoning_reconcile_conflicts(
        state=state,
        cfg=cfg,
        repo_root=repo_root,
        repo_files=repo_files,
        results_by_id=results_by_id,
    )
    if reconciliation_note:
        summary_lines.append(f"- reasoning reconciliation: {reconciliation_note}")

    for unit in state.get("implementation_units", []):
        unit_id = str(unit.get("id", "")).strip()
        result = results_by_id.get(unit_id)
        if not result:
            continue

        entry = dict(ledger.get(unit_id) or {})
        entry["implementation_attempts"] = int(
            entry.get("implementation_attempts", 0)
        ) + 1
        entry["patch_retries"] = int(entry.get("patch_retries", 0)) + int(
            result.get("patch_retries", 0)
        )
        entry["files_inspected"] = dedupe(
            [
                *entry.get("files_inspected", []),
                *[str(path) for path in result.get("files_inspected", [])],
            ]
        )
        if result.get("model"):
            entry["last_model"] = result.get("model")
            entry["last_provider"] = result.get("provider")
            entry["last_context_tokens"] = result.get("context_tokens", 0)
            entry["last_context_budget_tokens"] = result.get(
                "context_budget_tokens", 0
            )
        if result.get("reconciliation_model"):
            entry["last_reconciliation_model"] = result.get("reconciliation_model")
            entry["last_reconciliation_provider"] = result.get("reconciliation_provider")

        result_status = str(result.get("status", "retryable"))
        if result_status == "reconciled_pending":
            owner_id = str(result.get("reconciliation_owner", "")).strip()
            owner_entry = dict(ledger.get(owner_id) or {})
            owner_status = str(owner_entry.get("status", "retryable"))
            if owner_status in SUCCESSFUL_UNIT_STATUSES:
                entry["status"] = owner_status
                entry["last_error"] = ""
                entry["summary"] = str(result.get("summary", ""))
            else:
                entry["status"] = "retryable"
                entry["last_error"] = (
                    f"Reasoning reconciliation owner {owner_id} did not apply successfully."
                )
            ledger[unit_id] = entry
            summary_lines.append(
                f"- {unit_id}: {entry['status']} via reasoning reconciliation owner {owner_id}."
            )
            continue
        if result_status == "completed":
            entry["status"] = "completed"
            entry["last_error"] = ""
            entry["summary"] = str(result.get("summary", ""))
            ledger[unit_id] = entry
            validation_commands.extend(result.get("validation_commands", []))
            summary_lines.append(f"- {unit_id}: completed; no repository edit needed.")
            continue

        if result_status != "proposed":
            entry["status"] = "retryable"
            entry["last_error"] = str(
                result.get("blocking_reason")
                or result.get("summary")
                or "Worker did not produce a patch proposal."
            )
            ledger[unit_id] = entry
            summary_lines.append(f"- {unit_id}: retryable - {entry['last_error']}")
            continue

        raw_edits = [
            item for item in result.get("edits", []) if isinstance(item, dict)
        ]
        if not raw_edits:
            entry["status"] = "retryable"
            entry["last_error"] = "Worker reported proposed status without edits."
            ledger[unit_id] = entry
            summary_lines.append(f"- {unit_id}: retryable - proposal contained no edits.")
            continue

        try:
            intended_paths = _intended_edit_paths(
                repo_root=repo_root,
                repo_files=repo_files,
                raw_edits=raw_edits,
            )
        except Exception as exc:
            entry["status"] = "retryable"
            entry["last_error"] = f"Patch path validation failed: {exc}"
            ledger[unit_id] = entry
            errors.append(f"Unit {unit_id} patch path validation failed: {exc}")
            summary_lines.append(f"- {unit_id}: retryable - invalid patch path.")
            continue

        # Detect same-generation ownership conflicts *before* staging. Earlier units
        # may already have changed the sandbox, so staging a stale overlapping edit
        # first would turn a predictable scheduling conflict into a misleading
        # exact-replace failure.
        conflicts = sorted(set(intended_paths) & claimed_paths)
        if conflicts:
            entry["status"] = "retryable"
            entry["last_error"] = (
                "Patch overlaps a file already reconciled in this generation: "
                + ", ".join(conflicts)
            )
            ledger[unit_id] = entry
            summary_lines.append(
                f"- {unit_id}: retryable - overlapping files deferred to next iteration."
            )
            continue

        try:
            originals, original_exists, staged, metadata, intended_paths = _stage_unit_edits(
                repo_root=repo_root,
                repo_files=repo_files,
                raw_edits=raw_edits,
                cfg=cfg,
            )
        except Exception as exc:
            entry["status"] = "retryable"
            entry["last_error"] = f"Patch staging failed: {exc}"
            ledger[unit_id] = entry
            errors.append(f"Unit {unit_id} patch staging failed: {exc}")
            summary_lines.append(f"- {unit_id}: retryable - patch staging failed.")
            continue

        changed_paths = [
            path for path in intended_paths
            if staged.get(path, originals.get(path, "")) != originals.get(path, "")
        ]

        written_paths: list[str] = []
        unit_changes: list[dict[str, Any]] = []

        try:
            for path in changed_paths:
                before = originals[path]
                after = staged[path]
                result_text = write_file(
                    repo_root,
                    path,
                    after,
                    allow_write=allow_write,
                )
                if allow_write:
                    written_paths.append(path)
                meta = metadata[path]
                unit_changes.append(
                    {
                        "path": path,
                        "operation": meta["operation"],
                        "requested_operation": meta["requested_operation"],
                        "anchor_rebased": meta.get("anchor_rebased") == "true",
                        "anchor_similarity": float(meta.get("anchor_similarity", "1.0")),
                        "status": "modified" if original_exists.get(path, False) else "added",
                        "reason": meta["reason"],
                        "write_result": result_text,
                        "original": before,
                        "modified": after,
                        "implementation_unit": unit_id,
                    }
                )
        except Exception as exc:
            if allow_write:
                _rollback_unit(
                    repo_root=repo_root,
                    originals=originals,
                    original_exists=original_exists,
                    written_paths=written_paths,
                )
            entry["status"] = "retryable"
            entry["last_error"] = f"Patch commit failed and was rolled back: {exc}"
            ledger[unit_id] = entry
            errors.append(f"Unit {unit_id} patch commit failed: {exc}")
            summary_lines.append(f"- {unit_id}: retryable - patch commit failed.")
            continue

        for change in unit_changes:
            prior_original = next(
                (
                    str(existing.get("original", ""))
                    for existing in file_changes
                    if str(existing.get("path", "")) == str(change.get("path", ""))
                ),
                str(change.get("original", "")),
            )
            _upsert_file_change(file_changes, change)
            _upsert_diff(
                diffs,
                path=str(change.get("path", "")),
                original=prior_original,
                modified=str(change.get("modified", "")),
            )
        claimed_paths.update(changed_paths)
        validation_commands.extend(result.get("validation_commands", []))
        validation_commands.extend(unit.get("validation_commands", []))

        entry["status"] = "implemented" if allow_write else "proposed"
        entry["last_error"] = ""
        entry["summary"] = str(result.get("summary", ""))
        entry["files_changed"] = dedupe(
            [*entry.get("files_changed", []), *changed_paths]
        )
        ledger[unit_id] = entry
        summary_lines.append(
            f"- {unit_id}: {entry['status']} ({len(changed_paths)} changed file(s))."
        )

    mode = "WRITE MODE" if allow_write else "DRY RUN"
    return {
        "completion_ledger": ledger,
        "file_changes": file_changes,
        "diffs": diffs,
        "validation_commands": dedupe(
            [str(command) for command in validation_commands if str(command).strip()]
        ),
        "patch_summary": (
            f"{mode}: reconciled implementation generation {generation}.\n"
            + ("\n".join(summary_lines) if summary_lines else "No worker proposals were available.")
        ),
        "errors": dedupe(errors),
        "reasoning_reconciliations_used": reconciliations_used,
        "continue_loop": not _implementation_complete(
            {**state, "completion_ledger": ledger}
        ),
        "status": "patches_reconciled",
    }


def _blocking_dependencies(
    unit: dict[str, Any],
    ledger: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        dep
        for dep in unit.get("depends_on", [])
        if str((ledger.get(dep) or {}).get("status", "pending"))
        in {"blocked", "failed"}
    ]


def assess_progress_node(
    state: CodingAgentState,
    cfg: CodingAgentSettings = default_settings,
) -> CodingAgentState:
    """Advance fresh batches freely while tightly bounding true repair rounds."""

    cfg = _settings_for_state(state, cfg)
    ledger = deepcopy(state.get("completion_ledger") or {})
    iteration = max(1, int(state.get("implementation_iteration", 1)))
    max_iterations = max(
        1,
        int(
            state.get(
                "max_implementation_iterations",
                cfg.max_implementation_iterations,
            )
        ),
    )
    loop_notes = list(state.get("loop_notes", []))
    units = _unit_by_id(state)

    # Only genuine code-level blocking validation reaches this node. Infrastructure
    # failures are downgraded to advisory by validate_node and therefore never reopen code.
    if bool(state.get("blocking_validation_failed")):
        changed_unit_ids = {
            str(item.get("implementation_unit", "")).strip()
            for item in state.get("file_changes", [])
            if str(item.get("implementation_unit", "")).strip()
        }
        reopen = [
            unit_id
            for unit_id, entry in ledger.items()
            if str(entry.get("status", "")) in SUCCESSFUL_UNIT_STATUSES
            and (unit_id in changed_unit_ids or not changed_unit_ids)
        ]
        validation_feedback = format_failed_validation_results(
            state.get("validation_results", [])
        )
        for unit_id in reopen:
            entry = dict(ledger[unit_id])
            entry["status"] = "retryable"
            entry["last_error"] = (
                "Blocking code validation failed after reconciliation. "
                + validation_feedback[:2_000]
            )
            ledger[unit_id] = entry

    for unit_id, unit in units.items():
        entry = dict(ledger.get(unit_id) or {})
        if str(entry.get("status", "pending")) not in RETRYABLE_UNIT_STATUSES:
            continue
        blockers = _blocking_dependencies(unit, ledger)
        if blockers:
            entry["status"] = "blocked"
            entry["last_error"] = "Blocked by failed unit(s): " + ", ".join(blockers)
            ledger[unit_id] = entry

    if _implementation_complete({**state, "completion_ledger": ledger}):
        return {
            "completion_ledger": ledger,
            "continue_loop": False,
            "progress_reason": "All implementation units are complete.",
            "status": "assessed",
        }

    unfinished = [
        unit_id
        for unit_id, entry in ledger.items()
        if str(entry.get("status", "pending")) not in SUCCESSFUL_UNIT_STATUSES
    ]

    pending_ready = [
        unit_id
        for unit_id, unit in units.items()
        if str((ledger.get(unit_id) or {}).get("status", "pending")) == "pending"
        and _dependencies_satisfied(unit, ledger)
    ]
    retry_ready = [
        unit_id
        for unit_id, unit in units.items()
        if str((ledger.get(unit_id) or {}).get("status", "pending")) == "retryable"
        and _dependencies_satisfied(unit, ledger)
    ]

    # More units than worker slots simply require another generation; this is not a retry.
    if pending_ready:
        focus = "Schedule fresh dependency-ready implementation units: " + ", ".join(pending_ready)
        return {
            "completion_ledger": ledger,
            "implementation_iteration": iteration,
            "continue_loop": True,
            "remaining_tasks": pending_ready,
            "progress_reason": focus,
            "loop_context_focus": focus,
            "loop_notes": [*loop_notes, focus][-8:],
            "search_results": [],
            "repo_navigation_summary": "",
            "repo_navigation_files": [],
            "repo_navigation_missing_context": [],
            "blocking_validation_failed": False,
            "status": "assessed",
        }

    if retry_ready:
        if iteration >= max_iterations:
            for unit_id in retry_ready:
                entry = dict(ledger.get(unit_id) or {})
                entry["status"] = "failed"
                entry["last_error"] = (
                    entry.get("last_error")
                    or f"Repair-round limit reached ({iteration}/{max_iterations})."
                )
                ledger[unit_id] = entry
            return {
                "completion_ledger": ledger,
                "implementation_iteration": iteration,
                "continue_loop": False,
                "remaining_tasks": retry_ready,
                "progress_reason": f"Repair-round limit reached at {iteration}/{max_iterations}.",
                "loop_notes": [*loop_notes, "Deterministic repair limit reached."][-8:],
                "status": "loop_limit_reached",
            }

        next_iteration = iteration + 1
        focus = (
            f"Repair round {next_iteration}/{max_iterations}. Work only on: "
            + ", ".join(retry_ready)
        )
        return {
            "completion_ledger": ledger,
            "implementation_iteration": next_iteration,
            "continue_loop": True,
            "remaining_tasks": retry_ready,
            "progress_reason": focus,
            "loop_context_focus": focus,
            "loop_notes": [*loop_notes, focus][-8:],
            "search_results": [],
            "repo_navigation_summary": "",
            "repo_navigation_files": [],
            "repo_navigation_missing_context": [],
            "blocking_validation_failed": False,
            "status": "assessed",
        }

    for unit_id in unfinished:
        entry = dict(ledger.get(unit_id) or {})
        if str(entry.get("status", "pending")) in RETRYABLE_UNIT_STATUSES:
            entry["status"] = "blocked"
            entry["last_error"] = entry.get("last_error") or "No dependency-ready implementation path remains."
            ledger[unit_id] = entry

    return {
        "completion_ledger": ledger,
        "continue_loop": False,
        "remaining_tasks": unfinished,
        "progress_reason": "No dependency-ready implementation units remain.",
        "status": "assessed",
    }

