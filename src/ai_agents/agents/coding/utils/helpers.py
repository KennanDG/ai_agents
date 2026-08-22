from pathlib import Path
from ai_agents.agents.coding.utils.constants import PATCH_CONTEXT_MAX_CHARS
from ai_agents.agents.coding.utils.search import (
    derive_search_requests,
    filter_context_paths,
    legacy_queries_to_search_requests,
)
from ai_agents.agents.coding.coding_agent_schemas import (
    PlanDecision,
    RepoNavigationDecision,
)
from ai_agents.agents.coding.coding_agent_settings import CodingAgentSettings 
from ai_agents.agents.coding.state import CodingAgentState
from ai_agents.agents.coding.utils.text import bullets, dedupe



from ai_agents.agents.coding.skill_registry import SkillRegistry, route_skill
from ai_agents.agents.coding.coding_agent_schemas import (
    PlanDecision,
    RepoNavigationDecision,
)

from ai_agents.agents.coding.utils.search import (
    derive_search_requests,
    filter_context_paths,
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



def _search_requests_from_state(state: CodingAgentState) -> list[dict[str, object]]:
    search_requests = list(state.get("search_requests") or [])

    if search_requests:
        return search_requests

    # Fallback
    search_requests = legacy_queries_to_search_requests(state.get("search_queries", []))

    if search_requests:
        return search_requests

    return derive_search_requests(state["user_request"]) # Extra fallback



def _format_search_result_dicts(results: list[dict[str, object]]) -> str:
    lines: list[str] = []

    for result in results:
        path = str(result.get("path", "")).strip()
        if not path:
            continue

        line_no = result.get("line_no") or 1
        snippet = str(result.get("snippet", "")).strip()
        score = result.get("score", 0.0)
        reason = str(result.get("reason", "")).strip()

        try:
            score_text = f"{float(score):.1f}"
        except (TypeError, ValueError):
            score_text = "0.0"

        lines.append(f"{path}:{line_no}: {snippet} [score={score_text}; {reason}]")

    return "\n".join(lines)



def _repo_navigation_path_reasons(
    decision: RepoNavigationDecision,
    search_results: list[dict[str, object]],
) -> dict[str, str]:
    reasons = {
        item.path.strip(): item.reason.strip()
        for item in decision.files_to_inspect
        if item.path.strip()
    }

    for result in search_results:
        path = str(result.get("path", "")).strip()
        if not path or path in reasons:
            continue

        reason = str(result.get("reason", "")).strip()
        score = result.get("score", 0.0)
        try:
            reason = f"Ranked search fallback: {reason} (score={float(score):.1f})"
        except (TypeError, ValueError):
            reason = f"Ranked search fallback: {reason}"

        reasons[path] = reason

    return reasons




def _route_with_fallback(
    *,
    state: CodingAgentState,
    registry: SkillRegistry,
    error: str | None = None,
) -> CodingAgentState:
    errors = list(state.get("errors", []))
    if error:
        errors.append(error)

    selected_skill = route_skill(
        state["user_request"],
        registry.list_names(),
        default_skill=registry.default_skill_name(),
    )
    skill = registry.get(selected_skill)

    return {
        "selected_skill": skill.name,
        "skill_instructions": skill.instructions,
        "route_confidence": 0.0,
        "route_reason": "Deterministic fallback route was used.",
        "route_alternatives": [],
        "errors": errors,
        "status": "routed",
    }




def _repo_attachment_paths(state: CodingAgentState) -> list[str]:
    """Return explicit repo-relative attachment paths in user-provided order."""

    paths: list[str] = []

    for item in state.get("attached_files") or []:
        if str(item.get("source", "upload")).strip() != "repo":
            continue

        path = str(item.get("path", "") or "").strip()
        if path:
            paths.append(path)

    return dedupe(filter_context_paths(paths))



def _format_attached_files_for_context(state: CodingAgentState) -> tuple[str, list[str]]:
    """Format only external uploads as inline context.

    Repository attachments are represented by their repo-relative paths and loaded
    through the normal repository reader. This prevents the same file contents from
    being injected once as an attachment and again as an inspected repository file.
    """

    attached_files = list(state.get("attached_files") or [])

    if not attached_files:
        return "", []

    blocks: list[str] = []
    used: list[str] = []

    for item in attached_files:
        name = str(item.get("name", "")).strip() or "attachment"
        source = str(item.get("source", "upload")).strip()
        path = str(item.get("path", "") or "").strip()
        content = str(item.get("content", ""))
        truncated = bool(item.get("truncated", False))

        label = path if source == "repo" and path else name
        used.append(f"{source}:{label}")

        if source == "repo" and path:
            continue

        header = f"Attachment: {label} | source={source or 'upload'}"
        if truncated:
            header += " | truncated=true"

        blocks.append(
            f"{header}\n"
            "```text\n"
            f"{content}\n"
            "```"
        )

    if not blocks:
        return "", used

    introduction = [
        "# External user-attached files",
        (
            "These uploads are read-only context and may not exist in the repository. "
            "Only edit files loaded from the repository through normal patch operations."
        ),
    ]

    return "\n\n".join([*introduction, *blocks]), used



def _attached_file_summary(state: CodingAgentState) -> str:
    attached_files = list(state.get("attached_files") or [])

    if not attached_files:
        return ""

    lines: list[str] = []

    for item in attached_files:
        name = str(item.get("name", "")).strip() or "attachment"
        source = str(item.get("source", "upload")).strip()
        path = str(item.get("path", "") or "").strip()
        content = str(item.get("content", ""))
        label = path if path else name
        lines.append(f"- {label} ({source}, {len(content)} chars)")

    return "\n".join(lines)


def _resolve_existing_repo_path(
    *,
    repo_root: Path,
    candidate: str,
    repo_files: list[str],
) -> str | None:
    
    candidate = candidate.strip().replace("\\", "/").lstrip("/")

    if not candidate:
        return None

    candidate_path = Path(candidate)

    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        return None

    exact = (repo_root / candidate).resolve()

    try:
        root = repo_root.resolve()

        if exact.is_file() and (exact == root or root in exact.parents):
            return exact.relative_to(root).as_posix()
        
    except OSError:
        return None

    suffix_matches = [
        path for path in repo_files
        if path == candidate or path.endswith(f"/{candidate}")
    ]

    if len(suffix_matches) == 1:
        return suffix_matches[0]


    basename = candidate_path.name

    basename_matches = [
        path for path in repo_files
        if Path(path).name == basename
    ]

    if len(basename_matches) == 1:
        return basename_matches[0]

    return None



def _resolve_context_candidate_paths(
    *,
    repo_root: Path,
    candidate_paths: list[str],
    repo_files: list[str],
) -> tuple[list[str], list[str]]:

    resolved: list[str] = []
    unresolved: list[str] = []

    for candidate in candidate_paths:
        path = _resolve_existing_repo_path(
            repo_root=repo_root,
            candidate=candidate,
            repo_files=repo_files,
        )

        if path:
            resolved.append(path)
        else:
            unresolved.append(candidate)

    return dedupe(resolved), dedupe(unresolved)



def _format_loop_context_focus(state: CodingAgentState) -> str:
    """Build focused guidance for context-refresh loops."""

    lines: list[str] = []
    loop_context_focus = str(state.get("loop_context_focus", "")).strip()
    progress_reason = str(state.get("progress_reason", "")).strip()
    remaining_tasks = list(state.get("remaining_tasks") or [])
    loop_notes = list(state.get("loop_notes") or [])[-5:]

    if loop_context_focus:
        lines.append(loop_context_focus)

    if progress_reason:
        lines.append(f"Progress reason: {progress_reason}")

    if remaining_tasks:
        lines.append("Remaining tasks:")
        lines.append(bullets([str(item) for item in remaining_tasks]))

    if loop_notes:
        lines.append("Recent loop notes:")
        lines.append(bullets([str(item) for item in loop_notes]))

    return "\n".join(lines).strip()



def _derive_loop_search_requests(
    *,
    state: CodingAgentState,
    remaining_tasks: list[str],
    next_iteration_notes: str,
    reason: str,
) -> list[dict[str, object]]:
    """Derive focused search requests when the assessor did not provide any."""

    loop_search_text = "\n".join(
        item
        for item in [
            state.get("user_request", ""),
            reason,
            next_iteration_notes,
            "\n".join(remaining_tasks),
            "\n".join(str(item) for item in state.get("errors", [])[-5:]),
        ]
        if item
    )

    derived = derive_search_requests(loop_search_text) if loop_search_text else []

    return derived or list(state.get("search_requests") or [])



def _context_file_read_limit(
    *,
    path: str,
    explicit_repo_paths: set[str],
    explicit_repo_file_count: int,
    cfg: CodingAgentSettings,
) -> int:
    """Allocate more context to files explicitly attached by the user."""

    if path not in explicit_repo_paths:
        return cfg.max_file_chars

    # Keep room for the request, plan, errors, navigation notes, and supporting
    # files. Divide the remaining budget across explicit repository files.
    reserved_chars = min(20_000, PATCH_CONTEXT_MAX_CHARS // 5)
    available_chars = max(
        cfg.max_file_chars,
        PATCH_CONTEXT_MAX_CHARS - reserved_chars,
    )

    return max(
        cfg.max_file_chars,
        available_chars // max(1, explicit_repo_file_count),
    )




def _same_file_content(existing: str, requested: str) -> bool:
    """Return True when an attempted create is effectively already applied."""
    return existing == requested or existing.rstrip("\n") == requested.rstrip("\n")

