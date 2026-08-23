from __future__ import annotations

import ast
import re
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from langchain_core.output_parsers import PydanticOutputParser

from fastapi import APIRouter, HTTPException, Query

from ai_agents.agents.coding.skill_registry import SkillRegistry, extract_allowed_tools
from ai_agents.agents.coding.tool_registry import (
    CODING_TOOLS_DIR,
    ApprovedCustomToolRegistry,
    CustomToolValidationError,
    validate_approved_custom_tool_source,
)
from ai_agents.agents.voice.tool_registry import (
    VOICE_TOOLS_DIR,
    ApprovedCustomVoiceToolRegistry,
    validate_voice_custom_tool_source,
)
from ai_agents.agents.coding.model_factory import build_chat_model
from ai_agents.agents.coding.utils.text import message_content_to_text
from ai_agents.config.model_catalog import ModelCapability, discover_models
from ai_agents.config.runtime_configuration import runtime_agent_configuration
from ai_agents.config.constants import ChatProvider, AgentKind
from ai_agents.config.settings import settings as config_settings

from ai_agents.api.api_schemas import (
    NAME_RE,
    AgentConfigurationUpdate,
    SkillWriteRequest,
    SkillDraftRequest,
    SkillDraftResponse,
    SkillDraftDecision,
    ToolQuarantineRequest,
    ToolFileUpdateRequest,
    SkillSummary,
    ToolSummary,
    ToolReviewResponse,
)
from ai_agents.config.constants import (
    AI_AGENTS_ROOT,
    VOICE_SKILLS_DIR,
    CUSTOM_PREFIX,
    CODING_RUNTIME_FIELDS,
    CODING_RUNTIME_BOUNDS, 
    MODEL_CONFIGURATION_FIELDS,
)



SKILL_DIRS: dict[AgentKind, Path] = {
    # Use the same directory as the runtime SkillRegistry so custom skills saved
    # from the UI are immediately visible to route_node on the next run.
    "coding": SkillRegistry().skills_dir.resolve(),
    "voice": VOICE_SKILLS_DIR.resolve(),
}
TOOL_DIRS: dict[AgentKind, Path] = {
    # Resolve coding tools from the importable ai_agents package itself. This avoids
    # accidentally creating src/agents/coding/tools beside src/ai_agents/... .
    "coding": CODING_TOOLS_DIR,
    "voice": VOICE_TOOLS_DIR.resolve(),
}




router = APIRouter(prefix="/admin", tags=["admin"])


def _coding_runtime_path() -> Path:
    base = Path(config_settings.runtime_agent_config_path).expanduser().resolve()
    return base.with_name(f"{base.stem}-coding-runtime{base.suffix}")


def _coding_runtime_snapshot() -> dict[str, int]:
    return {field: int(getattr(config_settings, field)) for field in CODING_RUNTIME_FIELDS}


def _load_coding_runtime_configuration() -> None:
    path = _coding_runtime_path()
    if not path.exists():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(raw, dict):
        return
    for field in CODING_RUNTIME_FIELDS:
        value = raw.get(field)
        minimum, maximum = CODING_RUNTIME_BOUNDS[field]
        if isinstance(value, int) and minimum <= value <= maximum:
            setattr(config_settings, field, value)


def _save_coding_runtime_configuration(values: dict[str, int]) -> None:
    for field, value in values.items():
        setattr(config_settings, field, value)
    _atomic_write(
        _coding_runtime_path(),
        json.dumps(_coding_runtime_snapshot(), indent=2, sort_keys=True) + "\n",
    )


_load_coding_runtime_configuration()


def _safe_agent_dir(mapping: dict[AgentKind, Path], agent: AgentKind) -> Path:
    directory = mapping[agent].resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _migrate_legacy_coding_assets() -> None:
    """Move custom assets created under the old src/agents/... root into ai_agents."""

    legacy_skill_root = (AI_AGENTS_ROOT / "agents" / "coding" / "skills").resolve()
    canonical_skill_root = SKILL_DIRS["coding"].resolve()

    if legacy_skill_root != canonical_skill_root and legacy_skill_root.exists():
        canonical_skill_root.mkdir(parents=True, exist_ok=True)

        for source in legacy_skill_root.glob("custom_*.md"):
            target = canonical_skill_root / source.name
            if not target.exists():
                os.replace(source, target)

    legacy_tool_root = (AI_AGENTS_ROOT / "agents" / "coding" / "tools").resolve()
    canonical_tool_root = TOOL_DIRS["coding"].resolve()

    if legacy_tool_root != canonical_tool_root and legacy_tool_root.exists():
        for directory_name in ("custom_pending", "custom_approved"):
            source_dir = legacy_tool_root / directory_name
            target_dir = canonical_tool_root / directory_name
            if not source_dir.exists():
                continue

            target_dir.mkdir(parents=True, exist_ok=True)
            for source in source_dir.glob("*.py"):
                target = target_dir / source.name
                if not target.exists():
                    os.replace(source, target)


def _migrate_legacy_voice_assets() -> None:
    """Move voice assets created under the old src/agents/... root into ai_agents."""

    legacy_root = (AI_AGENTS_ROOT / "agents" / "voice").resolve()
    canonical_skill_root = SKILL_DIRS["voice"].resolve()
    canonical_tool_root = TOOL_DIRS["voice"].resolve()

    legacy_skill_root = legacy_root / "skills"

    if legacy_skill_root != canonical_skill_root and legacy_skill_root.exists():
        canonical_skill_root.mkdir(parents=True, exist_ok=True)
        for source in legacy_skill_root.glob("custom_*.md"):
            target = canonical_skill_root / source.name
            if not target.exists():
                os.replace(source, target)

    legacy_tool_root = legacy_root / "tools"

    if legacy_tool_root != canonical_tool_root and legacy_tool_root.exists():
        for directory_name in ("custom_pending", "custom_approved"):
            source_dir = legacy_tool_root / directory_name
            target_dir = canonical_tool_root / directory_name
            if not source_dir.exists():
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            for source in source_dir.glob("*.py"):
                target = target_dir / source.name
                if not target.exists():
                    os.replace(source, target)



def _migrate_agent_assets(agent: AgentKind) -> None:
    if agent == "coding":
        _migrate_legacy_coding_assets()
    else:
        _migrate_legacy_voice_assets()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, path)


def _validate_skill_markdown(content: str) -> str:
    """Enforce the canonical skill document contract used by the runtime and UI."""

    normalized = content.replace("\r\n", "\n").strip() + "\n"
    if "\x00" in normalized:
        raise HTTPException(status_code=400, detail="Skill content contains a null byte.")

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines or not re.match(r"^#\s+Skill:\s+.+$", lines[0], flags=re.IGNORECASE):
        raise HTTPException(
            status_code=400,
            detail="Skill Markdown must start with '# Skill: <display name>'.",
        )

    if not any(line.lower().startswith("purpose:") and line.split(":", 1)[1].strip() for line in lines):
        raise HTTPException(
            status_code=400,
            detail="Skill Markdown must include a non-empty top-level 'Purpose:' line.",
        )

    normalized_labels = {
        re.sub(r"^#{1,6}\s+", "", line).lower().rstrip(":")
        for line in lines
    }
    required_sections = ("use when", "allowed tools", "steps", "rules")
    missing = [section for section in required_sections if section not in normalized_labels]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Skill Markdown is not in the canonical format. Missing section(s): "
                + ", ".join(missing)
                + ". Expected: # Skill, Purpose, Use when, Allowed tools, Steps, Rules."
            ),
        )

    return normalized


def _normalize_custom_skill_name(value: str, *, fallback: str = "custom_skill") -> str:
    normalized = value.strip().lower().replace("-", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized).strip("_") or fallback
    if not normalized[0].isalpha():
        normalized = f"skill_{normalized}"
    if not normalized.startswith(CUSTOM_PREFIX):
        normalized = f"{CUSTOM_PREFIX}{normalized}"
    return normalized[:128]


def _executable_tool_catalog(agent: AgentKind) -> list[ToolSummary]:
    return [
        tool
        for tool in list_tools(agent)
        if tool.status in {"builtin", "approved"}
    ]


def _validate_skill_tool_references(agent: AgentKind, content: str) -> list[str]:
    declared = list(extract_allowed_tools(content))
    if not declared:
        return []

    available = {tool.name for tool in _executable_tool_catalog(agent)}
    missing = [tool for tool in declared if tool not in available]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Skill references tools that are not executable for this agent: "
                + ", ".join(missing)
                + ". Pending-review tools do not count as available until approved."
            ),
        )
    return declared


def _render_skill_markdown(
    *,
    display_name: str,
    purpose: str,
    use_when: list[str],
    allowed_tools: list[str],
    steps: list[str],
    rules: list[str],
    unavailable_tools: list[str] | None = None,
) -> str:
    safe_use_when = [item.strip() for item in use_when if item.strip()] or [
        "The user's request matches this skill's stated purpose."
    ]
    safe_steps = [item.strip() for item in steps if item.strip()] or [
        "Inspect the relevant context.",
        "Perform the requested work using existing project patterns.",
        "Validate the result and report any limitations.",
    ]
    safe_rules = [item.strip() for item in rules if item.strip()] or [
        "Do not expose secrets.",
        "Avoid unrelated changes.",
    ]
    unavailable = [item for item in unavailable_tools or [] if item]
    if unavailable:
        safe_rules.append(
            "Unavailable tool dependency detected during import/generation: "
            + ", ".join(unavailable)
            + ". Do not assume these tools exist; use only the Allowed tools above."
        )

    tool_lines = "\n".join(f"- {tool}" for tool in allowed_tools)
    use_lines = "\n".join(f"- {item}" for item in safe_use_when)
    step_lines = "\n".join(f"{index}. {item}" for index, item in enumerate(safe_steps, start=1))
    rule_lines = "\n".join(f"- {item}" for item in safe_rules)

    return (
        f"# Skill: {display_name.strip()}\n\n"
        f"Purpose: {purpose.strip()}\n\n"
        f"Use when:\n{use_lines}\n\n"
        f"Allowed tools:\n{tool_lines}\n\n"
        f"Steps:\n{step_lines}\n\n"
        f"Rules:\n{rule_lines}\n"
    )


def _generate_skill_draft(request: SkillDraftRequest) -> SkillDraftResponse:
    tools = _executable_tool_catalog(request.agent)
    available_names = {tool.name for tool in tools}
    tool_catalog = "\n".join(
        f"- {tool.name}: {tool.purpose or tool.module}"
        for tool in tools
    ) or "- No executable tools are currently registered for this agent."

    parser = PydanticOutputParser(pydantic_object=SkillDraftDecision)
    source_context = (
        f"\n\n# Imported Markdown to preserve\n{request.source_markdown}"
        if request.source_markdown
        else ""
    )
    suggested = request.suggested_name or "custom_skill"
    system_prompt = (
        "You create safe AI-agent skill playbooks. Return only the requested structured object. "
        "Preserve the user's intent and operational behavior. Do not invent tools. "
        "Allowed tools must be selected only from the supplied executable tool catalog. "
        "Keep the skill narrow, practical, and non-destructive."
    )
    user_prompt = f"""
Create or normalize a custom skill for the {request.agent} agent.

# User instruction
{request.prompt}

# Suggested registry name
{suggested}

# Executable tool catalog
{tool_catalog}
{source_context}

# Requirements
- registry_name must be lowercase snake_case and start with custom_.
- Preserve the intent of imported Markdown when it is present.
- Do not list a tool unless its exact name appears in the executable tool catalog.
- Use concise use_when conditions, concrete ordered steps, and explicit safety rules.

{parser.get_format_instructions()}
""".strip()

    try:
        model = build_chat_model(
            provider=config_settings.coding_provider,
            model_name=config_settings.coding_model,
            max_tokens=2_400,
            temperature=0.1,
        )
        response = model.invoke(
            [("system", system_prompt), ("human", user_prompt)],
            config={
                "run_name": "admin_skill_draft",
                "tags": ["admin", "skill-draft", request.agent],
            },
        )
        decision = parser.parse(message_content_to_text(response.content))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Skill generation failed: {exc}",
        ) from exc

    requested_tools = list(dict.fromkeys(tool.strip() for tool in decision.allowed_tools if tool.strip()))
    valid_tools = [tool for tool in requested_tools if tool in available_names]
    missing_tools = [tool for tool in requested_tools if tool not in available_names]

    # Also preserve tool dependencies explicitly named by imported Markdown. They
    # are never placed into Allowed tools unless the backend really exposes them.
    imported_tools = list(extract_allowed_tools(request.source_markdown or ""))
    for tool in imported_tools:
        if tool in available_names and tool not in valid_tools:
            valid_tools.append(tool)
        elif tool not in available_names and tool not in missing_tools:
            missing_tools.append(tool)

    name = _normalize_custom_skill_name(decision.registry_name or suggested)
    content = _render_skill_markdown(
        display_name=decision.display_name,
        purpose=decision.purpose,
        use_when=decision.use_when,
        allowed_tools=valid_tools,
        steps=decision.steps,
        rules=decision.rules,
        unavailable_tools=missing_tools,
    )
    
    content = _validate_skill_markdown(content)

    warnings: list[str] = []
    if missing_tools:
        warnings.append(
            "Some requested/imported tools are not executable and were omitted from Allowed tools: "
            + ", ".join(missing_tools)
        )

    return SkillDraftResponse(
        agent=request.agent,
        name=name,
        purpose=decision.purpose.strip(),
        allowed_tools=valid_tools,
        missing_tools=missing_tools,
        warnings=warnings,
        content=content,
    )


def _skill_summary(agent: AgentKind, skill: Any) -> SkillSummary:
    return SkillSummary(
        agent=agent,
        name=skill.name,
        purpose=skill.purpose,
        allowed_tools=list(skill.allowed_tools),
        content=skill.instructions,
        custom=skill.custom,
    )


def _module_name(tool_root: Path, path: Path) -> str:
    relative = path.relative_to(tool_root).with_suffix("")
    return ".".join(relative.parts)


def _function_purpose(node: ast.FunctionDef | ast.AsyncFunctionDef, fallback: str) -> str:
    docstring = ast.get_docstring(node, clean=True)
    return docstring.splitlines()[0].strip() if docstring else fallback


def _scan_tool_file(
    *,
    agent: AgentKind,
    tool_root: Path,
    path: Path,
    status: Literal["builtin", "approved", "pending_review"],
) -> list[ToolSummary]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError):
        return []

    module = _module_name(tool_root, path)
    module_docstring = ast.get_docstring(tree, clean=True) or ""
    module_purpose = module_docstring.splitlines()[0].strip() if module_docstring else ""
    tools: list[ToolSummary] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            tools.append(
                ToolSummary(
                    agent=agent,
                    name=node.name,
                    module=module,
                    purpose=_function_purpose(
                        node,
                        module_purpose or f"Public function from {module}.",
                    ),
                    status=status,
                )
            )
    return tools


def _validate_quarantined_tool_source(name: str, source: str) -> str:
    normalized = source.replace("\r\n", "\n").strip() + "\n"
    if "\x00" in normalized:
        raise HTTPException(status_code=400, detail="Tool source contains a null byte.")

    try:
        tree = ast.parse(normalized, filename=f"{name}.py")
    except SyntaxError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Tool source is not valid Python: line {exc.lineno}: {exc.msg}",
        ) from exc

    public_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    if name not in public_functions:
        raise HTTPException(
            status_code=400,
            detail=f"The quarantined module must define a public function named '{name}'.",
        )

    # This file is deliberately never imported by the runtime. Reject obvious
    # top-level execution anyway so a later human review starts from a safer file.
    allowed_top_level = (
        ast.Expr,       # module docstring only; checked below
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Assign,
        ast.AnnAssign,
        ast.If,         # TYPE_CHECKING guards
    )
    for index, node in enumerate(tree.body):
        if not isinstance(node, allowed_top_level):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported top-level statement in quarantined tool: {type(node).__name__}.",
            )
        if isinstance(node, ast.Expr) and not (
            index == 0 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            raise HTTPException(
                status_code=400,
                detail="Only a module docstring may appear as a top-level expression.",
            )

    return normalized


@router.get("/agent-configuration")
def get_agent_configuration() -> dict[str, Any]:
    return {
        **runtime_agent_configuration.public_snapshot(),
        **_coding_runtime_snapshot(),
    }


@router.get("/models")
def list_available_models(
    provider: ChatProvider = Query(...),
    capability: ModelCapability = Query(...),
) -> dict[str, Any]:
    """Return live compatible models when credentials permit, otherwise fallbacks."""
    return discover_models(provider, capability)


@router.put("/agent-configuration")
def update_agent_configuration(request: AgentConfigurationUpdate) -> dict[str, Any]:
    values = request.model_dump(exclude={"secrets"})
    model_values = {field: values[field] for field in MODEL_CONFIGURATION_FIELDS}
    coding_values = {
        field: int(values[field])
        if values.get(field) is not None
        else int(getattr(config_settings, field))
        for field in CODING_RUNTIME_FIELDS
    }
    try:
        snapshot = runtime_agent_configuration.update(
            model_values,
            secrets=request.secrets,
        )
        _save_coding_runtime_configuration(coding_values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # VoiceAgentService keeps a provider client. Clear it so the next turn uses the
    # new model/key. The coding-agent model factory already resolves every run.
    try:
        from ai_agents.api.routers.voice_agent import get_voice_service

        get_voice_service.cache_clear()
    except (ImportError, AttributeError):
        pass

    return {**snapshot, **_coding_runtime_snapshot()}


@router.get("/skills", response_model=list[SkillSummary])
def list_skills(
    agent: AgentKind = Query(...),
) -> list[SkillSummary]:
    _migrate_agent_assets(agent)
    registry = SkillRegistry(_safe_agent_dir(SKILL_DIRS, agent)).load()
    return [_skill_summary(agent, skill) for skill in registry.list()]


@router.post("/skills/draft", response_model=SkillDraftResponse)
def draft_skill(request: SkillDraftRequest) -> SkillDraftResponse:
    """Generate a new skill or translate imported Markdown into canonical format."""

    return _generate_skill_draft(request)


@router.post("/skills", response_model=SkillSummary)
def save_skill(request: SkillWriteRequest) -> SkillSummary:
    if not request.name.startswith(CUSTOM_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=f"User-created skills must start with '{CUSTOM_PREFIX}'.",
        )

    skill_dir = _safe_agent_dir(SKILL_DIRS, request.agent)
    path = (skill_dir / f"{request.name}.md").resolve()
    if path.parent != skill_dir:
        raise HTTPException(status_code=400, detail="Unsafe skill path.")
    if path.exists() and not request.overwrite:
        raise HTTPException(status_code=409, detail=f"Skill already exists: {request.name}")

    normalized_content = _validate_skill_markdown(request.content)
    _validate_skill_tool_references(request.agent, normalized_content)
    _atomic_write(path, normalized_content)
    skill = SkillRegistry(skill_dir).load().get(request.name)
    
    return _skill_summary(request.agent, skill)


@router.delete("/skills/{agent}/{name}")
def delete_skill(agent: AgentKind, name: str) -> dict[str, bool]:
    normalized = name.strip().lower().replace("-", "_")
    if not NAME_RE.fullmatch(normalized) or not normalized.startswith(CUSTOM_PREFIX):
        raise HTTPException(status_code=403, detail="Only custom_ skills may be deleted.")

    skill_dir = _safe_agent_dir(SKILL_DIRS, agent)
    path = (skill_dir / f"{normalized}.md").resolve()
    if path.parent != skill_dir:
        raise HTTPException(status_code=400, detail="Unsafe skill path.")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Skill does not exist.")
    path.unlink()
    return {"deleted": True}


@router.get("/tools", response_model=list[ToolSummary])
def list_tools(agent: AgentKind = Query(...)) -> list[ToolSummary]:
    _migrate_agent_assets(agent)

    tool_root = _safe_agent_dir(TOOL_DIRS, agent)
    pending_root = (tool_root / "custom_pending").resolve()
    approved_root = (tool_root / "custom_approved").resolve()
    pending_root.mkdir(parents=True, exist_ok=True)
    approved_root.mkdir(parents=True, exist_ok=True)

    tools: list[ToolSummary] = []
    for path in sorted(tool_root.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue

        if pending_root in path.parents:
            status: Literal["builtin", "approved", "pending_review"] = "pending_review"
        elif approved_root in path.parents:
            status = "approved"
        else:
            status = "builtin"

        matches = _scan_tool_file(
            agent=agent,
            tool_root=tool_root,
            path=path,
            status=status,
        )

        if status != "builtin":
            # One custom file is one tool. Helpers must be private (_helper).
            matches = [item for item in matches if item.name == path.stem]
        tools.extend(matches)

    return sorted(tools, key=lambda item: (item.status, item.name, item.module))


def _custom_tool_path(agent: AgentKind, name: str, *, status: Literal["pending_review", "approved"]) -> Path:
    normalized = name.strip().lower().replace("-", "_")
    if not NAME_RE.fullmatch(normalized):
        raise HTTPException(status_code=400, detail="Invalid tool name.")
    
    tool_root = _safe_agent_dir(TOOL_DIRS, agent)
    directory_name = "custom_pending" if status == "pending_review" else "custom_approved"
    directory = (tool_root / directory_name).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / f"{normalized}.py").resolve()
    if path.parent != directory:
        raise HTTPException(status_code=400, detail="Unsafe tool path.")
    return path


def _approval_validation_errors(agent: AgentKind, name: str, source: str) -> list[str]:
    try:
        _validate_quarantined_tool_source(name, source)
        validate_approved_custom_tool_source(name, source)
        if agent == "voice":
            validate_voice_custom_tool_source(name, source)
    except (HTTPException, CustomToolValidationError) as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        return [str(detail)]
    return []


def _candidate_registry(agent: AgentKind, directory: Path):
    if agent == "coding":
        return ApprovedCustomToolRegistry(directory)
    return ApprovedCustomVoiceToolRegistry(directory)


@router.get("/tools/{agent}/{name}", response_model=ToolReviewResponse)
def review_tool(agent: AgentKind, name: str) -> ToolReviewResponse:
    path = _custom_tool_path(agent, name, status="pending_review")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Pending tool does not exist.")

    source = path.read_text(encoding="utf-8")
    matches = _scan_tool_file(
        agent=agent,
        tool_root=_safe_agent_dir(TOOL_DIRS, agent),
        path=path,
        status="pending_review",
    )

    match = next((item for item in matches if item.name == path.stem), None)
    if match is None:
        raise HTTPException(status_code=400, detail="Could not discover the pending tool function.")

    validation_errors = _approval_validation_errors(agent, path.stem, source)

    return ToolReviewResponse(
        **match.model_dump(),
        source=source,
        approval_ready=not validation_errors,
        validation_errors=validation_errors,
    )


@router.post("/tools/content", response_model=ToolReviewResponse)
def update_tool_file(request: ToolFileUpdateRequest) -> ToolReviewResponse:
    name = Path(request.path).stem
    path = _custom_tool_path(request.agent, name, status="pending_review")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Pending tool does not exist.")

    source = _validate_quarantined_tool_source(name, request.content)
    _atomic_write(path, source)
    return review_tool(request.agent, name)


@router.post("/tools/{agent}/{name}/approve", response_model=ToolSummary)
def approve_tool(agent: AgentKind, name: str) -> ToolSummary:
    pending_path = _custom_tool_path(agent, name, status="pending_review")
    approved_path = _custom_tool_path(agent, name, status="approved")

    if not pending_path.exists():
        raise HTTPException(status_code=404, detail="Pending tool does not exist.")
    if approved_path.exists():
        raise HTTPException(status_code=409, detail=f"Approved tool already exists: {name}")

    source = pending_path.read_text(encoding="utf-8")
    source = _validate_quarantined_tool_source(pending_path.stem, source)

    try:
        source = validate_approved_custom_tool_source(pending_path.stem, source)
        if agent == "voice":
            source = validate_voice_custom_tool_source(pending_path.stem, source)
    except CustomToolValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Import/signature-check the candidate outside custom_approved first. This keeps
    # approval atomic from the runtime registry's perspective: a concurrent agent
    # run can see either the pending file or the fully validated approved file.
    with tempfile.TemporaryDirectory(prefix="ai-agents-tool-approval-") as temporary_dir:
        candidate_path = Path(temporary_dir) / pending_path.name
        _atomic_write(candidate_path, source)
        candidate_registry = _candidate_registry(agent, Path(temporary_dir)).load()

        if not candidate_registry.has(pending_path.stem):
            raise HTTPException(
                status_code=400,
                detail="Tool passed static review but could not be loaded by the runtime registry.",
            )

    _atomic_write(pending_path, source)
    os.replace(pending_path, approved_path)

    match = next(
        (
            item
            for item in _scan_tool_file(
                agent=agent,
                tool_root=_safe_agent_dir(TOOL_DIRS, agent),
                path=approved_path,
                status="approved",
            )
            if item.name == approved_path.stem
        ),
        None,
    )

    if match is None:
        raise HTTPException(status_code=500, detail="Approved tool could not be indexed.")
    return match


@router.delete("/tools/{agent}/{name}/reject")
def reject_tool(agent: AgentKind, name: str) -> dict[str, bool]:
    pending_path = _custom_tool_path(agent, name, status="pending_review")
    if not pending_path.exists():
        raise HTTPException(status_code=404, detail="Pending tool does not exist.")
    pending_path.unlink()
    return {"rejected": True}


@router.post("/tools/quarantine", response_model=ToolSummary)
def quarantine_tool(request: ToolQuarantineRequest) -> ToolSummary:
    tool_root = _safe_agent_dir(TOOL_DIRS, request.agent)
    path = _custom_tool_path(request.agent, request.name, status="pending_review")
    approved_path = _custom_tool_path(request.agent, request.name, status="approved")
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Pending tool already exists: {request.name}")

    if approved_path.exists() or any(
        item.name == request.name and item.status == "builtin"
        for item in list_tools(request.agent)
    ):
        raise HTTPException(
            status_code=409,
            detail=f"An executable tool named '{request.name}' already exists.",
        )

    source = _validate_quarantined_tool_source(request.name, request.source)
    if not source.lstrip().startswith(('"""', "'''")):
        source = f'"""{request.purpose}"""\n\n' + source
    _atomic_write(path, source)

    matches = _scan_tool_file(
        agent=request.agent,
        tool_root=tool_root,
        path=path,
        status="pending_review",
    )
    match = next((item for item in matches if item.name == request.name), None)
    if match is None:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not discover the submitted tool function.")

    return match.model_copy(update={"purpose": request.purpose})
