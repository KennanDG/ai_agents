from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from ai_agents.agents.coding.registry import SkillRegistry
from ai_agents.config.runtime_configuration import runtime_agent_configuration


router = APIRouter(prefix="/admin", tags=["admin"])

AgentKind = Literal["coding", "voice"]
ChatProvider = Literal["groq", "deepseek", "openrouter", "openai"]

_AI_AGENTS_ROOT = Path(__file__).resolve().parents[2]
_SKILL_DIRS: dict[AgentKind, Path] = {
    "coding": _AI_AGENTS_ROOT / "agents" / "coding" / "skills",
    "voice": _AI_AGENTS_ROOT / "agents" / "voice" / "skills",
}
_TOOL_DIRS: dict[AgentKind, Path] = {
    "coding": _AI_AGENTS_ROOT / "agents" / "coding" / "tools",
    "voice": _AI_AGENTS_ROOT / "agents" / "voice" / "tools",
}

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_CUSTOM_PREFIX = "custom_"
_MAX_SKILL_CHARS = 50_000
_MAX_TOOL_CHARS = 100_000


class AgentConfigurationUpdate(BaseModel):
    coding_provider: ChatProvider
    coding_model: str = Field(min_length=1, max_length=255)
    reasoning_provider: ChatProvider
    reasoning_model: str = Field(min_length=1, max_length=255)
    caption_model: str = Field(min_length=1, max_length=255)
    voice_chat_model: str = Field(min_length=1, max_length=255)
    voice_stt_model: str = Field(min_length=1, max_length=255)
    voice_tts_model: str = Field(min_length=1, max_length=255)
    voice_tts_voice: str = Field(min_length=1, max_length=100)
    voice_tts_enabled: bool = True
    secrets: dict[ChatProvider, str] = Field(default_factory=dict)

    @field_validator(
        "coding_model",
        "reasoning_model",
        "caption_model",
        "voice_chat_model",
        "voice_stt_model",
        "voice_tts_model",
        "voice_tts_voice",
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class SkillWriteRequest(BaseModel):
    agent: AgentKind
    name: str
    content: str = Field(min_length=1, max_length=_MAX_SKILL_CHARS)
    overwrite: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not _NAME_RE.fullmatch(normalized):
            raise ValueError(
                "Skill names must start with a letter and contain only lowercase "
                "letters, numbers, underscores, or hyphens."
            )
        return normalized


class ToolQuarantineRequest(BaseModel):
    agent: AgentKind
    name: str
    purpose: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=_MAX_TOOL_CHARS)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not _NAME_RE.fullmatch(normalized):
            raise ValueError("Tool names must use lowercase snake_case.")
        return normalized

    @field_validator("purpose")
    @classmethod
    def normalize_purpose(cls, value: str) -> str:
        return value.strip()


class SkillSummary(BaseModel):
    agent: AgentKind
    name: str
    purpose: str
    allowed_tools: list[str] = Field(default_factory=list)
    content: str
    custom: bool


class ToolSummary(BaseModel):
    agent: AgentKind
    name: str
    module: str
    purpose: str
    status: Literal["builtin", "pending_review"]


def _safe_agent_dir(mapping: dict[AgentKind, Path], agent: AgentKind) -> Path:
    directory = mapping[agent].resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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
    normalized = content.replace("\r\n", "\n").strip() + "\n"
    if "\x00" in normalized:
        raise HTTPException(status_code=400, detail="Skill content contains a null byte.")
    if not any(line.lower().startswith("purpose:") for line in normalized.splitlines()):
        raise HTTPException(
            status_code=400,
            detail="Skill Markdown must include a top-level 'Purpose:' line.",
        )
    return normalized


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
    status: Literal["builtin", "pending_review"],
) -> list[ToolSummary]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError):
        return []

    module = _module_name(tool_root, path)
    tools: list[ToolSummary] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            tools.append(
                ToolSummary(
                    agent=agent,
                    name=node.name,
                    module=module,
                    purpose=_function_purpose(node, f"Public function from {module}."),
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
    return runtime_agent_configuration.public_snapshot()


@router.put("/agent-configuration")
def update_agent_configuration(request: AgentConfigurationUpdate) -> dict[str, Any]:
    values = request.model_dump(exclude={"secrets"})
    try:
        snapshot = runtime_agent_configuration.update(values, secrets=request.secrets)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # VoiceAgentService keeps a provider client. Clear it so the next turn uses the
    # new model/key. The coding-agent model factory already resolves every run.
    try:
        from ai_agents.api.routers.voice_agent import get_voice_service

        get_voice_service.cache_clear()
    except (ImportError, AttributeError):
        pass

    return snapshot


@router.get("/skills", response_model=list[SkillSummary])
def list_skills(
    agent: AgentKind = Query(...),
) -> list[SkillSummary]:
    registry = SkillRegistry(_safe_agent_dir(_SKILL_DIRS, agent)).load()
    return [_skill_summary(agent, skill) for skill in registry.list()]


@router.post("/skills", response_model=SkillSummary)
def save_skill(request: SkillWriteRequest) -> SkillSummary:
    if not request.name.startswith(_CUSTOM_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=f"User-created skills must start with '{_CUSTOM_PREFIX}'.",
        )

    skill_dir = _safe_agent_dir(_SKILL_DIRS, request.agent)
    path = (skill_dir / f"{request.name}.md").resolve()
    if path.parent != skill_dir:
        raise HTTPException(status_code=400, detail="Unsafe skill path.")
    if path.exists() and not request.overwrite:
        raise HTTPException(status_code=409, detail=f"Skill already exists: {request.name}")

    _atomic_write(path, _validate_skill_markdown(request.content))
    skill = SkillRegistry(skill_dir).load().get(request.name)
    return _skill_summary(request.agent, skill)


@router.delete("/skills/{agent}/{name}")
def delete_skill(agent: AgentKind, name: str) -> dict[str, bool]:
    normalized = name.strip().lower().replace("-", "_")
    if not _NAME_RE.fullmatch(normalized) or not normalized.startswith(_CUSTOM_PREFIX):
        raise HTTPException(status_code=403, detail="Only custom_ skills may be deleted.")

    skill_dir = _safe_agent_dir(_SKILL_DIRS, agent)
    path = (skill_dir / f"{normalized}.md").resolve()
    if path.parent != skill_dir:
        raise HTTPException(status_code=400, detail="Unsafe skill path.")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Skill does not exist.")
    path.unlink()
    return {"deleted": True}


@router.get("/tools", response_model=list[ToolSummary])
def list_tools(agent: AgentKind = Query(...)) -> list[ToolSummary]:
    tool_root = _safe_agent_dir(_TOOL_DIRS, agent)
    pending_root = (tool_root / "custom_pending").resolve()
    pending_root.mkdir(parents=True, exist_ok=True)

    tools: list[ToolSummary] = []
    for path in sorted(tool_root.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        status: Literal["builtin", "pending_review"] = (
            "pending_review" if pending_root in path.parents else "builtin"
        )
        tools.extend(
            _scan_tool_file(
                agent=agent,
                tool_root=tool_root,
                path=path,
                status=status,
            )
        )

    return sorted(tools, key=lambda item: (item.status, item.name, item.module))


@router.post("/tools/quarantine", response_model=ToolSummary)
def quarantine_tool(request: ToolQuarantineRequest) -> ToolSummary:
    tool_root = _safe_agent_dir(_TOOL_DIRS, request.agent)
    pending_root = (tool_root / "custom_pending").resolve()
    pending_root.mkdir(parents=True, exist_ok=True)
    path = (pending_root / f"{request.name}.py").resolve()
    if path.parent != pending_root:
        raise HTTPException(status_code=400, detail="Unsafe tool path.")
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Pending tool already exists: {request.name}")

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
