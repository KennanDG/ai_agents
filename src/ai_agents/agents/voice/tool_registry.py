from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from ai_agents.agents.coding.tool_registry import (
    CustomToolValidationError,
    validate_approved_custom_tool_source,
)


VOICE_TOOLS_DIR = Path(__file__).resolve().parent / "tools"
VOICE_CUSTOM_PENDING_DIR = VOICE_TOOLS_DIR / "custom_pending"
VOICE_CUSTOM_APPROVED_DIR = VOICE_TOOLS_DIR / "custom_approved"

# Voice custom tools run before the intake model and receive only backend-owned,
# read-only context. Required parameters outside this set cannot be satisfied safely
# and are rejected during approval.
VOICE_RUNTIME_ARGUMENTS = {
    "repo_root",
    "workspace_root",
    "active_path",
    "transcript",
    "prompt_text",
    "attached_files",
    "history",
    "repo_context",
    "context_summary",
}

MAX_VOICE_CUSTOM_TOOL_CALLS = 4
MAX_VOICE_CUSTOM_TOOL_RESULT_CHARS = 6_000
MAX_VOICE_CUSTOM_TOOL_TOTAL_CHARS = 18_000


@dataclass(frozen=True)
class ApprovedVoiceTool:
    name: str
    path: Path
    purpose: str
    signature: str
    callable: Callable[..., Any]


def _load_module(path: Path) -> ModuleType:
    module_name = f"ai_agents_voice_custom_tool_{path.stem}_{path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CustomToolValidationError(f"Could not create an import spec for {path.name}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool_purpose(source: str, name: str) -> str:
    tree = ast.parse(source, filename=f"{name}.py")
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    if target is not None:
        docstring = ast.get_docstring(target, clean=True)
        if docstring:
            return docstring.splitlines()[0].strip()

    module_docstring = ast.get_docstring(tree, clean=True)
    if module_docstring:
        return module_docstring.splitlines()[0].strip()
    return f"Approved custom voice tool {name}."


def validate_voice_custom_tool_source(name: str, source: str) -> str:
    """Validate the additional runtime contract for a voice custom tool.

    The shared coding-tool validator supplies the security restrictions. This layer
    adds a voice-specific interface rule: every required argument must be one that
    the backend can inject deterministically before the intake model runs.
    """

    normalized = validate_approved_custom_tool_source(name, source)
    tree = ast.parse(normalized, filename=f"{name}.py")
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    if target is None:
        raise CustomToolValidationError(
            f"Approved voice tool must define a function named '{name}'."
        )

    if target.args.posonlyargs or target.args.vararg or target.args.kwarg:
        raise CustomToolValidationError(
            "Voice custom tools may use only named parameters; positional-only "
            "parameters, *args, and **kwargs are not supported."
        )

    positional = [*target.args.args]
    positional_defaults = [None] * (len(positional) - len(target.args.defaults)) + list(
        target.args.defaults
    )
    required: list[str] = [
        argument.arg
        for argument, default in zip(positional, positional_defaults, strict=True)
        if default is None
    ]
    required.extend(
        argument.arg
        for argument, default in zip(
            target.args.kwonlyargs,
            target.args.kw_defaults,
            strict=True,
        )
        if default is None
    )

    unsupported = [item for item in required if item not in VOICE_RUNTIME_ARGUMENTS]
    if unsupported:
        raise CustomToolValidationError(
            "Voice custom tools may only require backend-injected arguments. "
            "Unsupported required argument(s): " + ", ".join(unsupported) + ". "
            "Supported names: " + ", ".join(sorted(VOICE_RUNTIME_ARGUMENTS)) + "."
        )

    return normalized


def _render_result(value: Any) -> tuple[str, bool]:
    try:
        rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        rendered = repr(value)

    if len(rendered) <= MAX_VOICE_CUSTOM_TOOL_RESULT_CHARS:
        return rendered, False
    return (
        rendered[:MAX_VOICE_CUSTOM_TOOL_RESULT_CHARS]
        + "\n...[voice custom tool result truncated]",
        True,
    )


class ApprovedCustomVoiceToolRegistry:
    """Discover and invoke only approved custom tools for voice intake.

    Tools are reloaded from disk on each ``load`` so approval takes effect on the
    next voice turn without rebuilding the cached VoiceAgentService or graph.
    """

    def __init__(self, approved_dir: Path = VOICE_CUSTOM_APPROVED_DIR) -> None:
        self.approved_dir = approved_dir.resolve()
        self._tools: dict[str, ApprovedVoiceTool] = {}

    def load(self) -> "ApprovedCustomVoiceToolRegistry":
        self._tools.clear()
        self.approved_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(self.approved_dir.glob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
                validate_voice_custom_tool_source(path.stem, source)
                module = _load_module(path)
                function = getattr(module, path.stem, None)
                if not callable(function):
                    continue
                signature = inspect.signature(function)
                self._tools[path.stem] = ApprovedVoiceTool(
                    name=path.stem,
                    path=path,
                    purpose=_tool_purpose(source, path.stem),
                    signature=str(signature),
                    callable=function,
                )
            except (OSError, ImportError, AttributeError, CustomToolValidationError):
                # One invalid approved file must not break voice intake startup.
                continue

        return self

    def list(self) -> list[ApprovedVoiceTool]:
        return [self._tools[name] for name in sorted(self._tools)]

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ApprovedVoiceTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Approved voice custom tool is unavailable: {name}") from exc

    def invoke_from_state(self, name: str, state: dict[str, Any]) -> dict[str, Any]:
        tool = self.get(name)
        signature = inspect.signature(tool.callable)

        repo_root = state.get("repo_root")
        workspace_root = state.get("workspace_root")
        runtime_values: dict[str, Any] = {
            "repo_root": (
                str(Path(str(repo_root)).expanduser().resolve()) if repo_root else None
            ),
            "workspace_root": (
                str(Path(str(workspace_root)).expanduser().resolve())
                if workspace_root
                else None
            ),
            "active_path": state.get("active_path"),
            "transcript": state.get("transcript", ""),
            "prompt_text": state.get("prompt_text", ""),
            "attached_files": copy.deepcopy(list(state.get("attached_files") or [])),
            "history": copy.deepcopy(list(state.get("history") or [])),
            "repo_context": copy.deepcopy(dict(state.get("repo_context") or {})),
            "context_summary": state.get("context_summary", ""),
        }

        arguments: dict[str, Any] = {}
        for parameter_name in signature.parameters:
            if parameter_name not in VOICE_RUNTIME_ARGUMENTS:
                # Optional custom parameters keep their function defaults.
                continue
            value = runtime_values.get(parameter_name)
            if value is not None:
                arguments[parameter_name] = value

        try:
            signature.bind(**arguments)
        except TypeError as exc:
            raise ValueError(
                f"Voice runtime could not satisfy {name}{tool.signature}: {exc}"
            ) from exc

        value = tool.callable(**arguments)
        output, truncated = _render_result(value)
        return {
            "tool_name": name,
            "output": output,
            "truncated": truncated,
            "success": True,
        }
