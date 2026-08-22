from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable


CODING_TOOLS_DIR = Path(__file__).resolve().parent / "tools"
CUSTOM_PENDING_DIR = CODING_TOOLS_DIR / "custom_pending"
CUSTOM_APPROVED_DIR = CODING_TOOLS_DIR / "custom_approved"

# Custom tools are deliberately much more restricted than built-in application
# code. Expand this allowlist only when a concrete reviewed tool requires it.
SAFE_IMPORT_ROOTS = {
    "__future__",
    "ast",
    "collections",
    "dataclasses",
    "datetime",
    "enum",
    "fnmatch",
    "functools",
    "hashlib",
    "itertools",
    "json",
    "math",
    "operator",
    "pathlib",
    "re",
    "statistics",
    "string",
    "textwrap",
    "typing",
    "tomllib",
}

FORBIDDEN_DIRECT_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "locals",
    "open",
    "setattr",
    "vars",
}

FORBIDDEN_ATTRIBUTE_CALLS = {
    # File mutation
    "chmod",
    "chown",
    "mkdir",
    "rename",
    "rmdir",
    "touch",
    "unlink",
    "write_bytes",
    "write_text",
    # Process / shell / dynamic loading
    "Popen",
    "exec_module",
    "open",
    "popen",
    "system",
}

MAX_CUSTOM_TOOL_RESULT_CHARS = 24_000
MAX_CUSTOM_TOOL_CALLS = 4


@dataclass(frozen=True)
class ApprovedTool:
    name: str
    path: Path
    purpose: str
    signature: str
    callable: Callable[..., Any]


class CustomToolValidationError(ValueError):
    pass


def _function_purpose(node: ast.FunctionDef | ast.AsyncFunctionDef, fallback: str) -> str:
    docstring = ast.get_docstring(node, clean=True)
    return docstring.splitlines()[0].strip() if docstring else fallback


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    )


def validate_approved_custom_tool_source(name: str, source: str) -> str:
    """Validate the stronger contract required before runtime activation.

    This is defense in depth, not an OS sandbox. Approved custom tools still run
    as Python inside the local backend process, so the user must review the source.
    """

    normalized = source.replace("\r\n", "\n").strip() + "\n"

    try:
        tree = ast.parse(normalized, filename=f"{name}.py")

    except SyntaxError as exc:
        raise CustomToolValidationError(
            f"Tool source is not valid Python: line {exc.lineno}: {exc.msg}"
        ) from exc

    target: ast.FunctionDef | None = None
    public_functions: list[str] = []

    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Expr):
            if not (
                index == 0
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                raise CustomToolValidationError(
                    "Only a module docstring may appear as a top-level expression."
                )
            
            continue

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules: list[str] = []

            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]

            elif node.module:
                modules = [node.module]

            for module in modules:
                root = module.split(".", 1)[0]

                if root not in SAFE_IMPORT_ROOTS:
                    raise CustomToolValidationError(
                        f"Import '{module}' is not allowed in approved custom tools."
                    )
                
            continue

        if isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                raise CustomToolValidationError(
                    "Custom tool functions may not use decorators because decorators execute at import time."
                )
            
            default_nodes = [*node.args.defaults, *[item for item in node.args.kw_defaults if item is not None]]

            if any(isinstance(item, ast.Call) for default in default_nodes for item in ast.walk(default)):
                raise CustomToolValidationError(
                    "Custom tool default argument values may not execute function calls."
                )
            
            if not node.name.startswith("_"):
                public_functions.append(node.name)

            if node.name == name:
                target = node

            continue

        if isinstance(node, ast.AsyncFunctionDef):
            raise CustomToolValidationError(
                "Async custom tools are not supported yet; use a synchronous function."
            )

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value

            if value is not None and any(isinstance(item, ast.Call) for item in ast.walk(value)):
                raise CustomToolValidationError(
                    "Top-level assignments may not execute function calls."
                )
            
            continue

        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            continue

        raise CustomToolValidationError(
            f"Unsupported top-level statement for an approved tool: {type(node).__name__}."
        )

    if target is None:
        raise CustomToolValidationError(
            f"Approved tool must define a synchronous public function named '{name}'."
        )

    if public_functions != [name]:
        raise CustomToolValidationError(
            "Custom tool modules must expose exactly one public function and it must "
            f"match the filename. Found: {', '.join(public_functions) or 'none'}."
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_DIRECT_CALLS:
                raise CustomToolValidationError(
                    f"Reference to '{node.id}' is not allowed in approved custom tools."
                )

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif node.module:
                modules = [node.module]
            for module in modules:
                root = module.split(".", 1)[0]
                if root not in SAFE_IMPORT_ROOTS:
                    raise CustomToolValidationError(
                        f"Import '{module}' is not allowed in approved custom tools."
                    )

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_DIRECT_CALLS:
                raise CustomToolValidationError(
                    f"Call to '{node.func.id}' is not allowed in approved custom tools."
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRIBUTE_CALLS:
                raise CustomToolValidationError(
                    f"Call to '.{node.func.attr}(...)' is not allowed in approved custom tools."
                )

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CustomToolValidationError(
                "Dunder attribute access is not allowed in approved custom tools."
            )

    return normalized


def validate_tool_signature(function: Callable[..., Any]) -> inspect.Signature:

    signature = inspect.signature(function)

    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }:
            raise CustomToolValidationError(
                "Custom tools may use only named parameters; *args, **kwargs, and "
                "positional-only parameters are not supported."
            )
        
    return signature


def _load_module(path: Path, source: str) -> ModuleType:
    module_name = f"ai_agents_custom_tool_{path.stem}_{path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise CustomToolValidationError(f"Could not create an import spec for {path.name}.")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _json_result(value: Any) -> tuple[str, bool]:
    try:
        text = json.dumps(value, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(value)

    if len(text) <= MAX_CUSTOM_TOOL_RESULT_CHARS:
        return text, False
    
    return text[:MAX_CUSTOM_TOOL_RESULT_CHARS] + "\n...[tool result truncated]", True


class ApprovedCustomToolRegistry:
    """Load and invoke only user-approved custom coding tools."""

    def __init__(self, approved_dir: Path = CUSTOM_APPROVED_DIR) -> None:
        self.approved_dir = approved_dir.resolve()
        self._tools: dict[str, ApprovedTool] = {}

    def load(self) -> "ApprovedCustomToolRegistry":
        self._tools.clear()
        self.approved_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(self.approved_dir.glob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
                validate_approved_custom_tool_source(path.stem, source)
                module = _load_module(path, source)
                function = getattr(module, path.stem, None)

                if not callable(function):
                    continue

                signature = validate_tool_signature(function)
                tree = ast.parse(source, filename=str(path))
                node = next(
                    (
                        item
                        for item in tree.body
                        if isinstance(item, ast.FunctionDef) and item.name == path.stem
                    ),
                    None,
                )
                module_docstring = ast.get_docstring(tree, clean=True) or ""
                module_purpose = (
                    module_docstring.splitlines()[0].strip() if module_docstring else ""
                )
                purpose = (
                    _function_purpose(
                        node,
                        module_purpose or f"Approved custom tool {path.stem}.",
                    )
                    if node is not None
                    else module_purpose or f"Approved custom tool {path.stem}."
                )
                self._tools[path.stem] = ApprovedTool(
                    name=path.stem,
                    path=path,
                    purpose=purpose,
                    signature=str(signature),
                    callable=function,
                )
            except (OSError, CustomToolValidationError, ImportError, AttributeError):
                # Invalid approved files are treated as unavailable rather than making
                # the whole coding-agent startup fail.
                continue

        return self

    def list(self) -> list[ApprovedTool]:
        if not self._tools:
            self.load()
        return [self._tools[name] for name in sorted(self._tools)]

    def has(self, name: str) -> bool:
        if not self._tools:
            self.load()
        return name in self._tools

    def prompt_catalog(self, allowed_names: Iterable[str] | None = None) -> str:
        if not self._tools:
            self.load()

        allowed = set(allowed_names or self._tools)
        lines = []

        for tool in self.list():
            if tool.name not in allowed:
                continue
            lines.append(f"- {tool.name}{tool.signature}: {tool.purpose}")

        return "\n".join(lines)

    def invoke(
        self,
        name: str,
        *,
        repo_root: Path,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._tools:
            self.load()

        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Approved custom tool is not available: {name}")

        supplied = dict(arguments or {})
        # Runtime-owned path values may never be overridden by model output.
        supplied.pop("repo_root", None)
        signature = inspect.signature(tool.callable)

        if "repo_root" in signature.parameters:
            supplied["repo_root"] = str(repo_root.resolve())

        try:
            signature.bind(**supplied)
        except TypeError as exc:
            raise ValueError(f"Invalid arguments for {name}{tool.signature}: {exc}") from exc

        result = tool.callable(**supplied)
        rendered, truncated = _json_result(result)
        
        return {
            "tool_name": name,
            "arguments": {key: value for key, value in supplied.items() if key != "repo_root"},
            "output": rendered,
            "truncated": truncated,
            "success": True,
        }
