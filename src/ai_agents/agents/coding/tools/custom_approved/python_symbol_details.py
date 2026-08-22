"""Inspect Python definitions using the standard AST module."""

from __future__ import annotations

import ast
from pathlib import Path


def python_symbol_details(
    repo_root: str,
    symbol: str,
    relative_path: str = ".",
    max_results: int = 25,
) -> dict:
    """Find Python definitions and return structured symbol metadata."""

    root = Path(repo_root).expanduser().resolve()
    target = (root / relative_path).resolve()

    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Requested path is outside the repository root.") from exc

    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {relative_path}")

    paths = [target] if target.is_file() else sorted(target.rglob("*.py"))

    matches: list[dict] = []

    for path in paths:
        if len(matches) >= max_results:
            break

        if any(
            part in {
                ".git",
                ".venv",
                "venv",
                "__pycache__",
                "node_modules",
                "dist",
                "build",
            }
            for part in path.parts
        ):
            continue

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                continue

            if node.name != symbol:
                continue

            if isinstance(node, ast.ClassDef):
                kind = "class"
                signature = node.name
                bases = [
                    ast.unparse(base)
                    for base in node.bases
                ]
            else:
                kind = (
                    "async_function"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else "function"
                )
                signature = (
                    f"{node.name}("
                    + ", ".join(arg.arg for arg in node.args.args)
                    + ")"
                )
                bases = []

            decorators = [
                ast.unparse(item)
                for item in node.decorator_list
            ]

            end_lineno = node.end_lineno
            

            matches.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "name": node.name,
                    "kind": kind,
                    "line_start": node.lineno,
                    "line_end": end_lineno,
                    "signature": signature,
                    "bases": bases,
                    "decorators": decorators,
                    "docstring": ast.get_docstring(node) or "",
                    "preview": "\n".join(
                        lines[
                            max(0, node.lineno - 1):
                            min(len(lines), node.lineno + 7)
                        ]
                    ),
                }
            )

            if len(matches) >= max_results:
                break

    return {
        "symbol": symbol,
        "match_count": len(matches),
        "matches": matches,
    }
