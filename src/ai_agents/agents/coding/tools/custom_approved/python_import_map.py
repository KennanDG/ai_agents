"""Build a structured Python import map for a repository."""

from __future__ import annotations

import ast
from pathlib import Path


def python_import_map(
    repo_root: str,
    relative_path: str = ".",
    max_files: int = 500,
) -> dict:
    """Return imports used by Python files beneath a repository path."""

    root = Path(repo_root).expanduser().resolve()
    target = (root / relative_path).resolve()

    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Requested path is outside the repository root.") from exc

    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {relative_path}")

    paths = [target] if target.is_file() else sorted(target.rglob("*.py"))

    files: dict[str, dict] = {}
    errors: list[dict[str, str]] = []

    for path in paths[:max_files]:
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

        relative = path.relative_to(root).as_posix()

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            errors.append(
                {
                    "path": relative,
                    "error": str(exc),
                }
            )
            continue

        imports: list[dict] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "import",
                            "module": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                        }
                    )

            elif isinstance(node, ast.ImportFrom):
                imports.append(
                    {
                        "type": "from",
                        "module": node.module or "",
                        "level": node.level,
                        "names": [alias.name for alias in node.names],
                        "line": node.lineno,
                    }
                )

        files[relative] = {
            "imports": imports,
            "import_count": len(imports),
        }

    return {
        "root": str(root),
        "path": relative_path,
        "file_count": len(files),
        "files": files,
        "errors": errors,
    }
