from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from .shell import run_command


def scaffold_component(component_name: str, props: list[str] | None = None) -> str:
    """Generate a boilerplate React component .tsx file content.

    Args:
        component_name: Name of the component (PascalCase).
        props: List of prop names with types (e.g., "title:string").

    Returns:
        A string containing the component code.
    """
    # Stub that returns a simple placeholder.
    if props is None:
        props = []
    prop_lines = "\n".join(
        f"  {p.split(':')[0]}: {p.split(':')[1] if ':' in p else 'any'};"
        for p in props
    )
    return f"""import React from 'react';

interface {component_name}Props {{
{prop_lines}
}}

export const {component_name}: React.FC<{component_name}Props> = ({{ {" ,".join(p.split(':')[0] for p in props)}}}) => {{
  return <div>{component_name}</div>;
}};
"""


def lint_component_imports(file_path: str, frontend_dir: Path = Path.cwd() / "agents" / "frontend") -> list[str]:
    """Check a component file for unused or missing imports using TypeScript compiler.

    Args:
        file_path: Path to the .tsx file relative to the frontend directory.
        frontend_dir: Directory containing the frontend project (tsconfig.json).

    Returns:
        A list of import-related lint messages.
    """
    errors = _run_tsc_check(file_path, frontend_dir)
    import_error_codes = {"TS2304", "TS2307", "TS2305", "TS6133"}
    messages = []
    for line in errors:
        if any(f"error {code}" in line for code in import_error_codes):
            messages.append(line.strip())
    return messages


def validate_component_props(file_path: str, frontend_dir: Path = Path.cwd() / "agents" / "frontend") -> list[str]:
    """Validate component props: run TypeScript type-check and flag 'any' usage.

    Args:
        file_path: Path to the .tsx file relative to the frontend directory.
        frontend_dir: Directory containing the frontend project (tsconfig.json).

    Returns:
        A list of validation messages.
    """
    errors = _run_tsc_check(file_path, frontend_dir)
    prop_error_codes = {"TS2322", "TS2345", "TS2554", "TS2741", "TS7006"}
    messages = []
    for line in errors:
        if any(f"error {code}" in line for code in prop_error_codes):
            messages.append(line.strip())
    # Also scan for explicit 'any' usage in the source file.
    target_file = (frontend_dir / file_path).resolve()
    if target_file.exists():
        content = target_file.read_text(encoding="utf-8")
        for i, source_line in enumerate(content.splitlines(), start=1):
            if re.search(r"(:\s*any\b|\bas\s+any\b)", source_line) and not source_line.strip().startswith("//"):
                messages.append(f"{target_file.name}:{i} - usage of 'any' detected: {source_line.strip()}")
    return messages


def _run_tsc_check(file_path: str, frontend_dir: Path) -> list[str]:
    """Run tsc --noEmit on a single file and return error lines."""
    target_file = (frontend_dir / file_path).resolve()
    if not target_file.exists():
        return [f"File not found: {target_file}"]
    try:
        relative_path = target_file.relative_to(frontend_dir)
    except ValueError:
        return [f"File is not under frontend directory: {target_file}"]
    command = f"npx tsc --noEmit -p tsconfig.app.json --pretty false {shlex.quote(str(relative_path))}"
    result = run_command(repo_root=frontend_dir, command=command, timeout_seconds=60)
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    combined_lines = (stdout + stderr).splitlines()
    error_lines = [line for line in combined_lines if "error TS" in line]
    return error_lines
