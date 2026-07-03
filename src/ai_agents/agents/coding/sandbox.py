from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


SANDBOX_IGNORE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


@dataclass(frozen=True)
class CodingSandbox:
    sandbox_root: Path
    original_workspace_root: Path
    original_repo_root: Path
    workspace_root: Path
    repo_root: Path


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in SANDBOX_IGNORE_DIRS or name.endswith(".egg-info")}


def create_coding_sandbox(
    *,
    repo_root: Path,
    workspace_root: Path | None,
    run_id: str,
) -> CodingSandbox:
    
    original_repo_root = repo_root.expanduser().resolve()
    original_workspace_root = (
        workspace_root.expanduser().resolve()
        if workspace_root
        else original_repo_root
    )

    try:
        repo_relative = original_repo_root.relative_to(original_workspace_root)
    except ValueError:
        original_workspace_root = original_repo_root
        repo_relative = Path(".")

    sandbox_root = Path(tempfile.mkdtemp(prefix=f"coding-agent-{run_id}-"))
    sandbox_workspace_root = sandbox_root / "workspace"

    shutil.copytree(
        original_workspace_root,
        sandbox_workspace_root,
        ignore=_ignore,
    )

    sandbox_repo_root = sandbox_workspace_root / repo_relative

    return CodingSandbox(
        sandbox_root=sandbox_root,
        original_workspace_root=original_workspace_root,
        original_repo_root=original_repo_root,
        workspace_root=sandbox_workspace_root,
        repo_root=sandbox_repo_root,
    )


def cleanup_coding_sandbox(sandbox: CodingSandbox, *, keep: bool = False) -> None:
    if keep:
        return

    shutil.rmtree(sandbox.sandbox_root, ignore_errors=True)


def apply_sandbox_files_to_repo(
    *,
    sandbox: CodingSandbox,
    changed_paths: list[str],
) -> list[str]:
    applied: list[str] = []

    for relative in changed_paths:
        relative_path = Path(relative)

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe changed path: {relative}")

        source = (sandbox.repo_root / relative_path).resolve()
        target = (sandbox.original_repo_root / relative_path).resolve()

        if sandbox.repo_root not in source.parents and source != sandbox.repo_root:
            raise ValueError(f"Source path escaped sandbox repo: {relative}")

        if sandbox.original_repo_root not in target.parents and target != sandbox.original_repo_root:
            raise ValueError(f"Target path escaped original repo: {relative}")

        if not source.exists():
            raise FileNotFoundError(f"Changed file does not exist in sandbox: {relative}")

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        applied.append(relative)

    return applied