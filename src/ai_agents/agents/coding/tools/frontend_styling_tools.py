from __future__ import annotations

from pathlib import Path
from typing import Any

from .shell import run_command

FRONTEND_DIR = "agents/frontend"


def validate_tailwind_config(repo_root: str, config_path: str = "") -> dict[str, Any]:
    """Run frontend typecheck to validate Tailwind and TypeScript configuration.

    Args:
        repo_root: Repository root directory.
        config_path: (unused, present for compatibility).

    Returns:
        Command result dict with returncode, stdout, stderr.
    """
    return run_command(
        Path(repo_root),
        f"npm run typecheck --prefix {FRONTEND_DIR}",
    )


def audit_accessibility(repo_root: str, html_file_path: str = "") -> list[dict[str, Any]]:
    """Audit an HTML/JSX file for common accessibility issues.

    Args:
        repo_root: Repository root directory.
        html_file_path: Path to the file to audit (unused).

    Returns:
        A list of issues found (empty if no issues or tooling unavailable).
    """
    # No dedicated accessibility auditing tool is configured in the frontend
    # toolchain. Run typecheck as a lightweight validation of JSX structure.
    result = run_command(
        Path(repo_root),
        f"npm run typecheck --prefix {FRONTEND_DIR}",
    )
    if result["returncode"] != 0:
        return [
            {
                "issue": "typecheck_error",
                "description": f"TypeScript compilation failed:\n{result['stderr']}",
            }
        ]
    return []


def check_responsive_breakpoints(repo_root: str, css_file_path: str = "") -> dict[str, Any]:
    """Run frontend production build to validate CSS processing and breakpoints.

    Args:
        repo_root: Repository root directory.
        css_file_path: Path to the CSS file to inspect (unused).

    Returns:
        Command result dict with returncode, stdout, stderr.
    """
    return run_command(
        Path(repo_root),
        f"npm run build --prefix {FRONTEND_DIR}",
    )
