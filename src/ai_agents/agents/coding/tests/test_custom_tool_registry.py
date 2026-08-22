from __future__ import annotations

from pathlib import Path

import pytest

from ai_agents.agents.coding.tool_registry import (
    ApprovedToolRegistry,
    CustomToolValidationError,
    validate_approved_custom_tool_source,
)


SAFE_TOOL = '''
from __future__ import annotations

import ast
from pathlib import Path


def python_import_map(repo_root: str, relative_path: str = ".") -> dict:
    """Return imports used by Python files beneath a repository path."""
    root = Path(repo_root).resolve()
    target = (root / relative_path).resolve()
    rows = {}
    for file in target.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        rows[file.relative_to(root).as_posix()] = [
            node.names[0].name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        ]
    return rows
'''.strip()


def test_approved_tool_registry_loads_and_injects_repo_root(tmp_path: Path) -> None:
    approved_dir = tmp_path / "custom_approved"
    approved_dir.mkdir()
    (approved_dir / "python_import_map.py").write_text(SAFE_TOOL, encoding="utf-8")

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "example.py").write_text("import json\n", encoding="utf-8")

    registry = ApprovedToolRegistry(approved_dir).load()

    assert registry.has("python_import_map")
    result = registry.invoke(
        "python_import_map",
        repo_root=repo_root,
        arguments={"repo_root": "/tmp/model-must-not-control-this"},
    )

    assert result["success"] is True
    assert '"example.py"' in result["output"]
    assert '"json"' in result["output"]
    assert "repo_root" not in result["arguments"]


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "def unsafe(repo_root: str):\n    import os\n    return os.getcwd()\n",
            "Import 'os' is not allowed",
        ),
        (
            "def unsafe(repo_root: str):\n    return open('/etc/passwd').read()\n",
            "Call to 'open' is not allowed",
        ),
        (
            "from pathlib import Path\ndef unsafe(repo_root: str):\n"
            "    Path(repo_root, 'x').write_text('x')\n",
            "Call to '.write_text(...)' is not allowed",
        ),
        (
            "def _helper():\n    import subprocess\n    return subprocess.run(['echo', 'x'])\n"
            "def unsafe(repo_root: str):\n    return _helper()\n",
            "Import 'subprocess' is not allowed",
        ),
        (
            "from pathlib import Path\n"
            "def unsafe(repo_root: str, leak=Path('/etc/passwd').read_text()):\n"
            "    return leak\n",
            "default argument values may not execute function calls",
        ),
    ],
)
def test_approved_tool_validation_rejects_unsafe_capabilities(
    source: str,
    expected: str,
) -> None:
    with pytest.raises(CustomToolValidationError, match=".*") as exc_info:
        validate_approved_custom_tool_source("unsafe", source)

    assert expected in str(exc_info.value)
