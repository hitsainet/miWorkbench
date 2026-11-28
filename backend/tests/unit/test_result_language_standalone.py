"""045 FR-10: miWorkbench works fully without miAssist (BR-MX-7, BR-G-4, M5).

The result language is a VENDORED contract read from the package copy at build time; nothing under `mcp_server/`
imports, calls or names a miAssist address, and a server builds and answers with no miAssist checkout anywhere.

Controls (0xcc/reviews/045_implementation_controls_2026-11-16.md, FTASKS 11.1):
  a constant `MIASSIST_URL = "http://miassist…"` in a tool module     → test_no_miassist_reach red
  `import src.schemas` in tool_classes.py                             → test_mcp_tools_stay_on_http.py
                                                                         ::test_the_045_modules_import_only_their_leaves red
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from tests.support import result_language_drive as drive

MCP_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "mcp_server"


def _docstrings(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def miassist_reach(path: pathlib.Path) -> list[str]:
    """Imports of a miAssist package and string constants naming miAssist (a docstring's prose excepted)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = _docstrings(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if "miassist" in a.name.lower()]
        elif isinstance(node, ast.ImportFrom) and "miassist" in (node.module or "").lower():
            found.append(node.module)
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
              and "miassist" in node.value.lower()):
            found.append(node.value[:80])
    return found


def test_no_miassist_reach():
    modules = sorted(MCP_DIR.rglob("*.py"))
    assert len(modules) > 40 and MCP_DIR / "result_language.py" in modules
    reach = {str(p.relative_to(MCP_DIR)): r for p in modules if (r := miassist_reach(p))}
    assert not reach, reach


def test_the_scan_sees_a_url(tmp_path):
    sample = tmp_path / "x.py"
    sample.write_text('"""miAssist is named in prose only."""\nURL = "http://miassist.hitsai.local/api"\n')
    assert miassist_reach(sample) == ["http://miassist.hitsai.local/api"]


def test_builds_with_no_sibling(monkeypatch, tmp_path):
    monkeypatch.setenv("MIASSIST_REPO", str(tmp_path))
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/features/f_1$", body={"id": "f_1"})
        mcp = drive.build("read")
        assert len(mcp._tool_manager.list_tools()) > 10  # noqa: SLF001
        result = drive.call(mcp, "get_feature", {"feature_id": "f_1"})
    # 046: the fake is completed with the route's always-sent fields (get_feature is typed from its route).
    assert not result.is_error and result.structured_content["id"] == "f_1"


def test_a_missing_package_copy_fails_the_build(monkeypatch, tmp_path):
    from src.mcp_server import result_language

    monkeypatch.setattr(result_language, "CONTRACT_PATH", tmp_path / "absent.json")
    result_language.load_contract.cache_clear()
    try:
        with pytest.raises(FileNotFoundError):
            drive.build("read")
    finally:
        result_language.load_contract.cache_clear()
