"""ENH-010 (044 task 10.1, FR-10; suite CLAUDE.md "WSP: use miForge's WSP provider, never a library directly").

miWorkbench never imports a weight-space programming library: `weightspace` is a pinned research dependency of
miForge ONLY, and `wsplab` / `dspy-wsp` are comparators. miWorkbench reaches WSP through miForge's API and its
published schemas. Walks the AST of every module under `backend/src` (imports, `__import__` and
`importlib.import_module` with a constant name), so a docstring that names a library cannot trip it.

Control (0xcc/reviews/044_implementation_controls_2026-11-16.md): `import weightspace` in a service module → red.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
FORBIDDEN = {"weightspace", "wsplab", "dspy_wsp"}


def _imported_roots(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module.split(".")[0]
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in ("__import__", "import_module"):
                yield node.args[0].value.split(".")[0]


def test_no_module_imports_a_wsp_library():
    modules = sorted(SRC.rglob("*.py"))
    assert len(modules) > 200, f"only {len(modules)} modules found under {SRC}: the walk would check nothing"
    offenders = [
        f"{path.relative_to(SRC)} imports {root}"
        for path in modules
        for root in _imported_roots(ast.parse(path.read_text(encoding="utf-8")))
        if root in FORBIDDEN
    ]
    assert offenders == []


def test_the_walk_sees_each_import_form():
    for source in ("import weightspace", "from weightspace.edit import x", "import wsplab as w",
                   "__import__('dspy_wsp')", "importlib.import_module('weightspace.core')"):
        assert set(_imported_roots(ast.parse(source))) & FORBIDDEN, source
    assert not set(_imported_roots(ast.parse('"""uses weightspace"""\nfrom . import weightspace_notes'))) & FORBIDDEN
