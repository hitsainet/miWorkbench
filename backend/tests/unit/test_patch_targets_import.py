"""Every string patch target under `backend/tests` names something that exists (043 FR-043.15; FTID §8; FTASKS 5.3).

A `patch("src.services.old_name.f")` whose module was renamed raises at test time, which is loud; but one that
still resolves under an alias patches an object nothing calls, which is silent (miWorkbench memory "monkeypatches
that silently do nothing"). ENH-009 P9 renames modules (`millm_read` → `miserver_read`, `millm_utils` →  (p9: keep, brand-guard plant or rule)
`miserver_utils`), so every literal target is resolved here by import and attribute lookup, read from the test
sources' AST (never a regex over text).
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
PATCHERS = {"patch", "setattr", "delattr"}


def _targets(tree: ast.AST) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        first = node.args[0]
        creates = any(k.arg == "create" and isinstance(k.value, ast.Constant) and k.value.value for k in node.keywords)
        if name in PATCHERS and isinstance(first, ast.Constant) and isinstance(first.value, str) \
                and first.value.startswith("src.") and not creates:  # `create=True` makes the attribute on purpose
            out.append((node.lineno, first.value))
    return out


def resolves(target: str) -> bool:
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return False
        return True
    return False


def all_targets() -> list[tuple[str, int, str]]:
    found = []
    for path in sorted(TESTS.rglob("*.py")):
        for line, target in _targets(ast.parse(path.read_text(encoding="utf-8"))):
            found.append((str(path.relative_to(TESTS)), line, target))
    return found


def test_every_string_patch_target_resolves():
    targets = all_targets()
    assert len(targets) > 50, f"only {len(targets)} string patch targets found: the scan reads nothing"
    bad = [f"{p}:{n}: {t}" for p, n, t in targets if not resolves(t)]
    assert not bad, bad


def test_an_old_module_name_does_not_resolve():
    assert resolves("src.services.miserver_read.SessionReader")
    assert not resolves("src.services.mi" "llm_read.SessionReader")
