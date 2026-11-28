"""Every string patch target under `backend/tests` names something that exists (043 FR-043.15; FTID §8; FTASKS 5.3),
and no `create=True` patch goes unexplained (ENH-011 WB-1, #16).

A `patch("src.services.old_name.f")` whose module was renamed raises at test time, which is loud; but one that
still resolves under an alias patches an object nothing calls, which is silent (miWorkbench memory "monkeypatches
that silently do nothing"). ENH-009 P9 renames modules (`millm_read` → `miserver_read`, `millm_utils` →  (p9: keep, brand-guard plant or rule)
`miserver_utils`), so every literal target is resolved here by import and attribute lookup, read from the test
sources' AST (never a regex over text).

`create=True` was skipped outright, and that hid two inert patches: `src.services.extraction_service.emit_progress`
(the service imports it inside the function, so the module attribute the patch created was never read) and
`src.workers.steering_tasks.__name__` (patched for no purpose). `create=True` makes a patch of a missing attribute
succeed, which is exactly the inert case, so now EVERY `create=True` patch, by string or by `patch.object`, needs an
entry in `CREATE_TRUE_ALLOWED` saying why the created attribute is read. A string target with no entry is also
resolved like any other.

Controls (0xcc/reviews/ENH-011_WB_controls_2026-11-26.md): plant a `create=True` on a missing `src.` name (C8) and
blank an allowlist reason (C9); each turns this file red.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import NamedTuple, Optional

from tests.support.source_tree import python_sources

TESTS = Path(__file__).resolve().parents[1]
PATCHERS = {"patch", "setattr", "delattr"}

#: `(file under backend/tests, target) → why the attribute `create=True` makes is read`. A string target is the
#: patched dotted name; a `patch.object(obj, "attr", …)` target is `<obj source>.<attr>`.
CREATE_TRUE_ALLOWED: dict[tuple[str, str], str] = {
    ("unit/test_steering_worker_lifecycle.py", "src.api.v1.endpoints.steering.open"):
        "builtin shadow: `open` is a builtin, so the module has no attribute until the patch makes one, and the "
        "module's own `open(STEERING_LOG, 'a')` resolves module globals before builtins, so it reads the patch",
    ("unit/test_finalize_wiring.py", "task_cls.get_db"):
        "a bound Celery task reads `get_db` off its class on each call; judged not inert 2026-11-15 (ENH-011 §1)",
    ("unit/test_labeling_claim_release.py", "mod.cleanup_stuck_labeling_task.get_db"):
        "a bound task reads the instance attribute `self.get_db` before its class's; judged not inert 2026-11-15",
    ("unit/test_labeling_resume_sweep.py", "mod.cleanup_stuck_labeling_task.get_db"):
        "a bound task reads the instance attribute `self.get_db` before its class's; judged not inert 2026-11-15",
}


class Target(NamedTuple):
    line: int
    target: str
    creates: bool
    by_string: bool


def _creates(node: ast.Call) -> bool:
    return any(k.arg == "create" and isinstance(k.value, ast.Constant) and k.value.value for k in node.keywords)


def _patch_object_target(node: ast.Call) -> Optional[str]:
    """`<obj>.<attr>` for `patch.object(obj, "attr", …)` (or `mock.patch.object`), else None."""
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "object"):
        return None
    owner = func.value
    named = owner.attr if isinstance(owner, ast.Attribute) else owner.id if isinstance(owner, ast.Name) else ""
    if named != "patch" or len(node.args) < 2:
        return None
    attr = node.args[1]
    if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
        return None
    return f"{ast.unparse(node.args[0])}.{attr.value}"


def _targets(tree: ast.AST) -> list[Target]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        first = node.args[0]
        if name in PATCHERS and isinstance(first, ast.Constant) and isinstance(first.value, str) \
                and first.value.startswith("src."):
            out.append(Target(node.lineno, first.value, _creates(node), True))
        elif (obj := _patch_object_target(node)) is not None and _creates(node):
            out.append(Target(node.lineno, obj, True, False))
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


def all_targets(root: Path = TESTS) -> list[tuple[str, Target]]:
    """A file another xdist worker deletes after the listing is skipped (`tests/support/source_tree.py`)."""
    found = []
    for path, text in python_sources(root):
        for target in _targets(ast.parse(text)):
            found.append((str(path.relative_to(root)), target))
    return found


def findings(root: Path = TESTS, allowed: dict[tuple[str, str], str] = CREATE_TRUE_ALLOWED) -> list[str]:
    """Every patch target that does not resolve, and every `create=True` patch the allowlist does not explain."""
    bad = []
    for path, t in all_targets(root):
        if t.creates:
            reason = allowed.get((path, t.target))
            if reason is None:
                bad.append(f"{path}:{t.line}: create=True on {t.target} has no CREATE_TRUE_ALLOWED reason: a created "
                           "attribute nothing reads makes the patch inert (patch where the code looks the name up)")
                if t.by_string and not resolves(t.target):
                    bad.append(f"{path}:{t.line}: {t.target} does not resolve")
            elif not reason.strip():
                bad.append(f"{path}:{t.line}: the CREATE_TRUE_ALLOWED reason for {t.target} is blank")
        elif t.by_string and not resolves(t.target):
            bad.append(f"{path}:{t.line}: {t.target}")
    return bad


def test_every_string_patch_target_resolves():
    targets = all_targets()
    assert len(targets) > 50, f"only {len(targets)} string patch targets found: the scan reads nothing"
    bad = findings()
    assert not bad, bad


def test_every_create_true_allowance_has_a_reason_and_is_used():
    """A blank reason explains nothing; an entry no patch uses would excuse the next one written there."""
    used = {(path, t.target) for path, t in all_targets() if t.creates}
    assert set(CREATE_TRUE_ALLOWED) <= used, sorted(set(CREATE_TRUE_ALLOWED) - used)
    assert all(reason.strip() for reason in CREATE_TRUE_ALLOWED.values())
    assert len(used) == len(CREATE_TRUE_ALLOWED) == 4, sorted(used)


def test_the_guard_sees_create_true_by_string_and_by_patch_object(tmp_path):
    (tmp_path / "unit").mkdir()
    (tmp_path / "unit" / "test_x.py").write_text(
        "from unittest.mock import patch\n"
        "with patch('src.services.extraction_service.emit_progress', create=True):\n    pass\n"
        "with patch.object(task, 'get_db', lambda: None, create=True):\n    pass\n"
        "with patch('src.workers.websocket_emitter.emit_progress'):\n    pass\n"
    )
    bad = findings(tmp_path, allowed={})
    assert len([b for b in bad if "has no CREATE_TRUE_ALLOWED reason" in b]) == 2, bad
    assert any("src.services.extraction_service.emit_progress does not resolve" in b for b in bad), bad
    assert findings(tmp_path, allowed={("unit/test_x.py", "src.services.extraction_service.emit_progress"): "r",
                                       ("unit/test_x.py", "task.get_db"): "r"}) == []
    assert any("is blank" in b for b in findings(tmp_path, allowed={
        ("unit/test_x.py", "src.services.extraction_service.emit_progress"): " ", ("unit/test_x.py", "task.get_db"): "r"}))


def test_a_file_deleted_after_the_listing_is_skipped(tmp_path, monkeypatch):
    """The xdist race (2026-11-28): another worker deletes its `test_zz_totals_probe_<pid>.py` between this scan's
    listing and its read. Simulated by a listed file whose read raises `FileNotFoundError`."""
    (tmp_path / "unit").mkdir()
    (tmp_path / "unit" / "test_x.py").write_text(
        "from unittest.mock import patch\nwith patch('src.workers.websocket_emitter.emit_progress'):\n    pass\n")
    gone = tmp_path / "unit" / "test_zz_totals_probe_1.py"
    gone.write_text("")
    real_read = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == gone:
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert [(rel, t.target) for rel, t in all_targets(tmp_path)] == [
        ("unit/test_x.py", "src.workers.websocket_emitter.emit_progress")]


def test_an_old_module_name_does_not_resolve():
    assert resolves("src.services.miserver_read.SessionReader")
    assert not resolves("src.services.mi" "llm_read.SessionReader")
