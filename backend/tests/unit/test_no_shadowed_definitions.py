"""No module under `backend/src` defines a top-level function or class twice (044 R2 review fix OR2-044a-L3,
2026-11-17).

`probe_monitor_run.py` held two `_load_tokenizer_for_run` (G5's backfill added the second): Python keeps the last, so
every caller of the first — the render check, the memory forecast, the rule recommendation, the WSP bundle build —
silently ran the backfill's version and got its missing-model error. Nothing failed. The guard reads every module's
AST (top-level statements only), and the behaviour test pins that the two loaders are two functions again.

Controls (`0xcc/reviews/044a_r2_review_fixes_2026-11-17.md`): rename `_load_backfill_tokenizer` back to
`_load_tokenizer_for_run` → both tests red.
"""
from __future__ import annotations

import ast
import collections
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.wsp_world import sync_db  # noqa: F401 (fixture)

SRC = Path(__file__).resolve().parents[2] / "src"


def test_no_module_defines_a_name_twice():
    modules, shadowed = 0, {}
    for path in sorted(SRC.rglob("*.py")):
        modules += 1
        names = collections.Counter(node.name for node in ast.parse(path.read_text()).body
                                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        twice = sorted(name for name, count in names.items() if count > 1)
        if twice:
            shadowed[str(path.relative_to(SRC))] = twice
    assert modules > 300, f"only {modules} modules read: the scan is not looking at backend/src"
    assert shadowed == {}, f"a later definition silently replaces an earlier one: {shadowed}"


def test_the_shared_tokenizer_loader_keeps_its_own_missing_model_error(sync_db, monkeypatch):
    """The render check's loader answers a missing model with its `ValueError`; the backfill's answers its own 404."""
    from sqlalchemy.orm import sessionmaker

    from src.core import database
    from src.services import probe_monitor_run as run_module

    monkeypatch.setattr(database, "SyncSessionLocal", sessionmaker(bind=sync_db.get_bind()))
    absent = SimpleNamespace(model_id="m_absent00000000")
    with pytest.raises(ValueError, match="not found") as shared:
        run_module._load_tokenizer_for_run(absent)
    assert not isinstance(shared.value, run_module.TruncationBackfillRefused)
    with pytest.raises(run_module.TruncationBackfillRefused) as backfill:
        run_module._load_backfill_tokenizer(absent)
    assert (backfill.value.code, backfill.value.status) == ("model_not_found", 404)
