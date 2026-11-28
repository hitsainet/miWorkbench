"""R3 review fix (OR3-045a0-2), 2026-11-16 (the reviewer's candidate test for R3-10, adopted).

A Celery task RUN INLINE in the API process (`task.apply(...)`, `task.run(...)`, or calling the task object itself) is
work the REST gate, the route walk and the dispatch scans never see: a GPU task run that way from an ungated route
would be GPU work in the API pod with no approval. So no code in src/api, src/services, src/core, src/utils, src/ml or
src/db may run a task imported from `src.workers` inline, except the reviewed rows below (pinned both ways). The
worker package is exempt: it IS the worker.

Controls (record `0xcc/reviews/045_A0_implementation_controls_2026-11-16.md`, R3 section):
  `run_probe_monitor.apply(kwargs=…)` in `POST /jlens/annotation-sweeps`   → red
  `run_probe_monitor(…)` called directly there                              → red
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
API_PROCESS = ("api/", "services/", "core/", "utils/", "ml/", "db/")


def _resolve(module: str, level: int, here: str) -> str:
    if level == 0:
        return module
    parts = ("src/" + here).rsplit("/", 1)[0].replace("/", ".").split(".")
    base = parts[: len(parts) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def _task_names(tree: ast.AST, here: str) -> dict[str, tuple[str, str]]:
    from celery.app.task import Task

    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = _resolve(node.module or "", node.level, here)
            if ".workers" not in mod and not mod.startswith("src.workers"):
                continue
            try:
                m = importlib.import_module(mod)
            except Exception:  # noqa: BLE001
                continue
            for alias in node.names:
                if isinstance(getattr(m, alias.name, None), Task):
                    out[alias.asname or alias.name] = (mod, alias.name)
    return out


def inline_task_runs():
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if not rel.startswith(API_PROCESS):
            continue
        tree = ast.parse(path.read_text())
        tasks = _task_names(tree, rel)
        if not tasks:
            continue
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            f = call.func
            if isinstance(f, ast.Name) and f.id in tasks:
                hits.append((rel, call.lineno, f.id, "called"))
            if (isinstance(f, ast.Attribute) and f.attr in ("apply", "run", "__call__")
                    and isinstance(f.value, ast.Name) and f.value.id in tasks):
                hits.append((rel, call.lineno, f.value.id, f.attr))
    return hits


#: Inline runs reviewed as not GPU work: (file, task name) -> reason. Pinned both ways.
INLINE_RUNS_REVIEWED = {
    ("api/v1/endpoints/datasets.py", "cancel_task"): "revokes a download task and marks the row; no model, no GPU",
    ("api/v1/endpoints/models.py", "cancel_download"): "revokes a model download and marks the row; no model, no GPU",
}


def test_the_scan_sees_the_reviewed_rows():
    """The scan is looking where the tasks are imported (a scan that finds nothing proves nothing)."""
    assert {(f, name) for f, _, name, _ in inline_task_runs()} >= set(INLINE_RUNS_REVIEWED)


def test_no_task_runs_inline_in_the_api_process():
    seen = {(f, name) for f, _, name, _ in inline_task_runs()}
    assert seen - set(INLINE_RUNS_REVIEWED) == set(), sorted(seen - set(INLINE_RUNS_REVIEWED))
    assert set(INLINE_RUNS_REVIEWED) <= seen, "a reviewed row no longer exists: remove it"
