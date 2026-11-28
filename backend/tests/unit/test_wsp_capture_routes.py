"""ENH-010 044 task 4.3: `POST /wsp/captures` plans the sets, makes one row per set and dispatches ONE GPU task with
exactly those rows; a refused plan answers its code; `GET /wsp/captures` lists the rows. Through the real app."""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from src.core.config import settings
from tests.support.wsp_world import add_view, seed_probe, sync_db  # noqa: F401 (fixture)


@pytest.fixture
def dispatched(monkeypatch):
    from types import SimpleNamespace

    from src.services import gpu_dispatch

    calls = []

    # The route sends through `dispatch_gpu_task` under the id of the task-queue row it opened first (2026-11-27).
    def fake_dispatch(task, *, gpu_request, kwargs, task_id):
        calls.append({"task": task.name, "gpu_request": gpu_request, "kwargs": kwargs})
        return SimpleNamespace(id=f"task-{len(calls)}")

    monkeypatch.setattr(gpu_dispatch, "dispatch_gpu_task", fake_dispatch)
    return calls


@pytest.fixture
def gates_pass(monkeypatch):
    """The precision and render gates answer for a real model snapshot this world does not have; their wiring on the
    route is pinned by `test_the_precision_gate_is_on_the_route` below."""
    from src.api.v1.endpoints import probe_monitors

    async def passes(_probe_id):
        return None

    monkeypatch.setattr(probe_monitors, "refuse_precision_mismatch", passes)
    monkeypatch.setattr(probe_monitors, "refuse_render_mismatch", passes)


def _evaluated_probe(db, tmp_path, *, with_positions=True):
    from src.models.probe_monitor import ProbeMonitorEvaluation

    world = seed_probe(db, weights_path="/x.safetensors", threshold=0.5)
    view = add_view(db, world, role="eval")
    paths = {}
    for name, values, dtype in (("scores", [0.1, 0.9, 0.4], np.float32), ("labels", [0, 1, 0], np.int8),
                                ("positions", [3, 5, 8], np.int64)):
        paths[name] = tmp_path / f"{view}_{name}.npy"
        np.save(paths[name], np.asarray(values, dtype=dtype))
    evaluation = ProbeMonitorEvaluation(probe_id=world.probe_id, dataset_id=view, status="completed", metrics={},
                                        scores_path=str(paths["scores"]), labels_path=str(paths["labels"]),
                                        positions_path=str(paths["positions"]) if with_positions else None)
    db.add(evaluation)
    db.commit()
    return world, view, evaluation.id


@pytest.mark.asyncio
async def test_one_task_with_exactly_the_planned_rows(client, sync_db, tmp_path, dispatched, gates_pass):
    from src.models.wsp import WspActivationCapture

    world, view, evaluation_id = _evaluated_probe(sync_db, tmp_path)
    response = await client.post("/api/v1/wsp/captures", json={"probe_id": world.probe_id})
    assert response.status_code == 202, response.text
    body = response.json()
    rows = sync_db.query(WspActivationCapture).filter(WspActivationCapture.probe_id == world.probe_id).all()
    assert [(r.dataset_id, r.role, r.source_evaluation_id, r.status) for r in rows] == [
        (view, "eval", evaluation_id, "pending")]
    assert dispatched == [{"task": "src.workers.wsp_tasks.capture_wsp_heldout", "gpu_request": "auto",
                           "kwargs": {"probe_id": world.probe_id, "capture_ids": [rows[0].id]}}]
    assert body["started"] == [rows[0].id] and body["task_id"] == "task-1"

    # Asked again while pending: the row is kept and nothing is dispatched twice.
    again = await client.post("/api/v1/wsp/captures", json={"probe_id": world.probe_id})
    assert again.status_code == 202 and again.json()["started"] == [] and len(dispatched) == 1

    listed = await client.get("/api/v1/wsp/captures", params={"probe_id": world.probe_id})
    assert [c["id"] for c in listed.json()["captures"]] == [rows[0].id]


@pytest.mark.asyncio
async def test_a_failed_capture_is_rearmed_not_duplicated(client, sync_db, tmp_path, dispatched, gates_pass):
    from src.models.wsp import WspActivationCapture

    world, view, evaluation_id = _evaluated_probe(sync_db, tmp_path)
    old = WspActivationCapture(probe_id=world.probe_id, dataset_id=view, role="eval",
                               source_evaluation_id=evaluation_id, layer=1, scope="all", status="parity_failed",
                               parity={"passed": False})
    sync_db.add(old)
    sync_db.commit()
    response = await client.post("/api/v1/wsp/captures", json={"probe_id": world.probe_id})
    assert response.status_code == 202, response.text
    sync_db.refresh(old)
    assert old.status == "pending" and old.parity is None
    assert dispatched[0]["kwargs"]["capture_ids"] == [old.id]


@pytest.mark.asyncio
async def test_unrecorded_positions_answer_409_and_dispatch_nothing(client, sync_db, tmp_path, dispatched, gates_pass):
    world, _view, _ = _evaluated_probe(sync_db, tmp_path, with_positions=False)
    response = await client.post("/api/v1/wsp/captures", json={"probe_id": world.probe_id})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "evaluation_positions_not_recorded" and dispatched == []


@pytest.mark.asyncio
async def test_the_precision_gate_is_on_the_route(client, sync_db, tmp_path, dispatched):
    """A run that predates precision recording refuses BEFORE a GPU task is queued (the evaluate route's rule)."""
    world, _view, _ = _evaluated_probe(sync_db, tmp_path)
    response = await client.post("/api/v1/wsp/captures", json={"probe_id": world.probe_id})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "precision_mismatch"
    assert dispatched == []


# ── R2 review fix (OR2-044a-L6), 2026-11-17: two requests at once make one row per set ──────────────────────────


def _calibrated_probe(db, tmp_path):
    """`_evaluated_probe` plus a calibration set with recorded rows, so the plan holds an eval AND a calibration set
    (the calibration capture's unique key holds a NULL `source_evaluation_id`)."""
    from src.models.probe_monitor import ProbeMonitor

    world, view, evaluation_id = _evaluated_probe(db, tmp_path)
    calibration = add_view(db, world, role="calibration")
    for name, values, dtype in (("cal_scores", [0.2, 0.3], np.float32), ("cal_positions", [1, 4], np.int64)):
        np.save(tmp_path / f"{name}.npy", np.asarray(values, dtype=dtype))
    probe = db.get(ProbeMonitor, world.probe_id)
    probe.threshold_source, probe.calibration_dataset_id = "calibration_set", calibration
    probe.calibration_scores_path = str(tmp_path / "cal_scores.npy")
    probe.calibration_positions_path = str(tmp_path / "cal_positions.npy")
    db.commit()
    return world, view, calibration


def test_concurrent_plans_make_one_row_per_set(sync_db, tmp_path, monkeypatch):
    """Two plans for one probe at once, each pausing between reading the existing rows and writing its own: the
    per-probe advisory lock lets the second see the first's rows, so there is ONE eval row and ONE calibration row, one
    caller has both to run and the other none (no second GPU capture, no IntegrityError 500). Control: drop the
    `pg_advisory_xact_lock` → red."""
    import threading
    import time

    from sqlalchemy.orm import sessionmaker

    from src.api.v1.endpoints import wsp
    from src.models.wsp import WspActivationCapture

    world, view, calibration = _calibrated_probe(sync_db, tmp_path)
    monkeypatch.setattr(wsp, "_BETWEEN_PLAN_AND_WRITE", lambda: time.sleep(1.0))
    Session = sessionmaker(bind=sync_db.get_bind())
    outcomes, errors = [], []

    def plan():
        session = Session()
        try:
            outcomes.append(len(wsp.plan_capture_rows(session, world.probe_id, None)["to_run"]))
        except Exception as exc:  # noqa: BLE001 - the race's old answer (an IntegrityError) is the failure
            errors.append(repr(exc)[:300])
        finally:
            session.close()

    threads = [threading.Thread(target=plan) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and sorted(outcomes) == [0, 2], (outcomes, errors)
    sync_db.expire_all()
    rows = sync_db.query(WspActivationCapture).filter(WspActivationCapture.probe_id == world.probe_id).all()
    assert sorted((r.dataset_id, r.role) for r in rows) == sorted([(view, "eval"), (calibration, "calibration")])


def test_the_database_refuses_a_second_calibration_capture(sync_db, tmp_path):
    """The backstop: `uq_wsp_capture_calibration` refuses a second calibration capture of one view for one probe,
    whatever the code above does (a unique key over a NULL never would). An eval capture whose evaluation was deleted
    (`source_evaluation_id` NULL) is not caught by it. Control: drop the index from the model → red."""
    from sqlalchemy.exc import IntegrityError

    from src.models.wsp import WspActivationCapture

    world, view, calibration = _calibrated_probe(sync_db, tmp_path)

    def capture(dataset_id, role):
        return WspActivationCapture(probe_id=world.probe_id, dataset_id=dataset_id, role=role, layer=1, scope="all",
                                    source_evaluation_id=None, status="pending")

    sync_db.add_all([capture(view, "eval"), capture(view, "eval"), capture(calibration, "calibration")])
    sync_db.commit()
    sync_db.add(capture(calibration, "calibration"))
    with pytest.raises(IntegrityError, match="uq_wsp_capture_calibration"):
        sync_db.commit()
    sync_db.rollback()

    # R4 (C044a-M1): the index covers the CURRENT row only. A superseded calibration capture is history beside its
    # replacement, and the replacement may itself be superseded; two current rows are still refused.
    from src.core.clock import utc_now

    current = (sync_db.query(WspActivationCapture)
               .filter(WspActivationCapture.probe_id == world.probe_id, WspActivationCapture.role == "calibration")
               .one())
    for _ in range(2):
        replacement = capture(calibration, "calibration")
        replacement.id = f"wac_{uuid.uuid4().hex[:12]}"
        current.superseded_at, current.superseded_by = utc_now(), replacement.id
        sync_db.flush()
        sync_db.add(replacement)
        sync_db.commit()
        current = replacement
    sync_db.add(capture(calibration, "calibration"))
    with pytest.raises(IntegrityError, match="uq_wsp_capture_calibration"):
        sync_db.commit()
    sync_db.rollback()
    rows = sync_db.query(WspActivationCapture).filter(WspActivationCapture.probe_id == world.probe_id,
                                                      WspActivationCapture.role == "calibration").all()
    assert len(rows) == 3 and [r.id for r in rows if r.superseded_at is None] == [current.id]


# ── R3 review fix (OR3-044a-L3), 2026-11-17: a ready calibration capture follows the probe's bar ────────────────────


@pytest.mark.asyncio
async def test_a_calibration_capture_of_an_old_bar_is_superseded_not_reset(client, sync_db, tmp_path, dispatched,
                                                                          gates_pass):
    """A READY calibration capture records the bar and the calibration rows it was checked against. While both still
    match the probe it is kept (asked again: not re-run, and bundles use it). After a recalibration (a new threshold)
    or a G5 re-cut (new calibration arrays, same threshold) it is no longer offered to a bundle, and the next capture
    request SUPERSEDES it with a new row (R4 review fix C044a-M1): the old row keeps its status, parity and directory
    (an earlier bundle names it and is scored on it), gains `superseded_at` and `superseded_by`, and the new row is
    the one captured and offered. One CURRENT calibration row per view throughout.
    Controls: never supersede a ready row (OR3-044a-L3 reverted) → red; reset it in place (round 3) → red."""
    from src.models.probe_monitor import ProbeMonitor
    from src.models.wsp import WspActivationCapture
    from src.services.wsp_bundle import calibration_capture_missing, ready_captures
    from src.services.wsp_capture import calibration_arrays_sha256

    world, _view, calibration = _calibrated_probe(sync_db, tmp_path)
    probe = sync_db.get(ProbeMonitor, world.probe_id)
    checked = {"threshold_served": 0.5, "passed": True,
               "calibration_arrays_sha256": calibration_arrays_sha256([1, 4], [0.2, 0.3], None)}
    row = WspActivationCapture(probe_id=world.probe_id, dataset_id=calibration, role="calibration", layer=1,
                               scope="all", status="ready", parity=checked, dir_path="/sealed/first")
    sync_db.add(row)
    sync_db.commit()

    async def ask():
        response = await client.post("/api/v1/wsp/captures", json={"probe_id": world.probe_id})
        assert response.status_code == 202, response.text
        sync_db.expire_all()
        return response.json()

    def calibration_rows():
        return (sync_db.query(WspActivationCapture)
                .filter(WspActivationCapture.probe_id == world.probe_id, WspActivationCapture.role == "calibration")
                .order_by(WspActivationCapture.created_at).all())

    def sealed(capture, parity, directory):
        sync_db.refresh(capture)
        return (capture.status, capture.parity, capture.dir_path) == ("ready", parity, directory)

    # Current: kept, and offered to a bundle.
    assert row.id not in (await ask())["started"] and sealed(row, checked, "/sealed/first")
    assert row.id in [c.id for c in ready_captures(sync_db, probe)]

    # A recalibration moved the bar: not offered, then superseded by a NEW row (the old one untouched).
    probe.threshold = 0.6
    sync_db.commit()
    assert row.id not in [c.id for c in ready_captures(sync_db, probe)]
    assert calibration_capture_missing(probe, ready_captures(sync_db, probe)) is not None
    answer = await ask()
    second = sync_db.get(WspActivationCapture, answer["started"][0])
    assert answer["started"] == [second.id] and second.id != row.id and second.status == "pending"
    assert sealed(row, checked, "/sealed/first")
    assert row.superseded_at is not None and row.superseded_by == second.id and second.superseded_at is None
    listed = {c["id"]: c for c in answer["captures"]}
    assert row.id not in listed and listed[second.id]["superseded_by"] is None
    assert row.id not in [c.id for c in ready_captures(sync_db, probe)]   # superseded: never offered again
    assert second.id not in (await ask())["started"]                       # pending: kept, not asked twice

    # The new row captured against the new bar; then a re-cut rewrites the calibration rows under the same threshold.
    second.status, second.parity, second.dir_path = "ready", {**checked, "threshold_served": 0.6}, "/sealed/second"
    sync_db.commit()
    assert [c.id for c in ready_captures(sync_db, probe) if c.role == "calibration"] == [second.id]
    np.save(tmp_path / "cal_scores.npy", np.asarray([0.25, 0.3], dtype=np.float32))
    assert second.id not in [c.id for c in ready_captures(sync_db, probe)]
    third_id = (await ask())["started"][0]
    assert third_id not in (row.id, second.id)
    assert sealed(row, checked, "/sealed/first") and sealed(second, {**checked, "threshold_served": 0.6},
                                                             "/sealed/second")
    rows = calibration_rows()
    assert [r.id for r in rows] == [row.id, second.id, third_id]
    assert [r.superseded_by for r in rows] == [second.id, third_id, None]
    assert [r.id for r in rows if r.superseded_at is None] == [third_id]
