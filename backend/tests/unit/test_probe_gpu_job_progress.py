"""Long probe GPU jobs show live progress: the meter, the task's row and Celery state, the routes, and the readers.

The operator asked of a two-hour probe evaluation (2026-11-27) "what is this job? where can I see its progress in a
UI?" The evaluation (`evaluate_probe_monitor`) and the WSP held-out capture (`capture_wsp_heldout`) had no row: they
were bare GPU-queue messages with `progress: null`. Now:

* the route opens a `task_queue` row (`probe_evaluation` / `wsp_heldout_capture`, entity the probe) under the Celery
  id it sends with, before sending (the ENH-006 G17 order);
* the task moves it (`jlens_progress.RowNarration`): running, the percentage on the row, and the reading — view k of
  n, rows x of y, rate, ETA — in its Celery state as `progress_detail`, then completed or failed;
* `list_active_jobs` returns that reading beside the row, and the gpu-queue message is no longer a second, blind entry;
* `GET /task-queue/active` (the Monitor and the Probes page) carries it in `entity_info`, with the probe named.

The real scoring loops are exercised in `test_probe_gpu_job_progress_pipeline.py`.

Controls (0xcc/reviews/probe_gpu_job_progress_controls_2026-11-27.md): the publish call, each route's row, the
task's row writes, and `attach_live_progress` are each removed in turn; each must turn this file red.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from tests.support.wsp_world import add_view, seed_probe, sync_db  # noqa: F401 (fixture)


# ── the meter ────────────────────────────────────────────────────────────────────────────────────


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _meter(clock, every=5.0):
    from src.services.live_progress import ViewRowsProgress

    writes = []
    meter = ViewRowsProgress("probe_evaluation", lambda p, d: writes.append((round(p, 3), d)), clock=clock,
                             every_seconds=every)
    return meter, writes


def _short(detail):
    return (detail["phase"], detail["set_index"], detail["set_rows_done"], detail["set_rows_total"],
            detail["rows_done"], detail["rows_total"], detail["rows_estimated"])


def test_the_meter_writes_at_each_boundary_and_on_a_time_throttle_inside_a_view():
    clock = Clock()
    meter, writes = _meter(clock)
    meter.plan([("pmd_a", "alpha", 100), ("pmd_b", "beta", 50)])
    meter.begin()
    meter.segment(0, 2, "alpha")
    clock.now += 6.0
    meter.rows(30, 90)                  # the view rendered 90 rows, not the estimated 100
    meter(1, 3)                         # 6 s after the last write: written
    clock.now += 1.0
    meter.rows(60, 90)
    meter(2, 3)                         # 1 s later: throttled
    clock.now += 1.0
    meter.rows(90, 90)
    meter(3, 3)                         # a view's last batch always writes
    meter.segment(1, 2, "beta")
    clock.now += 1.0
    meter.rows(50, 50)
    meter(1, 1)
    meter.finish()

    assert [_short(d) for _p, d in writes] == [
        ("loading_model", None, None, None, 0, 150, True),
        ("preparing", 0, 0, 100, 0, 150, True),
        ("scoring", 0, 30, 90, 30, 140, True),
        ("scoring", 0, 90, 90, 90, 140, True),
        ("preparing", 1, 0, 50, 90, 140, True),
        ("scoring", 1, 50, 50, 140, 140, False),
        ("done", None, None, None, 140, 140, False),
    ]
    assert [p for p, _d in writes] == [0.0, 0.0, 21.429, 64.286, 64.286, 100.0, 100.0]
    assert writes[2][1]["set_name"] == "alpha" and writes[4][1]["set_name"] == "beta"
    assert {d["kind"] for _p, d in writes} == {"probe_evaluation"} and writes[-1][1]["set_count"] == 2


def test_a_view_that_never_began_leaves_the_total_and_a_write_failure_breaks_nothing():
    from src.services.live_progress import ViewRowsProgress

    clock = Clock()
    seen = []

    def write(progress, detail):
        seen.append(detail["rows_total"])
        raise RuntimeError("database gone")

    meter = ViewRowsProgress("wsp_heldout_capture", write, clock=clock)
    meter.plan([("a", "a", 10), ("gone", "gone", 40)], exact=True)
    meter.begin()
    meter.segment(0, 2, "")
    meter.rows(10, 10)
    meter(1, 1)
    meter.finish()                      # "gone" never began: it scored nothing
    assert seen == [50, 50, 50, 10]
    assert meter.writes == 4


# ── the task moves its row and publishes its reading ────────────────────────────────────────────


def _row(task_id):
    from src.core.database import get_sync_db
    from src.models.task_queue import TaskQueue

    with get_sync_db() as db:
        row = db.query(TaskQueue).filter(TaskQueue.task_id == task_id).one()
        return {"status": row.status, "progress": row.progress, "error": row.error_message,
                "gpu_uuid": row.gpu_uuid, "gpu_uuids": row.gpu_uuids, "started": row.started_at is not None}


def _celery_info(task_id):
    from celery.result import AsyncResult

    from src.core.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)
    return result.state, result.info


def _open(task_type, entity_id, task_id):
    from src.workers import jlens_progress

    assert jlens_progress.open_row(task_type, entity_id, task_id, {"gpu": "auto"}, "probe_monitor")


def _drive(progress, check):
    """What a two-view job tells its meter; `check()` looks at the row and Celery state as view 2 begins."""
    progress.plan([("pmd_a", "alpha", 100), ("pmd_b", "beta", 60)])
    progress.begin()
    progress.placed("GPU-aaaa", ["GPU-aaaa", "GPU-bbbb"])
    progress.segment(0, 2, "alpha")
    progress.rows(40, 100)
    progress(1, 3)
    progress.rows(100, 100)
    progress(3, 3)
    progress.segment(1, 2, "beta")
    check()
    progress.rows(60, 60)
    progress(1, 1)
    progress.finish()


#: What the row and the reading say as view 2 of 2 begins: 100 of 160 rows.
MID_DETAIL = {"phase": "preparing", "set_index": 1, "set_count": 2, "set_name": "beta", "set_rows_done": 0,
              "set_rows_total": 60, "rows_done": 100, "rows_total": 160}


def _run(task, task_id, **kwargs):
    task.push_request(id=task_id)
    try:
        return task.run(**kwargs)
    finally:
        task.pop_request()


@pytest.fixture
def evaluation(monkeypatch, async_session):
    from src.services import probe_monitor_run as run_module
    from src.workers import probe_monitor_tasks as tasks

    monkeypatch.setattr(tasks, "_require_row", lambda *a, **k: None)
    monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)
    seen = {}

    def fake_evaluate(db, probe_id, dataset_ids, sample_rows=None, progress=None):
        assert progress is not None, "the task must hand the evaluation a meter"
        _drive(progress, lambda: seen.setdefault("mid", (_row(seen["task_id"]), _celery_info(seen["task_id"]))))
        if seen.get("raise"):
            raise seen["raise"]
        return {"probe_id": probe_id, "evaluations": 2}

    monkeypatch.setattr(run_module, "evaluate_probe", fake_evaluate)
    return seen


def test_the_evaluation_task_moves_its_row_and_publishes_view_and_rows(evaluation):
    from src.workers import jlens_progress
    from src.workers import probe_monitor_tasks as tasks

    task_id = str(uuid.uuid4())
    evaluation["task_id"] = task_id
    _open(jlens_progress.PROBE_EVALUATION, "pm_x", task_id)

    assert _run(tasks.evaluate_probe_monitor, task_id, probe_id="pm_x", dataset_ids=["pmd_a", "pmd_b"]) == {
        "probe_id": "pm_x", "evaluations": 2}

    row, (state, info) = evaluation["mid"]
    # While it runs: the row is running at the overall percentage, on BOTH cards it was placed on (1+n GPUs).
    assert (row["status"], row["progress"], row["gpu_uuid"], row["gpu_uuids"], row["started"]) == (
        "running", 62.5, "GPU-aaaa", ["GPU-aaaa", "GPU-bbbb"], True)
    assert state == "PROGRESS" and set(info) == {"progress_detail", "heartbeat"}
    detail = info["progress_detail"]
    assert detail["kind"] == "probe_evaluation"
    assert {k: detail[k] for k in MID_DETAIL} == MID_DETAIL
    # Then: completed at 100 %.
    end = _row(task_id)
    assert (end["status"], end["progress"]) == ("completed", 100.0)


def test_a_failed_evaluation_fails_its_row_with_the_reason(evaluation):
    from src.workers import jlens_progress
    from src.workers import probe_monitor_tasks as tasks

    task_id = str(uuid.uuid4())
    evaluation.update(task_id=task_id, **{"raise": ValueError("view pmd_b is too short")})
    _open(jlens_progress.PROBE_EVALUATION, "pm_x", task_id)
    with pytest.raises(ValueError):
        _run(tasks.evaluate_probe_monitor, task_id, probe_id="pm_x", dataset_ids=["pmd_a", "pmd_b"])
    end = _row(task_id)
    assert (end["status"], end["error"]) == ("failed", "ValueError: view pmd_b is too short")


def test_a_hand_off_puts_the_row_back_in_the_queue_not_failed(evaluation):
    from src.services.gpu_job_claim import JobHandoff
    from src.workers import jlens_progress
    from src.workers import probe_monitor_tasks as tasks

    task_id = str(uuid.uuid4())
    evaluation.update(task_id=task_id, **{"raise": JobHandoff("gpu.auto", "both cards are busy")})
    _open(jlens_progress.PROBE_EVALUATION, "pm_x", task_id)
    with pytest.raises(JobHandoff):
        _run(tasks.evaluate_probe_monitor, task_id, probe_id="pm_x", dataset_ids=["pmd_a"])
    end = _row(task_id)
    assert (end["status"], end["error"]) == ("queued", None)


def test_the_capture_task_moves_its_row_and_publishes_set_and_rows(monkeypatch, async_session):
    from src.services import wsp_capture
    from src.workers import jlens_progress, wsp_tasks

    monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)
    task_id = str(uuid.uuid4())
    seen = {}

    def fake_run_captures(db, probe_id, capture_ids, *, model_loader=None, progress=None):
        assert progress is not None, "the task must hand the capture a meter"
        _drive(progress, lambda: seen.setdefault("mid", (_row(task_id), _celery_info(task_id))))
        return {"probe_id": probe_id, "captures": {c: "ready" for c in capture_ids}}

    monkeypatch.setattr(wsp_capture, "run_captures", fake_run_captures)
    _open(jlens_progress.WSP_HELDOUT_CAPTURE, "pm_x", task_id)
    out = _run(wsp_tasks.capture_wsp_heldout, task_id, probe_id="pm_x", capture_ids=["wac_1", "wac_2"])
    assert out["captures"] == {"wac_1": "ready", "wac_2": "ready"}
    row, (state, info) = seen["mid"]
    assert (row["status"], row["progress"], row["gpu_uuids"]) == ("running", 62.5, ["GPU-aaaa", "GPU-bbbb"])
    assert state == "PROGRESS" and info["progress_detail"]["kind"] == "wsp_heldout_capture"
    assert {k: info["progress_detail"][k] for k in MID_DETAIL} == MID_DETAIL
    assert _row(task_id)["status"] == "completed"


def test_a_task_with_no_row_still_publishes_and_does_not_raise(monkeypatch, async_session):
    """Dispatched by something that opened no row: the Celery state still carries the reading."""
    from src.workers.jlens_progress import RowNarration
    from src.workers.probe_monitor_tasks import evaluate_probe_monitor

    task_id = str(uuid.uuid4())
    narration = RowNarration(SimpleNamespace(request=SimpleNamespace(id=task_id),
                                             update_state=evaluate_probe_monitor.update_state))
    import src.workers.jlens_progress as jp

    monkeypatch.setattr(jp, "mark_running", lambda *a, **k: False)
    narration.start()
    meter = narration.meter(jp.PROBE_EVALUATION)
    meter.plan([("a", "a", 10)])
    meter.begin()
    state, info = _celery_info(task_id)
    assert state == "PROGRESS" and info["progress_detail"]["rows_total"] == 10
    narration.completed()


# ── the routes open the row, under the id they send with, before they send ──────────────────────


@pytest.fixture
def sent(monkeypatch):
    from src.api.v1.endpoints import probe_monitors as ep
    from src.services import gpu_dispatch

    calls = []

    def dispatch(task, *, gpu_request, kwargs, task_id):
        calls.append({"task": task.name, "gpu_request": gpu_request, "kwargs": kwargs, "task_id": task_id,
                      "row_at_send": _row(task_id)})
        return SimpleNamespace(id=task_id)

    # The evaluate route holds the name from its module import; the capture route imports it per call.
    monkeypatch.setattr(ep, "dispatch_gpu_task", dispatch)
    monkeypatch.setattr(gpu_dispatch, "dispatch_gpu_task", dispatch)
    return calls


async def _no_refusal(*_a, **_k):
    return None


async def test_the_evaluate_route_opens_its_row_then_sends_under_its_id(client, sync_db, sent, monkeypatch):
    from src.api.v1.endpoints import probe_monitors as ep
    from src.models.task_queue import TaskQueue

    monkeypatch.setattr(ep, "refuse_precision_mismatch", _no_refusal)
    monkeypatch.setattr(ep, "refuse_render_mismatch", _no_refusal)
    world = seed_probe(sync_db)
    view = add_view(sync_db, world, role="eval")
    r = await client.post(f"/api/v1/probe-monitors/probes/{world.probe_id}/evaluate", json=[view])
    assert r.status_code == 202, r.text
    body = r.json()
    assert len(sent) == 1
    call = sent[0]
    assert (call["task"], call["gpu_request"], call["kwargs"]) == (
        "src.workers.probe_monitor_tasks.evaluate_probe_monitor", "auto",
        {"probe_id": world.probe_id, "dataset_ids": [view], "sample_rows": None})
    assert call["row_at_send"]["status"] == "queued", "the row must be committed before the task is sent"
    assert body["task_id"] == call["task_id"] and body["task_queue_id"].startswith("tq_")
    sync_db.expire_all()
    row = sync_db.query(TaskQueue).filter(TaskQueue.id == body["task_queue_id"]).one()
    assert (row.task_type, row.entity_id, row.entity_type, row.task_id, row.status) == (
        "probe_evaluation", world.probe_id, "probe_monitor", call["task_id"], "queued")


async def test_the_capture_route_opens_its_row_then_sends_under_its_id(client, sync_db, tmp_path, sent, monkeypatch):
    from src.api.v1.endpoints import probe_monitors as ep
    from src.models.task_queue import TaskQueue
    from tests.unit.test_wsp_capture_routes import _evaluated_probe

    monkeypatch.setattr(ep, "refuse_precision_mismatch", _no_refusal)
    monkeypatch.setattr(ep, "refuse_render_mismatch", _no_refusal)
    world, _view, _evaluation = _evaluated_probe(sync_db, tmp_path)
    r = await client.post("/api/v1/wsp/captures", json={"probe_id": world.probe_id})
    assert r.status_code == 202, r.text
    body = r.json()
    assert len(sent) == 1 and sent[0]["task"] == "src.workers.wsp_tasks.capture_wsp_heldout"
    assert sent[0]["kwargs"] == {"probe_id": world.probe_id, "capture_ids": body["started"]}
    assert sent[0]["row_at_send"]["status"] == "queued"
    sync_db.expire_all()
    row = sync_db.query(TaskQueue).filter(TaskQueue.id == body["task_queue_id"]).one()
    assert (row.task_type, row.entity_id, row.entity_type, row.task_id) == (
        "wsp_heldout_capture", world.probe_id, "probe_monitor", body["task_id"])


async def test_a_send_that_fails_fails_its_row(client, sync_db, monkeypatch):
    """The row was opened first, so a broker that refuses the send must not leave it `queued`."""
    from src.api.v1.endpoints import probe_monitors as ep

    monkeypatch.setattr(ep, "refuse_precision_mismatch", _no_refusal)
    monkeypatch.setattr(ep, "refuse_render_mismatch", _no_refusal)
    ids = []

    def dispatch(task, *, gpu_request, kwargs, task_id):
        ids.append(task_id)
        raise ConnectionError("broker refused")

    monkeypatch.setattr(ep, "dispatch_gpu_task", dispatch)
    world = seed_probe(sync_db)
    view = add_view(sync_db, world, role="eval")
    with pytest.raises(ConnectionError):
        await client.post(f"/api/v1/probe-monitors/probes/{world.probe_id}/evaluate", json=[view])
    row = _row(ids[0])
    assert (row["status"], row["error"]) == ("failed", "not dispatched: ConnectionError: broker refused")


async def test_a_row_that_cannot_be_written_sends_nothing(client, sync_db, sent, monkeypatch):
    from src.api.v1.endpoints import probe_monitors as ep
    from src.workers import jlens_progress

    monkeypatch.setattr(ep, "refuse_precision_mismatch", _no_refusal)
    monkeypatch.setattr(ep, "refuse_render_mismatch", _no_refusal)
    monkeypatch.setattr(jlens_progress, "open_row", lambda *a, **k: None)
    world = seed_probe(sync_db)
    view = add_view(sync_db, world, role="eval")
    r = await client.post(f"/api/v1/probe-monitors/probes/{world.probe_id}/evaluate", json=[view])
    assert r.status_code == 503 and "nothing was dispatched" in r.json()["detail"]
    assert sent == []


# ── the readers: list_active_jobs and GET /task-queue/active ───────────────────────────────────


def _running_evaluation(probe_id):
    """A running evaluation row with a published reading, as the task leaves it mid-run."""
    from src.workers import jlens_progress
    from src.workers.probe_monitor_tasks import evaluate_probe_monitor

    task_id = str(uuid.uuid4())
    _open(jlens_progress.PROBE_EVALUATION, probe_id, task_id)
    narration = jlens_progress.RowNarration(SimpleNamespace(request=SimpleNamespace(id=task_id),
                                                            update_state=evaluate_probe_monitor.update_state))
    narration.start()
    from src.services.live_progress import ViewRowsProgress

    # Unthrottled, so the reading mid-view is the one left behind (the task's meter writes every 5 s).
    meter = ViewRowsProgress(jlens_progress.PROBE_EVALUATION, narration.write, on_placed=narration.placed,
                             every_seconds=0.0)
    meter.plan([("pmd_a", "alpha", 100), ("pmd_b", "beta", 300)])
    meter.begin()
    meter.placed("GPU-cccc", None)
    meter.segment(0, 2, "alpha")
    meter.rows(100, 100)
    meter(1, 1)
    meter.segment(1, 2, "beta")
    meter.rows(150, 300)
    meter(1, 2)
    return task_id


async def test_list_active_jobs_returns_the_evaluation_with_its_reading_once(async_session, sync_db, monkeypatch):
    from src.services import active_jobs, gpu_queue_state

    world = seed_probe(sync_db)
    task_id = _running_evaluation(world.probe_id)
    # The broker also holds the evaluation's message: it must not come back as a second, blind job.
    monkeypatch.setattr(gpu_queue_state, "read_gpu_messages", lambda **k: [
        {"task_id": task_id, "task": "src.workers.probe_monitor_tasks.evaluate_probe_monitor", "queue": "gpu.auto",
         "state": "delivered", "orphan_reason": None}])
    body = await active_jobs.list_active_jobs(async_session, cards=[])
    mine = [job for job in body["jobs"] if job.get("task_id") == task_id]
    assert len(mine) == 1, mine
    job = mine[0]
    assert (job["kind"], job["task_type"], job["entity_id"], job["entity_type"], job["status"], job["progress"],
            job["gpu_uuid"]) == ("task_queue", "probe_evaluation", world.probe_id, "probe_monitor", "running", 62.5,
                                 "GPU-cccc")
    detail = job["progress_detail"]
    assert (detail["set_index"], detail["set_count"], detail["set_name"], detail["set_rows_done"],
            detail["set_rows_total"], detail["rows_done"], detail["rows_total"]) == (1, 2, "beta", 150, 300, 250, 400)
    assert body["errors"] == []


async def test_a_task_queue_job_that_publishes_nothing_reads_null(async_session, monkeypatch):
    from src.services import active_jobs, gpu_queue_state
    from src.workers import jlens_progress

    monkeypatch.setattr(gpu_queue_state, "read_gpu_messages", lambda **k: [])
    task_id = str(uuid.uuid4())
    _open(jlens_progress.FIT, "m_x", task_id)
    jlens_progress.update_row(task_id, status="running", progress=5.0)
    body = await active_jobs.list_active_jobs(async_session, kind="task_queue", cards=[])
    [job] = [j for j in body["jobs"] if j.get("task_id") == task_id]
    assert job["progress_detail"] is None and job["progress"] == 5.0


async def test_the_monitor_listing_names_the_probe_and_carries_the_reading(client, sync_db):
    world = seed_probe(sync_db)
    task_id = _running_evaluation(world.probe_id)
    r = await client.get("/api/v1/task-queue/active")
    assert r.status_code == 200, r.text
    [row] = [t for t in r.json()["data"] if t["task_id"] == task_id]
    assert (row["task_type"], row["status"], row["progress"]) == ("probe_evaluation", "running", 62.5)
    info = row["entity_info"]
    assert info["name"] == f"Probe {world.probe_id} · layer 1 · mean"
    assert (info["progress_detail"]["set_index"], info["progress_detail"]["rows_done"],
            info["progress_detail"]["rows_total"]) == (1, 250, 400)
    # No server-written subtitle for this kind: the page formats the reading, so it is never said twice.
    assert "details" not in info
