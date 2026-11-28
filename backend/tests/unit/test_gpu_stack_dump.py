"""A wedged GPU worker's stack is captured, once, without stopping it (ENH-011 WB-2, #9, G12 "Defect B").

The wedge watch (`workers/gpu_worker_watch.py`) classifies a worker `wedged` and every worker registers faulthandler on
SIGUSR2 (`core/celery_app.install_stack_dump_signal`), but nothing ever sent the signal, so no stack was captured. Now:

* the GPU job wrapper's completion stamp names the worker that ran the task (Celery's `request.hostname`);
* on a `wedged` reading the watch writes ONE Redis mark per stuck task naming that worker (`<spec name>`, the part of the
  hostname before `@`, which is how the supervisor's `WorkerSpec` names it), with a TTL;
* the supervisor (PID 1 of `celery-gpu-workers`, which knows each child's pid) reads the marks of its OWN running
  children on every poll, consumes each, sends `SIGUSR2` once to the named child and logs
  `stack_dump_requested worker=<name> pid=<pid>`. It never restarts, kills or stops anything.

Controls (0xcc/reviews/ENH-011_WB_controls_2026-11-26.md): C1 the supervisor's mark check removed → (a) red; C2 the mark
not consumed → (b) red; C3 the watch's mark write removed → (e) red (the wiring control); C4 SIGTERM for SIGUSR2 → (a).
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
import uuid

import pytest

from src.workers import gpu_supervisor as S
from tests.unit.test_gpu_supervisor import RTX, TI, Clock, Launcher, forever, restore_signals, run  # noqa: F401

RTX_NAME = f"gpu-{RTX.index}-{RTX.uuid.removeprefix('GPU-')[:8]}"
TI_NAME = f"gpu-{TI.index}-{TI.uuid.removeprefix('GPU-')[:8]}"


class FakeRedis:
    """The calls the mark uses: `set` (with `nx`, `ex`), `get`, `getdel`, `delete`, `ttl`."""

    def __init__(self):
        self.data: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.data:
            return None
        self.data[key] = value.encode() if isinstance(value, str) else value
        if ex is not None:
            self.ttls[key] = int(ex)
        return True

    def get(self, key):
        return self.data.get(key)

    def getdel(self, key):
        self.ttls.pop(key, None)
        return self.data.pop(key, None)

    def delete(self, *keys):
        for key in keys:
            self.data.pop(key, None)
            self.ttls.pop(key, None)

    def ttl(self, key):
        return self.ttls.get(key, -1)


@pytest.fixture
def kills(monkeypatch):
    """`os.kill`, recorded. The loop's own stop (the test's SIGTERM to this process) still goes through."""
    real = os.kill
    sent = []

    def kill(pid, signum):
        if pid == os.getpid():
            return real(pid, signum)
        sent.append((pid, signum))

    monkeypatch.setattr(os, "kill", kill)
    return sent


def _supervise_with_marks(clock, redis, marks_at, specs, *, stop=6.0):
    """Run the real supervisor with the relay wired as `main()` wires it; `marks_at` = {time: [worker names]}."""
    from src.workers import gpu_stack_dump as D

    launcher = Launcher(clock, {spec.card: forever(clock) for spec in specs})
    placed = set()

    def tick():
        for when, names in marks_at.items():
            if clock() >= when and when not in placed:
                placed.add(when)
                for name in names:
                    D.request_stack_dump(redis, name, task_id=f"t-{name}-{when}", now=time.time())
        if clock() >= stop:
            os.kill(os.getpid(), signal.SIGTERM)

    run(launcher, clock, specs, on_tick=tick, on_stack_dump=D.StackDumpRelay(lambda: redis))
    return launcher


def _children(launcher):
    return {card: workers[0].pid for card, workers in launcher.workers.items()}


def test_a_mark_sends_exactly_one_sigusr2_to_the_named_child(restore_signals, kills, caplog):  # noqa: F811
    """(a) One `os.kill(pid, SIGUSR2)`, to the named child's pid, and the log line; nothing restarted."""
    clock, redis = Clock(), FakeRedis()
    specs = S.worker_specs([TI, RTX], python="py")
    with caplog.at_level(logging.INFO, logger="src.workers.gpu_stack_dump"):
        launcher = _supervise_with_marks(clock, redis, {2.0: [RTX_NAME]}, specs)
    pid = _children(launcher)[RTX.uuid]
    assert kills == [(pid, signal.SIGUSR2)], kills
    assert f"stack_dump_requested worker={RTX_NAME} pid={pid}" in caplog.text
    assert launcher.starts(RTX.uuid) == [0.0] and launcher.starts(TI.uuid) == [0.0], "nothing may restart"
    assert all(w.signals[:1] == [(signal.SIGTERM, w.signals[0][1])] for ws in launcher.workers.values() for w in ws), (
        "the only signal a child got from the supervisor itself is the stop's SIGTERM")


def test_a_consumed_mark_sends_no_second_signal(restore_signals, kills):  # noqa: F811
    """(b) The mark is consumed: the polls after it send nothing more."""
    clock, redis = Clock(), FakeRedis()
    from src.workers import gpu_stack_dump as D

    _supervise_with_marks(clock, redis, {1.0: [RTX_NAME]}, S.worker_specs([RTX], python="py"), stop=8.0)
    assert len(kills) == 1, kills
    assert redis.get(D.MARK_PREFIX + RTX_NAME) is None, "the mark must be consumed"


def test_no_mark_no_signal(restore_signals, kills):  # noqa: F811
    """(c)"""
    clock, redis = Clock(), FakeRedis()
    _supervise_with_marks(clock, redis, {}, S.worker_specs([TI, RTX], python="py"))
    assert kills == []


def test_a_mark_for_a_worker_that_is_not_its_child_sends_nothing(restore_signals, kills):  # noqa: F811
    """(d) Another pod's worker (or a renamed one): no signal, and the mark is left to its TTL, not consumed here."""
    from src.workers import gpu_stack_dump as D

    clock, redis = Clock(), FakeRedis()
    _supervise_with_marks(clock, redis, {1.0: ["gpu-7-deadbeef"]}, S.worker_specs([TI, RTX], python="py"))
    assert kills == []
    assert redis.get(D.MARK_PREFIX + "gpu-7-deadbeef") is not None
    assert 0 < redis.ttl(D.MARK_PREFIX + "gpu-7-deadbeef") <= D.MARK_TTL_S


def test_main_wires_the_relay_to_the_broker(monkeypatch):
    """The production entry point passes the relay, reading the broker's Redis."""
    from src.workers import gpu_stack_dump as D

    seen = {}
    monkeypatch.setattr("src.services.gpu_placement.list_cards", lambda: [RTX])
    monkeypatch.setattr("src.services.gpu_worker_queues.maintenance_tick", lambda live: (lambda: None))
    monkeypatch.setattr(S.CardAvailability, "reset", lambda self, cards: None)
    monkeypatch.setattr(S, "supervise", lambda specs, **kwargs: seen.update(kwargs) or 0)
    assert S.main() == 0
    relay = seen["on_stack_dump"]
    assert isinstance(relay, D.StackDumpRelay) and relay.client_factory is D.broker_client


# ── (e) the watch writes the mark, once, on `wedged` only ─────────────────────────────────────────────────────────

from tests.support.real_broker import real_broker  # noqa: E402,F401 - a fixture
from tests.unit.test_gpu_worker_watch import EVALUATE, RUN, broker, result_success  # noqa: E402,F401


def _stamp(client, task_id, seconds_ago, hostname):
    from datetime import datetime, timedelta, timezone

    from src.workers import task_completion

    when = (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()
    client.set(task_completion.DONE_PREFIX + task_id,
               json.dumps({"completed_at": when, "task": RUN, "worker": hostname}))


def _marks(client):
    from src.workers import gpu_stack_dump as D

    return {k.decode(): json.loads(client.get(k)) for k in client.scan_iter(match=D.MARK_PREFIX + "*")}


@pytest.fixture
def no_rows(monkeypatch):
    """The watch's task_queue bookkeeping is not what this test reads."""
    import contextlib
    from unittest.mock import MagicMock

    from src.workers import gpu_worker_watch as watch

    monkeypatch.setattr(watch, "get_sync_db", lambda: contextlib.nullcontext(MagicMock()))
    monkeypatch.setattr("src.services.gpu_queue_state._cards_held", lambda: 0)


def test_the_watch_marks_the_worker_once_on_wedged_and_never_otherwise(broker, no_rows):  # noqa: F811
    """(e) A real stuck message whose completion stamp names `gpu-1-247aa582@pod`: `suspect` writes nothing; `wedged`
    writes one mark naming `gpu-1-247aa582`, with the task id and a TTL; a second `wedged` pass writes no second."""
    from src.services import gpu_queue_state as state
    from src.workers import gpu_stack_dump as D
    from src.workers import gpu_worker_watch as watch

    finished = broker.publish(RUN, run_id="pmr_1")
    broker.deliver()
    _stamp(broker.client, finished, 600, f"{RTX_NAME}@miworkbench-backend-0")
    broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])

    assert watch.watch_gpu_workers(client=broker.client)["state"] == "suspect"
    assert _marks(broker.client) == {}, "suspect must write no mark"

    broker.client.set(state.FIRST_SEEN_KEY, str(time.time() - state.WEDGE_CONFIRM_S - 1))
    assert watch.watch_gpu_workers(client=broker.client)["state"] == "wedged"
    marks = _marks(broker.client)
    assert list(marks) == [D.MARK_PREFIX + RTX_NAME], marks
    assert marks[D.MARK_PREFIX + RTX_NAME]["task_id"] == finished
    assert 0 < broker.client.ttl(D.MARK_PREFIX + RTX_NAME) <= D.MARK_TTL_S

    broker.client.delete(D.MARK_PREFIX + RTX_NAME)  # the supervisor consumed it
    assert watch.watch_gpu_workers(client=broker.client)["state"] == "wedged"
    assert _marks(broker.client) == {}, "one mark per stuck task, however many passes"


def test_an_ok_or_unknown_pass_writes_no_mark(broker, no_rows):  # noqa: F811
    from src.workers import gpu_worker_watch as watch

    task_id = broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
    broker.deliver()
    assert watch.watch_gpu_workers(client=broker.client)["state"] == "ok"
    assert _marks(broker.client) == {}

    class Down:
        def __getattr__(self, name):
            raise ConnectionError("broker down")

    assert watch.watch_gpu_workers(client=Down())["state"] == "unknown"
    assert task_id and _marks(broker.client) == {}


def test_a_stuck_task_whose_worker_is_not_named_is_reported_not_guessed(broker, no_rows, caplog):  # noqa: F811
    """A stamp from before this change has no worker: no mark (never a guess at which child), and the log says why."""
    from src.services import gpu_queue_state as state
    from src.workers import gpu_worker_watch as watch

    finished = broker.publish(RUN, run_id="pmr_1")
    broker.deliver()
    result_success(broker.client, finished, 600)
    broker.publish(EVALUATE, probe_id="pm_1", dataset_ids=["d"])
    broker.client.set(state.FIRST_SEEN_KEY, str(time.time() - state.WEDGE_CONFIRM_S - 1))
    with caplog.at_level(logging.WARNING, logger="src.workers.gpu_worker_watch"):
        assert watch.watch_gpu_workers(client=broker.client)["state"] == "wedged"
    assert _marks(broker.client) == {}
    assert "cannot name the worker" in caplog.text and finished in caplog.text


# ── the stamp names the worker that ran the task (the watch's only source of the name) ──────────────────────────

def test_the_completion_stamp_names_the_worker(real_broker, monkeypatch, async_session):  # noqa: F811
    from unittest.mock import MagicMock

    from src.core.celery_app import celery_app
    from src.services import probe_monitor_run as run_module
    from src.workers import probe_monitor_tasks as tasks
    from src.workers import task_completion

    celery_app.loader.import_default_modules()
    client, _url = real_broker
    monkeypatch.setattr(task_completion, "_client", lambda: client)
    monkeypatch.setattr("src.workers.gpu_job.per_card_workers", lambda: False)
    monkeypatch.setattr(run_module, "evaluate_probe", MagicMock(return_value={"evaluated": 1}))
    monkeypatch.setattr(tasks, "_require_row", lambda *a, **k: MagicMock())
    task_id = str(uuid.uuid4())
    task = tasks.evaluate_probe_monitor
    task.push_request(id=task_id, hostname=f"{RTX_NAME}@miworkbench-backend-0", delivery_info={"redelivered": False})
    try:
        task.run(probe_id="pm", dataset_ids=["d"])
    finally:
        task.pop_request()
    stamp = json.loads(client.get(task_completion.DONE_PREFIX + task_id))
    assert stamp["worker"] == f"{RTX_NAME}@miworkbench-backend-0"
    assert task_completion.completed_by(task_id, client=client) == f"{RTX_NAME}@miworkbench-backend-0"
