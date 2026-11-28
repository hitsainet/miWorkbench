"""R4 review fix (C044a-L1), 2026-11-17: the capture task CLAIMS each row, as the bundle task does.

Round 3 (OR3-044a-L2) filtered `pending` once, at the task's start, and `capture_one` then wrote `running` and `ready`
unconditionally. A multi-set task keeps its later sets `pending` for as long as the earlier ones take on the GPU, so the
janitor could reap them `never_started` and the task captured them anyway: `ready` with the janitor's error (the
review's probe P2). Now `capture_one` claims the row only while it is `pending` (`claim_capture`), each heartbeat only
while it is `running`, and reads its status under `FOR UPDATE` before the final write; a row reaped meanwhile is left
exactly as the janitor wrote it and reported `skipped:<status>`.

One real tiny run, module scope (its own database); no capture exists before the first test.

Controls (`0xcc/reviews/044a_r4_review_fixes_2026-11-17.md`): an unconditional `running` at the start again →
`test_a_row_reaped_while_it_waits_behind_the_tasks_earlier_set_is_not_captured` red; no locked read before the final
write → `test_a_row_reaped_after_its_last_heartbeat_never_turns_ready` red; a heartbeat that does not check the row →
`test_a_row_reaped_mid_capture_stops_at_its_next_heartbeat` red; a failure written whatever the row's status →
`test_a_failure_never_overwrites_the_janitors_reason` red.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

pytest.importorskip("datasets")

from tests.support.wsp_pipeline import (  # noqa: E402,F401 (fixtures)
    _loader_that_records_its_placement,
    arrow_dataset,
    heldout_arrow,
    monkeypatch_module,
    pipeline_db,
    tiny_lm,
    tiny_tokenizer,
    wsp_data_dir,
    wsp_probe,
    wsp_world,
)


@pytest.fixture
def gates_pass(monkeypatch):
    """The fixture run predates precision recording; these two refusals are not under test (patched where they
    live, as `run_captures` imports them per call)."""
    monkeypatch.setattr("src.services.probe_monitor_run.require_matching_precision", lambda *a, **k: None)
    monkeypatch.setattr("src.services.probe_monitor_run.require_served_render", lambda *a, **k: None)


def _janitor(db, *, hours):
    """The real janitor, from its OWN session (another process in production), `hours` from now."""
    from sqlalchemy.orm import sessionmaker

    from src.core.clock import utc_now
    from src.workers.cleanup_stuck_wsp_rows import sweep

    with sessionmaker(bind=db.get_bind())() as session:
        return sweep(session, now=utc_now() + timedelta(hours=hours))


def _rows(db, ids):
    from src.models.wsp import WspActivationCapture

    db.expire_all()
    return {i: db.get(WspActivationCapture, i) for i in ids}


def test_a_row_reaped_while_it_waits_behind_the_tasks_earlier_set_is_not_captured(pipeline_db, wsp_probe, tiny_lm,
                                                                                  tiny_tokenizer, monkeypatch,
                                                                                  gates_pass):
    """The review's probe P2, made a regression. A two-set task: while its first set is captured, the janitor runs a
    day later and reaps the second set, still `pending`, as `never_started`. The task then reaches that row and
    leaves it exactly as the janitor wrote it (`skipped:failed`, no files, no parity); the first set is captured."""
    from src.api.v1.endpoints.wsp import plan_capture_rows
    from src.services import wsp_capture
    from src.workers.cleanup_stuck_wsp_rows import WORDS

    ids = plan_capture_rows(pipeline_db, wsp_probe.id, None)["to_run"]
    pipeline_db.commit()
    assert len(ids) == 2, ids
    real, calls, reaped = wsp_capture.capture_one, [], []

    def first_then_janitor(db, capture, planned, **kwargs):
        calls.append(capture.id)
        out = real(db, capture, planned, **kwargs)
        if len(calls) == 1:   # hours on a real GPU; the other row is still `pending` the whole time
            reaped.append(_janitor(db, hours=25))
        return out

    monkeypatch.setattr(wsp_capture, "capture_one", first_then_janitor)
    result = wsp_capture.run_captures(pipeline_db, wsp_probe.id, ids,
                                      model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
    assert len(calls) == 2 and reaped == [{"captures": 1, "bundles": 0}], (calls, reaped)
    first, second = calls
    rows = _rows(pipeline_db, ids)
    assert result["captures"] == {first: rows[first].status, second: "skipped:failed"}, result
    assert rows[first].status in ("ready", "parity_failed") and rows[first].parity is not None
    assert (rows[second].status, rows[second].error, rows[second].parity, rows[second].dir_path,
            rows[second].rows) == ("failed", WORDS["never_started"], None, None, None)


def test_a_row_reaped_after_its_last_heartbeat_never_turns_ready(pipeline_db, wsp_probe, tiny_lm, tiny_tokenizer,
                                                                 monkeypatch, gates_pass):
    """Reaped `worker_lost` after its activations are written and its last heartbeat beaten, while parity is checked:
    the status read under `FOR UPDATE` before the final write finds the janitor's `failed` and the row keeps it."""
    from src.api.v1.endpoints.wsp import plan_capture_rows
    from src.services import wsp_capture
    from src.workers.cleanup_stuck_wsp_rows import WORDS

    ids = plan_capture_rows(pipeline_db, wsp_probe.id, None)["to_run"]   # the row the first test's janitor failed
    pipeline_db.commit()
    assert len(ids) == 1, ids
    real, reaped = wsp_capture.check_parity, []

    def janitor_meanwhile(*args, **kwargs):
        reaped.append(_janitor(pipeline_db, hours=2))   # past CAPTURE_STALE (90 min) since the last heartbeat
        return real(*args, **kwargs)

    monkeypatch.setattr(wsp_capture, "check_parity", janitor_meanwhile)
    result = wsp_capture.run_captures(pipeline_db, wsp_probe.id, ids,
                                      model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
    row = _rows(pipeline_db, ids)[ids[0]]
    assert reaped == [{"captures": 1, "bundles": 0}] and result["captures"] == {row.id: "skipped:failed"}, result
    assert (row.status, row.error, row.parity) == ("failed", WORDS["worker_lost"], None)


def test_a_failure_never_overwrites_the_janitors_reason(pipeline_db, wsp_probe, monkeypatch, gates_pass):
    """A task that fails at its model load tells each of its rows why, but only rows still its own: a row the janitor
    reaped meanwhile keeps the janitor's reason."""
    from src.api.v1.endpoints.wsp import plan_capture_rows
    from src.services import wsp_capture
    from src.workers.cleanup_stuck_wsp_rows import WORDS

    ids = plan_capture_rows(pipeline_db, wsp_probe.id, None)["to_run"]   # re-armed again
    pipeline_db.commit()
    assert len(ids) == 1, ids

    def loader(_run):
        _janitor(pipeline_db, hours=25)
        raise RuntimeError("stopped by the test at the model load")

    with pytest.raises(RuntimeError):
        wsp_capture.run_captures(pipeline_db, wsp_probe.id, ids, model_loader=loader)
    row = _rows(pipeline_db, ids)[ids[0]]
    assert (row.status, row.error) == ("failed", WORDS["never_started"])


def test_a_row_reaped_mid_capture_stops_at_its_next_heartbeat(pipeline_db, wsp_probe, tiny_lm, tiny_tokenizer,
                                                              monkeypatch, gates_pass):
    """Reaped `worker_lost` while its forward pass runs: the next heartbeat (between batches) is refused, so the
    capture stops there, before any parity work, and the row keeps the janitor's reason. Control: a heartbeat that
    does not check the row → parity is checked (only the final locked read stops `ready`) → red."""
    from src.api.v1.endpoints.wsp import plan_capture_rows
    from src.services import wsp_capture
    from src.workers.cleanup_stuck_wsp_rows import WORDS

    ids = plan_capture_rows(pipeline_db, wsp_probe.id, None)["to_run"]   # re-armed again
    pipeline_db.commit()
    assert len(ids) == 1, ids
    real_write, real_parity, reaped, parity_calls = wsp_capture.write_capture, wsp_capture.check_parity, [], []

    def reaped_at_the_first_batch(*args, progress=None, **kwargs):
        def beat(done, total):
            if not reaped:
                reaped.append(_janitor(pipeline_db, hours=2))   # past CAPTURE_STALE since the claim
            return progress(done, total)
        return real_write(*args, progress=beat, **kwargs)

    def parity(*args, **kwargs):
        parity_calls.append(1)
        return real_parity(*args, **kwargs)

    monkeypatch.setattr(wsp_capture, "write_capture", reaped_at_the_first_batch)
    monkeypatch.setattr(wsp_capture, "check_parity", parity)
    result = wsp_capture.run_captures(pipeline_db, wsp_probe.id, ids,
                                      model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
    row = _rows(pipeline_db, ids)[ids[0]]
    assert reaped == [{"captures": 1, "bundles": 0}] and parity_calls == [], (reaped, parity_calls)
    assert result["captures"] == {row.id: "skipped:failed"}, result
    assert (row.status, row.error, row.parity) == ("failed", WORDS["worker_lost"], None)
