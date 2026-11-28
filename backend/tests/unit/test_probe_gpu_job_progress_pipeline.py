"""A real probe evaluation and a real WSP held-out capture publish view k of n and rows x of y as they run (2026-11-27).

The operator asked of a running evaluation "what is this job? where can I see its progress in a UI?" and the answer
was nowhere: `evaluate_probe_monitor` and `capture_wsp_heldout` reported nothing until each view finished. Both now
narrate through `live_progress.ViewRowsProgress`. Here they run for REAL on the tiny model of
`tests/support/wsp_pipeline.py` (its own database, module scope), with the batch planner forced to several batches
per view so a reading inside a view exists to be seen, and every reading the meter would write is recorded.

What is pinned, against what was actually scored rather than against the meter's own arithmetic:
* the plan is written before the model loads, and the card(s) the model was placed on are reported;
* each view begins (`preparing`, set k of n), then counts its rows batch by batch, monotone, to the exact rendered
  count, which is the length of the arrays the evaluation stored;
* the last reading is `done` with rows_done == rows_total == the rows really scored, and nothing estimated.

Controls (0xcc/reviews/probe_gpu_job_progress_controls_2026-11-27.md): drop the `rows` report in
`forward_scores_many` or `capture_tokens`, drop `progress=progress` from `evaluate_probe`'s scoring call or from
`run_captures`' `capture_one` call, or drop the `segment` call in `run_captures` -> red here.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("datasets")

from tests.unit.test_probe_monitor_pipeline import FAKE_GPU_UUID  # noqa: E402
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
    """The fixture run predates precision recording; these refusals are not under test (patched where they live)."""
    monkeypatch.setattr("src.services.probe_monitor_run.require_matching_precision", lambda *a, **k: None)
    monkeypatch.setattr("src.services.probe_monitor_run.require_served_render", lambda *a, **k: None)


@pytest.fixture
def several_batches(monkeypatch):
    """A tiny token budget, so every view is several batches and a reading INSIDE a view exists. Patched at the call
    (a default argument is bound at import; see `test_probe_monitor_pipeline.completed_run`)."""
    from src.services import probe_monitor_capture as capture_module

    real = capture_module.plan_batches
    monkeypatch.setattr(capture_module, "plan_batches",
                        lambda examples, *, token_budget=None: real(examples, token_budget=8))


def _meter(kind):
    from src.services.live_progress import ViewRowsProgress

    writes, placed = [], []
    meter = ViewRowsProgress(kind, lambda progress, detail: writes.append((progress, detail)),
                             on_placed=lambda uuid, uuids: placed.append((uuid, uuids)), every_seconds=0.0)
    return meter, writes, placed


def _check_sequence(writes, expected_sets):
    """The readings of a whole job: [(name, rows)] per view in order, rows exactly as scored."""
    details = [d for _p, d in writes]
    assert details[0]["phase"] == "loading_model" and details[0]["set_index"] is None, details[0]
    assert details[0]["set_count"] == len(expected_sets)
    for index, (name, rows) in enumerate(expected_sets):
        mine = [d for d in details if d["set_index"] == index and d["phase"] in ("preparing", "scoring")]
        assert mine and mine[0]["phase"] == "preparing" and mine[0]["set_rows_done"] == 0, mine[:1]
        assert all(d["set_name"] == name for d in mine), {d["set_name"] for d in mine}
        scoring = [d for d in mine if d["phase"] == "scoring"]
        assert len(scoring) >= 2, f"view {index}: one reading per view cannot show progress inside it ({scoring})"
        counts = [d["set_rows_done"] for d in scoring]
        assert counts == sorted(counts) and counts[-1] == rows, (index, counts, rows)
        assert counts[0] < rows, f"view {index}: the first reading must come before the view ends ({counts})"
        assert all(d["set_rows_total"] == rows for d in scoring), (index, rows)
    overall = [d["rows_done"] for d in details]
    assert overall == sorted(overall), "rows done never goes backwards"
    percents = [p for p, d in writes if d["phase"] == "scoring"]
    assert all(0.0 <= p <= 100.0 for p in percents)
    last = details[-1]
    total = sum(rows for _name, rows in expected_sets)
    assert (last["phase"], last["rows_done"], last["rows_total"], last["rows_estimated"]) == ("done", total, total,
                                                                                            False), last
    assert writes[-1][0] == 100.0


def test_a_real_evaluation_publishes_each_view_and_its_rows(pipeline_db, wsp_world, wsp_probe, tiny_lm,
                                                          tiny_tokenizer, monkeypatch, gates_pass, several_batches):
    from src.models.probe_monitor import ProbeMonitorEvaluation
    from src.services import probe_monitor_run as run_module

    monkeypatch.setattr(run_module, "_load_model_for_run", _loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
    views = [wsp_world["eval"], wsp_world["calibration"]]
    meter, writes, placed = _meter("probe_evaluation")
    before = {e.id for e in pipeline_db.query(ProbeMonitorEvaluation).all()}

    result = run_module.evaluate_probe(pipeline_db, wsp_probe.id, [v.id for v in views], sample_rows=100,
                                       progress=meter)

    assert result["evaluations"] == 2
    new = {e.dataset_id: e for e in pipeline_db.query(ProbeMonitorEvaluation).all() if e.id not in before}
    scored = [(v.name, int(len(np.load(new[v.id].scores_path)))) for v in views]
    assert all(rows > 0 for _n, rows in scored), scored
    _check_sequence(writes, scored)
    # The plan came first, as an estimate (these views store no counts, so the sample is the estimate).
    first = writes[0][1]
    assert (first["rows_total"], first["rows_estimated"]) == (200, True), first
    # 1+n GPUs: the card the loader placed the model on, as the loader wrote it to the run row.
    assert placed == [(FAKE_GPU_UUID, None)]


def test_a_real_capture_publishes_each_set_and_its_rows(pipeline_db, wsp_probe, tiny_lm, tiny_tokenizer, gates_pass,
                                                       several_batches):
    from src.api.v1.endpoints.wsp import plan_capture_rows
    from src.models.probe_monitor import ProbeMonitorDataset
    from src.models.wsp import WspActivationCapture
    from src.services import wsp_capture

    ids = plan_capture_rows(pipeline_db, wsp_probe.id, None)["to_run"]
    pipeline_db.commit()
    assert len(ids) == 2, ids
    rows = {r.id: r for r in pipeline_db.query(WspActivationCapture).filter_by(probe_id=wsp_probe.id).all()}
    planned = {(p.view_id, p.role): p for p in wsp_capture.plan_capture(pipeline_db, wsp_probe)}
    meter, writes, placed = _meter("wsp_heldout_capture")

    out = wsp_capture.run_captures(pipeline_db, wsp_probe.id, ids,
                                   model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer),
                                   progress=meter)

    assert sorted(out["captures"].values()) == ["ready", "ready"], out
    pending = [rows[i] for i in ids]
    names = {v.id: v.name for v in pipeline_db.query(ProbeMonitorDataset).all()}
    expected = []
    for row in pending:
        pipeline_db.refresh(row)
        label = names[row.dataset_id] + (" (calibration)" if row.role == "calibration" else "")
        # What was captured, from the row the capture wrote — and what the plan said, before anything ran.
        assert row.rows == len(planned[(row.dataset_id, row.role)].positions)
        expected.append((label, int(row.rows)))
    _check_sequence(writes, expected)
    first = writes[0][1]
    assert (first["rows_total"], first["rows_estimated"]) == (sum(r for _l, r in expected), False), first
    assert placed == [(FAKE_GPU_UUID, None)]
