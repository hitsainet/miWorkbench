"""ENH-010 044 tasks 4.1, 4.2: held-out capture and its parity, on the REAL pipeline (FTID I44-D1: the tiny real
model of `test_probe_monitor_pipeline.py`, never a stub — a stub forward would agree by construction).

One real `execute_probe_run` with an eval view (scored on a seeded sample, so the capture must re-draw the same rows)
and a calibration view (so the probe's bar is cut on a calibration set and a calibration capture exists). Then the
capture is run for real and compared with what the run stored.

Controls (0xcc/reviews/044_implementation_controls_2026-11-16.md): skip `check_parity` → the tampered-set tests red;
compare the eval AUROC over all rows instead of under the stored mask → `test_parity_compares_under_the_same_mask` red;
skip the calibration statistic → `test_calibration_capture_parity` red; a control that cannot fail → the controls
tests red.
"""
from __future__ import annotations

import dataclasses
import uuid

import numpy as np
import pytest

pytest.importorskip("datasets")

from tests.unit.test_probe_monitor_pipeline import (  # noqa: E402,F401 (fixtures)
    _loader_that_records_its_placement,
    arrow_dataset,
    monkeypatch_module,
    pipeline_db,
    tiny_lm,
    tiny_tokenizer,
)

EVAL_SAMPLE = 60


@pytest.fixture(scope="module")
def wsp_prepared(pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun

    model_row = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny-random", status=ModelStatus.READY,
                      file_path=str(arrow_dataset), architecture="LlamaForCausalLM", params_count=1_000)
    dataset_row = Dataset(id=uuid.uuid4(), name="probe-corpus", source="local", status=DatasetStatus.READY,
                          raw_path=str(arrow_dataset), extra_metadata={})
    pipeline_db.add_all([model_row, dataset_row])
    pipeline_db.commit()
    mapping = {"high": "positive", "low": "negative"}
    train = ProbeMonitorDataset(name="train", dataset_id=dataset_row.id, input_column="inputs",
                                label_column="stakes", label_mapping=mapping, pair_column="pair", role="train",
                                counts={})
    heldout = ProbeMonitorDataset(name="unseen", dataset_id=dataset_row.id, input_column="inputs",
                                  label_column="stakes", label_mapping=mapping, role="eval",
                                  distribution="out_of_distribution", counts={})
    calibration = ProbeMonitorDataset(name="calm", dataset_id=dataset_row.id, input_column="inputs",
                                      label_column="stakes", label_mapping={"low": "negative"}, role="calibration",
                                      counts={})
    pipeline_db.add_all([train, heldout, calibration])
    pipeline_db.commit()
    run = ProbeMonitorRun(
        model_id=model_row.id, train_dataset_id=train.id, eval_dataset_ids=[heldout.id],
        calibration_dataset_id=calibration.id,
        config={"layers": [1, 2], "rules": ["mean"], "scope": "all", "max_length": 64, "val_fraction": 0.25,
                "seed": 1337, "top_n_layers": 1, "target_fpr": 0.1, "eval_max_rows": EVAL_SAMPLE},
        status="pending", environment={})
    pipeline_db.add(run)
    pipeline_db.commit()
    return {"run": run, "eval": heldout, "calibration": calibration}


@pytest.fixture(scope="module")
def wsp_run(pipeline_db, wsp_prepared, tiny_lm, tiny_tokenizer, tmp_path_factory, monkeypatch_module):
    from src.core import config as config_module
    from src.models.probe_monitor import ProbeMonitor
    from src.services import probe_monitor_run as run_module

    monkeypatch_module.setattr(config_module.settings, "data_dir", tmp_path_factory.mktemp("wsp_artifacts"),
                               raising=False)
    run_module.execute_probe_run(pipeline_db, wsp_prepared["run"].id,
                                 model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
    probe = (pipeline_db.query(ProbeMonitor).filter(ProbeMonitor.run_id == wsp_prepared["run"].id,
                                                    ProbeMonitor.weights_path.isnot(None)).first())
    assert probe is not None and probe.threshold_source == "calibration_set", "the fixture must calibrate on a set"
    return probe


def _plan(db, probe, role):
    from src.services.wsp_capture import plan_capture

    return next(p for p in plan_capture(db, probe) if p.role == role)


def _new_capture(db, probe, planned):
    from src.models.wsp import WspActivationCapture

    row = WspActivationCapture(probe_id=probe.id, dataset_id=planned.view_id, role=planned.role,
                               source_evaluation_id=planned.evaluation_id, layer=probe.layer, scope="all")
    db.add(row)
    db.commit()
    return row


@pytest.fixture(scope="module")
def captured(pipeline_db, wsp_run, tiny_lm, tiny_tokenizer):
    """Both sets captured once, for real."""
    from src.services.wsp_capture import capture_one

    out = {}
    for role in ("eval", "calibration"):
        planned = _plan(pipeline_db, wsp_run, role)
        row = capture_one(pipeline_db, _new_capture(pipeline_db, wsp_run, planned), planned, model=tiny_lm,
                          tokenizer=tiny_tokenizer, architecture="LlamaForCausalLM")
        out[role] = (row, planned)
    return out


def _reload(row, planned):
    """The captured files of `row`, as `check_parity` reads them."""
    from pathlib import Path

    from src.services.wsp_capture import CapturedSet

    directory = Path(row.dir_path)
    tokens = next(directory.glob("tokens.f*"))
    return CapturedSet(tokens, np.load(directory / "offsets.npy"), row.d_model, row.dtype,
                       np.load(directory / "truncated.npy"))


# ── 4.1 planning ──────────────────────────────────────────────────────────────


def test_rows_are_the_evaluations_rows(pipeline_db, wsp_run, wsp_prepared, tiny_tokenizer):
    from src.models.probe_monitor import ProbeMonitorEvaluation, ProbeMonitorRun
    from src.services.probe_monitor_run import context_from_row
    from src.services.wsp_capture import render_planned

    planned = _plan(pipeline_db, wsp_run, "eval")
    evaluation = pipeline_db.get(ProbeMonitorEvaluation, planned.evaluation_id)
    assert planned.view_id == wsp_prepared["eval"].id
    assert np.array_equal(planned.positions, np.load(evaluation.positions_path))
    assert planned.sample_rows == EVAL_SAMPLE and len(planned.positions) == EVAL_SAMPLE
    context = context_from_row(pipeline_db.get(ProbeMonitorRun, wsp_run.run_id))
    rendered = render_planned(pipeline_db, context, tiny_tokenizer, planned)
    assert len(rendered) == len(planned.positions)
    # The SEEDED SAMPLE is re-drawn: a whole-view render keeps the same rows only by matching positions.
    unsampled = dataclasses.replace(planned, sample_rows=None)
    assert len(render_planned(pipeline_db, context, tiny_tokenizer, unsampled)) == len(planned.positions)


def test_positions_not_recorded_refused(pipeline_db, wsp_run):
    from src.models.probe_monitor import ProbeMonitorEvaluation
    from src.services.wsp_capture import CaptureRefused, plan_capture

    evaluation = pipeline_db.get(ProbeMonitorEvaluation, _plan(pipeline_db, wsp_run, "eval").evaluation_id)
    saved, evaluation.positions_path = evaluation.positions_path, None
    pipeline_db.commit()
    try:
        with pytest.raises(CaptureRefused) as refused:
            plan_capture(pipeline_db, wsp_run)
        assert refused.value.code == "evaluation_positions_not_recorded"
    finally:
        evaluation.positions_path = saved
        pipeline_db.commit()


def test_calibration_positions_not_recorded_refused(pipeline_db, wsp_run):
    from src.services.wsp_capture import CaptureRefused, plan_capture

    saved, wsp_run.calibration_positions_path = wsp_run.calibration_positions_path, None
    pipeline_db.commit()
    try:
        with pytest.raises(CaptureRefused) as refused:
            plan_capture(pipeline_db, wsp_run)
        assert refused.value.code == "calibration_positions_not_recorded"
    finally:
        wsp_run.calibration_positions_path = saved
        pipeline_db.commit()


def test_a_view_never_evaluated_is_refused_not_skipped(pipeline_db, wsp_run):
    from src.services.wsp_capture import CaptureRefused, plan_capture

    with pytest.raises(CaptureRefused) as refused:
        plan_capture(pipeline_db, wsp_run, dataset_ids=["pmd_000000000000"])
    assert refused.value.code == "evaluation_positions_not_recorded"


# ── 4.2 capture and parity ────────────────────────────────────────────────────


def test_parity_passes_on_the_stored_evaluation(captured):
    row, planned = captured["eval"]
    assert row.status == "ready", row.parity
    parity = row.parity
    assert parity["passed"] and parity["controls"] == {"permuted_w": True, "rolled_rows": True}
    assert abs(parity["auroc_stored"] - parity["auroc_capture"]) <= 1e-4
    assert parity["count_stored"] == parity["count_capture"] and parity["rows"] == len(planned.positions)
    assert row.rows == len(planned.positions) and row.tokens > 0 and row.bytes > 0 and row.row_ids_sha256


def test_the_files_are_the_five_arrays_under_the_capture_dir(captured):
    from pathlib import Path

    row, planned = captured["eval"]
    directory = Path(row.dir_path)
    assert directory.parts[-3:-1] == ("wsp", "captures") and directory.name == row.id
    assert {p.name for p in directory.iterdir()} == {"tokens.f32", "offsets.npy", "positions.npy", "labels.npy",
                                                     "n_scored.npy", "truncated.npy"}
    assert np.array_equal(np.load(directory / "positions.npy"), planned.positions)


def test_permuted_weight_fails_parity(pipeline_db, captured, wsp_run):
    from src.services.probe_monitor_run import load_probe
    from src.services.wsp_capture import check_parity

    row, planned = captured["eval"]
    head, trained = load_probe(pipeline_db, wsp_run.id)
    permuted = dataclasses.replace(head, weight=head.weight[np.random.default_rng(1).permutation(
        int(head.weight.shape[0]))])
    result = check_parity(permuted, trained, _reload(row, planned), planned, wsp_run)
    assert result["passed"] is False


def test_rolled_rows_fail_parity(pipeline_db, captured, wsp_run):
    from src.services.probe_monitor_run import load_probe
    from src.services.wsp_capture import check_parity

    row, planned = captured["eval"]
    head, trained = load_probe(pipeline_db, wsp_run.id)
    rolled = dataclasses.replace(planned, scores=np.roll(planned.scores, 1), labels=np.roll(planned.labels, 1))
    assert check_parity(head, trained, _reload(row, planned), rolled, wsp_run)["passed"] is False


def test_parity_compares_under_the_same_mask(pipeline_db, captured, wsp_run):
    """Rows the stored evaluation marks truncated are not compared: their stored scores are scrambled here and parity
    still passes, with `auroc_stored` the AUROC recomputed over the untruncated rows only."""
    from src.services.probe_monitor_metrics import auroc
    from src.services.probe_monitor_run import load_probe
    from src.services.wsp_capture import check_parity

    row, planned = captured["eval"]
    head, trained = load_probe(pipeline_db, wsp_run.id)
    mask = np.zeros(len(planned.positions), dtype=np.int8)
    mask[: len(mask) // 3] = 1
    scrambled = planned.scores.copy()
    scrambled[mask == 1] = -scrambled[mask == 1] + 7.0
    tampered = dataclasses.replace(planned, scores=scrambled, truncated=mask)
    files = dataclasses.replace(_reload(row, planned), truncated=mask)
    result = check_parity(head, trained, files, tampered, wsp_run)
    keep = mask == 0
    assert result["passed"] and result["mask_source"] == "stored" and result["rows_counted"] == int(keep.sum())
    assert result["auroc_stored"] == auroc(planned.scores[keep].tolist(), planned.labels[keep].tolist())


def test_a_truncation_disagreement_fails(pipeline_db, captured, wsp_run):
    from src.services.probe_monitor_run import load_probe
    from src.services.wsp_capture import check_parity

    row, planned = captured["eval"]
    head, trained = load_probe(pipeline_db, wsp_run.id)
    mask = np.zeros(len(planned.positions), dtype=np.int8)
    mask[0] = 1
    result = check_parity(head, trained, _reload(row, planned), dataclasses.replace(planned, truncated=mask), wsp_run)
    assert (result["passed"], result["reason"]) == (False, "truncation_differs")


def test_calibration_capture_parity(pipeline_db, captured, wsp_run):
    from src.services.probe_monitor_run import load_probe
    from src.services.wsp_capture import check_parity

    row, planned = captured["calibration"]
    assert row.status == "ready" and row.role == "calibration" and row.source_evaluation_id is None, row.parity
    assert row.parity["threshold_row_holds_the_bar"] and all(row.parity["controls"].values())
    head, trained = load_probe(pipeline_db, wsp_run.id)
    rolled = dataclasses.replace(planned, scores=np.roll(planned.scores, 1))
    assert check_parity(head, trained, _reload(row, planned), rolled, wsp_run)["passed"] is False


def test_a_tampered_set_is_recorded_parity_failed(pipeline_db, wsp_run, tiny_lm, tiny_tokenizer):
    """Through `capture_one` (the task's path): a set whose stored scores no longer match is `parity_failed`."""
    from src.services.wsp_capture import capture_one

    planned = _plan(pipeline_db, wsp_run, "eval")
    tampered = dataclasses.replace(planned, scores=planned.scores + 0.5)
    from src.models.wsp import WspActivationCapture

    row = WspActivationCapture(probe_id=wsp_run.id, dataset_id=planned.view_id, role="eval",
                               source_evaluation_id=None, layer=wsp_run.layer, scope="all")
    pipeline_db.add(row)
    pipeline_db.commit()
    row = capture_one(pipeline_db, row, tampered, model=tiny_lm, tokenizer=tiny_tokenizer,
                      architecture="LlamaForCausalLM")
    assert row.status == "parity_failed" and row.parity["passed"] is False and row.finished_at is not None


def test_a_control_that_cannot_fail_makes_the_capture_unusable(pipeline_db, captured, wsp_run):
    """A head whose weight is constant is unchanged by any permutation, so the permuted-weight control cannot fail:
    the capture proves nothing and must read `parity_failed`, even though its own scores agree."""
    import torch

    from src.services.probe_monitor_run import _aggregate_rows, load_probe
    from src.services.wsp_capture import check_parity

    row, planned = captured["eval"]
    head, trained = load_probe(pipeline_db, wsp_run.id)
    flat = dataclasses.replace(head, weight=torch.full_like(head.weight, 0.05))
    files = _reload(row, planned)
    own = np.asarray(_aggregate_rows(flat, trained.rule, dict(trained.rule_params or {}), files.rows()),
                     dtype=np.float32)
    result = check_parity(flat, trained, files, dataclasses.replace(planned, scores=own), wsp_run)
    assert result["controls"]["permuted_w"] is False
    assert (result["passed"], result["reason"]) == (False, "controls_did_not_fail")


# ── R2 review fix (OR2-044a-M1), 2026-11-17: the calibration capture reproduces `probe.threshold` ────────────────


def test_calibration_parity_compares_the_probes_threshold(pipeline_db, captured, wsp_run):
    """FR-1 (R2B-19): `calibrate` over the stored calibration negatives must reproduce `probe.threshold`, and the
    capture must put as many rows at or above it. The same real capture passes with the probe's own threshold and fails
    with a believable stale one (the next stored negative below the bar) and with ±1e9 — it used to pass all three
    (review experiment E2). Controls: `checks.append(bool(out["bar_reproduced"]))` → True, or
    `stored_reproduces_the_threshold` → True → red."""
    from types import SimpleNamespace

    from src.services.probe_monitor_run import load_probe
    from src.services.wsp_capture import check_parity

    row, planned = captured["calibration"]
    head, trained = load_probe(pipeline_db, wsp_run.id)
    files = _reload(row, planned)
    own = check_parity(head, trained, files, planned, wsp_run)
    assert own["passed"] and own["bar_reproduced"] and own["stored_reproduces_the_threshold"], own
    assert own["count_stored"] == own["count_capture"] and own["threshold_served"] == wsp_run.threshold
    cut = planned.scores if planned.truncated is None else planned.scores[np.asarray(planned.truncated) == 0]
    ranked = np.sort(cut.astype(np.float64))[::-1]
    allowed = int(float(wsp_run.target_fpr) * len(ranked))
    assert 0 < allowed < len(ranked), "fixture: the bar sits inside the calibration negatives"
    stale = float(ranked[allowed])   # the next negative below the bar: a re-cut nobody re-stored
    assert np.float32(stale) != np.float32(wsp_run.threshold), "fixture: the stale bar must differ from the real one"
    for threshold in (stale, 1e9, -1e9):
        result = check_parity(head, trained, files, planned,
                              SimpleNamespace(threshold=threshold, target_fpr=wsp_run.target_fpr))
        assert result["passed"] is False and result["stored_reproduces_the_threshold"] is False, (threshold, result)
        assert result["reason"] == "scores_differ" and result["bar_reproduced"] is False


def test_calibration_bar_counts_at_the_served_threshold():
    """The count half alone, on arrays built so that ONLY the count differs: every row within the per-row tolerance,
    the stored array reproducing the threshold, the stored threshold row still supplying the capture's bar — and one
    row just under the bar in the stored array that the capture puts just over it; then the threshold-row check alone
    (two rows swapping at the bar). Controls: drop the count, or the threshold row, from `bar_reproduced` → red."""
    from types import SimpleNamespace

    from src.services.wsp_capture import PlannedSet, calibration_bar

    stored = np.asarray([9.0, 8.0, 7.0, 5.0, 4.9996, 3.0, 2.0, 1.0, 0.5, 0.0], dtype=np.float32)
    probe = SimpleNamespace(threshold=5.0, target_fpr=0.4)       # allowed = 4: the 4th highest negative, 5.0
    planned = PlannedSet(view_id="v", role="calibration", evaluation_id=None, positions=np.arange(10),
                         labels=np.zeros(10, np.int8), scores=stored, truncated=None)
    same = calibration_bar(stored.copy(), planned, probe)
    assert same["bar_reproduced"] and same["count_stored"] == same["count_capture"] == 4
    moved = stored.copy()
    moved[4] = 5.0   # 4e-4 from the stored 4.9996 (inside ROW_ATOL): a tie with the bar row, so now AT the bar
    result = calibration_bar(moved, planned, probe)
    assert result["stored_reproduces_the_threshold"] and result["threshold_row_holds_the_bar"], result
    assert (result["count_stored"], result["count_capture"]) == (4, 5)
    assert result["bar_reproduced"] is False
    # And the threshold-row half alone: the rows either side of the bar swap places inside the tolerance, so the
    # count at the served bar is unchanged (4) but the capture's bar row is no longer the stored one.
    swapped = stored.copy()
    swapped[3], swapped[4] = np.float32(4.9997), np.float32(5.0)
    alone = calibration_bar(swapped, planned, probe)
    assert alone["stored_reproduces_the_threshold"] and alone["count_stored"] == alone["count_capture"] == 4
    assert alone["threshold_row_holds_the_bar"] is False and alone["bar_reproduced"] is False


def test_calibration_bar_is_cut_on_the_probes_own_mask():
    """G5: a bar cut without its truncated negatives is reproduced from the stored array UNDER the probe's mask (and an
    all-rows bar from every row). A truncated row above the bar changes the order statistic, so the two differ here."""
    from types import SimpleNamespace

    from src.services.wsp_capture import PlannedSet, calibration_bar

    stored = np.asarray([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    mask = np.zeros(10, np.int8)
    mask[0] = 1                                   # the highest negative was cut at max_length
    masked = PlannedSet(view_id="v", role="calibration", evaluation_id=None, positions=np.arange(10),
                        labels=np.zeros(10, np.int8), scores=stored, truncated=mask)
    # 9 untruncated rows at 0.25 → allowed 2 → the 2nd highest untruncated, 7.0; all rows → allowed 2 → 8.0.
    assert calibration_bar(stored.copy(), masked, SimpleNamespace(threshold=7.0, target_fpr=0.25))["bar_reproduced"]
    assert not calibration_bar(stored.copy(), masked, SimpleNamespace(threshold=8.0, target_fpr=0.25))["bar_reproduced"]
    unmasked = PlannedSet(view_id="v", role="calibration", evaluation_id=None, positions=np.arange(10),
                          labels=np.zeros(10, np.int8), scores=stored, truncated=None)
    assert calibration_bar(stored.copy(), unmasked, SimpleNamespace(threshold=8.0, target_fpr=0.25))["bar_reproduced"]


def test_the_task_captures_only_pending_rows(pipeline_db, wsp_run):
    """R3 review fix (OR3-044a-L2, the bundle's twin): a capture the janitor failed `never_started` while its message
    waited in the GPU queue is not captured when the message finally arrives; it keeps the janitor's reason and the model
    is never loaded. The same row re-armed `pending` IS taken up (here the test's loader stops it). Control: run every
    listed row whatever its status → the reaped row is taken up → red."""
    from src.models.wsp import WspActivationCapture
    from src.services.wsp_capture import run_captures
    from src.workers.cleanup_stuck_wsp_rows import WORDS

    planned = _plan(pipeline_db, wsp_run, "eval")
    # No evaluation id: the (probe, view, evaluation) key holds a NULL, so this row collides with no other capture.
    row = WspActivationCapture(probe_id=wsp_run.id, dataset_id=planned.view_id, role="eval", layer=wsp_run.layer,
                               scope="all", status="failed", error=WORDS["never_started"])
    pipeline_db.add(row)
    pipeline_db.commit()
    loads = []

    def loader(run):
        loads.append(run.id)
        raise RuntimeError("stopped by the test at the model load")

    try:
        out = run_captures(pipeline_db, wsp_run.id, [row.id], model_loader=loader)
        pipeline_db.refresh(row)
        assert out["captures"] == {row.id: "skipped:failed"} and loads == []
        assert (row.status, row.error, row.parity, row.dir_path) == ("failed", WORDS["never_started"], None, None)

        row.status, row.error = "pending", None
        pipeline_db.commit()
        with pytest.raises(Exception):
            run_captures(pipeline_db, wsp_run.id, [row.id], model_loader=loader)
        pipeline_db.refresh(row)
        assert row.status == "failed" and row.error and not row.error.startswith("never_started")
    finally:
        pipeline_db.rollback()
        pipeline_db.delete(pipeline_db.get(WspActivationCapture, row.id))
        pipeline_db.commit()
