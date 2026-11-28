"""ENH-010 G5 (GR-10): every reader of the evaluation arrays and calibration negatives has a decision, and keeps it.

The registry `probe_evaluation_scores.G5_READERS` is checked against the readers found by walking `backend/src`'s AST
(attribute reads of the array columns, `getattr(..., "<column>")`, and calls of the array loaders) — so a new reader
without a decision fails here. Each "apply" reader is then driven over TWIN fixtures that differ ONLY in the mask.

Controls (0xcc/reviews/G5_implementation_controls_2026-11-16.md):
  drop the mask in `probe_breakdown`                    → test_probe_breakdown_respects_the_mask red
  drop the mask in `detector_success_report._figures`   → test_detector_success_report_respects_the_mask red
  drop the mask in `probe_blend.service.load_member_arrays` → test_probe_blend_respects_the_mask red
  drop the mask in `recall_at_decision`                 → test_selection_respects_the_mask red
  remove a registry row                                 → test_every_reader_has_a_decision red
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.config import settings
from src.services import probe_evaluation_scores as pes

SRC = Path(__file__).resolve().parents[2] / "src"

#: The columns whose readers must decide, and the loaders that read them.
ARRAY_ATTRS = {"scores_path", "labels_path", "positions_path", "n_scored_path", "truncated_path",
               "calibration_scores_path", "calibration_truncated_path"}
LOADERS = {"load_evaluation_arrays", "load_calibration_array", "load_member_arrays"}


def _readers():
    found = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = "src." + ".".join(path.relative_to(SRC).with_suffix("").parts)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ARRAY_ATTRS:
                found.add(module)
            elif isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name in LOADERS:
                    found.add(module)
                if name == "getattr" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                        and node.args[1].value in ARRAY_ATTRS:
                    found.add(module)
    return found


def test_every_reader_has_a_decision():
    readers = _readers()
    assert len(readers) >= 8, f"the walk found only {sorted(readers)}: it would check nothing"
    assert sorted(readers - set(pes.G5_READERS)) == [], "a reader of the arrays has no G5 decision"
    assert {decision for decision, _why in pes.G5_READERS.values()} <= {
        "apply", "label", "apply_new_calibrations", "owner"}


def test_untruncated_drops_exactly_the_masked_rows():
    loaded = pes.EvaluationArrays(arrays={
        "score": np.array([0.1, 0.9, 0.5, 0.7]), "position": np.array([3, 5, 8, 9]),
        "n_scored": np.array([1, 2, 3, 4]), "label": np.array([0, 1, 0, 1]),
        "truncated": np.array([0, 1, 0, 0], dtype=np.int8)}, total=4)
    arrays, note = pes.untruncated(loaded)
    assert arrays["position"].tolist() == [3, 8, 9] and note == {"rows": "untruncated", "excluded": 1,
                                                                  "policy": "excluded"}
    del loaded.arrays["truncated"]
    arrays, note = pes.untruncated(loaded)
    assert arrays["position"].tolist() == [3, 5, 8, 9] and note["rows"] == "all"
    assert note["reason"] == "truncation_not_recorded"


# ── twins: the same evaluation with and without a mask ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


def _mask_every_third_positive(evaluation, labels):
    """Mark every third positive truncated, on the evaluation's own files — the twin differs only in this."""
    mask = np.zeros(len(labels), dtype=np.int8)
    positives = np.flatnonzero(np.asarray(labels) == 1)
    mask[positives[::3]] = 1
    path = Path(evaluation.scores_path).with_name(Path(evaluation.scores_path).stem + "__truncated.npy")
    np.save(path, mask)
    return mask, str(path)


async def _breakdown_world(async_session, data_dir, request):
    from tests.support.random_behaviour import rng_for
    from tests.unit.test_probe_breakdown import world

    return await world(async_session, data_dir, rng_for(request.node.nodeid))


async def test_probe_breakdown_respects_the_mask(client, async_session, data_dir, request):
    from tests.unit.test_probe_breakdown import url

    w = await _breakdown_world(async_session, data_dir, request)
    before = (await client.get(url(w), params={"by_column": "kind"})).json()
    mask, path = _mask_every_third_positive(w["evaluation"], w["labels"])
    w["evaluation"].truncated_path = path
    await async_session.commit()
    after = (await client.get(url(w), params={"by_column": "kind"})).json()
    total = lambda body: sum(g["positives"]["n"] for g in body["groups"])  # noqa: E731
    assert total(before) - total(after) == int(mask.sum()) > 0
    assert before["truncation"]["rows"] == "all" and after["truncation"] == {
        "rows": "untruncated", "excluded": int(mask.sum()), "policy": "excluded"}


async def test_detector_success_report_respects_the_mask(async_session, data_dir, request):
    from src.services.detector_success_report import _figures

    w = await _breakdown_world(async_session, data_dir, request)
    root = Path(settings.data_dir) / "probe_monitors"
    before = _figures(w["probe"], w["run"], w["evaluation"], w["view"], None, root)
    mask, path = _mask_every_third_positive(w["evaluation"], w["labels"])
    w["evaluation"].truncated_path = path
    after = _figures(w["probe"], w["run"], w["evaluation"], w["view"], None, root)
    assert before != after, "the mask changed nothing the report states"
    # R1 review fix (OR1-044-5), 2026-11-16: the unmasked (pre-G5) twin is LABELLED as judged over every row.
    assert before["truncation"] == {"rows": "all", "excluded": None, "reason": "truncation_not_recorded"}
    assert after["truncation"]["rows"] == "untruncated"


def _twin_files(tmp_path, *, masked):
    """An evaluation row's files under the probe root: 40 rows, every third positive marked when `masked`."""
    directory = Path(settings.data_dir) / "probe_monitors" / "pmr_twin"
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / ("masked" if masked else "plain")
    labels = np.array([i % 2 for i in range(40)], dtype=np.int8)
    scores = np.where(labels == 1, 2.0, -2.0).astype(np.float32)
    np.save(f"{stem}.npy", scores)
    np.save(f"{stem}__positions.npy", np.arange(40, dtype=np.int64) * 2)
    np.save(f"{stem}__n_scored.npy", np.full(40, 5, dtype=np.int32))
    np.save(f"{stem}__labels.npy", labels)
    row = SimpleNamespace(id=f"pme_{stem.name}", scores_path=f"{stem}.npy", positions_path=f"{stem}__positions.npy",
                          n_scored_path=f"{stem}__n_scored.npy", labels_path=f"{stem}__labels.npy",
                          truncated_path=None)
    if masked:
        mask = np.zeros(40, dtype=np.int8)
        mask[np.flatnonzero(labels == 1)[::3]] = 1
        np.save(f"{stem}__truncated.npy", mask)
        row.truncated_path = f"{stem}__truncated.npy"
    return row


def test_probe_blend_respects_the_mask(tmp_path):
    from src.services.probe_blend.service import load_member_arrays

    root = Path(settings.data_dir) / "probe_monitors"
    plain = load_member_arrays(_twin_files(tmp_path, masked=False), root)
    masked = load_member_arrays(_twin_files(tmp_path, masked=True), root)
    assert len(plain.positions) == 40 and len(masked.positions) == 40 - 7
    assert plain.truncation["rows"] == "all" and masked.truncation["rows"] == "untruncated"   # OR1-044-5
    assert set(masked.positions.tolist()) < set(plain.positions.tolist())


def test_selection_respects_the_mask(tmp_path):
    """`recall_at_decision` (selection decides) reads untruncated rows: the masked positives score LOW here, so
    leaving them out raises the recall."""
    from src.services.probe_monitor_run import recall_at_decision

    probe = SimpleNamespace(threshold=0.0, length_bands=None, window_decisions=None)
    run = SimpleNamespace(config={"scope": "all"})
    plain, masked = _twin_files(tmp_path, masked=False), _twin_files(tmp_path, masked=True)
    for row in (plain, masked):     # the masked positives miss the bar
        scores, labels = np.load(row.scores_path), np.load(row.labels_path)
        scores[np.flatnonzero(labels == 1)[::3]] = -5.0
        np.save(row.scores_path, scores)
    assert recall_at_decision(probe, run, plain) == pytest.approx(13 / 20)
    assert recall_at_decision(probe, run, masked) == pytest.approx(1.0)


def test_relays_carry_no_mask_bookkeeping():
    """`label` / `apply_new_calibrations` relays: a window entry's `truncated_path` and `truncated_excluded` never
    reach a contract document or miServer."""
    from src.mcp_server.tools.miserver_probes import contract_windows
    from src.services.probe_decision import window_decisions_or_none

    entry = {"threshold": 1.0, "target_fpr": 0.01, "realised_fpr": 0.01, "n_negatives": 900, "scope": "all",
             "length_bands": None, "scores_path": "/x.npy", "truncated_path": "/t.npy", "truncated_excluded": 100}
    relayed = contract_windows({"all": entry})["all"]
    assert "truncated_path" not in relayed and "truncated_excluded" not in relayed
    exported = window_decisions_or_none({"all": entry})["all"].model_dump()
    assert "truncated_path" not in exported and "truncated_excluded" not in exported


async def test_rules_evaluation_respects_the_mask(client, async_session, data_dir, monkeypatch, request):
    """rules ∪ probe (`rules_evaluation`) over the same world twice: the second time one evaluated row is marked
    truncated, and every detector's row count drops by exactly that row."""
    from tests.support import rules_world as rw

    world = await rw.build(client, async_session, data_dir, request.node.nodeid)
    labels = np.load(world.evaluation.labels_path)
    mask = np.zeros(len(labels), dtype=np.int8)
    mask[int(np.flatnonzero(labels == 1)[0])] = 1
    path = Path(world.evaluation.scores_path).with_name(Path(world.evaluation.scores_path).stem + "__truncated.npy")
    np.save(path, mask)
    world.evaluation.truncated_path = str(path)
    await async_session.commit()
    out = await rw.run_one(client, monkeypatch, world, calibration=False)
    assert out["outcome"]["status"] == "completed", out
    view = (await client.get(f"{rw.API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]["views"][0]
    # The unmasked world's figures (test_rules_evaluation_task.py) are 6 positives and 6 negatives per detector.
    for name in ("rules_alone", "probe_alone", "rules_or_probe"):
        assert view["detectors"][name]["positives"]["n"] == 5, name
        assert view["detectors"][name]["negatives"]["n"] == 6, name


# ── R1 review fix (OR1-044-1), 2026-11-16: the CALIBRATION mask in the three "apply" readers ─────────────────────
#
# Twin fixtures differ ONLY in `calibration_truncated_path`. Controls (G5 controls record, D7/R1/R2): drop the mask in
# `probe_breakdown` target mode, in `rules_evaluation.run`'s union bar, in `calibration_firing_rate` → each test red.


def _calibration_mask(path_dir: Path, probe_id: str, mask) -> str:
    path = Path(path_dir) / f"{probe_id}__calibration_truncated.npy"
    np.save(path, np.asarray(mask, dtype=np.int8))
    return str(path)


def test_calibration_firing_rate_respects_the_calibration_mask(tmp_path):
    """40 negatives; the 10 highest are truncated and fire. With the mask the rate is over the 30 the bar was cut from
    (none fire); without it, 10 of 40."""
    from src.services.probe_monitor_run import calibration_firing_rate

    directory = Path(settings.data_dir) / "probe_monitors" / "pmr_cal_twin"
    directory.mkdir(parents=True, exist_ok=True)
    negatives = np.concatenate([np.full(10, 5.0), np.full(30, -5.0)]).astype(np.float32)
    np.save(directory / "pm_twin__calibration_negatives.npy", negatives)
    run = SimpleNamespace(config={"scope": "all"})

    def probe(mask_path):
        return SimpleNamespace(id="pm_twin", calibration_scores_path=str(directory / "pm_twin__calibration_negatives.npy"),
                               calibration_truncated_path=mask_path, calibration_lengths_path=None, threshold=0.0,
                               length_bands=None, window_decisions=None, realised_fpr=None)

    plain = calibration_firing_rate(probe(None), run)
    masked = calibration_firing_rate(probe(_calibration_mask(directory, "pm_twin", [1] * 10 + [0] * 30)), run)
    assert plain == (pytest.approx(0.25), "global_bar")
    assert masked == (pytest.approx(0.0), "global_bar")


async def test_probe_breakdown_target_mode_respects_the_calibration_mask(client, async_session, data_dir, request):
    """Target mode cuts a bar at a requested FPR on the calibration negatives: with the mask, on the untruncated ones
    only (the 50 highest negatives are marked truncated, so the masked bar is lower)."""
    from src.services.probe_monitor_trainer import calibrate
    from tests.unit.test_probe_breakdown import url

    w = await _breakdown_world(async_session, data_dir, request)
    params = {"by_column": "kind", "mode": "target_fpr", "target_fpr": 0.05}
    plain = (await client.get(url(w), params=params)).json()
    cal = np.asarray(w["cal"], dtype=np.float32)
    mask = np.zeros(len(cal), dtype=np.int8)
    mask[np.argsort(-cal)[:50]] = 1
    w["probe"].calibration_truncated_path = _calibration_mask(Path(w["probe"].calibration_scores_path).parent,
                                                              w["probe"].id, mask)
    await async_session.commit()
    masked = (await client.get(url(w), params=params)).json()
    expected = calibrate([float(x) for x in cal[mask == 0]], target_fpr=0.05, source="calibration_set").threshold
    assert masked["decision"]["threshold"] == pytest.approx(expected, abs=1e-6)
    assert masked["decision"]["threshold"] < plain["decision"]["threshold"]


async def test_rules_union_bar_respects_the_calibration_mask(client, async_session, data_dir, monkeypatch, request):
    """The rules ∪ probe union bar spends 1% of the calibration negatives the probe's bar was cut from: with 200 of the
    1,000 marked truncated, the budget is 8, not 10."""
    from src.models.probe_monitor import ProbeMonitor
    from tests.support import rules_world as rw

    plain_world = await rw.build(client, async_session, data_dir, request.node.nodeid + "-plain")
    plain = await rw.run_one(client, monkeypatch, plain_world, calibration=True)
    masked_world = await rw.build(client, async_session, data_dir, request.node.nodeid + "-masked")
    probe = await async_session.get(ProbeMonitor, masked_world.probe.id)
    mask = np.zeros(rw.CAL_N, dtype=np.int8)
    mask[:200] = 1                       # the 200 highest-scoring negatives (`calibration_scores` is descending)
    probe.calibration_truncated_path = _calibration_mask(Path(probe.calibration_scores_path).parent, probe.id, mask)
    await async_session.commit()
    masked = await rw.run_one(client, monkeypatch, masked_world, calibration=True)

    async def calibration(out):
        body = (await client.get(f"{rw.API}/rules-evaluations/{out['evaluation_id']}")).json()["result"]
        return body["calibration"]["global"]

    assert (await calibration(plain))["budget"] == 10
    assert (await calibration(masked))["budget"] == 8
