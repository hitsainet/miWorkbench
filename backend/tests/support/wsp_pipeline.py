"""A REAL tiny probe run with WSP captures, for the bundle, baseline and scorer tests (ENH-010 044; FTID I44-D1).

The run trains on the pipeline test's Arrow corpus (five words a row) and is evaluated and calibrated on a SEPARATE
held-out corpus (six words a row), so no held-out row can share a location, text or row key with a practice row and a
bundle builds. Import the fixtures by name into a test module (module scope: one run per module).
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from tests.unit.test_probe_monitor_pipeline import (  # noqa: F401 (re-exported fixtures)
    _loader_that_records_its_placement,
    arrow_dataset,
    monkeypatch_module,
    pipeline_db,
    tiny_lm,
    tiny_tokenizer,
)

HELDOUT_ROWS = 60


@pytest.fixture(scope="module")
def heldout_arrow(tmp_path_factory):
    """Six words a row (the practice corpus has five), so every held-out text differs from every practice text."""
    from datasets import Dataset as HFDataset

    rng = np.random.default_rng(11)
    inputs, labels = [], []
    for i in range(HELDOUT_ROWS):
        positive = i % 2 == 0
        marker = "risk" if positive else "calm"
        words = [f"{marker}{rng.integers(0, 8)}" for _ in range(3)] + [f"filler{rng.integers(0, 16)}"
                                                                         for _ in range(3)]
        inputs.append(" ".join(words))
        labels.append("high" if positive else "low")
    path = tmp_path_factory.mktemp("wsp_heldout") / "arrow"
    HFDataset.from_dict({"inputs": inputs, "stakes": labels}).save_to_disk(str(path))
    return path


def make_world(db, practice_arrow, heldout_path, *, calibration: bool = True, eval_max_rows=None,
               name: str = "unseen", target_fpr: float = 0.1, calibration_path=None, calibration_max_rows=None):
    """A model, the two datasets, a training view, an eval view and (with `calibration`) a calibration view, and a
    pending run over them. Without `calibration` the probe's bar is cut on its practice validation negatives
    (`threshold_source = validation_negatives`, the practice basis); `eval_max_rows` makes the held-out evaluation a
    seeded SAMPLE of the view, so its positions are not the identity. `calibration_path` puts the calibration view on
    a THIRD dataset (the eval view keeps `heldout_path`); `calibration_max_rows` makes the calibration rows a seeded
    sample of that view."""
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun

    model_row = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny-random", status=ModelStatus.READY,
                      file_path=str(practice_arrow), architecture="LlamaForCausalLM", params_count=1_000)
    practice_ds = Dataset(id=uuid.uuid4(), name="practice-corpus", source="local", status=DatasetStatus.READY,
                          raw_path=str(practice_arrow), extra_metadata={})
    heldout_ds = Dataset(id=uuid.uuid4(), name="heldout-corpus", source="local", status=DatasetStatus.READY,
                         raw_path=str(heldout_path), extra_metadata={})
    db.add_all([model_row, practice_ds, heldout_ds])
    db.commit()
    mapping = {"high": "positive", "low": "negative"}
    train = ProbeMonitorDataset(name="train", dataset_id=practice_ds.id, input_column="inputs",
                                label_column="stakes", label_mapping=mapping, pair_column="pair", role="train",
                                counts={})
    heldout = ProbeMonitorDataset(name=name, dataset_id=heldout_ds.id, input_column="inputs",
                                  label_column="stakes", label_mapping=mapping, role="eval",
                                  distribution="out_of_distribution", counts={})
    views = [train, heldout]
    cal = None
    cal_ds = heldout_ds
    if calibration and calibration_path is not None:
        cal_ds = Dataset(id=uuid.uuid4(), name="calibration-corpus", source="local", status=DatasetStatus.READY,
                         raw_path=str(calibration_path), extra_metadata={})
        db.add(cal_ds)
        db.commit()
    if calibration:
        cal = ProbeMonitorDataset(name="calm", dataset_id=cal_ds.id, input_column="inputs",
                                  label_column="stakes", label_mapping={"low": "negative"}, role="calibration",
                                  counts={})
        views.append(cal)
    db.add_all(views)
    db.commit()
    config = {"layers": [1, 2], "rules": ["mean"], "scope": "all", "max_length": 64, "val_fraction": 0.25,
              "seed": 1337, "top_n_layers": 1, "target_fpr": float(target_fpr)}
    if eval_max_rows:
        config["eval_max_rows"] = int(eval_max_rows)
    if calibration_max_rows:
        config["calibration_max_rows"] = int(calibration_max_rows)
    run = ProbeMonitorRun(
        model_id=model_row.id, train_dataset_id=train.id, eval_dataset_ids=[heldout.id],
        calibration_dataset_id=cal.id if cal is not None else None, config=config, status="pending", environment={})
    db.add(run)
    db.commit()
    return {"run": run, "train": train, "eval": heldout, "calibration": cal, "practice_ds": practice_ds,
            "heldout_ds": heldout_ds, "calibration_ds": cal_ds if cal is not None else None}


def run_probe(db, world, tiny_lm, tiny_tokenizer):
    """Run the world's probe run for real; the probe with weights."""
    from src.models.probe_monitor import ProbeMonitor
    from src.services import probe_monitor_run as run_module

    run_module.execute_probe_run(db, world["run"].id,
                                 model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
    probe = (db.query(ProbeMonitor).filter(ProbeMonitor.run_id == world["run"].id,
                                           ProbeMonitor.weights_path.isnot(None)).first())
    assert probe is not None
    return probe


def capture_all(db, probe, tiny_lm, tiny_tokenizer):
    """Every held-out set of the probe captured for real; all must reach `ready`."""
    from src.models.wsp import WspActivationCapture
    from src.services.wsp_capture import capture_one, plan_capture

    rows = []
    for planned in plan_capture(db, probe):
        row = WspActivationCapture(probe_id=probe.id, dataset_id=planned.view_id, role=planned.role,
                                   source_evaluation_id=planned.evaluation_id, layer=probe.layer, scope="all")
        db.add(row)
        db.commit()
        rows.append(capture_one(db, row, planned, model=tiny_lm, tokenizer=tiny_tokenizer,
                                architecture="LlamaForCausalLM"))
    assert [r.status for r in rows] == ["ready"] * len(rows), [r.parity for r in rows]
    return rows


@pytest.fixture(scope="module")
def wsp_data_dir(tmp_path_factory, monkeypatch_module):
    """ONE artifact root per module: every run's files (and so every bundle's) resolve under it, whichever world's
    fixture ran last (`artifact_dir_for` reads `settings.data_dir` at each call)."""
    from src.core import config as config_module

    path = tmp_path_factory.mktemp("wsp_artifacts")
    monkeypatch_module.setattr(config_module.settings, "data_dir", path, raising=False)
    return path


@pytest.fixture(scope="module")
def wsp_world(pipeline_db, arrow_dataset, heldout_arrow):
    return make_world(pipeline_db, arrow_dataset, heldout_arrow)


@pytest.fixture(scope="module")
def wsp_probe(pipeline_db, wsp_world, tiny_lm, tiny_tokenizer, wsp_data_dir):
    probe = run_probe(pipeline_db, wsp_world, tiny_lm, tiny_tokenizer)
    assert probe.threshold_source == "calibration_set"
    return probe


@pytest.fixture(scope="module")
def wsp_captures(pipeline_db, wsp_probe, tiny_lm, tiny_tokenizer):
    return capture_all(pipeline_db, wsp_probe, tiny_lm, tiny_tokenizer)


# ── R2 review (OR2-044a-M2, -M3, -M4, -L4): the PRACTICE basis, over a SAMPLED held-out view ────────────────────
#
# Every fixture above cuts its bars on a calibration set and evaluates the whole held-out view in order, so the
# practice basis (the default for any probe without a calibration set) and a sampled evaluation's positions were
# never exercised: on the fixtures the stored positions equalled `arange(rows)` by construction. The pool is larger
# than the sample so the sample still holds the evaluator's floor of 20 rows of each class.

PB_ROWS, PB_SAMPLE = 160, 80


def write_heldout(path, rows, seed, *, repeat_from=None, practice_texts=None):
    """Six words a row (never a practice text), alternating labels; from `repeat_from` on, row `i` repeats practice
    row `i - repeat_from` instead (same label: both corpora alternate from an even index)."""
    from datasets import Dataset as HFDataset

    rng = np.random.default_rng(seed)
    inputs, labels = [], []
    for i in range(rows):
        positive = i % 2 == 0
        if repeat_from is not None and i >= repeat_from:
            inputs.append(practice_texts[i - repeat_from])
        else:
            marker = "risk" if positive else "calm"
            words = [f"{marker}{rng.integers(0, 8)}" for _ in range(3)] + [f"filler{rng.integers(0, 16)}"
                                                                             for _ in range(3)]
            inputs.append(" ".join(words))
        labels.append("high" if positive else "low")
    HFDataset.from_dict({"inputs": inputs, "stakes": labels}).save_to_disk(str(path))
    return path


@pytest.fixture(scope="module")
def pb_heldout_arrow(tmp_path_factory):
    return write_heldout(tmp_path_factory.mktemp("wsp_pb_heldout") / "arrow", PB_ROWS, 17)


@pytest.fixture(scope="module")
def pb_world(pipeline_db, arrow_dataset, pb_heldout_arrow):
    # 0.25: the practice validation split holds about ten negatives, so a 10% target would place no bar at all.
    return make_world(pipeline_db, arrow_dataset, pb_heldout_arrow, calibration=False, eval_max_rows=PB_SAMPLE,
                      name="unseen-sampled", target_fpr=0.25)


@pytest.fixture(scope="module")
def pb_probe(pipeline_db, pb_world, tiny_lm, tiny_tokenizer, wsp_data_dir):
    probe = run_probe(pipeline_db, pb_world, tiny_lm, tiny_tokenizer)
    assert (probe.threshold_source, probe.threshold is not None) == ("validation_negatives", True), \
        f"fixture: practice basis ({probe.threshold_source}, {probe.threshold}, {probe.calibration_n_negatives})"
    return probe


@pytest.fixture(scope="module")
def pb_captures(pipeline_db, pb_probe, tiny_lm, tiny_tokenizer):
    rows = capture_all(pipeline_db, pb_probe, tiny_lm, tiny_tokenizer)
    assert [r.role for r in rows] == ["eval"], "fixture: a practice-basis probe has no calibration capture"
    positions = np.load(Path(rows[0].dir_path) / "positions.npy")
    assert len(positions) == PB_SAMPLE and not np.array_equal(positions, np.arange(PB_SAMPLE)), \
        "fixture: the held-out evaluation is a sample, so its positions are not the identity"
    return rows


#: The overlap corpus: rows from `OV_REPEAT_FROM` on repeat practice texts; its evaluation samples `OV_SAMPLE` rows.
OV_ROWS, OV_SAMPLE, OV_REPEAT_FROM = 160, 80, 80


@pytest.fixture(scope="module")
def overlap_arrow(tmp_path_factory, arrow_dataset):
    """A held-out corpus whose second half repeats PRACTICE texts: a sample of it holds overlapping rows only at
    positions a whole-view-in-order reading of `OV_SAMPLE` rows never reaches."""
    from datasets import load_from_disk

    practice = load_from_disk(str(arrow_dataset))["inputs"]
    assert len(practice) >= OV_ROWS - OV_REPEAT_FROM
    return write_heldout(tmp_path_factory.mktemp("wsp_overlap") / "arrow", OV_ROWS, 13, repeat_from=OV_REPEAT_FROM,
                         practice_texts=practice)


@pytest.fixture(scope="module")
def ov_world(pipeline_db, arrow_dataset, overlap_arrow):
    return make_world(pipeline_db, arrow_dataset, overlap_arrow, calibration=False, eval_max_rows=OV_SAMPLE,
                      name="overlapping-sampled")


@pytest.fixture(scope="module")
def ov_probe(pipeline_db, ov_world, tiny_lm, tiny_tokenizer, wsp_data_dir):
    return run_probe(pipeline_db, ov_world, tiny_lm, tiny_tokenizer)


@pytest.fixture(scope="module")
def ov_captures(pipeline_db, ov_probe, tiny_lm, tiny_tokenizer):
    return capture_all(pipeline_db, ov_probe, tiny_lm, tiny_tokenizer)


# ── R3 review (OR3-044a-M2): a CALIBRATION set that repeats practice texts at sampled positions ─────────────────────
#
# Every overlap fixture above is a practice-basis world, or puts the overlap in an eval set, so leaving the calibration
# capture out of the P45/P67 check (review survivor X39) stayed green. Here the eval view is the clean held-out corpus
# and the calibration view is a third corpus whose rows from `CO_REPEAT_FROM` on repeat practice texts; the run
# calibrates on a seeded sample of `CO_SAMPLE` of its negatives.
CO_ROWS, CO_SAMPLE, CO_REPEAT_FROM = 160, 64, 80


@pytest.fixture(scope="module")
def cal_overlap_arrow(tmp_path_factory, arrow_dataset):
    from datasets import load_from_disk

    practice = load_from_disk(str(arrow_dataset))["inputs"]
    assert len(practice) >= CO_ROWS - CO_REPEAT_FROM
    return write_heldout(tmp_path_factory.mktemp("wsp_cal_overlap") / "arrow", CO_ROWS, 19,
                         repeat_from=CO_REPEAT_FROM, practice_texts=practice)


@pytest.fixture(scope="module")
def co_world(pipeline_db, arrow_dataset, heldout_arrow, cal_overlap_arrow):
    return make_world(pipeline_db, arrow_dataset, heldout_arrow, name="unseen-cal-overlap",
                      calibration_path=cal_overlap_arrow, calibration_max_rows=CO_SAMPLE)


@pytest.fixture(scope="module")
def co_probe(pipeline_db, co_world, tiny_lm, tiny_tokenizer, wsp_data_dir):
    probe = run_probe(pipeline_db, co_world, tiny_lm, tiny_tokenizer)
    assert probe.threshold_source == "calibration_set", "fixture: the bar is cut on the calibration set"
    return probe


@pytest.fixture(scope="module")
def co_captures(pipeline_db, co_probe, tiny_lm, tiny_tokenizer):
    return capture_all(pipeline_db, co_probe, tiny_lm, tiny_tokenizer)


def new_bundle(db, probe):
    from src.models.wsp import WspBundle

    row = WspBundle(probe_id=probe.id, status="building")
    db.add(row)
    db.commit()
    return row
