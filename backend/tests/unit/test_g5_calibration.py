"""ENH-010 G5 (GR-6): calibrations cut after G5 leave truncated negatives out; re-cuts apply the stored mask.

The calibrating stage runs for real (`_prepare_calibration_rows` → `_score_calibration_group` → `_calibrate_probes`)
over an Arrow split on disk; only the renderer and the model forward are faked. The fake renderer marks every 10th
row truncated, and the fake forward scores those rows highest — so a bar cut over every negative sits above the
truncated rows and a bar cut over the untruncated ones does not. The arithmetic re-cut is driven through
`probe_recalibration.propose` / `recut_windows` over arrays on disk.

Controls (0xcc/reviews/G5_implementation_controls_2026-11-16.md):
  apply the exclusion when the mask is absent        → test_legacy_recut_reproduces_the_shipped_bar_exactly red
  skip the mask in `recut_windows`                    → test_window_and_band_recuts_apply_the_mask red
  calibrate the run's bar over every negative         → test_new_calibration_excludes_truncated_negatives red
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.config import settings
from src.models.dataset import Dataset
from src.models.probe_monitor import ProbeMonitorDataset
from src.services import probe_monitor_capture, probe_monitor_render, probe_monitor_run as run_mod
from src.services import probe_recalibration as R
from src.services.probe_monitor_trainer import calibrate

N_SPLIT = 600


@pytest.fixture
def world(tmp_path, monkeypatch):
    from datasets import Dataset as HFDataset

    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    raw = tmp_path / "datasets" / "cal_g5"
    HFDataset.from_dict({
        "text": [f"row {i}" for i in range(N_SPLIT)],
        "kind": ["neg"] * N_SPLIT,
    }).save_to_disk(str(raw))

    def fake_render_all(tokenizer, conversations, **kwargs):
        out = []
        for conversation in conversations:
            text = " ".join(m["content"] for m in conversation)
            out.append(SimpleNamespace(text=text, truncated=int(text.split()[1]) % 10 == 0))
        return out, SimpleNamespace(as_dict=lambda: {})

    def fake_by_scope(model, rendered, specs, *, scopes, **kwargs):
        def row(r):
            number = int(r.text.split()[1])
            # Truncated rows fire hardest (D128: every truncated row fired); the rest spread 0..~600.
            value = 10_000.0 + number if r.truncated else float(number)
            return SimpleNamespace(aggregate=value, n_scored=5 + number % 7)
        return {scope: [[row(r) for r in rendered] for _ in specs] for scope in scopes}

    monkeypatch.setattr(probe_monitor_render, "render_all", fake_render_all)
    monkeypatch.setattr(probe_monitor_capture, "forward_scores_by_scope", fake_by_scope)
    dataset = SimpleNamespace(id="ds-cal", raw_path=str(raw))
    view = SimpleNamespace(id="pmd_cal", dataset_id="ds-cal", input_column="text", label_column="kind",
                           pair_column=None, label_mapping={"neg": "negative"}, keyword_filter=None, split=None,
                           name="cal")
    directory = tmp_path / "probe_monitors" / "pmr_g5"
    directory.mkdir(parents=True)

    class DB:
        def query(self, model):
            target = {Dataset: dataset, ProbeMonitorDataset: view}[model]
            return SimpleNamespace(filter=lambda *a, **k: SimpleNamespace(first=lambda: target))

    return SimpleNamespace(db=DB(), dir=directory)


def test_new_calibration_excludes_truncated_negatives(world, monkeypatch):
    from src.services import probe_decision

    monkeypatch.setattr(probe_decision, "calibration_or_not_formable", lambda db, cid, source: None)
    monkeypatch.setattr(probe_decision, "write_decision_snapshot", lambda probe, calibration: {})
    ctx = SimpleNamespace(artifact_dir=world.dir, scope="all", max_length=64, target_fpr=0.05, seed=0,
                          skip_template_prefix=False, config={"calibration_max_rows": N_SPLIT})
    probe = SimpleNamespace(id="pm_g5", variant="dense", sae_id=None, sae_feature_indices=None)
    run_mod._calibrate_probes(world.db, [(probe, SimpleNamespace(layer=1), SimpleNamespace(rule="mean", rule_params={}),
                                          [], [])], "pmd_cal", ctx, None, SimpleNamespace(pad_token_id=0), "llama")
    negatives = np.load(probe.calibration_scores_path)
    mask = np.load(probe.calibration_truncated_path)
    assert mask.dtype == np.int8 and len(mask) == len(negatives) == N_SPLIT, "every negative stays on disk"
    assert int(mask.sum()) == N_SPLIT // 10
    kept = [float(v) for v, m in zip(negatives.tolist(), mask.tolist()) if not m]
    want = calibrate(kept, target_fpr=0.05, source="calibration_set")
    everything = calibrate([float(v) for v in negatives.tolist()], target_fpr=0.05, source="calibration_set")
    assert probe.threshold == want.threshold
    assert probe.threshold != everything.threshold, "fixture too weak: the exclusion moved nothing"
    assert probe.calibration_n_negatives == len(kept)
    # Each contract window that was cut carries its own mask and the count it left out.
    for window, entry in (probe.window_decisions or {}).items():
        assert entry["truncated_excluded"] == N_SPLIT // 10, window
        assert np.load(entry["truncated_path"]).tolist() == mask.tolist(), window
        assert entry["threshold"] == want.threshold, window
    assert probe.window_decisions, "no window was cut, so the window half proves nothing"


def test_the_global_bar_records_its_untruncated_count_without_windows(world, monkeypatch):
    """R2 review fix (OR2-044-1), 2026-11-16: the window pass re-writes `calibration_n_negatives`, so the twin above
    cannot see the global cut's own count. With no window pass, the record must be the UNTRUNCATED count the bar was
    cut from (OVN-8), not every negative. Control: count `negatives_scores` at the global cut → red."""
    from src.services import probe_decision

    monkeypatch.setattr(probe_decision, "calibration_or_not_formable", lambda db, cid, source: None)
    monkeypatch.setattr(probe_decision, "write_decision_snapshot", lambda probe, calibration: {})
    monkeypatch.setattr(run_mod, "_calibrate_windows", lambda *args, **kwargs: None)
    ctx = SimpleNamespace(artifact_dir=world.dir, scope="all", max_length=64, target_fpr=0.05, seed=0,
                          skip_template_prefix=False, config={"calibration_max_rows": N_SPLIT})
    probe = SimpleNamespace(id="pm_g5w", variant="dense", sae_id=None, sae_feature_indices=None)
    run_mod._calibrate_probes(world.db, [(probe, SimpleNamespace(layer=1), SimpleNamespace(rule="mean", rule_params={}),
                                          [], [])], "pmd_cal", ctx, None, SimpleNamespace(pad_token_id=0), "llama")
    mask = np.load(probe.calibration_truncated_path)
    assert int(mask.sum()) == N_SPLIT // 10, "fixture too weak: nothing truncated"
    assert probe.calibration_n_negatives == N_SPLIT - N_SPLIT // 10


def _save(path: Path, values, dtype=np.float32) -> str:
    np.save(path, np.asarray(list(values), dtype=dtype))
    return str(path)


def _stub(tmp_path: Path, negatives, *, mask=None, lengths=None, bands=None, windows=None, threshold=None):
    return SimpleNamespace(
        id="pm_g5", calibration_scores_path=_save(tmp_path / "neg.npy", negatives),
        calibration_truncated_path=_save(tmp_path / "trunc.npy", mask, np.int8) if mask is not None else None,
        calibration_lengths_path=_save(tmp_path / "len.npy", lengths) if lengths is not None else None,
        threshold=threshold, target_fpr=0.01, realised_fpr=0.01, threshold_source="calibration_set",
        window_decisions=windows, length_bands=bands, published=None, calibration_history=None,
        definition_path=None, run_id="pmr_g5", decision_snapshot=None, decision_digest=None, created_at=None,
    )


def _negatives_and_mask():
    values = [float(v) for v in np.linspace(-10.0, 30.0, 2000, dtype=np.float32)]
    mask = [1 if i % 25 == 0 else 0 for i in range(2000)]
    for i, flag in enumerate(mask):
        if flag:
            values[i] = float(np.float32(1000.0 + i))      # truncated rows score highest
    return values, mask


def test_legacy_recut_reproduces_the_shipped_bar_exactly(tmp_path):
    """No mask (a calibration cut before G5): the re-cut uses the full array, so the shipped bar comes back bit for
    bit (memory `calibrate-is-an-order-statistic`; G5-D5)."""
    values, _mask = _negatives_and_mask()
    shipped = calibrate(values, target_fpr=0.01, source="calibration_set")
    proposed = R.propose(_stub(tmp_path, values, threshold=shipped.threshold), [], target_fpr=0.01)["proposed"]
    assert proposed["threshold"] == shipped.threshold
    assert proposed["realised_fpr"] == shipped.realised_fpr
    # Every negative was used: a control that dropped one shipped row still landed on the same order statistic
    # (int(0.01 × 1999) = 19 → the same value), so the count is pinned as well as the bar.
    assert proposed["n_negatives"] == len(values)
    assert proposed["truncated_excluded"] is None


def test_global_recut_applies_the_mask(tmp_path):
    values, mask = _negatives_and_mask()
    kept = [v for v, m in zip(values, mask) if not m]
    proposed = R.propose(_stub(tmp_path, values, mask=mask), [], target_fpr=0.01)["proposed"]
    assert proposed["threshold"] == calibrate(kept, target_fpr=0.01, source="calibration_set").threshold
    assert proposed["n_negatives"] == len(kept) and proposed["truncated_excluded"] == sum(mask)


def test_window_and_band_recuts_apply_the_mask(tmp_path):
    """GR-6 for the per-window and per-length re-cuts (`recut_windows`, `recut_length_bands`)."""
    values, mask = _negatives_and_mask()
    kept = [v for v, m in zip(values, mask) if not m]
    lengths = [50 + (i % 800) for i in range(2000)]
    kept_lengths = [n for n, m in zip(lengths, mask) if not m]
    window_dir = tmp_path / "w"
    window_dir.mkdir()
    windows = {"prompt": {
        "threshold": 1.0, "target_fpr": 0.01, "realised_fpr": 0.01, "n_negatives": len(kept), "scope": "input",
        "scores_path": _save(window_dir / "neg.npy", values),
        "truncated_path": _save(window_dir / "trunc.npy", mask, np.int8),
        "lengths_path": _save(window_dir / "len.npy", lengths),
        "length_bands": [{"min_tokens": 0, "max_tokens": None, "threshold": 9.9, "threshold_source": "band",
                          "target_fpr": 0.01, "realised_fpr": 0.01, "n_negatives": len(kept)}],
    }}
    bands = [{"min_tokens": 0, "max_tokens": None, "threshold": 9.9, "threshold_source": "band",
              "target_fpr": 0.01, "realised_fpr": 0.01, "n_negatives": len(kept)}]
    got = R.propose(_stub(tmp_path, values, mask=mask, lengths=lengths, bands=bands, windows=windows), [],
                    target_fpr=0.02)
    want = calibrate(kept, target_fpr=0.02, source="calibration_set")
    window = got["window_decisions"]["prompt"]
    assert window["threshold"] == want.threshold and window["n_negatives"] == len(kept)
    from src.services.probe_monitor_metrics import length_band_decisions

    want_bands = length_band_decisions(kept, kept_lengths, target_fpr=0.02, global_threshold=want.threshold)
    assert got["length_bands"] == want_bands
    assert window["length_bands"] == want_bands


def test_a_recorded_mask_that_is_gone_is_refused(tmp_path):
    values, mask = _negatives_and_mask()
    probe = _stub(tmp_path, values, mask=mask)
    Path(probe.calibration_truncated_path).unlink()
    with pytest.raises(R.RecalibrationRefused) as caught:
        R.propose(probe, [], target_fpr=0.01)
    assert caught.value.code == "calibration_mask_missing"
