"""ENH-010 044 tasks 6.1 – 6.4: the baseline registry and its five fits, on practice rows only (FTID §3.4)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.ml.probe_monitor_model import ProbeHead
from src.schemas import wsp_scoring
from src.services import wsp_baselines as wb
from src.services.probe_monitor_metrics import auroc

D, T = 8, 3


def _practice(n=80, seed=0, planted=None):
    rng = np.random.default_rng(seed)
    planted = planted if planted is not None else np.eye(D)[2]
    labels = (np.arange(n) % 2).astype(int)
    rows = [rng.standard_normal((T, D)).astype(np.float32) + (2.0 * planted * y).astype(np.float32) for y in labels]
    validation = (np.arange(n) % 5 == 0)
    return wb.Practice(rows=rows, labels=labels.tolist(), validation_mask=validation.tolist())


def _parent():
    g = torch.Generator().manual_seed(1)
    return ProbeHead(weight=torch.randn(D, generator=g), bias=0.3, mean=torch.zeros(D), std=torch.ones(D), layer=4)


def _parent_scores(practice):
    """float32 values, as the scorer's parent scores are (`scores_for` returns float32)."""
    return [float(np.float32(np.asarray(r).mean(0) @ np.arange(1, D + 1))) for r in practice.rows]


def test_the_registry_is_the_contracts_kind_list():
    """The scorer's `baseline:<kind>` grammar and the fits are one list (the contract's tuple is generated from it)."""
    assert tuple(wb.BASELINE_KINDS) == wsp_scoring.BASELINE_KINDS
    assert all(spec.uses_rows == "practice" for spec in wb.BASELINE_KINDS.values())


def test_every_kind_is_fitted(monkeypatch):
    """Derived from the registry: removing an entry, or a fit that returns nothing storable, is red. (Storage of the
    fitted heads under `baselines/` is 6.6's, with the bundle task.)"""
    from src.services import probe_monitor_trainer

    practice, parent = _practice(), _parent()
    monkeypatch.setattr(probe_monitor_trainer, "train_rule", _fake_train_rule([]))
    # The registry's own fits, as `wsp_bundle.fit_and_store_baselines` runs them (`fit_all` left with OR2-044a-L4: no
    # production caller; the stored path is pinned by `test_wsp_bundle_build.py::test_every_kind_is_fitted_and_stored`).
    fitted = {kind: spec.fit(parent, practice, rule="mean", rule_params={}, parent_scores=_parent_scores(practice),
                             target_fpr=0.1) for kind, spec in wb.BASELINE_KINDS.items()}
    assert list(fitted) == list(wb.BASELINE_KINDS)
    for kind, result in fitted.items():
        spec = wb.BASELINE_KINDS[kind]
        assert result.kind == kind
        assert (result.head is not None) == spec.stores_weights, kind
        assert spec.stores_weights or result.bar is not None, kind


def _fake_train_rule(calls):
    from src.services.probe_monitor_trainer import TrainedRule

    def fake(rule, train_rows, train_labels, val_rows, val_labels, **kw):
        calls.append((rule, list(train_rows), list(train_labels), list(val_rows), list(val_labels), kw))
        return TrainedRule(rule=rule, weight=torch.ones(D), bias=0.0, mean=torch.zeros(D), std=torch.ones(D),
                           attention_query=None, rule_params={}, val_auroc=0.9, epochs_run=3, best_epoch=2)
    return fake


def test_lr_baseline_uses_practice_split_only(monkeypatch):
    from src.services import probe_monitor_trainer

    calls = []
    monkeypatch.setattr(probe_monitor_trainer, "train_rule", _fake_train_rule(calls))
    practice = _practice()
    wb.fit_logistic_regression(_parent(), practice, rule="mean", rule_params={}, seed=7,
                               train_config={"epochs": 5, "not_a_trainer_arg": 1})
    assert len(calls) == 1
    rule, train_rows, train_labels, val_rows, val_labels, kw = calls[0]
    val = practice.validation_mask
    assert [id(r) for r in train_rows] == [id(r) for r, v in zip(practice.rows, val) if not v]
    assert [id(r) for r in val_rows] == [id(r) for r, v in zip(practice.rows, val) if v]
    assert kw["device"] == torch.device("cpu") and kw["seed"] == 7 and kw["epochs"] == 5
    assert "not_a_trainer_arg" not in kw


def test_diff_of_means_recovers_a_planted_direction():
    planted = np.zeros(D)
    planted[5] = 1.0
    parent = _parent()
    fitted = wb.fit_diff_of_means(parent, _practice(n=400, planted=planted))
    w = fitted.head.weight.double().numpy()
    cosine = float(w @ planted / np.linalg.norm(w))
    assert cosine > 0.95, cosine
    assert np.isclose(np.linalg.norm(w), float(torch.linalg.vector_norm(parent.weight.double())))
    assert fitted.head.bias == parent.bias and fitted.head.layer == parent.layer


def test_lda_shrinkage_is_auto(monkeypatch):
    import sklearn.discriminant_analysis as da

    seen = {}
    real = da.LinearDiscriminantAnalysis

    class Spy(real):
        def __init__(self, **kw):
            seen.update(kw)
            super().__init__(**kw)

    monkeypatch.setattr(da, "LinearDiscriminantAnalysis", Spy)
    fitted = wb.fit_lda(_parent(), _practice(n=200))
    assert seen == {"solver": "lsqr", "shrinkage": "auto"}
    assert fitted.head.weight.shape == (D,)


def test_operating_point_baselines_leave_auroc_unchanged():
    practice = _practice()
    scores = _parent_scores(practice)
    recut = wb.fit_threshold_recut(_parent(), practice, parent_scores=scores, target_fpr=0.1)
    affine = wb.fit_affine(_parent(), practice, parent_scores=scores, target_fpr=0.1)
    base = auroc(scores, practice.labels)
    assert auroc(wb.scores_for("threshold_recut", recut.transform, scores).tolist(), practice.labels) == base
    assert auroc(wb.scores_for("affine_recalibration", affine.transform, scores).tolist(),
                 practice.labels) == pytest.approx(base, abs=0)
    assert recut.bar["source"] == "practice" and affine.bar == {**affine.bar, "threshold": 0.0,
                                                                  "source": "practice_logit_zero"}


def test_threshold_recut_cuts_on_practice_negatives_only():
    """R2 review fix (OR2-044a-L1), 2026-11-17: the re-cut bar is cut on the practice VALIDATION negatives that were
    not truncated — the scorer's practice basis and G5 GR-6 — never on training rows or truncated rows (this test used
    to pin every practice negative). The highest validation negative is then marked truncated: it no longer places the
    bar. Controls: cut on every practice negative, or keep truncated rows → red."""
    from src.services.probe_monitor_trainer import calibrate

    target = 0.3
    practice = _practice()
    scores = _parent_scores(practice)
    recut = wb.fit_threshold_recut(_parent(), practice, parent_scores=scores, target_fpr=target)
    rows = [i for i, (y, v) in enumerate(zip(practice.labels, practice.validation_mask)) if y == 0 and v]
    assert 0 < len(rows) < sum(1 for y in practice.labels if y == 0), "fixture: validation is a strict subset"
    assert recut.bar["n_negatives"] == len(rows)
    assert recut.bar["threshold"] == calibrate([scores[i] for i in rows], target_fpr=target, source="x").threshold
    every = calibrate([s for s, y in zip(scores, practice.labels) if y == 0], target_fpr=target, source="x").threshold
    assert recut.bar["threshold"] is not None and every != recut.bar["threshold"], "fixture: the rules must differ"

    top = max(rows, key=lambda i: scores[i])
    cut = wb.Practice(rows=practice.rows, labels=practice.labels, validation_mask=practice.validation_mask,
                      truncated_mask=[i == top for i in range(len(scores))])
    kept = [i for i in rows if i != top]
    expected = calibrate([scores[i] for i in kept], target_fpr=target, source="x").threshold
    assert expected is not None and expected != recut.bar["threshold"], "fixture: the truncated row placed the bar"
    recut2 = wb.fit_threshold_recut(_parent(), cut, parent_scores=scores, target_fpr=target)
    assert (recut2.bar["n_negatives"], recut2.bar["threshold"]) == (len(kept), expected)
    assert wb.fit_affine(_parent(), cut, parent_scores=scores, target_fpr=target).bar["n_negatives"] == len(kept)


def test_affine_refuses_a_decreasing_map():
    practice = _practice()
    flipped = [-s for s in _parent_scores(practice)]
    with pytest.raises(wb.BaselineRefused) as refused:
        wb.fit_affine(_parent(), practice, parent_scores=flipped, target_fpr=0.1)
    assert refused.value.code == "affine_not_increasing"


def test_unaffordable_target_is_reported_not_inf():
    practice = _practice(n=20)
    recut = wb.fit_threshold_recut(_parent(), practice, parent_scores=_parent_scores(practice), target_fpr=0.01)
    assert recut.bar["threshold"] is None and recut.bar["realised_fpr"] == 0.0
