"""Baselines fitted on PRACTICE rows only (ENH-010 044 tasks 6.1 – 6.5; FR-4; D7, P4, P20; FTID §3.4).

A WSP candidate is judged against what a cheap, conventional fix would have done on the same practice rows, so an
"improvement" that any refit or a re-cut bar would match is not read as one (R-4). Five kinds, in one registry
(`BASELINE_KINDS`) the tests derive from — never a hand list:

* `logistic_regression` — the run's own fit (`train_rule`, CPU, the run's config) on practice-train PLUS the practice
  supplement, early-stopped on practice-validation;
* `lda`, `diff_of_means` — on mean-pooled standardised practice-train rows PLUS the practice supplement; `w` rescaled
  to ‖w_parent‖; the parent's `mean`, `std`, `attention_query` and `bias` kept (degenerate channels are zeroed by
  `standardise`, as the head does);
* `threshold_recut` — the parent's scores, its bar re-cut by `calibrate` on the parent's PRACTICE negatives at the
  probe's target FPR (`threshold=None` when the target is unaffordable: reported, never `inf`). The negatives are
  `practice_bar_indices`' — the VALIDATION negatives that were not truncated, the scorer's own practice basis (R2
  review fix OR2-044a-L1, 2026-11-17: it used every practice negative, training rows and truncated rows included,
  against G5 GR-6 and the scorer);
* `affine_recalibration` — `a·s + c` of the parent's score from a 1-D logistic regression on practice; refused
  `affine_not_increasing` when `a ≤ 0` (a decreasing map is not a recalibration); its bar is `a·s + c ≥ 0`.

⚠ THE THREE REFITS SEE EVERY PRACTICE ROW THE EDITOR MAY USE (operator decision, 2026-11-27 about 22:10 ET, locked;
record `0xcc/reviews/wsp_baselines_supplement_2026-11-27.md`). Fitted on practice-train alone, `logistic_regression`
re-ran the parent's own trainer on the parent's own rows with the parent's seed and came out the parent bit for bit
(the phase-2 verdicts: the same `scores_sha256` and bar on every set), so "must beat the LR refit" added nothing to
"must beat the parent". miForge's editor also reads the bundle's practice supplement (FU-7a C2), so LR, LDA and DoM fit
on practice-train AND every supplement row (`Practice.fit_rows`). The supplement is TRAIN-ONLY: practice-validation
stays exactly the parent's validation rows, because it is the early-stopping basis of the LR refit and the bar basis
of a practice-basis probe (`practice_bar_indices`), and a negatives-only supplement there would move both away from
the parent's. The two operating-point kinds (`threshold_recut`, `affine_recalibration`) take the parent's scores on
the practice rows and never the supplement.

**No held-out row reaches any fit** (6.5): every fit takes a `Practice` value, which holds only practice rows and the
practice supplement (checked against every held-out set by the bundle, `heldout_overlaps_practice`), and nothing here
reads a held-out capture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..ml.probe_monitor_model import ProbeHead


class BaselineRefused(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class Practice:
    """The practice partition of a bundle, as the run's token file holds it (FTDD §4.5): per-row (t, d) activation
    blocks, labels, and the validation mask. Nothing else is ever passed to a fit."""

    rows: Sequence[np.ndarray]
    labels: Sequence[int]
    validation_mask: Sequence[bool]
    #: G5: 1 = the row was cut at `max_length`. Empty = none recorded (no row is treated as truncated).
    truncated_mask: Sequence[bool] = ()
    #: The bundle's practice supplement (FU-7a C2): every supplement's rows in `practice.supplements` order, and their
    #: labels. TRAIN-ONLY (module docstring): `split` never returns them; `fit_rows` adds them to practice-train.
    supplement_rows: Sequence[np.ndarray] = ()
    supplement_labels: Sequence[int] = ()

    def __post_init__(self) -> None:
        if not (len(self.rows) == len(self.labels) == len(self.validation_mask)):
            raise ValueError("practice rows, labels and validation mask must align")
        if len(self.truncated_mask) not in (0, len(self.rows)):
            raise ValueError("the practice truncation mask must align with the rows")
        if len(self.supplement_rows) != len(self.supplement_labels):
            raise ValueError("the supplement rows and labels must align")

    def split(self, validation: bool) -> tuple[list, list]:
        idx = [i for i, v in enumerate(self.validation_mask) if bool(v) == validation]
        return [self.rows[i] for i in idx], [int(self.labels[i]) for i in idx]

    def fit_rows(self) -> tuple[List[np.ndarray], List[int]]:
        """What a refit (LR, LDA, DoM) fits on: practice-train, then every supplement row (operator decision
        2026-11-27)."""
        rows, labels = self.split(validation=False)
        return rows + list(self.supplement_rows), labels + [int(y) for y in self.supplement_labels]


@dataclass(frozen=True)
class Fitted:
    """A fitted baseline: a head (weights, for LR/LDA/DoM), a bar (operating-point kinds) or a transform (affine)."""

    kind: str
    head: Optional[ProbeHead] = None
    bar: Optional[Dict[str, Any]] = None
    transform: Optional[Dict[str, float]] = None
    fit: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineSpec:
    kind: str
    #: Every kind fits on practice rows only (6.5).
    uses_rows: str
    #: True for kinds whose head is stored as `baselines/<kind>.safetensors`.
    stores_weights: bool
    fit: Callable[..., Fitted]


def practice_bar_indices(labels: Sequence[int], validation_mask: Sequence[bool],
                         truncated_mask: Sequence[bool] = ()) -> np.ndarray:
    """THE practice rows a bar is cut on (one rule for the scorer's practice basis, the import's derived bar and the
    operating-point baselines): the VALIDATION negatives that were not truncated — the rows a `validation_negatives`
    probe's bar was cut from, less the rows G5 (GR-6) says never place a bar."""
    labels = np.asarray(labels)
    validation = np.asarray(validation_mask).astype(bool)
    truncated = (np.asarray(truncated_mask).astype(bool) if len(truncated_mask)
                 else np.zeros(len(labels), dtype=bool))
    return np.flatnonzero(validation & ~truncated & (labels == 0))


def _pooled(parent: ProbeHead, rows: Sequence[np.ndarray]) -> np.ndarray:
    """Mean-pooled standardised rows, (n, d), float64, by the parent's own statistics."""
    out = []
    for block in rows:
        z = parent.standardise(torch.as_tensor(np.asarray(block), dtype=torch.float32))
        out.append(z.mean(dim=0).double().numpy())
    return np.stack(out) if out else np.zeros((0, int(parent.weight.shape[0])))


def _rescaled(parent: ProbeHead, w: np.ndarray) -> ProbeHead:
    norm = float(np.linalg.norm(w))
    if not np.isfinite(norm) or norm == 0.0:
        raise BaselineRefused("baseline_degenerate", "the fitted direction is zero or not finite")
    target = float(torch.linalg.vector_norm(parent.weight.double()))
    scaled = torch.as_tensor(w * (target / norm), dtype=parent.weight.dtype)
    return ProbeHead(weight=scaled, bias=parent.bias, mean=parent.mean, std=parent.std,
                     attention_query=parent.attention_query, layer=parent.layer, eps=parent.eps)


def fit_diff_of_means(parent: ProbeHead, practice: Practice, **_: Any) -> Fitted:
    rows, labels = practice.fit_rows()
    pooled, y = _pooled(parent, rows), np.asarray(labels)
    if not (y == 1).any() or not (y == 0).any():
        raise BaselineRefused("baseline_one_class", "practice-train holds one class")
    w = pooled[y == 1].mean(axis=0) - pooled[y == 0].mean(axis=0)
    return Fitted("diff_of_means", head=_rescaled(parent, w),
                  fit={"rows": int(len(y)), "supplement_rows": len(practice.supplement_rows)})


def fit_lda(parent: ProbeHead, practice: Practice, **_: Any) -> Fitted:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    rows, labels = practice.fit_rows()
    pooled, y = _pooled(parent, rows), np.asarray(labels)
    if not (y == 1).any() or not (y == 0).any():
        raise BaselineRefused("baseline_one_class", "practice-train holds one class")
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(pooled, y)
    return Fitted("lda", head=_rescaled(parent, np.asarray(lda.coef_[0], dtype=np.float64)),
                  fit={"rows": int(len(y)), "supplement_rows": len(practice.supplement_rows), "solver": "lsqr",
                       "shrinkage": "auto"})


def fit_logistic_regression(parent: ProbeHead, practice: Practice, *, rule: str, rule_params: Dict[str, Any],
                            seed: int = 1337, train_config: Optional[Dict[str, Any]] = None, **_: Any) -> Fitted:
    """The run's own trainer on CPU: practice-train and the supplement to fit, practice-validation to stop early."""
    from .probe_monitor_trainer import train_rule

    train_rows, train_labels = practice.fit_rows()
    val_rows, val_labels = practice.split(validation=True)
    config = {k: v for k, v in (train_config or {}).items()
              if k in ("tau", "window", "epochs", "patience", "lr", "weight_decay", "batch_bytes")}
    config.update({k: rule_params[k] for k in ("tau", "window") if k in rule_params})
    trained = train_rule(rule, train_rows, train_labels, val_rows, val_labels, seed=seed,
                         device=torch.device("cpu"), **config)
    head = ProbeHead(weight=trained.weight.detach().cpu(), bias=float(trained.bias), mean=trained.mean.detach().cpu(),
                     std=trained.std.detach().cpu(),
                     attention_query=None if trained.attention_query is None else trained.attention_query.cpu(),
                     layer=parent.layer, eps=parent.eps)
    return Fitted("logistic_regression", head=head,
                  fit={"rows": len(train_labels), "supplement_rows": len(practice.supplement_rows),
                       "validation_rows": len(val_labels), "seed": seed,
                       "epochs": trained.epochs_run, "best_epoch": trained.best_epoch,
                       "practice_validation_auroc": trained.val_auroc})


def fit_threshold_recut(parent: ProbeHead, practice: Practice, *, parent_scores: Sequence[float], target_fpr: float,
                        **_: Any) -> Fitted:
    """The parent's bar re-cut on its PRACTICE negatives; scores unchanged, so AUROC is unchanged."""
    from .probe_monitor_trainer import calibrate

    labels = np.asarray(practice.labels)
    if len(parent_scores) != len(labels):
        raise ValueError("parent_scores must align with the practice rows")
    rows = practice_bar_indices(labels, practice.validation_mask, practice.truncated_mask)
    if not len(rows):
        raise BaselineRefused("baseline_no_negatives", "no untruncated practice validation negative to cut a bar on")
    negatives = [float(parent_scores[i]) for i in rows]
    cal = calibrate(negatives, target_fpr=target_fpr, source="wsp_practice")
    bar = {"threshold": cal.threshold, "source": "practice", "n_negatives": cal.n_negatives,
           "realised_fpr": cal.realised_fpr, "target_fpr": cal.target_fpr}
    return Fitted("threshold_recut", bar=bar, fit={"rows": int(len(labels)), "bar_rows": int(len(rows))})


def fit_affine(parent: ProbeHead, practice: Practice, *, parent_scores: Sequence[float], target_fpr: float,
               **_: Any) -> Fitted:
    """`a·s + c` from a 1-D logistic regression on the parent's practice scores; bar `a·s + c ≥ 0`."""
    from sklearn.linear_model import LogisticRegression

    s = np.asarray(parent_scores, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(practice.labels)
    if len(s) != len(y):
        raise ValueError("parent_scores must align with the practice rows")
    model = LogisticRegression(C=1e6).fit(s, y)
    a, c = float(model.coef_[0][0]), float(model.intercept_[0])
    if not a > 0.0:
        raise BaselineRefused("affine_not_increasing",
                              f"the fitted map has slope {a}; a decreasing map is not a recalibration")
    transform = {"a": a, "c": c}
    # The realised rate on the rows every practice bar is measured on (OR2-044a-L1), from the scores exactly as the
    # scorer serves them (`scores_for`, float32; OR2-044a-L4: one copy of the map).
    rows = practice_bar_indices(y, practice.validation_mask, practice.truncated_mask)
    mapped = scores_for("affine_recalibration", transform, np.asarray(parent_scores)[rows])
    bar = {"threshold": 0.0, "source": "practice_logit_zero", "n_negatives": int(len(rows)),
           "realised_fpr": float((mapped >= np.float32(0.0)).sum() / max(int(len(rows)), 1)),
           "target_fpr": float(target_fpr)}
    return Fitted("affine_recalibration", transform=transform, bar=bar, fit={"rows": int(len(y))})


#: The registry every test derives from (6.1). Order is the order the bundle lists them in.
BASELINE_KINDS: Dict[str, BaselineSpec] = {
    "logistic_regression": BaselineSpec("logistic_regression", "practice", True, fit_logistic_regression),
    "lda": BaselineSpec("lda", "practice", True, fit_lda),
    "diff_of_means": BaselineSpec("diff_of_means", "practice", True, fit_diff_of_means),
    "threshold_recut": BaselineSpec("threshold_recut", "practice", False, fit_threshold_recut),
    "affine_recalibration": BaselineSpec("affine_recalibration", "practice", False, fit_affine),
}


def scores_for(kind: str, transform: Optional[Dict[str, float]], parent_scores: Sequence[float]) -> np.ndarray:
    """The operating-point kinds' float32 scores from the parent's: the parent's own (`threshold_recut`) or
    `a·s + c` computed in float64 and cast once (`affine_recalibration`). THE one place the affine map is applied: the
    scorer (`wsp_score.score`) and the fit's realised rate both call it (R2 review fix OR2-044a-L4)."""
    parent = np.asarray(parent_scores)
    if kind == "threshold_recut":
        return parent.astype(np.float32, copy=True)
    if kind == "affine_recalibration":
        a, c = float(transform["a"]), float(transform["c"])
        return (a * parent.astype(np.float64) + c).astype(np.float32)
    raise ValueError(f"{kind} is scored through its head, not from the parent's scores")
