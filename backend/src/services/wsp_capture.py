"""Held-out capture for weight-space programming (ENH-010 044 FR-1; tasks 4.1, 4.2; FTDD §6, FTID §3.2).

miForge never sees a held-out row. miWorkbench captures each held-out set's token activations at the probe's layer,
ONCE, under the run's own render and the stored evaluation's own sampling, so the scorer can score any head on exactly
the rows the stored evaluation scored. A capture is usable only when it REPRODUCES that evaluation (parity), and the
parity check is itself checked: two controls that must break it (a permuted weight and rows rolled by one against
their labels). A capture whose controls do not fail proves nothing and is `parity_failed`.

The sets (`plan_capture`): for each held-out view the probe was evaluated on (`eval`, `everyday_text`,
`nonsense_text`), the NEWEST `completed`/`refused` evaluation with recorded positions; and, for a probe whose bar was
cut on a calibration set, that set's recorded calibration rows. A probe calibrated on its practice validation
negatives has no calibration capture: its bar is cut on practice (`operating_point.threshold_source = practice`).

Parity (`check_parity`), per set, over the rows the stored evaluation counts (its truncation mask, G5):
* eval: AUROC within `AUROC_TOLERANCE` of the AUROC RECOMPUTED from the stored scores under the same mask — never
  the stored all-rows `metrics.auroc` — and the positives caught at the shipped bar, exactly;
* everyday / nonsense text: the false alarms at the shipped bar, exactly;
* calibration (over ALL its rows, FR-1): `calibrate` over the stored negatives the probe's bar was cut on (its own G5
  mask, `calibration_truncated_path`; every row when none is recorded) reproduces `probe.threshold`; the capture puts
  exactly as many of those rows at or above `probe.threshold` as the stored array does; and the stored threshold row
  still supplies the capture's own bar (`calibrate`'s order statistic). R2 review fix OR2-044a-M1: the first two are
  what ties the capture to the bar AS SERVED — without them a capture passed whatever `probe.threshold` was;
* every set: each row's score within `ROW_ATOL + ROW_RTOL·|stored|` of the stored one, and the per-row scored-token
  counts equal (OVN-wsp-workbench-25: without a per-row check a negatives-only set's count is invariant under the
  rolled-rows control, so that control could never fail).
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HELDOUT_ROLES = ("eval", "everyday_text", "nonsense_text")
#: ENH-010 FU-7a C2: a `supplement` view is captured in the same job as the held-out sets, with the same parity (the
#: false alarms at the shipped bar and every row's score, as on a negatives-only text set), from an evaluation that
#: never counts towards the rung. It goes into a bundle only as `practice.supplements`, never into `heldout_ref`.
CAPTURE_ROLES = HELDOUT_ROLES + ("supplement",)
AUROC_TOLERANCE = 1e-4
ROW_ATOL = 1e-3
ROW_RTOL = 1e-3
#: The permutation of the permuted-weight control is seeded, so a parity record reproduces.
CONTROL_SEED = 0


class CaptureRefused(Exception):
    """A capture that cannot be planned: `{code, message}`, answered 409 by the route (FTDD §5.3)."""

    def __init__(self, code: str, message: str, status: int = 409, **extra: Any):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        #: Figures the refusal carries beside its words (the GPU check's need and cards).
        self.extra = extra

    def detail(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.extra}


@dataclass
class PlannedSet:
    """One held-out set to capture: which rows (stored positions, in stored order) and what they scored."""

    view_id: str
    role: str
    evaluation_id: Optional[str]
    positions: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    truncated: Optional[np.ndarray]
    n_scored: Optional[np.ndarray] = None
    #: A sampled evaluation's size (its seed is the view and this size, `sample_seed_for`); None = the whole view.
    sample_rows: Optional[int] = None

    @property
    def calibration(self) -> bool:
        return self.role == "calibration"


def _load(path: Optional[str]) -> Optional[np.ndarray]:
    from ..core.config import settings

    if not path:
        return None
    resolved = Path(settings.resolve_data_path(path))
    return np.load(resolved, allow_pickle=False) if resolved.is_file() else None


def plan_capture(db: Any, probe: Any, dataset_ids: Optional[Sequence[str]] = None) -> List[PlannedSet]:
    """The held-out sets of `probe` (optionally only `dataset_ids`), each with its stored rows. Refuses
    `evaluation_positions_not_recorded` / `calibration_positions_not_recorded` rather than guessing rows."""
    from ..models.probe_monitor import ProbeMonitorDataset, ProbeMonitorEvaluation

    wanted = set(dataset_ids or ())
    evaluations = (
        db.query(ProbeMonitorEvaluation)
        .filter(ProbeMonitorEvaluation.probe_id == probe.id,
                ProbeMonitorEvaluation.status.in_(("completed", "refused")))
        .order_by(ProbeMonitorEvaluation.created_at.desc())
        .all()
    )
    newest: Dict[str, Any] = {}
    for evaluation in evaluations:
        newest.setdefault(evaluation.dataset_id, evaluation)
    views = {v.id: v for v in db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id.in_(list(newest))).all()} \
        if newest else {}
    calibration_id = None
    if probe.threshold_source == "calibration_set":
        from ..models.probe_monitor import ProbeMonitorRun

        run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).first()
        # The run's set, else the probe's own (a recalibrated probe), as `recut_probe_windows_on_gpu` reads it.
        calibration_id = getattr(run, "calibration_dataset_id", None) or probe.calibration_dataset_id
        if not calibration_id:
            raise CaptureRefused("calibration_positions_not_recorded",
                                 f"probe {probe.id} records a calibration-set bar but names no calibration set.")

    planned: List[PlannedSet] = []
    for dataset_id, evaluation in newest.items():
        view = views.get(dataset_id)
        if view is None or view.role not in CAPTURE_ROLES or (wanted and dataset_id not in wanted):
            continue
        positions, labels, scores = (_load(evaluation.positions_path), _load(evaluation.labels_path),
                                     _load(evaluation.scores_path))
        if positions is None or labels is None or scores is None:
            raise CaptureRefused(
                "evaluation_positions_not_recorded",
                f"evaluation {evaluation.id} of view {dataset_id} has no recorded positions, labels or scores; the "
                f"rows it scored cannot be identified. Re-evaluate the probe on this view first.")
        sample = (evaluation.metrics or {}).get("sample") or {}
        planned.append(PlannedSet(
            view_id=dataset_id, role=view.role, evaluation_id=evaluation.id,
            positions=positions.astype(np.int64), labels=labels.astype(np.int8), scores=scores.astype(np.float32),
            truncated=_load(evaluation.truncated_path), n_scored=_load(evaluation.n_scored_path),
            sample_rows=int(sample["rows"]) if sample.get("rows") else None))

    missing = wanted - {p.view_id for p in planned} - ({calibration_id} if calibration_id else set())
    if missing:
        raise CaptureRefused(
            "evaluation_positions_not_recorded",
            f"no recorded held-out evaluation of this probe on {sorted(missing)}; evaluate it there first.")
    if calibration_id and (not wanted or calibration_id in wanted):
        positions, scores = _load(probe.calibration_positions_path), _load(probe.calibration_scores_path)
        if positions is None or scores is None or len(positions) != len(scores):
            raise CaptureRefused(
                "calibration_positions_not_recorded",
                f"probe {probe.id} was calibrated on view {calibration_id} but its calibration rows' positions are "
                f"not recorded (or not aligned with its calibration scores); recalibrate the probe first.")
        planned.append(PlannedSet(
            view_id=calibration_id, role="calibration", evaluation_id=None,
            positions=positions.astype(np.int64), labels=np.zeros(len(positions), dtype=np.int8),
            scores=scores.astype(np.float32), truncated=_load(probe.calibration_truncated_path)))
    if not planned:
        raise CaptureRefused(
            "evaluation_positions_not_recorded",
            f"probe {probe.id} has no held-out evaluation with recorded rows; evaluate it on a held-out view first.")
    return planned


#: The refusal a capture answers when no card it may run on has the memory its model needs now (2026-11-19: a capture
#: started on a full card and every set failed CUDA out of memory; nothing said so before it was queued).
GPU_MEMORY_INSUFFICIENT = "gpu_memory_insufficient"


def capture_gpu_check(db: Any, probe: Any) -> Dict[str, Any]:
    """Can the capture's model be loaded on a card NOW? Read at request time, before anything is queued.

    The capture loads the probe run's model at the row's own quantization (`probe_monitor_run._load_model_for_run`)
    on ONE card, the run's `gpu_request` ("auto": the card with the most free memory). The need is the loader's own
    figure (`resource_config.load_need_gb`: weights at the quantization plus the activation headroom), so this refuses
    exactly what the loader would refuse, only before the job is queued instead of after it starts.

    Answers `{checked, fits, required_mb, requested, card, cards, message, model_id, quantization}`. `checked: false`
    (no GPU visible to this process, or no parameter count on the model row) is NOT a no: the capture goes ahead as
    before and the loader checks again when it starts. Every card is read from NVML at this moment (1+n cards;
    `gpu_placement.list_cards`), so another app's memory on a card (miServer's resident model) counts."""
    from ..models.model import Model
    from ..models.probe_monitor import ProbeMonitorRun
    from . import gpu_placement
    from .base_model_budget import params_for_sizing
    from .resource_config import load_need_gb

    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).first()
    model = db.query(Model).filter(Model.id == run.model_id).first() if run is not None else None
    requested = (getattr(run, "gpu_request", None) or "auto") if run is not None else "auto"
    quantization = getattr(getattr(model, "quantization", None), "value", getattr(model, "quantization", None))
    out: Dict[str, Any] = {"checked": False, "fits": None, "required_mb": None, "requested": requested,
                           "card": None, "cards": [], "model_id": getattr(run, "model_id", None),
                           "quantization": quantization, "message": None}
    params = params_for_sizing(getattr(model, "params_count", None), quantization,
                               getattr(model, "architecture_config", None)) if model is not None else None
    if params is None:
        out["message"] = "the model row has no parameter count, so its memory need is unknown; not checked"
        return out
    _weights_gb, needed_gb = load_need_gb(params, quantization)
    required_mb = needed_gb * 1024
    out["required_mb"] = int(round(required_mb))
    cards = gpu_placement.list_cards()
    out["cards"] = [card.to_dict() for card in cards]
    if not cards:
        out["message"] = "no GPU is visible to the backend, so free memory cannot be read; not checked"
        return out
    out["checked"] = True
    try:
        card = gpu_placement.resolve_card(requested, required_mb=required_mb, cards=cards)
    except gpu_placement.GpuPlacementError as exc:
        out["fits"] = False
        out["message"] = (f"{exc} The capture loads model {out['model_id']} at {quantization} on one card "
                          f"(~{out['required_mb']:,} MB with headroom). Free memory on a card (another job, or "
                          "miServer's resident model), then request the capture again.")
        return out
    out["fits"] = True
    out["card"] = card.to_dict()
    out["message"] = (f"{card.describe()} can take it: the capture needs ~{out['required_mb']:,} MB "
                      f"(model {out['model_id']} at {quantization}, with headroom).")
    return out


def refuse_if_gpu_too_full(check: Dict[str, Any]) -> None:
    """Raise `CaptureRefused(gpu_memory_insufficient)` (409) when `capture_gpu_check` read the cards and none fits."""
    if check.get("checked") and check.get("fits") is False:
        raise CaptureRefused(GPU_MEMORY_INSUFFICIENT, check["message"], required_mb=check["required_mb"],
                             requested=check["requested"], cards=check["cards"])


def render_planned(db: Any, context: Any, tokenizer: Any, planned: PlannedSet) -> List[Any]:
    """The planned set's rendered rows, in STORED order: re-rendered with the evaluation's own sampling, kept to the
    stored positions. A stored position that no longer renders refuses (the render drifted since the evaluation)."""
    from ..models.probe_monitor import ProbeMonitorDataset
    from .probe_monitor_run import (
        LEGACY_CALIBRATION_MAX_ROWS,
        _prepare_examples,
        preparation_role,
        sample_seed_for,
    )

    view = db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id == planned.view_id).first()
    if view is None:
        raise CaptureRefused("evaluation_positions_not_recorded", f"view {planned.view_id} is gone")
    if planned.calibration:
        recorded = context.config.get("calibration_max_rows")
        kwargs = {"role": "calibration", "max_rows": int(recorded) if recorded else LEGACY_CALIBRATION_MAX_ROWS}
    else:
        rows = planned.sample_rows
        kwargs = {"role": preparation_role(view), "max_rows": rows,
                  "sample_seed": sample_seed_for(view.id, rows) if rows else None}
    examples, rendered, _summary, labels = _prepare_examples(db, view, context, tokenizer, **kwargs)
    by_position = {int(example.index): (render, int(label))
                   for example, render, label in zip(examples, rendered, labels)}
    absent = [int(p) for p in planned.positions if int(p) not in by_position]
    if absent:
        code = "calibration_positions_not_recorded" if planned.calibration else "evaluation_positions_not_recorded"
        raise CaptureRefused(code, f"{len(absent)} stored position(s) of view {view.id} no longer render "
                                   f"(first: {absent[:5]}); the view or its render changed since it was scored.")
    relabelled = [p for p, y in zip(planned.positions, planned.labels) if by_position[int(p)][1] != int(y)]
    if relabelled:
        raise CaptureRefused("evaluation_positions_not_recorded",
                             f"{len(relabelled)} stored row(s) of view {view.id} now map to another label.")
    return [by_position[int(p)][0] for p in planned.positions]


@dataclass
class CapturedSet:
    """The files of one capture (FTDD §4.4)."""

    tokens_path: Path
    offsets: np.ndarray
    d_model: int
    dtype: str
    truncated: np.ndarray
    files: Dict[str, Path] = field(default_factory=dict)

    def rows(self) -> List[np.ndarray]:
        total = int(self.offsets[-1])
        memmap = np.memmap(self.tokens_path, dtype=np.dtype(self.dtype), mode="r", shape=(total, self.d_model))
        return [memmap[int(self.offsets[i]):int(self.offsets[i + 1])] for i in range(len(self.offsets) - 1)]


def write_capture(directory: Path, rendered: Sequence[Any], planned: PlannedSet, model: Any, *, layer: int,
                  scope: str, architecture: str, pad_id: int,
                  progress: Optional[Callable[[int, int], None]] = None) -> CapturedSet:
    """Capture the rows' token activations at `layer` and write the five arrays beside them."""
    from .probe_monitor_capture import capture_tokens

    directory.mkdir(parents=True, exist_ok=True)
    capture = capture_tokens(model, rendered, layer, directory / "tokens.bin", scope=scope,
                             architecture=architecture, pad_id=pad_id, progress=progress)
    suffix = ".f16" if np.dtype(capture.dtype) == np.float16 else ".f32"
    tokens = directory / f"tokens{suffix}"
    Path(capture.path).replace(tokens)
    truncated = np.asarray([1 if r.truncated else 0 for r in rendered], dtype=np.int8)
    files = {"tokens": tokens}
    for name, values, dtype in (("offsets", capture.offsets, np.int64), ("positions", planned.positions, np.int64),
                                ("labels", planned.labels, np.int8),
                                ("n_scored", np.diff(capture.offsets), np.int32), ("truncated", truncated, np.int8)):
        files[name] = directory / f"{name}.npy"
        np.save(files[name], np.asarray(values, dtype=dtype))
    return CapturedSet(tokens, np.asarray(capture.offsets), int(capture.d_model), str(np.dtype(capture.dtype)),
                       truncated, files)


def _row_agreement(capture: np.ndarray, stored: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    deviation = np.abs(capture[mask].astype(np.float64) - stored[mask].astype(np.float64))
    allowed = ROW_ATOL + ROW_RTOL * np.abs(stored[mask].astype(np.float64))
    return {"max_deviation": float(deviation.max()) if deviation.size else 0.0,
            "rows_outside_tolerance": int((deviation > allowed).sum())}


def _statistics(scores: np.ndarray, planned: PlannedSet, mask: np.ndarray, probe: Any) -> Dict[str, Any]:
    """The parity statistics of one score vector against the stored set, over `mask`; `passed` is their AND."""
    from .probe_monitor_metrics import auroc
    from .probe_monitor_rung import everyday_false_alarms, shipped_bar_recall
    from .probe_monitor_trainer import calibrate

    out: Dict[str, Any] = _row_agreement(scores, planned.scores, mask)
    checks = [out["rows_outside_tolerance"] == 0]
    labels, stored = planned.labels[mask], planned.scores[mask]
    if planned.role == "eval":
        out["auroc_stored"] = auroc(stored.tolist(), labels.tolist())
        out["auroc_capture"] = auroc(scores[mask].tolist(), labels.tolist())
        both = out["auroc_stored"] is not None and out["auroc_capture"] is not None
        checks.append(both and abs(out["auroc_stored"] - out["auroc_capture"]) <= AUROC_TOLERANCE)
        count = lambda s: (shipped_bar_recall(s.tolist(), labels.tolist(), probe.threshold) or {}).get("caught")  # noqa: E731
    elif planned.role == "calibration":
        out.update(calibration_bar(scores, planned, probe))
        checks.append(bool(out["bar_reproduced"]))
        count = None
    else:
        count = lambda s: (everyday_false_alarms(s.tolist(), labels.tolist(), probe.threshold) or {}).get(  # noqa: E731
            "false_alarms")
    if count is not None:
        out["count_stored"], out["count_capture"] = count(stored), count(scores[mask])
        checks.append(out["count_stored"] == out["count_capture"])
    out["passed"] = bool(all(checks))
    return out


def calibration_arrays_sha256(positions: Any, scores: Any, truncated: Any) -> str:
    """The digest of a calibration basis as `plan_capture` reads it (positions int64, scores float32, the G5 mask int8;
    an absent mask is named as absent), recorded on a calibration capture's parity (`calibration_arrays_sha256`)."""
    import hashlib

    digest = hashlib.sha256(b"miworkbench.wsp-calibration-basis/1")
    for name, values, dtype in (("positions", positions, "<i8"), ("scores", scores, "<f4"), ("truncated", truncated, "i1")):
        digest.update(f"|{name}|".encode())
        if values is None:
            digest.update(b"absent")
            continue
        array = np.ascontiguousarray(np.asarray(values).reshape(-1).astype(dtype))
        digest.update(f"{array.size}|".encode() + array.tobytes())
    return digest.hexdigest()


def calibration_capture_stale(probe: Any, capture: Any) -> Optional[str]:
    """Why a READY calibration capture no longer describes the probe's bar, or None (R3 review fix OR3-044a-L3,
    2026-11-17). Its parity was checked once, against the bar and the calibration rows of that day; a recalibration or
    G5's GPU re-cut since then rewrites `probe.threshold` or the calibration arrays, and every later bundle would carry
    the old capture. Compared: the recorded `threshold_served` with `probe.threshold` (as float32, the stored dtype),
    and the recorded `calibration_arrays_sha256` with the arrays the probe names now. A capture recorded before the
    digest existed is stale (it cannot be shown current)."""
    if getattr(capture, "role", None) != "calibration" or getattr(capture, "status", None) != "ready":
        return None
    parity = capture.parity or {}
    recorded, served = parity.get("threshold_served"), probe.threshold
    if (recorded is None) != (served is None) or (
            served is not None and np.float32(float(served)) != np.float32(float(recorded))):
        return f"the probe's bar is {served}; this calibration capture was checked against {recorded}"
    now = calibration_arrays_sha256(_load(probe.calibration_positions_path), _load(probe.calibration_scores_path),
                                    _load(probe.calibration_truncated_path))
    if parity.get("calibration_arrays_sha256") != now:
        return "the probe's calibration rows changed since this capture was checked (recalibrated or re-cut)"
    return None


def calibration_bar(scores: np.ndarray, planned: PlannedSet, probe: Any) -> Dict[str, Any]:
    """FR-1's calibration parity (R2B-19; R2 review fix OR2-044a-M1, 2026-11-17), over the rows the probe's bar was cut
    on: its own G5 mask (`planned.truncated`, from `calibration_truncated_path`), else every row (a legacy all-rows
    bar). Three facts, ANDed into `bar_reproduced`:

    * `stored_reproduces_the_threshold`: `calibrate(stored rows, probe.target_fpr)` IS `probe.threshold` (compared as
      float32, the dtype the stored array holds). A re-cut in place, a stale `calibration_scores_path` or a threshold
      edited by hand fails here — the bar the scorer would cut is then not the bar miServer serves;
    * `count_capture == count_stored`: the capture puts as many of those rows at or above `probe.threshold` as the
      stored array does ("`probe.threshold`'s count at the target FPR");
    * `threshold_row_holds_the_bar`: the stored threshold row still supplies the capture's own bar."""
    from .probe_monitor_trainer import calibrate

    target = float(probe.target_fpr)
    rows = (np.asarray(planned.truncated, dtype=np.int8) == 0) if planned.truncated is not None \
        else np.ones(len(planned.scores), dtype=bool)
    stored, captured = planned.scores[rows], scores[rows]
    allowed = int(target * len(stored))
    order = np.argsort(-stored.astype(np.float64), kind="stable")
    bar = calibrate(captured.tolist(), target_fpr=target, source="wsp_parity").threshold if len(stored) else None
    holds = (bar is None and allowed <= 0) or (
        bar is not None and allowed > 0 and float(captured[order[allowed - 1]]) == float(bar))
    served = None if probe.threshold is None else float(probe.threshold)
    recut = calibrate(stored.tolist(), target_fpr=target, source="wsp_parity").threshold if len(stored) else None
    reproduces = (served is None and recut is None) or (
        served is not None and recut is not None and np.float32(served) == np.float32(recut))
    bar32 = None if served is None else np.float32(served)
    count = (lambda values: 0 if bar32 is None else int((values >= bar32).sum()))  # noqa: E731
    out = {"bar_rows": int(rows.sum()), "threshold_served": served,
           "threshold_recut_from_stored": None if recut is None else float(recut),
           "stored_reproduces_the_threshold": bool(reproduces), "threshold_row_holds_the_bar": bool(holds),
           "count_stored": count(stored), "count_capture": count(captured)}
    out["bar_reproduced"] = bool(reproduces and holds and out["count_stored"] == out["count_capture"])
    return out


def check_parity(head: Any, trained: Any, captured: CapturedSet, planned: PlannedSet, probe: Any) -> Dict[str, Any]:
    """Score the capture with the parent and compare with the stored set; then require both controls to FAIL."""
    from .probe_monitor_run import _aggregate_rows

    rows = captured.rows()
    if len(rows) != len(planned.scores):
        return {"passed": False, "reason": "row_count_differs", "rows_capture": len(rows),
                "rows_stored": int(len(planned.scores))}
    mask_source = "stored" if planned.truncated is not None else "capture"
    truncated = planned.truncated if planned.truncated is not None else captured.truncated
    if planned.truncated is not None and not np.array_equal(planned.truncated.astype(np.int8), captured.truncated):
        return {"passed": False, "reason": "truncation_differs"}
    if planned.n_scored is not None and not np.array_equal(planned.n_scored.astype(np.int64),
                                                           np.diff(captured.offsets).astype(np.int64)):
        return {"passed": False, "reason": "scored_tokens_differ"}
    mask = np.asarray(truncated, dtype=np.int8) == 0
    if planned.calibration:
        # FR-1: the calibration capture's parity is over ALL its rows (the set the shipped bar was cut on, so a legacy
        # all-rows `probe.threshold` reproduces); the bar's own rows are the probe's mask (`calibration_bar`).
        mask = np.ones(len(planned.scores), dtype=bool)

    def score(with_head):
        return np.asarray(_aggregate_rows(with_head, trained.rule, dict(trained.rule_params or {}), rows),
                          dtype=np.float32)

    parent = score(head)
    result = _statistics(parent, planned, mask, probe)
    permutation = np.random.default_rng(CONTROL_SEED).permutation(int(head.weight.shape[0]))
    permuted = dataclasses.replace(head, weight=head.weight[permutation])
    controls = {
        "permuted_w": not _statistics(score(permuted), planned, mask, probe)["passed"],
        "rolled_rows": not _statistics(np.roll(parent, 1), planned, mask, probe)["passed"],
    }
    result.update({"mask_source": mask_source, "rows": int(mask.size), "rows_counted": int(mask.sum()),
                   "controls": controls})
    if result["passed"] and not all(controls.values()):
        result.update({"passed": False, "reason": "controls_did_not_fail"})
    elif not result["passed"]:
        result["reason"] = "scores_differ"
    return result


class CaptureNotPending(Exception):
    """The row is no longer this task's (the janitor reaped it, or another task took it): the capture stops and writes
    nothing to the row (R4 review fix C044a-L1, the capture twin of `wsp_bundle.BundleNotBuilding`)."""

    def __init__(self, capture_id: str, status: Optional[str]):
        super().__init__(f"capture {capture_id} is {status!r}, not this task's")
        self.status = status


def claim_capture(db: Any, capture_id: str, *, from_status: str) -> bool:
    """`status = running`, `heartbeat_at = now` on the row only while it is `from_status`, in one statement; False
    (rolled back, nothing written) when it is not. `pending` claims the row at the start of its capture; `running` is
    each heartbeat. As `wsp_bundle.claim_building` (C044a-L1)."""
    from sqlalchemy import update

    from ..core.clock import utc_now
    from ..models.wsp import WspActivationCapture

    done = db.execute(update(WspActivationCapture)
                      .where(WspActivationCapture.id == capture_id, WspActivationCapture.status == from_status)
                      .values(status="running", heartbeat_at=utc_now()).execution_options(synchronize_session=False))
    if not done.rowcount:
        db.rollback()
        return False
    db.commit()
    return True


def _capture_status(db: Any, capture_id: str, *, lock: bool = False) -> Optional[str]:
    from ..models.wsp import WspActivationCapture

    query = db.query(WspActivationCapture.status).filter(WspActivationCapture.id == capture_id)
    return (query.with_for_update() if lock else query).scalar()


def capture_one(db: Any, capture: Any, planned: PlannedSet, *, model: Any, tokenizer: Any, architecture: str,
                beat: Optional[Callable[[], None]] = None, progress: Optional[Any] = None) -> Any:
    """Run one planned set into its row: `pending` → `running` → `ready` | `parity_failed` (files kept for the record).

    R4 review fix (C044a-L1), 2026-11-17: the row is CLAIMED, as a bundle is (`claim_capture`): at the start only
    while it is still `pending`, at every heartbeat only while it is `running`, and before the final write its status
    is read under `FOR UPDATE`. A row the janitor reaped while it waited behind the task's earlier sets (or while it
    ran) raises `CaptureNotPending` and is left exactly as the janitor wrote it."""
    from ..core.clock import utc_now
    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorRun
    from .probe_monitor_run import _pad_id, context_from_row, load_probe
    from .wsp_digests import row_ids_sha256
    from .wsp_layout import capture_dir

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == capture.probe_id).one()
    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).one()
    context = context_from_row(run)
    if not claim_capture(db, capture.id, from_status="pending"):
        raise CaptureNotPending(capture.id, _capture_status(db, capture.id))
    db.refresh(capture)
    rendered = render_planned(db, context, tokenizer, planned)

    def on_batch(done, total):
        if not claim_capture(db, capture.id, from_status="running"):
            raise CaptureNotPending(capture.id, _capture_status(db, capture.id))
        if beat is not None:
            beat()
        if progress is not None:
            progress(done, total)

    if progress is not None:
        # The rows of this set, for `ViewRowsProgress` (2026-11-27): the batch loop reports them here.
        on_batch.rows = progress.rows  # type: ignore[attr-defined]
    captured = write_capture(capture_dir(run.id, capture.id), rendered, planned, model, layer=probe.layer,
                             scope=context.scope, architecture=architecture, pad_id=_pad_id(tokenizer),
                             progress=on_batch)
    head, trained = load_probe(db, probe.id)
    parity = check_parity(head, trained, captured, planned, probe)
    if planned.calibration:   # OR3-044a-L3: what this capture was checked against, so a later re-cut supersedes it
        parity["calibration_arrays_sha256"] = calibration_arrays_sha256(planned.positions, planned.scores,
                                                                        planned.truncated)
    # C044a-L1: the row is still ours, read under its lock, so the janitor cannot reap it between here and the commit.
    status = _capture_status(db, capture.id, lock=True)
    if status != "running":
        raise CaptureNotPending(capture.id, status)
    capture.rows, capture.tokens = int(len(planned.positions)), int(captured.offsets[-1])
    capture.d_model, capture.dtype = captured.d_model, captured.dtype
    capture.dir_path = str(captured.tokens_path.parent)
    capture.row_ids_sha256 = row_ids_sha256(planned.view_id, planned.positions)
    capture.bytes = int(sum(p.stat().st_size for p in captured.files.values()))
    capture.parity = parity
    capture.status = "ready" if parity["passed"] else "parity_failed"
    capture.finished_at = utc_now()
    db.commit()
    return capture


def run_captures(db: Any, probe_id: str, capture_ids: Sequence[str], *,
                 model_loader: Optional[Callable[[Any], Any]] = None,
                 progress: Optional[Any] = None) -> Dict[str, Any]:
    """The task body: load the run's model once, then capture each row's set. A set that fails is `failed` with its
    reason; the others still run. Refuses work that would mix precisions or renders into the probe's evidence.

    `progress` (a `live_progress.ViewRowsProgress`, from the task) is told the plan — each set's stored rows, exact —
    before the model loads, the card(s) it was placed on, and each set and batch as it is captured (2026-11-27)."""
    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorRun
    from ..models.wsp import WspActivationCapture
    from .probe_monitor_run import _load_model_for_run, placed_cards, require_matching_precision, require_served_render

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).one()
    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).one()
    listed = db.query(WspActivationCapture).filter(WspActivationCapture.id.in_(list(capture_ids)),
                                                   WspActivationCapture.probe_id == probe_id).all()
    outcome: Dict[str, Any] = {"probe_id": probe_id, "captures": {}}
    # R3 review fix (OR3-044a-L2, the bundle's twin): only a row still `pending` is captured. A row the janitor failed
    # `never_started` while its message waited in the GPU queue, or one already captured, is left exactly as it is.
    rows = [row for row in listed if row.status == "pending"]
    for row in listed:
        if row.status != "pending":
            logger.warning("WSP capture %s not run: it is %s, not pending", row.id, row.status)
            outcome["captures"][row.id] = f"skipped:{row.status}"
    if not rows:
        return outcome
    try:
        require_matching_precision(db, run)
        require_served_render(run, probe)
        planned = {(p.view_id, p.role): p for p in plan_capture(db, probe, sorted({r.dataset_id for r in rows}))}
        if progress is not None:
            progress.plan(capture_progress_plan(db, rows, planned), exact=True)
            progress.begin()
        loaded = (model_loader or _load_model_for_run)(run)
        model, tokenizer, architecture = loaded
        if progress is not None:
            # THIS capture's card(s), for its own task-queue row; the run row keeps its training card.
            progress.placed(*placed_cards(loaded))
    except Exception as exc:  # noqa: BLE001 - every row is told why, then the task fails
        db.rollback()
        for row in rows:
            _fail_if_ours(db, row.id, str(exc))
        raise
    for index, row in enumerate(rows):
        plan = planned.get((row.dataset_id, row.role))
        if progress is not None:
            progress.segment(index, len(rows), "")
        try:
            if plan is None or (plan.evaluation_id or None) != (row.source_evaluation_id or None):
                raise CaptureRefused("evaluation_positions_not_recorded",
                                     "the evaluation this capture was planned from is no longer the newest")
            capture_one(db, row, plan, model=model, tokenizer=tokenizer, architecture=architecture,
                        progress=progress)
        except CaptureNotPending as gone:   # C044a-L1: reaped (or taken) since the task started; left as it is
            db.rollback()
            logger.warning("WSP capture %s not run: %s", row.id, gone)
            outcome["captures"][row.id] = f"skipped:{gone.status}"
            continue
        except Exception as exc:  # noqa: BLE001 - one set's failure is recorded on its row
            db.rollback()
            _fail_if_ours(db, row.id, str(exc))
            logger.exception("WSP capture %s failed", row.id)
        db.refresh(row)
        outcome["captures"][row.id] = row.status
    if progress is not None:
        progress.finish()
    return outcome


def capture_progress_plan(db: Any, rows: Sequence[Any], planned: Dict[Any, PlannedSet]) -> List[Tuple[str, str, int]]:
    """`[(view_id, name, rows)]` for the capture's live progress, one per capture row in run order: each set's
    STORED positions, so the plan is exact. A row with no plan (it will be refused) captures 0 rows."""
    from ..models.probe_monitor import ProbeMonitorDataset

    ids = sorted({row.dataset_id for row in rows})
    names = {v.id: v.name for v in db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id.in_(ids)).all()} \
        if ids else {}
    out: List[Tuple[str, str, int]] = []
    for row in rows:
        plan = planned.get((row.dataset_id, row.role))
        label = names.get(row.dataset_id, row.dataset_id)
        if row.role == "calibration":
            label = f"{label} (calibration)"
        out.append((row.dataset_id, label, int(len(plan.positions)) if plan is not None else 0))
    return out


def _fail_if_ours(db: Any, capture_id: str, error: str) -> None:
    """`failed` with the task's reason, only on a row still `pending` or `running` (C044a-L1): a row the janitor
    reaped keeps the janitor's reason."""
    from sqlalchemy import update

    from ..core.clock import utc_now
    from ..models.wsp import WspActivationCapture

    db.execute(update(WspActivationCapture)
               .where(WspActivationCapture.id == capture_id, WspActivationCapture.status.in_(("pending", "running")))
               .values(status="failed", error=error[:2000], finished_at=utc_now())
               .execution_options(synchronize_session=False))
    db.commit()
