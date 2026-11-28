"""The WSP scorer (ENH-010 044 FR-5, FR-6; tasks 7.2 – 7.5; FTDD §5.7.1, FTID §3.5).

miForge proposes heads; miWorkbench scores them on held-out rows miForge never sees and answers per-row scores,
bars and metrics — never an activation and never a row's text (P46; 7.4's leak test pins the exact key sets).

Order of the checks (the first failing one is the answer, each with its code from `schemas/wsp_scoring.REFUSALS`):
bundle (found, ready, digest, kind) → `heldout_ref` exactly the bundle's → head count → each head (tensors where
required and nowhere else; dtype, shape, digest equal to the parent's) → the entries clamp → the captures (the ones
the bundle was sealed on, `capture_changed`; ready; parity passed) → the per-row head budget (P64, P70) under one
advisory lock → scoring → the call record.

⚠ THE BUDGET IS THE ONLY BOUND ON READING A HELD-OUT ROW (FTDD §8). A score is linear in `w`, so a one-entry edit
reads one coordinate of every row's pooled activation. Each DISTINCT tensor-bearing head (by `head_tensors_sha256`)
that reads a row is counted against that row's content key under (model, layer, scope), so a re-registered view, a
re-download or an overlapping re-sample shares it; past `wsp_score_max_heads_per_set`, 409.

Phase 2: tensor-bearing heads are bound by the budget, the full reference and the clamp alone (no miForge client yet;
044-C adds the run binding, 7.2a).
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..core.runtime_settings import effective
from ..schemas.wsp_scoring import REFUSALS, RESPONSE_KIND, carries_tensors, refusal_status

logger = logging.getLogger(__name__)

#: T44-D1: one scoring call at a time per process; a second answers 429 `scorer_busy` at once.
SEMAPHORE = threading.Semaphore(1)
#: `pg_advisory_xact_lock` key of the head budget (one fixed key: an empty table still locks; R3B-17).
WSP_HEAD_BUDGET_LOCK = 0x57535042  # "WSPB"
#: Pause between the budget count and its insert; a test widens it to prove the lock serialises two callers.
_BETWEEN_COUNT_AND_INSERT: Callable[[], None] = lambda: None
#: The clock `wsp_score_timeout_seconds` is measured on (a test moves it).
_NOW: Callable[[], float] = time.monotonic


def check_deadline(deadline: float, where: str) -> None:
    """Past `wsp_score_timeout_seconds`: 503 `score_timeout`, before anything is recorded (the caller rolls back, so no
    call row and no budget row survive: a call that did not answer used nothing)."""
    if _NOW() > deadline:
        raise ScoreRefused("score_timeout", f"the call was still scoring ({where}) after wsp_score_timeout_seconds = "
                                            f"{effective().wsp_score_timeout_seconds}; nothing was recorded")


class ScoreAbandoned(Exception):
    """The caller hung up (the route's disconnect watch set the call's stop event): the call stops at the next bucket
    and records nothing. Not a refusal: nobody is left to read an answer, so it has no code in the contract."""


def checkpoint(deadline: float, stop: Optional[threading.Event], where: str) -> None:
    """Called before EVERY bucket (`_aggregate_rows_many(between_buckets=…)`) and before the record: an abandoned call
    stops (`ScoreAbandoned`), a late one is refused `score_timeout`. So a call ends at `wsp_score_timeout_seconds` plus
    at most one bucket, never after a whole set (2026-11-22: D128's everyday_text set alone is 39 GB, and the deadline
    was checked only after each set)."""
    if stop is not None and stop.is_set():
        raise ScoreAbandoned(f"the caller disconnected; stopped before {where}, nothing recorded")
    check_deadline(deadline, where)


@contextlib.contextmanager
def scorer_threads(bound: Optional[int] = None) -> Iterator[int]:
    """Bound torch's intra-op threads to `wsp_score_torch_threads` (at most the CPUs this process may use) while a call
    scores, and put the previous count back after. `bound`, when given, is used exactly instead: the WSP import passes
    the bound a score call RECORDED (`arithmetic.scorer_threads`, operator decision 2026-11-24), so its re-scoring runs
    at the scorer's count even if the setting or the pod's CPU set changed since. The scorer runs inside the API process: unbounded, D128 held about
    9.4 cores of it for an hour (2026-11-22). The pack threads take the same width. The matmul does not take it: it
    runs on one MKL thread whatever this bound is (B5a, 2026-11-24: MKL's Intel kernels change gemv's bits with the
    thread count; `packed_scoring.one_blas_thread`). The rules' reductions are torch's own and give the same bits at
    any thread count except a sum over one row longer than 32,768 tokens, which torch splits across threads: so the
    WSP import re-scores under this same bound (`wsp_import.import_candidate`)."""
    import torch

    wanted = int(bound) if bound is not None else max(1, min(int(effective().wsp_score_torch_threads),
                                                             _usable_cpus()))
    before = torch.get_num_threads()
    torch.set_num_threads(wanted)
    try:
        yield wanted
    finally:
        torch.set_num_threads(before)


#: The fields of a score call's arithmetic record (`wsp_score_calls.arithmetic`; B5a follow-up, operator 2026-11-24):
#: what decides a score's bits beyond the call's own inputs. P23 is bit for bit, and bit equality holds only within one
#: MKL kernel family (B5a: a score made on MKL's `def` kernels, an AMD host, does not reproduce on Intel AVX-512) and,
#: for a rule's sum over one row longer than 32,768 tokens, one torch thread count. So the import, when P23 fails,
#: names which of these differs instead of answering a bare `persistence_mismatch`. The import re-scores AT a recorded
#: `scorer_threads` (operator decision 2026-11-24), so that field differs only in the comparison's own unit tests.
ARITHMETIC_FIELDS = ("mkl_kernel_family", "torch_version", "scorer_threads", "numpy_version")


def arithmetic_record(threads: int) -> Dict[str, Any]:
    """This process's arithmetic, scoring at `threads` torch threads (the bound `scorer_threads` yields). A field this
    host cannot answer is None, never a guess (`packed_scoring.mkl_kernel_family`)."""
    import torch

    from .packed_scoring import mkl_kernel_family

    return {"mkl_kernel_family": mkl_kernel_family(), "torch_version": str(torch.__version__),
            "scorer_threads": int(threads), "numpy_version": str(np.__version__)}


def _usable_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


class ScoreRefused(Exception):
    def __init__(self, code: str, message: str, **extra: Any):
        if code not in REFUSALS:   # every code the scorer answers is in the published table (7.6)
            raise KeyError(f"{code} is not in the scoring contract's refusal table")
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra
        self.status = refusal_status(code)

    def detail(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.extra}


@dataclass
class Head:
    label: str
    arrays: Dict[str, np.ndarray]
    #: `head_tensors_sha256` of a tensor-bearing head; None for `parent` / `baseline:*` (they cost nothing).
    digest: Optional[str] = None


def _parent_arrays(bundle: Any) -> Dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    return dict(load_file(str(Path(bundle.dir_path) / "heads.safetensors")))


def decode_head(head: Any, parent: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """A candidate/control head's arrays, checked against the parent's names, dtypes, shapes and digests."""
    from .wsp_patch import tensor_sha256

    names = [t.name for t in head.tensors]
    if sorted(names) != sorted(parent) or len(set(names)) != len(names):
        raise ScoreRefused("head_shape_mismatch", f"head {head.label} carries {sorted(names)}; the parent has "
                                                  f"{sorted(parent)}")
    out = {}
    for tensor in head.tensors:
        expected = parent[tensor.name]
        raw = base64.b64decode(tensor.data_b64, validate=True)
        if (tensor.dtype != str(expected.dtype) or list(tensor.shape) != list(expected.shape)
                or len(raw) != expected.nbytes):
            raise ScoreRefused("head_shape_mismatch",
                               f"head {head.label} tensor {tensor.name} is {tensor.dtype} {tensor.shape} "
                               f"({len(raw)} bytes); the parent's is {expected.dtype} {list(expected.shape)}")
        array = np.frombuffer(raw, dtype=np.dtype(tensor.dtype).newbyteorder("<")).reshape(expected.shape).copy()
        if tensor_sha256(array) != tensor.sha256:
            raise ScoreRefused("head_shape_mismatch", f"head {head.label} tensor {tensor.name}: sha256 does not "
                                                      f"match its bytes")
        out[tensor.name] = array
    return out


def check_clamp(label: str, arrays: Dict[str, np.ndarray], parent: Dict[str, np.ndarray],
                editable: Dict[str, bool], limit: Optional[int]) -> int:
    """A head may differ from the parent only in editable tensors, in at most `limit` entries (P56). Entries are
    compared by their BITS (a NaN, a signed zero), never by float equality."""
    changed = 0
    for name, array in arrays.items():
        unsigned = np.dtype(f"u{array.dtype.itemsize}")
        differs = int(np.count_nonzero(array.view(unsigned) != parent[name].view(unsigned)))
        if differs and not editable.get(name, False):
            raise ScoreRefused("candidate_outside_patch_limits",
                               f"head {label} changes {name}, which the bundle marks not editable")
        changed += differs
    if limit is not None and changed > limit:
        raise ScoreRefused("candidate_outside_patch_limits",
                           f"head {label} changes {changed} entries; the limit is {limit}")
    return changed


def entries_limit(run_ref: Any, parent: Dict[str, np.ndarray], editable: Dict[str, bool]) -> int:
    """`min(run_ref.max_changed_entries, ceiling)`; a null takes the ceiling; the ceiling defaults to every editable
    entry of the parent (`wsp_score_max_changed_entries_ceiling` empty)."""
    ceiling = effective().wsp_score_max_changed_entries_ceiling
    if ceiling is None:
        ceiling = int(sum(a.size for n, a in parent.items() if editable.get(n, False)))
    asked = run_ref.max_changed_entries
    return int(ceiling if asked is None else min(int(asked), int(ceiling)))


def to_probe_head(arrays: Dict[str, np.ndarray], layer: int) -> Any:
    import torch

    from ..ml.probe_monitor_model import ProbeHead

    def tensor(name):
        return torch.from_numpy(np.ascontiguousarray(arrays[name])) if name in arrays else None

    return ProbeHead(weight=tensor("weight"), bias=float(arrays["bias"].reshape(-1)[0]), mean=tensor("mean"),
                     std=tensor("std"), attention_query=tensor("attention_query"), layer=layer)


# ── captures and their rows ───────────────────────────────────────────────────


@dataclass
class CapturedRows:
    capture: Any
    name: str
    positions: np.ndarray
    labels: np.ndarray
    truncated: np.ndarray
    rows: List[np.ndarray]
    content_keys: List[str]


def load_set(db: Any, capture: Any, name: str) -> CapturedRows:
    from ..models.probe_monitor import ProbeMonitorDataset
    from .wsp_bundle import row_keys

    directory = Path(capture.dir_path)
    offsets = np.load(directory / "offsets.npy")
    tokens = next(directory.glob("tokens.f*"))
    memmap = np.memmap(tokens, dtype=np.dtype(capture.dtype), mode="r", shape=(int(capture.tokens),
                                                                                 int(capture.d_model)))
    positions = np.load(directory / "positions.npy")
    view = db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id == capture.dataset_id).one()
    return CapturedRows(capture=capture, name=name, positions=positions, labels=np.load(directory / "labels.npy"),
                        truncated=np.load(directory / "truncated.npy"),
                        rows=[memmap[int(offsets[i]):int(offsets[i + 1])] for i in range(len(offsets) - 1)],
                        content_keys=row_keys(db, view, positions).text)


def practice_validation_negatives(bundle: Any) -> List[np.ndarray]:
    """The bundle's practice VALIDATION negatives that were not truncated (`wsp_baselines.practice_bar_indices`): the
    rows a `validation_negatives` probe's bar was cut from (FTDD §5.7.4's `threshold_source` mapping), less the rows
    G5 (GR-6) says never place a bar."""
    from .wsp_baselines import practice_bar_indices

    directory = Path(bundle.dir_path)
    practice = bundle.manifest["practice"]
    offsets = np.load(directory / "practice_offsets.npy")
    labels = np.load(directory / "practice_labels.npy")
    validation = np.load(directory / "practice_validation_mask.npy")
    truncated = np.load(directory / "practice_truncated_mask.npy")
    name = next(f["name"] for f in bundle.manifest["files"] if f["role"] == "practice_activations")
    memmap = np.memmap(directory.parents[2] / name, dtype=np.dtype(practice["dtype"]), mode="r",
                       shape=(int(practice["tokens"]), int(practice["d_model"])))
    keep = practice_bar_indices(labels, validation, truncated)
    return [memmap[int(offsets[i]):int(offsets[i + 1])] for i in keep]


# ── the budget (P64, P70) ─────────────────────────────────────────────────────


def check_budget(db: Any, *, model: str, layer: int, scope: str, sets: Sequence[CapturedRows],
                 heads: Sequence[Head], call_id: str, ceiling: int) -> List[Dict[str, Any]]:
    """Take the budget's advisory lock (held until the transaction ends), count every (row content key,
    tensor-bearing head) this call would add, and refuse 409 before anything is scored. Returns the rows to record
    (`record_budget`, after the call row exists)."""
    from sqlalchemy import text

    from ..models.wsp import WspHeldoutRowHead

    counted = [h for h in heads if h.digest is not None]
    if not counted:
        return []
    db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": WSP_HEAD_BUDGET_LOCK})
    new_digests = {h.digest for h in counted}
    for one in sets:
        keys = sorted(set(one.content_keys))
        existing: Dict[str, set] = {}
        for key, digest in db.query(WspHeldoutRowHead.row_content_key, WspHeldoutRowHead.head_tensors_sha256).filter(
                WspHeldoutRowHead.model == model, WspHeldoutRowHead.layer == layer,
                WspHeldoutRowHead.scope == scope, WspHeldoutRowHead.row_content_key.in_(keys)).all():
            existing.setdefault(key, set()).add(digest)
        over = [k for k in keys if len(existing.get(k, set()) | new_digests) > ceiling]
        if over:
            at_ceiling = sum(1 for k in one.content_keys if len(existing.get(k, set())) >= ceiling)
            raise ScoreRefused(
                "heldout_set_head_budget_exhausted",
                f"{len(over)} row(s) of set {one.name} would be read by more than {ceiling} distinct tensor-bearing "
                f"heads (wsp_score_max_heads_per_set). The way forward is a capture over DISJOINT rows.",
                set=one.name, row_ids_sha256=one.capture.row_ids_sha256, rows_exhausted=int(at_ceiling))
    _BETWEEN_COUNT_AND_INSERT()
    return [{"model": model, "layer": layer, "scope": scope, "row_content_key": key,
             "head_tensors_sha256": h.digest, "label": h.label, "first_call_id": call_id}
            for one in sets for key in sorted(set(one.content_keys)) for h in counted]


def record_budget(db: Any, values: List[Dict[str, Any]]) -> None:
    """Insert the counted rows (`ON CONFLICT DO NOTHING`: a head that already read a row costs nothing again)."""
    from sqlalchemy.dialects.postgresql import insert

    from ..models.wsp import WspHeldoutRowHead

    for start in range(0, len(values), 5000):
        db.execute(insert(WspHeldoutRowHead).values(values[start:start + 5000]).on_conflict_do_nothing())


# ── bars and metrics ──────────────────────────────────────────────────────────


def _f32(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def cut_bar(scores: np.ndarray, target_fpr: float, source: str) -> Dict[str, Any]:
    from .probe_monitor_trainer import calibrate

    cal = calibrate([float(s) for s in scores], target_fpr=target_fpr, source=source)
    return {"threshold": None if cal.threshold is None else float(cal.threshold), "source": source,
            "n_negatives": int(cal.n_negatives), "realised_fpr": float(cal.realised_fpr),
            "target_fpr": float(cal.target_fpr)}


def cut_calibration_bar(scores: np.ndarray, calibration: "CapturedRows", target_fpr: float) -> Dict[str, Any]:
    """A head's bar on the calibration capture: its UNTRUNCATED negatives (FR-1, G5), from that head's float32 scores
    over every row of the capture."""
    keep = (calibration.truncated == 0) & (calibration.labels == 0)
    return cut_bar(scores[keep], target_fpr, "calibration")


def cut_practice_bar(head: Any, negatives: Sequence[np.ndarray], rule: str, params: Dict[str, Any],
                     target_fpr: float, between_buckets: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    """A head's bar on the practice basis: `negatives` are `practice_validation_negatives` (validation, untruncated)."""
    from .probe_monitor_run import _aggregate_rows_many

    return cut_bar(_f32(_aggregate_rows_many([head], rule, params, negatives, between_buckets=between_buckets)[0]),
                   target_fpr, "practice")


def bar_rows(db: Any, bundle: Any, sets: Sequence["CapturedRows"]) -> Optional["CapturedRows"]:
    """The calibration set every head's bar is cut on, or None for the practice basis. R1 review fix (OR1-044a-1): the
    basis is the MANIFEST's, never "whichever set is present" — a calibration-basis bundle without its calibration set
    is refused, and a practice-basis bundle ignores any calibration capture it holds."""
    from .wsp_bundle import bar_basis

    calibration = next((s for s in sets if s.capture.role == "calibration"), None)
    basis = bar_basis(db, bundle)
    if basis == "calibration" and calibration is None:
        raise ScoreRefused("capture_not_ready", "the bundle's bars are cut on its calibration set (operating_point."
                                                "threshold_source = calibration), and no calibration capture is in it")
    return calibration if basis == "calibration" else None


def scorer_parent_bar(db: Any, bundle: Any) -> Dict[str, Any]:
    """FR-1: the parent's bar AS THE SCORER CUTS IT (the same helpers `score` uses, on the same rows), beside the bar
    AS SERVED (`operating_point.threshold`) and their difference. They differ when the served bar was cut on rows the
    scorer leaves out: on the practice basis the served validation-negatives bar keeps truncated negatives (G5 has no
    mask on that path), the scorer drops them (GR-6). Stored on the bundle at build (`wsp_bundles.scorer_parent_bar`).
    R2 review fix OR2-044a-M3, 2026-11-17."""
    from ..models.wsp import WspActivationCapture
    from .probe_monitor_run import _aggregate_rows_many
    from .wsp_bundle import bar_target_fpr, heldout_set_name

    manifest = bundle.manifest
    function = manifest["function"]["linear_probe"]
    layer, rule = int(function["layer"]), function["rule"]
    params = {k: v for k, v in function["params"].items() if v is not None}
    parent = to_probe_head(_parent_arrays(bundle), layer)
    target = bar_target_fpr(db, bundle)
    captures = db.query(WspActivationCapture).filter(WspActivationCapture.id.in_(list(bundle.capture_ids or []))).all()
    sets = [load_set(db, c, heldout_set_name(c)) for c in captures if c.role == "calibration"]
    calibration = bar_rows(db, bundle, sets)
    if calibration is not None:
        bar = cut_calibration_bar(_f32(_aggregate_rows_many([parent], rule, params, calibration.rows)[0]),
                                  calibration, target)
    else:
        bar = cut_practice_bar(parent, practice_validation_negatives(bundle), rule, params, target)
    point = manifest.get("operating_point") or {}
    served = point.get("threshold")
    scorer = bar["threshold"]
    return {"scorer": bar, "served": {"threshold": served, "threshold_source": point.get("threshold_source")},
            "difference": (None if scorer is None or served is None else float(scorer) - float(served)),
            "same": bool((scorer is None and served is None)
                         or (scorer is not None and served is not None
                             and np.float32(scorer) == np.float32(served)))}


def scores_sha256(scores: np.ndarray) -> str:
    """sha256 of the float32 little-endian scores in row order: miForge's per-head `scores_sha256` (P23)."""
    return hashlib.sha256(np.asarray(scores, dtype="<f4").tobytes()).hexdigest()


def identical_to_parent(values: Dict[str, np.ndarray], keep: np.ndarray, labels: Sequence[str]) -> Dict[str, bool]:
    """Per `baseline:<kind>` label: are its scores on the scored rows the parent's (equal `scores_sha256`)? Operator
    decision 2026-11-27: the phase-2 LR refit was the parent bit for bit, and a verdict must say so, not count it as
    a separate bar the candidate beat."""
    parent = scores_sha256(values["parent"][keep])
    return {label: scores_sha256(values[label][keep]) == parent for label in labels if label.startswith("baseline:")}


def metrics_of(scores: np.ndarray, labels: np.ndarray, threshold: Optional[float]) -> Dict[str, Any]:
    """From exactly the float32 scores the response carries (R1B-X3): AUROC by `probe_monitor_metrics.auroc`, recall a
    ratio and false alarms a COUNT at `score >= threshold` (a None bar fires on nothing)."""
    from .probe_monitor_metrics import auroc

    values = [float(s) for s in scores]
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    fires = np.zeros(len(values), dtype=bool) if threshold is None else (scores >= np.float32(threshold))
    return {"auroc": auroc(values, labels.astype(int).tolist()),
            "recall_at_bar": (float(fires[labels == 1].sum()) / positives) if positives else None,
            "false_alarms_at_bar": int(fires[labels == 0].sum()),
            "negatives": negatives, "positives": positives}


# ── the call ──────────────────────────────────────────────────────────────────


def _bundle(db: Any, request: Any) -> Any:
    from ..models.wsp import WspBundle

    bundle = db.query(WspBundle).filter(WspBundle.id == request.bundle_id).first()
    if bundle is None:
        raise ScoreRefused("bundle_not_found", f"no bundle {request.bundle_id}")
    if bundle.status != "ready" or not bundle.manifest:
        raise ScoreRefused("bundle_not_ready", f"bundle {bundle.id} is {bundle.status}")
    if bundle.bundle_sha256 != request.bundle_sha256:
        raise ScoreRefused("bundle_sha256_mismatch", "bundle_sha256 is not this bundle's")
    if bundle.bundle_kind != "linear_probe":
        raise ScoreRefused("bundle_kind_unsupported", f"bundle kind {bundle.bundle_kind} is scored from phase 4")
    return bundle


def resolve_heads(db: Any, request: Any, bundle: Any, parent: Dict[str, np.ndarray]) -> List[Head]:
    from ..models.wsp import WspBaseline
    from .wsp_digests import head_tensors_sha256

    editable = {t["name"]: bool(t["editable"]) for t in bundle.manifest["tensors"]}
    limit = None
    heads: List[Head] = []
    for head in request.heads:
        if not carries_tensors(head.label):
            if head.tensors:
                raise ScoreRefused("parent_tensors_not_accepted",
                                   f"head {head.label} carries tensors; parent and baselines are the bundle's own")
            if head.label == "parent":
                heads.append(Head("parent", parent))
                continue
            kind = head.label.split(":", 1)[1]
            row = db.query(WspBaseline).filter(WspBaseline.bundle_id == bundle.id, WspBaseline.kind == kind).first()
            if row is None or (row.fit or {}).get("pending") or (row.fit or {}).get("refused"):
                raise ScoreRefused("baseline_not_available",
                                   f"the bundle has no fitted {kind} baseline (pending, refused or never fitted)")
            heads.append(Head(head.label, {"__baseline__": np.zeros(0)}))
            continue
        if not head.tensors:
            raise ScoreRefused("head_tensors_required", f"head {head.label} carries no tensors")
        arrays = decode_head(head, parent)
        if limit is None:
            limit = entries_limit(request.run_ref, parent, editable)
        check_clamp(head.label, arrays, parent, editable, limit)
        heads.append(Head(head.label, arrays, digest=head_tensors_sha256(arrays)))
    return heads


def score(db: Any, request: Any, *, caller: Optional[Dict[str, Any]] = None,
          stop: Optional[threading.Event] = None) -> Dict[str, Any]:
    """One scoring call, start to record. The caller holds `SEMAPHORE`. `stop` is set when the caller has gone (the
    route's disconnect watch): the call then ends before its next bucket with `ScoreAbandoned`, recording nothing."""
    from ..core.clock import utc_now
    from ..models.wsp import WspActivationCapture, WspBaseline, WspScoreCall, _wsc_id
    from .probe_monitor_run import _aggregate_rows_many
    from .wsp_baselines import scores_for
    from .wsp_bundle import bar_target_fpr, capture_seal_broken, heldout_set_name, load_baseline_head
    from .wsp_digests import row_ids_sha256
    from .wsp_patch import jcs_sha256

    deadline = _NOW() + float(effective().wsp_score_timeout_seconds)
    bundle = _bundle(db, request)
    manifest = bundle.manifest
    request_ref = request.heldout_ref.model_dump(mode="json")
    if request_ref != manifest["heldout_ref"]:
        raise ScoreRefused("heldout_ref_mismatch",
                           "heldout_ref must equal the bundle's exactly (every set, in order); a subset is refused")
    if len(request.heads) > int(effective().wsp_score_max_heads):
        raise ScoreRefused("too_many_heads", f"{len(request.heads)} heads; at most "
                                             f"{effective().wsp_score_max_heads} (wsp_score_max_heads)")
    parent = _parent_arrays(bundle)
    heads = resolve_heads(db, request, bundle, parent)

    captures = {c.id: c for c in db.query(WspActivationCapture).filter(
        WspActivationCapture.id.in_(list(bundle.capture_ids or []))).all()}
    by_name = {heldout_set_name(c): c for c in captures.values()}
    sets: List[CapturedRows] = []
    for entry in manifest["heldout_ref"]["sets"]:
        capture = by_name.get(entry["name"])
        if capture is None:
            raise ScoreRefused("capture_not_ready", f"the capture of set {entry['name']} is not ready")
        # R4 review fix (C044a-M1): only the capture the bundle was sealed on is ever scored.
        broken = capture_seal_broken(bundle, entry, capture)
        if broken:
            raise ScoreRefused("capture_changed", broken)
        if capture.status not in ("ready", "parity_failed"):
            raise ScoreRefused("capture_not_ready", f"the capture of set {entry['name']} is not ready")
        if capture.status == "parity_failed":
            raise ScoreRefused("capture_parity_failed", f"the capture of set {entry['name']} failed parity")
        sets.append(load_set(db, capture, entry["name"]))

    function = manifest["function"]["linear_probe"]
    layer, scope = int(function["layer"]), str(function["scope"])
    rule, params = function["rule"], {k: v for k, v in function["params"].items() if v is not None}
    call_id = _wsc_id()
    ceiling = int(effective().wsp_score_max_heads_per_set)
    budget = check_budget(db, model=manifest["identity"]["model"], layer=layer, scope=scope, sets=sets, heads=heads,
                          call_id=call_id, ceiling=ceiling)

    # Scoring: every weight-bearing head on the same packed rows (`_aggregate_rows_many`), cast to float32 FIRST.
    baselines = {b.kind: b for b in db.query(WspBaseline).filter(WspBaseline.bundle_id == bundle.id).all()}
    weighted: Dict[str, Any] = {}
    for head in heads:
        if head.label.startswith("baseline:"):
            kind = head.label.split(":", 1)[1]
            if baselines[kind].weights_path:
                weighted[head.label] = load_baseline_head(Path(baselines[kind].weights_path), layer)
        else:
            weighted[head.label] = to_probe_head(head.arrays, layer)
    if "parent" not in weighted:
        weighted["parent"] = to_probe_head(parent, layer)   # the operating-point baselines are the parent's scores
    order = list(weighted)
    per_set: Dict[str, Dict[str, np.ndarray]] = {}
    # The thread bound covers every matmul of the call; the checkpoint runs before EVERY bucket (2026-11-22).
    with scorer_threads() as threads:
        for one in sets:
            scored = _aggregate_rows_many(
                [weighted[k] for k in order], rule, params, one.rows,
                between_buckets=lambda where=f"set {one.name}": checkpoint(deadline, stop, where))
            values = {label: _f32(s) for label, s in zip(order, scored)}
            for head in heads:
                if head.label in ("baseline:threshold_recut", "baseline:affine_recalibration"):
                    kind = head.label.split(":", 1)[1]
                    values[head.label] = scores_for(kind, baselines[kind].transform, values["parent"])
            per_set[one.name] = values
            checkpoint(deadline, stop, f"set {one.name}")

        # Bars: cut ONCE per head, written into every set (P29), at the probe's own target (OR2-044a-L5).
        target = bar_target_fpr(db, bundle)
        calibration = bar_rows(db, bundle, sets)   # OR1-044a-1: the manifest's basis
        labels_out = [h.label for h in heads]
        bars: Dict[str, Dict[str, Any]] = {}
        practice_negatives: Optional[List[np.ndarray]] = None
        for label in labels_out:
            if label in ("baseline:threshold_recut", "baseline:affine_recalibration"):
                stored = dict(baselines[label.split(":", 1)[1]].bar)
                bars[label] = {"threshold": stored["threshold"], "source": stored["source"],
                               "n_negatives": int(stored["n_negatives"]), "realised_fpr": float(stored["realised_fpr"]),
                               "target_fpr": float(stored["target_fpr"])}
            elif calibration is not None:
                bars[label] = cut_calibration_bar(per_set[calibration.name][label], calibration, target)
            else:
                # A probe calibrated on its practice validation negatives (OVN-wsp-workbench-26): every head's bar is
                # cut on the same practice rows — the validation negatives that were NOT truncated (G5 GR-6). The parent's
                # served bar was cut on every validation negative (that path records no mask), so where a validation
                # negative was truncated the scorer's parent bar differs from `operating_point.threshold`; the bundle
                # records both and the difference (`scorer_parent_bar`, FR-1; R2 review fix OR2-044a-M3).
                if practice_negatives is None:
                    practice_negatives = practice_validation_negatives(bundle)
                bars[label] = cut_practice_bar(weighted[label if label in weighted else "parent"], practice_negatives,
                                               rule, params, target,
                                               lambda: checkpoint(deadline, stop, "the practice bars"))
    checkpoint(deadline, stop, "bars")

    response_sets = []
    for one in sets:
        keep = one.truncated == 0
        entry = {
            "name": one.name, "view_id": one.capture.dataset_id, "role": one.capture.role,
            "row_ids_sha256": one.capture.row_ids_sha256,
            "scored_row_ids_sha256": row_ids_sha256(one.capture.dataset_id, one.positions[keep]),
            "rows": int(len(one.positions)),
            "excluded": {"truncated": int((~keep).sum()), "truncated_fired": {
                label: int(0 if bars[label]["threshold"] is None else
                           (per_set[one.name][label][~keep] >= np.float32(bars[label]["threshold"])).sum())
                for label in labels_out}},
            "bars": {label: bars[label] for label in labels_out},
        }
        if one.capture.role != "calibration":
            labels = one.labels[keep].astype(np.int8)
            entry["labels"] = [int(y) for y in labels]
            entry["scores"] = {label: [float(v) for v in per_set[one.name][label][keep]] for label in labels_out}
            entry["metrics"] = {label: metrics_of(per_set[one.name][label][keep], labels, bars[label]["threshold"])
                                for label in labels_out}
            if request.report_identical_to_parent:   # opt-in: an answer nobody asked it of stays the old shape
                entry["identical_to_parent"] = identical_to_parent(per_set[one.name], keep, labels_out)
        response_sets.append(entry)

    response = {"kind": RESPONSE_KIND, "bundle_sha256": bundle.bundle_sha256,
                "heldout_ref_sha256": jcs_sha256(request_ref),
                "function": {"rule": rule, "params": dict(function["params"]), "layer": layer, "scope": scope,
                             "skip_template_prefix": bool(function["skip_template_prefix"])},
                "sets": response_sets, "scored_at": utc_now().isoformat().replace("+00:00", "Z"), "call_id": call_id}
    candidate = next((h for h in heads if h.label == "candidate"), None)
    candidate_scores = None
    if candidate is not None:
        digest = hashlib.sha256()
        for entry in response_sets:
            if "scores" in entry:
                digest.update(np.asarray(entry["scores"]["candidate"], dtype=np.float32).tobytes())
        candidate_scores = digest.hexdigest()
    db.add(WspScoreCall(
        id=call_id, bundle_id=bundle.id, heldout_ref_sha256=response["heldout_ref_sha256"], labels=labels_out,
        miforge_run_id=request.run_ref.miforge_run_id, patch_sha256=request.run_ref.patch_sha256,
        candidate_tensors_sha256=candidate.digest if candidate is not None else None,
        heads=[{"label": h.label, "head_tensors_sha256": h.digest} for h in heads if h.digest is not None],
        caller=caller or {}, request_sha256=jcs_sha256(request.model_dump(mode="json")),
        response_sha256=jcs_sha256(response), candidate_scores_sha256=candidate_scores, status="scored",
        arithmetic=arithmetic_record(threads)))
    db.flush()
    record_budget(db, budget)
    db.commit()   # releases the budget's advisory lock
    return response
