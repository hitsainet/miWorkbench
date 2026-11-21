"""Import a verdicted WSP candidate as a NEW probe (ENH-010 044 FR-8; tasks 8.2 – 8.6; FTID §3.6).

miForge proposes a patch; miWorkbench decides whether it becomes a probe. Nothing miForge sends is trusted: the run
document is validated against the vendored schema and its `run_sha256` recomputed (P65), the patch is applied here
with numpy in miForge's published forms (`wsp_patch`, R1B-X4), and every edited tensor's hash is reproduced.

**Order of the checks** (FPRD FR-8; the first failing one is the answer, and nothing is written before the last):
1. schemas — the run document (`run_document_invalid`), then the patch manifest and its two digests
   (`patch_malformed`, `patch_hash_mismatch`); a run with no verdict (`run_not_evaluated`); the run's `run_sha256`
   recomputed from the document itself (`run_sha256_mismatch`); the patch is the run's (`patch_hash_mismatch`);
2. the bundle — one of this app's ready bundles, and the patch's `bundle_base_sha256` is that bundle's
   (`patch_base_mismatch`);
3. the verdict — `no_change` (or no candidate) → `nothing_to_import` (P22); anything but `improved` without an
   acknowledgement naming that verdict → `acknowledgement_required`;
4. the chain — a chained patch only on top of the imported patch it names, from the same bundle
   (`patch_chain_unresolved`, P21);
5. the patch — applied to the stored parent tensors; base hashes, edited hashes, change counts (`wsp_patch`);
6. the limits — frozen and non-editable tensors and the run's `max_changed_entries` (`patch_outside_limits`);
7. P23's reference — the score call of THIS run whose candidate is exactly the tensors applied here
   (`persistence_reference_missing`; found by run id AND `candidate_tensors_sha256`, R1A-W8).

Then: the new probe (rung 0, never selectable, window and length-band bars `null`), P23's bit-equality on the
re-loaded head (`persistence_mismatch` rolls the import back: never "imported but different"), the bar cut on the
calibration capture, an evaluation per held-out set from its capture (`metrics.source: "wsp_capture"`, the five G5
arrays) and `recompute_rung`.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Refusal code → HTTP status. FTDD §5.3's codes, plus `run_document_invalid` and `run_not_evaluated`
#: (OVN-wsp-workbench-34: neither of §5.3's names says truthfully what is wrong with such a document).
REFUSALS: Dict[str, int] = {
    "run_document_invalid": 422,
    "patch_malformed": 422,
    "patch_hash_mismatch": 422,
    "run_not_evaluated": 409,
    "run_sha256_mismatch": 422,
    "patch_base_mismatch": 409,
    "nothing_to_import": 409,
    "acknowledgement_required": 409,
    "patch_chain_unresolved": 409,
    "patch_outside_limits": 422,
    "persistence_reference_missing": 409,
    "capture_not_ready": 409,
    "capture_changed": 409,
    "persistence_mismatch": 500,
}

#: R1A-W6: what a derived probe's bars are, in the words every surface shows.
BARS_NOTE = "global bar only; window and length-band bars were not re-cut"
#: `metrics.source` of an evaluation computed from WSP captures.
CAPTURE_SOURCE = "wsp_capture"
RUN_KIND = "miforge.wsp-run/v1"


class ImportRefused(Exception):
    def __init__(self, code: str, message: str, **extra: Any):
        if code not in REFUSALS:
            raise KeyError(f"{code} is not an import refusal")
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra
        self.status = REFUSALS[code]

    def detail(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.extra}


# ── the run document ──────────────────────────────────────────────────────────


def _run_schema() -> dict:
    from importlib import resources

    return json.loads(resources.files("src.schemas.vendored").joinpath("miforge-wsp-run-v1.json").read_text("utf-8"))


def run_sha256_of(document: dict) -> str:
    """miForge's `run_sha256` (P63, P65): RFC 8785 over the document WITHOUT `run_sha256` and `program_reruns`."""
    from .wsp_patch import jcs_sha256

    return jcs_sha256({k: v for k, v in document.items() if k not in ("run_sha256", "program_reruns")})


def validate_run_document(document: Any) -> dict:
    from jsonschema.exceptions import best_match
    from jsonschema.validators import validator_for

    schema = _run_schema()
    error = best_match(validator_for(schema)(schema).iter_errors(document))
    if error is not None:
        path = "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in error.absolute_path)
        raise ImportRefused("run_document_invalid", f"the run document is invalid at {path}: {error.message}",
                            path=path)
    return document


def decode_arrays(edit_npz_b64: str) -> bytes:
    try:
        return base64.b64decode(edit_npz_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImportRefused("patch_malformed", f"edit_npz_b64 is not base64: {exc}") from exc


# ── the checks (no writes) ────────────────────────────────────────────────────


@dataclass
class Checked:
    run: dict
    manifest: dict
    bundle: Any
    parent: Any
    #: The probe the patch applies to: the bundle's own probe, or the earlier derived probe a chained patch names.
    base_probe: Any
    edited: Dict[str, Any]
    call: Any
    candidate_digest: str
    heldout_uses: Dict[str, int] = field(default_factory=dict)


def _head_tensors(probe: Any) -> Dict[str, Any]:
    """A stored probe head as WSP tensors (bits), `bias` float64 `[1]` read exactly from the metadata."""
    from .wsp_bundle import head_arrays
    from .wsp_patch import from_array

    return {name: from_array(array) for name, array in head_arrays(probe).items()}


def _bundle_heads(bundle: Any) -> Dict[str, Any]:
    from .wsp_patch import read_safetensors

    return read_safetensors((Path(bundle.dir_path) / "heads.safetensors").read_bytes())


def _arrays(tensors: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """WSP tensors (bits) as arrays of their float dtype (float16/32/64; a probe head holds no bfloat16)."""
    out = {}
    for name, tensor in tensors.items():
        out[name] = tensor.bits.view({"float16": np.float16, "float32": np.float32,
                                      "float64": np.float64}[tensor.dtype])
    return out


def check(db: Any, *, run_document: Any, patch_manifest: Any, edit_npz: bytes,
          acknowledgement: Optional[Dict[str, Any]] = None) -> Checked:
    """FR-8's checks in order; raises `ImportRefused` (or answers what an import would write)."""
    from ..models.probe_monitor import ProbeMonitor
    from ..models.wsp import WspBundle, WspScoreCall
    from .wsp_digests import head_tensors_sha256
    from .wsp_patch import WspPatchRefused, apply_patch, check_limits, validate_manifest

    # 1. schemas and digests
    run = validate_run_document(run_document)
    try:
        manifest = validate_manifest(patch_manifest)
    except WspPatchRefused as refused:
        raise ImportRefused(refused.code, str(refused), **refused.details) from refused
    verdict = run.get("verdict")
    if verdict is None or run.get("run_sha256") is None:
        raise ImportRefused("run_not_evaluated", f"run {run['run_id']} has no verdict; only an evaluated candidate "
                                                 "is imported")
    if run_sha256_of(run) != run["run_sha256"]:
        raise ImportRefused("run_sha256_mismatch",
                            "run_sha256 does not reproduce from the document (RFC 8785, without run_sha256 and "
                            "program_reruns): the document was changed after its verdict")
    if run.get("patch_sha256") != manifest["patch_sha256"] or run.get("content_sha256") != manifest["content_sha256"]:
        raise ImportRefused("patch_hash_mismatch", f"the patch is not run {run['run_id']}'s (its patch_sha256 or "
                                                   "content_sha256 differs)")

    # 2. the bundle
    ref = run["request"]["bundle"]
    bundle = None
    if ref["owner"] == "miworkbench":
        bundle = db.query(WspBundle).filter(WspBundle.id == ref["bundle_id"],
                                            WspBundle.bundle_sha256 == ref["bundle_sha256"]).first()
    if bundle is None or bundle.status != "ready" or not bundle.manifest:
        raise ImportRefused("patch_base_mismatch", f"the run's bundle {ref['bundle_id']} is not a ready bundle of "
                                                   "this miWorkbench")
    if manifest["bundle_base_sha256"] != bundle.bundle_sha256 or run["lineage"]["bundle_sha256"] != bundle.bundle_sha256:
        raise ImportRefused("patch_base_mismatch", "the patch's bundle_base_sha256 (or the run's lineage) is not "
                                                   f"bundle {bundle.id}'s")
    parent = db.query(ProbeMonitor).filter(ProbeMonitor.id == bundle.probe_id).first()
    if parent is None or not parent.weights_path:
        raise ImportRefused("patch_base_mismatch", f"bundle {bundle.id}'s probe is gone or has no weights")

    # 3. the verdict
    if run.get("outcome") != "candidate" or verdict["value"] == "no_change" or not run.get("candidate_id"):
        raise ImportRefused("nothing_to_import", f"run {run['run_id']} has no candidate to import (outcome "
                                                 f"{run.get('outcome')}, verdict {verdict['value']})")
    if verdict["value"] != "improved":
        acknowledged = (acknowledgement or {}).get("verdict")
        if acknowledged != verdict["value"]:
            raise ImportRefused("acknowledgement_required",
                                f"the verdict is {verdict['value']}; importing it needs an acknowledgement naming "
                                "that verdict", verdict=verdict["value"])

    # 4. the chain (P21)
    base_probe = parent
    kind = manifest["parent"]["kind"]
    if manifest["parent_patch_sha256"] is not None or kind == "candidate":
        wanted = manifest["parent_patch_sha256"]
        earlier = None
        if wanted is not None and kind == "candidate" and manifest["parent"]["patch_sha256"] == wanted:
            earlier = next((p for p in db.query(ProbeMonitor).filter(ProbeMonitor.run_id == parent.run_id,
                                                                     ProbeMonitor.derived_from.isnot(None)).all()
                            if (p.derived_from or {}).get("patch_sha256") == wanted
                            and (p.derived_from or {}).get("bundle_id") == bundle.id and p.weights_path), None)
        if earlier is None:
            raise ImportRefused("patch_chain_unresolved",
                                "a chained patch is imported only on top of the patch it names, imported from the "
                                "same bundle first")
        base_probe = earlier
        base = _head_tensors(earlier)
    elif kind == "bundle" and manifest["parent"]["ref"] == bundle.bundle_sha256:
        base = _bundle_heads(bundle)
    else:
        raise ImportRefused("patch_base_mismatch", f"a patch on a {kind} parent ({manifest['parent']['ref']}) is not "
                                                   f"one this bundle's probe can take")

    # 5. the patch, applied here
    try:
        _, edited = apply_patch(manifest, edit_npz, base)
    except WspPatchRefused as refused:
        raise ImportRefused(refused.code, str(refused), **refused.details) from refused

    # 6. the limits
    limits = run["request"]["change_limits"]
    editable = {t["name"] for t in bundle.manifest["tensors"] if t["editable"]}
    if limits.get("allowed_tensors") is not None:
        editable &= set(limits["allowed_tensors"])
    editable -= set(limits.get("frozen_tensors") or [])
    try:
        check_limits(manifest, editable=editable, max_changed_entries=limits.get("max_changed_entries"))
    except WspPatchRefused as refused:
        raise ImportRefused(refused.code, str(refused), **refused.details) from refused

    # 7. P23's reference: THIS run's call whose candidate is exactly what was applied here (R1A-W8)
    digest = head_tensors_sha256({name: t.bits for name, t in edited.items()})
    call = (db.query(WspScoreCall)
            .filter(WspScoreCall.bundle_id == bundle.id, WspScoreCall.miforge_run_id == run["run_id"],
                    WspScoreCall.candidate_tensors_sha256 == digest,
                    WspScoreCall.candidate_scores_sha256.isnot(None))
            .order_by(WspScoreCall.created_at.desc()).first())
    if call is None:
        raise ImportRefused("persistence_reference_missing",
                            f"no score call of run {run['run_id']} scored exactly these tensors; P23 has nothing to "
                            "reproduce")
    uses = {s["name"]: int(s["heldout_uses"]) for s in verdict.get("per_set") or []}
    return Checked(run=run, manifest=manifest, bundle=bundle, parent=parent, base_probe=base_probe, edited=edited,
                   call=call, candidate_digest=digest, heldout_uses=uses)


# ── the new probe ─────────────────────────────────────────────────────────────


def derived_from_of(checked: Checked) -> Dict[str, Any]:
    """`DerivedFrom`'s fields (published, 9.4) plus miWorkbench's own (never exported: the builder reads only
    `DerivedFrom`'s fields)."""
    run, created_by = checked.run, checked.manifest["created_by"]
    return {
        "parent_probe_id": checked.parent.id,
        "parent_detector_digest": getattr(checked.parent, "built_detector_digest", None),
        "patch_sha256": checked.manifest["patch_sha256"],
        "provider": created_by["provider"],
        "provider_version": created_by["provider_version"],
        "verdict": {"value": run["verdict"]["value"], "run_sha256": run["run_sha256"]},
        # miWorkbench's own (not in the contract):
        "request_limits": run["request"]["change_limits"],
        "bundle_id": checked.bundle.id,
        "miforge_run_id": run["run_id"],
        "candidate_id": run["candidate_id"],
        "verdict_id": run["verdict"]["verdict_id"],
        "score_call_id": checked.call.id,
        "chained_on": checked.base_probe.id if checked.base_probe is not checked.parent else None,
        "heldout_uses": dict(checked.heldout_uses),
    }


def write_head(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """The `_persist_probe` writer shape: `weight`, `mean`, `std` (and `attention_query`) as tensors in their own
    dtype, `bias` in the metadata as `repr(float)` (exact for a float64)."""
    import torch
    from safetensors.torch import save_file

    tensors = {name: torch.from_numpy(np.ascontiguousarray(arrays[name]).copy())
               for name in ("weight", "mean", "std", "attention_query") if name in arrays}
    save_file(tensors, str(path), metadata={"bias": repr(float(np.asarray(arrays["bias"]).reshape(-1)[0]))})


def create_derived_probe(db: Any, checked: Checked) -> Any:
    """A NEW `ProbeMonitor`, field by field from the parent (never a copy of the ORM row): rung 0, never selected,
    window and length-band bars `null` (R1A-W6). Flushed, not committed; its weights are on disk."""
    from ..models.probe_monitor import ProbeMonitor
    from .probe_monitor_run import artifact_dir_for

    parent = checked.parent
    probe = ProbeMonitor(
        run_id=parent.run_id, layer=parent.layer, rule=parent.rule, rule_params=dict(parent.rule_params or {}),
        variant=parent.variant, render_form=parent.render_form, behaviour=parent.behaviour,
        behaviour_label=parent.behaviour_label, sub_question=parent.sub_question,
        behaviour_version=parent.behaviour_version, streamable=parent.streamable,
        val_metrics={"derived": True, "note": "a weight-space candidate imported from miForge; never trained here"},
        selected=False, rung=0, target_fpr=parent.target_fpr, calibration_dataset_id=parent.calibration_dataset_id,
        window_decisions=None, length_bands=None, derived_from=derived_from_of(checked),
        rung_detail={"bars_note": BARS_NOTE},
    )
    db.add(probe)
    db.flush()
    directory = artifact_dir_for(parent.run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{probe.id}.safetensors"
    write_head(path, _arrays(checked.edited))
    probe.weights_path = str(path)
    probe.norm_path = str(path)
    db.flush()
    return probe


# ── captures ──────────────────────────────────────────────────────────────────


@dataclass
class CaptureRows:
    capture: Any
    name: str
    rows: List[np.ndarray]
    positions: np.ndarray
    labels: np.ndarray
    truncated: np.ndarray
    n_scored: np.ndarray


def bundle_captures(db: Any, bundle: Any) -> List[CaptureRows]:
    """The bundle's held-out sets in `heldout_ref` order, each from its ready capture."""
    from ..models.wsp import WspActivationCapture
    from .wsp_bundle import capture_seal_broken, heldout_set_name

    captures = {heldout_set_name(c): c for c in db.query(WspActivationCapture).filter(
        WspActivationCapture.id.in_(list(bundle.capture_ids or []))).all()}
    out = []
    for entry in bundle.manifest["heldout_ref"]["sets"]:
        capture = captures.get(entry["name"])
        broken = None if capture is None else capture_seal_broken(bundle, entry, capture)
        if broken:   # R4 review fix (C044a-M1): the import evaluates on the sealed captures only, as the scorer does
            raise ImportRefused("capture_changed", broken)
        if capture is None or capture.status != "ready":
            raise ImportRefused("capture_not_ready", f"the capture of set {entry['name']} is not ready")
        directory = Path(capture.dir_path)
        offsets = np.load(directory / "offsets.npy")
        tokens = next(directory.glob("tokens.f*"))
        memmap = np.memmap(tokens, dtype=np.dtype(capture.dtype), mode="r",
                           shape=(int(capture.tokens), int(capture.d_model)))
        out.append(CaptureRows(capture=capture, name=entry["name"],
                               rows=[memmap[int(offsets[i]):int(offsets[i + 1])] for i in range(len(offsets) - 1)],
                               positions=np.load(directory / "positions.npy"),
                               labels=np.load(directory / "labels.npy"),
                               truncated=np.load(directory / "truncated.npy"),
                               n_scored=np.load(directory / "n_scored.npy")))
    return out


def _scores(head: Any, trained: Any, rows: Sequence[np.ndarray]) -> np.ndarray:
    from .probe_monitor_run import _aggregate_rows

    return np.asarray(_aggregate_rows(head, trained.rule, trained.rule_params, rows), dtype=np.float32)


def candidate_scores_sha256(head: Any, trained: Any, sets: Sequence[CaptureRows]) -> str:
    """The scorer's `candidate_scores_sha256` (FTID §3.5) recomputed from a head: float32 scores of every
    non-calibration set, untruncated rows only, sets in `heldout_ref` order."""
    digest = hashlib.sha256()
    for one in sets:
        if one.capture.role == "calibration":
            continue
        scores = _scores(head, trained, one.rows)
        digest.update(scores[one.truncated == 0].tobytes())
    return digest.hexdigest()


def persistence_check(db: Any, probe: Any, checked: Checked, sets: Sequence[CaptureRows]) -> None:
    """P23: the probe AS RE-LOADED (`load_probe`) scores the held-out rows bit for bit as the scorer scored the
    candidate in that run's call; else `persistence_mismatch`."""
    from .probe_monitor_run import load_probe

    head, trained = load_probe(db, probe.id)
    if candidate_scores_sha256(head, trained, sets) != checked.call.candidate_scores_sha256:
        raise ImportRefused("persistence_mismatch",
                            "the imported probe, re-loaded, does not reproduce the candidate's scores from score call "
                            f"{checked.call.id} bit for bit; the import was rolled back")


def calibrate_from_capture(db: Any, probe: Any, bundle: Any, sets: Sequence[CaptureRows], head: Any,
                           trained: Any) -> None:
    """The global bar: on the calibration capture's untruncated negatives (the scorer's own rows), or, for a probe
    whose parent was calibrated on practice validation negatives, on the bundle's practice validation negatives
    (OVN-wsp-workbench-31). Window and length-band bars are never cut here (R1A-W6)."""
    from .probe_monitor_run import artifact_dir_for, persist_calibration_negatives, untruncated_only
    from .probe_monitor_trainer import calibrate
    from .wsp_score import practice_validation_negatives

    directory = artifact_dir_for(probe.run_id)
    from .wsp_bundle import bar_target_fpr

    # R2 review fix (OR2-044a-L5): the derived probe's own target (copied from its parent), else the bundle's bar
    # target — the parent's, never a constant 0.01.
    target = float(probe.target_fpr if probe.target_fpr is not None else bar_target_fpr(db, bundle))
    calibration = next((s for s in sets if s.capture.role == "calibration"), None)
    # R1 review fix (OR1-044a-1), 2026-11-16: the derived probe's bar is cut on the basis the bundle's manifest
    # states; a calibration-basis bundle without its calibration set is refused, never re-based on practice rows.
    from .wsp_bundle import bar_basis

    basis = bar_basis(db, bundle)
    if basis == "calibration" and calibration is None:
        raise ImportRefused("capture_not_ready", "the bundle's bars are cut on its calibration set, and no "
                                                 "calibration capture is in it; the derived probe is not re-based "
                                                 "on practice rows")
    if basis != "calibration":
        calibration = None
    if calibration is not None:
        negative = calibration.labels == 0
        scores = _scores(head, trained, calibration.rows)[negative]
        truncated = calibration.truncated[negative].astype(int).tolist()
        probe.calibration_scores_path = persist_calibration_negatives(directory, probe.id, scores.tolist())
        probe.calibration_truncated_path = persist_calibration_negatives(directory, probe.id, truncated,
                                                                         name="truncated", dtype=np.int8)
        probe.calibration_positions_path = persist_calibration_negatives(
            directory, probe.id, calibration.positions[negative].tolist(), name="positions", dtype=np.int64)
        probe.calibration_lengths_path = persist_calibration_negatives(
            directory, probe.id, calibration.n_scored[negative].tolist(), name="lengths")
        bar_negatives = untruncated_only(scores.tolist(), truncated)
        source = "calibration_set"
    else:
        bar_negatives = _scores(head, trained, practice_validation_negatives(bundle)).tolist()
        probe.calibration_scores_path = persist_calibration_negatives(directory, probe.id, bar_negatives)
        source = "validation_negatives"
    probe.calibration_n_negatives = len(bar_negatives)
    if not bar_negatives:
        probe.threshold = None
        return
    cal = calibrate(bar_negatives, target_fpr=target, source=source)
    probe.threshold, probe.target_fpr = cal.threshold, cal.target_fpr
    probe.realised_fpr, probe.threshold_source = cal.realised_fpr, cal.source


def evaluate_from_captures(db: Any, probe: Any, sets: Sequence[CaptureRows], head: Any, trained: Any) -> List[Any]:
    """One evaluation per non-calibration held-out set, through the evaluator's own metrics
    (`evaluation_metrics`) and the five G5 arrays, `metrics.source: "wsp_capture"`. Returns the files written."""
    from ..models.probe_monitor import (
        ProbeMonitorDataset,
        ProbeMonitorEvaluation,
        ProbeMonitorRun,
        _pme_id,
    )
    from .probe_monitor_run import (
        artifact_dir_for,
        context_from_row,
        evaluation_array_paths,
        evaluation_metrics,
    )

    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).one()
    max_length = context_from_row(run).max_length
    written: List[Path] = []
    for one in sets:
        if one.capture.role == "calibration":
            continue
        view = db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id == one.capture.dataset_id).one()
        scores = _scores(head, trained, one.rows)
        labels = [int(v) for v in one.labels]
        lengths = [int(v) for v in one.n_scored]
        truncated = [int(v) for v in one.truncated]
        aggregates = [float(v) for v in scores]
        metrics = evaluation_metrics(view, aggregates, labels, lengths, truncated, target_fpr=probe.target_fpr,
                                     threshold=probe.threshold, max_length=max_length)
        metrics["source"] = CAPTURE_SOURCE
        metrics["capture_id"] = one.capture.id
        source = None
        if one.capture.source_evaluation_id:
            source = db.query(ProbeMonitorEvaluation).filter(
                ProbeMonitorEvaluation.id == one.capture.source_evaluation_id).first()
        if source is not None and (source.metrics or {}).get("sample"):
            metrics["sample"] = dict(source.metrics["sample"])
        row = ProbeMonitorEvaluation(id=_pme_id(), probe_id=probe.id, dataset_id=view.id,
                                     status="completed" if metrics.get("scored") else "refused",
                                     n_positive=metrics.get("n_positive"), n_negative=metrics.get("n_negative"),
                                     metrics=metrics)
        paths = evaluation_array_paths(artifact_dir_for(probe.run_id), probe.id, view.id, row.id)
        for column, values, dtype in (("scores_path", scores, np.float32),
                                      ("positions_path", one.positions, np.int64),
                                      ("n_scored_path", lengths, np.int32), ("labels_path", labels, np.int8),
                                      ("truncated_path", truncated, np.int8)):
            np.save(paths[column], np.asarray(values, dtype=dtype))
            written.append(paths[column])
            setattr(row, column, str(paths[column]))
        db.add(row)
    return written


def derived_rung_detail(db: Any, probe: Any) -> Dict[str, Any]:
    """`rung_detail.bars_note` and `rung_detail.heldout_reuse` (F44-D17): per held-out set, the verdict's use count
    (miForge's, P24) beside miWorkbench's own count of score calls that read the set (every call reads every set)."""
    from ..models.wsp import WspScoreCall

    stored = probe.derived_from or {}
    calls = 0
    if stored.get("bundle_id"):
        calls = db.query(WspScoreCall).filter(WspScoreCall.bundle_id == stored["bundle_id"]).count()
    reuse = {name: {"verdict_uses": int(uses), "score_calls": int(calls)}
             for name, uses in (stored.get("heldout_uses") or {}).items()}
    return {"bars_note": BARS_NOTE, "heldout_reuse": reuse}


# ── the import ────────────────────────────────────────────────────────────────


def import_candidate(db: Any, *, run_document: Any, patch_manifest: Any, edit_npz: bytes,
                     acknowledgement: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check, create, prove P23, calibrate, evaluate, recompute the rung. Raises `ImportRefused`; on any failure after
    the probe exists, the session is rolled back and every file this import wrote is removed."""
    from .probe_monitor_run import _unlink_quietly, load_probe, recompute_rung

    checked = check(db, run_document=run_document, patch_manifest=patch_manifest, edit_npz=edit_npz,
                    acknowledgement=acknowledgement)
    sets = bundle_captures(db, checked.bundle)
    written: List[Path] = []
    try:
        probe = create_derived_probe(db, checked)
        written.append(Path(probe.weights_path))
        persistence_check(db, probe, checked, sets)
        head, trained = load_probe(db, probe.id)
        calibrate_from_capture(db, probe, checked.bundle, sets, head, trained)
        written += [Path(p) for p in (probe.calibration_scores_path, probe.calibration_truncated_path,
                                      probe.calibration_positions_path, probe.calibration_lengths_path) if p]
        written += evaluate_from_captures(db, probe, sets, head, trained)
        db.commit()
    except BaseException:
        try:
            db.rollback()
        finally:
            _unlink_quietly(written)
        raise
    rung = recompute_rung(db, probe.id)
    logger.info("imported WSP candidate %s of run %s as probe %s (rung %s)", checked.run["candidate_id"],
                checked.run["run_id"], probe.id, rung.get("rung"))
    return {"probe_id": probe.id, "parent_probe_id": checked.parent.id, "bundle_id": checked.bundle.id,
            "miforge_run_id": checked.run["run_id"], "patch_sha256": checked.manifest["patch_sha256"],
            "verdict": checked.run["verdict"]["value"], "rung": int(probe.rung or 0),
            "score_call_id": checked.call.id}
