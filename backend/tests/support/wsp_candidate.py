"""A miForge-shaped candidate for the import tests (ENH-010 044 tasks 8.2 – 8.6): a `miforge.wsp-patch/v1` patch in
miForge's sparse form over a REAL bundle's head, and a `miforge.wsp-run/v1` run document that carries it, each
validated against the VENDORED schema before a test sees it (a fixture that drifted from the contract would make every
refusal test pass for the wrong reason).

`score_candidate` records the score call P23 reads: the scorer itself scores the edited head for the run, exactly as
miForge's `evaluate` would make it do.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.services.wsp_import import run_sha256_of, validate_run_document
from src.services.wsp_patch import (
    WspTensor,
    content_sha256,
    jcs_sha256,
    patch_sha256,
    read_safetensors,
    tensor_sha256,
    validate_manifest,
)

CREATED_AT = "2026-11-16T05:00:00Z"


@dataclass
class Candidate:
    run: Dict[str, Any]
    manifest: Dict[str, Any]
    npz: bytes
    edited: Dict[str, WspTensor]

    @property
    def npz_b64(self) -> str:
        return base64.b64encode(self.npz).decode()

    def body(self, **extra: Any) -> Dict[str, Any]:
        return {"run_document": copy.deepcopy(self.run), "patch_manifest": copy.deepcopy(self.manifest),
                "edit_npz_b64": self.npz_b64, **extra}


def bundle_heads(bundle: Any) -> Dict[str, WspTensor]:
    return read_safetensors((Path(bundle.dir_path) / "heads.safetensors").read_bytes())


def sparse_patch(bundle: Any, base: Dict[str, WspTensor], *, entries: List[int], delta: float = 0.25,
                 tensor: str = "weight", parent: Optional[Dict[str, Any]] = None,
                 parent_patch_sha256: Optional[str] = None, run_id: str = "wsp_000000000000",
                 tamper_edited_sha256: bool = False) -> tuple[Dict[str, Any], bytes, Dict[str, WspTensor]]:
    """One sparse tensor patch: `entries` of `tensor` moved by `delta` (bits written as bits)."""
    held = base[tensor]
    values = held.bits.view(np.float32).reshape(-1).copy()
    idx = np.asarray(sorted(entries), dtype=np.int64)
    values[idx] = values[idx] + np.float32(delta)
    new_bits = values.view(np.uint32)
    edited_bits = held.bits.reshape(-1).copy()
    edited_bits[idx] = new_bits[idx]
    edited = WspTensor(held.dtype, edited_bits.reshape(held.bits.shape))
    buffer = io.BytesIO()
    np.savez(buffer, t0_idx=idx, t0_val=new_bits[idx].astype(np.uint32))
    npz = buffer.getvalue()
    deltas = values[idx].astype(np.float64) - held.bits.view(np.float32).reshape(-1)[idx].astype(np.float64)
    entry = {"append": None, "base_sha256": held.sha256, "changed_entries": int(idx.size), "dtype": held.dtype,
             "edited_sha256": "0" * 64 if tamper_edited_sha256 else tensor_sha256(edited.bits),
             "edited_shape": held.shape, "form": "sparse", "index": 0,
             "l2_delta": float(np.sqrt((deltas ** 2).sum())), "max_abs_delta": float(np.abs(deltas).max()),
             "name": tensor, "rank": None, "shape": held.shape}
    manifest = {"arrays_file": "edit.npz", "arrays_sha256": hashlib.sha256(npz).hexdigest(),
                "bundle_base_sha256": bundle.bundle_sha256, "changed_entries_total": int(idx.size),
                "content_sha256": "", "created_at": CREATED_AT,
                "created_by": {"provider": "weightspace", "provider_version": "0.3.0", "run_id": run_id},
                "format_version": 1, "kind": "miforge.wsp-patch/v1",
                "parent": parent or {"kind": "bundle", "patch_sha256": None, "ref": bundle.bundle_sha256},
                "parent_patch_sha256": parent_patch_sha256, "patch_sha256": "", "tensors": [entry]}
    manifest["content_sha256"] = content_sha256(manifest)
    manifest["patch_sha256"] = patch_sha256(manifest)
    validate_manifest(manifest)
    return manifest, npz, {**base, tensor: edited}


#: miForge's verdict keeps one truncated-row count per head label the scorer answered (`truncated_fired`, a map
#: since miForge F-2, 2026-11-19; it was an object with one literal key `label`). Every head's count is 0 here
#: because `excluded_truncated` is 0.
TRUNCATED_FIRED = {"candidate": 0, "control": 0, "parent": 0}
#: miForge 016-P2 review (R16P2-4): the verdict records the candidate's and the cast control's changed count and
#: delta norm (`control_figures`, required in /v1). Carried here so every import test runs it through the package copy.
CONTROL_FIGURES = {"candidate": {"changed_entries": 1, "l2_delta": 0.25},
                   "control_post_cast": {"changed_entries": 1, "l2_delta": 0.2500000037252903}}


def run_document(bundle: Any, manifest: Dict[str, Any], *, run_id: str, verdict: str = "improved",
                 outcome: str = "candidate", max_changed_entries: Optional[int] = None,
                 frozen_tensors: Optional[List[str]] = None, uses: int = 1) -> Dict[str, Any]:
    """`uses` is the PRIMARY set's `heldout_uses` (the others get `uses + i`). Its default is 1, a fresh primary set,
    so an `improved` fixture stands under FU-7b's rule (`wsp_standing`); 2 or more makes it `improved_unconfirmed`
    (2026-11-25: the default was 3, which the rule reads as unconfirmed)."""
    sets = bundle.manifest["heldout_ref"]["sets"]
    request = {
        "budgets": {"cost_usd": None, "forward_passes": 0, "max_steps": 10, "time_s": 600, "tokens": 1000},
        "bundle": {"bundle_id": bundle.id, "bundle_sha256": bundle.bundle_sha256, "owner": "miworkbench"},
        "chain": manifest["parent_patch_sha256"] is not None,
        "change_limits": {"allowed_tensors": None, "frozen_tensors": list(frozen_tensors or []),
                          "max_changed_entries": max_changed_entries},
        "endpoint": None, "goal": None, "keep_weights": False, "kind": "repair", "mode": "session",
        "observation_tools": [], "of_run_id": None, "parent": {"kind": "bundle", "ref": bundle.bundle_sha256},
        "primary_sets": [sets[0]["name"]], "program_sha256": None,
        "provider": {"id": "weightspace", "version": "0.3.0"}, "sensitivity_declared": "unknown",
        "support_set": None, "task": "repair the probe", "thinking": None,
    }
    document = {
        "approval": None, "boot_id": None, "caller_ref": None, "candidate_id": f"wspc_{uuid.uuid4().hex[:12]}",
        "content_sha256": manifest["content_sha256"], "created_at": CREATED_AT, "failure": None,
        "forward_passes_used": 0, "kind": "miforge.wsp-run/v1",
        "lineage": {"bundle_sha256": bundle.bundle_sha256, "chain": [manifest["patch_sha256"]],
                    "parent": {"kind": "bundle", "ref": bundle.bundle_sha256}, "provider": "weightspace",
                    "provider_version": "0.3.0"},
        "outcome": outcome, "patch_sha256": manifest["patch_sha256"], "program_reruns": [], "program_sha256": None,
        "replay": None, "request": request, "request_sha256": jcs_sha256(request), "result_sha256": None,
        "run_id": run_id, "run_sha256": None,
        "sensitivity": {"effective": "sensitive", "source": "none", "value": "unknown"}, "status": "completed",
        "stop_reason": "submitted", "transcript_sha256": None,
        "usage": {"cost_usd": 0, "input_tokens": 0, "metered_by": "wsp_gateway", "output_tokens": 0, "requests": 0,
                  "self_reported": None},
        "verdict": {
            "B": 1000, "ci": 0.95, "completeness": {"missing": [], "ok": True}, "control_seed": 7,
            "control_figures": CONTROL_FIGURES,
            "evaluated_at": CREATED_AT, "guarded_comparisons": 0,
            "per_set": [{"deltas": [], "excluded_truncated": 0, "heads": [], "heldout_uses": uses + i,
                         "name": s["name"], "negatives": 1, "positives": 1, "primary": i == 0, "role": s["role"],
                         "rows": s["rows"], "truncated_fired": TRUNCATED_FIRED, "value": verdict}
                        for i, s in enumerate(sets)],
            "primary_metric": "auroc", "reasons": [], "request_sha256": jcs_sha256(request),
            "scorer": {"call_id": "wsc_x", "contract": "miworkbench-wsp-scoring-v1", "metrics_agree": True,
                       "owner": "miworkbench"},
            "seed": 0, "statistic": "paired_bootstrap", "value": verdict, "verdict_id": f"wsv_{uuid.uuid4().hex[:12]}",
        },
    }
    document["run_sha256"] = run_sha256_of(document)
    return validate_run_document(document)


def make_candidate(bundle: Any, *, entries: List[int], delta: float = 0.25, verdict: str = "improved",
                   base: Optional[Dict[str, WspTensor]] = None, **run_kwargs: Any) -> Candidate:
    run_id = f"wsp_{uuid.uuid4().hex[:12]}"
    patch_kwargs = {k: run_kwargs.pop(k) for k in ("parent", "parent_patch_sha256", "tamper_edited_sha256")
                    if k in run_kwargs}
    manifest, npz, edited = sparse_patch(bundle, base or bundle_heads(bundle), entries=entries, delta=delta,
                                         run_id=run_id, **patch_kwargs)
    return Candidate(run=run_document(bundle, manifest, run_id=run_id, verdict=verdict, **run_kwargs),
                     manifest=manifest, npz=npz, edited=edited)


def head_body(label: str, tensors: Dict[str, WspTensor]) -> Dict[str, Any]:
    out = []
    for name, tensor in tensors.items():
        array = tensor.bits.view({"float16": np.float16, "float32": np.float32, "float64": np.float64}[tensor.dtype])
        out.append({"name": name, "dtype": tensor.dtype, "shape": tensor.shape,
                    "data_b64": base64.b64encode(np.ascontiguousarray(array).tobytes()).decode(),
                    "sha256": tensor_sha256(array)})
    return {"label": label, "tensors": out}


def score_candidate(db: Any, bundle: Any, candidate: Candidate, *,
                    tensors: Optional[Dict[str, WspTensor]] = None) -> Dict[str, Any]:
    """The scorer scores `tensors` (default: the candidate's edited head) as the run's `candidate`."""
    from src.schemas.wsp_scoring import WspScoreRequest
    from src.services import wsp_score

    body = {"kind": "miworkbench.wsp-score-request/v1", "bundle_id": bundle.id, "bundle_sha256": bundle.bundle_sha256,
            "run_ref": {"miforge_run_id": candidate.run["run_id"], "patch_sha256": candidate.manifest["patch_sha256"],
                        "max_changed_entries": None},
            "heldout_ref": json.loads(json.dumps(bundle.manifest["heldout_ref"])),
            "heads": [{"label": "parent", "tensors": None}, head_body("candidate", tensors or candidate.edited)],
            "primary_sets": []}
    return wsp_score.score(db, WspScoreRequest.model_validate(body))
