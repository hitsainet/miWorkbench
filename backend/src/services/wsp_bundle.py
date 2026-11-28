"""The WSP bundle (ENH-010 044 FR-2; tasks 5.1, 5.2; FTDD §4.4, §4.5, §5.7.4; FTID §3.3).

A bundle is what miForge may read about one probe: the parent head, the PRACTICE rows (the run's own token
activations, served in place, with the arrays that say which row is which) and a reference to the held-out sets it
will never read (their names, row counts and row-id digests only). Its manifest is exactly miForge's
`miforge.wsp-bundle/v1` (validated against the vendored schema; no pydantic model of it here, IDL-55).

⚠ THE PRACTICE ROWS ARE REBUILT, NOT REMEMBERED (§4.5). The run's token file holds rows in the order the run rendered
them; nothing on disk says which row is which. So the bundle re-runs the run's own preparation — view, render, split —
and refuses `practice_capture_mismatch` unless the rebuilt offsets size the file EXACTLY. A tokenizer, template or
dataset that drifted since the run fails here, before any number is produced.

⚠ NO HELD-OUT ROW IS A PRACTICE ROW (P45, P57, P67). Three key kinds, ANY intersection refuses
`heldout_overlaps_practice`: the LOCATION (dataset, revision, config, split, position) — two views over one split share
rows a (view, position) key cannot see; the TEXT key (sha256 of the row's source text column after `dw.rowkey/v1`'s
normalisation: NFC, CRLF→LF, strip) — two downloads of one split have different dataset ids and the same rows; and
miDataworks' `_dw_row_key` when BOTH sides carry it, IN ADDITION to the text key, never instead of it.

⚠ A BUNDLE NAMES ITS HELD-OUT SETS, AND A SET IS NEVER BUNDLED TWICE (FU-7a C1, 2026-11-25). `POST /wsp/bundles` may
name exactly which ready captures go in (`heldout_view_ids`; the calibration capture, the parent's bar basis, always
goes in and is exempt). A held-out set that shares a location, text, `_dw_row_key`, `carrier_id` or `pair_id` key
with any held-out set or practice row of an EARLIER bundle of the same probe is refused `heldout_reused` here; the
route also refuses any set miForge's own held-out uses (`GET /api/v1/wsp/heldout-uses`, the authority) say was scored
(`services/wsp_heldout_uses.py`). Before this a new bundle quietly re-bundled every old set (D128: seven sets, each
scored 18 times).

⚠ A SUPPLEMENT IS PRACTICE, FOR VALIDATION ONLY (FU-7a C2, miForge contract revision 2026-11-25). `supplement_view_ids`
names ready captures of `supplement` views (for example, negatives from a generator the probe never saw); their rows
are written to the five `supplement_*` files and listed in `practice.supplements`, each with its own parity. miForge
reads them, so each is checked against every held-out set (this bundle's and every earlier one's) exactly as practice
is (`heldout_overlaps_practice`). "Validation only" is miForge's use of them (its editor checks an edit on them); the
LR, LDA and DoM baselines FIT on them beside practice-train (operator decision 2026-11-27, `fit_and_store_baselines`),
so a refit sees every practice row the editor may use and is no longer the parent itself.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

KIND = "miforge.wsp-bundle/v1"
PARITY_ROWS = 16
#: R4 review fix (C044a-L2): a baseline fit is waited on in slices this long, with a heartbeat between them, so a fit
#: longer than the janitor's `BUNDLE_STALE` (60 min) never reads as a lost worker. Must stay well below it
#: (`test_wsp_bundle_build.py::test_the_fit_heartbeat_is_well_inside_the_janitor_window`).
BASELINE_BEAT_SECONDS = 60.0
PARITY_TOLERANCE = 1e-5
HASH_CHUNK = 256 * 1024 * 1024
#: `mean` and `std` are the standardisation the head was trained with: part of the function, never edited (P40).
EDITABLE = {"weight": True, "bias": True, "attention_query": True, "mean": False, "std": False}
TENSOR_ORDER = ("weight", "bias", "mean", "std", "attention_query")
THRESHOLD_SOURCE = {"calibration_set": "calibration", "validation_negatives": "practice"}
#: The capture roles a bundle's `heldout_ref` may carry (`calibration` is the parent's bar basis).
HELDOUT_CAPTURE_ROLES = ("eval", "everyday_text", "nonsense_text", "calibration")
#: The view (and capture) role of a validation-only practice supplement (FU-7a C2).
SUPPLEMENT_ROLE = "supplement"
#: The five supplement files: every supplement's rows, one after another, in `practice.supplements` order.
SUPPLEMENT_ARRAYS = (("supplement_offsets", "supplement_offsets.npy"),
                     ("supplement_positions", "supplement_positions.npy"),
                     ("supplement_labels", "supplement_labels.npy"),
                     ("supplement_truncated_mask", "supplement_truncated_mask.npy"))
PRACTICE_ARRAYS = (("practice_offsets", "practice_offsets.npy"), ("practice_positions", "practice_positions.npy"),
                   ("practice_labels", "practice_labels.npy"),
                   ("practice_validation_mask", "practice_validation_mask.npy"),
                   ("practice_truncated_mask", "practice_truncated_mask.npy"))


class BundleRefused(Exception):
    """`{code, message}` (FTDD §5.3); the bundle row reads `failed` with it."""

    def __init__(self, code: str, message: str, status: int = 409, **extra: Any):
        super().__init__(message)
        self.code, self.message, self.status, self.extra = code, message, status, extra

    def detail(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.extra}


from .wsp_bundle_schema import load_schema, load_validator, schema_errors  # noqa: E402,F401


def bundle_sha256(manifest: dict) -> str:
    """`sha256(rfc8785.dumps(manifest without bundle_sha256))` — `bundle_id` is inside the hashed bytes (P40)."""
    from .wsp_patch import jcs_sha256

    return jcs_sha256({k: v for k, v in manifest.items() if k != "bundle_sha256"})


def file_sha256(path: Path, beat: Optional[Callable[[], None]] = None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
            if beat is not None:
                beat()
    return digest.hexdigest()


# ── row keys and the overlap check (P45, P57, P67) ──────────────────────────────


def normalise(value: Any) -> Any:
    """`dw.rowkey/v1`'s normalisation, applied to every string (also inside chat messages); keys untouched."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    return value


def text_key(value: Any) -> str:
    """sha256 of the RFC 8785 bytes of the normalised source text value."""
    from .wsp_patch import jcs_sha256

    return jcs_sha256(normalise(value))


@dataclass
class RowKeys:
    """One side's keys: per row, in the order the positions were given."""

    view_id: str
    location: List[Tuple[str, Optional[str], Optional[str], Optional[str], int]]
    text: List[str]
    dw: Optional[List[str]] = None
    #: miDataworks' `carrier_id` and `pair_id` columns when the split carries them (FU-7a C1); a blank value is no
    #: key (natural rows have no carrier), so it never matches another blank.
    carrier: Optional[List[Optional[str]]] = None
    pair: Optional[List[Optional[str]]] = None


def row_keys(db: Any, view: Any, positions: Sequence[int]) -> RowKeys:
    """The three keys of `positions` (split positions) of `view`'s downloaded split."""
    from ..models.dataset import Dataset
    from .probe_monitor_service import open_split, resolve_dataset_path

    dataset = db.query(Dataset).filter(Dataset.id == view.dataset_id).first()
    if dataset is None:
        raise BundleRefused("practice_capture_mismatch", f"dataset {view.dataset_id} of view {view.id} is gone")
    data = open_split(resolve_dataset_path(dataset.raw_path), view.split)
    texts = data[view.input_column]
    dw = data["_dw_row_key"] if "_dw_row_key" in data.column_names else None
    positions = [int(p) for p in positions]
    revision = getattr(dataset, "revision", None)

    def column(name: str) -> Optional[List[Optional[str]]]:
        if name not in data.column_names:
            return None
        values = data[name]
        return [_key_value(values[p]) for p in positions]

    return RowKeys(
        view_id=view.id,
        location=[(str(view.dataset_id), revision, view.config, view.split, p) for p in positions],
        text=[text_key(texts[p]) for p in positions],
        dw=[str(dw[p]) for p in positions] if dw is not None else None,
        carrier=column("carrier_id"),
        pair=column("pair_id"),
    )


def _key_value(value: Any) -> Optional[str]:
    """A `carrier_id` / `pair_id` cell as a key, or None for a blank one (never a key that matches every blank)."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def shared_key(one: RowKeys, other: RowKeys) -> Optional[Tuple[str, int]]:
    """`(kind, rows)` of the first key kind `one` and `other` share, or None: location, text, then `_dw_row_key`,
    `carrier_id` and `pair_id` when both sides carry the column (blank cells never match). FU-7a C1."""
    pairs: List[Tuple[str, Any, Any]] = [("location", one.location, other.location), ("text", one.text, other.text),
                                         ("dw_row_key", one.dw, other.dw), ("carrier_id", one.carrier, other.carrier),
                                         ("pair_id", one.pair, other.pair)]
    for kind, mine, theirs in pairs:
        if mine is None or theirs is None:
            continue
        shared = {k for k in mine if k is not None} & {k for k in theirs if k is not None}
        if shared:
            return kind, len(shared)
    return None


def check_overlap(practice: RowKeys, heldout: Sequence[RowKeys]) -> None:
    """Refuse `heldout_overlaps_practice` on ANY shared key: location, text, or `_dw_row_key` when both carry it."""
    kinds = [("location", set(practice.location)), ("text", set(practice.text))]
    for side in heldout:
        pairs = [("location", side.location), ("text", side.text)]
        if practice.dw is not None and side.dw is not None:
            pairs.append(("dw_row_key", side.dw))
        for kind, keys in pairs:
            own = dict(kinds).get(kind) if kind != "dw_row_key" else set(practice.dw)
            shared = own & set(keys)
            if shared:
                raise BundleRefused(
                    "heldout_overlaps_practice",
                    f"{len(shared)} held-out row(s) of view {side.view_id} are practice rows of view "
                    f"{practice.view_id} by {kind} key; a bundle must never let miForge read a held-out row.",
                    practice_view=practice.view_id, heldout_view=side.view_id, key=kind, rows=len(shared))


def check_partition_roles(practice_view: Any, heldout_views: Sequence[Any]) -> None:
    """miDataworks' recorded split roles (044-DV, P33): a `wsp_heldout` split is never practice and a `wsp_practice`
    split is never held out (5.5b). A view with no recorded role (a plain download) is not refused here; the overlap
    keys still apply."""
    def role(view):
        record = getattr(view, "dataset_version", None) or {}
        return record.get("wsp_role") if isinstance(record, dict) else None

    if role(practice_view) == "wsp_heldout":
        raise BundleRefused("heldout_overlaps_practice",
                            f"the training view {practice_view.id} is a miDataworks wsp_heldout split; its rows may "
                            f"never be practice rows.", practice_view=practice_view.id, key="partition_role")
    for view in heldout_views:
        if role(view) == "wsp_practice":
            raise BundleRefused("heldout_overlaps_practice",
                                f"view {view.id} is a miDataworks wsp_practice split; it cannot be a held-out set.",
                                heldout_view=view.id, key="partition_role")


def check_supplement_roles(supplement_views: Sequence[Any]) -> None:
    """A practice supplement is practice (FU-7a C2): a miDataworks `wsp_heldout` split is never one (the plan marks a
    supplement split `wsp_practice`)."""
    for view in supplement_views:
        record = getattr(view, "dataset_version", None) or {}
        if isinstance(record, dict) and record.get("wsp_role") == "wsp_heldout":
            raise BundleRefused("heldout_overlaps_practice",
                                f"supplement view {view.id} is a miDataworks wsp_heldout split; its rows may never be "
                                f"practice rows.", practice_view=view.id, key="partition_role")


# ── the practice rebuild (§4.5) ────────────────────────────────────────────────


@dataclass
class Practice:
    view: Any
    rendered: List[Any]
    labels: np.ndarray
    offsets: np.ndarray
    positions: np.ndarray
    validation_mask: np.ndarray
    truncated_mask: np.ndarray
    tokens_path: Path
    dtype: str
    d_model: int

    def rows(self, indices: Sequence[int]) -> List[np.ndarray]:
        total = int(self.offsets[-1])
        memmap = np.memmap(self.tokens_path, dtype=np.dtype(self.dtype), mode="r", shape=(total, self.d_model))
        return [memmap[int(self.offsets[i]):int(self.offsets[i + 1])] for i in indices]


def practice_tokens_file(run_dir: Path, layer: int) -> Tuple[Path, str]:
    for suffix, dtype in (("f16", "float16"), ("f32", "float32")):
        path = run_dir / f"tokens_layer{layer}.{suffix}"
        if path.is_file():
            return path, dtype
    raise BundleRefused("practice_capture_mismatch",
                        f"the run's token file for layer {layer} is not in {run_dir}; the practice rows are gone")


def rebuild_practice(db: Any, probe: Any, run: Any, context: Any, tokenizer: Any, d_model: int) -> Practice:
    """The run's training view, rendered and split as the run did; refused unless it sizes the token file exactly."""
    from ..models.probe_monitor import ProbeMonitorDataset
    from .probe_monitor_capture import count_scored_tokens
    from .probe_monitor_run import _prepare_examples, _split_indices

    view = db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id == run.train_dataset_id).first()
    if view is None:
        raise BundleRefused("practice_capture_mismatch", f"the run's training view {run.train_dataset_id} is gone")
    examples, rendered, _summary, labels = _prepare_examples(db, view, context, tokenizer, role="train")
    _train, validation = _split_indices(examples, context)
    total, offsets = count_scored_tokens(rendered, context.scope)
    tokens_path, dtype = practice_tokens_file(Path(context.artifact_dir), probe.layer)
    expected = int(total) * int(d_model) * np.dtype(dtype).itemsize
    actual = tokens_path.stat().st_size
    if actual != expected:
        raise BundleRefused(
            "practice_capture_mismatch",
            f"the rebuilt practice rows need {expected} bytes ({total} tokens x {d_model} x {dtype}) but "
            f"{tokens_path.name} holds {actual}: the render, tokenizer or dataset changed since the run.")
    validation_mask = np.zeros(len(examples), dtype=np.int8)
    validation_mask[list(validation)] = 1
    return Practice(
        view=view, rendered=list(rendered), labels=np.asarray(labels, dtype=np.int8),
        offsets=np.asarray(offsets, dtype=np.int64),
        positions=np.asarray([int(e.index) for e in examples], dtype=np.int64),
        validation_mask=validation_mask,
        truncated_mask=np.asarray([1 if r.truncated else 0 for r in rendered], dtype=np.int8),
        tokens_path=tokens_path, dtype=dtype, d_model=int(d_model))


# ── validation-only supplements (FU-7a C2) ──────────────────────────────────────


@dataclass
class Supplement:
    """One ready `supplement` capture as the bundle carries it: the capture's own rows, files and labels."""

    capture: Any
    view: Any
    positions: np.ndarray
    labels: np.ndarray
    truncated: np.ndarray
    offsets: np.ndarray
    tokens_path: Path
    dtype: str
    d_model: int

    @property
    def rows(self) -> int:
        return int(len(self.positions))

    def token_rows(self, indices: Sequence[int]) -> List[np.ndarray]:
        total = int(self.offsets[-1])
        memmap = np.memmap(self.tokens_path, dtype=np.dtype(self.dtype), mode="r", shape=(total, self.d_model))
        return [memmap[int(self.offsets[i]):int(self.offsets[i + 1])] for i in indices]


def load_supplement(db: Any, capture: Any, practice: Practice) -> Supplement:
    """A ready supplement capture's files, refused `practice_capture_mismatch` unless they are whole and match the
    practice rows' dtype and width (miForge reads both through one `practice.dtype` and `d_model`)."""
    from ..models.probe_monitor import ProbeMonitorDataset

    view = db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id == capture.dataset_id).first()
    directory = Path(capture.dir_path or "")
    tokens = sorted(directory.glob("tokens.f*")) if capture.dir_path else []
    names = ("offsets", "positions", "labels", "truncated")
    if view is None or len(tokens) != 1 or not all((directory / f"{n}.npy").is_file() for n in names):
        raise BundleRefused("practice_capture_mismatch",
                            f"supplement capture {capture.id} (view {capture.dataset_id}) has lost its files or view")
    arrays = {n: np.load(directory / f"{n}.npy", allow_pickle=False) for n in names}
    dtype = str(capture.dtype)
    if dtype != practice.dtype or int(capture.d_model or 0) != int(practice.d_model):
        raise BundleRefused("practice_capture_mismatch",
                            f"supplement capture {capture.id} holds {dtype} x {capture.d_model}; the practice rows are "
                            f"{practice.dtype} x {practice.d_model}, and miForge reads both as one")
    supplement = Supplement(capture=capture, view=view, positions=arrays["positions"].astype(np.int64),
                            labels=arrays["labels"].astype(np.int8), truncated=arrays["truncated"].astype(np.int8),
                            offsets=arrays["offsets"].astype(np.int64), tokens_path=tokens[0], dtype=dtype,
                            d_model=int(capture.d_model))
    expected = int(supplement.offsets[-1]) * supplement.d_model * np.dtype(dtype).itemsize
    if supplement.rows < 8 or supplement.offsets.size != supplement.rows + 1 \
            or tokens[0].stat().st_size != expected:
        raise BundleRefused("practice_capture_mismatch",
                            f"supplement capture {capture.id}: {supplement.rows} rows (at least 8, the parity "
                            f"minimum) whose offsets and token file must agree")
    return supplement


def supplement_parity(trained: Any, head: Any, supplement: Supplement, seed: int) -> Dict[str, Any]:
    """Seeded rows WITHIN the supplement and the parent's float32 scores over them: miForge's evaluator must agree."""
    from .probe_monitor_run import _aggregate_rows

    n = supplement.rows
    rows = sorted(random.Random(f"{seed}|{supplement.view.id}").sample(range(n), min(PARITY_ROWS, n)))
    scores = _aggregate_rows(head, trained.rule, dict(trained.rule_params or {}), supplement.token_rows(rows))
    return {"rows": rows, "scores": [float(np.float32(s)) for s in scores], "tolerance": PARITY_TOLERANCE}


def supplement_entries(trained: Any, head: Any, supplements: Sequence[Supplement], seed: int) -> List[Dict[str, Any]]:
    """`practice.supplements`: one entry per supplement, in file order."""
    from .wsp_digests import row_ids_sha256

    entries, first = [], 0
    for supplement in supplements:
        labels = supplement.labels
        entries.append({
            "view_id": supplement.view.id, "first_row": first, "rows": supplement.rows,
            "tokens": int(supplement.offsets[-1]),
            "label_counts": {"positive": int((labels == 1).sum()), "negative": int((labels == 0).sum())},
            "truncated_rows": int(supplement.truncated.sum()),
            "row_ids_sha256": row_ids_sha256(supplement.view.id, supplement.positions),
            "validation_only": True,
            "parity": supplement_parity(trained, head, supplement, seed)})
        first += supplement.rows
    return entries


def write_supplement_files(directory: Path, supplements: Sequence[Supplement],
                           beat: Optional[Callable[[], None]] = None) -> List[Dict[str, Any]]:
    """The five `supplement_*` files in the bundle dir: every supplement's token rows copied one after another (the
    captures stay sealed where they are), offsets over the concatenation, positions, labels and truncation."""
    if not supplements:
        return []
    suffix = ".f16" if np.dtype(supplements[0].dtype) == np.float16 else ".f32"
    activations = directory / f"supplement_activations{suffix}"
    with open(activations, "wb") as out:
        for supplement in supplements:
            with open(supplement.tokens_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
                    out.write(chunk)
                    if beat is not None:
                        beat()
    shift, parts = 0, []
    for supplement in supplements:
        parts.append(supplement.offsets[(1 if parts else 0):] + shift)
        shift += int(supplement.offsets[-1])
    values = {"supplement_offsets": np.concatenate(parts).astype(np.int64),
              "supplement_positions": np.concatenate([s.positions for s in supplements]).astype(np.int64),
              "supplement_labels": np.concatenate([s.labels for s in supplements]).astype(np.int8),
              "supplement_truncated_mask": np.concatenate([s.truncated for s in supplements]).astype(np.int8)}
    entries: List[Tuple[str, Path]] = [("supplement_activations", activations)]
    for role, name in SUPPLEMENT_ARRAYS:
        np.save(directory / name, values[role])
        entries.append((role, directory / name))
    return [{"name": path.name, "role": role, "sha256": file_sha256(path, beat), "bytes": path.stat().st_size}
            for role, path in entries]


# ── heads ─────────────────────────────────────────────────────────────────────


def head_arrays(probe: Any) -> Dict[str, np.ndarray]:
    """The parent's tensors in their stored dtypes, `bias` as float64 `[1]` read exactly from the file metadata."""
    from safetensors import safe_open
    from safetensors.numpy import load_file

    from ..core.config import settings

    path = str(settings.resolve_data_path(probe.weights_path))
    tensors = load_file(path)
    with safe_open(path, framework="np") as handle:
        metadata = handle.metadata() or {}
    out = {name: np.ascontiguousarray(tensors[name]) for name in TENSOR_ORDER if name in tensors}
    out["bias"] = np.asarray([float(metadata.get("bias", "0.0"))], dtype=np.float64)
    return {name: out[name] for name in TENSOR_ORDER if name in out}


def tensor_entries(arrays: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    from .wsp_patch import tensor_sha256

    return [{"name": name, "shape": [int(d) for d in array.shape], "dtype": str(array.dtype),
             "sha256": tensor_sha256(array), "editable": EDITABLE[name]} for name, array in arrays.items()]


# ── the manifest ──────────────────────────────────────────────────────────────


def heldout_set_name(capture: Any) -> str:
    return f"{capture.role}-{capture.dataset_id}"[:64]


def heldout_ref(captures: Sequence[Any]) -> Dict[str, Any]:
    return {"owner": "miworkbench",
            "sets": [{"name": heldout_set_name(c), "view_id": c.dataset_id, "role": c.role, "rows": int(c.rows),
                      "row_ids_sha256": c.row_ids_sha256} for c in captures]}


def parity_block(probe: Any, trained: Any, head: Any, practice: Practice, seed: int) -> Dict[str, Any]:
    """16 seeded practice rows and the parent's float32 scores over them (P40): miForge's own scorer must agree."""
    from .probe_monitor_run import _aggregate_rows

    n = len(practice.positions)
    if n < 8:
        raise BundleRefused("practice_capture_mismatch", f"only {n} practice rows; parity needs at least 8")
    rows = sorted(random.Random(seed).sample(range(n), min(PARITY_ROWS, n)))
    scores = _aggregate_rows(head, trained.rule, dict(trained.rule_params or {}), practice.rows(rows))
    return {"rows": rows, "scores": [float(np.float32(s)) for s in scores], "tolerance": PARITY_TOLERANCE}


def build_manifest(*, bundle_id: str, created_at: str, probe: Any, run: Any, model_name: str, context: Any,
                   trained: Any, head: Any, practice: Practice, files: List[Dict[str, Any]],
                   arrays: Dict[str, np.ndarray], captures: Sequence[Any], sensitivity: Dict[str, Any],
                   baselines_available: Sequence[str],
                   supplements: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    from .wsp_digests import row_ids_sha256

    params = dict(trained.rule_params or {})
    window = params.get("window")
    operating_point = None
    if probe.threshold is not None and probe.threshold_source in THRESHOLD_SOURCE:
        operating_point = {"target_fpr": float(probe.target_fpr), "threshold": float(probe.threshold),
                           "threshold_source": THRESHOLD_SOURCE[probe.threshold_source]}
    labels = practice.labels
    manifest = {
        "kind": KIND,
        "bundle_kind": "linear_probe",
        "bundle_id": bundle_id,
        "created_at": created_at,
        "identity": {"owner": "miworkbench", "artifact_kind": "probe", "artifact_id": probe.id,
                     "detector_digest": getattr(probe, "built_detector_digest", None), "model": model_name,
                     "layer": int(probe.layer), "run_id": probe.run_id},
        "files": files,
        "tensors": tensor_entries(arrays),
        "function": {
            "kind": "linear_probe/v1",
            "description": (f"miWorkbench linear probe {probe.id}: rule {trained.rule} over the standardised "
                            f"token activations of layer {probe.layer} (scope {context.scope}); "
                            f"score = (x - mean) / std . weight + bias per token, combined by the rule."),
            "linear_probe": {"rule": trained.rule,
                             "params": {"tau": params.get("tau"), "window": int(window) if window else None},
                             "layer": int(probe.layer), "scope": context.scope,
                             "skip_template_prefix": bool(context.skip_template_prefix)},
            "sae_fn": None,
            "parity": parity_block(probe, trained, head, practice, context.seed),
        },
        "practice": {"format": "token_activations", "dtype": practice.dtype, "rows": int(len(practice.positions)),
                     "tokens": int(practice.offsets[-1]), "d_model": int(practice.d_model),
                     "view_id": practice.view.id,
                     "row_ids_sha256": row_ids_sha256(practice.view.id, practice.positions),
                     "label_counts": {"positive": int((labels == 1).sum()), "negative": int((labels == 0).sum())},
                     "validation_rows": int(practice.validation_mask.sum()),
                     "truncated_rows": int(practice.truncated_mask.sum())},
        "heldout_ref": heldout_ref(captures),
        "baselines_available": list(baselines_available),
        "operating_point": operating_point,
        "sensitivity": sensitivity,
        "sae": None,
    }
    if supplements:   # FU-7a C2: absent, never [], when the bundle has none (the contract's one way to say none)
        manifest["practice"]["supplements"] = list(supplements)
    manifest["bundle_sha256"] = bundle_sha256(manifest)
    errors = schema_errors(manifest)
    if errors:
        raise BundleRefused("bundle_schema_invalid", f"the manifest does not validate: {errors[:5]}", status=500)
    return manifest


def write_bundle_files(directory: Path, practice: Practice, arrays: Dict[str, np.ndarray],
                       beat: Optional[Callable[[], None]] = None) -> List[Dict[str, Any]]:
    """`heads.safetensors` and the five practice arrays in the bundle dir; the run's token file is served in place.
    Returns `files[]` (flat names, roles, sha256, bytes)."""
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    heads = directory / "heads.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in arrays.items()}, str(heads))
    values = {"practice_offsets": practice.offsets, "practice_positions": practice.positions,
              "practice_labels": practice.labels, "practice_validation_mask": practice.validation_mask,
              "practice_truncated_mask": practice.truncated_mask}
    entries: List[Tuple[str, Path]] = [("heads", heads), ("practice_activations", practice.tokens_path)]
    for role, name in PRACTICE_ARRAYS:
        np.save(directory / name, values[role])
        entries.append((role, directory / name))
    return [{"name": path.name, "role": role, "sha256": file_sha256(path, beat), "bytes": path.stat().st_size}
            for role, path in entries]


def file_path_of(bundle: Any, name: str) -> Optional[Path]:
    """Where a manifest-listed file lives, or None when `name` is not in the manifest's `files[]` (FR-3): a name is
    resolved ONLY from that list — never joined to a directory as given — and a capture file is never in it."""
    for entry in (bundle.manifest or {}).get("files", []):
        if entry.get("name") == name:
            bundle_directory = Path(bundle.dir_path)
            if entry["role"] == "practice_activations":
                return bundle_directory.parents[2] / entry["name"]   # the run's token file, served in place
            return bundle_directory / entry["name"]
    return None


@dataclass
class Built:
    manifest: Dict[str, Any]
    practice: Practice
    arrays: Dict[str, np.ndarray]
    captures: List[Any] = field(default_factory=list)


# ── baselines: fitted on practice, stored, never listed in files[] (FR-4; tasks 6.1 storage, 6.6) ─────────────


def save_baseline_head(path: Path, head: Any) -> None:
    """The `_persist_probe` shape: tensors plus the bias in the metadata, so `load_baseline_head` reads it back."""
    from safetensors.torch import save_file

    tensors = {"weight": head.weight.detach().cpu().contiguous()}
    for name in ("mean", "std", "attention_query"):
        value = getattr(head, name)
        if value is not None:
            tensors[name] = value.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"bias": repr(float(head.bias))})


def load_baseline_head(path: Path, layer: int) -> Any:
    from safetensors import safe_open
    from safetensors.torch import load_file

    from ..ml.probe_monitor_model import ProbeHead

    tensors = load_file(str(path))
    with safe_open(str(path), framework="pt") as handle:
        bias = float((handle.metadata() or {}).get("bias", "0.0"))
    return ProbeHead(weight=tensors["weight"], bias=bias, mean=tensors.get("mean"), std=tensors.get("std"),
                     attention_query=tensors.get("attention_query"), layer=layer)


def fit_and_store_baselines(db: Any, bundle: Any, probe: Any, head: Any, trained: Any, practice: Practice,
                            context: Any, directory: Path, *, max_seconds: Optional[float] = None,
                            beat: Optional[Callable[[], None]] = None,
                            supplements: Sequence[Supplement] = ()) -> Dict[str, str]:
    """Every registered kind on the PRACTICE rows only, the bundle's practice `supplements` included (operator
    decision 2026-11-27: the refits see every practice row miForge's editor may use; `wsp_baselines` docstring); one
    `wsp_baselines` row each. Returns kind → `fitted`, `pending` (the fit ran past `wsp_baseline_max_seconds`:
    reported, never silently missing) or `refused:<code>`."""
    import concurrent.futures

    from ..core.runtime_settings import effective
    from ..models.wsp import WspBaseline
    from .probe_monitor_run import _aggregate_rows
    from .wsp_baselines import BASELINE_KINDS, BaselineRefused
    from .wsp_baselines import Practice as FitPractice
    from .wsp_layout import baselines_dir

    limit = float(max_seconds if max_seconds is not None else effective().wsp_baseline_max_seconds)
    rows = practice.rows(range(len(practice.positions)))
    fit_practice = FitPractice(rows=rows, labels=[int(y) for y in practice.labels],
                               validation_mask=[bool(v) for v in practice.validation_mask],
                               truncated_mask=[bool(t) for t in practice.truncated_mask],
                               supplement_rows=[r for s in supplements for r in s.token_rows(range(s.rows))],
                               supplement_labels=[int(y) for s in supplements for y in s.labels])
    parent_scores = _aggregate_rows(head, trained.rule, dict(trained.rule_params or {}), rows)
    kwargs = {"rule": trained.rule, "rule_params": dict(trained.rule_params or {}), "seed": context.seed,
              "train_config": dict(context.config or {}), "parent_scores": parent_scores,
              "target_fpr": float(probe.target_fpr or context.target_fpr)}
    outcome: Dict[str, str] = {}
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="wsp-baseline")
    try:
        for kind, spec in BASELINE_KINDS.items():
            row = WspBaseline(bundle_id=bundle.id, kind=kind)
            future = pool.submit(spec.fit, head, fit_practice, **kwargs)
            try:
                fitted = _wait_beating(future, limit, beat)
            except concurrent.futures.TimeoutError:
                # The fit cannot be interrupted; its thread finishes on its own and its result is dropped.
                row.fit = {"pending": True, "reason": f"the fit ran past wsp_baseline_max_seconds = {limit:g}"}
                outcome[kind] = "pending"
                pool.shutdown(wait=False)
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="wsp-baseline")
            except BaselineRefused as refused:
                row.fit = {"refused": refused.code, "reason": str(refused)}
                outcome[kind] = f"refused:{refused.code}"
            else:
                if spec.stores_weights:
                    path = baselines_dir(probe.run_id, bundle.id) / f"{kind}.safetensors"
                    save_baseline_head(path, fitted.head)
                    row.weights_path = str(path)
                row.bar, row.transform, row.fit = fitted.bar, fitted.transform, dict(fitted.fit)
                outcome[kind] = "fitted"
            db.add(row)
            if beat is not None:
                beat()   # C044a-I1: the claim commits this row only while the bundle is still building
            db.commit()
    finally:
        pool.shutdown(wait=False)
    return outcome


def _wait_beating(future: Any, limit: float, beat: Optional[Callable[[], None]]) -> Any:
    """`future.result(timeout=limit)`, waited in `BASELINE_BEAT_SECONDS` slices with `beat()` between them (C044a-L2,
    2026-11-17). `wsp_baseline_max_seconds` may be set up to a day, and the janitor reaps a started bundle silent for
    `BUNDLE_STALE`; the build's heartbeat is also its claim, so a bundle reaped meanwhile stops here
    (`BundleNotBuilding`). Raises `TimeoutError` once `limit` has passed."""
    import concurrent.futures
    import time

    deadline = time.monotonic() + limit
    while True:
        remaining = deadline - time.monotonic()
        try:
            return future.result(timeout=max(0.0, min(BASELINE_BEAT_SECONDS, remaining)))
        except concurrent.futures.TimeoutError:
            if future.done() or time.monotonic() >= deadline:   # the fit's own TimeoutError, or the limit
                raise
            if beat is not None:
                beat()


# ── the whole build (task 6.6 runs it in the `wsp` worker) ────────────────────────────────────────────────────────


def _newest_ready(db: Any, probe: Any, roles: Sequence[str]) -> List[Any]:
    from ..models.wsp import WspActivationCapture
    from .wsp_capture import calibration_capture_stale

    rows = (db.query(WspActivationCapture)
            .filter(WspActivationCapture.probe_id == probe.id, WspActivationCapture.status == "ready",
                    WspActivationCapture.superseded_at.is_(None), WspActivationCapture.role.in_(list(roles)))
            .order_by(WspActivationCapture.created_at.desc()).all())
    newest: Dict[Tuple[str, str], Any] = {}
    for row in rows:
        newest.setdefault((row.dataset_id, row.role), row)
    current = [c for c in newest.values() if calibration_capture_stale(probe, c) is None]
    return sorted(current, key=lambda c: (c.role, c.dataset_id))


def ready_captures(db: Any, probe: Any, view_ids: Optional[Sequence[str]] = None) -> List[Any]:
    """The newest READY capture per held-out set; a `parity_failed` capture is never referenced, nor a superseded one
    (C044a-M1), nor a calibration capture checked against a bar the probe no longer serves
    (`calibration_capture_stale`, OR3-044a-L3): a calibration-set probe is then refused `capture_not_ready` until it
    is captured again. A `supplement` capture is never a held-out set (FU-7a C2).

    `view_ids` (FU-7a C1): only the captures of those held-out views, plus the calibration capture (the parent's bar
    basis), which always goes in. None = every held-out set (the request named none)."""
    current = _newest_ready(db, probe, HELDOUT_CAPTURE_ROLES)
    if view_ids is None:
        return current
    wanted = set(view_ids)
    return [c for c in current if c.dataset_id in wanted or c.role == "calibration"]


def ready_supplements(db: Any, probe: Any, view_ids: Sequence[str]) -> List[Any]:
    """The newest READY `supplement` capture of each named view, in the order the views were named (FU-7a C2)."""
    by_view = {c.dataset_id: c for c in _newest_ready(db, probe, (SUPPLEMENT_ROLE,))}
    return [by_view[v] for v in view_ids if v in by_view]


def resolve_request(db: Any, probe: Any, heldout_view_ids: Optional[Sequence[str]],
                    supplement_view_ids: Optional[Sequence[str]]) -> Tuple[List[Any], List[Any]]:
    """`(held-out captures, supplement captures)` for a bundle request, or `BundleRefused` (FU-7a C1, C2): a view
    named twice or in both lists (422 `bundle_request_invalid`), a view that does not exist (404 `view_not_found`), a
    named view of the wrong role (422 `bundle_request_invalid`), or a named view with no ready capture (409
    `capture_not_ready`, every one named). Read by the route before anything is queued and by the build again."""
    from ..models.probe_monitor import ProbeMonitorDataset

    heldout = list(heldout_view_ids) if heldout_view_ids is not None else None
    supplements = list(supplement_view_ids or [])
    named = (heldout or []) + supplements
    if len(set(named)) != len(named):
        raise BundleRefused("bundle_request_invalid", "a view is named twice (or as both a held-out set and a "
                                                      "supplement); name each view once", status=422)
    views = {v.id: v for v in db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id.in_(named)).all()} \
        if named else {}
    unknown = [v for v in named if v not in views]
    if unknown:
        raise BundleRefused("view_not_found", f"no probe dataset view {unknown}", status=404, views=unknown)
    wrong = [v for v in (heldout or []) if views[v].role not in HELDOUT_CAPTURE_ROLES] + \
        [v for v in supplements if views[v].role != SUPPLEMENT_ROLE]
    if wrong:
        raise BundleRefused("bundle_request_invalid",
                            f"view(s) {wrong} are not of the role their list needs: heldout_view_ids takes eval, "
                            f"everyday_text, nonsense_text or calibration views; supplement_view_ids takes "
                            f"{SUPPLEMENT_ROLE} views", status=422, views=wrong)
    captures = ready_captures(db, probe, heldout)
    extras = ready_supplements(db, probe, supplements)
    missing = [v for v in (heldout or []) if v not in {c.dataset_id for c in captures}] + \
        [v for v in supplements if v not in {c.dataset_id for c in extras}]
    if missing:
        raise BundleRefused("capture_not_ready",
                            f"view(s) {missing} have no ready capture for probe {probe.id}; capture them "
                            f"(POST /wsp/captures) and wait for parity", views=missing)
    return captures, extras


def used_set(capture: Any, used: Sequence[Dict[str, Any]]) -> bool:
    """Whether miForge's held-out uses (`used`: the rows `wsp_heldout_uses.used_by_us` keeps) count `capture`, by
    its set name or its row-id digest."""
    name = heldout_set_name(capture)
    return any(u.get("set_name") == name or u.get("row_ids_sha256") == capture.row_ids_sha256 for u in used)


def earlier_bundle_keys(db: Any, probe: Any, bundle_id: str,
                        used: Sequence[Dict[str, Any]]) -> Tuple[List[RowKeys], List[RowKeys]]:
    """`(held-out keys, practice keys)` of the EARLIER bundles of `probe` (FU-7a C1): every held-out capture one was
    sealed on that miForge has scored (`used`, read by the route from miForge's held-out uses; calibration included:
    it was scored too), and every practice supplement one carried (miForge may have read it), each from its own
    manifest and files. An earlier bundle's training view is this bundle's practice view (one probe, one run),
    checked by `check_overlap` already. A set that was bundled but never scored is not "used": a bundle that failed to
    import may be built again on the same sets. A record that can no longer be read is refused
    `heldout_history_unreadable`: the check is never skipped for a bundle it cannot see."""
    from ..models.probe_monitor import ProbeMonitorDataset
    from ..models.wsp import WspActivationCapture, WspBundle

    earlier = (db.query(WspBundle).filter(WspBundle.probe_id == probe.id, WspBundle.id != bundle_id)
               .order_by(WspBundle.created_at).all())
    capture_ids = list(dict.fromkeys(cid for b in earlier for cid in (b.capture_ids or [])))
    every = (db.query(WspActivationCapture).filter(WspActivationCapture.id.in_(capture_ids)).all()
             if capture_ids else [])
    found = {c.id for c in every}
    lost = [c for c in capture_ids if c not in found]
    captures = [c for c in every if used_set(c, used)]
    sources = [(c.dataset_id, Path(c.dir_path or "") / "positions.npy", None) for c in captures]
    for bundle in earlier:
        for entry in ((bundle.manifest or {}).get("practice") or {}).get("supplements") or []:
            sources.append((entry["view_id"], Path(bundle.dir_path or "") / "supplement_positions.npy",
                            (int(entry["first_row"]), int(entry["rows"]))))
    view_ids = list(dict.fromkeys(v for v, _p, _s in sources))
    views = {v.id: v for v in db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id.in_(view_ids)).all()} \
        if view_ids else {}
    unreadable = lost + [str(path) for v, path, _s in sources if v not in views or not path.is_file()]
    if unreadable:
        raise BundleRefused("heldout_history_unreadable",
                            f"the rows of an earlier bundle of probe {probe.id} can no longer be read "
                            f"({unreadable[:3]}); without them no held-out set can be shown unused", status=409)
    heldout, practice = [], []
    for view_id, path, span in sources:
        positions = np.load(path, allow_pickle=False)
        if span is not None:
            positions = positions[span[0]:span[0] + span[1]]
        (practice if span is not None else heldout).append(row_keys(db, views[view_id], positions))
    return heldout, practice


def check_heldout_unused(new: Sequence[Tuple[Any, RowKeys]], earlier_heldout: Sequence[RowKeys],
                         earlier_practice: Sequence[RowKeys]) -> None:
    """Refuse `heldout_reused` when a held-out set of this bundle shares ANY key (`shared_key`) with a held-out set or
    a practice supplement of an earlier bundle of the probe (FU-7a C1). The calibration capture is exempt: it is the
    parent's bar basis, kept across rebuilds by decision (FU-7a D4)."""
    for capture, keys in new:
        if capture.role == "calibration":
            continue
        for old in list(earlier_heldout) + list(earlier_practice):
            hit = shared_key(keys, old)
            if hit:
                raise BundleRefused(
                    "heldout_reused",
                    f"{hit[1]} row(s) of held-out view {capture.dataset_id} share a {hit[0]} key with view "
                    f"{old.view_id} of an earlier bundle of this probe; a held-out set is scored in one bundle's "
                    f"history only. Bundle fresh held-out rows (heldout_view_ids).",
                    heldout_view=capture.dataset_id, earlier_view=old.view_id, key=hit[0], rows=hit[1])


def check_supplements_unseen(supplements: Sequence[RowKeys], heldout: Sequence[RowKeys]) -> None:
    """Refuse `heldout_overlaps_practice` when a supplement shares ANY key with a held-out set (this bundle's or an
    earlier one's): miForge reads every supplement row (FU-7a C2; P45, P57, P67)."""
    for supplement in supplements:
        for side in heldout:
            hit = shared_key(supplement, side)
            if hit:
                raise BundleRefused(
                    "heldout_overlaps_practice",
                    f"{hit[1]} row(s) of supplement view {supplement.view_id} share a {hit[0]} key with held-out "
                    f"view {side.view_id}; a bundle must never let miForge read a held-out row.",
                    practice_view=supplement.view_id, heldout_view=side.view_id, key=hit[0], rows=hit[1])


def capture_parity_sha256(parity: Any) -> Optional[str]:
    """The RFC 8785 sha256 of a capture's parity record, as a bundle seals it (C044a-M1); None for no record."""
    from .wsp_patch import jcs_sha256

    return None if parity is None else jcs_sha256(parity)


def capture_seal_broken(bundle: Any, entry: Dict[str, Any], capture: Any) -> Optional[str]:
    """Why `capture` is no longer the capture `bundle` was sealed on for its `heldout_ref` set `entry`, or None
    (R4 review fix C044a-M1, 2026-11-17). Compared: the capture's `rows` and `row_ids_sha256` with the manifest's
    entry, and its parity record with the digest the bundle recorded at build (`capture_seals`). A bundle with no seal
    cannot be shown unchanged, and is refused the same way. The scorer and the import answer `capture_changed`."""
    seals = getattr(bundle, "capture_seals", None)
    if not isinstance(seals, dict) or capture.id not in seals:
        return f"bundle {bundle.id} records no seal for capture {capture.id} (set {entry['name']})"
    if capture.rows is None or int(capture.rows) != int(entry["rows"]) \
            or capture.row_ids_sha256 != entry["row_ids_sha256"]:
        return (f"the capture of set {entry['name']} now holds {capture.rows} rows with row ids "
                f"{capture.row_ids_sha256}; the bundle's heldout_ref names {entry['rows']} rows, {entry['row_ids_sha256']}")
    if capture_parity_sha256(capture.parity) != seals[capture.id]:
        return (f"the capture of set {entry['name']} carries another parity record than the one bundle {bundle.id} "
                f"was built on (it was captured again)")
    return None


def calibration_capture_missing(probe: Any, captures: Sequence[Any]) -> Optional[str]:
    """R1 review fix (OR1-044a-1), 2026-11-16: a probe calibrated on its CALIBRATION SET is bundled only with a ready
    calibration capture. Without one every head's bar would be cut on practice rows — rows miForge reads — while the
    manifest's operating point says `calibration`. Returns the refusal message, or None."""
    if getattr(probe, "threshold_source", None) != "calibration_set":
        return None
    if any(getattr(c, "role", None) == "calibration" for c in captures):
        return None
    return (f"probe {probe.id} was calibrated on its calibration set, and that set has no ready capture with parity "
            f"against the bar it serves now (none, or one taken before a recalibration); capture it (POST "
            f"/wsp/captures, including the calibration view) — its bars are never cut on practice rows")


def bar_basis(db: Any, bundle: Any) -> str:
    """`calibration` or `practice`: the basis every head's bar is cut on, as the MANIFEST states it (OR1-044a-1). A
    bundle whose probe has no bar (`operating_point` null) takes its probe's own threshold source."""
    from ..models.probe_monitor import ProbeMonitor

    source = ((bundle.manifest or {}).get("operating_point") or {}).get("threshold_source")
    if source is None:
        probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == bundle.probe_id).first()
        source = THRESHOLD_SOURCE.get(getattr(probe, "threshold_source", None), "practice")
    return str(source)


def bar_target_fpr(db: Any, bundle: Any) -> float:
    """The target FPR every head's bar is cut at: the manifest's operating point, else the probe's own `target_fpr` (a
    probe with no bar still has a target), else its run's. R2 review fix (OR2-044a-L5), 2026-11-17: a bundle with no
    operating point used to fall back to a constant 0.01, whatever the probe was trained for."""
    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorRun
    from .probe_monitor_run import context_from_row

    target = ((bundle.manifest or {}).get("operating_point") or {}).get("target_fpr")
    if target is not None:
        return float(target)
    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == bundle.probe_id).first()
    if probe is not None and probe.target_fpr is not None:
        return float(probe.target_fpr)
    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).one()
    return float(context_from_row(run).target_fpr)


def build_bundle(db: Any, bundle: Any, *, tokenizer: Any = None, beat: Optional[Callable[[], None]] = None,
                 max_seconds: Optional[float] = None, heldout_view_ids: Optional[Sequence[str]] = None,
                 supplement_view_ids: Optional[Sequence[str]] = None,
                 used_heldout: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Build `bundle` (a `building` row) end to end: refusals first, then files, baselines and the manifest.

    `heldout_view_ids` (FU-7a C1): exactly those held-out sets, plus the calibration capture; None = every ready
    held-out set. `supplement_view_ids` (FU-7a C2): the validation-only practice supplements, in that order.
    `used_heldout`: the held-out uses miForge reported to the route (`wsp_heldout_uses.used_by_us`), which decide
    which earlier sets count as used here (the `wsp` worker has no miForge address)."""
    from ..core.clock import utc_now
    from ..models.model import Model
    from ..models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun
    from .probe_monitor_run import _load_tokenizer_for_run, context_from_row, load_probe
    from .wsp_layout import bundle_dir
    from .wsp_patch import jcs_sha256  # noqa: F401 (the manifest's digest; imported for the RFC 8785 dependency)
    from .wsp_sensitivity import effective as effective_sensitivity

    def heartbeat():
        # Only while the row is still `building` (OR3-044a-L2): a reaped row stops the build here.
        if not claim_building(db, bundle.id):
            raise BundleNotBuilding(bundle.id, _status_now())
        if beat is not None:
            beat()

    def _status_now():
        from ..models.wsp import WspBundle

        return db.query(WspBundle.status).filter(WspBundle.id == bundle.id).scalar()

    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == bundle.probe_id).one()
    if getattr(probe, "variant", "dense") == "sae":
        raise BundleRefused("bundle_kind_unsupported",
                            f"probe {probe.id} reads SAE features; an SAE-probe bundle is not defined (phase 4 bundles "
                            f"an SAE itself, not an SAE probe).", status=422)
    if not probe.weights_path:
        raise BundleRefused("practice_capture_mismatch", f"probe {probe.id} has no weights on disk")
    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).one()
    context = context_from_row(run)
    captures, supplement_captures = resolve_request(db, probe, heldout_view_ids, supplement_view_ids)
    if not captures:
        raise BundleRefused("capture_not_ready",
                            f"probe {probe.id} has no ready held-out capture; capture its held-out sets first "
                            f"(POST /wsp/captures) and wait for parity.")
    missing = calibration_capture_missing(probe, captures)
    if missing:
        raise BundleRefused("capture_not_ready", missing)
    head, trained = load_probe(db, probe.id)
    arrays = head_arrays(probe)
    heartbeat()
    practice = rebuild_practice(db, probe, run, context, tokenizer or _load_tokenizer_for_run(run),
                                d_model=int(arrays["weight"].shape[0]))
    heartbeat()
    views = {v.id: v for v in db.query(ProbeMonitorDataset).filter(
        ProbeMonitorDataset.id.in_([c.dataset_id for c in captures])).all()}
    check_partition_roles(practice.view, [views[c.dataset_id] for c in captures])
    heldout_keys = [row_keys(db, views[c.dataset_id], np.load(Path(c.dir_path) / "positions.npy"))
                    for c in captures]
    check_overlap(row_keys(db, practice.view, practice.positions), heldout_keys)
    # FU-7a C1: no held-out set of this bundle was a held-out set or practice row of an earlier one.
    earlier_heldout, earlier_practice = earlier_bundle_keys(db, probe, bundle.id, list(used_heldout or []))
    check_heldout_unused(list(zip(captures, heldout_keys)), earlier_heldout, earlier_practice)
    # FU-7a C2: the supplements, checked against every held-out set miForge must never read.
    supplements = [load_supplement(db, c, practice) for c in supplement_captures]
    check_supplement_roles([s.view for s in supplements])
    check_supplements_unseen([row_keys(db, s.view, s.positions) for s in supplements],
                             heldout_keys + earlier_heldout)
    heartbeat()
    sensitivity = effective_sensitivity(db, [practice.view] + [views[c.dataset_id] for c in captures]
                                        + [s.view for s in supplements])
    directory = bundle_dir(run.id, bundle.id)
    files = write_bundle_files(directory, practice, arrays, beat=heartbeat)
    files += write_supplement_files(directory, supplements, beat=heartbeat)
    fitted = fit_and_store_baselines(db, bundle, probe, head, trained, practice, context, directory,
                                     max_seconds=max_seconds, beat=heartbeat, supplements=supplements)
    model = db.query(Model).filter(Model.id == run.model_id).first()
    manifest = build_manifest(
        bundle_id=bundle.id, created_at=utc_now().isoformat().replace("+00:00", "Z"), probe=probe, run=run,
        model_name=(getattr(model, "repo_id", None) or getattr(model, "name", None) or run.model_id),
        context=context, trained=trained, head=head, practice=practice, files=files, arrays=arrays,
        captures=captures, sensitivity=sensitivity,
        baselines_available=[kind for kind, state in fitted.items() if state == "fitted"],
        supplements=supplement_entries(trained, head, supplements, context.seed) if supplements else None)
    import rfc8785

    (directory / "manifest.json").write_bytes(bytes(rfc8785.dumps(manifest)))
    bundle.manifest, bundle.bundle_sha256 = manifest, manifest["bundle_sha256"]
    bundle.sensitivity, bundle.sensitivity_basis = sensitivity["value"], sensitivity["basis"]
    bundle.capture_ids, bundle.dir_path = [c.id for c in captures], str(directory)
    # C044a-M1: what the bundle is sealed on; the scorer and the import refuse `capture_changed` on any difference.
    bundle.capture_seals = {c.id: capture_parity_sha256(c.parity) for c in captures + supplement_captures}
    # FR-1 (R2 review fix OR2-044a-M3): the parent's bar as the scorer cuts it, beside the bar as served, so "parent"
    # in a verdict is never silently a different operating point from the parent miServer serves.
    from .wsp_score import scorer_parent_bar

    bundle.scorer_parent_bar = scorer_parent_bar(db, bundle)
    # The row is still ours (OR3-044a-L2): read under a row lock, so the janitor cannot reap it between this check and
    # the commit that makes it `ready`.
    from ..models.wsp import WspBundle

    status = (db.query(WspBundle.status).filter(WspBundle.id == bundle.id).with_for_update().scalar())
    if status != "building":
        raise BundleNotBuilding(bundle.id, status)
    # C044a-M1: the captures read at the start are still the current ones for the bar the manifest states (a
    # recalibration or a re-capture during the build would seal an operating point no capture was checked against).
    db.refresh(probe)
    current = {c.id for c in ready_captures(db, probe)} | \
        {c.id for c in ready_supplements(db, probe, [c.dataset_id for c in supplement_captures])}
    moved = [c.id for c in captures + supplement_captures if c.id not in current]
    if moved:
        raise BundleRefused("capture_not_ready", f"held-out capture(s) {moved} were superseded or went stale while "
                                                 f"the bundle was built (the probe was recalibrated or re-captured); "
                                                 f"build it again")
    bundle.status = "ready"
    db.commit()
    return {"bundle_id": bundle.id, "status": bundle.status, "baselines": fitted}


class BundleNotBuilding(Exception):
    """The row is no longer `building` (the janitor reaped it, or another build finished it): this build stops and
    writes nothing to the row (R3 review fix OR3-044a-L2)."""

    def __init__(self, bundle_id: str, status: Optional[str]):
        super().__init__(f"bundle {bundle_id} is {status!r}, not 'building'")
        self.status = status


def claim_building(db: Any, bundle_id: str) -> bool:
    """Write `heartbeat_at` on the row only while it is still `building`, in one statement; False when it is not
    (the janitor's `worker_lost`/`never_started`, or a finished build), and then the session is rolled back, so work
    pending in it (a baseline row) is never committed for a reaped bundle (R4 review fix C044a-I1). A started build is
    the only row the janitor ages by heartbeat, so this is also the task's first sign of life."""
    from sqlalchemy import update

    from ..core.clock import utc_now
    from ..models.wsp import WspBundle

    done = db.execute(update(WspBundle).where(WspBundle.id == bundle_id, WspBundle.status == "building")
                      .values(heartbeat_at=utc_now()).execution_options(synchronize_session=False))
    if not done.rowcount:   # C044a-I1: a refused claim commits nothing, pending work included
        db.rollback()
        return False
    db.commit()
    return True


def build_bundle_row(db: Any, bundle_id: str, **kwargs: Any) -> Dict[str, Any]:
    """The task body: a refusal or failure is recorded on the row (`failed`, `error` = `code: message`).

    R3 review fix (OR3-044a-L2), 2026-11-17: the `wsp` worker is solo, so a bundle can wait in its queue behind a long
    build; the janitor used to reap it from `created_at` and the late task then built it `ready` anyway. Now the task
    claims the row at start (`claim_building`: its first heartbeat, only while `building`), the janitor ages only
    started rows by heartbeat, and a row that is no longer `building` (here, or at any heartbeat, or before `ready`)
    stops the build without writing to it."""
    from ..models.wsp import WspBundle

    bundle = db.query(WspBundle).filter(WspBundle.id == bundle_id).one()
    if not claim_building(db, bundle_id):
        db.refresh(bundle)
        logger.warning("WSP bundle %s not built: it is %s, not building", bundle_id, bundle.status)
        return {"bundle_id": bundle_id, "status": bundle.status, "skipped": "bundle_not_building"}
    db.refresh(bundle)
    try:
        return build_bundle(db, bundle, **kwargs)
    except BundleNotBuilding as gone:
        db.rollback()
        logger.warning("WSP bundle %s stopped: %s", bundle_id, gone)
        return {"bundle_id": bundle_id, "status": gone.status, "skipped": "bundle_not_building"}
    except BundleRefused as refused:
        db.rollback()
        bundle.status, bundle.error = "failed", f"{refused.code}: {refused.message}"[:4000]
        db.commit()
        return {"bundle_id": bundle_id, "status": "failed", "refused": refused.detail()}
    except Exception as exc:
        db.rollback()
        bundle.status, bundle.error = "failed", str(exc)[:4000]
        db.commit()
        raise
