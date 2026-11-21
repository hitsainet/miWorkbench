"""Load, validate and page one evaluation's per-row arrays (Feature 035 XR-2; FTDD §5.4, D47; FTID TI8).

The evaluation stage writes four aligned `.npy` files per evaluation (`probe_monitor_run.
_evaluate_probes_on_sets`): scores (float32), positions (int64, each kept row's position in the
downloaded split), n_scored (int32) and labels (int8). This module is the only reader, and it is
pure over the paths a row records so it can be tested without the app.

⚠ A BROKEN ARTIFACT IS AN ERROR, NEVER AN EMPTY PAGE (FR-16). A missing file, a missing path
column, a path outside the probe artifact root, an unreadable array, or four arrays of different
lengths each raise `EvaluationArtifactError` naming the check that failed. Serving `rows: []` for
any of them would read as "this evaluation scored nothing", and a consumer reproducing an AUROC
from a partial page would report a number for data it never saw.

Arrays are memory-mapped (`mmap_mode="r"`) and only the requested slice is materialised, so a
50,000-row page of a million-row evaluation reads 50,000 rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: The four arrays and the evaluation-row attribute recording each path, in response order.
ARRAY_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("score", "scores_path"),
    ("position", "positions_path"),
    ("n_scored", "n_scored_path"),
    ("label", "labels_path"),
)

#: ENH-010 G5 (GR-10): EVERY module that reads an evaluation's per-row arrays or a probe's calibration negatives, and
#: what it does about truncated rows. `apply`: it judges a detector, so it leaves truncated rows out (through
#: `untruncated` / the calibration mask). `label`: it serves or relays rows, so it serves every row and says which are
#: truncated (or relays only contract fields, which carry no rows). `apply_new_calibrations`: it reads a bar or a count
#: whose calibration, since G5, was cut without truncated negatives. `owner`: it writes the arrays and applies the
#: mask itself. `test_g5_readers.py` walks `backend/src` for readers and fails on one missing here.
G5_READERS: Dict[str, Tuple[str, str]] = {
    "src.services.probe_breakdown": ("apply", "per-group recall judges the detector"),
    "src.services.detector_success_report": ("apply", "success criteria judge the detector"),
    "src.services.rules_evaluation": ("apply", "rules ∪ probe judges a detector; its union bar uses the mask too"),
    "src.services.probe_blend.service": ("apply", "blend members are joined after their truncated rows are dropped"),
    "src.services.wsp_capture": ("apply", "ENH-010 044: capture parity compares under the stored evaluation's mask"),
    "src.services.wsp_import": ("apply", "ENH-010 044 FR-8: a derived probe's bar is cut on the calibration capture's "
                                         "untruncated negatives, and its evaluations write the mask they were scored with"),
    "src.services.probe_evaluation_scores": ("label", "the 035 scores route serves every row with `truncated`"),
    "src.api.v1.endpoints.probe_monitors": ("label", "the scores route: every row, flagged; reported_auroc all rows"),
    "src.mcp_server.tools.miserver_probes": ("label", "relays contract window fields only; no row, no bookkeeping"),
    "src.services.calibration_size": ("apply_new_calibrations", "calibration_n_negatives is the untruncated count"),
    "src.services.probe_decision": ("apply_new_calibrations", "copies bars field by field; no mask bookkeeping"),
    "src.workers.probe_blend_tasks": ("apply", "reads members through `load_member_arrays`, which applies the mask"),
    "src.services.behaviour_checklist": ("label", "filters evaluations by `positions_path IS NOT NULL`; reads no row"),
    "src.services.probe_monitor_run": ("owner", "writes the arrays and the mask; rung, selection and calibration apply it"),
    "src.services.probe_recalibration": ("owner", "re-cuts apply the stored calibration mask (GR-6)"),
    "src.models.probe_monitor": ("owner", "the columns"),
}

DEFAULT_LIMIT = 10_000
MAX_LIMIT = 50_000


class EvaluationArtifactError(Exception):
    """One evaluation's arrays cannot be served. `check` names the failed check."""

    def __init__(self, check: str, message: str):
        super().__init__(message)
        self.check = check
        self.message = message


#: ENH-010 G5 (GR-1): the fifth array, the truncation mask (int8, 1 = cut at `max_length`). Optional: an evaluation
#: recorded before G5 has none (`truncated_path` NULL), which is "not recorded", never "none truncated".
TRUNCATED_COLUMN: Tuple[str, str] = ("truncated", "truncated_path")


@dataclass
class EvaluationArrays:
    arrays: Dict[str, np.ndarray]
    total: int

    @property
    def truncation_recorded(self) -> bool:
        return TRUNCATED_COLUMN[0] in self.arrays


def untruncated(loaded: EvaluationArrays) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """ENH-010 G5 (GR-10): the four arrays without the truncated rows, and the note a reader states beside its figures.

    The ONE place an "apply" reader (`G5_READERS`) drops truncated rows, so every reader that judges a detector drops
    the same ones. With no mask recorded, every row is kept and the note says so (`rows: "all"`,
    `truncation_not_recorded`) — the reader's output is then labelled rather than silently mixed.
    """
    arrays = {field: loaded.arrays[field] for field, _ in ARRAY_COLUMNS}
    if not loaded.truncation_recorded:
        return arrays, {"rows": "all", "excluded": None, "reason": "truncation_not_recorded"}
    keep = np.asarray(loaded.arrays[TRUNCATED_COLUMN[0]]) == 0
    return ({field: np.asarray(values)[keep] for field, values in arrays.items()},
            {"rows": "untruncated", "excluded": int((~keep).sum()), "policy": "excluded"})


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_evaluation_arrays(row: Any, *, root: Path) -> EvaluationArrays:
    """Open the four arrays `row` records, under `root`, and check they align.

    `root` is the probe artifact root (`<data_dir>/probe_monitors`): every path the writer records
    lives there (`artifact_dir_for`), so anything else is refused rather than opened (FTID TI9).
    """
    evaluation_id = getattr(row, "id", "?")
    arrays: Dict[str, np.ndarray] = {}
    # ENH-010 G5: the mask is read under the same checks as the four arrays when it is recorded, and skipped when not.
    columns = list(ARRAY_COLUMNS) + ([TRUNCATED_COLUMN] if getattr(row, TRUNCATED_COLUMN[1], None) else [])
    for field, column in columns:
        recorded: Optional[str] = getattr(row, column, None)
        if not recorded:
            raise EvaluationArtifactError(
                "path_not_recorded",
                f"evaluation {evaluation_id} records no {column}, so its {field} array cannot be served",
            )
        path = Path(recorded)
        if not _inside(path, root):
            raise EvaluationArtifactError(
                "path_outside_artifact_root",
                f"evaluation {evaluation_id}'s {column} ({recorded}) is not under {root}; refusing to read it",
            )
        if not path.is_file():
            raise EvaluationArtifactError(
                "file_missing",
                f"evaluation {evaluation_id}'s {field} array is missing on disk ({recorded})",
            )
        try:
            arrays[field] = np.load(path, mmap_mode="r", allow_pickle=False).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - a corrupt array must not read as an empty one
            raise EvaluationArtifactError(
                "file_unreadable",
                f"evaluation {evaluation_id}'s {field} array at {recorded} could not be read ({exc})",
            ) from exc
    lengths = {field: int(a.shape[0]) for field, a in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise EvaluationArtifactError(
            "length_mismatch",
            f"evaluation {evaluation_id}'s arrays are not aligned: {lengths}",
        )
    return EvaluationArrays(arrays=arrays, total=lengths["score"])


def page(loaded: EvaluationArrays, offset: int, limit: int) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Rows `[offset, offset+limit)` in stored order, and the next offset (None at the end)."""
    end = min(offset + limit, loaded.total)
    if offset >= loaded.total:
        return [], None
    sliced = {field: loaded.arrays[field][offset:end].tolist() for field, _ in ARRAY_COLUMNS}
    # ENH-010 G5 (GR-8): each served row says whether it was truncated, from the SAME slice of the stored mask;
    # `None` on every row when the evaluation recorded no mask (never a guessed False).
    mask = (loaded.arrays[TRUNCATED_COLUMN[0]][offset:end].tolist() if loaded.truncation_recorded
            else [None] * (end - offset))
    rows = [
        {
            "position": int(sliced["position"][i]),
            "score": float(sliced["score"][i]),
            "label": int(sliced["label"][i]),
            "n_scored": int(sliced["n_scored"][i]),
            "truncated": None if mask[i] is None else bool(mask[i]),
        }
        for i in range(end - offset)
    ]
    return rows, (end if end < loaded.total else None)
