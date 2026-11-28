"""Sensitivity of the data behind a WSP bundle (ENH-010 044 FR-7; task 5.4; FTDD §6; P11, P35, P41, P69, F44-D24).

A bundle may leave miWorkbench (miForge reads its practice activations), so whether its sources hold sensitive data
decides what may happen to it. The value is per SOURCE (a probe view's downloaded dataset), and the bundle's is the
most sensitive of its sources: `sensitive` > `unknown` > `not_sensitive`. `unknown` is treated as sensitive (P11).

Per source, READ LIVE at every call (P41, R1A-X1) — a value cached at bundle build is a snapshot, never the answer:
1. a source registered from a miDataworks version (its view carries the verified record, `dataset_version`):
   * the REGISTERED value (the manifest's `sensitivity` recorded at registration) and miDataworks' LIVE value
     (`GET /api/v1/versions/{id}` → `sensitivity`, 3 s timeout). A `null` means "not stated" (FR-7 as amended by the
     operator, 2026-11-17, F44-D27): the value is the most sensitive of the STATED values, nulls ignored, and the
     LIVE read decides whether anything is stated now:
     - live stated → the most sensitive of the stated values. A registration-time `null` followed by a live
       `not_sensitive` reads `not_sensitive` (the 2026-11-17 change); a registered `sensitive` is never lowered by
       a live `not_sensitive`;
     - live `null` beside a stated registered value → `unknown` (or `sensitive` if that was registered): a claim
       miDataworks withdrew is never trusted (R2 review fix OR2-044a-M8, kept), and no mark lifts it;
   * miDataworks unreachable, unset, non-JSON, or answering without the field → `unknown`; NO MARK LIFTS IT (F44-D24);
   * both `null` (nothing ever stated) → the operator's newest mark decides (P35, confirmed by the operator
     2026-11-17, F44-D28), else `unknown`;
   * otherwise a mark may only RAISE, to `sensitive` (P69) — never lower anything, and never lift an `unknown` that
     comes from a withdrawn claim (the mark decides only where BOTH are `null`);
2. a source with no miDataworks record (a plain Hub download): the operator's newest mark, else `unknown`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import httpx

logger = logging.getLogger(__name__)

ORDER = {"not_sensitive": 0, "unknown": 1, "sensitive": 2}
VALUES = tuple(ORDER)
MARK_VALUES = ("sensitive", "not_sensitive")
#: miDataworks' version read (exists; 021 FR-021.26 adds `sensitivity` to its answer).
VERSION_ROUTE = "/api/v1/versions/{version_id}"
TIMEOUT_S = 3.0
#: The live read could not answer (distinct from a stated `null`).
UNREACHABLE = object()


def most_sensitive(values: Iterable[str]) -> str:
    values = list(values)
    return max(values, key=ORDER.__getitem__) if values else "unknown"


def read_live(version_id: str, *, transport: Optional[httpx.BaseTransport] = None) -> Any:
    """miDataworks' current `sensitivity` for a version: `"sensitive"`, `"not_sensitive"`, `None` (stated null), or
    `UNREACHABLE` (unset URL, any failure, non-JSON, or no `sensitivity` field in the answer)."""
    from ..core.config import settings

    base = (settings.midataworks_api_url or "").rstrip("/")
    if not base:
        return UNREACHABLE
    try:
        with httpx.Client(base_url=base, timeout=TIMEOUT_S, transport=transport) as client:
            response = client.get(VERSION_ROUTE.format(version_id=version_id))
        if response.status_code != 200:
            return UNREACHABLE
        body = response.json()
    except Exception as exc:  # noqa: BLE001 - any failure is "cannot say", never "not sensitive"
        logger.info("miDataworks sensitivity read for %s failed: %s", version_id, exc)
        return UNREACHABLE
    if not isinstance(body, dict) or "sensitivity" not in body:
        return UNREACHABLE
    value = body["sensitivity"]
    return value if value in MARK_VALUES or value is None else UNREACHABLE


@dataclass
class SourceSensitivity:
    """One source's value and the facts it came from (the `basis` entries of FTDD §5.7.4)."""

    value: str
    basis: List[Dict[str, Any]]


def newest_mark(db: Any, dataset_id: Any) -> Optional[Any]:
    from ..models.wsp import WspSensitivityMark

    return (db.query(WspSensitivityMark).filter(WspSensitivityMark.dataset_id == dataset_id)
            .order_by(WspSensitivityMark.created_at.desc(), WspSensitivityMark.id.desc()).first())


def source_sensitivity(db: Any, view: Any, *, live: Callable[[str], Any] = read_live) -> SourceSensitivity:
    """The effective sensitivity of one view's source, read live (rules in the module docstring)."""
    mark = newest_mark(db, view.dataset_id)
    mark_basis = ({"source": "operator_mark", "ref": mark.id, "value": mark.value} if mark is not None else None)
    record = getattr(view, "dataset_version", None) or {}
    version_id = record.get("version_id") if isinstance(record, dict) else None
    if not version_id:
        if mark is None:
            return SourceSensitivity("unknown", [{"source": "none", "ref": str(view.dataset_id), "value": "unknown"}])
        return SourceSensitivity(mark.value, [mark_basis])

    registered = record.get("sensitivity")
    current = live(version_id)
    basis: List[Dict[str, Any]] = [{"source": "midataworks", "ref": f"registered:{version_id}",
                                    "value": registered if registered in MARK_VALUES else "unknown"}]
    if current is UNREACHABLE:
        basis.append({"source": "midataworks", "ref": f"live:{version_id}", "value": "unknown"})
        # F44-D24: an unreachable miDataworks is never lifted; a mark may only raise it.
        value = most_sensitive(["unknown"] + ([mark.value] if mark is not None and mark.value == "sensitive" else []))
        return SourceSensitivity(value, basis + ([mark_basis] if mark_basis and value == "sensitive" else []))
    basis.append({"source": "midataworks", "ref": f"live:{version_id}",
                  "value": current if current in MARK_VALUES else "unknown"})
    if registered not in MARK_VALUES and current not in MARK_VALUES:
        # P35 (confirmed by the operator, 2026-11-17, F44-D28): both null — the operator's mark decides.
        if mark is None:
            return SourceSensitivity("unknown", basis)
        return SourceSensitivity(mark.value, basis + [mark_basis])
    if current in MARK_VALUES:
        # FR-7 (operator, 2026-11-17, F44-D27): a null is "not stated"; the most sensitive STATED value wins. The live
        # read states something now, so a registration-time null is ignored and a stated registered value still counts.
        value = most_sensitive([v for v in (registered, current) if v in MARK_VALUES])
    else:
        # Live null beside a stated registered value: the claim was withdrawn → at least `unknown` (OR2-044a-M8, kept);
        # a registered `sensitive` stays `sensitive` (a withdrawal never lowers).
        value = most_sensitive(["unknown", registered])
    if value != "sensitive" and mark is not None and mark.value == "sensitive":
        return SourceSensitivity("sensitive", basis + [mark_basis])  # P69: a mark may raise, never lower
    return SourceSensitivity(value, basis)


def effective(db: Any, views: Iterable[Any], *, live: Callable[[str], Any] = read_live) -> Dict[str, Any]:
    """`{value, basis}` over every source of `views` (each dataset once): the most sensitive source wins."""
    seen, values, basis = set(), [], []
    for view in views:
        record = (getattr(view, "dataset_version", None) or {}) if view is not None else {}
        key = (view.dataset_id, record.get("version_id") if isinstance(record, dict) else None) if view else None
        if view is None or key in seen:
            continue
        seen.add(key)
        one = source_sensitivity(db, view, live=live)
        values.append(one.value)
        basis.extend(one.basis)
    return {"value": most_sensitive(values), "basis": basis}
