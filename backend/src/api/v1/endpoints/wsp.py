"""ENH-010 044: weight-space programming routes (FTDD §5.2), prefix `/wsp`.

miWorkbench never edits weights here. It captures held-out activations (FR-1), builds bundles in miForge's shape
(FR-2), scores heads miForge proposes on rows miForge never sees (FR-5), and imports a verdicted candidate as a NEW
probe (FR-8). Agent callers (`X-MiStudio-Agent`) are refused on the routes that carry activations, scores or an
operator's decision (`refuse_agent_callers`). The capture and import routes take a consumed approval instead (044
FTASKS 4.4, 14.2): `require_consumed_approval(<action>, exempt_agents=frozenset(), refuse_without_id=MCP_POD_AGENTS)` —
the operator (no agent header) is unchanged; the MCP pod without an approved id is 403 `approval_required` (its tools
`capture_wsp_heldout` / `import_wsp_candidate` file first and pass the id on the re-call); any other agent without an
id files one pending approval and gets 202, nothing queued; a call WITH an approved id spends it once, before the
handler runs, for exactly that payload.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from ....core.approval_gate import (
    MCP_POD_AGENTS,
    gated_responses,
    require_consumed_approval,
)
from ....schemas import workbench_answers as WA  # 046 FTASKS 8.10: documented answers (decorators only)
from ....services.probe_dataset_lifecycle import AGENT_HEADER

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wsp", tags=["wsp"])

PHASE2_WORDS = {
    "operator_or_service_only": "this route serves activations, scores or the manifest's file names; it is called by "
                                "the operator or miForge's backend, never by an agent. Agents read the bundle summary.",
}


def refuse_agent_callers(code: str, message: Optional[str] = None):
    """A dependency refusing any `X-MiStudio-Agent` caller with `{code, message}` (FTDD §8, F44-D16). Modelled on
    `code_rules.refuse_agents`, which raises its own `code_rule_operator_only` and so cannot be reused."""

    def _refuse(request: Request) -> None:
        if request.headers.get(AGENT_HEADER) is not None:
            raise HTTPException(status_code=403, detail={"code": code, "message": message or PHASE2_WORDS[code]})

    _refuse.refusal_code = code  # the registry tests read which code a route refuses with
    return _refuse


def approval_gated(action: str) -> Any:
    """044 FTASKS 4.4 / 14.2 (FTDD §5.6; 045 A29 NEEDS N-4): the route consumes `action`'s approval itself. No agent
    is exempt — the MCP pod included, whose tools file the approval and pass its id on the re-call — and the MCP pod
    without an id is refused 403 `approval_required` rather than filing a second request behind its own tool's back."""
    return require_consumed_approval(action, exempt_agents=frozenset(), refuse_without_id=MCP_POD_AGENTS)


def _session():
    from ....core.database import SyncSessionLocal

    return SyncSessionLocal()


# ── captures (FR-1; tasks 4.1 – 4.3) ──────────────────────────────────────────


class CaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_id: str = Field(min_length=1, max_length=64)
    #: Only these held-out views (and/or the calibration view); omitted = every held-out set the probe was scored on.
    dataset_ids: Optional[List[str]] = Field(None, max_length=64)


def capture_view(row: Any) -> Dict[str, Any]:
    return {"id": row.id, "probe_id": row.probe_id, "dataset_id": row.dataset_id, "role": row.role,
            "source_evaluation_id": row.source_evaluation_id, "layer": row.layer, "scope": row.scope,
            "rows": row.rows, "tokens": row.tokens, "d_model": row.d_model, "dtype": row.dtype,
            "row_ids_sha256": row.row_ids_sha256, "parity": row.parity, "status": row.status, "bytes": row.bytes,
            "error": row.error, "created_at": row.created_at.isoformat() if row.created_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "superseded_at": row.superseded_at.isoformat() if row.superseded_at else None,
            "superseded_by": row.superseded_by}


#: A capture in one of these states is not asked again: it is in hand or under way.
_KEPT = ("pending", "running", "ready")
#: `pg_advisory_xact_lock(key, hashtext(probe_id))`: one capture plan per probe at a time (R2 review fix OR2-044a-L6).
WSP_CAPTURE_PLAN_LOCK = 0x57535043  # "WSPC"
#: Pause between reading the existing rows and writing the new ones; a test widens it to prove the lock serialises.
_BETWEEN_PLAN_AND_WRITE = lambda: None  # noqa: E731


def plan_capture_rows(db: Any, probe_id: str, dataset_ids: Optional[List[str]], *,
                      check_gpu: bool = False) -> Dict[str, Any]:
    """Plan the sets (refusing as `plan_capture` does) and make one row per set: an existing pending, running or
    ready row is kept as it is; a failed one is re-armed `pending`; a ready calibration capture that no longer
    describes the probe's bar is SUPERSEDED by a new row; the rest are created.

    R4 review fix (C044a-M1), 2026-11-17: round 3 re-armed a stale ready calibration capture in place, and every
    earlier bundle naming it then lost its evidence (unscorable, or scored on other rows). A ready row is sealed
    evidence and is never reset: the stale one keeps its status, parity and files, gains `superseded_at` and
    `superseded_by`, and a new row (a new directory) serves new bundles. A failed row is still re-armed in place: no
    bundle ever names a row that was not ready (`ready_captures`).

    R2 review fix (OR2-044a-L6), 2026-11-17: two requests for one probe at once used to both find no row and both
    create one — two GPU captures of the calibration set (its unique key holds a NULL), or an IntegrityError 500 for
    an eval set. The plan now holds a transaction-scoped advisory lock on the probe until its rows are committed, so
    the second request finds the first one's rows and keeps them; `uq_wsp_capture_calibration` is the backstop."""
    from sqlalchemy import text

    from ....core.clock import utc_now
    from ....models.probe_monitor import ProbeMonitor
    from ....models.wsp import WspActivationCapture, _wac_id
    from ....services.wsp_capture import (
        calibration_capture_stale,
        capture_gpu_check,
        plan_capture,
        refuse_if_gpu_too_full,
    )

    db.execute(text("SELECT pg_advisory_xact_lock(:key, hashtext(:probe))"),
               {"key": WSP_CAPTURE_PLAN_LOCK, "probe": probe_id})
    probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
    if probe is None:
        raise HTTPException(status_code=404, detail={"code": "probe_not_found", "message": f"probe {probe_id}"})
    planned = plan_capture(db, probe, dataset_ids)
    from ....models.probe_monitor import ProbeMonitorRun

    run = db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == probe.run_id).one()
    scope = str((run.config or {}).get("scope", "all"))  # `context_from_row`'s default
    rows, to_run = [], []
    for plan in planned:
        # Exactly one CURRENT row or none: `uq_wsp_capture_source` for a held-out set (its evaluation id is never
        # NULL here), `uq_wsp_capture_calibration` for the calibration set. A superseded row is history, never current.
        row = _current_capture(db, probe, plan)
        stale = row is not None and row.status == "ready" and calibration_capture_stale(probe, row) is not None
        if row is None or stale:
            fresh = WspActivationCapture(id=_wac_id(), probe_id=probe.id, dataset_id=plan.view_id, role=plan.role,
                                         source_evaluation_id=plan.evaluation_id, layer=probe.layer,
                                         scope=scope, status="pending")
            if stale:
                # A ready calibration capture checked against a bar the probe no longer serves (OR3-044a-L3) is
                # superseded, never reset (C044a-M1). Written first, so the current-row index never sees two.
                row.superseded_at, row.superseded_by = utc_now(), fresh.id
                db.flush()
            db.add(fresh)
            row = fresh
            to_run.append(row)
        elif row.status not in _KEPT:
            # A failed row is re-armed: no bundle names it (only ready captures are ever bundled).
            row.status, row.error, row.parity, row.finished_at, row.heartbeat_at = "pending", None, None, None, None
            to_run.append(row)
        rows.append(row)
    _BETWEEN_PLAN_AND_WRITE()
    gpu = None
    if check_gpu and to_run:
        # 2026-11-19 (live): a capture queued onto a full card failed every set CUDA out of memory. Refused HERE, before
        # the rows are committed (the caller closes the session without committing, so nothing is written and a
        # re-request plans afresh), and only when a set will actually run.
        gpu = capture_gpu_check(db, probe)
        refuse_if_gpu_too_full(gpu)
    db.commit()   # releases the plan lock
    return {"probe": probe, "rows": rows, "to_run": [r.id for r in to_run], "gpu": gpu}


@router.post("/captures", status_code=202, dependencies=[approval_gated("capture_wsp_heldout")],
             responses=gated_responses(WA.WspCaptureStarted, status_code=202))
async def start_capture(body: CaptureRequest) -> Dict[str, Any]:
    """Capture a probe's held-out sets (GPU, queue `extraction`), each checked for parity with its stored evaluation.
    An agent needs an approved `capture_wsp_heldout` request, which this route spends (044 FTASKS 4.4). Refused 409
    `gpu_memory_insufficient`, before anything is queued or written, when a set would run and no card the run may use
    has the memory its model needs now (`wsp_capture.capture_gpu_check`)."""
    from ....services.wsp_capture import CaptureRefused
    from ....workers.wsp_tasks import capture_wsp_heldout
    from .probe_monitors import refuse_precision_mismatch, refuse_render_mismatch

    await refuse_precision_mismatch(body.probe_id)
    await refuse_render_mismatch(body.probe_id)

    def _work():
        db = _session()
        try:
            planned = plan_capture_rows(db, body.probe_id, body.dataset_ids, check_gpu=True)
            return planned["to_run"], [capture_view(r) for r in planned["rows"]], planned["gpu"]
        finally:
            db.close()

    try:
        to_run, captures, gpu = await run_in_threadpool(_work)
    except CaptureRefused as refused:
        raise HTTPException(status_code=refused.status, detail=refused.detail())
    task_id = task_queue_id = None
    if to_run:
        # 2026-11-27: with its task-queue row, so the capture shows set k of n and rows x of y while it runs.
        from ....services.gpu_dispatch import dispatch_gpu_task
        from ....workers.jlens_progress import WSP_HELDOUT_CAPTURE
        from ._task_rows import fail_unsent_row, open_task_row

        celery_id, task_queue_id = await open_task_row(
            WSP_HELDOUT_CAPTURE, body.probe_id, what=f"WSP held-out capture of probe {body.probe_id}")
        try:
            task_id = dispatch_gpu_task(capture_wsp_heldout, gpu_request="auto",
                                        kwargs={"probe_id": body.probe_id, "capture_ids": to_run},
                                        task_id=celery_id).id
        except Exception as exc:
            await fail_unsent_row(celery_id, exc)
            raise
    return {"probe_id": body.probe_id, "task_id": task_id, "task_queue_id": task_queue_id, "started": to_run,
            "captures": captures, "gpu": gpu}


@router.get("/captures/preflight",
            responses={200: {"model": WA.WspCapturePreflight, "description": "What a capture would do now"}})
async def capture_preflight(probe_id: str = Query(..., min_length=1, max_length=64),
                            dataset_ids: Optional[List[str]] = Query(None, max_length=64)) -> Dict[str, Any]:
    """What `POST /wsp/captures` would do now, and whether it would be refused, without writing or queueing anything
    (read by `capture_wsp_heldout` before it files an approval, and again before it spends one): the sets with their
    rows, which would run (a set already ready, pending or running is kept), and the GPU check
    (`wsp_capture.capture_gpu_check`). The route's own refusals answer here as they would there (404, 409)."""
    from ....models.probe_monitor import ProbeMonitor
    from ....services.wsp_capture import CaptureRefused, capture_gpu_check, plan_capture
    from .probe_monitors import refuse_precision_mismatch, refuse_render_mismatch

    await refuse_precision_mismatch(probe_id)
    await refuse_render_mismatch(probe_id)

    def _work():
        db = _session()
        try:
            probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == probe_id).first()
            if probe is None:
                raise HTTPException(status_code=404, detail={"code": "probe_not_found", "message": f"probe {probe_id}"})
            sets = []
            for plan in plan_capture(db, probe, dataset_ids):
                row = _current_capture(db, probe, plan)
                status = None if row is None else row.status
                runs = row is None or status not in _KEPT or _stale_ready(probe, row)
                sets.append({"dataset_id": plan.view_id, "role": plan.role, "rows": int(len(plan.positions)),
                             "capture": None if row is None else row.id, "status": status, "would_run": bool(runs)})
            gpu = capture_gpu_check(db, probe)
            return sets, gpu
        finally:
            db.rollback()
            db.close()

    try:
        sets, gpu = await run_in_threadpool(_work)
    except CaptureRefused as refused:
        raise HTTPException(status_code=refused.status, detail=refused.detail())
    to_run = [s for s in sets if s["would_run"]]
    refusal = None
    if to_run and gpu.get("checked") and gpu.get("fits") is False:
        refusal = {"code": "gpu_memory_insufficient", "message": gpu["message"]}
    return {"probe_id": probe_id, "sets": sets, "to_run": len(to_run), "rows_to_run": sum(s["rows"] for s in to_run),
            "gpu": gpu, "refusal": refusal}


def _current_capture(db: Any, probe: Any, plan: Any) -> Any:
    """The probe's CURRENT capture row for one planned set, or None (as `plan_capture_rows` finds it)."""
    from ....models.wsp import WspActivationCapture

    query = db.query(WspActivationCapture).filter(WspActivationCapture.probe_id == probe.id,
                                                  WspActivationCapture.dataset_id == plan.view_id,
                                                  WspActivationCapture.role == plan.role,
                                                  WspActivationCapture.superseded_at.is_(None))
    query = (query.filter(WspActivationCapture.source_evaluation_id.is_(None)) if plan.calibration
             else query.filter(WspActivationCapture.source_evaluation_id == plan.evaluation_id))
    return query.one_or_none()


def _stale_ready(probe: Any, row: Any) -> bool:
    from ....services.wsp_capture import calibration_capture_stale

    return row.status == "ready" and calibration_capture_stale(probe, row) is not None


@router.get("/captures")
async def list_captures(probe_id: str = Query(..., min_length=1, max_length=64)) -> Dict[str, Any]:
    """A probe's captures with their parity records (any caller: no activation or text is served)."""
    from ....models.wsp import WspActivationCapture

    def _work():
        db = _session()
        try:
            rows = (db.query(WspActivationCapture).filter(WspActivationCapture.probe_id == probe_id)
                    .order_by(WspActivationCapture.created_at.desc()).all())
            return [capture_view(r) for r in rows]
        finally:
            db.close()

    return {"probe_id": probe_id, "captures": await run_in_threadpool(_work)}


# ── sensitivity (FR-7; task 5.4) ──────────────────────────────────────────────

MARK_WORDS = ("a sensitivity mark is the operator's decision about their own data, made in miWorkbench; an agent "
              "never marks a dataset (P11).")


class SensitivityMarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=64)
    value: Literal["sensitive", "not_sensitive"]
    reason: str = Field(min_length=1, max_length=2000)


def _dataset_sensitivity(db: Any, dataset_id: str) -> Dict[str, Any]:
    from ....models.probe_monitor import ProbeMonitorDataset
    from ....services.wsp_sensitivity import effective, source_sensitivity

    views = db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.dataset_id == dataset_id).all()
    if views:
        return effective(db, views)
    # A dataset no view registers yet: its only basis is a mark.
    from types import SimpleNamespace

    one = source_sensitivity(db, SimpleNamespace(dataset_id=dataset_id, dataset_version=None))
    return {"value": one.value, "basis": one.basis}


def _uuid_or_404(dataset_id: str):
    import uuid

    try:
        return uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail={"code": "dataset_not_found", "message": dataset_id})


@router.post("/sensitivity-marks", status_code=201,
             dependencies=[Depends(refuse_agent_callers("operator_or_service_only", MARK_WORDS))])
async def mark_sensitivity(body: SensitivityMarkRequest) -> Dict[str, Any]:
    """Record the operator's sensitivity mark on a downloaded dataset (append-only), and answer its effective value
    now. A mark decides only where miDataworks states nothing, and may raise `not_sensitive`; it never lowers."""
    from ....models.dataset import Dataset
    from ....models.wsp import WspSensitivityMark

    dataset_uuid = _uuid_or_404(body.dataset_id)

    def _work():
        db = _session()
        try:
            if db.get(Dataset, dataset_uuid) is None:
                raise HTTPException(status_code=404, detail={"code": "dataset_not_found", "message": body.dataset_id})
            mark = WspSensitivityMark(dataset_id=dataset_uuid, value=body.value, reason=body.reason)
            db.add(mark)
            db.commit()
            return {"mark": {"id": mark.id, "dataset_id": str(mark.dataset_id), "value": mark.value,
                             "reason": mark.reason, "decided_by": mark.decided_by,
                             "created_at": mark.created_at.isoformat()},
                    "effective": _dataset_sensitivity(db, dataset_uuid)}
        finally:
            db.close()

    return await run_in_threadpool(_work)


@router.get("/sensitivity")
async def get_sensitivity(dataset_id: str = Query(..., min_length=1, max_length=64)) -> Dict[str, Any]:
    """A dataset's effective sensitivity and its basis, read live (any caller)."""
    from ....core.clock import utc_now

    dataset_uuid = _uuid_or_404(dataset_id)

    def _work():
        db = _session()
        try:
            return {"dataset_id": dataset_id, **_dataset_sensitivity(db, dataset_uuid),
                    "read_at": utc_now().isoformat()}
        finally:
            db.close()

    return await run_in_threadpool(_work)


# ── bundles (FR-2, FR-3; tasks 5.3, 5.6, 6.6) ─────────────────────────────────


class BundleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_id: str = Field(min_length=1, max_length=64)
    #: FU-7a C1: exactly these held-out views go into the bundle (each by its newest ready capture), plus the
    #: calibration capture, which always goes in. Omitted = every held-out set the probe has a ready capture of.
    heldout_view_ids: Optional[List[str]] = Field(None, min_length=1, max_length=64)
    #: FU-7a C2: views of role `supplement` whose ready captures go into `practice.supplements` (validation only), in
    #: this order. Omitted = none.
    supplement_view_ids: Optional[List[str]] = Field(None, min_length=1, max_length=16)


def _bundle_or_404(db: Any, bundle_id: str) -> Any:
    from ....models.wsp import WspBundle

    bundle = db.query(WspBundle).filter(WspBundle.id == bundle_id).first()
    if bundle is None:
        raise HTTPException(status_code=404, detail={"code": "bundle_not_found", "message": bundle_id})
    return bundle


def _baselines_of(db: Any, bundle_id: str) -> List[Dict[str, Any]]:
    from ....models.wsp import WspBaseline

    rows = db.query(WspBaseline).filter(WspBaseline.bundle_id == bundle_id).order_by(WspBaseline.created_at).all()
    return [{"kind": r.kind, "state": ("pending" if (r.fit or {}).get("pending") else
                                       "refused" if (r.fit or {}).get("refused") else "fitted"),
             "bar": r.bar, "transform": r.transform, "fit": r.fit} for r in rows]


def bundle_view(db: Any, bundle: Any) -> Dict[str, Any]:
    """The `/summary` answer: the row and its manifest without `files[]`."""
    manifest = dict(bundle.manifest or {})
    manifest.pop("files", None)   # the MCP view (R1A-W18): no file names, no route to a file
    baselines = _baselines_of(db, bundle.id)
    return {"bundle_id": bundle.id, "probe_id": bundle.probe_id, "bundle_kind": bundle.bundle_kind,
            "status": bundle.status, "error": bundle.error, "bundle_sha256": bundle.bundle_sha256,
            "sensitivity_snapshot": {"value": bundle.sensitivity, "basis": bundle.sensitivity_basis},
            "capture_ids": list(bundle.capture_ids or []),
            "baseline_pending": [b["kind"] for b in baselines if b["state"] == "pending"],
            "baselines": baselines, "manifest": manifest or None,
            # FR-1 (R2 review fix OR2-044a-M3): the parent's bar as the scorer cuts it, beside the bar as served
            # (`manifest.operating_point.threshold`) and their difference; null for a bundle built before it was kept.
            "parent_bar": bundle.scorer_parent_bar,
            "created_at": bundle.created_at.isoformat() if bundle.created_at else None}


@router.post("/bundles", status_code=202)
async def start_bundle(body: BundleRequest) -> Dict[str, Any]:
    """Build a bundle for a probe (CPU, queue `wsp`). Refused now, before queueing, when the probe reads SAE features
    (`bundle_kind_unsupported`) or has no ready held-out capture (`capture_not_ready`).

    FU-7a (2026-11-25): `heldout_view_ids` names exactly which held-out sets go in (the calibration capture always
    does) and `supplement_view_ids` the validation-only practice supplements. Each named view must exist (404
    `view_not_found`), have the right role (422 `bundle_request_invalid`) and a ready capture (409 `capture_not_ready`).
    A held-out set miForge has already scored is refused 409 `heldout_reused`, read from miForge's own held-out uses
    (503 `miforge_not_configured` / `heldout_uses_unavailable` when they cannot be read: never assumed); the build
    refuses `heldout_reused` too when a set shares a row key with an earlier bundle of the probe."""
    from ....core.config import settings
    from ....models.probe_monitor import ProbeMonitor
    from ....models.wsp import WspBundle
    from ....services.wsp_bundle import BundleRefused, calibration_capture_missing, resolve_request
    from ....services.wsp_heldout_uses import read_heldout_uses, refuse_used, used_by_us
    from ....workers.wsp_tasks import build_wsp_bundle

    def _work():
        db = _session()
        try:
            probe = db.query(ProbeMonitor).filter(ProbeMonitor.id == body.probe_id).first()
            if probe is None:
                raise HTTPException(status_code=404, detail={"code": "probe_not_found", "message": body.probe_id})
            if getattr(probe, "variant", "dense") == "sae":
                raise HTTPException(status_code=422, detail={
                    "code": "bundle_kind_unsupported",
                    "message": "a probe over SAE features has no bundle kind; phase 4 bundles an SAE itself"})
            try:
                ready, supplements = resolve_request(db, probe, body.heldout_view_ids, body.supplement_view_ids)
                if not ready:
                    raise HTTPException(status_code=409, detail={
                        "code": "capture_not_ready",
                        "message": "capture the probe's held-out sets first (POST /wsp/captures) and wait for parity"})
                missing = calibration_capture_missing(probe, ready)  # OR1-044a-1
                if missing:
                    raise HTTPException(status_code=409, detail={"code": "capture_not_ready", "message": missing})
                # FU-7a C1: miForge's held-out uses are the authority on what was scored; read, never assumed.
                uses = read_heldout_uses(settings.miforge_api_url)
                refuse_used(ready, supplements, uses)
            except BundleRefused as refused:
                raise HTTPException(status_code=refused.status, detail=refused.detail())
            bundle = WspBundle(probe_id=probe.id, status="building")
            db.add(bundle)
            db.commit()
            return bundle.id, used_by_us(uses)
        finally:
            db.close()

    bundle_id, used = await run_in_threadpool(_work)
    task = build_wsp_bundle.delay(bundle_id=bundle_id, heldout_view_ids=body.heldout_view_ids,
                                  supplement_view_ids=body.supplement_view_ids, used_heldout=used)
    return {"bundle_id": bundle_id, "task_id": task.id, "status": "building"}


@router.get("/bundles/{bundle_id}",
            dependencies=[Depends(refuse_agent_callers("operator_or_service_only"))])
async def get_bundle(bundle_id: str) -> Dict[str, Any]:
    """EXACTLY the bundle's manifest, `files[]` included (FTDD §5.2, FR-3): the `miforge.wsp-bundle/v1` document
    miForge's import validates, whose RFC 8785 digest is `bundle_sha256`. 409 `bundle_not_ready` until the bundle is
    `ready`. The operator's view of the row (status, baselines, parent bar, sensitivity snapshot) is `/summary`.

    R3 review fix (OR3-044a-H1), 2026-11-17: this route answered the operator's wrapper (`{bundle_id, status, …,
    manifest: {…}}`), so miForge's `fetch_bundle_manifest` → `_check_manifest` refused every real bundle
    `wsp_bundle_invalid`; `test_bundle_route_cross_repo.py` runs miForge's own client and checks on this answer."""
    def _work():
        db = _session()
        try:
            bundle = _bundle_or_404(db, bundle_id)
            if bundle.status != "ready" or not bundle.manifest:
                raise HTTPException(status_code=409, detail={"code": "bundle_not_ready", "message": bundle.status})
            return dict(bundle.manifest)
        finally:
            db.close()

    return await run_in_threadpool(_work)


@router.get("/bundles/{bundle_id}/summary")
async def get_bundle_summary(bundle_id: str) -> Dict[str, Any]:
    """The operator's and the MCP view of the bundle: the row (status, error, baselines, parent bar, sensitivity
    snapshot) and its manifest WITHOUT `files[]`: no file names, so nothing here leads to an activation (any
    caller). miForge reads the manifest itself from `GET /bundles/{id}`."""
    def _work():
        db = _session()
        try:
            return bundle_view(db, _bundle_or_404(db, bundle_id))
        finally:
            db.close()

    return await run_in_threadpool(_work)


def bundle_sensitivity_now(db: Any, bundle: Any) -> Dict[str, Any]:
    """FR-7 recomputed LIVE for the bundle's sources (P41): the practice view and every referenced capture's view."""
    from ....core.clock import utc_now
    from ....models.probe_monitor import ProbeMonitorDataset
    from ....models.wsp import WspActivationCapture
    from ....services.wsp_sensitivity import effective

    practice = (bundle.manifest or {}).get("practice", {})
    practice_id = practice.get("view_id")
    # FU-7a C2: miForge reads every supplement row too, so each supplement's view is a source.
    supplement_ids = [s.get("view_id") for s in practice.get("supplements") or []]
    listed = list(dict.fromkeys(bundle.capture_ids or []))
    captures = (db.query(WspActivationCapture).filter(WspActivationCapture.id.in_(listed)).all() if listed else [])
    view_ids = list(dict.fromkeys([practice_id] + supplement_ids + [c.dataset_id for c in captures]))
    views = db.query(ProbeMonitorDataset).filter(ProbeMonitorDataset.id.in_([v for v in view_ids if v])).all()
    out = {"bundle_id": bundle.id, **effective(db, views), "read_at": utc_now().isoformat()}
    # R1 review fix (OR1-044a-3), 2026-11-16: a source that no longer resolves (a capture row or a view gone, or no
    # practice view) is not "fewer sources": the answer is `unknown`, which no mark lifts, never a less-sensitive
    # value read from what is left.
    unresolved = ([] if practice_id else ["practice view"]) + \
        [f"capture {c}" for c in listed if c not in {row.id for row in captures}] + \
        [f"view {v}" for v in view_ids if v and v not in {row.id for row in views}]
    if unresolved:
        out["value"] = "unknown"
        out["unresolved"] = unresolved
    return out


@router.get("/bundles/{bundle_id}/sensitivity",
            dependencies=[Depends(refuse_agent_callers("operator_or_service_only"))])
async def get_bundle_sensitivity(bundle_id: str) -> Dict[str, Any]:
    """The bundle's sensitivity RECOMPUTED NOW (registered, miDataworks live, marks), never the manifest's sealed
    snapshot: miForge reads this at push time (P41, R1B-X1)."""
    def _work():
        db = _session()
        try:
            return bundle_sensitivity_now(db, _bundle_or_404(db, bundle_id))
        finally:
            db.close()

    return await run_in_threadpool(_work)


@router.get("/bundles/{bundle_id}/files/{name}",
            dependencies=[Depends(refuse_agent_callers("operator_or_service_only"))])
async def get_bundle_file(bundle_id: str, name: str):
    """One file the manifest lists, by its listed name only (FR-3); `X-Content-SHA256` is the listed digest. Any other
    name — a capture file, `../x`, the manifest itself — is 404."""
    from fastapi.responses import FileResponse

    from ....services.wsp_bundle import file_path_of

    def _work():
        db = _session()
        try:
            bundle = _bundle_or_404(db, bundle_id)
            if bundle.status != "ready":
                raise HTTPException(status_code=409, detail={"code": "bundle_not_ready", "message": bundle.status})
            path = file_path_of(bundle, name)
            entry = next((f for f in bundle.manifest.get("files", []) if f.get("name") == name), None)
            return path, entry
        finally:
            db.close()

    path, entry = await run_in_threadpool(_work)
    if path is None or entry is None or not path.is_file():
        raise HTTPException(status_code=404, detail={"code": "bundle_file_not_listed",
                                                     "message": f"{name!r} is not a file of bundle {bundle_id}"})
    return FileResponse(path, media_type="application/octet-stream", filename=entry["name"],
                        headers={"X-Content-SHA256": entry["sha256"]})


# ── the scorer (FR-5, FR-6; tasks 7.2 – 7.6) ──────────────────────────────────

from ....schemas.wsp_scoring import WspScoreRequest  # noqa: E402

SCORER_BUSY_RETRY_AFTER_S = 5
#: What an abandoned call answers. Nobody reads it (the client has gone), so it is no contract refusal: nginx's
#: "client closed request".
CLIENT_CLOSED_STATUS = 499


async def watch_client(receive: Any, stop: threading.Event) -> None:
    """Set `stop` when the HTTP client disconnects (2026-11-22, the D128 scorer defect: miForge gave up at 600 s and
    the scorer went on for the rest of the hour holding `SEMAPHORE`, so 46 calls got 429). FastAPI has read the JSON
    body before the route runs, so the next ASGI message is `http.disconnect`, sent when the connection is lost
    (miServer's `generation_stop.ClientWatch` waits the same way). A broken `receive` never fails the call."""
    try:
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                stop.set()
                return
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.warning("wsp score: the client watch failed; the call is not watched", exc_info=True)


@router.post("/score", dependencies=[Depends(refuse_agent_callers("operator_or_service_only"))])
async def score_heads(body: WspScoreRequest, request: Request) -> Dict[str, Any]:
    """Score heads on a bundle's held-out rows (`miworkbench-wsp-scoring-v1.json`). The operator and miForge's backend
    only; one call at a time per process (429 `scorer_busy` with `Retry-After`); every refusal is `{code, message}`
    with the contract's status."""
    from ....services import wsp_score

    if not wsp_score.SEMAPHORE.acquire(blocking=False):
        raise HTTPException(status_code=429, headers={"Retry-After": str(SCORER_BUSY_RETRY_AFTER_S)},
                            detail={"code": "scorer_busy", "message": "a score call is running; retry shortly"})
    # FR-5's caller address. Behind the ingress the socket peer is the proxy (uvicorn runs without
    # `--forwarded-allow-ips`), so the record keeps the peer AND the `X-Forwarded-For` chain as sent — unverified, the
    # header `steering.get_client_id` reads (R2 review fix OR2-044a-L2, 2026-11-17).
    caller = {"address": request.client.host if request.client else None,
              "forwarded_for": request.headers.get("x-forwarded-for"),
              "user_agent": request.headers.get("user-agent")}

    stop = threading.Event()

    def _work():
        db = _session()
        try:
            return wsp_score.score(db, body, caller=caller, stop=stop)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    watch = None
    try:
        # The scorer checks `stop` before every bucket; the thread is not cancellable, so this awaits its end and the
        # semaphore is released only once nothing scores any more.
        watch = asyncio.ensure_future(watch_client(request.receive, stop))
        return await run_in_threadpool(_work)
    except wsp_score.ScoreAbandoned as abandoned:
        logger.warning("wsp score: %s (bundle %s, caller %s)", abandoned, body.bundle_id, caller)
        return Response(status_code=CLIENT_CLOSED_STATUS)
    except wsp_score.ScoreRefused as refused:
        raise HTTPException(status_code=refused.status, detail=refused.detail())
    finally:
        if watch is not None:
            watch.cancel()
        wsp_score.SEMAPHORE.release()


# ── candidate import from files (FR-8; tasks 8.2 – 8.6) ───────────────────────


class ImportAcknowledgement(BaseModel):
    """Importing a verdict other than `improved` needs the operator to name the verdict they accept, and why. An
    `improved` verdict that is `improved_unconfirmed` under miForge's standing rule (FU-7b) needs the same, naming
    `improved_unconfirmed` (2026-11-25)."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["not_significant", "regressed", "unmeasured", "improved_unconfirmed"]
    reason: str = Field(min_length=1, max_length=2000)


class CandidateImportRequest(BaseModel):
    """FR-8 (R3B-11): JSON, never multipart, so the app's 1 MiB body cap bounds it and no route declares
    `UploadFile`; a probe patch is tens of KB."""

    model_config = ConfigDict(extra="forbid")

    run_document: Dict[str, Any]
    patch_manifest: Dict[str, Any]
    edit_npz_b64: str = Field(min_length=1, max_length=1_048_576)
    acknowledgement: Optional[ImportAcknowledgement] = None


@router.post("/candidates/import", status_code=202, dependencies=[approval_gated("import_wsp_candidate")],
             responses=gated_responses(WA.WspCandidateImporting, status_code=202))
async def import_candidate(body: CandidateImportRequest) -> Dict[str, Any]:
    """Import a verdicted miForge candidate as a NEW probe, never selected, that earns its rung at import (it is
    created at rung 0, then calibrated and evaluated from the held-out captures; FU-5). Every check of FR-8 runs here
    first and refuses `{code, message}`; the import itself (P23, calibration and evaluation from the captures, the rung)
    runs on the `wsp` worker, which repeats every check. An agent needs an approved `import_wsp_candidate` request,
    which this route spends (044 FTASKS 14.2; it replaced phase 2's refusal of every agent, R1A-W10)."""
    from ....services import wsp_import
    from ....workers.wsp_tasks import import_wsp_candidate

    acknowledgement = body.acknowledgement.model_dump() if body.acknowledgement else None

    def _work():
        db = _session()
        try:
            checked = wsp_import.check(db, run_document=body.run_document, patch_manifest=body.patch_manifest,
                                       edit_npz=wsp_import.decode_arrays(body.edit_npz_b64),
                                       acknowledgement=acknowledgement)
            return {"bundle_id": checked.bundle.id, "parent_probe_id": checked.parent.id,
                    "miforge_run_id": checked.run["run_id"], "score_call_id": checked.call.id}
        finally:
            db.rollback()
            db.close()

    try:
        checked = await run_in_threadpool(_work)
    except wsp_import.ImportRefused as refused:
        raise HTTPException(status_code=refused.status, detail=refused.detail())
    # 2026-11-27 (MCP gap, ENH-010 phase-2 round): a task-queue row under the Celery id the task is sent with, opened
    # BEFORE the send (the ENH-006 G17 order, `_task_rows`), so `get_task_status` answers the import while it runs and
    # after: the task completes it as the new probe, or fails it with the refusal's code. A row that cannot be written
    # is a 503 and nothing is sent.
    from ....workers.jlens_progress import WSP_CANDIDATE_IMPORT
    from ._task_rows import fail_unsent_row, open_task_row

    celery_id, task_queue_id = await open_task_row(
        WSP_CANDIDATE_IMPORT, checked["parent_probe_id"], gpu_request=None,
        what=f"import of miForge run {checked['miforge_run_id']}")
    try:
        task = import_wsp_candidate.apply_async(
            kwargs={"run_document": body.run_document, "patch_manifest": body.patch_manifest,
                    "edit_npz_b64": body.edit_npz_b64, "acknowledgement": acknowledgement},
            task_id=celery_id)
    except Exception as exc:
        await fail_unsent_row(celery_id, exc)
        raise
    return {**checked, "task_id": task.id, "task_queue_id": task_queue_id, "status": "importing"}
