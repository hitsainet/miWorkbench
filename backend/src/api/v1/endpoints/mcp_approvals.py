"""
Agent approval endpoints (Feature 010, operator-approval mode).

When the MCP server runs with MCP_STEERING_APPROVAL=true, agent steering
calls land here as pending requests instead of hitting /steering/async/*
directly. An operator approves or denies them (the approvals banner above every panel). Since 045 (FR-13a, A16 D5)
steering has ONE approval mechanism, the caller-executed one: approving only marks the row, and the re-call with
the id is consumed by the steering route's own `require_consumed_approval` dependency, whoever sends it (the MCP
steer tools pass it as the approval-request header). Before 045 the backend ran an approved steering request
itself by a self-call and stored `steering_task_id`; historical rows keep that id
(`e045_steering_switchover.py` moved them to `consumed`).

Feature 034 (FR-20) adds CALLER-EXECUTED tools. `miserver_load_model` evicts the model miServer serves,
so an agent's load waits for the operator too — but the backend has no miServer URL and cannot run
the load itself (T-98). So approving a caller-executed request only marks it `approved`; the MCP
tool re-calls with the approval id, CONSUMES it here (`POST /{id}/consume`, one conditional
UPDATE — the only once-guarantee) and then loads. Approve is the operator's act and consume the
executor's: no MCP code path can approve — and since OSD-56 (2026-10-24) the two decision routes
refuse any caller sending `X-MiStudio-Agent` (403 `approval_operator_only`).

Caller-executed approvals EXPIRE: pending, or approved and unconsumed, for longer than
`mcp_approval_ttl_hours` (default 24). Expiry is lazy — every route runs `expire_stale` first —
so a stale request never reads `pending`. Steering approvals lapse like every caller row since 045 (FR-13a; S3-03's
exclusion is withdrawn with the self-call).

ENH-008 (2026-10-28): miServer decides its own requests now, so the four `millm_*` tools no longer file here (the  (p9: keep, rename history)
proxies relay miServer's pending answer). Each row records who asked (`requested_by`) and who decided (`decided_by`),
`GET` computes `expires_at`, and `decided=true` lists the Approvals page's History, newest decision first.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.runtime_settings import effective
from src.core.deps import get_db
from src.models.agent_approval import AgentApprovalRequest, ApprovalStatus
from src.workers.websocket_emitter import emit_progress

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp/approvals", tags=["mcp-approvals"])

#: The header every MCP request carries (`mcp_server/client.py`); the frontend never sends it.
AGENT_HEADER = "X-MiStudio-Agent"

#: ENH-008: `decided_by` on approve and deny. The suite has no login (operator decision 2026-10-06), so the
#: decider is always "operator"; an expiry records "system".
OPERATOR = "operator"


def refuse_agent_decisions(request: Request) -> None:
    """OSD-56 (040 Appendix C.6, 040-A5): approving or denying is the OPERATOR's act.

    On `/approve` and `/deny` only — never on `POST /mcp/approvals` (agents file requests) or
    `/consume` (the MCP tool spends a caller-executed approval). Any value of the header, even an
    empty or malformed one, is refused, as `refuse_agents` does on the purge routes.
    """
    if request.headers.get(AGENT_HEADER) is not None:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "approval_operator_only",
                "message": (
                    "Approving or denying is the operator's, from miWorkbench's approvals banner. "
                    "An agent files a request and waits; it can never decide one."
                ),
            },
        )


class Execution(str, Enum):
    """Who carries out an approved request."""

    #: The backend, on approve, by an internal self-call (steering before 045). No tool uses it since 045 FR-13a
    #: (A16 D5): kept so `test_approvals_no_settings_change_requests.py` still reads the table's kinds.
    BACKEND = "backend"
    #: The MCP tool that asked, on re-call, after consuming the approval (miServer model load).
    CALLER = "caller"
    #: Feature 040 S4 (FTDD C.6, 040-A4): nothing is executed or consumed — approving WRITES a durable record (an
    #: operator `approve` row in `code_rule_approvals`, per code hash) in the same transaction that marks the request
    #: approved. It persists until the operator revokes it.
    RECORD = "record"


#: Steering tools: caller-executed since 045 FR-13a (A16 D5), consumed by the steering routes' own REST gate.
STEERING_TOOLS = frozenset({"steer_compare", "steer_sweep", "steer_combined"})

#: 045 (FTID §11a): the actions the REST gate files, beyond the five probe gates above them, the steering three
#: included (FTASKS 21.6).
REST_GATED_ACTIONS = STEERING_TOOLS | frozenset({
    "start_circuit_capture", "confirm_circuit_capture", "run_attribution_pass", "validate_circuit_edges",
    "reproduce_validation", "run_circuit_faithfulness", "calibrate_circuit_strength", "reproduce_calibration",
    "record_steering_samples", "fit_jlens_artifact", "revalidate_jlens_artifact", "compute_jlens_band_report",
    "jlens_readout", "jlens_probe", "jlens_decompose", "build_jlens_template_lens", "run_jlens_intervention",
    "acquire_jlens_artifact", "download_model", "redownload_model", "extract_activations", "retry_extraction",
    "score_probe_monitor", "score_probe_monitor_batch", "build_probe_definition", "score_probe_family_batch",
    "train_sae", "evaluate_training", "resume_training", "extract_features", "batch_extract_features",
    "start_labeling", "start_labeling_resume_sweep", "neuronpedia_export", "compute_dashboard_data",
    "push_to_neuronpedia_local", "retry_task", "run_labeling_trial", "trigger_nlp_analysis", "run_enhanced_labeling",
    "run_enhanced_labeling_batch", "try_judge_prompt", "publish_probe_definition", "publish_probe_family",
    "upload_sae", "publish_jlens_artifact",
})

#: ENH-010 044 (FTASKS 4.4, 14.2; FTDD §5.5): the WSP actions. Each is filed by its MCP tool (`capture_wsp_heldout`,
#: `import_wsp_candidate`) or by another agent's REST call, and spent by the ROUTE (`POST /wsp/captures`,
#: `POST /wsp/candidates/import`, `api/v1/endpoints/wsp.approval_gated`), never by `/consume` from the tool.
WSP_ACTIONS = frozenset({"capture_wsp_heldout", "import_wsp_candidate"})

#: tool name → (who executes it, backend self-call path or None).
APPROVABLE_TOOLS: dict[str, tuple[Execution, Optional[str]]] = {
    # ENH-008 (D1, 2026-10-28): the four `miserver_*` tools that used to file here — `millm_load_model`,
    # `millm_request_probe_limits`, `millm_measure_probe_capacity`, `millm_import_rule_set` — are GONE from this
    # table. miServer decides its own requests in its own queue, and the proxies relay its pending answer; filing one
    # here is refused 422. Requests still waiting at switch-over were expired by `e008_millm_approvals_moved.py`.  # p9: keep, a record's name
    # 2026-10-19 (MCP gap closure): an agent's probe training run and probe evaluation each hold
    # a GPU for minutes to hours, behind which every other GPU job queues. Caller-executed like
    # the miServer load: approving marks the row, the MCP tool consumes it and submits exactly the
    # payload the operator saw. Gated by `MCP_PROBE_GPU_APPROVAL` (default on).
    "start_probe_training_run": (Execution.CALLER, None),
    "evaluate_probe_monitor": (Execution.CALLER, None),
    # 2026-10-21 (judge-prompt library): an agent's judge run against miServer is thousands of
    # generations on the GPU that serves miServer's users, and may load the judge model (evicting the
    # served one). Gated when the endpoint is miServer (formerly miLLM), failing closed — see  # p9: keep, rename history
    # `judge_endpoint_is_miserver` (042 FR-038 P0 half: both stems, configured host, IP literals resolved).
    "start_probe_judge_run": (Execution.CALLER, None),
    # Feature 038 (D34): a spot-check judged by miServer queues live chats behind every judge call.
    "start_probe_spot_check": (Execution.CALLER, None),
    # Feature 040 S3 (review R1-5): the GPU arm of a recalibration (`recut_windows=true`) re-scores a probe's
    # per-window negatives on a card; the arithmetic arm never asks.
    "recalibrate_probe_monitor": (Execution.CALLER, None),
    # Feature 040 S4 (FTDD Appendix C.6): an agent's request to run operator code (a code rule, by hash) in the
    # isolated runner. Approving records the operator's per-hash approval; it is never consumed.
    "code_rule_approve": (Execution.RECORD, None),
    # 045 FR-13 (A12, A29; FTID §11a's action table): every REST route that starts GPU work, makes an outside paid call
    # or publishes to the Hub files under one of these when an agent other than the MCP pod calls it, and consumes it
    # on the approved re-call (`core.approval_gate.require_consumed_approval`). Caller-executed: approving marks the
    # row, the route runs it once. A route an MCP tool calls takes that tool's name.
    **{action: (Execution.CALLER, None) for action in sorted(REST_GATED_ACTIONS)},
    # ENH-010 044: a held-out capture holds a GPU (the probe's model is loaded once); an import registers a new probe
    # from a candidate miForge proposed. Caller-executed: approving marks the row, the route runs it once.
    **{action: (Execution.CALLER, None) for action in sorted(WSP_ACTIONS)},
}

CALLER_TOOLS = frozenset(
    tool for tool, (kind, _) in APPROVABLE_TOOLS.items() if kind is Execution.CALLER
)

#: Review R2 (S4): tools whose approval WRITES a durable record. Each is filed only by its own route (operator code:
#: `POST /code-rules/{sha}/approval-requests`, which runs the code's own cases first and stores the exact document),
#: never through `POST /mcp/approvals`; and an approved one is never consumed, so it never "expires unused".
RECORD_TOOLS = frozenset(
    tool for tool, (kind, _) in APPROVABLE_TOOLS.items() if kind is Execution.RECORD
)


#: The largest `forecast` a request may file, as JSON. A summary line and a few counts are well under it.
FORECAST_MAX_BYTES = 16_384


class ApprovalCreateRequest(BaseModel):
    tool_name: str
    payload: dict[str, Any]
    #: ENH-006 G15: what the asking tool forecast (a probe run's time), shown to the operator beside the
    #: payload. Not approved and not consumed: only `payload` is.
    forecast: Optional[dict[str, Any]] = None

    @field_validator("forecast")
    @classmethod
    def _small(cls, value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        import json

        if value is not None and len(json.dumps(value, default=str)) > FORECAST_MAX_BYTES:
            raise ValueError(f"forecast is larger than {FORECAST_MAX_BYTES} bytes as JSON")
        return value

    @field_validator("tool_name")
    @classmethod
    def _approvable(cls, value: str) -> str:
        if value not in APPROVABLE_TOOLS:
            raise ValueError(
                f"{value!r} is not an approvable tool (one of {sorted(APPROVABLE_TOOLS)})"
            )
        return value


class ConsumeRequest(BaseModel):
    """The executor's claim: this tool, with exactly this payload."""

    tool_name: str
    payload: dict[str, Any]


class ApprovalResponse(BaseModel):
    id: str
    tool_name: str
    payload: dict[str, Any]
    status: str
    reason: Optional[str] = None
    steering_task_id: Optional[str] = None
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    consumed_at: Optional[datetime] = None
    #: ENH-008 (§3.2 step 7): who asked ("mcp", "operator") and who decided ("operator", "system").
    requested_by: Optional[str] = None
    decided_by: Optional[str] = None
    #: ENH-008: when a request that can still lapse will lapse — pending (created + TTL), or approved and not yet
    #: consumed (approved + TTL). NULL for steering (never expires), for an approved record tool (never consumed)
    #: and for every decided row. Computed on read from `mcp_approval_ttl_hours`, never stored.
    expires_at: Optional[datetime] = None
    #: ENH-006 D81: shared by every request approved in one "Approve all" call; NULL otherwise.
    batch_id: Optional[str] = None
    #: ENH-006 G15: what the asking tool forecast when it filed the request; NULL when nothing was.
    forecast: Optional[dict[str, Any]] = None

    model_config = {"from_attributes": True}


class ApprovalListResponse(BaseModel):
    approvals: list[ApprovalResponse]
    total: int


#: Longest `requested_by` kept (the column's width).
REQUESTED_BY_MAX = 64


def requested_by_of(request: Request) -> str:
    """ENH-008: who filed a request — the `X-MiStudio-Agent` value (the MCP server sends "mcp"), else "operator"
    (the UI sends no header). Stripped and cut to the column; never refused here (filing must not break on a
    label)."""
    raw = request.headers.get(AGENT_HEADER)
    if raw is None:
        return "operator"
    return raw.strip()[:REQUESTED_BY_MAX] or "agent"


def expires_at_of(approval: AgentApprovalRequest, ttl_hours: float) -> Optional[datetime]:
    """When this request will lapse if nothing happens, or `None` (see `ApprovalResponse.expires_at`).

    Mirrors `expire_stale` exactly: a pending request lapses `ttl` after it was filed; an approved CALLER request
    (steering included since 045 FR-13a) lapses `ttl` after it was approved unless consumed; a RECORD approval never
    lapses."""
    ttl = timedelta(hours=ttl_hours)
    if approval.status == ApprovalStatus.PENDING.value and approval.created_at is not None:
        return approval.created_at + ttl
    if (approval.status == ApprovalStatus.APPROVED.value and approval.resolved_at is not None
            and approval.tool_name not in RECORD_TOOLS):
        return approval.resolved_at + ttl
    return None


def respond(approval: AgentApprovalRequest, ttl_hours: Optional[float] = None) -> ApprovalResponse:
    """The one serialiser for an approval row (memory: one list needs one serialiser)."""
    if ttl_hours is None:
        ttl_hours = effective().mcp_approval_ttl_hours
    out = ApprovalResponse.model_validate(approval)
    out.expires_at = expires_at_of(approval, ttl_hours)
    return out


class DenyRequest(BaseModel):
    #: ENH-008 (D8c): the UI sends the operator's reason (at most 1000 characters) or
    #: "Rejected by the operator." when the field is left blank.
    reason: Optional[str] = Field(None, max_length=2000)


def _emit_approval_event(event: str, data: dict[str, Any]) -> None:
    emit_progress(channel="mcp/approvals", event=f"approval:{event}", data=data)


async def expire_stale(db: AsyncSession) -> list[str]:
    """Move stale CALLER-EXECUTED approvals to `expired`; return their ids.

    Two conditional statements (steering is no longer excluded: 045 FR-13a made it caller-executed, so an approved
    steering request the agent never came back for lapses like any other):

    * pending longer than the TTL, by `created_at`;
    * approved and not consumed longer than the TTL, by `resolved_at` — an approval the agent
      never came back for must not stay spendable indefinitely.

    `synchronize_session=False`: a criteria UPDATE otherwise evaluates its WHERE in Python against
    the session's loaded rows (the ORM bulk-statement lesson), comparing an aware cutoff to
    whatever the row holds. The routes re-read rows after this runs.
    """
    # Read per call (Feature 036): an operator's override applies at the next approval check.
    ttl_hours = effective().mcp_approval_ttl_hours
    ttl = timedelta(hours=ttl_hours)
    now = datetime.now(timezone.utc)
    cutoff = now - ttl
    model = AgentApprovalRequest
    pending = (
        update(model)
        .where(
            model.status == ApprovalStatus.PENDING.value,
            model.created_at < cutoff,
        )
        .values(
            status=ApprovalStatus.EXPIRED.value,
            resolved_at=now,
            reason=f"expired: pending for more than {ttl_hours} h",
            # ENH-008: nobody decided; the TTL did.
            decided_by="system",
        )
        .returning(model.id)
        .execution_options(synchronize_session=False)
    )
    unconsumed = (
        update(model)
        .where(
            model.status == ApprovalStatus.APPROVED.value,
            model.resolved_at < cutoff,
            model.tool_name.not_in(sorted(RECORD_TOOLS)),
        )
        .values(
            status=ApprovalStatus.EXPIRED.value,
            reason=(
                f"expired: approved but not used within {ttl_hours} h"
            ),
        )
        .returning(model.id)
        .execution_options(synchronize_session=False)
    )
    expired = list((await db.execute(pending)).scalars().all())
    expired += list((await db.execute(unconsumed)).scalars().all())
    if expired:
        await db.commit()
        logger.info("expired %d stale agent approval(s)", len(expired))
        for request_id in expired:
            _emit_approval_event("resolved", {"request_id": request_id, "status": "expired"})
    return expired


@router.post("", response_model=ApprovalResponse, status_code=201)
async def create_approval_request(
    request: ApprovalCreateRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a pending approval request (called by the MCP server)."""
    if request.tool_name in RECORD_TOOLS:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "approval_filed_by_its_route",
                "message": (
                    f"{request.tool_name} is filed by its own route, which runs the code's own cases in "
                    "the isolated runner first (request_code_rule_approval); it cannot be filed here."
                ),
            },
        )
    if request.tool_name in CALLER_TOOLS:
        # 045 A43 (V-1): an agent other than the MCP pod files against a named gated route, validated like a call of
        # it, before any row is written. The MCP pod's and the operator's filings are unchanged.
        from src.core.approval_gate import validate_queue_filing

        await validate_queue_filing(http_request, request.tool_name, request.payload)
    approval = await file_request(request.tool_name, request.payload,
                                  requested_by=requested_by_of(http_request), forecast=request.forecast, db=db)
    return respond(approval)


async def file_request(tool_name: str, payload: dict[str, Any], *, requested_by: str,
                       forecast: Optional[dict[str, Any]] = None,
                       db: Optional[AsyncSession] = None) -> AgentApprovalRequest:
    """THE writer of a pending approval (045 FR-13; FTDD §16.3 commit point 1), shared by `POST /mcp/approvals` and
    the REST gate (`core.approval_gate`). Without `db` it opens its OWN session and commits before returning, so the
    row survives whatever the calling request does next (the gate answers 202 by raising)."""
    if db is None:
        from src.core.database import AsyncSessionLocal

        async with AsyncSessionLocal() as own:
            return await file_request(tool_name, payload, requested_by=requested_by, forecast=forecast, db=own)
    approval = AgentApprovalRequest(
        tool_name=tool_name,
        payload=payload,
        status=ApprovalStatus.PENDING.value,
        requested_by=requested_by,
        forecast=forecast,
    )
    db.add(approval)
    await db.commit()
    await db.refresh(approval)

    _emit_approval_event(
        "created",
        {
            "request_id": approval.id,
            "tool_name": approval.tool_name,
            "summary": _summarize(tool_name, payload),
        },
    )
    return approval


@router.get("", response_model=ApprovalListResponse)
async def list_approval_requests(
    status: Optional[str] = Query(
        None, description="Filter: pending|approved|denied|expired|consumed"
    ),
    decided: bool = Query(
        False,
        description="ENH-008: only DECIDED requests (every status but pending), the most recent decision "
                    "first — the Approvals page's History",
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0, description="Skip this many (History's Show older)"),
    db: AsyncSession = Depends(get_db),
):
    await expire_stale(db)
    model = AgentApprovalRequest
    if decided:
        # ⚠ THE HISTORY FILTER: everything that is no longer waiting, newest decision first.
        query = select(model).where(model.status != ApprovalStatus.PENDING.value).order_by(
            model.resolved_at.desc().nulls_last(), model.created_at.desc(), model.id
        )
    else:
        query = select(model).order_by(model.created_at.desc(), model.id)
    if status:
        query = query.where(model.status == status)
    result = await db.execute(
        query.offset(offset).limit(limit).execution_options(populate_existing=True)
    )
    approvals = result.scalars().all()
    ttl_hours = effective().mcp_approval_ttl_hours
    return ApprovalListResponse(
        approvals=[respond(a, ttl_hours) for a in approvals],
        total=len(approvals),
    )


@router.get("/{request_id}", response_model=ApprovalResponse)
async def get_approval_request(request_id: str, db: AsyncSession = Depends(get_db)):
    await expire_stale(db)
    approval = await _get_or_404(db, request_id)
    return respond(approval)


@router.post("/{request_id}/approve", response_model=ApprovalResponse,
             dependencies=[Depends(refuse_agent_decisions)])
async def approve_request(request_id: str, db: AsyncSession = Depends(get_db)):
    """Approve.

    Caller-executed (the probe GPU starts, steering since 045 FR-13a, every REST-gated action): the row is marked
    approved and nothing else happens here — the re-call consumes it. Record (operator code): the durable record is
    written. An expired request cannot be approved (409).
    """
    await expire_stale(db)
    approval = await _get_or_404(db, request_id)
    if approval.status != ApprovalStatus.PENDING.value:
        raise HTTPException(409, f"Request is {approval.status}, not pending")
    return respond(await _approve_pending(db, approval))


async def _approve_pending(db: AsyncSession, approval: AgentApprovalRequest,
                           batch_id: Optional[str] = None) -> AgentApprovalRequest:
    """THE approve path, shared by `POST /{id}/approve` and `POST /approve-batch` (ENH-006 D81): one request,
    already checked pending, decided by the operator and committed on its own. `batch_id` is stamped on the row
    in the same commit that marks it approved. Raises `HTTPException` when this request cannot be approved."""
    request_id = approval.id
    kind, _ = APPROVABLE_TOOLS.get(approval.tool_name, (None, None))
    if kind is Execution.RECORD:
        # ⚠ ONE TRANSACTION (C.6): the operator's `approve` row (origin 'operator', the request id) and the request's
        # `approved` status commit together, after the stored document is re-hashed.
        from ....services import code_rules
        from ....services.detector_next import DetectorRefused

        try:
            row = code_rules.record_approval(approval)
        except DetectorRefused as refusal:
            raise refusal.to_http() from refusal
        db.add(row)
        approval.status = ApprovalStatus.APPROVED.value
        approval.resolved_at = datetime.now(timezone.utc)
        approval.decided_by = OPERATOR
        approval.batch_id = batch_id
        await db.commit()
        await db.refresh(approval)
        _emit_approval_event("resolved", {"request_id": approval.id, "status": "approved"})
        logger.info("Approval %s approved (%s, recorded%s)", request_id, approval.tool_name,
                    f", batch {batch_id}" if batch_id else "")
        return approval
    if kind is Execution.CALLER:
        approval.status = ApprovalStatus.APPROVED.value
        approval.resolved_at = datetime.now(timezone.utc)
        approval.decided_by = OPERATOR
        approval.batch_id = batch_id
        await db.commit()
        await db.refresh(approval)
        _emit_approval_event(
            "resolved", {"request_id": approval.id, "status": "approved"}
        )
        logger.info(f"Approval {request_id} approved ({approval.tool_name}, caller-executed)")
        return approval
    # 045 FR-13a (A16 D5): no approve-time self-call any more. Every approvable tool is RECORD or CALLER; a row of
    # any other kind cannot be approved.
    raise HTTPException(422, f"Unknown tool {approval.tool_name}")


#: ENH-006 D81 / D79: what each batch kind may contain. Never mixed: a batch is ONE kind, and any id of another
#: kind refuses the whole batch. miWorkbench's queue has no settings-change request (D79), so "settings" is refused.
BATCH_KINDS: dict[str, frozenset[str]] = {"code_rules": RECORD_TOOLS & frozenset({"code_rule_approve"})}

#: Most ids one batch may carry (the list route's maximum page).
BATCH_MAX = 500


class ApproveBatchRequest(BaseModel):
    kind: str
    approval_ids: list[str] = Field(min_length=1, max_length=BATCH_MAX)


class ApproveBatchItem(BaseModel):
    id: str
    status: str  # "approved" | "failed"
    error: Optional[str] = None
    approval: Optional[ApprovalResponse] = None


class ApproveBatchResponse(BaseModel):
    batch_id: str
    kind: str
    results: list[ApproveBatchItem]


def _batch_refusal(code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(422, {"code": code, "message": message, **extra})


def _failure_text(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("code") or detail)
    return str(detail)


@router.post("/approve-batch", response_model=ApproveBatchResponse,
             dependencies=[Depends(refuse_agent_decisions)])
async def approve_batch(body: ApproveBatchRequest, db: AsyncSession = Depends(get_db)) -> ApproveBatchResponse:
    """Approve every listed request of ONE kind with one operator call (ENH-006 D81).

    1. The kind must be one miWorkbench batches (`code_rules`); `settings` is refused — there are none here (D79).
    2. The WHOLE batch is checked before anything is approved: an id that is missing, not pending, or of another
       kind is named in a 409 and nothing changes.
    3. Each id is approved through `_approve_pending`, the single approve's own path, committed on its own and
       stamped with one shared `batch_id`. A request that fails (a document that no longer hashes to its code, a
       request decided meanwhile) is reported and the rest carry on.
    There is no batch reject.
    """
    if body.kind == "settings":
        raise _batch_refusal("approve_batch_no_settings_requests",
                             "miWorkbench's queue holds no settings-change requests: runtime settings are the "
                             "operator's, from Settings → Environment, and are never filed for approval.")
    tools = BATCH_KINDS.get(body.kind)
    if tools is None:
        raise _batch_refusal("approve_batch_unknown_kind", f"{body.kind!r} is not a batch kind (one of "
                             f"{sorted(BATCH_KINDS)})")
    ids = list(body.approval_ids)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise _batch_refusal("approve_batch_duplicate_ids", "an id is listed more than once", ids=dupes)

    await expire_stale(db)
    model = AgentApprovalRequest
    found = {
        r.id: r for r in (await db.execute(
            select(model).where(model.id.in_(ids)).execution_options(populate_existing=True))).scalars()
    }
    refused = []
    for rid in ids:
        row = found.get(rid)
        if row is None:
            refused.append({"id": rid, "why": "not found"})
        elif row.tool_name not in tools:
            refused.append({"id": rid, "why": f"is a {row.tool_name} request, not one of kind {body.kind}"})
        elif row.status != ApprovalStatus.PENDING.value:
            refused.append({"id": rid, "why": f"is {row.status}, not pending"})
    if refused:
        raise HTTPException(409, {
            "code": "approve_batch_refused",
            "message": f"{len(refused)} of {len(ids)} ids are not waiting {body.kind} requests; nothing was "
                       "approved",
            "refused": refused,
        })

    batch_id = str(uuid.uuid4())
    ttl_hours = effective().mcp_approval_ttl_hours
    results: list[ApproveBatchItem] = []
    for rid in ids:
        try:
            # Re-read: an earlier item's rollback expires the session's rows.
            approval = await _get_or_404(db, rid)
            if approval.status != ApprovalStatus.PENDING.value:
                raise HTTPException(409, f"Request is {approval.status}, not pending")
            done = await _approve_pending(db, approval, batch_id=batch_id)
            results.append(ApproveBatchItem(id=rid, status="approved", approval=respond(done, ttl_hours)))
        except HTTPException as exc:
            await db.rollback()
            results.append(ApproveBatchItem(id=rid, status="failed", error=_failure_text(exc)))
        except Exception as exc:  # noqa: BLE001 — one item's fault never stops the others
            await db.rollback()
            logger.exception("batch %s: approving %s failed", batch_id, rid)
            results.append(ApproveBatchItem(id=rid, status="failed", error=f"{type(exc).__name__}: {exc}"))
    logger.info("batch %s (%s): %d of %d approved", batch_id, body.kind,
                sum(r.status == "approved" for r in results), len(ids))
    return ApproveBatchResponse(batch_id=batch_id, kind=body.kind, results=results)


@router.post("/{request_id}/deny", response_model=ApprovalResponse,
             dependencies=[Depends(refuse_agent_decisions)])
async def deny_request(
    request_id: str,
    request: DenyRequest,
    db: AsyncSession = Depends(get_db),
):
    await expire_stale(db)
    approval = await _get_or_404(db, request_id)
    if approval.status != ApprovalStatus.PENDING.value:
        raise HTTPException(409, f"Request is {approval.status}, not pending")

    approval.status = ApprovalStatus.DENIED.value
    approval.reason = request.reason
    approval.resolved_at = datetime.now(timezone.utc)
    approval.decided_by = OPERATOR
    await db.commit()
    await db.refresh(approval)

    _emit_approval_event("resolved", {"request_id": approval.id, "status": "denied"})
    return respond(approval)


@router.post("/{request_id}/consume", response_model=ApprovalResponse)
async def consume_request(
    request_id: str,
    request: ConsumeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Spend an approved caller-executed request — once, and only for the payload approved.

    Called by the MCP tool that executes the action (`miserver_load_model`), never by the operator.
    Refuses (422) a backend-executed tool, a different tool and a different payload; refuses
    (409, `{status, reason}`) anything not currently `approved` — pending, denied, expired or
    already consumed. The once-guarantee is ONE conditional UPDATE: two concurrent consumes of
    one approval get one 200 and one 409.
    """
    await expire_stale(db)
    return respond(await consume(db, request_id, request.tool_name, request.payload))


async def consume(db: AsyncSession, request_id: str, tool_name: str, payload: dict[str, Any]) -> AgentApprovalRequest:
    """THE spend of an approved caller-executed request (045 FTDD §16.3), shared by `/consume` and the REST gate:
    once, and only for the action and payload approved. 422 for a non-caller tool, another tool or another payload;
    409 `{status, reason}` for anything not currently `approved` (pending, denied, expired, consumed, failed). Commits.
    """
    await expire_stale(db)
    approval = await _get_or_404(db, request_id)

    kind, _ = APPROVABLE_TOOLS.get(approval.tool_name, (None, None))
    if kind is not Execution.CALLER:
        raise HTTPException(
            422,
            f"{approval.tool_name} is executed by the backend when approved; there is nothing "
            "to consume",
        )
    if tool_name != approval.tool_name:
        raise HTTPException(
            422,
            f"approval {request_id} is for {approval.tool_name}, not {tool_name}",
        )
    if payload != approval.payload:
        logger.info("consume refused for %s: payload differs from the approved one", request_id)
        raise HTTPException(
            422,
            "the payload differs from the one the operator approved; an approval covers "
            "exactly what was approved",
        )

    model = AgentApprovalRequest
    spent = await db.execute(
        update(model)
        .where(
            model.id == request_id,
            model.status == ApprovalStatus.APPROVED.value,
            model.tool_name == tool_name,
        )
        .values(status=ApprovalStatus.CONSUMED.value, consumed_at=datetime.now(timezone.utc))
        .returning(model.id)
        .execution_options(synchronize_session=False)
    )
    if spent.scalar_one_or_none() is None:
        await db.rollback()
        await db.refresh(approval)
        logger.info("consume refused for %s: status %s", request_id, approval.status)
        raise HTTPException(409, {"status": approval.status, "reason": approval.reason})
    await db.commit()
    await db.refresh(approval)
    _emit_approval_event("resolved", {"request_id": approval.id, "status": "consumed"})
    return approval



async def mark_failed(request_id: str, reason: str) -> bool:
    """A consumed approval whose call then failed (045 FTDD §16.3 commit point 4; A29, A40): `failed`, in a FRESH
    session, never re-opened. Returns whether a row moved."""
    from src.core.database import AsyncSessionLocal

    model = AgentApprovalRequest
    async with AsyncSessionLocal() as own:
        moved = await own.execute(
            update(model)
            .where(model.id == request_id, model.status == ApprovalStatus.CONSUMED.value)
            .values(status=ApprovalStatus.FAILED.value, reason=reason[:2000])
            .returning(model.id)
            .execution_options(synchronize_session=False)
        )
        done = moved.scalar_one_or_none() is not None
        await own.commit()
    if done:
        _emit_approval_event("resolved", {"request_id": request_id, "status": "failed"})
    return done


async def _get_or_404(db: AsyncSession, request_id: str) -> AgentApprovalRequest:
    # `populate_existing`: `expire_stale` updates with `synchronize_session=False`, so a row this
    # session already holds would otherwise be returned with its pre-expiry status.
    result = await db.execute(
        select(AgentApprovalRequest)
        .where(AgentApprovalRequest.id == request_id)
        .execution_options(populate_existing=True)
    )
    approval = result.scalar_one_or_none()
    if not approval:
        raise HTTPException(404, f"Approval request {request_id} not found")
    return approval


def _summarize(tool_name: str, payload: dict[str, Any]) -> str:
    """One-line payload summary for the operator surface, per tool."""
    if tool_name == "start_probe_training_run":
        return (
            f"Train probes on model {payload.get('model_id')} from probe dataset "
            f"{payload.get('train_dataset_id')} — a GPU run (minutes to hours)"
        )
    if tool_name == "evaluate_probe_monitor":
        sets = payload.get("dataset_ids") or []
        return (
            f"Evaluate probe {payload.get('probe_id')} on {len(sets)} set(s) — a GPU job"
            # Feature 041 T20: the operator sees the sample it approves.
            + (f", a sample of {payload['sample_rows']} rows of each" if payload.get("sample_rows") else "")
        )
    if tool_name == "start_probe_judge_run":
        sets = payload.get("dataset_ids") or []
        return (
            f"Judge {len(sets)} set(s) with {payload.get('model')} on miServer, up to "
            f"{payload.get('max_rows_per_set')} rows each (one generation per row)"
            + (f", prompt {payload.get('prompt_id')}" if payload.get("prompt_id") else "")
        )
    if tool_name == "recalibrate_probe_monitor":
        return (
            f"Re-score probe {payload.get('probe_id')}'s calibration windows on the GPU and re-cut at "
            f"{payload.get('target_fpr')}" + (" (for its calibration positions)" if payload.get("require_positions") else "")
        )
    if tool_name == "start_probe_spot_check":
        return (
            f"Spot-check {payload.get('size')} turns of {payload.get('behaviour')} with "
            f"{payload.get('model')} on miServer (one judge call per turn and probe; live chats wait)"
        )
    if tool_name == "capture_wsp_heldout":
        sets = payload.get("dataset_ids") or []
        return (
            f"Capture probe {payload.get('probe_id')}'s held-out activations on "
            + (f"{len(sets)} set(s)" if sets else "every set it was evaluated on")
            + " — a GPU job that loads the probe's model on one card"
        )
    if tool_name == "import_wsp_candidate":
        run = payload.get("run_document") if isinstance(payload.get("run_document"), dict) else {}
        verdict = (run.get("verdict") or {}).get("value") if isinstance(run.get("verdict"), dict) else None
        bundle = ((run.get("request") or {}).get("bundle") or {}).get("bundle_id") \
            if isinstance(run.get("request"), dict) else None
        ack = payload.get("acknowledgement") if isinstance(payload.get("acknowledgement"), dict) else None
        return (
            f"Import miForge run {run.get('run_id')}'s candidate {run.get('candidate_id')} (verdict {verdict}) "
            f"from bundle {bundle} as a NEW probe that earns its rung at import; the parent is unchanged"
            + (f"; acknowledged {ack.get('verdict')}: {str(ack.get('reason'))[:80]}" if ack else "")
        )
    if tool_name == "code_rule_approve":
        functions = [f.get("name", "?") for f in (payload.get("document") or {}).get("functions") or []]
        return (f"Run operator code {str(payload.get('code_sha256'))[:12]}… ({', '.join(functions)}) in the isolated "
                f"runner — its own cases passed; read its source before approving")
    features = payload.get("selected_features") or payload.get("feature_ids") or []
    prompt = (payload.get("prompt") or "")[:60]
    return f"{len(features)} feature(s), prompt: {prompt!r}"
