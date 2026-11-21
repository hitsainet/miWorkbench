"""045 FR-13 (FTASKS 21.0a, 21.1): the REST gate's dependency, its four commit points, and its refusals, on a fixture
router that carries `require_consumed_approval` exactly as the real routes do (no handler body changes, A40).

Written FIRST (21.1, R1A-16): these also prove that, on this FastAPI (0.159.1), an exception raised in the endpoint
reaches a `yield` dependency's `except` (FTDD §16.3 marked it unverified) — `test_handler_refusal_marks_failed` and
`test_failure_after_consume_marks_failed` cannot pass otherwise.

Every read of the queue goes through a FRESH session (`AsyncSessionLocal`), never the request's, so a row the request's
rollback would remove cannot be seen here.

Controls (record `0xcc/reviews/045_implementation_controls_2026-11-16.md`):
  file in a session the request rolls back            → test_the_filed_row_survives_the_request red
  consume before parsing the body                     → test_invalid_body_does_not_spend_the_approval red
  drop the `yield` exit's `mark_failed`               → test_handler_refusal_marks_failed, ..._after_consume_ red
  read the id from the canonical payload              → test_a_body_form_id_is_consumed_once red
  check the arm before reading the id                 → test_an_id_outside_the_arm_is_refused red
  drop `_method`/`_route` from the payload            → test_an_approval_is_consumed_only_on_its_route red
  file before refusing undeclared keys                → test_a_filing_with_an_ignored_key_is_refused red
  remove the `mcp` exemption                          → test_mcp_is_exempt red
  gate regardless of the header                       → test_no_header_behaves_as_today red
"""
from __future__ import annotations

from typing import Literal, Optional

import httpx
import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

ACTION = "fixture_gpu_action"
ARMED = "fixture_armed_action"


class Inner(BaseModel):
    depth: int = 1


class FixtureBody(BaseModel):
    rows: int
    inner: Optional[Inner] = None
    fail: Optional[str] = None


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: int


class ClashBody(BaseModel):
    """A body that DECLARES a field named like the route's path parameter (no real gated route does)."""

    thing_id: str
    rows: int


class ControlBody(BaseModel):
    """Row 35's body as `schemas.training.TrainingControlRequest` declares it (a Literal of four actions)."""

    action: Literal["pause", "resume", "stop", "stop_and_finalize"]


@pytest.fixture
def world(monkeypatch, async_session):
    """A fixture app: the real 202 handler, three gated routes (two share one action, one has an arm), and a list
    standing in for the GPU queue."""
    from src.api.v1.endpoints import mcp_approvals
    from src.core.approval_gate import PendingApproval, pending_approval_handler, require_consumed_approval

    monkeypatch.setitem(mcp_approvals.APPROVABLE_TOOLS, ACTION, (mcp_approvals.Execution.CALLER, None))
    monkeypatch.setitem(mcp_approvals.APPROVABLE_TOOLS, ARMED, (mcp_approvals.Execution.CALLER, None))
    dispatched: list[dict] = []
    router = APIRouter()

    @router.post("/things/{thing_id}/run", dependencies=[require_consumed_approval(ACTION)])
    async def run(thing_id: str, body: FixtureBody, sample: int = Query(10)):
        if body.fail == "refuse":
            raise HTTPException(409, "the handler refused")
        dispatched.append({"thing_id": thing_id, "rows": body.rows, "sample": sample})
        if body.fail == "late":
            raise RuntimeError("failed after the dispatch")
        return {"task_id": f"t{len(dispatched)}"}

    @router.post("/things/{thing_id}/other", dependencies=[require_consumed_approval(ACTION)])
    async def other(thing_id: str, body: FixtureBody):
        dispatched.append({"other": thing_id})
        return {"ok": True}

    from src.core.approval_gate import RECUT_WINDOWS_ARM

    # The REAL arm row 25 uses, so a change to it is seen here (NC: compare the raw string → spellings red).
    @router.post("/things/{thing_id}/recut", dependencies=[require_consumed_approval(ARMED, arm=RECUT_WINDOWS_ARM)])
    async def recut(thing_id: str, body: StrictBody, recut_windows: bool = Query(False)):
        dispatched.append({"recut": recut_windows})
        return {"ok": True}


    from src.core.approval_gate import RESUME_ARM

    # The REAL arm row 35 uses (training control: only `resume` starts GPU work).
    @router.post("/things/{thing_id}/control", dependencies=[require_consumed_approval(ARMED, arm=RESUME_ARM)])
    async def control(thing_id: str, body: ControlBody):
        dispatched.append({"control": body.action})
        return {"ok": True}

    def raising_arm(parsed):
        raise RuntimeError("this arm cannot decide")

    # OR1-045a0-9 (G7): an arm that raises GATES (fail closed).
    @router.post("/things/{thing_id}/undecided", dependencies=[require_consumed_approval(ARMED, arm=raising_arm)])
    async def undecided(thing_id: str, body: StrictBody):
        dispatched.append({"undecided": thing_id})
        return {"ok": True}

    @router.post("/things/{thing_id}/bodiless", dependencies=[require_consumed_approval(ACTION)])
    async def bodiless(thing_id: str):
        dispatched.append({"bodiless": thing_id})
        return {"ok": True}

    @router.post("/things/{thing_id}/clash", dependencies=[require_consumed_approval(ACTION)])
    async def clash(thing_id: str, body: ClashBody):
        dispatched.append({"clash": thing_id, "body": body.thing_id})
        return {"ok": True}

    @router.post("/things/{thing_id}/upload", dependencies=[require_consumed_approval(ACTION)])
    async def upload(thing_id: str, request: Request):
        dispatched.append({"upload": (await request.body()).decode()})
        return {"ok": True}

    app = FastAPI()
    app.add_exception_handler(PendingApproval, pending_approval_handler)
    app.include_router(router, prefix="/api/v1")
    return app, dispatched


async def _post(app, path, json=None, agent: Optional[str] = "miassist", approval_id: Optional[str] = None,
                params=None):
    headers = {}
    if agent is not None:
        headers["X-MiStudio-Agent"] = agent
    if approval_id is not None:
        headers["X-MiStudio-Approval-Request"] = approval_id
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.post(path, json=json, headers=headers, params=params)


async def _row(approval_id: str):
    from src.core.database import AsyncSessionLocal
    from src.models.agent_approval import AgentApprovalRequest

    async with AsyncSessionLocal() as fresh:
        return await fresh.get(AgentApprovalRequest, approval_id)


async def _set_status(approval_id: str, status: str):
    from datetime import datetime, timezone

    from sqlalchemy import update

    from src.core.database import AsyncSessionLocal
    from src.models.agent_approval import AgentApprovalRequest

    async with AsyncSessionLocal() as fresh:
        await fresh.execute(update(AgentApprovalRequest).where(AgentApprovalRequest.id == approval_id)
                            .values(status=status, resolved_at=datetime.now(timezone.utc)))
        await fresh.commit()


async def _file_and_approve(app, path, body, params=None) -> str:
    response = await _post(app, path, json=body, params=params)
    assert response.status_code == 202, response.text
    approval_id = response.json()["approval_request_id"]
    await _set_status(approval_id, "approved")
    return approval_id


async def test_the_filed_row_survives_the_request(world):
    app, dispatched = world
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": 5}, params={"sample": 3})
    assert response.status_code == 202, response.text
    assert response.headers["X-MiStudio-Approval"] == "pending"
    body = response.json()
    assert body["status"] == "pending_approval" and body["action"] == ACTION
    assert body["approval"]["then"] == "call_again_with" and body["approval"]["decide_in"] == "miworkbench"
    row = await _row(body["approval_request_id"])
    assert row is not None and row.status == "pending" and row.tool_name == ACTION
    assert row.requested_by == "miassist"
    assert row.payload == {"_method": "POST", "_route": "/api/v1/things/{thing_id}/run", "thing_id": "a1",
                           "sample": 3, "rows": 5}
    assert dispatched == [], "an agent's first call queues nothing"


async def test_approved_recall_runs_once(world):
    app, dispatched = world
    approval_id = await _file_and_approve(app, "/api/v1/things/a1/run", {"rows": 5})
    first = await _post(app, "/api/v1/things/a1/run", json={"rows": 5}, approval_id=approval_id)
    assert first.status_code == 200, first.text
    assert (await _row(approval_id)).status == "consumed"
    second = await _post(app, "/api/v1/things/a1/run", json={"rows": 5}, approval_id=approval_id)
    assert second.status_code == 409 and second.json()["detail"]["status"] == "consumed"
    assert len(dispatched) == 1


async def test_mismatched_payload_is_refused(world):
    app, dispatched = world
    approval_id = await _file_and_approve(app, "/api/v1/things/a1/run", {"rows": 5})
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": 6}, approval_id=approval_id)
    assert response.status_code == 422 and response.json()["detail"]["code"] == "mismatch"
    assert (await _row(approval_id)).status == "approved" and dispatched == []


async def test_an_approval_is_consumed_only_on_its_route(world):
    """A38: two routes, one action, the same body — the approval binds `_route`."""
    app, dispatched = world
    approval_id = await _file_and_approve(app, "/api/v1/things/a1/run", {"rows": 5})
    response = await _post(app, "/api/v1/things/a1/other", json={"rows": 5}, approval_id=approval_id)
    assert response.status_code == 422 and response.json()["detail"]["code"] == "mismatch"
    assert dispatched == []


async def test_invalid_body_does_not_spend_the_approval(world):
    app, dispatched = world
    approval_id = await _file_and_approve(app, "/api/v1/things/a1/run", {"rows": 5})
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": "many"}, approval_id=approval_id)
    assert response.status_code == 422
    assert (await _row(approval_id)).status == "approved", "an invalid body never spends an approval"
    assert dispatched == []


async def test_handler_refusal_marks_failed(world):
    app, dispatched = world
    approval_id = await _file_and_approve(app, "/api/v1/things/a1/run", {"rows": 5, "fail": "refuse"})
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": 5, "fail": "refuse"}, approval_id=approval_id)
    assert response.status_code == 409
    row = await _row(approval_id)
    assert row.status == "failed" and "the handler refused" in (row.reason or "")


async def test_failure_after_consume_marks_failed(world):
    app, dispatched = world
    approval_id = await _file_and_approve(app, "/api/v1/things/a1/run", {"rows": 5, "fail": "late"})
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": 5, "fail": "late"}, approval_id=approval_id)
    assert response.status_code == 500
    assert (await _row(approval_id)).status == "failed"
    assert len(dispatched) == 1


async def test_a_late_failure_cannot_reuse_the_id(world):
    app, dispatched = world
    approval_id = await _file_and_approve(app, "/api/v1/things/a1/run", {"rows": 5, "fail": "late"})
    await _post(app, "/api/v1/things/a1/run", json={"rows": 5, "fail": "late"}, approval_id=approval_id)
    again = await _post(app, "/api/v1/things/a1/run", json={"rows": 5, "fail": "late"}, approval_id=approval_id)
    assert again.status_code == 409 and again.json()["detail"]["status"] in ("failed", "consumed")
    assert len(dispatched) == 1, "one dispatch in all"


async def test_consume_is_committed_before_dispatch(world):
    """A40: the handler runs on a spent approval — its status read from a FRESH session while the handler runs."""
    from src.core.approval_gate import require_consumed_approval

    app, dispatched = world
    seen: list[str] = []
    holder: dict[str, str] = {}
    router = APIRouter()

    @router.post("/things/{thing_id}/peek", dependencies=[require_consumed_approval(ACTION)])
    async def peek(thing_id: str, body: FixtureBody):
        seen.append((await _row(holder["id"])).status)
        return {"ok": True}

    app.include_router(router, prefix="/api/v1")
    holder["id"] = await _file_and_approve(app, "/api/v1/things/a1/peek", {"rows": 1})
    done = await _post(app, "/api/v1/things/a1/peek", json={"rows": 1}, approval_id=holder["id"])
    assert done.status_code == 200, done.text
    assert seen == ["consumed"]


async def test_a_body_form_id_is_consumed_once(world):
    """A37, R2A-4: an id in the raw body (044's form) is read before the canonical payload strips it."""
    app, dispatched = world
    approval_id = await _file_and_approve(app, "/api/v1/things/a1/run", {"rows": 5})
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": 5, "approval_request_id": approval_id})
    assert response.status_code == 200, response.text
    again = await _post(app, "/api/v1/things/a1/run", json={"rows": 5, "approval_request_id": approval_id})
    assert again.status_code == 409 and len(dispatched) == 1


async def test_an_id_outside_the_arm_is_refused(world):
    app, dispatched = world
    response = await _post(app, "/api/v1/things/a1/recut", json={"rows": 1}, approval_id="whatever",
                           params={"recut_windows": "false"})
    assert response.status_code == 422 and response.json()["detail"]["code"] == "approval_not_applicable"
    assert dispatched == []


@pytest.mark.parametrize("spelling", ["true", "1", "yes", "on", "True"])
async def test_recut_windows_spellings_are_armed(world, spelling):
    """A41: the arm reads the PARSED bool, so every spelling FastAPI reads as true is gated."""
    app, dispatched = world
    response = await _post(app, "/api/v1/things/a1/recut", json={"rows": 1}, params={"recut_windows": spelling})
    assert response.status_code == 202, response.text
    assert dispatched == []


async def test_resume_is_armed(world):
    """A41 (row 35; FTASKS 21.0a; R1 review fix OR1-045a0-4, 2026-11-16): training control is gated only on
    `resume` (and on anything the arm cannot read as another action); pause and both stops run as before.

    Control: `RESUME_ARM` returns False → red."""
    app, dispatched = world
    resumed = await _post(app, "/api/v1/things/a1/control", json={"action": "resume"})
    assert resumed.status_code == 202, resumed.text
    assert resumed.headers.get("X-MiStudio-Approval") == "pending" and dispatched == []
    for action in ("pause", "stop", "stop_and_finalize"):
        ran = await _post(app, "/api/v1/things/a1/control", json={"action": action})
        assert ran.status_code == 200, (action, ran.text)
    assert dispatched == [{"control": "pause"}, {"control": "stop"}, {"control": "stop_and_finalize"}]


async def test_an_arm_that_raises_gates(world):
    """R1 review fix (OR1-045a0-9, G7), 2026-11-16: an arm that raises cannot decide, so the call is GATED (202,
    nothing dispatched); it is never let through.

    Control: `armed = False` in the arm's `except` → red."""
    app, dispatched = world
    response = await _post(app, "/api/v1/things/a1/undecided", json={"rows": 1})
    assert response.status_code == 202, response.text
    assert response.headers.get("X-MiStudio-Approval") == "pending"
    assert dispatched == []


async def test_an_approval_is_consumed_only_for_its_action(world):
    """R1 review fix (OR1-045a0-9, G10b), 2026-11-16: an approval whose payload equals the call's but whose ACTION is
    another is never spent (the explicit check and the conditional UPDATE's predicate both bind the action).

    Control: drop both the explicit `tool_name` check and the UPDATE's `tool_name` predicate in `consume` → red."""
    from sqlalchemy import update

    from src.core.database import AsyncSessionLocal
    from src.models.agent_approval import AgentApprovalRequest

    app, dispatched = world
    approval_id = await _file_and_approve(app, "/api/v1/things/a1/run", {"rows": 5})
    async with AsyncSessionLocal() as fresh:  # the same payload, approved for ANOTHER caller-executed action
        await fresh.execute(update(AgentApprovalRequest).where(AgentApprovalRequest.id == approval_id)
                            .values(tool_name=ARMED))
        await fresh.commit()
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": 5}, approval_id=approval_id)
    assert response.status_code in (409, 422), response.text
    assert (await _row(approval_id)).status == "approved", "an approval for another action is never spent"
    assert dispatched == []


async def test_a_body_sent_to_a_bodiless_route_is_refused(world):
    """R1 review fix (OR1-045a0-1), 2026-11-16: a route that declares no body ignores one, so an agent's body keys
    would be keys the card shows and the route never reads: `422 unknown_body_field`, nothing filed. No body, or `{}`,
    is filed as before. Control: drop the bodiless branch in `refuse_undeclared` → red."""
    app, dispatched = world
    refused = await _post(app, "/api/v1/things/a1/bodiless", json={"rows": 5})
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"] == {"code": "unknown_body_field", "field": "rows"}
    for body in (None, {}):
        filed = await _post(app, "/api/v1/things/a1/bodiless", json=body)
        assert filed.status_code == 202, (body, filed.text)
    assert dispatched == []


async def test_a_colliding_body_key_is_refused_before_filing(world):
    """R1 review fix (OR1-045a0-10), 2026-11-16: a DECLARED body field named like a path parameter passes
    `refuse_undeclared`, so the payload itself must refuse it: `422 colliding_field`, nothing filed or dispatched
    (before the fix the body's value silently replaced the path's in what the operator approves).
    Control: `out.update(body)` without the collision check → red."""
    from sqlalchemy import func, select

    from src.core.database import AsyncSessionLocal
    from src.models.agent_approval import AgentApprovalRequest

    app, dispatched = world
    async with AsyncSessionLocal() as fresh:
        before = (await fresh.execute(select(func.count()).select_from(AgentApprovalRequest))).scalar_one()
    response = await _post(app, "/api/v1/things/a1/clash", json={"thing_id": "b2", "rows": 1})
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == {"code": "colliding_field", "field": "thing_id"}
    async with AsyncSessionLocal() as fresh:
        after = (await fresh.execute(select(func.count()).select_from(AgentApprovalRequest))).scalar_one()
    assert after == before and dispatched == []


async def test_an_unarmed_call_runs_for_an_agent(world):
    app, dispatched = world
    response = await _post(app, "/api/v1/things/a1/recut", json={"rows": 1})
    assert response.status_code == 200 and dispatched == [{"recut": False}]


async def test_a_filing_with_an_ignored_key_is_refused(world):
    from sqlalchemy import func, select

    from src.core.database import AsyncSessionLocal
    from src.models.agent_approval import AgentApprovalRequest

    app, dispatched = world
    async with AsyncSessionLocal() as fresh:
        before = (await fresh.execute(select(func.count()).select_from(AgentApprovalRequest))).scalar_one()
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": 5, "gpu": "GPU-x"})
    assert response.status_code == 422
    assert response.json()["detail"] == {"code": "unknown_body_field", "field": "gpu"}
    query = await _post(app, "/api/v1/things/a1/run", json={"rows": 5}, params={"verbose": "1"})
    assert query.status_code == 422 and query.json()["detail"]["code"] == "unknown_query_field"
    async with AsyncSessionLocal() as fresh:
        after = (await fresh.execute(select(func.count()).select_from(AgentApprovalRequest))).scalar_one()
    assert after == before, "a refused filing creates no row"


async def test_a_nested_unknown_key_is_refused(world):
    app, dispatched = world
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": 5, "inner": {"depth": 2, "extra": 1}})
    assert response.status_code == 422
    assert response.json()["detail"] == {"code": "unknown_body_field", "field": "inner.extra"}


async def test_no_header_behaves_as_today(world):
    app, dispatched = world
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": 5, "anything": 1}, agent=None)
    assert response.status_code == 200 and len(dispatched) == 1


async def test_a_caller_without_the_header_is_not_parsed(world):
    """FR-15: a header-less call is untouched by the gate — no body read, no parse — so a route that reads its own
    non-JSON body (a form upload) works exactly as before. NC: let header-less calls through to the parse → 422."""
    app, dispatched = world
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        response = await c.post("/api/v1/things/a1/upload", content=b"not json at all")
    assert response.status_code == 200, response.text
    assert dispatched == [{"upload": "not json at all"}]


async def test_mcp_is_exempt(world):
    app, dispatched = world
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": 5}, agent="mcp")
    assert response.status_code == 200 and len(dispatched) == 1


@pytest.mark.parametrize("agent", ["miassist", "agent:miassist", "anything-else", ""])
async def test_which_agents_are_gated(world, agent):
    """A15, FR-16: every agent value but the MCP pod's own is gated; the exemption is the pod client's constant."""
    from src.core.approval_gate import MCP_POD_AGENTS
    from src.mcp_server.client import AGENT_NAME

    assert MCP_POD_AGENTS == frozenset({AGENT_NAME}) == frozenset({"mcp"})
    app, dispatched = world
    response = await _post(app, "/api/v1/things/a1/run", json={"rows": 5}, agent=agent)
    assert response.status_code == 202 and dispatched == []
