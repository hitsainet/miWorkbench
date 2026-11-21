"""045 FR-13a (CHAIN_PLAN A16 D5; FTDD §16.5; FTASKS 21.6): ONE approval mechanism for steering.

Steering approvals are caller-executed: approving only marks the row; the RE-CALL with the id is consumed by the
steering route's own `require_consumed_approval` dependency, whoever sends it, before the task is queued.

* `test_approve_no_longer_self_calls` — approving a steering request issues no HTTP request and stores no task id.
* `test_steering_approval_is_consumed_by_the_route` — an agent's REST call files (202); after approval the re-call
  with the header is consumed by the route (committed BEFORE the handler runs) and the handler runs once.
* `test_recall_passes_the_id` — the MCP steer tools file the route's canonical payload and pass the id on re-call
  as the approval-request header, with the SAME body; a refused consume is a structured status, not a crash.
* `test_recall_consumes_through_the_real_route` — end to end on the test database through `httpx.ASGITransport`: the
  MCP tool files, the operator approves, the MCP re-call reaches the route, the route consumes it.

Controls (0xcc/reviews/045_A0_implementation_controls_2026-11-16.md):
  restore the steering rows as Execution.BACKEND           → test_approve_no_longer_self_calls red
  drop the id header from `_submit`'s re-call                → test_recall_passes_the_id red
  file the nested payload ({"tool_name", "payload": body}) → test_recall_consumes_through_the_real_route red
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import HTTPException

from src.core.approval_payload import APPROVAL_REQUEST_HEADER
from src.mcp_server.client import BackendError
from src.mcp_server.config import MCPSettings

ROUTE = "/api/v1/steering/async/compare"


def _features():
    from src.mcp_server.tools.steering import SteerFeature

    return [SteerFeature(feature_idx=3, layer=1, strength=5.0)]


TOOL_ARGS = {"sae_id": "sae_fx", "prompt": "hello", "features": _features()}


class _Recorder:
    """A miWorkbench client stand-in recording (method, path, body, headers)."""

    def __init__(self, fail=None):
        self.calls: list[tuple] = []
        self.fail = fail

    async def post(self, path, json_body=None, *, timeout=None, headers=None, **params):
        self.calls.append(("POST", path, json_body, headers))
        if self.fail is not None and path != "/mcp/approvals":
            raise self.fail
        if path == "/mcp/approvals":
            return {"id": "apr_1", "status": "pending"}
        return {"task_id": "t_1", "status": "queued"}


def _tools(client, *, approval=True):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tools import steering

    steering._inflight.clear()
    mcp = MCPServer("t")
    steering.register(mcp, client, MCPSettings(auth_token="x" * 32, steering_approval=approval))
    return mcp._tool_manager._tools  # noqa: SLF001


def test_recall_passes_the_id():
    from src.core.approval_payload import canonical_payload

    client = _Recorder()
    tools = _tools(client)
    first = asyncio.run(tools["steer_compare"].fn(**TOOL_ARGS))
    assert first["status"] == "pending_approval" and first["approval_request_id"] == "apr_1"
    assert "re-call steer_compare" in first["hint"]
    (_, path, filed, headers), = client.calls
    assert path == "/mcp/approvals" and headers is None
    assert filed["tool_name"] == "steer_compare"
    assert filed["payload"]["_method"] == "POST" and filed["payload"]["_route"] == ROUTE

    client.calls.clear()
    again = asyncio.run(tools["steer_compare"].fn(**TOOL_ARGS, approval_request_id="apr_1"))
    assert again["task_id"] == "t_1"
    (_, path, body, headers), = client.calls
    assert path == "/steering/async/compare"
    assert headers == {APPROVAL_REQUEST_HEADER: "apr_1"}
    # The route is sent exactly the body the approval was filed for (the gate rebuilds the same payload from it).
    assert canonical_payload("POST", ROUTE, {}, {}, body) == filed["payload"]


@pytest.mark.parametrize("status_code,detail,expected", [
    (409, {"status": "pending", "reason": None}, "pending"),
    (409, {"status": "denied", "reason": "no"}, "denied"),
    (404, "missing", "not_found"),
    (422, {"code": "mismatch", "action": "steer_sweep"}, "mismatch"),
])
def test_a_refused_recall_is_a_status(status_code, detail, expected):
    client = _Recorder(fail=BackendError(status_code, detail))
    tools = _tools(client)
    args = {"sae_id": "s", "prompt": "p", "feature_idx": 1, "layer": 0, "strength_values": [1.0]}
    result = asyncio.run(tools["steer_sweep"].fn(**args, approval_request_id="apr_9"))
    assert result["status"] == expected and result["approval_request_id"] == "apr_9"
    assert len(client.calls) == 1


def test_another_error_on_a_recall_still_raises():
    client = _Recorder(fail=BackendError(400, "SAE is not ready"))
    tools = _tools(client)
    with pytest.raises(BackendError):
        asyncio.run(tools["steer_compare"].fn(**TOOL_ARGS, approval_request_id="apr_1"))


@pytest.fixture
def handler_reached(monkeypatch):
    """The steering handler's first dispatch step: records the approval's status at that moment, then stops."""
    from src.api.v1 import steering_workers

    seen: list[dict] = []

    async def _stop(gpu_request):
        from sqlalchemy import select

        from src.core.database import AsyncSessionLocal
        from src.models.agent_approval import AgentApprovalRequest

        async with AsyncSessionLocal() as fresh:
            rows = (await fresh.execute(select(AgentApprovalRequest.id, AgentApprovalRequest.status))).all()
        seen.append({r.id: r.status for r in rows})
        raise HTTPException(503, "stopped by the test before any worker starts")

    monkeypatch.setattr(steering_workers, "steering_card_for", _stop)
    return seen


async def _status(request_id: str) -> str:
    from sqlalchemy import select

    from src.core.database import AsyncSessionLocal
    from src.models.agent_approval import AgentApprovalRequest

    async with AsyncSessionLocal() as fresh:
        return (await fresh.execute(select(AgentApprovalRequest.status).where(
            AgentApprovalRequest.id == request_id))).scalar_one()


def _asgi():
    from src.main import app

    return httpx.ASGITransport(app=app, raise_app_exceptions=False)


async def test_approve_no_longer_self_calls(async_session):
    async with httpx.AsyncClient(transport=_asgi(), base_url="http://t") as c:
        filed = await c.post("/api/v1/mcp/approvals", headers={"X-MiStudio-Agent": "mcp"}, json={
            "tool_name": "steer_compare", "payload": {"_method": "POST", "_route": ROUTE, "prompt": "x"}})
        assert filed.status_code == 201, filed.text
    from src.api.v1.endpoints.mcp_approvals import APPROVABLE_TOOLS, Execution

    assert APPROVABLE_TOOLS["steer_compare"][0] is Execution.CALLER
    original = httpx.AsyncClient
    try:
        httpx.AsyncClient = _refusing_client  # the approve below must build no client of its own
        async with original(transport=_asgi(), base_url="http://t") as c:
            approved = await c.post(f"/api/v1/mcp/approvals/{filed.json()['id']}/approve")
    finally:
        httpx.AsyncClient = original
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "approved" and approved.json()["steering_task_id"] is None


def _refusing_client(*a, **k):
    raise AssertionError("approve built an HTTP client (the steering self-call is gone since 045)")


async def test_steering_approval_is_consumed_by_the_route(async_session, handler_reached):
    body = {"sae_id": "sae_fx", "prompt": "hello", "selected_features": [{"feature_idx": 3, "layer": 1,
                                                                            "strength": 5.0}]}
    async with httpx.AsyncClient(transport=_asgi(), base_url="http://t") as c:
        filed = await c.post(ROUTE, json=body, headers={"X-MiStudio-Agent": "agent:miassist"})
        assert filed.status_code == 202, filed.text
        rid = filed.json()["approval_request_id"]
        assert handler_reached == []
        assert (await c.post(f"/api/v1/mcp/approvals/{rid}/approve")).status_code == 200
        ran = await c.post(ROUTE, json=body, headers={"X-MiStudio-Agent": "agent:miassist",
                                                      APPROVAL_REQUEST_HEADER: rid})
        assert ran.status_code == 503, ran.text  # the test's stop inside the handler
        again = await c.post(ROUTE, json=body, headers={"X-MiStudio-Agent": "agent:miassist",
                                                        APPROVAL_REQUEST_HEADER: rid})
    assert len(handler_reached) == 1, "the handler ran once"
    assert handler_reached[0][rid] == "consumed", "the consume was committed before the handler ran"
    assert await _status(rid) == "failed", "the handler's refusal marks the spent approval failed"
    assert again.status_code == 409 and again.json()["detail"]["status"] == "failed"


async def test_recall_consumes_through_the_real_route(async_session, handler_reached):
    """The MCP pod's steer tool, its real client and the real route, over ASGI on the test database."""
    from src.mcp_server.client import AGENT_HEADER, AGENT_NAME, MiWorkbenchClient

    client = MiWorkbenchClient("http://t")
    await client.close()
    client._http = httpx.AsyncClient(transport=_asgi(), base_url="http://t/api/v1",  # noqa: SLF001
                                     headers={AGENT_HEADER: AGENT_NAME})
    tools = _tools(client)
    try:
        first = await tools["steer_compare"].fn(**TOOL_ARGS)
        rid = first["approval_request_id"]
        assert await _status(rid) == "pending" and handler_reached == []
        async with httpx.AsyncClient(transport=_asgi(), base_url="http://t") as operator:
            assert (await operator.post(f"/api/v1/mcp/approvals/{rid}/approve")).status_code == 200
        with pytest.raises(BackendError) as stopped:
            await tools["steer_compare"].fn(**TOOL_ARGS, approval_request_id=rid)
        assert stopped.value.status_code == 503
        assert handler_reached and handler_reached[0][rid] == "consumed"
        assert await _status(rid) == "failed"
        # A second re-call is refused by the route: the approval is spent.
        refused = await tools["steer_compare"].fn(**TOOL_ARGS, approval_request_id=rid)
        assert refused["status"] == "failed" and len(handler_reached) == 1
    finally:
        await client.close()
