"""ENH-010 044 FTASKS 14.2: `POST /wsp/candidates/import` consumes an `import_wsp_candidate` approval, through the real
app on the test database (the backstop's both halves for row W1).

Phase 2 refused every agent here (`agent_route_not_yet_gated`, R1A-W10); this replaces it in the same commit as the
MCP tool (FTDD §5.6):

* the MCP pod without an approved id: 403 `approval_required`; nothing checked, nothing queued, nothing filed;
* any other agent without an id: ONE pending approval with the route's canonical payload (the documents and the
  patch included), 202; nothing checked, nothing queued;
* the operator (no agent header): unchanged — the checks run and ONE import task is queued with the body checked;
* the MCP tool end to end over ASGI: files; the operator approves; the re-call is consumed by the route, checked, and
  queues ONE task with exactly the approved body; a second re-call answers `consumed` and queues nothing;
* a refused import spends the approval (`failed`), never re-opened.

The FR-8 checks themselves run on miForge-shaped documents in `test_wsp_import.py`; here `wsp_import.check` is
replaced by a recorder, because what is under test is the GATE in front of it (whether the check and the queue are
reached, and with what).

Controls (record 0xcc/reviews/wsp_gated_capture_import_2026-11-19.md): drop the route's `approval_gated(...)`
dependency → the agent tests red; re-allow raw agent callers → `test_mcp_agent_without_an_approval_is_refused` red.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import httpx
import pytest

from src.core.approval_payload import APPROVAL_REQUEST_HEADER

ROUTE = "/api/v1/wsp/candidates/import"
RUN_DOC = {"run_id": "wsr_gate", "candidate_id": "wsc_gate", "verdict": {"value": "improved"},
           "request": {"bundle": {"bundle_id": "wsb_gate"}}}
BODY = {"run_document": RUN_DOC, "patch_manifest": {"patch_sha256": "b" * 64}, "edit_npz_b64": "QUJDRA=="}


@pytest.fixture
def import_world(monkeypatch):
    """Records every `wsp_import.check` the route makes and every import task it queues; `refuse` makes the check
    refuse with that code."""
    from src.services import wsp_import
    from src.workers import wsp_tasks

    state = SimpleNamespace(checked=[], queued=[], refuse=None)

    def check(db, *, run_document, patch_manifest, edit_npz, acknowledgement=None):
        state.checked.append({"run_document": run_document, "patch_manifest": patch_manifest, "edit_npz": edit_npz,
                              "acknowledgement": acknowledgement})
        if state.refuse:
            raise wsp_import.ImportRefused(state.refuse, "refused by the test")
        return SimpleNamespace(bundle=SimpleNamespace(id="wsb_gate"), parent=SimpleNamespace(id="pm_parent"),
                               run={"run_id": run_document["run_id"]}, call=SimpleNamespace(id="wsc_call"))

    monkeypatch.setattr(wsp_import, "check", check)
    # The route sends with `apply_async(kwargs=..., task_id=<the row's Celery id>)` (2026-11-27: its task-queue row).
    monkeypatch.setattr(wsp_tasks.import_wsp_candidate, "apply_async",
                        lambda kwargs=None, task_id=None, **_: state.queued.append(kwargs) or SimpleNamespace(id=task_id))
    return state


def _asgi():
    from src.main import app

    return httpx.ASGITransport(app=app, raise_app_exceptions=False)


async def _post(body, headers=None):
    async with httpx.AsyncClient(transport=_asgi(), base_url="http://t") as c:
        return await c.post(ROUTE, json=body, headers=headers or {})


async def _approvals():
    from sqlalchemy import select

    from src.core.database import AsyncSessionLocal
    from src.models.agent_approval import AgentApprovalRequest

    async with AsyncSessionLocal() as fresh:
        return (await fresh.execute(select(AgentApprovalRequest).where(
            AgentApprovalRequest.tool_name == "import_wsp_candidate"))).scalars().all()


async def _approve(rid):
    async with httpx.AsyncClient(transport=_asgi(), base_url="http://t") as operator:
        response = await operator.post(f"/api/v1/mcp/approvals/{rid}/approve")
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_mcp_agent_without_an_approval_is_refused(async_session, import_world):
    response = await _post(BODY, {"X-MiStudio-Agent": "mcp"})
    assert response.status_code == 403, response.text
    assert response.json()["detail"] == {"code": "approval_required", "action": "import_wsp_candidate"}
    assert import_world.checked == [] and import_world.queued == [] and await _approvals() == []


@pytest.mark.asyncio
async def test_another_agent_files_one_approval_and_queues_nothing(async_session, import_world):
    response = await _post(BODY, {"X-MiStudio-Agent": "agent:miassist"})
    assert response.status_code == 202 and response.headers["X-MiStudio-Approval"] == "pending", response.text
    (row,) = await _approvals()
    assert row.status == "pending" and row.payload == {"_method": "POST", "_route": ROUTE, **BODY}
    assert import_world.checked == [] and import_world.queued == []


@pytest.mark.asyncio
async def test_the_operator_call_is_unchanged(async_session, import_world):
    response = await _post(BODY)
    assert response.status_code == 202, response.text
    # The task id is the Celery id of the import's task-queue row (2026-11-27; `test_wsp_import_task_status.py`).
    assert response.json()["status"] == "importing" and uuid.UUID(response.json()["task_id"])
    assert response.json()["task_queue_id"].startswith("tq_")
    assert len(import_world.checked) == 1 and import_world.checked[0]["edit_npz"] == b"ABCD"
    assert import_world.queued == [{**BODY, "acknowledgement": None}]
    assert await _approvals() == []


async def _mcp_tool():
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.client import AGENT_HEADER, AGENT_NAME, MiWorkbenchClient
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import wsp

    client = MiWorkbenchClient("http://t")
    await client.close()
    client._http = httpx.AsyncClient(transport=_asgi(), base_url="http://t/api/v1",  # noqa: SLF001
                                     headers={AGENT_HEADER: AGENT_NAME})
    mcp = MCPServer("t")
    wsp.register(mcp, client, MCPSettings(auth_token="x" * 32))
    return client, mcp._tool_manager._tools["import_wsp_candidate"]  # noqa: SLF001


@pytest.mark.asyncio
async def test_approved_mcp_import_queues_once(async_session, import_world):
    client, tool = await _mcp_tool()
    try:
        first = await tool.fn(**BODY)
        assert first["status"] == "pending_approval", first
        rid = first["approval_request_id"]
        (row,) = await _approvals()
        assert row.payload == {"_method": "POST", "_route": ROUTE, **BODY} and row.requested_by == "mcp"
        assert import_world.checked == [] and import_world.queued == []

        await _approve(rid)
        started = await tool.fn(**BODY, approval_request_id=rid)
        assert started["status"] == "importing" and uuid.UUID(started["task_id"]), started
        assert len(import_world.checked) == 1
        assert import_world.queued == [{**BODY, "acknowledgement": None}]
        assert (await _approvals())[0].status == "consumed"

        again = await tool.fn(**BODY, approval_request_id=rid)
        assert again["status"] == "consumed" and len(import_world.queued) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_refused_import_spends_the_approval(async_session, import_world):
    client, tool = await _mcp_tool()
    import_world.refuse = "patch_base_mismatch"
    try:
        rid = (await tool.fn(**BODY))["approval_request_id"]
        await _approve(rid)
        with pytest.raises(Exception) as raised:
            await tool.fn(**BODY, approval_request_id=rid)
        assert "patch_base_mismatch" in str(raised.value)
        assert import_world.queued == []
        assert (await _approvals())[0].status == "failed"
        again = await _post(BODY, {"X-MiStudio-Agent": "mcp", APPROVAL_REQUEST_HEADER: rid})
        assert again.status_code == 409 and again.json()["detail"]["status"] == "failed"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_approval_for_another_body_is_a_mismatch(async_session, import_world):
    filed = await _post(BODY, {"X-MiStudio-Agent": "agent:miassist"})
    rid = filed.json()["approval_request_id"]
    await _approve(rid)
    other = {**BODY, "edit_npz_b64": "RUZHSA=="}
    response = await _post(other, {"X-MiStudio-Agent": "mcp", APPROVAL_REQUEST_HEADER: rid})
    assert response.status_code == 422 and response.json()["detail"]["code"] == "mismatch"
    assert import_world.checked == [] and import_world.queued == []
    assert (await _approvals())[0].status == "approved", "a mismatched call spends nothing"
