"""ENH-010 044 FTASKS 4.4: `POST /wsp/captures` (a GPU start) consumes a `capture_wsp_heldout` approval, through the
real app on the test database, with a real evaluated probe (the backstop's both halves for row 53).

Phase 2 refused every agent here (`agent_route_not_yet_gated`, R1A-W10); this replaces it, in the same commit as the
MCP tool (FTDD §5.6; `require_consumed_approval("capture_wsp_heldout", exempt_agents=frozenset(),
refuse_without_id=MCP_POD_AGENTS)`):

* the MCP pod without an approved id: 403 `approval_required`, nothing filed, nothing written, nothing dispatched;
* any other agent without an id: ONE pending approval with the route's canonical payload, 202, nothing dispatched;
* the operator (no agent header): unchanged — one dispatch with exactly the planned rows;
* the MCP tool end to end (its real client over ASGI): files, the operator approves, the re-call is consumed by the
  route and dispatches ONCE; a second re-call answers `consumed` and dispatches nothing;
* the GPU check: no card with the memory → 409 `gpu_memory_insufficient` before any row is written (a re-request
  then plans afresh), for the operator and for an approved re-call alike; the preflight route says so without writing.

Controls (record 0xcc/reviews/wsp_gated_capture_import_2026-11-19.md): drop the route's `approval_gated(...)`
dependency → the agent tests red; re-allow raw agent callers (`exempt_agents=MCP_POD_AGENTS`, no `refuse_without_id`)
→ `test_mcp_agent_without_an_approval_is_refused` red; drop `refuse_if_gpu_too_full` from `plan_capture_rows` → the
GPU tests red.
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from src.core.approval_payload import APPROVAL_REQUEST_HEADER
from tests.support.wsp_world import add_view, seed_probe, sync_db  # noqa: F401 (fixture)

ROUTE = "/api/v1/wsp/captures"


@pytest.fixture
def dispatched(monkeypatch):
    from src.services import gpu_dispatch

    calls = []

    def fake_gpu_delay(task, gpu_request):
        def send(**kwargs):
            calls.append({"task": task.name, "gpu_request": gpu_request, "kwargs": kwargs})
            return SimpleNamespace(id=f"task-{len(calls)}")
        return send

    monkeypatch.setattr(gpu_dispatch, "gpu_delay", fake_gpu_delay)
    return calls


@pytest.fixture
def gates_pass(monkeypatch):
    """The precision and render gates answer for a real model snapshot this world does not have (their wiring on the
    route is pinned by `test_wsp_capture_routes.py::test_the_precision_gate_is_on_the_route`)."""
    from src.api.v1.endpoints import probe_monitors

    async def passes(_probe_id):
        return None

    monkeypatch.setattr(probe_monitors, "refuse_precision_mismatch", passes)
    monkeypatch.setattr(probe_monitors, "refuse_render_mismatch", passes)


@pytest.fixture
def cards(monkeypatch):
    """The GPUs the backend sees (NVML), set per test; none by default (the check then reads `checked: false`)."""
    from src.services import gpu_placement

    state = {"cards": []}
    monkeypatch.setattr(gpu_placement, "list_cards", lambda: list(state["cards"]))
    return state


def _card(free_mb: int, index: int = 0):
    from src.services.gpu_placement import GpuCard

    return GpuCard(index=index, uuid=f"GPU-{index:08d}", name="test card", total_mb=24576, free_mb=free_mb)


@pytest.fixture
def world(sync_db, tmp_path):
    """A probe evaluated on one eval view, with its rows recorded; its model is 1B parameters at FP16 (~3.9 GB of
    weights plus 2 GB headroom: `resource_config.load_need_gb`)."""
    from src.models.model import Model, QuantizationFormat
    from src.models.probe_monitor import ProbeMonitorEvaluation

    w = seed_probe(sync_db, weights_path="/x.safetensors", threshold=0.5)
    model = sync_db.query(Model).filter(Model.id == w.model_id).one()
    model.params_count, model.quantization = 1_000_000_000, QuantizationFormat.FP16
    view = add_view(sync_db, w, role="eval")
    paths = {}
    for name, values, dtype in (("scores", [0.1, 0.9, 0.4], np.float32), ("labels", [0, 1, 0], np.int8),
                                ("positions", [3, 5, 8], np.int64)):
        paths[name] = tmp_path / f"{view}_{name}.npy"
        np.save(paths[name], np.asarray(values, dtype=dtype))
    sync_db.add(ProbeMonitorEvaluation(probe_id=w.probe_id, dataset_id=view, status="completed", metrics={},
                                       scores_path=str(paths["scores"]), labels_path=str(paths["labels"]),
                                       positions_path=str(paths["positions"])))
    sync_db.commit()
    return SimpleNamespace(probe_id=w.probe_id, view=view, body={"probe_id": w.probe_id, "dataset_ids": [view]})


def _asgi():
    from src.main import app

    return httpx.ASGITransport(app=app, raise_app_exceptions=False)


async def _post(body, headers=None):
    async with httpx.AsyncClient(transport=_asgi(), base_url="http://t") as c:
        return await c.post(ROUTE, json=body, headers=headers or {})


def _captures(sync_db, probe_id):
    from src.models.wsp import WspActivationCapture

    sync_db.expire_all()
    return sync_db.query(WspActivationCapture).filter(WspActivationCapture.probe_id == probe_id).all()


def _approvals(sync_db, tool="capture_wsp_heldout"):
    from src.models.agent_approval import AgentApprovalRequest

    sync_db.expire_all()
    return sync_db.query(AgentApprovalRequest).filter(AgentApprovalRequest.tool_name == tool).all()


# ── a raw agent call ─────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_agent_without_an_approval_is_refused(async_session, world, sync_db, gates_pass, dispatched, cards):
    response = await _post(world.body, {"X-MiStudio-Agent": "mcp"})
    assert response.status_code == 403, response.text
    assert response.json()["detail"] == {"code": "approval_required", "action": "capture_wsp_heldout"}
    assert dispatched == [] and _captures(sync_db, world.probe_id) == [] and _approvals(sync_db) == []


@pytest.mark.asyncio
async def test_another_agent_files_one_approval_and_queues_nothing(async_session, world, sync_db, gates_pass,
                                                                   dispatched, cards):
    response = await _post(world.body, {"X-MiStudio-Agent": "agent:miassist"})
    assert response.status_code == 202 and response.headers["X-MiStudio-Approval"] == "pending", response.text
    filed = _approvals(sync_db)
    assert len(filed) == 1 and filed[0].status == "pending" and filed[0].requested_by == "agent:miassist"
    assert filed[0].payload == {"_method": "POST", "_route": ROUTE, **world.body}
    assert dispatched == [] and _captures(sync_db, world.probe_id) == []


@pytest.mark.asyncio
async def test_an_agent_with_an_unapproved_id_runs_nothing(async_session, world, sync_db, gates_pass, dispatched,
                                                           cards):
    filed = await _post(world.body, {"X-MiStudio-Agent": "agent:miassist"})
    rid = filed.json()["approval_request_id"]
    response = await _post(world.body, {"X-MiStudio-Agent": "mcp", APPROVAL_REQUEST_HEADER: rid})
    assert response.status_code == 409 and response.json()["detail"]["status"] == "pending"
    assert dispatched == [] and _captures(sync_db, world.probe_id) == []


# ── the operator ─────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_operator_call_is_unchanged(async_session, world, sync_db, gates_pass, dispatched, cards):
    response = await _post(world.body)
    assert response.status_code == 202, response.text
    rows = _captures(sync_db, world.probe_id)
    assert len(dispatched) == 1 and len(rows) == 1
    assert dispatched[0]["kwargs"] == {"probe_id": world.probe_id, "capture_ids": [rows[0].id]}
    assert response.json()["started"] == [rows[0].id] and _approvals(sync_db) == []


# ── the MCP tool, end to end ─────────────────────────────────────────────────────────────────────────────────────


def _tool(client):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import wsp

    mcp = MCPServer("t")
    wsp.register(mcp, client, MCPSettings(auth_token="x" * 32))
    return mcp._tool_manager._tools["capture_wsp_heldout"]  # noqa: SLF001


async def _mcp_client():
    from src.mcp_server.client import AGENT_HEADER, AGENT_NAME, MiWorkbenchClient

    client = MiWorkbenchClient("http://t")
    await client.close()
    client._http = httpx.AsyncClient(transport=_asgi(), base_url="http://t/api/v1",  # noqa: SLF001
                                     headers={AGENT_HEADER: AGENT_NAME})
    return client


async def _approve(rid):
    async with httpx.AsyncClient(transport=_asgi(), base_url="http://t") as operator:
        response = await operator.post(f"/api/v1/mcp/approvals/{rid}/approve")
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_approved_mcp_capture_starts_once(async_session, world, sync_db, gates_pass, dispatched, cards):
    cards["cards"] = [_card(free_mb=20_000)]
    client = await _mcp_client()
    tool = _tool(client)
    try:
        first = await tool.fn(probe_id=world.probe_id, dataset_ids=[world.view])
        assert first["status"] == "pending_approval", first
        rid = first["approval_request_id"]
        (row,) = _approvals(sync_db)
        assert row.id == rid and row.status == "pending" and row.requested_by == "mcp"
        assert row.payload == {"_method": "POST", "_route": ROUTE, **world.body}
        assert row.forecast["gpu"]["fits"] is True and row.forecast["sets"][0]["rows"] == 3
        assert dispatched == [] and _captures(sync_db, world.probe_id) == []

        early = await tool.fn(probe_id=world.probe_id, dataset_ids=[world.view], approval_request_id=rid)
        assert early["status"] == "pending" and dispatched == []

        await _approve(rid)
        started = await tool.fn(probe_id=world.probe_id, dataset_ids=[world.view], approval_request_id=rid)
        assert started.get("task_id") == "task-1", started
        assert len(dispatched) == 1
        assert started["gpu"]["fits"] is True and started["gpu"]["card"]["index"] == 0
        assert _approvals(sync_db)[0].status == "consumed"

        again = await tool.fn(probe_id=world.probe_id, dataset_ids=[world.view], approval_request_id=rid)
        # The set is now pending, so the tool finds nothing to run and files nothing; a raw re-send of the same
        # request with the spent id is refused by the route (409 consumed).
        assert again["started"] == [] and len(dispatched) == 1
        response = await _post(world.body, {"X-MiStudio-Agent": "mcp", APPROVAL_REQUEST_HEADER: rid})
        assert response.status_code == 409 and response.json()["detail"]["status"] == "consumed"
        assert len(dispatched) == 1 and len(_approvals(sync_db)) == 1
    finally:
        await client.close()


# ── the GPU check ────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_full_card_is_refused_before_anything_is_written(async_session, world, sync_db, gates_pass,
                                                                 dispatched, cards):
    cards["cards"] = [_card(free_mb=900), _card(free_mb=1200, index=1)]
    response = await _post(world.body)
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "gpu_memory_insufficient" and 3500 < detail["required_mb"] < 4500  # 1B at FP16 (~1,907 MB) + 2,048 MB headroom
    assert len(detail["cards"]) == 2 and "Free memory on a card" in detail["message"]
    assert dispatched == [] and _captures(sync_db, world.probe_id) == []
    # Positive control: the same request once a card has room dispatches, and plans afresh (no stale pending row).
    cards["cards"] = [_card(free_mb=900), _card(free_mb=20_000, index=1)]
    response = await _post(world.body)
    assert response.status_code == 202, response.text
    assert len(dispatched) == 1 and response.json()["gpu"]["card"]["index"] == 1


@pytest.mark.asyncio
async def test_no_visible_gpu_is_not_a_refusal(async_session, world, sync_db, gates_pass, dispatched, cards):
    response = await _post(world.body)
    assert response.status_code == 202, response.text
    assert response.json()["gpu"]["checked"] is False and len(dispatched) == 1


@pytest.mark.asyncio
async def test_a_card_that_filled_after_approval_is_refused_without_spending(async_session, world, sync_db,
                                                                            gates_pass, dispatched, cards):
    cards["cards"] = [_card(free_mb=20_000)]
    client = await _mcp_client()
    tool = _tool(client)
    try:
        rid = (await tool.fn(probe_id=world.probe_id, dataset_ids=[world.view]))["approval_request_id"]
        await _approve(rid)
        cards["cards"] = [_card(free_mb=500)]
        refused = await tool.fn(probe_id=world.probe_id, dataset_ids=[world.view], approval_request_id=rid)
        assert refused["refused"] == "gpu_memory_insufficient"
        assert _approvals(sync_db)[0].status == "approved", "the approval is still unspent"
        assert dispatched == [] and _captures(sync_db, world.probe_id) == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_the_preflight_reads_and_writes_nothing(async_session, world, sync_db, gates_pass, dispatched, cards):
    cards["cards"] = [_card(free_mb=900)]
    async with httpx.AsyncClient(transport=_asgi(), base_url="http://t") as c:
        response = await c.get(f"{ROUTE}/preflight", params={"probe_id": world.probe_id,
                                                              "dataset_ids": [world.view]},
                               headers={"X-MiStudio-Agent": "mcp"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["to_run"] == 1 and body["rows_to_run"] == 3
    assert body["sets"] == [{"dataset_id": world.view, "role": "eval", "rows": 3, "capture": None, "status": None,
                             "would_run": True}]
    assert body["refusal"]["code"] == "gpu_memory_insufficient" and body["gpu"]["fits"] is False
    assert dispatched == [] and _captures(sync_db, world.probe_id) == [] and _approvals(sync_db) == []
