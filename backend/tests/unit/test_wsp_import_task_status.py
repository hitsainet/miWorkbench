"""An accepted WSP candidate import is pollable by `get_task_status` (MCP gap found by the ENH-010 phase-2 round,
2026-11-27).

Found live: `import_wsp_candidate` answered `{task_id, status: "importing"}` and said to poll `get_task_status`, which
answered `not_found` for that id: the import had no task-queue row, and its answer lived only in the Celery result
backend (`result_expires` one hour), so the round agent read the worker log. Now the route opens a
`wsp_candidate_import` task-queue row under the Celery id it sends the task with, BEFORE the send (the ENH-006 G17
order, `endpoints/_task_rows.py`), and the task moves it: `running`, then `completed` naming the NEW probe (its rung in
`entity_info`), or `failed` with `refused <code>: <message>`.

Through the real MCP tools on the built server (their client on the real app over ASGI), the real routes and the real
task on the test database: the agent files, the operator approves, the re-call is accepted and names its poll
(`next_steps`); that poll is CALLED and reads the import running, then finished; a second import reads failed.
`wsp_import.check` and `wsp_import.import_candidate` are stand-ins (their FR-8 behaviour is `test_wsp_import.py`'s, on
a real run); what is under test is the row around them.

Controls (record 0xcc/reviews/wsp_baselines_supplement_2026-11-27.md): drop the route's `open_task_row` → red; drop the
task's `complete_as` → red; drop the refusal's row write → red; drop the answer's poll step → red.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest

from tests.support.wsp_world import seed_probe, sync_db  # noqa: F401 (fixture)

ROUTE = "/api/v1/wsp/candidates/import"
RUN_DOC = {"run_id": "wsr_poll", "candidate_id": "wsc_poll", "verdict": {"value": "improved"},
           "request": {"bundle": {"bundle_id": "wsb_poll"}}}
BODY = {"run_document": RUN_DOC, "patch_manifest": {"patch_sha256": "c" * 64}, "edit_npz_b64": "QUJDRA=="}


def _asgi():
    from src.main import app

    return httpx.ASGITransport(app=app, raise_app_exceptions=False)


def _server(categories=None):
    """The REAL built server (default categories, or `categories`), its client on the real app over ASGI, as the MCP
    pod."""
    from src.mcp_server.client import AGENT_HEADER, AGENT_NAME
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    extra = {} if categories is None else {"tool_categories": categories}
    mcp, client = build_server(MCPSettings(allow_anonymous=True, **extra), stdio=True)
    client._http = httpx.AsyncClient(transport=_asgi(), base_url="http://t/api/v1",  # noqa: SLF001
                                     headers={AGENT_HEADER: AGENT_NAME})
    return mcp


async def _call(mcp, tool, args):
    return await mcp._tool_manager.call_tool(tool, args, context=None)  # noqa: SLF001


async def _approve_latest():
    from sqlalchemy import select

    from src.core.database import AsyncSessionLocal
    from src.models.agent_approval import AgentApprovalRequest

    async with AsyncSessionLocal() as fresh:
        rows = (await fresh.execute(select(AgentApprovalRequest).where(
            AgentApprovalRequest.tool_name == "import_wsp_candidate",
            AgentApprovalRequest.status == "pending"))).scalars().all()
    assert len(rows) == 1, [r.id for r in rows]
    async with httpx.AsyncClient(transport=_asgi(), base_url="http://t") as operator:
        response = await operator.post(f"/api/v1/mcp/approvals/{rows[0].id}/approve")
    assert response.status_code == 200, response.text
    return rows[0].id


@pytest.fixture
def world(sync_db, monkeypatch):
    """A parent probe and the probe the import will produce (rung 2, derived from it), on the test database; the
    check and the import stand in; the send is recorded, never published; the task's session is a stand-in (the
    stand-in import never reads it)."""
    from contextlib import contextmanager

    from src.models.probe_monitor import ProbeMonitor
    from src.services import wsp_import
    from src.workers import wsp_tasks

    parent = seed_probe(sync_db)
    derived = ProbeMonitor(run_id=parent.run_id, layer=1, rule="mean", rung=2,
                           derived_from={"parent_probe_id": parent.probe_id})
    sync_db.add(derived)
    sync_db.commit()
    state = SimpleNamespace(parent=parent.probe_id, derived=derived.id, sent=[], refuse=None,
                            entered=threading.Event(), release=threading.Event())

    def check(db, **_):
        return SimpleNamespace(bundle=SimpleNamespace(id="wsb_poll"), parent=SimpleNamespace(id=state.parent),
                               run={"run_id": RUN_DOC["run_id"]}, call=SimpleNamespace(id="wsc_000000000001"))

    def import_candidate(db, **_):
        state.entered.set()
        assert state.release.wait(30), "the test never released the import"
        if state.refuse:
            raise wsp_import.ImportRefused(state.refuse, "the arrays do not hash to the patch's edited_sha256")
        return {"probe_id": state.derived, "parent_probe_id": state.parent, "rung": 2}

    @contextmanager
    def no_session():
        yield None

    monkeypatch.setattr(wsp_import, "check", check)
    monkeypatch.setattr(wsp_import, "import_candidate", import_candidate)
    monkeypatch.setattr(wsp_tasks.import_wsp_candidate, "apply_async",
                        lambda kwargs=None, task_id=None, **_: state.sent.append((kwargs, task_id))
                        or SimpleNamespace(id=task_id))
    monkeypatch.setattr(wsp_tasks.import_wsp_candidate, "get_db", no_session, raising=False)
    return state


def _run_task(kwargs, task_id):
    """The real task body under its Celery id, as the `wsp` worker would run it (eagerly, in a thread)."""
    from src.workers import wsp_tasks

    return wsp_tasks.import_wsp_candidate.apply(kwargs=kwargs, task_id=task_id).get(disable_sync_subtasks=False)


async def _accepted(mcp, world):
    """File, approve, re-call: the accepted answer and the one task it sent."""
    first = await _call(mcp, "import_wsp_candidate", BODY)
    assert first["status"] == "pending_approval", first
    rid = await _approve_latest()
    started = await _call(mcp, "import_wsp_candidate", {**BODY, "approval_request_id": rid})
    assert started["status"] == "importing", started
    assert len(world.sent) == 1
    kwargs, task_id = world.sent.pop()
    assert task_id and started["task_id"] == task_id and started["task_queue_id"].startswith("tq_"), started
    return started, kwargs, task_id


async def _poll(mcp, started):
    """The poll the answer names, called as an agent would."""
    step = started["next_steps"][0]
    assert step == {**step, "tool": "get_task_status", "args": {"task_id": started["task_id"]}}, step
    return (await _call(mcp, step["tool"], step["args"]))["data"]


@pytest.mark.asyncio
async def test_an_import_reads_running_then_completed_as_the_new_probe(async_session, world):
    mcp = _server()
    started, kwargs, task_id = await _accepted(mcp, world)
    queued = await _poll(mcp, started)
    assert queued["id"] == started["task_queue_id"] and queued["task_type"] == "wsp_candidate_import"
    assert queued["status"] == "queued" and queued["entity_id"] == world.parent and queued["task_id"] == task_id

    worker = asyncio.create_task(asyncio.to_thread(_run_task, kwargs, task_id))
    assert await asyncio.to_thread(world.entered.wait, 30), "the task never reached the import"
    running = await _poll(mcp, started)
    assert running["status"] == "running" and running["entity_id"] == world.parent, running

    world.release.set()
    result = await worker
    assert result["probe_id"] == world.derived
    done = await _poll(mcp, started)
    assert done["status"] == "completed" and done["progress"] == 100.0, done
    assert done["entity_id"] == world.derived and done["entity_type"] == "probe_monitor", done
    assert done["entity_info"]["rung"] == 2 and done["entity_info"]["derived_from_probe_id"] == world.parent, done
    assert done["error_message"] is None and done["completed_at"]


@pytest.mark.asyncio
async def test_a_refused_import_reads_failed_with_its_code(async_session, world):
    mcp = _server()
    world.refuse = "patch_hash_mismatch"
    started, kwargs, task_id = await _accepted(mcp, world)
    world.release.set()
    result = await asyncio.to_thread(_run_task, kwargs, task_id)
    assert result["refused"] == "patch_hash_mismatch"
    failed = await _poll(mcp, started)
    assert failed["status"] == "failed" and failed["entity_id"] == world.parent, failed
    assert failed["error_message"].startswith("refused patch_hash_mismatch: "), failed


@pytest.mark.asyncio
async def test_an_import_that_raises_reads_failed_with_the_exception(async_session, world, monkeypatch):
    from src.services import wsp_import

    def boom(db, **_):
        raise RuntimeError("the worker lost its disk")

    monkeypatch.setattr(wsp_import, "import_candidate", boom)
    mcp = _server()
    started, kwargs, task_id = await _accepted(mcp, world)
    with pytest.raises(RuntimeError):
        await asyncio.to_thread(_run_task, kwargs, task_id)
    failed = await _poll(mcp, started)
    assert failed["status"] == "failed" and failed["error_message"] == "RuntimeError: the worker lost its disk"


@pytest.mark.asyncio
async def test_a_row_that_cannot_be_written_sends_nothing(async_session, world, monkeypatch):
    """No row, no send: the import would run unpollable, which is the defect."""
    from src.workers import jlens_progress

    monkeypatch.setattr(jlens_progress, "open_row", lambda *a, **k: None)
    async with httpx.AsyncClient(transport=_asgi(), base_url="http://t") as operator:
        response = await operator.post(ROUTE, json=BODY)
    assert response.status_code == 503, response.text
    assert world.sent == []


@pytest.mark.asyncio
async def test_a_send_that_fails_fails_its_row(async_session, world, monkeypatch, sync_db):
    from src.models.task_queue import TaskQueue
    from src.workers import wsp_tasks

    def refuse(**_):
        raise ConnectionError("broker down")

    monkeypatch.setattr(wsp_tasks.import_wsp_candidate, "apply_async", refuse)
    async with httpx.AsyncClient(transport=_asgi(), base_url="http://t") as operator:
        response = await operator.post(ROUTE, json=BODY)
    assert response.status_code == 500
    rows = sync_db.query(TaskQueue).filter(TaskQueue.task_type == "wsp_candidate_import",
                                           TaskQueue.entity_id == world.parent).all()
    assert [r.status for r in rows] == ["failed"] and rows[0].error_message.startswith("not dispatched: "), rows


def test_the_live_tool_names_a_poll_that_resolves_its_task():
    """From the LIVE registry (the categories `k8s/base/mcp.yaml` deploys): the import's description names
    `get_task_status` as its next step, that tool is registered beside it, and it takes the `task_id` the import
    answers (the tests above call it with exactly that and read the import)."""
    import re
    from pathlib import Path

    manifest = (Path(__file__).resolve().parents[3] / "k8s" / "base" / "mcp.yaml").read_text()
    found = re.search(r"- name: MCP_TOOL_CATEGORIES\n(?:\s+#.*\n)*\s+value: \"([^\"]+)\"", manifest)
    assert found, "MCP_TOOL_CATEGORIES not found in k8s/base/mcp.yaml"
    mcp = _server(found.group(1))
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    description = tools["import_wsp_candidate"].description
    named = re.findall(r"call `([a-z_]+)` with that `task_id`", description)
    assert named == ["get_task_status"], description
    assert "get_task_status" in tools
    assert "task_id" in tools["get_task_status"].input_schema["properties"]
    from src.mcp_server.tools.wsp import IMPORT_POLL_TOOL

    assert IMPORT_POLL_TOOL == named[0]
