"""ENH-010 044 (FR-15; FTASKS 4.4, 14.2, 15.1): the two `wsp` tools, held on the LIVE registry.

Every test here builds the real server (`build_server`) and calls the tool the server registered, through the real
`MiWorkbenchClient` whose HTTP transport records each request — never `wsp.register` on a server of its own, so a
tool whose registration (`CATEGORY_MODULES["wsp"]`, `VALID_CATEGORIES`, `DEFAULT_CATEGORIES`, the deployment's
`MCP_TOOL_CATEGORIES`) is removed fails here.

* both tools are in the registry the deployment builds (its manifest's categories) and the default one;
* `capture_wsp_heldout`, first call: reads its preflight, files ONE approval with the route's exact canonical payload
  (and its GPU card beside it), never calls the route; the approved re-call reads the approval, then calls the route
  ONCE with that body and the id in the approval-request header, and never consumes it itself (the route does);
* a capture the GPU cannot take is refused before anything is filed; a capture with nothing to run files nothing;
* `import_wsp_candidate`: the same two shapes, one call and two;
* both are gated with `MCP_PROBE_GPU_APPROVAL` off (the route refuses the MCP pod without an id whatever it says);
* a pending, mismatched or spent approval stops the tool before the route; a route refusal of the approval is a status;
  the handler's own refusal reaches the agent verbatim.

Controls (record 0xcc/reviews/wsp_gated_capture_import_2026-11-19.md): delete either `@mcp.tool()` in
`tools/wsp.py` → the registry tests red; call the route on the first call (drop the `_gate` line) → the first-call
tests red; key the gate on `settings.probe_gpu_approval` → the setting-off tests red.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest

from src.core.approval_payload import APPROVAL_REQUEST_HEADER, canonical_payload
from src.mcp_server.client import AGENT_HEADER, AGENT_NAME, BackendError
from src.mcp_server.config import MCPSettings
from mcp.server.mcpserver.exceptions import ToolError

REPO = Path(__file__).resolve().parents[3]
WSP_TOOLS = {"capture_wsp_heldout", "import_wsp_candidate"}

CAPTURE_ARGS = {"probe_id": "pm_0123456789ab", "dataset_ids": ["pmd_eval", "pmd_cal"]}
CAPTURE_BODY = {"probe_id": "pm_0123456789ab", "dataset_ids": ["pmd_eval", "pmd_cal"]}
CAPTURE_FILED = {"_method": "POST", "_route": "/api/v1/wsp/captures", **CAPTURE_BODY}

RUN_DOC = {"run_id": "wsr_1", "candidate_id": "wsc_1", "verdict": {"value": "regressed"}}
MANIFEST = {"patch_sha256": "a" * 64}
ACK = {"verdict": "regressed", "reason": "operator wants it measured anyway"}
IMPORT_ARGS = {"run_document": RUN_DOC, "patch_manifest": MANIFEST, "edit_npz_b64": "QUJD", "acknowledgement": ACK}
IMPORT_BODY = {"run_document": RUN_DOC, "patch_manifest": MANIFEST, "edit_npz_b64": "QUJD", "acknowledgement": ACK}
IMPORT_FILED = {"_method": "POST", "_route": "/api/v1/wsp/candidates/import", **IMPORT_BODY}

SETS = [{"dataset_id": "pmd_eval", "role": "eval", "rows": 3, "capture": None, "status": None, "would_run": True},
        {"dataset_id": "pmd_cal", "role": "calibration", "rows": 2, "capture": None, "status": None,
         "would_run": True}]
GPU_FITS = {"checked": True, "fits": True, "required_mb": 4096, "requested": "auto", "model_id": "m_1",
            "quantization": "FP16", "card": {"index": 0, "uuid": "GPU-x", "name": "card", "total_mb": 24576,
                                              "free_mb": 20000},
            "cards": [{"index": 0}], "message": "GPU 0 can take it"}
PREFLIGHT = {"probe_id": "pm_0123456789ab", "sets": SETS, "to_run": 2, "rows_to_run": 5, "gpu": GPU_FITS,
             "refusal": None}


def _manifest_categories() -> str:
    text = (REPO / "k8s" / "base" / "mcp.yaml").read_text()
    found = re.search(r"- name: MCP_TOOL_CATEGORIES\n(?:\s+#.*\n)*\s+value: \"([^\"]+)\"", text)
    assert found, "MCP_TOOL_CATEGORIES not found in k8s/base/mcp.yaml"
    return found.group(1)


def _registered(settings: MCPSettings) -> set[str]:
    from src.mcp_server.server import build_server

    mcp, _ = build_server(settings, stdio=True)
    return {t.name for t in asyncio.run(mcp.list_tools())}


class _Backend:
    """Answers the MCP client's requests as miWorkbench would, and records each one."""

    def __init__(self, *, preflight=None, approval=None, route=None):
        self.calls: list[dict] = []
        self.preflight = PREFLIGHT if preflight is None else preflight
        self.approval = approval  # the row GET /mcp/approvals/{id} answers; None → 404
        self.route = route  # (status, json) the gated route answers

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path.removeprefix("/api/v1")
        self.calls.append({"method": request.method, "path": path, "query": dict(request.url.params.multi_items()),
                           "query_lists": request.url.params.get_list("dataset_ids"), "json": body,
                           "approval_header": request.headers.get(APPROVAL_REQUEST_HEADER),
                           "agent": request.headers.get(AGENT_HEADER)})
        if path == "/wsp/captures/preflight":
            return httpx.Response(200, json=self.preflight)
        if request.method == "POST" and path == "/mcp/approvals":
            return httpx.Response(201, json={"id": "apr_1", "status": "pending"})
        if request.method == "GET" and path.startswith("/mcp/approvals/"):
            if self.approval is None:
                return httpx.Response(404, json={"detail": f"Approval request {path.rsplit('/', 1)[1]} not found"})
            return httpx.Response(200, json=self.approval)
        if path in ("/wsp/captures", "/wsp/candidates/import"):
            status, payload = self.route or (202, {"task_id": "t_1", "status": "queued"})
            return httpx.Response(status, json=payload)
        return httpx.Response(500, json={"detail": f"unexpected {request.method} {path}"})


def _server(backend: _Backend, *, gpu_approval: bool = True):
    """The REAL built server (default categories), its real client's transport recording into `backend`."""
    from src.mcp_server.server import build_server

    mcp, client = build_server(MCPSettings(allow_anonymous=True, probe_gpu_approval=gpu_approval), stdio=True)
    asyncio.run(client.close())
    client._http = httpx.AsyncClient(  # noqa: SLF001
        transport=httpx.MockTransport(backend.handler), base_url="http://t/api/v1",
        headers={AGENT_HEADER: AGENT_NAME})
    return mcp


def _call(mcp, tool: str, args: dict):
    return asyncio.run(mcp._tool_manager.call_tool(tool, args, context=None))  # noqa: SLF001


def _approved(tool: str, payload: dict, status: str = "approved") -> dict:
    return {"id": "apr_1", "tool_name": tool, "payload": payload, "status": status, "reason": None}


# ── the live registry ────────────────────────────────────────────────────────────────────────────────────────────


def test_the_deployment_registers_both_tools():
    """The categories k8s/base/mcp.yaml deploys build a server with both tools (memory: a category registered in
    code and not in the manifest is unreachable in production)."""
    names = _registered(MCPSettings(tool_categories=_manifest_categories(), allow_anonymous=True))
    assert WSP_TOOLS <= names, sorted(WSP_TOOLS - names)
    assert "wsp" in _manifest_categories().split(",")


def test_the_default_categories_register_both_tools():
    names = _registered(MCPSettings(allow_anonymous=True))
    assert WSP_TOOLS <= names, sorted(WSP_TOOLS - names)


def test_the_wsp_category_is_exactly_these_tools():
    with_wsp = _registered(MCPSettings(tool_categories="read,wsp", allow_anonymous=True))
    without = _registered(MCPSettings(tool_categories="read", allow_anonymous=True))
    assert with_wsp - without == WSP_TOOLS


# ── capture_wsp_heldout ──────────────────────────────────────────────────────────────────────────────────────────


def test_capture_first_call_files_one_approval_and_never_calls_the_route():
    backend = _Backend()
    result = _call(_server(backend), "capture_wsp_heldout", CAPTURE_ARGS)
    assert result["status"] == "pending_approval" and result["approval_request_id"] == "apr_1"
    assert [(c["method"], c["path"]) for c in backend.calls] == [("GET", "/wsp/captures/preflight"),
                                                                  ("POST", "/mcp/approvals")]
    assert backend.calls[0]["query_lists"] == ["pmd_eval", "pmd_cal"]
    assert backend.calls[0]["query"]["probe_id"] == "pm_0123456789ab"
    filed = backend.calls[1]["json"]
    assert filed["tool_name"] == "capture_wsp_heldout"
    assert filed["payload"] == CAPTURE_FILED
    # The card's GPU line and sets ride BESIDE the payload: never approved, never compared.
    assert filed["forecast"]["gpu"]["summary"] == "GPU 0 can take it"
    assert filed["forecast"]["gpu"]["required_mb"] == 4096 and filed["forecast"]["gpu"]["fits"] is True
    assert [s["dataset_id"] for s in filed["forecast"]["sets"]] == ["pmd_eval", "pmd_cal"]
    assert all(c["agent"] == "mcp" for c in backend.calls)


def test_capture_approved_recall_calls_the_route_once_with_the_id_and_never_consumes():
    backend = _Backend(approval=_approved("capture_wsp_heldout", CAPTURE_FILED))
    result = _call(_server(backend), "capture_wsp_heldout", {**CAPTURE_ARGS, "approval_request_id": "apr_1"})
    assert result == {"task_id": "t_1", "status": "queued"}
    assert [(c["method"], c["path"]) for c in backend.calls] == [
        ("GET", "/wsp/captures/preflight"), ("GET", "/mcp/approvals/apr_1"), ("POST", "/wsp/captures")]
    route = backend.calls[2]
    assert route["json"] == CAPTURE_BODY and route["approval_header"] == "apr_1"
    # The route's gate rebuilds the approved payload from exactly this body.
    assert canonical_payload("POST", "/api/v1/wsp/captures", {}, {}, route["json"]) == CAPTURE_FILED
    assert not [c for c in backend.calls if c["path"].endswith("/consume")]


def test_capture_without_dataset_ids_sends_none():
    backend = _Backend()
    _call(_server(backend), "capture_wsp_heldout", {"probe_id": "pm_0123456789ab"})
    assert backend.calls[0]["query_lists"] == []
    assert backend.calls[1]["json"]["payload"] == {"_method": "POST", "_route": "/api/v1/wsp/captures",
                                                   "probe_id": "pm_0123456789ab"}


@pytest.mark.parametrize("status", ["pending", "denied", "expired", "consumed", "failed"])
def test_capture_recall_on_an_unusable_approval_never_reaches_the_route(status):
    backend = _Backend(approval=_approved("capture_wsp_heldout", CAPTURE_FILED, status))
    result = _call(_server(backend), "capture_wsp_heldout", {**CAPTURE_ARGS, "approval_request_id": "apr_1"})
    assert result["status"] == status and result["approval_request_id"] == "apr_1"
    assert ("POST", "/wsp/captures") not in [(c["method"], c["path"]) for c in backend.calls]


def test_capture_recall_with_other_arguments_is_a_mismatch():
    backend = _Backend(approval=_approved("capture_wsp_heldout", CAPTURE_FILED))
    result = _call(_server(backend), "capture_wsp_heldout",
                   {"probe_id": "pm_0123456789ab", "dataset_ids": ["pmd_eval"], "approval_request_id": "apr_1"})
    assert result["status"] == "mismatch"
    assert ("POST", "/wsp/captures") not in [(c["method"], c["path"]) for c in backend.calls]


def test_capture_recall_on_another_tools_approval_is_a_mismatch():
    backend = _Backend(approval=_approved("start_probe_training_run", CAPTURE_FILED))
    result = _call(_server(backend), "capture_wsp_heldout", {**CAPTURE_ARGS, "approval_request_id": "apr_1"})
    assert result["status"] == "mismatch"
    assert ("POST", "/wsp/captures") not in [(c["method"], c["path"]) for c in backend.calls]


def test_capture_recall_on_an_unknown_id_is_not_found():
    backend = _Backend(approval=None)
    result = _call(_server(backend), "capture_wsp_heldout", {**CAPTURE_ARGS, "approval_request_id": "apr_9"})
    assert result["status"] == "not_found"
    assert len(backend.calls) == 2


def test_a_full_gpu_is_refused_before_anything_is_filed():
    """2026-11-19 (live): a capture started on a full card and every set failed CUDA out of memory."""
    full = {**PREFLIGHT, "gpu": {**GPU_FITS, "fits": False, "card": None, "message": "No single GPU has ~4,096 MB"},
            "refusal": {"code": "gpu_memory_insufficient", "message": "No single GPU has ~4,096 MB"}}
    for args in (CAPTURE_ARGS, {**CAPTURE_ARGS, "approval_request_id": "apr_1"}):
        backend = _Backend(preflight=full, approval=_approved("capture_wsp_heldout", CAPTURE_FILED))
        result = _call(_server(backend), "capture_wsp_heldout", args)
        assert result["refused"] == "gpu_memory_insufficient" and "4,096" in result["reason"]
        assert [(c["method"], c["path"]) for c in backend.calls] == [("GET", "/wsp/captures/preflight")]


def test_nothing_to_run_files_nothing():
    kept = {**PREFLIGHT, "to_run": 0, "sets": [{**s, "would_run": False, "status": "ready"} for s in SETS]}
    backend = _Backend(preflight=kept)
    result = _call(_server(backend), "capture_wsp_heldout", CAPTURE_ARGS)
    assert result["started"] == [] and "nothing filed" in result["note"]
    assert len(backend.calls) == 1


def test_capture_is_gated_with_gpu_approval_off():
    """FTASKS 4.4 (R4B-13): never keyed on `MCP_PROBE_GPU_APPROVAL` — the route refuses the MCP pod without an id."""
    backend = _Backend()
    result = _call(_server(backend, gpu_approval=False), "capture_wsp_heldout", CAPTURE_ARGS)
    assert result["status"] == "pending_approval"
    assert [(c["method"], c["path"]) for c in backend.calls] == [("GET", "/wsp/captures/preflight"),
                                                                  ("POST", "/mcp/approvals")]


# ── import_wsp_candidate ─────────────────────────────────────────────────────────────────────────────────────────


def test_import_first_call_files_one_approval_and_never_calls_the_route():
    backend = _Backend()
    result = _call(_server(backend), "import_wsp_candidate", IMPORT_ARGS)
    assert result["status"] == "pending_approval" and result["approval_request_id"] == "apr_1"
    assert [(c["method"], c["path"]) for c in backend.calls] == [("POST", "/mcp/approvals")]
    assert backend.calls[0]["json"] == {"tool_name": "import_wsp_candidate", "payload": IMPORT_FILED}


def test_import_approved_recall_calls_the_route_once_with_the_id():
    backend = _Backend(approval=_approved("import_wsp_candidate", IMPORT_FILED))
    result = _call(_server(backend), "import_wsp_candidate", {**IMPORT_ARGS, "approval_request_id": "apr_1"})
    assert result == {"task_id": "t_1", "status": "queued"}
    assert [(c["method"], c["path"]) for c in backend.calls] == [("GET", "/mcp/approvals/apr_1"),
                                                                  ("POST", "/wsp/candidates/import")]
    route = backend.calls[1]
    assert route["json"] == IMPORT_BODY and route["approval_header"] == "apr_1"
    assert canonical_payload("POST", "/api/v1/wsp/candidates/import", {}, {}, route["json"]) == IMPORT_FILED


def test_import_without_acknowledgement_sends_none():
    backend = _Backend()
    args = {k: v for k, v in IMPORT_ARGS.items() if k != "acknowledgement"}
    _call(_server(backend), "import_wsp_candidate", args)
    assert "acknowledgement" not in backend.calls[0]["json"]["payload"]


def test_import_is_gated_with_gpu_approval_off():
    """FTASKS 14.3: the import is always gated, never by the MCP pod's GPU setting."""
    backend = _Backend()
    result = _call(_server(backend, gpu_approval=False), "import_wsp_candidate", IMPORT_ARGS)
    assert result["status"] == "pending_approval"
    assert [(c["method"], c["path"]) for c in backend.calls] == [("POST", "/mcp/approvals")]


# ── the route's answers to an approved re-call ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("answer,status", [
    ((409, {"detail": {"status": "consumed", "reason": None}}), "consumed"),
    ((409, {"detail": {"status": "failed", "reason": "HTTPException: 409"}}), "failed"),
    ((404, {"detail": "Approval request apr_1 not found"}), "not_found"),
    ((422, {"detail": {"code": "mismatch", "action": "import_wsp_candidate"}}), "mismatch"),
])
def test_a_route_refusal_of_the_approval_is_a_status(answer, status):
    backend = _Backend(approval=_approved("import_wsp_candidate", IMPORT_FILED), route=answer)
    result = _call(_server(backend), "import_wsp_candidate", {**IMPORT_ARGS, "approval_request_id": "apr_1"})
    assert result["status"] == status and result["approval_request_id"] == "apr_1"


@pytest.mark.parametrize("answer", [
    (409, {"detail": {"code": "acknowledgement_required", "message": "name the verdict"}}),
    (404, {"detail": {"code": "probe_not_found", "message": "probe pm_x"}}),
])
def test_the_handlers_own_refusal_reaches_the_agent(answer):
    """A `{code, …}` 404/409 is the HANDLER's, after the approval was spent; it is never read as the approval's."""
    backend = _Backend(approval=_approved("import_wsp_candidate", IMPORT_FILED), route=answer)
    with pytest.raises(ToolError) as raised:  # the tool manager wraps the client's BackendError, text verbatim
        _call(_server(backend), "import_wsp_candidate", {**IMPORT_ARGS, "approval_request_id": "apr_1"})
    assert f"backend {answer[0]}" in str(raised.value) and answer[1]["detail"]["code"] in str(raised.value)
    assert isinstance(raised.value.__cause__, BackendError)


# ── the guidance names live tools ────────────────────────────────────────────────────────────────────────────────


def test_the_howto_topic_teaches_the_live_tools():
    from src.mcp_server.tools import howto
    from tests.support.next_calls import live_input_schemas

    text = howto.TOPICS["wsp"]
    named = set(re.findall(r"`([a-z][a-z0-9_]+)`", text))
    taught = WSP_TOOLS | {"get_approval_status", "get_task_status", "gpu_preflight", "evaluate_probe_monitor"}
    assert taught <= named, sorted(taught - named)
    assert taught <= set(live_input_schemas()), sorted(taught - set(live_input_schemas()))


def test_the_server_instructions_name_both_tools():
    from src.mcp_server.server import SERVER_INSTRUCTIONS

    for tool in WSP_TOOLS:
        assert tool in SERVER_INSTRUCTIONS
