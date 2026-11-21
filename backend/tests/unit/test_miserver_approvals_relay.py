"""miWorkbench's `miserver_*` proxies RELAY miServer's own approval answer (ENH-008 S2, S3; spec §3.2 steps 2-4).

miServer decides its own requests (D1). The six tools that change what miServer serves — `miserver_load_model`,
`miserver_unload_model`, `miserver_delete_model`, `miserver_request_probe_limits`, `miserver_measure_probe_capacity` and
`miserver_import_rule_set` for operator code — send the request to miServer with `X-miLLM-Agent`; miServer stores it and
answers 202 with `X-miLLM-Approval: pending`; the tool returns miServer's body unchanged plus `"decide_in": "miserver"`.

Everything here runs through the LIVE registry: `build_server` with the categories the deployed pod enables
(`k8s/base/mcp.yaml`) and a miServer URL, the real `MiServerClient` on an `httpx.MockTransport` standing in for miServer, and
miWorkbench's own client on a recorder. So each assertion covers the registration, the client's header and the tool:

* S2: a pending answer is returned as miServer sent it (+ `decide_in`), with exactly TWO miServer requests — the
  fail-closed `GET /api/approvals/actions`, then the route with its exact body — and ZERO requests to miWorkbench's
  `/mcp/approvals`;
* S3: a miServer without `/api/approvals/actions` (404), or one whose list lacks the action, gets the
  `millm_has_no_approval_queue` refusal and ZERO requests on the action route;
* the deprecated `approval_request_id` answers `approval_moved_to_millm` and sends nothing anywhere;
* the two new read tools issue exactly one request each, with the parameters given.

Mutation controls (0xcc/reviews/enh008_controls_2026-10-28.md): skip the queue check in one tool; make
`ApprovalQueueCheck.refusal` return None for a missing action; return the body without `relayed()`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

BACKEND = Path(__file__).resolve().parents[2]
MIWORKBENCH_URL = "http://mistudio.test"
MISERVER_URL = "http://millm.test"
ALL_ACTIONS = ["model_load", "model_unload", "model_delete", "probe_limits_write", "capacity_measure",
               "code_rule_import", "model_context_write"]
CODE_SHA = "c" * 64


def _manifest_categories() -> str:
    import yaml

    for doc in yaml.safe_load_all((BACKEND.parent / "k8s" / "base" / "mcp.yaml").read_text()):
        for container in (((doc or {}).get("spec") or {}).get("template") or {}).get("spec", {}).get("containers", []):
            for e in container.get("env", []):
                if e.get("name") == "MCP_TOOL_CATEGORIES":
                    return e["value"]
    raise AssertionError("no MCP_TOOL_CATEGORIES in k8s/base/mcp.yaml")


def _pending(action: str) -> dict[str, Any]:
    """miServer's 202 body (spec §3.1, `ApiResponse.ok({...})`), unwrapped by the client."""
    return {"approval_id": f"apr_{action}", "status": "pending", "action": action,
            "summary": {"model": "m"}, "request_digest": "sha256:" + "a" * 64,
            "expires_at": "2026-10-29T12:00:00Z",
            "hint": "Waiting for the operator to approve this in miServer's Approvals page."}


class _MiServer:
    """miServer with (or without) its approval queue. Records every request except the gate's health probe."""

    def __init__(self, actions: Any = "all", *, pending: bool = True):
        self.actions = ALL_ACTIONS if actions == "all" else actions
        self.pending = pending
        self.seen: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content) if request.content else None
        self.seen.append({"method": request.method, "path": path, "body": body,
                          "params": dict(request.url.params), "headers": dict(request.headers)})
        if path == "/api/approvals/actions":
            if self.actions is None:
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json={"success": True, "data": {
                "actions": [{"action": a, "label": a} for a in self.actions]}, "error": None})
        if path.startswith("/api/approvals"):
            return httpx.Response(200, json={"success": True, "data": {"approvals": [], "count": 0, "pending": 0}})
        action = _ACTION_FOR.get((request.method, path))
        if action and self.pending:
            return httpx.Response(202, json={"success": True, "data": _pending(action), "error": None},
                                  headers={"X-miLLM-Approval": "pending"})
        return httpx.Response(200, json={"success": True, "data": {"id": 7, "status": "loading"}, "error": None})


_ACTION_FOR = {
    ("POST", "/api/models/7/load"): "model_load",
    ("POST", "/api/models/7/unload"): "model_unload",
    ("DELETE", "/api/models/7"): "model_delete",
    ("PUT", "/api/models/7/probe-limits"): "probe_limits_write",
    ("POST", "/api/models/7/probe-capacity/measure"): "capacity_measure",
    ("POST", "/api/rule-sets/import"): "code_rule_import",
    # miServer contract v1.30: an agent's write of a model's stored context ceiling.
    ("PUT", "/api/models/7/context-ceiling"): "model_context_write",
}


def _code_rule_set() -> dict[str, Any]:
    return {"format": "midataworks.rule-set/v1", "name": "code", "behaviour_key": "<behaviour-key>",
            "signals": [{"id": "c", "kind": "code_rule", "code_sha256": CODE_SHA, "function": "f"}],
            "expressions": {"hit": {"signal": "c"}}, "outputs": [], "cases": []}


#: tool → (arguments, the ONE action request it must send: method, path, body, params).
RELAYED = {
    "miserver_load_model": ({"model_id": 7}, ("POST", "/api/models/7/load", None, {})),
    "miserver_unload_model": ({"model_id": 7}, ("POST", "/api/models/7/unload", None, {})),
    "miserver_delete_model": ({"model_id": 7, "acknowledge_permanent": True}, ("DELETE", "/api/models/7", None, {})),
    "miserver_request_probe_limits": (
        {"model_id": 7, "reason": "5% of the model's own time", "decode_budget": {"value": 5, "unit": "pct"}},
        ("PUT", "/api/models/7/probe-limits",
         {"reason": "5% of the model's own time", "set_by": "mistudio-agent", "source": "operator",
          "decode_budget": {"value": 5.0, "unit": "pct"}}, {})),
    "miserver_measure_probe_capacity": ({"model_id": 7}, ("POST", "/api/models/7/probe-capacity/measure", None, {})),
    "miserver_import_rule_set": ({"document": _code_rule_set(), "version": 2},
                              ("POST", "/api/rule-sets/import", _code_rule_set(), {"version": "2"})),
    "miserver_set_model_context_ceiling": ({"model_id": 7, "context_ceiling": 131072},
                                           ("PUT", "/api/models/7/context-ceiling", {"context_ceiling": 131072}, {})),
}


@pytest.fixture
def live(monkeypatch):
    """`build(miserver)` → `(call, miworkbench_seen)`: tool calls against the deployed-category server."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    def build(miserver: _MiServer):
        miworkbench_seen: list[tuple[str, str]] = []

        def miworkbench(request: httpx.Request) -> httpx.Response:
            miworkbench_seen.append((request.method, request.url.path))
            return httpx.Response(200, json={"status": "ok"})

        real = httpx.AsyncClient

        class Routed(real):
            def __init__(self, *args, **kwargs):
                base = str(kwargs.get("base_url", ""))
                kwargs["transport"] = httpx.MockTransport(miworkbench if base.startswith(MIWORKBENCH_URL) else miserver)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", Routed)
        monkeypatch.setenv("MIWORKBENCH_API_URL", MIWORKBENCH_URL)
        monkeypatch.setenv("MISERVER_API_URL", MISERVER_URL)
        monkeypatch.setenv("MCP_TOOL_CATEGORIES", _manifest_categories())
        mcp, _ = build_server(MCPSettings(auth_token="x" * 32), stdio=True)

        def call(tool: str, args: dict) -> Any:
            return asyncio.run(mcp._tool_manager.call_tool(tool, args, context=None))  # noqa: SLF001

        return call, miworkbench_seen

    return build


def _action_requests(miserver: _MiServer) -> list[dict]:
    return [r for r in miserver.seen if not r["path"].startswith("/api/approvals")]


# ── S2: a pending answer is relayed, with two miServer calls and none to miWorkbench's queue ─────────────────────────────


@pytest.mark.parametrize("tool", sorted(RELAYED))
def test_a_pending_answer_is_relayed_unchanged_with_decide_in(live, tool):
    args, (method, path, body, params) = RELAYED[tool]
    miserver = _MiServer()
    call, miworkbench_seen = live(miserver)

    result = call(tool, args)

    action = _ACTION_FOR[(method, path)]
    assert result == {**_pending(action), "decide_in": "miserver"}
    # Exactly two miServer requests, in this order: the fail-closed check, then the action with ITS body.
    assert [(r["method"], r["path"]) for r in miserver.seen] == [("GET", "/api/approvals/actions"), (method, path)]
    sent = miserver.seen[1]
    assert sent["body"] == body and sent["params"] == params
    assert all(r["headers"].get("x-millm-agent") == "agent:miworkbench-mcp" for r in miserver.seen)
    assert not [p for p in miworkbench_seen if p[1].startswith("/api/v1/mcp/approvals")], (
        "the relay filed in miWorkbench's queue: miServer decides its own requests now")


def test_the_load_sends_the_lease_ONLY_as_a_header(live):
    miserver = _MiServer()
    call, _ = live(miserver)
    call("miserver_load_model", {"model_id": 7, "lease_id": "L-secret"})
    load = _action_requests(miserver)
    assert len(load) == 1
    assert load[0]["headers"].get("x-millm-lease") == "L-secret"
    assert load[0]["body"] is None and "L-secret" not in json.dumps(load[0]["params"])


def test_a_load_miServer_simply_starts_reads_as_started(live):
    """No `X-miLLM-Approval: pending` (an older pending-less answer, or an operator call): the 034 shape."""
    miserver = _MiServer(pending=False)
    call, _ = live(miserver)
    result = call("miserver_load_model", {"model_id": 7})
    assert result["load_started"] is True
    assert "decide_in" not in result


def test_a_202_without_the_pending_header_is_NOT_a_pending_answer():
    from src.mcp_server.miserver_client import MiServerResponse
    from src.mcp_server.tools.miserver_approvals import relayed

    assert relayed(MiServerResponse(202, {"status": "loading"}, {})) is None
    assert relayed(MiServerResponse(200, {"x": 1}, {"x-millm-approval": "pending"})) is None
    assert relayed(MiServerResponse(202, {"approval_id": "a"}, {"X-miLLM-Approval": "Pending"})) == {
        "approval_id": "a", "decide_in": "miserver"}


def test_a_declarative_import_is_ONE_request_and_needs_no_queue(live):
    doc = {"format": "midataworks.rule-set/v1", "signals": [{"id": "p", "kind": "pattern"}]}
    miserver = _MiServer(actions=None)  # even a miServer with no queue: nothing here is gated
    call, _ = live(miserver)
    call("miserver_import_rule_set", {"document": doc})
    assert [(r["method"], r["path"]) for r in miserver.seen] == [("POST", "/api/rule-sets/import")]


# ── S3: fail closed against a miServer without its queue ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tool", sorted(RELAYED))
def test_a_miServer_without_the_queue_is_refused_and_sent_nothing(live, tool):
    args, (method, path, _body, _params) = RELAYED[tool]
    miserver = _MiServer(actions=None)
    call, _ = live(miserver)
    result = call(tool, args)
    assert result["refused"] == "millm_has_no_approval_queue", result
    assert _action_requests(miserver) == [], f"{tool} sent {method} {path} to a miServer that cannot gate it"


@pytest.mark.parametrize("tool", sorted(RELAYED))
def test_a_queue_that_does_not_list_the_action_is_refused_too(live, tool):
    args, (method, path, _body, _params) = RELAYED[tool]
    action = _ACTION_FOR[(method, path)]
    miserver = _MiServer(actions=[a for a in ALL_ACTIONS if a != action])
    call, _ = live(miserver)
    result = call(tool, args)
    assert result["refused"] == "millm_has_no_approval_queue" and result["action"] == action
    assert _action_requests(miserver) == []


def test_an_unreadable_queue_is_refused_not_trusted():
    """A 500 on the actions route is not an empty list and not a yes: refuse, and do not cache it."""
    from unittest.mock import MagicMock

    from src.mcp_server.client import BackendError
    from src.mcp_server.tools.miserver_approvals import ApprovalQueueCheck

    calls = []

    class Miserver:
        async def get(self, path, **_):
            calls.append(path)
            raise BackendError(500, "boom")

    gate = MagicMock()
    gate.ttl_s = 60.0
    check = ApprovalQueueCheck(Miserver(), gate)
    assert asyncio.run(check.refusal("model_load"))["refused"] == "millm_has_no_approval_queue"
    asyncio.run(check.refusal("model_load"))
    assert calls == ["/api/approvals/actions"] * 2


def test_the_answer_is_cached_for_the_gate_window_only():
    from unittest.mock import MagicMock

    from src.mcp_server.tools.miserver_approvals import ApprovalQueueCheck

    calls = []

    class Miserver:
        async def get(self, path, **_):
            calls.append(path)
            return {"actions": ["model_load"]}

    gate = MagicMock()
    gate.ttl_s = 60.0
    check = ApprovalQueueCheck(Miserver(), gate)
    assert asyncio.run(check.refusal("model_load")) is None
    assert asyncio.run(check.refusal("model_load")) is None
    assert len(calls) == 1
    gate.ttl_s = 0.0
    asyncio.run(check.refusal("model_load"))
    assert len(calls) == 2


@pytest.mark.parametrize("body,expected", [
    (["model_load"], {"model_load"}),
    ([{"action": "model_load"}, {"name": "model_unload"}, {"id": "model_delete"}],
     {"model_load", "model_unload", "model_delete"}),
    ({"actions": [{"action": "model_load"}]}, {"model_load"}),
    ({"actions": {"model_load": {"label": "x"}}}, {"model_load"}),
    ({"model_load": "Load a model"}, {"model_load"}),
    (None, set()),
    ("model_load", set()),
])
def test_the_action_list_is_read_in_any_documented_shape(body, expected):
    from src.mcp_server.tools.miserver_approvals import _action_names

    assert _action_names(body) == expected


# ── the deprecated parameter ──────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tool", ["miserver_load_model", "miserver_request_probe_limits", "miserver_measure_probe_capacity",
                                  "miserver_import_rule_set"])
def test_the_old_approval_id_is_refused_and_names_the_new_tool(live, tool):
    args = dict(RELAYED[tool][0], approval_request_id="ap_old")
    miserver = _MiServer()
    call, miworkbench_seen = live(miserver)
    result = call(tool, args)
    assert result["refused"] == "approval_moved_to_millm" and result["next"] == "miserver_get_approval_status"
    assert miserver.seen == [] and miworkbench_seen == []


# ── the two new read tools ────────────────────────────────────────────────────────────────────────────────────────


def test_list_approvals_is_one_GET_with_its_filters(live):
    miserver = _MiServer()
    call, _ = live(miserver)
    call("miserver_list_approvals", {"status": "pending", "limit": 200})
    assert [(r["method"], r["path"], r["params"]) for r in miserver.seen] == [
        ("GET", "/api/approvals", {"status": "pending", "limit": "200"})]


def test_get_approval_status_is_one_GET_and_says_what_next(live):
    miserver = _MiServer()

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/approvals/apr_1":
            miserver.seen.append({"method": request.method, "path": request.url.path})
            return httpx.Response(200, json={"success": True, "data": {"id": "apr_1", "status": "rejected",
                                                                       "reason": "no"}})
        return miserver(request)

    call, _ = live(answer)
    result = call("miserver_get_approval_status", {"approval_id": "apr_1"})
    assert [(r["method"], r["path"]) for r in miserver.seen] == [("GET", "/api/approvals/apr_1")]
    assert result["status"] == "rejected" and "Rejected" in result["next"]


def test_an_unknown_id_reads_not_found(live):
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"success": False, "error": {"code": "approval_not_found", "message": "x"}})

    call, _ = live(answer)
    result = call("miserver_get_approval_status", {"approval_id": "apr_x"})
    assert result["status"] == "not_found"


def test_both_new_tools_are_in_the_live_registry_once(live):
    call, _ = live(_MiServer())
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(auth_token="x" * 32), stdio=True)
    names = [t.name for t in asyncio.run(mcp.list_tools())]
    assert names.count("miserver_list_approvals") == 1 and names.count("miserver_get_approval_status") == 1
    for tool in RELAYED:
        assert names.count(tool) == 1, tool
