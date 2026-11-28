"""The GGUF context ceiling over MCP (miServer contract v1.30, 2026-11-21).

miServer takes a context ceiling two ways: for ONE load, as the body `{"context_ceiling": …}` of
`POST /api/models/{id}/load` (approved as `model_load`, on the card and in the digest), and stored on
the model row by `PUT /api/models/{id}/context-ceiling`, which an agent's write files as the new action
`model_context_write`. miWorkbench reaches both:

* `miserver_load_model(context_ceiling=…)` sends the body ONLY when a ceiling is named, and only after
  `GET /api/approvals/actions` lists `model_context_write`: a miServer before v1.30 ignores an unknown
  body field and would load at its own default with nobody noticing, so it is refused
  (`miserver_context_ceiling_unsupported`, with a `fix`) and sent nothing;
* `miserver_set_model_context_ceiling(model_id, context_ceiling)` is relayed like
  `miserver_set_telemetry`: the fail-closed queue check, ONE PUT, miServer's pending answer unchanged
  plus `decide_in: "miserver"`;
* `miserver_list_models` / `miserver_get_model` pass `context_ceiling`, `kv_cache_type`,
  `supports_embeddings` and `gguf_load` through unchanged.

Everything runs through the LIVE registry: `build_server` with the categories the deployed pod enables
(`k8s/base/mcp.yaml`) and the real `MiServerClient` over an `httpx.MockTransport` standing in for
miServer. Each test asserts the route, the method, the exact body and the request count.

Wiring controls (0xcc/reviews/miserver_context_ceiling_tools_2026-11-21.md): unregister the new tool;
drop the body from the load; drop the capability check; drop the queue check from the new tool.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.support.cross_repo import require_cross_repo
from tests.support.sibling_repos import miserver_package, miserver_path_or_none, miserver_repo
from tests.support.miserver_fakes import as_sent

BACKEND = Path(__file__).resolve().parents[2]
MIWORKBENCH_URL = "http://mistudio.test"
MISERVER_URL = "http://millm.test"
#: The actions a miServer before v1.30 lists (ENH-008 plus v1.23's telemetry_write).
OLD_ACTIONS = ["model_load", "model_unload", "model_delete", "probe_limits_write", "capacity_measure",
               "code_rule_import", "telemetry_write"]
V130_ACTIONS = OLD_ACTIONS + ["model_context_write"]

LOAD = ("POST", "/api/models/7/load")
PUT = ("PUT", "/api/models/7/context-ceiling")


def _manifest_categories() -> str:
    import yaml

    for doc in yaml.safe_load_all((BACKEND.parent / "k8s" / "base" / "mcp.yaml").read_text()):
        for container in (((doc or {}).get("spec") or {}).get("template") or {}).get("spec", {}).get("containers", []):
            for e in container.get("env", []):
                if e.get("name") == "MCP_TOOL_CATEGORIES":
                    return e["value"]
    raise AssertionError("no MCP_TOOL_CATEGORIES in k8s/base/mcp.yaml")


def _pending(action: str) -> dict[str, Any]:
    return {"approval_id": f"apr_{action}", "status": "pending", "action": action,
            "summary": {"Model": "m", "Context ceiling": "max"}, "request_digest": "sha256:" + "a" * 64,
            "expires_at": "2026-11-22T12:00:00Z",
            "hint": "Waiting for the operator to approve this in miServer's Approvals page."}


#: A v1.30 record of the loaded model (miServer `ModelResponse`), the four new fields included.
LOADED_ROW = {
    "id": 17, "name": "unsloth/Qwen3.6-27B-GGUF:Q4_K_M", "status": "loaded", "gguf_label": "Q4_K_M",
    "context_ceiling": 0, "context_length": 155648, "kv_cache_type": "q8_0", "supports_embeddings": False,
    "gguf_load": {"kv_cache_type": "q8_0", "kv_cache_reason": "configured",
                  "embeddings": False, "embeddings_reason": "recurrent state 38304 MiB at 256 sequences",
                  "context": {"ceiling": 0, "ceiling_source": "model", "declared": 262144,
                              "predicted_fit": 155648, "started_at": 155648, "loaded": 155648},
                  "attempts": [{"n_ctx": 155648, "ok": True}]},
}
READY_ROW = {"id": 18, "name": "g:IQ4_XS", "status": "ready", "context_ceiling": 65536,
             "kv_cache_type": None, "supports_embeddings": None, "gguf_load": None}


class _MiServer:
    """miServer listing `actions` (None: no queue at all). Records every request but the gate's probe."""

    def __init__(self, actions: Any = V130_ACTIONS, *, status: int = 202, body: Any = None,
                 rows: Any = None):
        self.actions = actions
        self.status = status
        self.body = body
        self.rows = rows
        self.seen: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content) if request.content else None
        self.seen.append({"method": request.method, "path": path, "body": body,
                          "raw": request.content, "headers": dict(request.headers)})
        if path == "/api/approvals/actions":
            if self.actions is None:
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json={"success": True, "data": {
                "actions": [{"action": a, "label": a} for a in self.actions]}, "error": None})
        if path == "/api/models":
            return httpx.Response(200, json={"success": True, "data": self.rows, "error": None})
        if path == "/api/models/17":
            return httpx.Response(200, json={"success": True, "data": LOADED_ROW, "error": None})
        if path == "/api/health/inference":
            return httpx.Response(200, json={"context_length": 4096})
        action = {LOAD: "model_load", PUT: "model_context_write"}.get((request.method, path))
        if action is not None:
            if self.status == 202:
                return httpx.Response(202, json={"success": True, "data": _pending(action), "error": None},
                                      headers={"X-miLLM-Approval": "pending"})
            return httpx.Response(self.status, json=self.body)
        return httpx.Response(404, json={"detail": "Not Found"})


@pytest.fixture
def live(monkeypatch):
    """`build(miserver)` → `(call, names, tools)` against the deployed-category server."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    def build(miserver):
        real = httpx.AsyncClient

        def miworkbench(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "ok"})

        class Routed(real):
            def __init__(self, *args, **kwargs):
                base = str(kwargs.get("base_url", ""))
                kwargs["transport"] = httpx.MockTransport(miworkbench if base.startswith(MIWORKBENCH_URL) else as_sent(miserver))
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", Routed)
        monkeypatch.setenv("MIWORKBENCH_API_URL", MIWORKBENCH_URL)
        monkeypatch.setenv("MISERVER_API_URL", MISERVER_URL)
        monkeypatch.setenv("MCP_TOOL_CATEGORIES", _manifest_categories())
        mcp, _ = build_server(MCPSettings(auth_token="x" * 32), stdio=True)

        def call(tool: str, args: dict) -> Any:
            return asyncio.run(mcp._tool_manager.call_tool(tool, args, context=None))  # noqa: SLF001

        tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
        names = [t.name for t in asyncio.run(mcp.list_tools())]
        return call, names, tools

    return build


def _requests(miserver: _MiServer) -> list[tuple[str, str]]:
    return [(r["method"], r["path"]) for r in miserver.seen]


# ── the live registry ─────────────────────────────────────────────────────────────────────────────


def test_the_new_tool_is_in_the_live_registry_once_with_the_deployed_categories(live):
    _call, names, _tools = live(_MiServer())
    assert names.count("miserver_set_model_context_ceiling") == 1
    assert names.count("miserver_load_model") == 1


def test_both_tools_take_the_ceiling_as_a_number_max_or_null(live):
    _call, _names, tools = live(_MiServer())
    for name, required in (("miserver_load_model", False), ("miserver_set_model_context_ceiling", True)):
        schema = tools[name].input_schema
        prop = schema["properties"]["context_ceiling"]
        kinds = json.dumps(prop)
        assert '"integer"' in kinds and '"max"' in kinds and '"null"' in kinds, (name, prop)
        assert ("context_ceiling" in schema.get("required", [])) is required, (name, schema.get("required"))
        assert "operator" in tools[name].description or "Approvals" in tools[name].description


def test_the_new_tool_is_relayed_as_model_context_write():
    from src.mcp_server.tools.miserver_approvals import RELAYED_ACTIONS
    from src.mcp_server.tools.miserver_models import CONTEXT_CEILING_MARKER

    assert RELAYED_ACTIONS["miserver_set_model_context_ceiling"] == "model_context_write"
    assert CONTEXT_CEILING_MARKER == "model_context_write"


# ── miserver_load_model: the body, only when named, only on a v1.30 miServer ──────────────────────────


@pytest.mark.parametrize("ceiling", [2048, 65536, "max"])
def test_a_named_ceiling_is_the_load_body_after_exactly_one_check(live, ceiling):
    miserver = _MiServer()
    call, _, _ = live(miserver)

    result = call("miserver_load_model", {"model_id": 7, "context_ceiling": ceiling})

    # 045 FR-7 adds `approval` beside the relayed body; every relayed key is unchanged (B4 port, 2026-11-24).
    assert {k: v for k, v in result.items() if k not in ("approval", "next_steps")} == {
        **_pending("model_load"), "decide_in": "miserver"}
    # 045 follow-up (2026-11-24): the poll comes first in `next_steps`, with miServer's id.
    assert result["next_steps"][0]["tool"] == "miserver_get_approval_status"
    assert result["next_steps"][0]["args"] == {"approval_id": result["approval_id"]}
    assert result["approval"]["decide_in"] == "miserver" and result["approval"]["then"] == "server_runs"
    assert result["approval"]["action"] == "model_load"
    # ONE read of the action list serves both the queue check and the capability check.
    assert _requests(miserver) == [("GET", "/api/approvals/actions"), LOAD]
    assert miserver.seen[1]["body"] == {"context_ceiling": ceiling}
    assert all(r["headers"].get("x-millm-agent") == "agent:miworkbench-mcp" for r in miserver.seen)


def test_a_load_with_a_ceiling_and_a_lease_keeps_the_lease_out_of_the_body(live):
    miserver = _MiServer()
    call, _, _ = live(miserver)
    call("miserver_load_model", {"model_id": 7, "context_ceiling": "max", "lease_id": "L-secret"})
    load = [r for r in miserver.seen if (r["method"], r["path"]) == LOAD]
    assert len(load) == 1
    assert load[0]["body"] == {"context_ceiling": "max"}
    assert load[0]["headers"].get("x-millm-lease") == "L-secret"


@pytest.mark.parametrize("actions", [OLD_ACTIONS, V130_ACTIONS])
def test_a_load_without_a_ceiling_sends_no_body_to_any_miserver(live, actions):
    """No ceiling is miServer's default, exactly as before v1.30, so an older miServer still loads."""
    miserver = _MiServer(actions=actions)
    call, _, _ = live(miserver)
    result = call("miserver_load_model", {"model_id": 7})
    assert result["status"] == "pending" and result["decide_in"] == "miserver"
    assert _requests(miserver) == [("GET", "/api/approvals/actions"), LOAD]
    assert miserver.seen[1]["raw"] == b""


def test_a_ceiling_on_a_miserver_before_v1_30_is_refused_with_a_fix_and_nothing_is_loaded(live):
    miserver = _MiServer(actions=OLD_ACTIONS)
    call, _, _ = live(miserver)

    result = call("miserver_load_model", {"model_id": 7, "context_ceiling": 131072})

    assert result["refused"] == "miserver_context_ceiling_unsupported", result
    assert result["marker"] == "model_context_write"
    assert "Nothing was sent" in result["reason"] and "ignore" in result["reason"]
    assert "without context_ceiling" in result["fix"] and "v1.30" in result["fix"]
    assert _requests(miserver) == [("GET", "/api/approvals/actions")]


def test_a_miserver_with_no_queue_is_refused_before_the_capability_is_asked(live):
    miserver = _MiServer(actions=None)
    call, _, _ = live(miserver)
    result = call("miserver_load_model", {"model_id": 7, "context_ceiling": "max"})
    assert result["refused"] == "millm_has_no_approval_queue"
    assert _requests(miserver) == [("GET", "/api/approvals/actions")]


@pytest.mark.parametrize("tool", ["miserver_load_model", "miserver_set_model_context_ceiling"])
@pytest.mark.parametrize("ceiling", [0, 2047, 1024, -1])
def test_a_ceiling_miserver_cannot_take_is_refused_here_with_zero_requests(live, tool, ceiling):
    miserver = _MiServer()
    call, _, _ = live(miserver)
    result = call(tool, {"model_id": 7, "context_ceiling": ceiling})
    assert result["refused"] == "invalid_context_ceiling" and result["context_ceiling"] == ceiling
    assert "2048" in result["fix"] and "Nothing was sent" in result["reason"]
    if ceiling == 0:
        assert '"max"' in result["fix"]
    assert miserver.seen == []


@pytest.mark.parametrize("tool", ["miserver_load_model", "miserver_set_model_context_ceiling"])
@pytest.mark.parametrize("ceiling", ["32k", "MAX", True, 65536.5, "65536"])
def test_a_value_that_is_not_a_number_or_max_never_leaves(live, tool, ceiling):
    """Strict: `true` is not read as 1 and "65536" is not read as a number (miServer refuses both)."""
    miserver = _MiServer()
    call, _, _ = live(miserver)
    try:
        result = call(tool, {"model_id": 7, "context_ceiling": ceiling})
    except Exception as exc:  # the argument schema refuses it before the tool runs
        assert "context_ceiling" in str(exc)
    else:
        assert result.get("refused") == "invalid_context_ceiling", result
    assert miserver.seen == []


def test_miservers_422_for_a_ceiling_reaches_the_agent_as_a_refusal(live):
    body = {"success": False, "data": None, "error": {
        "code": "VALIDATION_ERROR", "message": "context_ceiling 524288 is above the 262144 tokens the model declares",
        "details": {"declared": 262144}}}
    miserver = _MiServer(status=422, body=body)
    call, _, _ = live(miserver)
    result = call("miserver_load_model", {"model_id": 7, "context_ceiling": 524288})
    assert result == {"refused": "millm_refused", "status_code": 422, "code": "VALIDATION_ERROR",
                      "message": "context_ceiling 524288 is above the 262144 tokens the model declares",
                      "details": {"declared": 262144}}
    assert _requests(miserver) == [("GET", "/api/approvals/actions"), LOAD]


# ── miserver_set_model_context_ceiling: relayed like miserver_set_telemetry ───────────────────────────


@pytest.mark.parametrize("ceiling", [131072, "max", None])
def test_the_stored_ceiling_is_one_put_with_exactly_that_body(live, ceiling):
    miserver = _MiServer()
    call, _, _ = live(miserver)

    result = call("miserver_set_model_context_ceiling", {"model_id": 7, "context_ceiling": ceiling})

    # 045 FR-7 adds `approval` beside the relayed body; every relayed key is unchanged (B4 port, 2026-11-24).
    assert {k: v for k, v in result.items() if k not in ("approval", "next_steps")} == {
        **_pending("model_context_write"), "decide_in": "miserver"}
    # 045 follow-up (2026-11-24): the poll comes first in `next_steps`, with miServer's id.
    assert result["next_steps"][0]["tool"] == "miserver_get_approval_status"
    assert result["next_steps"][0]["args"] == {"approval_id": result["approval_id"]}
    assert result["approval"]["decide_in"] == "miserver" and result["approval"]["then"] == "server_runs"
    assert result["approval"]["action"] == "model_context_write"
    assert _requests(miserver) == [("GET", "/api/approvals/actions"), PUT]
    assert miserver.seen[1]["body"] == {"context_ceiling": ceiling}
    assert all(r["headers"].get("x-millm-agent") == "agent:miworkbench-mcp" for r in miserver.seen)


@pytest.mark.parametrize("actions", [OLD_ACTIONS, None])
def test_the_stored_ceiling_fails_closed_on_a_miserver_that_does_not_gate_it(live, actions):
    miserver = _MiServer(actions=actions)
    call, _, _ = live(miserver)
    result = call("miserver_set_model_context_ceiling", {"model_id": 7, "context_ceiling": "max"})
    assert result["refused"] == "millm_has_no_approval_queue"
    if actions is not None:
        assert result["action"] == "model_context_write"
    assert _requests(miserver) == [("GET", "/api/approvals/actions")]


def test_a_miserver_refusal_of_the_stored_ceiling_is_passed_through(live):
    body = {"success": False, "data": None, "error": {
        "code": "VALIDATION_ERROR", "message": "a context ceiling applies to GGUF models only",
        "details": None}}
    miserver = _MiServer(status=422, body=body)
    call, _, _ = live(miserver)
    result = call("miserver_set_model_context_ceiling", {"model_id": 7, "context_ceiling": 65536})
    assert result == {"refused": "millm_refused", "status_code": 422, "code": "VALIDATION_ERROR",
                      "message": "a context ceiling applies to GGUF models only", "details": None}
    assert _requests(miserver) == [("GET", "/api/approvals/actions"), PUT]


def test_the_stored_ceiling_with_miserver_down_answers_unavailable(live):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    call, _, _ = live(down)
    result = call("miserver_set_model_context_ceiling", {"model_id": 7, "context_ceiling": "max"})
    assert result["unavailable"] == "miserver"


# ── the record's new fields pass through unchanged ───────────────────────────────────────────────────

NEW_FIELDS = ("context_ceiling", "kv_cache_type", "supports_embeddings", "gguf_load")


def test_get_model_passes_the_four_fields_through_unchanged(live):
    miserver = _MiServer()
    call, _, _ = live(miserver)
    result = call("miserver_get_model", {"model_id": 17})
    assert {k: result[k] for k in NEW_FIELDS} == {k: LOADED_ROW[k] for k in NEW_FIELDS}
    assert result["context_length"] == 155648
    assert _requests(miserver) == [("GET", "/api/models/17")]


def test_list_models_passes_the_four_fields_through_and_trusts_the_records_window(live):
    miserver = _MiServer(rows=[LOADED_ROW, READY_ROW])
    call, _, _ = live(miserver)
    result = call("miserver_list_models", {})
    loaded, ready = result["models"]
    assert {k: loaded[k] for k in NEW_FIELDS} == {k: LOADED_ROW[k] for k in NEW_FIELDS}
    assert {k: ready[k] for k in NEW_FIELDS} == {k: READY_ROW[k] for k in NEW_FIELDS}
    # v1.30: the record carries the loaded window, so the engine is not asked (it would say 4096 here).
    assert loaded["context_length"] == 155648 and "context_length_note" not in loaded
    assert ready.get("context_length") is None  # 046: miServer always sends it (null on a row not loaded)
    assert _requests(miserver) == [("GET", "/api/models")]


def test_an_older_record_without_the_window_still_gets_the_engines(live):
    old = {"id": 17, "name": "m", "status": "loaded"}
    miserver = _MiServer(rows=[old])
    call, _, _ = live(miserver)
    loaded = call("miserver_list_models", {})["models"][0]
    assert loaded["context_length"] == 4096
    assert _requests(miserver) == [("GET", "/api/models"), ("GET", "/api/health/inference")]


# ── cross-repo: the marker, the floor and the values are miServer's ──────────────────────────────────

_DUMP = (
    "import json\n"
    "from {pkg}.main import create_app\n"
    "from {pkg}.api.approval_gate import APPROVAL_ACTIONS, gated_routes\n"
    "from {pkg}.services import context_ceiling as c\n"
    "from {pkg}.api.schemas.model import ModelLoadRequest as L, ContextCeilingRequest as C\n"
    "a = APPROVAL_ACTIONS.get('model_context_write')\n"
    "routes = sorted([m, p, x] for m, p, x in gated_routes(create_app())"
    " if x in ('model_context_write', 'model_load'))\n"
    "print('@@CTX@@' + json.dumps({'batch_kind': getattr(a, 'batch_kind', None), 'min': c.MIN_CONTEXT,"
    " 'max_word': c.CEILING_MAX_WORD, 'load': [L(context_ceiling=v).context_ceiling for v in (2048, 'max')],"
    " 'put_null': C(context_ceiling=None).context_ceiling, 'routes': routes}))\n"
)


def test_the_marker_floor_and_routes_are_miservers():
    """CROSS-REPO. The capability marker is an action miServer's LIVE route table gates on the PUT this
    tool calls; the floor is miServer's `MIN_CONTEXT`; the values the tools send are ones miServer's
    request schemas accept. Skips loudly without the sibling; fails under
    MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1."""
    from src.mcp_server.tools.miserver_models import CEILING_MAX, CONTEXT_CEILING_MARKER, MIN_CONTEXT_CEILING

    repo = miserver_repo()
    python = Path(os.environ.get("MISERVER_PYTHON") or repo / "venv" / "bin" / "python")
    module = miserver_path_or_none(repo, "services", "context_ceiling.py")
    if module is None or not python.is_file():
        reason = f"no miServer checkout/interpreter at {repo} / {python} (set MISERVER_REPO, MISERVER_PYTHON)"
        if require_cross_repo():
            pytest.fail(reason)
        pytest.skip("CROSS-REPO, SKIPPED LOUDLY: " + reason)
    if not module.is_file():
        reason = f"the miServer at {repo} predates contract v1.30 (no services/context_ceiling.py)"
        if require_cross_repo():
            pytest.fail(reason)
        pytest.skip("CROSS-REPO, SKIPPED LOUDLY: " + reason)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_URL", "PYTHONPATH"))}
    proc = subprocess.run([str(python), "-B", "-c", _DUMP.replace("{pkg}", miserver_package(repo))], cwd=repo,
                          env=env, capture_output=True, text=True, timeout=240)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@CTX@@")]
    if proc.returncode != 0 or not lines:
        pytest.fail(f"miServer's app did not build (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    facts = json.loads(lines[0][len("@@CTX@@"):])
    assert [CONTEXT_CEILING_MARKER, facts["batch_kind"]] == ["model_context_write", "settings"]
    assert ["PUT", "/api/models/{model_id}/context-ceiling", CONTEXT_CEILING_MARKER] in facts["routes"]
    assert ["POST", "/api/models/{model_id}/load", "model_load"] in facts["routes"]
    assert facts["min"] == MIN_CONTEXT_CEILING
    assert facts["max_word"] == CEILING_MAX
    assert facts["load"] == [2048, 0] and facts["put_null"] is None
