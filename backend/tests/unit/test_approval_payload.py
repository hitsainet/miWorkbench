"""045 FR-13 (A29, A38; FTDD §16.3; FTASKS 20.3): every MCP filer builds its payload with the ONE canonical function.

`test_mcp_filers_use_the_shared_function` drives each MCP tool that files an approval (`_approval_gate`'s five callers
and the three steer tools) twice through a recording client — the first call files, the approved re-call reaches its
route — and requires the FILED payload to equal what the REST gate builds from the route call the re-call makes
(`canonical_payload` over the live app's route template, path parameters, query and body). So an approval the MCP pod
files and one an agent files over REST for the same call are the same object, and its action is the route's.

Controls (0xcc/reviews/045_A0_implementation_controls_2026-11-16.md):
  a hand-built payload in one filer (the run filer files its bare body)   → red
  drop `recut_windows` from the recalibrate filer's query                  → red
"""
from __future__ import annotations

import asyncio
import re

import pytest

from src.mcp_server.config import MCPSettings
from tests.support.fake_clients import workbench_over  # 046 FTASKS 7.5: the fake behind a transport

MISERVER = "http://millm-backend.millm.svc.cluster.local:8000/v1"


@pytest.fixture(autouse=True)
def _no_miserver_address(monkeypatch):
    for name in ("MILLM_API_URL", "MISERVER_API_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_steering_slot_left_held():
    """A steering re-call here submits a task that nothing polls; its guardrail slot (module state) is released so a
    later test's steering call is not refused by it."""
    from src.mcp_server.tools import steering as steering_tools

    yield
    steering_tools._inflight.clear()  # noqa: SLF001


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, str, object, dict]] = []

    async def post(self, path, json_body=None, *, timeout=None, headers=None, **params):
        self.calls.append(("POST", path, json_body, {k: v for k, v in params.items() if v is not None}))
        if path == "/mcp/approvals":
            return {"id": "apr_1", "status": "pending"}
        # 046 FTASKS 7.5: each route's answer in its contract's shape (the fake is judged now)
        if path.endswith("/consume"):
            return {"id": path.split("/")[-2], "status": "consumed"}
        if path.startswith("/steering/async/"):
            return {"task_id": "t_1", "status": "queued"}
        # 046 P7: the probe routes' own queued answers (`ProbeTaskQueued`, `SpotCheckStarted`; the rest of each
        # answer is completed from its contract)
        if path == "/probe-monitors/runs/forecast":  # the forecast the run tool reads first (`ProbeRunForecast`)
            return {}
        if path.endswith(("/evaluate", "/recalibrate")):
            return {"probe_id": path.split("/")[-2], "task_id": "t_1", "status": "queued"}
        if path == "/probe-monitors/spot-checks":
            return {"id": "x_1", "status": "queued"}
        return {"id": "x_1", "task_id": "t_1", "status": "queued"}

    async def get(self, path, **params):
        self.calls.append(("GET", path, None, params))
        return {}


#: (category module, tool, arguments). Each files on its first call and calls one gated route on the re-call.
FILERS = [
    ("probe_monitors", "start_probe_training_run",
     {"model_id": "m_1", "train_dataset_id": "pmd_1", "eval_dataset_ids": ["pmd_2"],
      "config": {"layers": [11], "rules": ["mean"], "acknowledge_rule_choice": "fixture"}}),
    ("probe_monitors", "evaluate_probe_monitor", {"probe_id": "pm_1", "dataset_ids": ["pmd_2"], "sample_rows": 500}),
    ("probe_monitors", "recalibrate_probe_monitor",
     {"probe_id": "pm_1", "target_fpr": 0.01, "preview": False, "reason": "fixture", "recut_windows": True}),
    ("probe_monitors", "start_probe_judge_run",
     {"endpoint": MISERVER, "model": "qwen", "dataset_ids": ["pmd_2"], "max_rows_per_set": 200}),
    ("probe_monitors", "start_probe_spot_check",
     {"behaviour": "pii", "prompt_id": "jp_1", "endpoint": MISERVER, "model": "qwen", "size": 50}),
    ("steering", "steer_compare", {"sae_id": "s", "prompt": "p", "features": "FEATURES"}),
    ("steering", "steer_sweep",
     {"sae_id": "s", "prompt": "p", "feature_idx": 1, "layer": 0, "strength_values": [1.0, 5.0]}),
    ("steering", "steer_combined", {"sae_id": "s", "prompt": "p", "features": "FEATURES"}),
]


def _tool(module: str, name: str, client):
    import importlib

    from mcp.server.mcpserver import MCPServer

    mod = importlib.import_module(f"src.mcp_server.tools.{module}")
    mcp = MCPServer("t")
    mod.register(mcp, workbench_over(client, complete=True), MCPSettings(auth_token="x" * 32, probe_gpu_approval=True, steering_approval=True))
    return mcp._tool_manager._tools[name].fn  # noqa: SLF001


def _args(args):
    from src.mcp_server.tools.steering import SteerFeature

    return {k: ([SteerFeature(feature_idx=3, layer=1, strength=5.0)] if v == "FEATURES" else v)
            for k, v in args.items()}


def _gated_route(method: str, concrete: str):
    """(template, action, route) of the live gated route a concrete `/api/v1/...` path matches."""
    from src.core.approval_gate import gated_routes, iter_routes
    from src.main import app

    routes = {(m, path): route for path, route in iter_routes(app) for m in route.methods}
    for (m, template), action in gated_routes(app).items():
        pattern = "^" + re.sub(r"\{([^}]+)\}", r"(?P<\1>[^/]+)", template) + "$"
        hit = re.match(pattern, concrete)
        if m == method and hit:
            return template, action, routes[(m, template)], hit.groupdict()
    raise AssertionError(f"{method} {concrete} is not a gated route")


@pytest.mark.parametrize("module,tool,args", FILERS, ids=[f[1] for f in FILERS])
def test_mcp_filers_use_the_shared_function(module, tool, args):
    from src.core.approval_gate import body_param_name
    from src.core.approval_payload import canonical_payload

    client = _Recorder()
    fn = _tool(module, tool, client)
    asyncio.run(fn(**_args(args)))
    filings = [c for c in client.calls if c[1] == "/mcp/approvals"]
    assert len(filings) == 1, client.calls
    filed = filings[0][2]
    assert filed["tool_name"] == tool

    client.calls.clear()
    asyncio.run(fn(**_args(args), approval_request_id="apr_1"))
    sent = [c for c in client.calls if not c[1].startswith("/mcp/approvals")]
    assert len(sent) == 1, client.calls
    method, path, body, query = sent[0]
    template, action, route, path_params = _gated_route(method, "/api/v1" + path)
    assert action == tool, "the MCP tool files under the action its route consumes"
    rebuilt = canonical_payload(method, template, path_params, query, body, body_param=body_param_name(route))
    assert filed["payload"] == rebuilt, "the filed payload is the one the REST gate builds for that call"
