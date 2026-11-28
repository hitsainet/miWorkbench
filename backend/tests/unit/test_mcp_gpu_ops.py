"""G9/G10 MCP half (2026-10-21): `get_gpu_status`, `list_active_jobs`, `gpu_preflight` and the
unconditional `get_approval_status`, through the LIVE registry of the built server.

* Reachability: the tools are on `build_server()`'s `list_tools()` under the DEFAULT categories
  and the deployed manifest's categories — with steering approval OFF (the deployed setting),
  which is exactly where `get_approval_status` used to be missing.
* Payload + call count: each tool issues exactly the documented calls with exactly the documented
  arguments (a recording client; the preflight's sibling reads go over `httpx.MockTransport`
  through the REAL MiServerClient and the real miForge path).
* Standalone: with no MISERVER_API_URL and no MIFORGE_API_URL the preflight answers
  `configured: false` for each and still gives a verdict.

Controls (review record `0xcc/reviews/2026-10-21_mcp-gaps.md`, restored from saved bytes):
  C-REG     drop `gpu_ops` from CATEGORY_MODULES["jobs"] → the registry tests red
  C-APPREG  drop `approvals` from CATEGORY_MODULES["read"] → test_get_approval_status_is_on_… red
  C-MISERVER make `resident_on_this_card` always False → test_a_resident_miserver_model_makes_it_busy red
  C-FORGE   skip the miForge runs in `verdict` → test_miforge_runs_make_it_uncertain red
  Gaps (ag)/(ah), 2026-11-27 (`0xcc/reviews/gpu_preflight_idle_miserver_2026-11-27.md`): each fix line reverted
  turns the 16MB-unloaded row (and its named neighbours) red.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from tests.support.fake_clients import workbench_over  # 046 FTASKS 7.5: the fake behind a transport
from tests.support.miserver_fakes import as_sent

CARD = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"
OTHER = "GPU-f47ba814-49a2-603f-3595-275284140251"


def _built(categories: str, monkeypatch, **settings) -> set[str]:
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    for var in ("MISERVER_API_URL", "MIDATAWORKS_API_URL", "MIFORGE_API_URL"):
        monkeypatch.delenv(var, raising=False)
    mcp, _ = build_server(MCPSettings(tool_categories=categories, allow_anonymous=True,
                                      **settings), stdio=True)
    return {t.name for t in asyncio.run(mcp.list_tools())}


NEW = {"get_gpu_status", "list_active_jobs", "gpu_preflight", "get_approval_status"}


def test_the_new_tools_are_on_the_DEFAULT_build(monkeypatch):
    assert NEW <= _built("read,groups,steering,labeling,experiments,profiles,circuits,jlens,"
                         "jobs,models,probe_monitors", monkeypatch)


def test_the_new_tools_are_on_the_DEPLOYED_build(monkeypatch):
    from tests.unit.test_reachability import TestTheDEPLOYMENTEnablesWhatTheCodeRegisters as T

    deployed = ",".join(sorted(T()._deployed_categories()))
    assert NEW <= _built(deployed, monkeypatch, steering_approval=False)


def test_get_approval_status_is_on_a_read_only_build_with_approval_off(monkeypatch):
    names = _built("read", monkeypatch, steering_approval=False)
    assert "get_approval_status" in names
    assert "steer_sweep" not in names


def test_get_approval_status_is_registered_ONCE_with_steering_approval_on(monkeypatch):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(tool_categories="read,steering", allow_anonymous=True,
                                      steering_approval=True), stdio=True)
    names = [t.name for t in asyncio.run(mcp.list_tools())]
    assert names.count("get_approval_status") == 1


def test_the_gpu_tools_are_in_jobs_and_absent_without_it(monkeypatch):
    assert {"get_gpu_status", "list_active_jobs", "gpu_preflight"} <= _built("jobs", monkeypatch)
    assert not ({"get_gpu_status", "gpu_preflight"} & _built("read", monkeypatch))


# ── payload + call count ────────────────────────────────────────────────────────────────────


class Recorder:
    def __init__(self, answers=None):
        self.calls: list[tuple[str, str, dict]] = []
        self.answers = answers or {}

    async def get(self, path, **params):
        self.calls.append(("GET", path, params))
        answer = self.answers.get(path, {})
        if isinstance(answer, Exception):
            raise answer
        return answer


def _tools(client, **settings):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import approvals, gpu_ops

    mcp = MCPServer("t")
    s = MCPSettings(auth_token="x" * 32, _env_file=None, **settings)
    over = workbench_over(client, complete=True)
    gpu_ops.register(mcp, over, s)
    approvals.register(mcp, over, s)
    return mcp._tool_manager  # noqa: SLF001


def _call(manager, name, **args):
    return asyncio.run(manager.call_tool(name, args, context=None))


def test_get_gpu_status_payload():
    rec = Recorder()
    _call(_tools(rec), "get_gpu_status", gpu=CARD)
    assert rec.calls == [("GET", "/operations/gpu-status", {"gpu": CARD})]
    rec = Recorder()
    _call(_tools(rec), "get_gpu_status")
    assert rec.calls == [("GET", "/operations/gpu-status", {"gpu": None})]


def test_list_active_jobs_payload():
    rec = Recorder()
    _call(_tools(rec), "list_active_jobs", kind="probe_training", gpu=CARD)
    assert rec.calls == [("GET", "/operations/active-jobs", {"kind": "probe_training", "gpu": CARD})]


def test_list_active_jobs_names_the_probe_gpu_task_types_its_routes_write(monkeypatch):
    """Train W5 (2026-11-27): the description an agent reads in the LIVE registry names the `task_type` values the
    evaluate and WSP capture routes open their rows with, and the `progress_detail` they carry. Reverting the card +
    retry branch's description (evaluations back under "no row of its own") left every test green at the train head.
    The values come from the constants the routes and tasks import, so a renamed task type reds here too."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server
    from src.workers.jlens_progress import PROBE_EVALUATION, WSP_HELDOUT_CAPTURE

    for var in ("MISERVER_API_URL", "MIDATAWORKS_API_URL", "MIFORGE_API_URL"):
        monkeypatch.delenv(var, raising=False)
    mcp, _ = build_server(MCPSettings(tool_categories="jobs", allow_anonymous=True), stdio=True)
    tools = [t for t in asyncio.run(mcp.list_tools()) if t.name == "list_active_jobs"]
    assert len(tools) == 1
    text = " ".join((tools[0].description or "").split())
    for task_type in (PROBE_EVALUATION, WSP_HELDOUT_CAPTURE):
        assert task_type in text, task_type
    assert "progress_detail" in text
    assert "(an evaluation, a re-cut, a score)" not in text, "an evaluation has its own task_queue row now"


def _studio(card_in_use=False, jobs=()):
    card = {"uuid": CARD, "in_use": card_in_use,
            "in_use_by": ["miWorkbench lease held by training:t1"] if card_in_use else []}
    return {
        "/operations/gpu-status": {"cards": [card], "reason": None},
        "/operations/active-jobs": {"jobs": list(jobs), "errors": [], "read_at": "now"},
    }


def test_preflight_STANDALONE_is_clear_with_both_siblings_unconfigured(monkeypatch):
    for var in ("MISERVER_API_URL", "MIFORGE_API_URL"):
        monkeypatch.delenv(var, raising=False)
    rec = Recorder(_studio())
    out = _call(_tools(rec), "gpu_preflight", gpu=CARD)
    assert [c[:2] for c in rec.calls] == [("GET", "/operations/gpu-status"),
                                         ("GET", "/operations/active-jobs")]
    assert rec.calls[0][2] == {"gpu": CARD} and rec.calls[1][2] == {}
    assert out["verdict"] == "clear", out
    assert out["millm"]["configured"] is False and out["miforge"]["configured"] is False
    assert out["busy_reasons"] == [] and out["uncertain_reasons"] == []


def test_preflight_refuses_an_index():
    rec = Recorder(_studio())
    out = _call(_tools(rec), "gpu_preflight", gpu="0000")
    assert out["refused"] is True and rec.calls == []


def test_a_leased_card_is_busy(monkeypatch):
    for var in ("MISERVER_API_URL", "MIFORGE_API_URL"):
        monkeypatch.delenv(var, raising=False)
    out = _call(_tools(Recorder(_studio(card_in_use=True))), "gpu_preflight", gpu=CARD)
    assert out["verdict"] == "busy"
    assert "lease" in out["busy_reasons"][0]


def test_a_queued_job_with_no_card_makes_it_uncertain(monkeypatch):
    for var in ("MISERVER_API_URL", "MIFORGE_API_URL"):
        monkeypatch.delenv(var, raising=False)
    queued = {"kind": "probe_training", "id": "pmr_1", "status": "pending", "gpu_uuids": []}
    placed = {"kind": "sae_training", "id": "t2", "status": "queued", "gpu_uuids": [OTHER]}
    out = _call(_tools(Recorder(_studio(jobs=[queued, placed]))), "gpu_preflight", gpu=CARD)
    assert out["verdict"] == "uncertain"
    assert "pmr_1" in out["uncertain_reasons"][0] and "t2" not in out["uncertain_reasons"][0]


def test_a_backend_refusal_is_returned_not_raised(monkeypatch):
    from src.mcp_server.client import BackendError

    rec = Recorder({"/operations/gpu-status": BackendError(404, {"detail": "No GPU"})})
    out = _call(_tools(rec), "gpu_preflight", gpu=CARD)
    assert out["refused"] is True and out["status_code"] == 404


# ── siblings over the real clients ──────────────────────────────────────────────────────────


def _sibling_transports(monkeypatch, miserver_handler=None, forge_handler=None):
    from src.mcp_server.miserver_client import MiServerClient
    from src.mcp_server.tools import gpu_ops

    seen = {"millm": [], "forge": []}

    def miserver_factory(url):
        client = MiServerClient(url)

        def handler(request):
            seen["millm"].append((request.method, request.url.path))
            return miserver_handler(request)

        client._http = httpx.AsyncClient(base_url=url, transport=httpx.MockTransport(as_sent(handler)))  # noqa: SLF001
        return client

    def forge_factory(url):
        def handler(request):
            seen["forge"].append((request.method, request.url.path, dict(request.url.params)))
            return forge_handler(request)

        return httpx.AsyncClient(base_url=url, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(gpu_ops, "_miserver_client", miserver_factory)
    monkeypatch.setattr(gpu_ops, "_forge_client", forge_factory)
    return seen


def _miserver(resident_on=None, model="llama"):
    def handler(request):
        if request.url.path == "/api/health/detailed":
            return httpx.Response(200, json={"model_loaded": bool(resident_on), "model_name": model,
                                             "lease": None, "model_placement": None})
        cards = [{"smi_index": 0, "uuid": CARD.lower(), "torch_measured": resident_on == CARD,
                  "millm_allocated_mb": 15000 if resident_on == CARD else None,
                  "engine_memory": None}]
        return httpx.Response(200, json={"read_at": "x", "cards": cards})
    return handler


def test_a_resident_miserver_model_makes_it_busy(monkeypatch):
    monkeypatch.delenv("MIFORGE_API_URL", raising=False)
    seen = _sibling_transports(monkeypatch, miserver_handler=_miserver(resident_on=CARD))
    out = _call(_tools(Recorder(_studio()), miserver_api_url_override="http://millm.test"),
                "gpu_preflight", gpu=CARD)
    assert seen["millm"] == [("GET", "/api/health/detailed"), ("GET", "/api/health/gpus")]
    assert out["millm"]["resident_on_this_card"] is True
    assert out["verdict"] == "busy" and "miServer" in out["busy_reasons"][0]


def test_a_miserver_model_on_ANOTHER_card_leaves_this_one_clear(monkeypatch):
    monkeypatch.delenv("MIFORGE_API_URL", raising=False)
    _sibling_transports(monkeypatch, miserver_handler=_miserver(resident_on=OTHER))
    out = _call(_tools(Recorder(_studio()), miserver_api_url_override="http://millm.test"),
                "gpu_preflight", gpu=CARD)
    assert out["verdict"] == "clear", out


def test_a_configured_miserver_that_is_down_makes_it_uncertain_not_clear(monkeypatch):
    monkeypatch.delenv("MIFORGE_API_URL", raising=False)

    def down(request):
        raise httpx.ConnectError("refused")

    _sibling_transports(monkeypatch, miserver_handler=down)
    out = _call(_tools(Recorder(_studio()), miserver_api_url_override="http://millm.test"),
                "gpu_preflight", gpu=CARD)
    assert out["millm"] == {**out["millm"], "configured": True, "reachable": False}
    assert out["verdict"] == "uncertain"


def _forge(running=(), pending=(), body=None):
    def handler(request):
        if body is not None:
            return httpx.Response(200, content=body)
        status = request.url.params["status"]
        rows = running if status == "running" else pending
        return httpx.Response(200, json={"runs": [{"run_id": r, "status": status} for r in rows],
                                         "total": len(rows)})
    return handler


def test_miforge_runs_make_it_uncertain(monkeypatch):
    monkeypatch.delenv("MILLM_API_URL", raising=False)
    seen = _sibling_transports(monkeypatch, forge_handler=_forge(running=["run_a"],
                                                                 pending=["run_b"]))
    out = _call(_tools(Recorder(_studio()), miforge_api_url_override="http://forge.test"),
                "gpu_preflight", gpu=CARD)
    assert [(m, p, q["status"]) for m, p, q in seen["forge"]] == [
        ("GET", "/api/v1/runs", "running"), ("GET", "/api/v1/runs", "pending")]
    assert out["miforge"]["active_run_count"] == 2
    assert out["verdict"] == "uncertain"


def test_miforge_idle_is_clear(monkeypatch):
    monkeypatch.delenv("MILLM_API_URL", raising=False)
    _sibling_transports(monkeypatch, forge_handler=_forge())
    out = _call(_tools(Recorder(_studio()), miforge_api_url_override="http://forge.test"),
                "gpu_preflight", gpu=CARD)
    assert out["verdict"] == "clear" and out["miforge"]["active_run_count"] == 0


def test_a_miforge_html_page_is_NOT_an_empty_run_list(monkeypatch):
    monkeypatch.delenv("MILLM_API_URL", raising=False)
    _sibling_transports(monkeypatch, forge_handler=_forge(body=b"<html>splash</html>"))
    out = _call(_tools(Recorder(_studio()), miforge_api_url_override="http://forge.test"),
                "gpu_preflight", gpu=CARD)
    assert out["miforge"]["reachable"] is False and out["verdict"] == "uncertain"


def test_the_preflight_reads_the_env_urls(monkeypatch):
    monkeypatch.setenv("MISERVER_API_URL", "http://millm.env")
    monkeypatch.setenv("MIFORGE_API_URL", "http://forge.env/")
    from src.mcp_server.config import MCPSettings

    s = MCPSettings(auth_token="x" * 32, _env_file=None)
    assert s.miserver_api_url == "http://millm.env" and s.miforge_api_url == "http://forge.env"


# ── gaps (ag) and (ah), 2026-11-27: an idle miServer does not make a free card busy ────────────
#
# Live at 07:40 ET (W5 §3.1.2): miServer had NO model (`model_loaded: false`), the 3090 had 23,987 MiB free, and the
# preflight read `busy` twice over: "miServer's resident model None" (16 MB that torch keeps after an unload) and
# "1 unattributed process" (pid 627525, 380 MiB: miServer's own uvicorn holding its CUDA context). Every fixture here
# follows the real answers: miServer's from its vendored contract (`as_sent` completes them; the placement is the
# contract's `ModelPlacement`), miWorkbench's card from the REAL `gpu_status.assemble`.

MISERVER_PID = 627525
OTHER_PID = 700001


def _studio_card(pids, lease=None, jobs=()):
    """miWorkbench's card as `/operations/gpu-status` answers it: NVML's raw card through the real `assemble`."""
    from src.services.gpu_status import assemble

    raw = {"uuid": CARD, "index": 0, "name": "NVIDIA GeForce RTX 3090",
           "memory": {"total_mib": 24576.0, "used_mib": 589.0, "free_mib": 23987.0},
           "utilization": {"gpu_percent": 0, "memory_percent": 0},
           "processes": [{"pid": pid, "used_mib": mib} for pid, mib in pids]}
    leases = {CARD.lower(): lease} if lease else {}
    (card,) = assemble([raw], leases, list(jobs))
    return {"/operations/gpu-status": {"cards": [card], "reason": None, "worker_watch": {"state": "ok"}},
            "/operations/active-jobs": {"jobs": list(jobs), "errors": [], "read_at": "now"}}


def _miserver_card(uuid, torch_index, allocated, pids, *, measured=None, engine=None):
    """One `GpuCardMemory` (the contract's `/api/health/gpus` card), every required field set."""
    return {"smi_index": torch_index, "uuid": uuid, "name": "NVIDIA GeForce RTX 3090", "total_mb": 24576,
            "used_mb": 589, "free_mb": 23987, "torch_index": torch_index,
            "torch_measured": allocated is not None if measured is None else measured,
            "millm_allocated_mb": allocated, "millm_reserved_mb": None if allocated is None else allocated + 4,
            "engine_memory": engine, "processes": [{"pid": pid, "used_mb": mb} for pid, mb in pids],
            "processes_reason": None}


def _miserver_answers(cards, loaded=False, on=(), name="qwen"):
    """miServer's two health answers: `model_loaded`, and `model_placement` (null unless loaded, as miServer sends
    it) naming the torch indices `on`."""
    from tests.support.miserver_fakes import complete_def

    placement = None
    if loaded:
        placement = {**complete_def("ModelPlacement", {
            "mode": "single" if len(on) == 1 else "shard", "reason": "most_free_card_fits",
            "gpu_indices": list(on), "devices": [f"cuda:{i}" for i in on]}),
            "memory_by_device_mb": {f"cuda:{i}": 15000 for i in on}}

    def handler(request):
        if request.url.path == "/api/health/detailed":
            return httpx.Response(200, json={"model_loaded": loaded, "model_name": name if loaded else None,
                                             "lease": None, "model_placement": placement})
        return httpx.Response(200, json={"read_at": "2026-11-27T12:40:30Z", "cards": cards, "reason": None})
    return handler


def _preflight(monkeypatch, studio, handler):
    monkeypatch.delenv("MIFORGE_API_URL", raising=False)
    seen = _sibling_transports(monkeypatch, miserver_handler=handler)
    out = _call(_tools(Recorder(studio), miserver_api_url_override="http://millm.test"), "gpu_preflight", gpu=CARD)
    assert seen["millm"] == [("GET", "/api/health/detailed"), ("GET", "/api/health/gpus")]
    return out


@pytest.mark.parametrize("allocated,loaded,pids,expected", [
    (16, False, [(MISERVER_PID, 380)], "clear"),       # the live case: residual torch memory, no model
    (15000, True, [(MISERVER_PID, 15400)], "busy"),    # a model really loaded on this card
    (None, False, [], "clear"),                        # miServer never touched this card
], ids=["16MB-unloaded", "15000MB-loaded", "unmeasured"])
def test_the_residual_after_an_unload_is_not_a_resident_model(monkeypatch, allocated, loaded, pids, expected):
    out = _preflight(monkeypatch, _studio_card([(pid, mb - 0.5) for pid, mb in pids]),
                     _miserver_answers([_miserver_card(CARD.lower(), 0, allocated, pids)], loaded=loaded,
                                       on=[0] if loaded else []))
    assert out["verdict"] == expected, out
    assert out["millm"]["resident_on_this_card"] is loaded
    assert out["uncertain_reasons"] == []
    if loaded:
        assert out["busy_reasons"] == ["miServer's resident model 'qwen' is on this card"], out["busy_reasons"]
    else:
        assert out["busy_reasons"] == [], out["busy_reasons"]
    attribution = out["millm"]["process_attribution"]
    if pids:
        assert attribution == {"miserver_pid": MISERVER_PID,
                               "miserver_state": "model_resident" if loaded else "idle_context",
                               "unknown_count": 0, "unknown_pids": []}
    else:
        assert attribution == {"miserver_pid": None, "miserver_state": None, "unknown_count": 0, "unknown_pids": []}


def test_an_unknown_pid_beside_an_idle_miserver_is_still_busy(monkeypatch):
    pids = [(MISERVER_PID, 380), (OTHER_PID, 9000)]
    out = _preflight(monkeypatch, _studio_card(pids),
                     _miserver_answers([_miserver_card(CARD.lower(), 0, 16, pids)]))
    assert out["verdict"] == "busy", out
    (reason,) = out["busy_reasons"]
    assert reason.startswith("1 compute process(es) not attributable to a miWorkbench job or to miServer beside "
                             "miServer's own CUDA context"), reason
    assert out["millm"]["process_attribution"]["unknown_count"] == 1


def test_a_pid_miserver_does_not_list_is_unknown(monkeypatch):
    """miServer holds a context here, but the only process miWorkbench sees is one miServer's list does not show."""
    out = _preflight(monkeypatch, _studio_card([(OTHER_PID, 380)]),
                     _miserver_answers([_miserver_card(CARD.lower(), 0, 16, [(MISERVER_PID, 380)])]))
    assert out["verdict"] == "busy", out
    assert out["busy_reasons"][0].startswith("1 compute process(es) not attributable"), out["busy_reasons"]
    assert str(OTHER_PID) in out["busy_reasons"][0]


def test_being_listed_by_miserver_is_not_being_miservers(monkeypatch):
    """miServer's list is nvidia-smi's for the WHOLE card (every pod's processes), so a listed pid on a card where
    miServer holds no context (`torch_measured: false`, no GGUF) is someone else's: still busy."""
    pids = [(OTHER_PID, 9000)]
    out = _preflight(monkeypatch, _studio_card(pids),
                     _miserver_answers([_miserver_card(CARD.lower(), 0, None, pids)]))
    assert out["verdict"] == "busy", out
    assert out["millm"]["process_attribution"]["miserver_pid"] is None


def test_a_gguf_model_on_this_card_is_resident_and_its_process_is_miservers(monkeypatch):
    pids = [(MISERVER_PID, 9000)]
    out = _preflight(monkeypatch, _studio_card(pids),
                     _miserver_answers([_miserver_card(CARD.lower(), 0, None, pids, engine="not_measured_by_torch")],
                                       loaded=True, on=[0]))
    assert out["verdict"] == "busy" and out["busy_reasons"] == ["miServer's resident model 'qwen' is on this card"]
    assert out["millm"]["process_attribution"]["miserver_state"] == "model_resident"


def test_a_model_on_ANOTHER_card_leaves_this_cards_idle_context_clear(monkeypatch):
    """1+n cards: the model is on torch index 1; this card (index 0) keeps an earlier load's 16 MB and context."""
    pids = [(MISERVER_PID, 380)]
    cards = [_miserver_card(CARD.lower(), 0, 16, pids), _miserver_card(OTHER.lower(), 1, 15000, [(MISERVER_PID, 15400)])]
    out = _preflight(monkeypatch, _studio_card(pids), _miserver_answers(cards, loaded=True, on=[1]))
    assert out["millm"]["resident_on_this_card"] is False
    assert out["verdict"] == "clear", out
    assert out["millm"]["process_attribution"]["miserver_state"] == "idle_context"


def test_a_lease_and_a_running_job_stay_busy_beside_an_idle_miserver(monkeypatch):
    """The card's own reasons survive the process re-count, in `assemble`'s order (lease, jobs, process line)."""
    pids = [(MISERVER_PID, 380)]
    lease = {"holder": "probe_training:pmr_1", "kind": "probe_training", "task_id": "t"}
    job = {"kind": "probe_training", "id": "pmr_1", "status": "running", "gpu_uuids": [CARD]}
    out = _preflight(monkeypatch, _studio_card(pids, lease=lease, jobs=[job]),
                     _miserver_answers([_miserver_card(CARD.lower(), 0, 16, pids)]))
    assert out["verdict"] == "busy"
    assert out["busy_reasons"] == ["miWorkbench lease held by probe_training:pmr_1",
                                   "miWorkbench probe_training pmr_1 (running)"], out["busy_reasons"]


def test_miservers_attribution_needs_miserver_to_answer(monkeypatch):
    """Unconfigured miServer: the process is unknown (busy), exactly as before this fix."""
    for var in ("MISERVER_API_URL", "MIFORGE_API_URL"):
        monkeypatch.delenv(var, raising=False)
    out = _call(_tools(Recorder(_studio_card([(MISERVER_PID, 380)]))), "gpu_preflight", gpu=CARD)
    assert out["verdict"] == "busy" and len(out["busy_reasons"]) == 1, out
    assert out["busy_reasons"][0].startswith("1 compute process(es) not attributable")


# ── get_approval_status ─────────────────────────────────────────────────────────────────────


ROW = {"id": "a1", "tool_name": "start_probe_training_run", "payload": {"model_id": "m"},
       "status": "approved", "reason": None, "steering_task_id": None,
       "created_at": "2026-10-20T21:47:24Z", "resolved_at": "2026-10-20T21:58:00Z",
       "consumed_at": None}


def test_get_approval_status_payload_and_shape():
    rec = Recorder({"/mcp/approvals/a1": ROW})
    out = _call(_tools(rec), "get_approval_status", approval_request_id="a1")
    assert rec.calls == [("GET", "/mcp/approvals/a1", {})]
    assert out["status"] == "approved" and out["decided"] is True
    assert out["decided_at"] == ROW["resolved_at"] and out["decided_by"].startswith("operator")
    assert "Re-call" in out["next"]


@pytest.mark.parametrize("status,reason,who", [
    ("pending", None, None),
    ("denied", "too big", "operator"),
    ("consumed", None, "operator"),
    ("expired", "expired: pending for more than 24 h", "expiry"),
    ("expired", "expired: approved but not used within 24 h", "operator"),
])
def test_who_decided_per_status(status, reason, who):
    from src.mcp_server.tools.approvals import describe

    out = describe({**ROW, "status": status, "reason": reason})
    assert out["decided"] is (status != "pending")
    if who is None:
        assert out["decided_by"] is None
    else:
        assert out["decided_by"].startswith(who)
    assert out["next"]


def test_an_unknown_approval_is_not_found_not_an_exception():
    from src.mcp_server.client import BackendError

    rec = Recorder({"/mcp/approvals/zz": BackendError(404, "nope")})
    out = _call(_tools(rec), "get_approval_status", approval_request_id="zz")
    assert out["status"] == "not_found"


async def test_get_approval_status_against_the_REAL_route_never_consumes(async_session, client):
    """Through the real approvals route: create, approve, read with the tool — and the row is
    still `approved` afterwards (reading never spends it)."""
    from mcp.server.mcpserver import MCPServer

    from src.main import app
    from src.mcp_server.client import MiWorkbenchClient
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import approvals

    created = (await client.post("/api/v1/mcp/approvals", json={
        "tool_name": "start_probe_training_run", "payload": {"model_id": "m_x"}})).json()
    real = MiWorkbenchClient("http://test")
    real._http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),  # noqa: SLF001
                                   base_url="http://test/api/v1")
    mcp = MCPServer("t")
    approvals.register(mcp, real, MCPSettings(auth_token="x" * 32, _env_file=None))
    tool = mcp._tool_manager._tools["get_approval_status"]  # noqa: SLF001

    pending = await tool.fn(approval_request_id=created["id"])
    assert pending["status"] == "pending" and pending["decided"] is False
    await client.post(f"/api/v1/mcp/approvals/{created['id']}/approve")
    approved = await tool.fn(approval_request_id=created["id"])
    again = await tool.fn(approval_request_id=created["id"])
    assert approved["status"] == again["status"] == "approved"
    assert approved["decided_at"] and approved["consumed_at"] is None
    row = (await client.get(f"/api/v1/mcp/approvals/{created['id']}")).json()
    assert row["status"] == "approved", "reading must never consume"
    await real.close()
