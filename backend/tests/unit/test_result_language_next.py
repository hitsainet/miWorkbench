"""045 FR-8: `next_steps` — the contract's list of `{tool, args, why}` — beside every legacy `next`, which is never
touched (CHAIN_PLAN A6 amended 2026-11-15; A30).

`test_every_next_site_speaks_the_shape` drives each site of FTDD §8's inventory through the SDK's in-memory client
(the client also validates the answer against the tool's advertised output schema, so a string `next` passes only
because the envelope never declares a `next` property, A30/R1A-14) and asserts: `next_steps` validates item by item
against the vendored `next_step` `$def`; the legacy `next` equals the unwrapped tool's answer; the expected tools are
named. `result_language.STRICT` is on for the drive, so a site whose legacy `next` is prose or a bare name and which
sets no `next_steps` raises instead of passing.

Controls (0xcc/reviews/045_implementation_controls_2026-11-16.md, FTASKS 9.4):
  drop the `next_steps` addition in `add_next_steps`             → test_every_next_site_speaks_the_shape red
  let the wrapper rewrite the legacy `next`                       → the legacy comparison red
  misspell one site's tool (`list_circuit_capture`)               → test_next_tools_are_live red
  leave one string site unconverted (`miserver_models` load)      → the drive red (STRICT raises)
  pass relay `next_steps` through unfiltered                      → test_a_relay_drops_unregistered_next_steps red
"""
from __future__ import annotations

import ast
import pathlib
from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.support import result_language_drive as drive
from tests.support.miserver_fakes import approval_actions

MCP_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "mcp_server"
CODE = "a" * 64

#: 046 FTASKS 7.3: why these tests' fakes are off-contract (`mcp_drive.OffContractTransport`).
LEGACY_KEY_PLANTED = ("a typed route's answer planted with a backend's legacy key (`next`, `next_steps` or "
                      "`approval`) its model does not declare, to prove the result language converts, filters or drops it")


@dataclass
class Site:
    id: str
    category: str
    tool: str
    args: dict
    routes: list = field(default_factory=list)  # (method, path, status, body, headers)
    settings: dict = field(default_factory=dict)
    tools: list = field(default_factory=list)  # the tools `next_steps` must name, in order
    at: str = ""  # a nested key holding the legacy `next` (e.g. `positive_labels`)


def _miserver_error(code: str, message: str = "refused") -> dict:
    return {"error": {"code": code, "message": message}}


SITES = [
    Site("jobs.not_found", "jobs", "get_task_status", {"task_id": "t-404"},
         [("GET", r"/task-queue/lookup/t-404$", 200, {"found": []}, None)], tools=["list_active_jobs"]),
    Site("jobs.reader_list", "jobs,read", "get_task_status", {"task_id": "t-1"},
         [("GET", r"/task-queue/lookup/t-1$", 200,
           {"found": [{"table": "trainings", "row_id": "tr_1", "status": "running"}]}, None)],
         tools=["get_training"]),
    Site("jobs.prose", "jobs", "get_task_status", {"task_id": "t-2"},
         [("GET", r"/task-queue/lookup/t-2$", 200,
           {"found": [{"table": "some_table", "row_id": "x_1", "status": "running"}]}, None)],
         tools=["list_active_jobs"]),
    Site("jobs.queue_not_found", "jobs", "get_task_status", {"task_queue_id": "tq_9"},
         [("GET", r"/task-queue/tq_9$", 404, {"detail": "not found"}, None),
          ("GET", r"/task-queue/lookup/tq_9$", 200, {"found": []}, None)], tools=["list_active_jobs"]),
    Site("circuits.capture_estimate", "circuits", "start_circuit_capture",
         {"dataset_id": "ds_1", "layers": [{"layer": 3, "sae_id": "sae_1"}]},
         # the route's own 202 (`CaptureQueued`, 046 P7): an estimate is a queued run with confirmed false
         [("POST", r"/circuit-capture$", 202, {"id": "cc_1", "task_id": "t_1", "status": "queued", "confirmed": False},
           None)],
         tools=["list_circuit_captures", "confirm_circuit_capture"]),
    Site("miserver_models.load_started", "miserver_models", "miserver_load_model", {"model_id": 3},
         [("GET", r"/api/approvals/actions$", 200, approval_actions(["model_load"]), None),
          ("POST", r"/api/models/3/load$", 202, {"id": 3}, None)], tools=["miserver_list_models"]),
    Site("miserver_approvals.moved", "miserver_models", "miserver_load_model",
         {"model_id": 3, "approval_request_id": "old"}, tools=["miserver_get_approval_status"]),
    Site("miserver_approvals.pending", "miserver_models", "miserver_get_approval_status", {"approval_id": "a1"},
         [("GET", r"/api/approvals/a1$", 200, {"approval_id": "a1", "status": "pending"}, None)],
         tools=["miserver_get_approval_status"]),
    Site("miserver_approvals.executed", "miserver_models", "miserver_get_approval_status", {"approval_id": "a2"},
         [("GET", r"/api/approvals/a2$", 200, {"approval_id": "a2", "status": "executed"}, None)], tools=[]),
    Site("miserver_approvals.not_found", "miserver_models", "miserver_get_approval_status", {"approval_id": "a3"},
         [("GET", r"/api/approvals/a3$", 404, {"detail": "gone"}, None)], tools=[]),
    Site("approvals.pending", "read", "get_approval_status", {"approval_request_id": "r1"},
         [("GET", r"/mcp/approvals/r1$", 200, {"id": "r1", "tool_name": "evaluate_probe_monitor",
                                               "status": "pending"}, None)], tools=["get_approval_status"]),
    Site("approvals.approved", "read,probe_monitors", "get_approval_status", {"approval_request_id": "r2"},
         [("GET", r"/mcp/approvals/r2$", 200, {"id": "r2", "tool_name": "evaluate_probe_monitor",
                                               "status": "approved"}, None)], tools=["evaluate_probe_monitor"],
         settings={"probe_gpu_approval": True}),
    # OR1-045-1: the original tool's category is disabled here, so the status tool names no call (never a tool
    # this server does not register).
    Site("approvals.approved_elsewhere", "read", "get_approval_status", {"approval_request_id": "r2"},
         [("GET", r"/mcp/approvals/r2$", 200, {"id": "r2", "tool_name": "evaluate_probe_monitor",
                                               "status": "approved"}, None)], tools=[]),
    Site("approvals.code_rule_approved", "read,probe_monitors", "get_approval_status", {"approval_request_id": "c2"},
         [("GET", r"/mcp/approvals/c2$", 200, {"id": "c2", "tool_name": "code_rule_approve",
                                               "status": "approved"}, None)], tools=[]),
    Site("approvals.denied", "read", "get_approval_status", {"approval_request_id": "r3"},
         [("GET", r"/mcp/approvals/r3$", 200, {"id": "r3", "tool_name": "evaluate_probe_monitor",
                                               "status": "denied", "reason": "no"}, None)], tools=[]),
    Site("approvals.not_found", "read", "get_approval_status", {"approval_request_id": "r4"},
         [("GET", r"/mcp/approvals/r4$", 404, {"detail": "no"}, None)], tools=[]),
    Site("miserver_send.load_refusal", "miserver_inference,miserver_models", "miserver_score_completion",
         {"model": "m", "prompt": ["hi"]},
         [("POST", r"/v1/completions$", 409, _miserver_error("model_not_resident", "not loaded"), None)],
         tools=["miserver_load_model"], settings={}),
    Site("miserver_send.precision", "miserver_inference", "miserver_score_completion",
         {"model": "m", "prompt": ["hi"]},
         [("POST", r"/v1/completions$", 409, _miserver_error("scoring_precision_unavailable"), None)],
         tools=["miserver_score_completion"]),
    Site("miserver_send.name", "miserver_inference,miserver_models", "miserver_score_completion", {"model": "m", "prompt": ["hi"]},
         [("POST", r"/v1/completions$", 400, _miserver_error("model_not_generative"), None)],
         tools=["miserver_list_models"]),
    Site("miserver_send.503", "miserver_inference", "miserver_score_completion", {"model": "m", "prompt": ["hi"]},
         [("POST", r"/v1/completions$", 503, _miserver_error("busy"), None)], tools=[]),
    Site("probe_monitors.positive_labels", "probe_monitors", "register_probe_dataset",
         {"name": "v", "dataset_id": "d", "input_column": "text", "label_column": "label",
          "label_mapping": {"1": "positive", "0": "negative"}, "role": "train", "behaviour": "pii"},
         [("POST", r"/probe-monitors/datasets$", 200, {"id": "pd_1", "positive_labels_confirmed": False}, None)],
         tools=[], at="positive_labels"),
    Site("rest.pass_through", "read", "get_feature", {"feature_id": "f_1"},
         [("GET", r"/features/f_1$", 200, {"id": "f_1", "next": [
             {"tool": "get_feature_examples", "arguments": {"feature_id": "f_1"}, "why": "see where it fires"},
             {"tool": "build_probe_definition", "arguments": {"probe_id": "pm_1"}, "why": "not here",
              "needs": ["acknowledge_below_rung2_reason"]}]}, None)],
         tools=["get_feature_examples"]),
]


def _drive(monkeypatch, site: Site):
    from src.mcp_server import result_language

    monkeypatch.setattr(result_language, "STRICT", True)
    with drive.routed(monkeypatch, off_contract=LEGACY_KEY_PLANTED) as router:
        for method, path, status, body, headers in site.routes:
            router.on(method, path, status=status, body=body, headers=headers)
        mcp = drive.build(site.category, **site.settings)
        result = drive.call(mcp, site.tool, site.args)
        old = drive.unwrapped(mcp, site.tool, site.args)
    return result, old


@pytest.mark.parametrize("site", SITES, ids=[s.id for s in SITES])
def test_every_next_site_speaks_the_shape(monkeypatch, site):
    result, old = _drive(monkeypatch, site)
    assert not result.is_error, result.content
    answer = result.structured_content
    holder, before = (answer[site.at], old[site.at]) if site.at else (answer, old)
    assert "next" in holder, "the site's legacy next is gone"
    assert holder["next"] == before["next"], "the legacy next changed"
    steps = holder["next_steps"]
    check = drive.validator("next_step")
    for step in steps:
        errors = sorted(e.message for e in check.iter_errors(step))
        assert not errors, (step, errors)
    assert len(steps) <= 5
    assert [s["tool"] for s in steps] == site.tools


def _site_literals() -> set[str]:
    """Every tool name a `next_step(...)` call in mcp_server names as a literal (first positional argument)."""
    names: set[str] = set()
    for path in sorted(MCP_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "next_step"
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                names.add(node.args[0].value)
    return names


def test_next_tools_are_live():
    from src.mcp_server.miserver_send import NAME_REFUSALS
    from src.mcp_server.tools.jobs import TASK_ROW_READERS
    from tests.support.mcp_drive import build_every_category

    live = {t.name for m in build_every_category().values() for t in m._tool_manager.list_tools()}  # noqa: SLF001
    named = _site_literals()
    assert len(named) >= 10, sorted(named)  # the scan sees the sites
    named |= set(NAME_REFUSALS.values()) | {tool for tool, _ in TASK_ROW_READERS.values()}
    named |= {t for s in SITES for t in s.tools}
    assert not named - live, sorted(named - live)


def test_a_relay_drops_unregistered_next_steps(monkeypatch):
    steps = [{"tool": "dataworks_save_recipe", "args": {}, "why": "miDataworks' own tool, not relayed here"},
             {"tool": "dataworks_job_status", "args": {"job_id": "j1"}, "why": "poll the job"}]
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/v1/jobs/j1$", body={"id": "j1", "status": "running", "next_steps": steps})
        mcp = drive.build("dataworks")
        result = drive.call(mcp, "dataworks_job_status", {"job_id": "j1"})
    assert not result.is_error, result.content
    assert result.structured_content["next_steps"] == [steps[1]]


def test_a_native_answer_with_its_own_next_steps_is_left_alone():
    from src.mcp_server.result_language import add_next_steps

    value: dict[str, Any] = {"next": "prose", "next_steps": [{"tool": "nowhere", "args": {}, "why": "w"}]}
    assert add_next_steps(value, "get_feature", frozenset({"get_feature"})) is value


def test_the_envelope_never_declares_next():
    from tests.support.mcp_drive import build_every_category, wire_tools

    for mcp in build_every_category().values():
        for tool in wire_tools(mcp):
            props = (tool.output_schema or {}).get("properties") or {}
            assert "next" not in props, tool.name
            assert {"approval", "next_steps"} <= set(props), tool.name


def test_a_legacy_list_converts_one_item_each_and_keeps_needs_only_in_the_legacy_list():
    from src.mcp_server.result_language import add_next_steps

    legacy = [{"tool": "build_probe_definition", "arguments": {"probe_id": "pm_1"}, "why": "build it",
               "needs": ["acknowledge_below_rung2_reason"]}]
    out = add_next_steps({"next": legacy}, "get_probe_family", frozenset({"build_probe_definition"}))
    assert out["next"] is legacy and out["next"][0]["needs"] == ["acknowledge_below_rung2_reason"]
    assert out["next_steps"] == [{"tool": "build_probe_definition", "args": {"probe_id": "pm_1"}, "why": "build it"}]


# ── R1 review fixes (OR1-045-2 M23, M5; OR1-045-5), 2026-11-16 ──────────────────────────────────────────────────


@pytest.mark.parametrize("status", ["pending", "executed"])
def test_a_miserver_status_row_keeps_its_own_next_steps(monkeypatch, status):
    """A30 / OVN-b-9 (OR1-045-2 M23): a miServer approval row that already carries `next_steps` keeps them; the
    status tool's own poll (pending) or `[]` (decided) never overwrites a source that speaks the contract.
    Control: always set the tool's own list in `miserver_get_approval_status` → red."""
    own = [{"tool": "miserver_list_models", "args": {}, "why": "miServer's own words: see what it loaded"}]
    with drive.routed(monkeypatch, off_contract="the source answers its own `next_steps` or `approval`, which miServer's contract does not declare: the result language must keep, filter or repair them") as router:
        router.on("GET", r"/api/approvals/a9$", body={"approval_id": "a9", "action": "model_load", "status": status,
                                                       "next_steps": own})
        mcp = drive.build("miserver_models")
        result = drive.call(mcp, "miserver_get_approval_status", {"approval_id": "a9"})
    assert not result.is_error, result.content
    assert result.structured_content["next_steps"] == own


def test_a_long_legacy_list_is_cut_to_the_contract_limit(monkeypatch):
    """The contract's `next_steps` has `maxItems: 5`, and the SDK client raises on the whole answer past it
    (OR1-045-2 M5): a backend `next` of seven items converts to its first five, the legacy list untouched.
    Control: drop the `[:NEXT_STEPS_MAX]` on the legacy conversion in `add_next_steps` → red."""
    legacy = [{"tool": "get_feature_examples", "arguments": {"feature_id": f"f_{i}"}, "why": f"example {i}"}
              for i in range(7)]
    with drive.routed(monkeypatch, off_contract=LEGACY_KEY_PLANTED) as router:
        router.on("GET", r"/features/f_1$", body={"id": "f_1", "next": legacy})
        mcp = drive.build("read")
        result = drive.call(mcp, "get_feature", {"feature_id": "f_1"})
    assert not result.is_error, result.content
    answer = result.structured_content
    assert answer["next"] == legacy
    assert [s["args"]["feature_id"] for s in answer["next_steps"]] == [f"f_{i}" for i in range(5)]


@pytest.mark.parametrize(("bad", "kept"), [
    ("list_circuits", None),
    ({"tool": "list_circuits"}, None),
    ([{"tool": "get_feature_examples", "args": {}, "why": "ok"}, {"tool": "x"}, "y"],
     [{"tool": "get_feature_examples", "args": {}, "why": "ok"}]),
    ([{"tool": "get_feature_examples", "args": {}, "why": "w"}] * 6, [{"tool": "get_feature_examples", "args": {},
                                                                        "why": "w"}] * 5),
])
def test_a_non_conforming_backend_next_steps_costs_only_that_field(monkeypatch, bad, kept):
    """OR1-045-5: a backend's top-level `next_steps` in another shape is left out (a list keeps its conforming items,
    at most five); the rest of the answer arrives. Control: skip `conform` in `result_language.speak` → red."""
    with drive.routed(monkeypatch, off_contract=LEGACY_KEY_PLANTED) as router:
        router.on("GET", r"/features/f_1$", body={"id": "f_1", "label": "x", "next_steps": bad})
        mcp = drive.build("read")
        result = drive.call(mcp, "get_feature", {"feature_id": "f_1"})
    assert not result.is_error, result.content
    answer = result.structured_content
    assert (answer["id"], answer["label"]) == ("f_1", "x")
    assert answer.get("next_steps") == kept


# ── R2 review fixes (OR2-045-2 F2; OR2-045-3), 2026-11-16 ────────────────────────────────────────────────────────


def test_a_null_legacy_next_reads_as_no_next_steps(monkeypatch):
    """F2 (review-045ab-r2 candidate): a legacy `next: null` becomes `next_steps: []` (add_next_steps' docstring),
    never no field. Control: answer a null legacy `next` with no `next_steps` → red."""
    with drive.routed(monkeypatch, off_contract=LEGACY_KEY_PLANTED) as router:
        router.on("GET", r"/features/f_1$", body={"id": "f_1", "next": None})
        mcp = drive.build("read")
        result = drive.call(mcp, "get_feature", {"feature_id": "f_1"})
    assert not result.is_error, result.content
    assert result.structured_content["next_steps"] == []
    assert result.structured_content["next"] is None


@pytest.mark.parametrize("status", ["pending", "executing"])
def test_a_bad_source_step_never_costs_the_poll(monkeypatch, status):
    """OR2-045-3: a miServer row whose own poll step names an argument the tool does not declare loses that step,
    never the poll: the status tool's own poll comes back, and the row's good steps stay (A30).
    Control: drop `_keep_the_poll` from `add_next_steps` → red."""
    bad = {"tool": "miserver_get_approval_status", "args": {"id": "a9"}, "why": "poll (wrong argument name)"}
    good = {"tool": "miserver_list_models", "args": {}, "why": "see what is loaded"}
    with drive.routed(monkeypatch, off_contract="the source answers its own `next_steps` or `approval`, which miServer's contract does not declare: the result language must keep, filter or repair them") as router:
        router.on("GET", r"/api/approvals/a9$", body={"approval_id": "a9", "action": "model_load", "status": status,
                                                       "next_steps": [bad, good]})
        mcp = drive.build("miserver_models")
        result = drive.call(mcp, "miserver_get_approval_status", {"approval_id": "a9"})
    assert not result.is_error, result.content
    steps = result.structured_content["next_steps"]
    assert [(s["tool"], s["args"]) for s in steps] == [("miserver_get_approval_status", {"approval_id": "a9"}),
                                                       ("miserver_list_models", {})]


def test_a_decided_status_gets_no_poll_back(monkeypatch):
    """OR2-045-3's limit: a decided request has nothing to poll, so a dropped step is not replaced."""
    bad = {"tool": "miserver_get_approval_status", "args": {"id": "a9"}, "why": "poll (wrong argument name)"}
    with drive.routed(monkeypatch, off_contract="the source answers its own `next_steps` or `approval`, which miServer's contract does not declare: the result language must keep, filter or repair them") as router:
        router.on("GET", r"/api/approvals/a9$", body={"approval_id": "a9", "status": "executed", "next_steps": [bad]})
        mcp = drive.build("miserver_models")
        result = drive.call(mcp, "miserver_get_approval_status", {"approval_id": "a9"})
    assert result.structured_content["next_steps"] == []
