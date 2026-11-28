"""045 FR-7: the shared `approval` object, on every gated tool's pending or decided answer and on the three status
tools, from the vendored `mistudio.result-language/v1` (CHAIN_PLAN A6, A30).

Each paradigm is driven through the SDK's in-memory client against scripted HTTP answers (`result_language_drive`),
so the client also validates the answer against the tool's advertised output schema (the contract's `approval`,
inlined). Every test asserts (i) the answer before phase B is a subset of the new one, key for key and value for
value; (ii) `approval` validates against the vendored `$defs` (draft 2020-12); (iii) for `call_again_with`, re-calling
`call_again_with.tool` with `call_again_with.args` reaches the consume route ONCE with the payload the first call
filed, and the job route once.

Controls (0xcc/reviews/045_implementation_controls_2026-11-16.md, FTASKS 8.4):
  drop `speak_result_language(mcp)` in build_server        → every paradigm and status-tool test red
  `start_probe_training_run` row `approval=None`            → test_the_gated_set_is_derived red
  map `consumed` → `approved`                               → test_status_map red
  delete `dataworks_delete_version`'s approval              → test_the_gated_set_is_derived red
  always rebuild `approval` on a relay                      → test_a_relay_keeps_the_source_approval red
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from tests.support import result_language_drive as drive
from tests.support.miserver_fakes import approval_actions

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "mcp_server" / "tools"

#: 046 FTASKS 7.3: why these tests' fakes are off-contract (`mcp_drive.OffContractTransport`).
LEGACY_KEY_PLANTED = ("a typed route's answer planted with a backend's legacy key (`next`, `next_steps` or "
                      "`approval`) its model does not declare, to prove the result language converts, filters or drops it")


def _check(mcp, name: str, args: dict, result) -> dict:
    """The structured answer, after asserting it is not an error, keeps the pre-045 answer and carries a valid
    `approval`."""
    assert not result.is_error, result.content
    answer = result.structured_content
    approval = answer["approval"]
    errors = sorted(e.message for e in drive.validator("approval").iter_errors(approval))
    assert not errors, errors
    return answer


def _keeps_old(old: dict, new: dict) -> None:
    for key, value in old.items():
        assert key in new and new[key] == value, key
    assert set(new) - set(old) <= {"approval", "next_steps"}, set(new) - set(old)


# ── the five paradigms ───────────────────────────────────────────────────────────────────────────────────────────


def test_native_recall_files_once_and_the_recall_consumes_once(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("POST", r"/mcp/approvals$", body={"id": "apr_eval_1"})
        router.on("POST", r"/mcp/approvals/apr_eval_1/consume$", body={"status": "consumed"})
        router.on("POST", r"/probe-monitors/probes/pm_1/evaluate$", body={"task_id": "t-1"})
        mcp = drive.build("probe_monitors", probe_gpu_approval=True)
        args = {"probe_id": "pm_1", "dataset_ids": ["pd_a", "pd_b"]}

        first = drive.call(mcp, "evaluate_probe_monitor", args)
        answer = _check(mcp, "evaluate_probe_monitor", args, first)
        filed = router.body(router.calls("POST", r"/mcp/approvals$")[0])
        old = drive.unwrapped(mcp, "evaluate_probe_monitor", args)
        _keeps_old(old | {"approval_request_id": "apr_eval_1"}, answer)
        approval = answer["approval"]
        assert approval["id"] == "apr_eval_1" and approval["status"] == "pending"
        assert approval["decide_in"] == "miworkbench" and approval["then"] == "call_again_with"
        assert approval["call_again_with"]["tool"] == "evaluate_probe_monitor"
        assert approval["call_again_with"]["args"]["approval_request_id"] == "apr_eval_1"

        router.requests.clear()
        again = approval["call_again_with"]
        second = drive.call(mcp, again["tool"], again["args"])
        assert not second.is_error, second.content
        consumes = router.calls("POST", r"/mcp/approvals/apr_eval_1/consume$")
        assert len(consumes) == 1, len(consumes)
        assert router.body(consumes[0]) == filed, (router.body(consumes[0]), filed)
        assert len(router.calls("POST", r"/probe-monitors/probes/pm_1/evaluate$")) == 1
        assert router.calls("POST", r"/mcp/approvals$") == [], "the re-call filed a second request"
        assert "approval" not in second.structured_content


def test_native_steering_files_a_request(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("POST", r"/mcp/approvals$", body={"id": "apr_steer_1"})
        mcp = drive.build("steering", steering_approval=True)
        tool = mcp._tool_manager.get_tool("steer_compare")  # noqa: SLF001
        from tests.support.mcp_drive import _sample

        schema = tool.parameters
        args = {k: _sample(schema["properties"][k], schema.get("$defs") or {}) for k in schema.get("required", [])}
        result = drive.call(mcp, "steer_compare", args)
        answer = _check(mcp, "steer_compare", args, result)
        approval = answer["approval"]
        assert (approval["id"], approval["status"], approval["decide_in"], approval["then"]) == (
            "apr_steer_1", "pending", "miworkbench", "call_again_with")
        # An agent is never told to send an argument the tool refuses: the re-call object appears exactly when the
        # tool declares approval_request_id (045-A0 21.6 adds it to the steering tools).
        assert ("call_again_with" in approval) == ("approval_request_id" in schema["properties"])
        assert answer["approval_request_id"] == "apr_steer_1" and answer["status"] == "pending_approval"


def test_every_recall_row_but_the_code_rule_takes_the_id():
    """Every `call_again_with` row but the code rule's declares `approval_request_id` (A0's 21.6 gave it to `steer_*`;
    the strict xfail that held this until B sat on A0 was removed at the rebase, 2026-11-16)."""
    from src.mcp_server.tool_classes import TOOL_CLASSES
    from tests.support.mcp_drive import build_every_category

    tools = {t.name: t for m in build_every_category().values() for t in m._tool_manager.list_tools()}  # noqa: SLF001
    missing = sorted(n for n, r in TOOL_CLASSES.items() if r.approval == "call_again_with"
                     and n != "request_code_rule_approval"
                     and "approval_request_id" not in (tools[n].parameters.get("properties") or {}))
    assert not missing, missing


def test_miserver_relay(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/approvals/actions$", body=approval_actions(["model_load", "model_unload", "model_delete"]))
        router.on("POST", r"/api/models/7/load$", status=202, headers={"X-miLLM-Approval": "pending"},
                  body={"approval_id": "apr_m1", "status": "pending", "action": "model_load",
                        "summary": {"Model": "7"}, "expires_at": "2026-11-17T00:00:00Z"})
        mcp = drive.build("miserver_models")
        args = {"model_id": 7}
        answer = _check(mcp, "miserver_load_model", args, drive.call(mcp, "miserver_load_model", args))
        _keeps_old(drive.unwrapped(mcp, "miserver_load_model", args), answer)
        approval = answer["approval"]
        assert approval == {
            "id": "apr_m1", "status": "pending", "decide_in": "miserver", "then": "server_runs",
            "summary": approval["summary"], "action": "model_load", "expires_at": "2026-11-17T00:00:00Z"}
        assert "miserver_get_approval_status" in approval["summary"]


def test_midataworks_relay(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("DELETE", r"/api/v1/versions/v_1$", body={"approval_id": "apr_d1", "status": "pending",
                                                              "action": "version_delete"})
        mcp = drive.build("dataworks")
        args = {"version_id": "v_1", "reason": "superseded"}
        answer = _check(mcp, "dataworks_delete_version", args, drive.call(mcp, "dataworks_delete_version", args))
        _keeps_old(drive.unwrapped(mcp, "dataworks_delete_version", args), answer)
        approval = answer["approval"]
        assert (approval["id"], approval["status"], approval["decide_in"], approval["then"]) == (
            "apr_d1", "pending", "midataworks", "server_runs")
        assert "call_again_with" not in approval


def test_code_rule_approval_has_no_recall(monkeypatch):
    code = "a" * 64
    with drive.routed(monkeypatch) as router:
        router.on("POST", rf"/code-rules/{code}/approval-requests$",
                  body={"approval_request_id": "apr_cr1", "status": "pending_approval"})
        mcp = drive.build("probe_monitors")
        args = {"code_sha256": code, "cases": [{"id": "c1"}]}
        answer = _check(mcp, "request_code_rule_approval", args, drive.call(mcp, "request_code_rule_approval", args))
        _keeps_old(drive.unwrapped(mcp, "request_code_rule_approval", args), answer)
        approval = answer["approval"]
        assert (approval["id"], approval["status"], approval["decide_in"], approval["then"]) == (
            "apr_cr1", "pending", "miworkbench", "call_again_with")
        assert "call_again_with" not in approval  # C3: the operator approves the hash; nothing is re-called with it


# ── the three status tools ───────────────────────────────────────────────────────────────────────────────────────


def _native_row(status: str, **extra) -> dict:
    return {"id": "apr_9", "tool_name": "start_probe_training_run", "status": status, "reason": None,
            "created_at": "2026-11-16T00:00:00Z", "expires_at": "2026-11-17T00:00:00Z", **extra}


def test_get_approval_status(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/mcp/approvals/apr_9$", body=_native_row("consumed"))
        mcp = drive.build("read")
        args = {"approval_request_id": "apr_9"}
        answer = _check(mcp, "get_approval_status", args, drive.call(mcp, "get_approval_status", args))
        _keeps_old(drive.unwrapped(mcp, "get_approval_status", args), answer)
        approval = answer["approval"]
        assert (approval["id"], approval["status"], approval["decide_in"], approval["then"]) == (
            "apr_9", "executed", "miworkbench", "call_again_with")
        assert approval["action"] == "start_probe_training_run"
        assert "call_again_with" not in approval  # T45-D5: a status read does not know the original arguments


def test_miserver_get_approval_status(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/approvals/apr_m1$", body={"approval_id": "apr_m1", "action": "model_load",
                                                         "status": "executing"})
        mcp = drive.build("miserver_models")
        args = {"approval_id": "apr_m1"}
        answer = _check(mcp, "miserver_get_approval_status", args,
                        drive.call(mcp, "miserver_get_approval_status", args))
        _keeps_old(drive.unwrapped(mcp, "miserver_get_approval_status", args), answer)
        approval = answer["approval"]
        assert (approval["id"], approval["status"], approval["decide_in"], approval["then"]) == (
            "apr_m1", "executing", "miserver", "server_runs")


def test_dataworks_get_approval_status(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/v1/approvals/apr_d1$", body={"id": "apr_d1", "action": "version_delete",
                                                            "summary": "Delete v_1", "status": "rejected",
                                                            "reason": "still in use"})
        mcp = drive.build("dataworks")
        args = {"approval_id": "apr_d1"}
        answer = _check(mcp, "dataworks_get_approval_status", args,
                        drive.call(mcp, "dataworks_get_approval_status", args))
        _keeps_old(drive.unwrapped(mcp, "dataworks_get_approval_status", args), answer)
        approval = answer["approval"]
        assert (approval["id"], approval["status"], approval["decide_in"], approval["then"]) == (
            "apr_d1", "rejected", "midataworks", "server_runs")
        assert approval["reason"] == "still in use"


# ── the map, the derived gated set, and relays ───────────────────────────────────────────────────────────────────

#: FPRD FR-7's mapping (A6 amended 2026-11-15 adds executing and failed).
EXPECTED_STATUS_MAP = {
    "pending_approval": "pending", "pending": "pending", "approved": "approved", "denied": "rejected",
    "rejected": "rejected", "consumed": "executed", "executed": "executed", "executing": "executing",
    "failed": "failed", "expired": "expired",
}


def test_status_map():
    from src.mcp_server.result_language import STATUS_MAP

    assert STATUS_MAP == EXPECTED_STATUS_MAP
    enum = set(drive.contract()["$defs"]["approval_status"]["enum"])
    assert set(STATUS_MAP.values()) <= enum


@pytest.mark.parametrize("native", ["pending", "approved", "denied", "expired", "consumed", "not_found_status"])
def test_each_native_status_reads_through_the_status_tool(monkeypatch, native):
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/mcp/approvals/apr_9$", body=_native_row(native))
        mcp = drive.build("read")
        result = drive.call(mcp, "get_approval_status", {"approval_request_id": "apr_9"})
        answer = result.structured_content
        if native not in EXPECTED_STATUS_MAP:
            assert "approval" not in answer
            return
        assert answer["approval"]["status"] == EXPECTED_STATUS_MAP[native]
        assert answer["status"] == native  # the legacy field keeps its own word


@pytest.mark.parametrize("status", ["not_found", "mismatch"])
def test_a_refused_recall_is_not_an_approval(monkeypatch, status):
    code = {"not_found": 404, "mismatch": 422}[status]
    with drive.routed(monkeypatch) as router:
        router.on("POST", r"/mcp/approvals/apr_x/consume$", status=code, body={"detail": {"code": status}})
        mcp = drive.build("probe_monitors", probe_gpu_approval=True)
        result = drive.call(mcp, "evaluate_probe_monitor",
                            {"probe_id": "pm_1", "dataset_ids": ["pd_a"], "approval_request_id": "apr_x"})
        assert not result.is_error, result.content
        assert result.structured_content["status"] == status and "approval" not in result.structured_content


def _approval_filers(trees) -> set[str]:
    """Every function (a tool body's helper or a tool itself) that files a miWorkbench approval: its body POSTs the
    literal `/mcp/approvals`. Derived, so a new filer (044's `wsp._gate`, B4 port 2026-11-24) cannot be missed by a
    hand-kept helper list."""
    found: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "post" and call.args
                        and isinstance(call.args[0], ast.Constant) and call.args[0].value == "/mcp/approvals"):
                    found.add(node.name)
    return found


def _gate_callers() -> set[str]:
    """Tools whose bodies call a gate helper, found by AST over every tool module: a miWorkbench filer (derived by
    `_approval_filers`: `_approval_gate`, steering's `_submit`, `wsp._gate`) or a relay's (`relayed`,
    `with_poll_hint`)."""
    trees = [ast.parse(path.read_text(encoding="utf-8")) for path in sorted(TOOLS_DIR.glob("*.py"))]
    filers = _approval_filers(trees)
    assert {"_approval_gate", "_submit", "_gate"} <= filers, sorted(filers)  # the scan sees the filers
    helpers = filers | {"relayed", "with_poll_hint"}
    found: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef) or not any(
                    "tool" in ast.unparse(d) for d in node.decorator_list):
                continue
            calls = {c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", "")
                     for c in ast.walk(node) if isinstance(c, ast.Call)}
            if calls & helpers:
                found.add(node.name)
    return found


def test_the_gated_set_is_derived():
    from src.mcp_server.tool_classes import TOOL_CLASSES

    rows = {name for name, row in TOOL_CLASSES.items() if row.approval}
    derived = _gate_callers() | {"request_code_rule_approval"}
    assert len(derived) >= 20, sorted(derived)  # the scan sees the gates (a blind scan would pass an empty set)
    assert rows == derived, {"row but no gate call": sorted(rows - derived),
                             "gate call but no row": sorted(derived - rows)}
    assert all(TOOL_CLASSES[n].decide_in for n in rows), "every gated row names where the operator decides"


def test_a_relay_keeps_the_source_approval(monkeypatch):
    source = {"id": "apr_d2", "status": "expired", "decide_in": "midataworks", "then": "server_runs",
              "summary": "miDataworks' own words for an unswept expired request", "reason": "ttl"}
    with drive.routed(monkeypatch) as router:
        router.on("DELETE", r"/api/v1/versions/v_2$", body={"approval_id": "apr_d2", "status": "pending",
                                                              "approval": source})
        mcp = drive.build("dataworks")
        result = drive.call(mcp, "dataworks_delete_version", {"version_id": "v_2", "reason": "old"})
        assert not result.is_error, result.content
        assert result.structured_content["approval"] == source
        assert result.structured_content["status"] == "pending"


def test_a_plain_answer_gains_no_approval(monkeypatch):
    with drive.routed(monkeypatch) as router:
        # the evaluation queued, the route's own 202 (`ProbeTaskQueued`, 046 P7)
        router.on("POST", r"/probe-monitors/probes/pm_1/evaluate$",
                  body={"probe_id": "pm_1", "task_id": "t-1", "status": "queued"})
        mcp = drive.build("probe_monitors", probe_gpu_approval=False)
        result = drive.call(mcp, "evaluate_probe_monitor", {"probe_id": "pm_1", "dataset_ids": ["pd_a"]})
        assert not result.is_error, result.content
        assert "approval" not in result.structured_content


# ── R1 review fixes (OR1-045-1, OR1-045-2 M2/M3, OR1-045-5), 2026-11-16 ───────────────────────────────────────────


def _callable_on(mcp, steps: list) -> None:
    """Every `next_steps` item names a tool this server registers and sends only arguments that tool declares."""
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}  # noqa: SLF001
    for step in steps:
        assert step["tool"] in tools, (step, "not registered on this server")
        declared = set((tools[step["tool"]].parameters or {}).get("properties") or {})
        assert set(step["args"]) <= declared, (step, sorted(declared))


@pytest.mark.parametrize("categories", ["read", "read,probe_monitors"])
@pytest.mark.parametrize("native", ["pending", "approved", "denied", "expired", "consumed"])
def test_a_code_rule_approval_reads_through_the_status_tool(monkeypatch, native, categories):
    """OR1-045-1: a code-rule request is stored as `code_rule_approve`, which is no tool. The status tool still says
    `approval` (the `request_code_rule_approval` row: `then: call_again_with`, no re-call object, the stored action
    kept as `action`), and its `next_steps` name only callable tools: none at all once approved (the operator approved
    a hash; the agent retries the call that was refused `code_rule_not_approved`).
    Control: drop the `code_rule_approve` mapping in `result_language.STORED_ACTIONS` → red."""
    row = {"id": "apr_c", "tool_name": "code_rule_approve", "status": native, "reason": None,
           "created_at": "2026-11-16T00:00:00Z", "expires_at": "2026-11-17T00:00:00Z"}
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/mcp/approvals/apr_c$", body=row)
        mcp = drive.build(categories)
        args = {"approval_request_id": "apr_c"}
        answer = _check(mcp, "get_approval_status", args, drive.call(mcp, "get_approval_status", args))
        _keeps_old({k: v for k, v in drive.unwrapped(mcp, "get_approval_status", args).items() if k != "next_steps"},
                   answer)
    approval = answer["approval"]
    assert (approval["id"], approval["status"], approval["decide_in"], approval["then"], approval["action"]) == (
        "apr_c", EXPECTED_STATUS_MAP[native], "miworkbench", "call_again_with", "code_rule_approve")
    assert "call_again_with" not in approval
    _callable_on(mcp, answer["next_steps"])
    if native == "approved":
        assert answer["next_steps"] == []
        assert "code_rule_not_approved" in approval["summary"] and "code_rule_not_approved" in answer["next"]
    if native == "pending":
        assert [s["tool"] for s in answer["next_steps"]] == ["get_approval_status"]


def _every_native_row() -> list[str]:
    from src.mcp_server.result_language import STORED_ACTIONS
    from src.mcp_server.tool_classes import TOOL_CLASSES

    native = sorted(n for n, r in TOOL_CLASSES.items() if r.approval == "call_again_with" and r.decide_in == "miworkbench")
    return native + sorted(STORED_ACTIONS)


@pytest.mark.parametrize("stored", _every_native_row())
def test_every_status_answer_names_only_callable_tools(monkeypatch, stored):
    """OR1-045-1: `get_approval_status` lives in `read`, which every deployment enables; on a server with only `read`
    an approved request of any native gated tool names no tool from a disabled category, and on a server with every
    category it names only tools that take the arguments it sends. Control: drop the status-tool filter in
    `result_language.add_next_steps` → red."""
    from src.mcp_server.tools import all_categories

    row = {"id": "apr_n", "tool_name": stored, "status": "approved"}
    with drive.routed(monkeypatch) as router:  # built inside: every client the servers make is routed
        router.on("GET", r"/mcp/approvals/apr_n$", body=row)
        everything = drive.build(",".join(all_categories()), probe_gpu_approval=True, steering_approval=True)
        for mcp in (drive.build("read"), everything):
            result = drive.call(mcp, "get_approval_status", {"approval_request_id": "apr_n"})
            assert not result.is_error, result.content
            _callable_on(mcp, result.structured_content["next_steps"])


@pytest.mark.parametrize(("categories", "expected"), [("read,steering", ["get_steering_result"]), ("read", [])])
def test_an_approved_steering_row_polls_its_task(monkeypatch, categories, expected):
    """OVN-b-9 (OR1-045-2 M2): an approved steering request the backend already submitted is polled by its task id
    with `get_steering_result`, where that tool is registered. Control: disable the `steering_task_id` branch in
    `tools/approvals.next_steps_for` → red."""
    row = {"id": "apr_s", "tool_name": "steer_compare", "status": "approved", "steering_task_id": "st-9"}
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/mcp/approvals/apr_s$", body=row)
        mcp = drive.build(categories)
        result = drive.call(mcp, "get_approval_status", {"approval_request_id": "apr_s"})
    assert not result.is_error, result.content
    steps = result.structured_content["next_steps"]
    assert [s["tool"] for s in steps] == expected
    if expected:
        assert steps[0]["args"] == {"task_id": "st-9"}


@pytest.mark.parametrize(("status", "recall"), [("pending", True), ("expired", False), ("consumed", False),
                                                ("denied", False)])
def test_call_again_with_only_while_a_recall_can_run(monkeypatch, status, recall):
    """OVN-b-3 (OR1-045-2 M3): a refused re-call (the consume route answers 409 with the approval's status) offers
    `call_again_with` only while a re-call can still run it (pending, approved); a spent, rejected or expired one is
    never offered again (an agent would loop on a consume the gate refuses). Control: offer it in any status → red."""
    with drive.routed(monkeypatch) as router:
        router.on("POST", r"/mcp/approvals/apr_r/consume$", status=409,
                  body={"detail": {"status": status, "reason": "r"}})
        mcp = drive.build("probe_monitors", probe_gpu_approval=True)
        args = {"probe_id": "pm_1", "dataset_ids": ["pd_a"], "approval_request_id": "apr_r"}
        answer = _check(mcp, "evaluate_probe_monitor", args, drive.call(mcp, "evaluate_probe_monitor", args))
        assert router.calls("POST", r"/probe-monitors/probes/pm_1/evaluate$") == []
    approval = answer["approval"]
    assert approval["status"] == EXPECTED_STATUS_MAP[status]
    assert ("call_again_with" in approval) is recall
    if recall:
        again = approval["call_again_with"]
        assert again["tool"] == "evaluate_probe_monitor"
        assert {k: v for k, v in again["args"].items() if v is not None} == args  # defaulted optionals ride as None


@pytest.mark.parametrize("bad", [None, "pending", {"id": "apr_x"}, {"id": "apr_x", "status": "nope", "decide_in":
                                                                  "miworkbench", "then": "server_runs", "summary": "s"}])
def test_a_non_conforming_backend_approval_costs_only_that_field(monkeypatch, bad):
    """OR1-045-5: a backend answer whose top-level `approval` the advertised schema refuses (null, a string, an object
    of another shape) loses that one field; the rest of the answer arrives (the SDK client raised on the whole answer
    before). Control: skip `conform` in `result_language.speak` → red."""
    with drive.routed(monkeypatch, off_contract=LEGACY_KEY_PLANTED) as router:
        router.on("GET", r"/features/f_1$", body={"id": "f_1", "label": "x", "approval": bad})
        mcp = drive.build("read")
        result = drive.call(mcp, "get_feature", {"feature_id": "f_1"})
    assert not result.is_error, result.content
    # 046: the fake is completed with the route's always-sent fields (it is a typed tool); the backend's own fields
    # arrive, and only the refused `approval` is lost.
    answer = result.structured_content
    assert "approval" not in answer and (answer["id"], answer["label"]) == ("f_1", "x"), answer


def test_a_relay_with_a_non_conforming_approval_is_rebuilt_from_its_fields(monkeypatch):
    """OR1-045-5 on a relay: the source's unusable `approval: null` is left out, and the relay's own recognition
    (`approval_id` + `status`) builds the contract's object, as for a source that never sent one."""
    with drive.routed(monkeypatch) as router:
        router.on("DELETE", r"/api/v1/versions/v_3$", body={"approval_id": "apr_d3", "status": "pending",
                                                              "approval": None})
        mcp = drive.build("dataworks")
        args = {"version_id": "v_3", "reason": "old"}
        answer = _check(mcp, "dataworks_delete_version", args, drive.call(mcp, "dataworks_delete_version", args))
    assert (answer["approval"]["id"], answer["approval"]["decide_in"]) == ("apr_d3", "midataworks")
    assert answer["approval_id"] == "apr_d3" and answer["status"] == "pending"


def test_the_status_site_itself_names_no_call_for_an_approved_code_rule():
    """OR1-045-1 at its site, beside the wrapper's filter (which would also drop the unregistered `code_rule_approve`):
    `next_steps_for` itself answers `[]` for an approved code rule, so the site never relies on a filter to stop
    naming a stored action. Control: disable the stored-action branch in `tools/approvals.next_steps_for` → red."""
    from src.mcp_server.tools.approvals import describe, next_steps_for

    row = {"id": "apr_c", "tool_name": "code_rule_approve", "status": "approved"}
    assert next_steps_for(row) == []
    assert "code_rule_not_approved" in describe(row)["next"]
    assert next_steps_for({**row, "tool_name": "evaluate_probe_monitor"})[0]["tool"] == "evaluate_probe_monitor"


# ── R2 review fixes (OR2-045-1; OR2-045-2 F8), 2026-11-16 ────────────────────────────────────────────────────────

#: A valid contract approval plus one key this vendored copy does not know (a later minor version of the contract).
EXTRA = {"id": "apr_d", "status": "pending", "decide_in": "midataworks", "then": "server_runs", "summary": "Delete v",
         "created_at": "2026-11-16T00:00:00Z"}


@pytest.mark.parametrize("legacy_status", ["pending", "queued"])
def test_a_gated_relay_whose_approval_was_dropped_still_says_it_waits(monkeypatch, legacy_status):
    """OR2-045-1 (review-045ab-r2 candidate): whatever the legacy fields say, a gated relay answer whose source
    `approval` carries one unknown key still carries an `approval` (repaired, its own status kept); it never reads as
    a plain, finished result. Control: drop instead of repairing in `conform` → red (`queued`)."""
    with drive.routed(monkeypatch) as router:
        router.on("DELETE", r"/api/v1/versions/v$", status=202,
                  body={"approval_id": "apr_d", "status": legacy_status, "action": "version_delete", "approval": EXTRA})
        mcp = drive.build("dataworks")
        result = drive.call(mcp, "dataworks_delete_version", {"version_id": "v", "reason": "r"})
    assert not result.is_error, result.content
    approval = result.structured_content.get("approval")
    assert approval is not None, result.structured_content
    assert approval["status"] == "pending" and approval["decide_in"] == "midataworks"
    assert approval == {k: v for k, v in EXTRA.items() if k != "created_at"}  # the source's own words, cut to v1


def test_a_dropped_status_approval_keeps_the_sources_status(monkeypatch):
    """OR2-045-1 (review-045ab-r2 candidate): the source says `expired` (miDataworks 022-D9: past its expiry, the
    legacy column still says `pending`); a repair never turns it back into `pending`. Control: drop instead of
    repairing in `conform` → red."""
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/v1/approvals/apr_d$",
                  body={"id": "apr_d", "status": "pending", "action": "version_delete",
                        "approval": {**EXTRA, "status": "expired"}})
        mcp = drive.build("dataworks")
        result = drive.call(mcp, "dataworks_get_approval_status", {"approval_id": "apr_d"})
    assert not result.is_error, result.content
    assert result.structured_content["approval"]["status"] == "expired", result.structured_content["approval"]


@pytest.mark.parametrize("terminal", ["expired", "rejected", "executed", "failed"])
def test_an_unrepairable_source_approval_keeps_its_status_when_rebuilt(monkeypatch, terminal):
    """OR2-045-1: a source `approval` that cannot be repaired (no `summary`) is rebuilt from the answer's legacy
    fields, but with the SOURCE's contract status, never the lagging legacy `pending`. Control: ignore
    `source_status` in `add_approval` → red."""
    source = {"id": "apr_d", "status": terminal, "decide_in": "midataworks", "then": "server_runs"}
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/v1/approvals/apr_d$",
                  body={"id": "apr_d", "status": "pending", "action": "version_delete", "approval": source})
        mcp = drive.build("dataworks")
        result = drive.call(mcp, "dataworks_get_approval_status", {"approval_id": "apr_d"})
    assert not result.is_error, result.content
    approval = result.structured_content["approval"]
    assert approval["status"] == terminal and approval["id"] == "apr_d"
    assert result.structured_content["status"] == "pending"  # the legacy field keeps its own word


@pytest.mark.parametrize(("tool", "category", "method", "path", "args"), [
    ("dataworks_delete_version", "dataworks", "DELETE", r"/api/v1/versions/v$", {"version_id": "v", "reason": "r"}),
    ("dataworks_get_approval_status", "dataworks", "GET", r"/api/v1/approvals/apr_d$", {"approval_id": "apr_d"}),
])
def test_an_unreadable_approval_on_a_gated_answer_is_an_error(monkeypatch, tool, category, method, path, args):
    """OR2-045-1: a gated, relayed or status answer whose source `approval` can neither be repaired nor rebuilt (no
    recognisable legacy status) is an error that says nothing is confirmed done, never a plain result.
    Control: remove the refusal in `speak` → red."""
    with drive.routed(monkeypatch) as router:
        router.on(method, path, status=202, body={"approval_id": "apr_d", "status": "queued",
                                                  "approval": {"id": "apr_d", "status": "queued"}})
        mcp = drive.build(category)
        result = drive.call(mcp, tool, args)
    assert result.is_error, result.structured_content
    text = result.content[0].text
    assert text.startswith("approval_unreadable: "), text
    assert "NOT done" in text.split("\n")[1]


def test_a_plain_answer_with_an_unreadable_approval_only_loses_the_field(monkeypatch):
    """The refusal is for gated, relayed and status answers only: a plain read keeps OR1-045-5's behaviour."""
    with drive.routed(monkeypatch, off_contract=LEGACY_KEY_PLANTED) as router:
        router.on("GET", r"/features/f_1$", body={"id": "f_1", "approval": {"id": "x", "status": "queued"}})
        mcp = drive.build("read")
        result = drive.call(mcp, "get_feature", {"feature_id": "f_1"})
    assert not result.is_error, result.content
    answer = result.structured_content  # 046: completed with the route's always-sent fields (a typed tool)
    assert "approval" not in answer and answer["id"] == "f_1", answer


@pytest.mark.parametrize(("native", "has_reason"), [("pending", False), ("approved", False), ("consumed", False),
                                                    ("denied", True), ("expired", True)])
def test_approval_reason_only_when_rejected_failed_or_expired(monkeypatch, native, has_reason):
    """F8 (review-045ab-r2 candidate): the contract's `reason` is "why it was rejected, failed or expired"; a pending
    or spent row's own reason text is not put there. Control: copy `reason` in every status → red."""
    row = {"id": "apr_9", "tool_name": "start_probe_training_run", "status": native, "reason": "some text"}
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/mcp/approvals/apr_9$", body=row)
        mcp = drive.build("read")
        result = drive.call(mcp, "get_approval_status", {"approval_request_id": "apr_9"})
    assert not result.is_error, result.content
    assert ("reason" in result.structured_content["approval"]) is has_reason


# ── R3 review fixes (OR3-045b-1 – -4), 2026-11-16 ────────────────────────────────────────────────────────────────

#: Repair fails (no decide_in/then/summary) and no legacy status rebuilds it.
BAD = {"id": "apr_x", "status": "queued"}


@pytest.mark.parametrize(("label", "tool", "category", "method", "path", "args"), [
    ("gated native only", "recalibrate_probe_monitor", "probe_monitors", "POST",
     r"/probe-monitors/probes/pm_1/recalibrate$", {"probe_id": "pm_1", "target_fpr": 0.01}),
    ("relay only", "miserver_get_model", "miserver_models", "GET", r"/api/models/7$", {"model_id": 7}),
])
def test_each_refusal_branch_alone(monkeypatch, label, tool, category, method, path, args):
    """OR3-045b-1 (review-045-a0r2-br3 candidate): each branch of `speak`'s `approval_unreadable` refusal pinned ALONE
    (the R2 test drives only dataworks_* tools, which are relayed AND gated, or relayed AND status, at once).
    Controls: drop the gated-row branch → red (gated case); drop the relay branch → red (relay case)."""
    with drive.routed(monkeypatch, off_contract="the source answers its own `approval`, which miServer's contract does not declare: the result language must repair or drop it") as router:
        router.on(method, path, status=200, body={"id": "x", "approval": BAD})
        mcp = drive.build(category)
        result = drive.call(mcp, tool, args)
    assert result.is_error, (label, result.structured_content)
    assert result.content[0].text.startswith("approval_unreadable: "), (label, result.content[0].text[:200])


#: A source `approval` with a CONTRACT status but missing required fields: not foreign, so R3's first refusal does
#: not fire; `repair_approval` cannot complete it and nothing can be rebuilt (B4 port, 2026-11-24).
UNREBUILDABLE = {"id": "apr_y", "status": "pending"}


@pytest.mark.parametrize(("label", "tool", "category", "method", "path", "args"), [
    ("gated native", "recalibrate_probe_monitor", "probe_monitors", "POST",
     r"/probe-monitors/probes/pm_1/recalibrate$", {"probe_id": "pm_1", "target_fpr": 0.01}),
    ("relay", "miserver_get_model", "miserver_models", "GET", r"/api/models/7$", {"model_id": 7}),
])
def test_an_unrebuildable_approval_with_a_contract_status_is_an_error(monkeypatch, label, tool, category, method,
                                                                     path, args):
    """B4 port survivor R2-P3 (2026-11-24): every earlier refusal test sends an approval whose status the contract does
    not know (`queued`), which R3's FIRST refusal (`foreign`) catches, so removing the SECOND refusal in `speak` (a
    dropped approval with a contract status that nothing rebuilt) stayed green. Here the status is the contract's own
    `pending` and the object lacks `decide_in`, `then` and `summary`: the answer must still be `approval_unreadable`,
    never a plain result. Control: delete the second refusal → red."""
    with drive.routed(monkeypatch, off_contract="the source answers its own `approval`, which miServer's contract does not declare: the result language must repair or drop it") as router:
        router.on(method, path, status=200, body={"id": "x", "approval": UNREBUILDABLE})
        mcp = drive.build(category)
        result = drive.call(mcp, tool, args)
    assert result.is_error, (label, result.structured_content)
    assert result.content[0].text.startswith("approval_unreadable: "), (label, result.content[0].text[:200])
    assert "(approval apr_y)" in result.content[0].text, result.content[0].text[:400]


def test_a_status_outside_the_contract_is_refused_not_rebuilt(monkeypatch):
    """OR3-045b-2: a source status this vendored contract does not know (a later minor version's `withdrawn`) beside a
    lagging legacy `pending` is refused, never rebuilt as `pending`. Control: accept any source status → red."""
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/v1/approvals/apr_d$",
                  body={"id": "apr_d", "status": "pending", "action": "version_delete",
                        "approval": {**EXTRA, "status": "withdrawn"}})
        mcp = drive.build("dataworks")
        result = drive.call(mcp, "dataworks_get_approval_status", {"approval_id": "apr_d"})
    assert result.is_error, result.structured_content
    assert result.content[0].text.startswith("approval_unreadable: ")


def test_an_unreadable_approval_names_its_id_and_a_poll(monkeypatch):
    """OR3-045b-3: the error carries the approval id the answer carried, and line 3 polls the deciding app's status
    tool when it is registered here. Control: raise the error without its steps → red."""
    import json

    with drive.routed(monkeypatch) as router:
        router.on("DELETE", r"/api/v1/versions/v$", status=202,
                  body={"approval_id": "apr_d", "status": "queued", "approval": BAD})
        mcp = drive.build("dataworks")
        result = drive.call(mcp, "dataworks_delete_version", {"version_id": "v", "reason": "r"})
    assert result.is_error
    lines = result.content[0].text.split("\n")
    assert "apr_x" in "\n".join(lines[3:]), lines  # the source approval's own id
    assert json.loads(lines[2][len("next: "):]) == [
        {"tool": "dataworks_get_approval_status", "args": {"approval_id": "apr_x"},
         "why": "read the request where it is decided; nothing here confirms it waits, ran or was refused"}]


def test_an_unreadable_approval_on_a_status_tool_names_no_circular_poll(monkeypatch):
    """OR3-045b-3: on the status tool itself, polling it again would answer the same error: line 3 is `[]`."""
    import json

    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/v1/approvals/apr_d$", body={"id": "apr_d", "status": "queued", "approval": BAD})
        mcp = drive.build("dataworks")
        result = drive.call(mcp, "dataworks_get_approval_status", {"approval_id": "apr_d"})
    assert result.is_error
    assert json.loads(result.content[0].text.split("\n")[2][len("next: "):]) == []


@pytest.mark.parametrize("terminal", ["expired", "rejected", "executed", "failed"])
def test_a_decided_source_status_is_not_polled(monkeypatch, terminal):
    """OR3-045b-4: when the source's shared status (decided) wins over a lagging legacy `pending`, the status tool's
    poll step, derived from the legacy column, is taken out. Control: skip `_no_poll_once_decided` → red."""
    source = {"approval_id": "a9", "status": terminal, "decide_in": "miserver", "then": "server_runs"}  # no summary
    with drive.routed(monkeypatch, off_contract="the source answers its own `approval`, which miServer's contract does not declare: the result language must repair or drop it") as router:
        router.on("GET", r"/api/approvals/a9$", body={"approval_id": "a9", "action": "model_load",
                                                       "status": "pending", "approval": {**source, "id": "a9"}})
        mcp = drive.build("miserver_models")
        result = drive.call(mcp, "miserver_get_approval_status", {"approval_id": "a9"})
    assert not result.is_error, result.content
    answer = result.structured_content
    assert answer["approval"]["status"] == terminal
    assert answer["next_steps"] == [], answer["next_steps"]


def test_an_expired_queue_summary_names_no_action_as_a_tool(monkeypatch):
    """OR3-045b-4: a miDataworks status row's `action` is the app's action name, not a tool here: the expired summary
    says "the original tool". Control: name the action as the tool to call again → red."""
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/v1/approvals/apr_d$", body={"id": "apr_d", "status": "expired",
                                                            "action": "version_delete"})
        mcp = drive.build("dataworks")
        result = drive.call(mcp, "dataworks_get_approval_status", {"approval_id": "apr_d"})
    summary = result.structured_content["approval"]["summary"]
    assert "Call version_delete" not in summary and "the original tool" in summary, summary


# ── B4 port (2026-11-24): the tools main gained after 045-B was built ────────────────────────────────────────────


def _wsp_case(tool: str):
    if tool == "capture_wsp_heldout":
        return {"probe_id": "pm_w1"}, "/wsp/captures", {"probe_id": "pm_w1"}
    document = {"schema": "miforge.wsp-run/v1"}
    manifest = {"schema": "miforge.wsp-patch/v1"}
    args = {"run_document": document, "patch_manifest": manifest, "edit_npz_b64": "AAAA"}
    return args, "/wsp/candidates/import", dict(args)


@pytest.mark.parametrize("tool", ["capture_wsp_heldout", "import_wsp_candidate"])
def test_a_wsp_tool_files_once_and_the_recall_reaches_its_route_once(monkeypatch, tool):
    """044's two `wsp` tools (always gated, the ROUTE spends the approval) speak the same `approval` as every native
    gated tool: pending, decided in miWorkbench, `call_again_with` this tool and the id; the re-call reaches the route
    once with the id in the approval-request header and files nothing."""
    from src.core import approval_payload
    from src.mcp_server.tools import wsp

    args, route, body = _wsp_case(tool)
    filed = wsp.filed_payload(route, body)
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/wsp/captures/preflight$", body={"to_run": 1, "rows_to_run": 1, "sets": [
            {"dataset_id": "pd_a", "would_run": True}], "gpu": {"fits": True}})
        router.on("POST", r"/mcp/approvals$", body={"id": "apr_w1"})
        router.on("GET", r"/mcp/approvals/apr_w1$", body={"id": "apr_w1", "tool_name": tool, "payload": filed,
                                                          "status": "approved"})
        # The route's own 202 once the approval is spent (its contract's `WspCaptureStarted` or
        # `WspCandidateImporting`; 046 P7), completed beneath these fields.
        started = {"/wsp/captures": {"task_id": "t-w1", "started": ["cap_a"]},
                   "/wsp/candidates/import": {"task_id": "t-w1", "status": "importing"}}[route]
        router.on("POST", route + "$", status=202, body=started)
        mcp = drive.build("read,wsp")

        first = drive.call(mcp, tool, args)
        answer = _check(mcp, tool, args, first)
        assert router.body(router.calls("POST", r"/mcp/approvals$")[0])["payload"] == filed
        approval = answer["approval"]
        assert (approval["id"], approval["status"], approval["decide_in"], approval["then"]) == (
            "apr_w1", "pending", "miworkbench", "call_again_with")
        assert approval["call_again_with"]["tool"] == tool
        assert approval["call_again_with"]["args"]["approval_request_id"] == "apr_w1"
        assert answer["approval_request_id"] == "apr_w1" and answer["status"] == "pending_approval"
        assert router.calls("POST", route + "$") == [], "the first call reached the route"

        router.requests.clear()
        again = approval["call_again_with"]
        second = drive.call(mcp, again["tool"], again["args"])
        assert not second.is_error, second.content
        sent = router.calls("POST", route + "$")
        assert len(sent) == 1, len(sent)
        assert sent[0].headers.get(approval_payload.APPROVAL_REQUEST_HEADER) == "apr_w1"
        assert router.body(sent[0]) == body
        assert router.calls("POST", r"/mcp/approvals$") == [], "the re-call filed a second request"
        assert "approval" not in second.structured_content


def test_the_context_ceiling_relay_speaks_server_runs(monkeypatch):
    """miServer contract v1.30's `miserver_set_model_context_ceiling` (main, 2026-11-21) relays miServer's pending
    answer; the shared `approval` says miServer decides and runs it."""
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/approvals/actions$", body=approval_actions(["model_load", "model_context_write"]))
        router.on("PUT", r"/api/models/7/context-ceiling$", status=202, headers={"X-miLLM-Approval": "pending"},
                  body={"approval_id": "apr_c1", "status": "pending", "action": "model_context_write",
                        "summary": {"Context ceiling": "131072"}, "expires_at": "2026-11-25T00:00:00Z"})
        mcp = drive.build("miserver_models")
        args = {"model_id": 7, "context_ceiling": 131072}
        tool = "miserver_set_model_context_ceiling"
        answer = _check(mcp, tool, args, drive.call(mcp, tool, args))
        _keeps_old(drive.unwrapped(mcp, tool, args), answer)
        approval = answer["approval"]
        assert approval == {
            "id": "apr_c1", "status": "pending", "decide_in": "miserver", "then": "server_runs",
            "summary": approval["summary"], "action": "model_context_write", "expires_at": "2026-11-25T00:00:00Z"}
        assert "miserver_get_approval_status" in approval["summary"]
