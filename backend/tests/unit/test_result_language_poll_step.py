"""045 follow-up (2026-11-24): a gated tool's pending answer names the poll in `next_steps`.

Before this, a pending answer from a gated tool carried the contract's `approval` (045 FR-7) but no `next_steps` item
naming the status tool to poll, so an agent reading `next_steps` alone (the contract's key) was told nothing to call.
Now the deciding app's status tool comes FIRST, with the approval's id, in miDataworks 022's shape
(`{tool: dataworks_get_approval_status, args: {approval_id}, why}`): `get_approval_status {approval_request_id}` for
miWorkbench's own gates, `miserver_get_approval_status {approval_id}` and `dataworks_get_approval_status {approval_id}`
for the relays (`result_language.add_poll_step`).

The gated set is DERIVED, never a hand list: every tool of the live registry (one server per category, as
`howto._all_tools` builds it) whose `TOOL_CLASSES` row carries an `approval`, minus the three status tools. Each is
driven through the SDK's in-memory client (which validates the answer against the tool's advertised output schema) to
a pending answer, and the poll it names is then CALLED, as an agent would: it must reach the deciding app's status
route exactly once with that id (the junction, memory `mutation-survivors-cluster-on-callers`).

Controls (record `0xcc/reviews/045_followup_controls_2026-11-24.md`):
  drop `add_poll_step` from `speak`                        → every case of test_a_pending_answer_names_its_poll red
  poll in any status                                       → test_a_decided_answer_names_no_poll red
  skip the registered/argument check                       → test_no_poll_when_the_status_tool_is_not_here red
  skip the "a step already names it" check                 → test_a_source_poll_is_not_doubled red
  poll only for rows with an `approval`                    → test_an_ungated_relay_whose_source_waits_names_the_poll red
"""
from __future__ import annotations

import functools

import pytest

from tests.support import result_language_drive as drive
from tests.support.mcp_drive import _sample
from tests.support.miserver_fakes import approval_actions

ID = "apr_poll_1"

#: The deciding app → (its status tool, the argument that names the id, the status route that tool reads).
POLL = {
    "miworkbench": ("get_approval_status", "approval_request_id", rf"/mcp/approvals/{ID}$"),
    "miserver": ("miserver_get_approval_status", "approval_id", rf"^/api/approvals/{ID}$"),
    "midataworks": ("dataworks_get_approval_status", "approval_id", rf"/api/v1/approvals/{ID}$"),
}

#: Arguments a tool's sampled required arguments do not reach its gate with (every other gated tool files from them).
GATE_ARGS = {
    "recalibrate_probe_monitor": {"recut_windows": True},
    "request_code_rule_approval": {"code_sha256": "a" * 64, "cases": [{"id": "c1"}]},
    "miserver_delete_model": {"acknowledge_permanent": True},
    "miserver_set_model_context_ceiling": {"context_ceiling": 131072},
    "miserver_set_telemetry": {"values": {"TELEMETRY_SAMPLE_RATE": 0.5}},
    # FU-4 (2026-11-25): no argument is required (a candidate id OR the three documents); the documents need no miForge.
    "import_wsp_candidate": {"run_document": {"run_id": "wsp_1"}, "patch_manifest": {}, "edit_npz_b64": "QUJD"},
}


@functools.lru_cache(maxsize=1)
def _categories() -> dict[str, str]:
    """tool → its category, from the live registry."""
    from tests.support.mcp_drive import build_every_category

    return {t.name: category for category, mcp in build_every_category().items()
            for t in mcp._tool_manager.list_tools()}  # noqa: SLF001


def _gated() -> list[str]:
    from src.mcp_server.result_language import STATUS_TOOLS
    from src.mcp_server.tool_classes import TOOL_CLASSES

    return sorted(n for n in _categories() if TOOL_CLASSES[n].approval and n not in STATUS_TOOLS)


def test_the_gated_set_is_the_registrys():
    """Every gated row is served by the registry and every served gated tool has a case below (both ways)."""
    from src.mcp_server.result_language import STATUS_TOOLS
    from src.mcp_server.tool_classes import TOOL_CLASSES

    rows = {n for n, r in TOOL_CLASSES.items() if r.approval and n not in STATUS_TOOLS}
    assert set(_gated()) == rows, {"row not served": sorted(rows - set(_gated()))}
    assert len(rows) >= 20, sorted(rows)  # a blind derivation would pass an empty set
    assert {TOOL_CLASSES[n].decide_in for n in rows} == set(POLL)


def _args(mcp, name: str) -> dict:
    schema = mcp._tool_manager.get_tool(name).parameters  # noqa: SLF001
    sampled = {k: _sample(schema["properties"][k], schema.get("$defs") or {}) for k in schema.get("required", [])}
    return {**sampled, **GATE_ARGS.get(name, {})}


def _pending_routes(router, decide_in: str, name: str) -> None:
    """What each app answers when it files the request."""
    from src.mcp_server.tools.miserver_approvals import RELAYED_ACTIONS

    if decide_in == "miworkbench":
        router.on("POST", r"/mcp/approvals$", body={"id": ID})
        router.on("POST", r"/code-rules/a{64}/approval-requests$",
                  body={"approval_request_id": ID, "status": "pending_approval"})
        # the run's forecast, read before filing (`ProbeRunForecast`, 046 P7): none from this stand-in
        router.on("POST", r"/probe-monitors/runs/forecast$", body={"checked": False, "fits": None,
                                                                  "time_forecast": {"forecast": False,
                                                                                    "summary": "No time forecast."}})
        router.on("GET", r"/wsp/captures/preflight$",
                  body={"to_run": 1, "rows_to_run": 1, "sets": [{"dataset_id": "pd_a", "would_run": True}],
                        "gpu": {"fits": True}})
    elif decide_in == "miserver":
        router.on("GET", r"^/api/approvals/actions$", body=approval_actions(sorted(RELAYED_ACTIONS.values())))
        body = {"approval_id": ID, "status": "pending", "action": RELAYED_ACTIONS[name], "summary": {"Action": "s"},
                "expires_at": "2026-11-26T00:00:00Z"}
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            router.on(method, r"^/api/(?!v1/|approvals)", status=202, headers={"X-miLLM-Approval": "pending"},
                      body=body)
    else:
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            router.on(method, r"^/api/v1/(?!approvals)", status=202,
                      body={"approval_id": ID, "status": "pending", "action": "a"})
    poll_tool, poll_arg, poll_route = POLL[decide_in]
    # Each app's status row in ITS shape (046 WT3: miServer's is its contract's `ApprovalRead`).
    rows = {"miserver": {"approval_id": ID, "status": "pending", "action": "a"}}
    # miWorkbench's is its contract's `ApprovalResponse` (046 FTASKS 7.3; the router completes its other fields).
    router.on("GET", poll_route, body=rows.get(decide_in, {"id": ID, "status": "pending", "tool_name": name}))


def _server(name: str, decide_in: str):
    poll_tool = POLL[decide_in][0]
    categories = sorted({_categories()[name], _categories()[poll_tool]})
    return drive.build(",".join(categories), probe_gpu_approval=True, steering_approval=True)


@pytest.mark.parametrize("name", _gated())
def test_a_pending_answer_names_its_poll(monkeypatch, name):
    """The pending answer's FIRST `next_steps` item is the deciding app's status tool with the approval's id, once;
    calling it reaches that app's status route exactly once with the id."""
    from src.mcp_server.tool_classes import TOOL_CLASSES

    decide_in = TOOL_CLASSES[name].decide_in
    poll_tool, poll_arg, poll_route = POLL[decide_in]
    with drive.routed(monkeypatch) as router:
        _pending_routes(router, decide_in, name)
        mcp = _server(name, decide_in)
        result = drive.call(mcp, name, _args(mcp, name))
        assert not result.is_error, result.content
        answer = result.structured_content
        assert answer["approval"]["status"] == "pending" and answer["approval"]["id"] == ID, answer
        steps = answer["next_steps"]
        assert steps[0]["tool"] == poll_tool and steps[0]["args"] == {poll_arg: ID}, steps
        assert [s["tool"] for s in steps].count(poll_tool) == 1, steps
        assert steps[0]["why"].strip(), steps[0]

        router.requests.clear()
        polled = drive.call(mcp, steps[0]["tool"], steps[0]["args"])
        assert not polled.is_error, polled.content
        assert len(router.calls("GET", poll_route)) == 1, [r.url.path for r in router.requests]


def test_a_refused_recall_that_still_waits_names_the_poll(monkeypatch):
    """A re-call the consume route refuses because the operator has not decided (409 `pending`) is still a pending
    answer: it names the poll too."""
    with drive.routed(monkeypatch) as router:
        router.on("POST", r"/mcp/approvals/apr_r/consume$", status=409, body={"detail": {"status": "pending"}})
        mcp = drive.build("read,probe_monitors", probe_gpu_approval=True)
        args = {"probe_id": "pm_1", "dataset_ids": ["pd_a"], "approval_request_id": "apr_r"}
        result = drive.call(mcp, "evaluate_probe_monitor", args)
    assert not result.is_error, result.content
    assert result.structured_content["next_steps"][0] == {
        "tool": "get_approval_status", "args": {"approval_request_id": "apr_r"},
        "why": result.structured_content["next_steps"][0]["why"]}


@pytest.mark.parametrize("status", ["consumed", "denied", "expired"])
def test_a_decided_answer_names_no_poll(monkeypatch, status):
    """A spent, rejected or expired approval has nothing left to poll."""
    with drive.routed(monkeypatch) as router:
        router.on("POST", r"/mcp/approvals/apr_r/consume$", status=409, body={"detail": {"status": status}})
        mcp = drive.build("read,probe_monitors", probe_gpu_approval=True)
        args = {"probe_id": "pm_1", "dataset_ids": ["pd_a"], "approval_request_id": "apr_r"}
        result = drive.call(mcp, "evaluate_probe_monitor", args)
    assert not result.is_error, result.content
    answer = result.structured_content
    assert answer["approval"]["status"] != "pending"
    assert "get_approval_status" not in [s["tool"] for s in answer.get("next_steps") or []], answer


def test_no_poll_when_the_status_tool_is_not_here(monkeypatch):
    """`get_approval_status` lives in `read`; a server without it names no call it cannot make (the `approval`
    summary still says where the operator decides)."""
    with drive.routed(monkeypatch) as router:
        router.on("POST", r"/mcp/approvals$", body={"id": ID})
        mcp = drive.build("probe_monitors", probe_gpu_approval=True)
        result = drive.call(mcp, "evaluate_probe_monitor", {"probe_id": "pm_1", "dataset_ids": ["pd_a"]})
    assert not result.is_error, result.content
    answer = result.structured_content
    assert answer["approval"]["status"] == "pending"
    assert "get_approval_status" not in [s["tool"] for s in answer.get("next_steps") or []], answer


def _miserver_pending(router, next_steps: list) -> None:
    router.on("GET", r"^/api/approvals/actions$", body=approval_actions(["model_load"]))
    router.on("POST", r"^/api/models/7/load$", status=202, headers={"X-miLLM-Approval": "pending"},
              body={"approval_id": ID, "status": "pending", "action": "model_load", "summary": {"Model": "7"},
                    "next_steps": next_steps})


def test_a_source_poll_is_not_doubled(monkeypatch):
    """A relayed answer whose source already names the poll keeps that one step."""
    own = {"tool": "miserver_get_approval_status", "args": {"approval_id": ID}, "why": "miServer's own words"}
    with drive.routed(monkeypatch, off_contract="the source answers its own `next_steps` or `approval`, which miServer's contract does not declare: the result language must keep, filter or repair them") as router:
        _miserver_pending(router, [{"tool": "miserver_list_models", "args": {}, "why": "w"}, own])
        mcp = drive.build("miserver_models")
        result = drive.call(mcp, "miserver_load_model", {"model_id": 7})
    assert not result.is_error, result.content
    steps = result.structured_content["next_steps"]
    assert steps == [{"tool": "miserver_list_models", "args": {}, "why": "w"}, own], steps


def test_the_poll_goes_first_and_the_list_stays_within_five(monkeypatch):
    """Five source steps and the poll: the poll first, then four, the contract's `maxItems`."""
    source = [{"tool": t, "args": {}, "why": "w"} for t in (
        "miserver_list_models", "miserver_list_approvals", "miserver_lease_status", "miserver_status",
        "miserver_list_profiles")]
    with drive.routed(monkeypatch, off_contract="the source answers its own `next_steps` or `approval`, which miServer's contract does not declare: the result language must keep, filter or repair them") as router:
        _miserver_pending(router, source)
        mcp = drive.build("miserver_models,miserver_runtime")
        result = drive.call(mcp, "miserver_load_model", {"model_id": 7})
    assert not result.is_error, result.content
    steps = result.structured_content["next_steps"]
    assert len(steps) == 5 and steps[0]["tool"] == "miserver_get_approval_status", steps
    assert steps[1:] == source[:4], steps


def test_an_ungated_relay_whose_source_waits_names_the_poll(monkeypatch):
    """The poll follows the `approval` object, not the table: a relay whose row files nothing, answering a source app's
    pending `approval` (an app that gates a call before this table knows it), still names the source's status tool."""
    from src.mcp_server.tool_classes import TOOL_CLASSES

    assert TOOL_CLASSES["miserver_get_model"].approval is None
    source = {"id": ID, "status": "pending", "decide_in": "miserver", "then": "server_runs", "summary": "s"}
    with drive.routed(monkeypatch, off_contract="the source answers its own `next_steps` or `approval`, which miServer's contract does not declare: the result language must keep, filter or repair them") as router:
        router.on("GET", r"^/api/models/7$", body={"id": 7, "approval": source})
        mcp = drive.build("miserver_models")
        result = drive.call(mcp, "miserver_get_model", {"model_id": 7})
    assert not result.is_error, result.content
    answer = result.structured_content
    assert answer["approval"] == source
    assert answer["next_steps"][0]["tool"] == "miserver_get_approval_status"
    assert answer["next_steps"][0]["args"] == {"approval_id": ID}
