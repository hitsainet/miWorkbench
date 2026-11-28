"""045 FR-9: errors that teach. Every tool error an agent reads starts with the contract's error text (line 1
`<code>: <one-line message>`, line 2 `fix: …`, line 3 `next: <JSON list of next_step>`), then the original text
unchanged; the code is the vendored `error_code` (lower case; an app's own upper-case code keeps its spelling in the
original text). Each code is driven through the SDK's in-memory client; the patterns are read from the VENDORED
contract (`x-error-text-format`, `$defs.error_code`), never a local regex.

Logging (A30; R1A-10): a converted validation error keeps the pydantic `ValidationError` as its cause, so the SDK logs
the field names only, never the rejected value; a crash stays an `UnexpectedToolError`, so its traceback is logged.

Controls (0xcc/reviews/045_implementation_controls_2026-11-16.md, FTASKS 10.4):
  remove the re-raise in guard_and_teach (`except ToolError`)  → every code test red
  delete one code's fix in ERROR_CODES                        → test_every_code_has_a_fix red
  put the original text first in `teach`                       → test_the_first_word_is_the_code red
  `raise … from exc` for a validation error                   → test_rejected_arguments_are_not_logged red
  convert a crash into a plain ToolError                       → test_a_crash_is_still_logged red
  append a backend code without lower-casing                   → the error_code check red
"""
from __future__ import annotations

import json
import logging
import re

import pytest

from tests.support import result_language_drive as drive

SECRET = "the-users-private-prompt-7f3a"


def _format():
    c = drive.contract()
    lines = [re.compile(line["pattern"]) for line in c["x-error-text-format"]["lines"]]
    return lines, c["$defs"]["error_code"]


def _parse(result) -> dict:
    """The error text's parts, after asserting it is an error in the contract's layout."""
    assert result.is_error, result.structured_content
    text = result.content[0].text
    lines, code_def = _format()
    head = text.split("\n")
    assert len(head) >= 4, text
    first, fix, nxt = (pattern.match(line) for pattern, line in zip(lines, head[:3]))
    assert first and fix and nxt, text
    code = first.group(1)
    assert re.match(code_def["pattern"], code) and len(code) <= code_def["maxLength"], code
    steps = json.loads(nxt.group(1))
    check = drive.validator("next_step")
    for step in steps:
        assert not list(check.iter_errors(step)), step
    return {"code": code, "message": first.group(3), "fix": fix.group(1), "next": steps,
            "original": "\n".join(head[3:]), "text": text}


def test_unknown_arguments(monkeypatch):
    with drive.routed(monkeypatch):
        mcp = drive.build("read")
        err = _parse(drive.call(mcp, "get_feature", {"feature_id": "f_1", "featureid": "x"}))
    assert err["code"] == "unknown_arguments"
    assert "unknown argument: featureid" in err["original"] and "get_feature accepts" in err["original"]


def test_invalid_arguments(monkeypatch):
    with drive.routed(monkeypatch):
        mcp = drive.build("read")
        err = _parse(drive.call(mcp, "get_feature", {"feature_id": {"nested": SECRET}}))
    assert err["code"] == "invalid_arguments"
    assert err["original"].startswith("Error executing tool get_feature")


def test_steering_guardrail(monkeypatch):
    from src.mcp_server.tools import steering

    monkeypatch.setattr(steering, "_inflight", {"t-a", "t-b"})
    with drive.routed(monkeypatch):
        mcp = drive.build("steering", steering_max_concurrent=2)
        tool = mcp._tool_manager.get_tool("steer_compare")  # noqa: SLF001
        from tests.support.mcp_drive import _sample

        schema = tool.parameters
        args = {k: _sample(schema["properties"][k], schema.get("$defs") or {}) for k in schema.get("required", [])}
        err = _parse(drive.call(mcp, "steer_compare", args))
    assert err["code"] == "steering_guardrail"
    assert "Guardrail: 2 agent steering task(s) already in flight" in err["original"]


def test_backend_status_with_its_own_upper_case_code(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/features/f_1$", status=409, body={"detail": {"code": "PROBE_NOT_TRAINED",
                                                                       "message": "train it first"}})
        mcp = drive.build("read")
        err = _parse(drive.call(mcp, "get_feature", {"feature_id": "f_1"}))
    assert err["code"] == "backend_409_probe_not_trained"
    assert "PROBE_NOT_TRAINED" in err["original"] and "backend 409" in err["original"]
    assert err["message"] == "miWorkbench refused get_feature."


def test_backend_status_without_a_code(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/features/f_1$", status=404, body={"detail": "Feature not found"})
        mcp = drive.build("read")
        err = _parse(drive.call(mcp, "get_feature", {"feature_id": "f_1"}))
    assert err["code"] == "backend_404"


def test_a_backend_next_list_becomes_the_next_line(monkeypatch):
    nxt = [{"tool": "get_feature_examples", "arguments": {"feature_id": "f_1"}, "why": "see where it fires"}]
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/features/f_1$", status=409, body={"detail": {"code": "stale", "next": nxt}})
        mcp = drive.build("read")
        err = _parse(drive.call(mcp, "get_feature", {"feature_id": "f_1"}))
    assert err["code"] == "backend_409_stale"
    assert err["next"] == [{"tool": "get_feature_examples", "args": {"feature_id": "f_1"}, "why": "see where it fires"}]


def test_miserver_status_and_its_next_hint(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/models$", status=500,
                  body={"error": {"code": "ENGINE_FAILED", "message": "boom", "next": "miserver_list_models"}})
        mcp = drive.build("miserver_models")
        err = _parse(drive.call(mcp, "miserver_list_models", {}))
    assert err["code"].startswith("miserver_500"), err["code"]
    assert err["message"] == "miServer refused miserver_list_models."
    assert [s["tool"] for s in err["next"]] == ["miserver_list_models"]


def test_dataworks_status(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/v1/versions/v_1$", status=404,
                  body={"code": "version-not-found", "message": "no such version"})
        mcp = drive.build("dataworks")
        err = _parse(drive.call(mcp, "dataworks_get_version", {"version_id": "v_1"}))
    assert err["code"].startswith("dataworks_404"), err["code"]
    assert "version-not-found" in err["original"] or "no such version" in err["original"]


def _crashing_server(monkeypatch):
    mcp = drive.build("read")

    @mcp.tool()
    async def crash_for_test(prompt: str) -> dict:
        raise RuntimeError("crash-detail-only-the-log-sees")

    return mcp


def test_internal_error(monkeypatch):
    with drive.routed(monkeypatch):
        mcp = _crashing_server(monkeypatch)
        err = _parse(drive.call(mcp, "crash_for_test", {"prompt": "x"}))
    assert err["code"] == "internal_error"
    assert "crash-detail-only-the-log-sees" not in err["text"]  # a crash's own text stays on the server


def test_every_code_has_a_fix():
    from src.mcp_server.result_language import ERROR_CODES

    families = {"unknown_arguments", "invalid_arguments", "steering_guardrail", "backend", "miserver", "dataworks",
                "tool_error", "internal_error"}
    assert families <= set(ERROR_CODES), families - set(ERROR_CODES)
    assert all(isinstance(fix, str) and fix.strip() for fix in ERROR_CODES.values())


def test_the_first_word_is_the_code(monkeypatch):
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/features/f_1$", status=409, body={"detail": {"code": "Some Code!"}})
        mcp = drive.build("read")
        result = drive.call(mcp, "get_feature", {"feature_id": "f_1"})
    text = result.content[0].text
    assert text.split(":", 1)[0] == "backend_409_some_code", text
    assert text.split("\n")[1].startswith("fix: ") and text.split("\n")[2].startswith("next: [")


def test_the_code_is_lower_cased_and_bounded():
    from src.mcp_server.client import BackendError
    from src.mcp_server.result_language import classify

    _lines, code_def = _format()
    for raw in ("MODEL_NOT_RESIDENT", "a.b-c d", "X" * 200):
        try:
            try:
                raise BackendError(409, {"error": {"code": raw}}, product="miserver")
            except BackendError as inner:
                raise RuntimeError("wrapper") from inner
        except RuntimeError as exc:
            code = classify(exc)
        assert re.match(code_def["pattern"], code) and len(code) <= code_def["maxLength"], code


def test_rejected_arguments_are_not_logged(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    with drive.routed(monkeypatch):
        mcp = drive.build("read")
        result = drive.call(mcp, "get_feature", {"feature_id": {"nested": SECRET}})
    assert result.is_error
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "rejected arguments" in logged and "feature_id" in logged, logged[-2000:]
    assert SECRET not in logged


def test_a_crash_is_still_logged(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    with drive.routed(monkeypatch):
        mcp = _crashing_server(monkeypatch)
        drive.call(mcp, "crash_for_test", {"prompt": "x"})
    crashes = [r for r in caplog.records if r.levelno >= logging.ERROR and r.exc_info]
    assert crashes, "no traceback logged for a crash"
    import traceback

    logged = "".join(traceback.format_exception(*crashes[0].exc_info))
    assert "crash-detail-only-the-log-sees" in logged


@pytest.mark.parametrize("code", ["unknown_arguments", "backend_409_x", "miserver_unreachable", "internal_error"])
def test_teach_layout_matches_the_contract(code):
    from src.mcp_server.result_language import teach

    lines, _ = _format()
    text = teach(code, "some_tool", "Error executing tool some_tool: original", [])
    head = text.split("\n")
    assert all(p.match(line) for p, line in zip(lines, head[:3])), text
    assert head[3] == "Error executing tool some_tool: original"


def test_an_unreachable_backend_reads_unreachable(monkeypatch):
    """OVN-b-10 (OR1-045-2 M19): a connection failure (`BackendError(0)`) reads `backend_unreachable`, through the
    server, never `backend_0`. Control: always use `str(status_code)` in `classify` → red."""
    import httpx

    with drive.routed(monkeypatch) as router:
        def refuse(request):
            raise httpx.ConnectError("connection refused", request=request)

        router.rules.insert(0, ("GET", re.compile(r"/features/f_1$"), refuse))
        mcp = drive.build("read")
        err = _parse(drive.call(mcp, "get_feature", {"feature_id": "f_1"}))
    assert err["code"] == "backend_unreachable"
    assert "unreachable" in err["original"]


def test_an_error_hint_naming_no_tool_here_is_left_out(monkeypatch):
    """F4 (review-045ab-r2 candidate): a product error's string `next` that names no tool on this server (miServer's
    own MCP name, or a disabled category's tool) never reaches the error's `next:` line.
    Control: keep a string hint whatever it names → red."""
    with drive.routed(monkeypatch) as router:
        router.on("GET", r"/api/models$", status=500,
                  body={"error": {"code": "ENGINE_FAILED", "message": "boom", "next": "load_model"}})
        mcp = drive.build("miserver_models")
        err = _parse(drive.call(mcp, "miserver_list_models", {}))
    assert err["next"] == [], err["next"]


def test_an_unreadable_approval_code_teaches():
    from src.mcp_server.result_language import ERROR_CODES, teach

    lines, code_def = _format()
    text = teach("approval_unreadable", "some_tool", "original", [])
    assert all(p.match(line) for p, line in zip(lines, text.split("\n")[:3])), text
    assert re.match(code_def["pattern"], "approval_unreadable") and "approval_unreadable" in ERROR_CODES
