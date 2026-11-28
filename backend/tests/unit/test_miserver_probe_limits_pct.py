"""miWorkbench's probe-limit tools follow miServer's slowdown budget (miServer MCP contract v1.25, ENH-006 D83).

miServer no longer limits probes by behaviour count: the cost budget alone decides, and its default is 5% of the
model's own measured time (decode ms per generated token, prefill ms per 1,000 prompt tokens). So:

* `miserver_request_probe_limits` has no `max_behaviours` parameter (it would do nothing), and a caller who sends one
  anyway never gets it forwarded to miServer;
* each budget half is `{value, unit}` with `unit` `pct` (default) or `ms`, sent to miServer as EXACTLY that object
  (miServer's `BudgetSetting` has `extra="forbid"` and a required `unit`), the unit always present;
* approving the recommendation is still `source: "recommended-approved"` + its digest, with no value fields;
* the two reads relay the v1.25 fields (`decode_budget_pct`, `capacity`, `superseded`, null `max_behaviours`)
  unchanged.

Every call goes through the LIVE registry (the deployed categories, the real `MiServerClient` on a stub miServer) and
asserts the body AND the request count. Negative controls: 0xcc/reviews/probe_limits_pct_controls_2026-10-29.md.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from tests.unit.test_miserver_approvals_relay import _MiServer, _pending, live  # noqa: F401 - `live` is a fixture
from tests.support.cross_repo import require_cross_repo

DIGEST = "d" * 64
PATH = "/api/models/7/probe-limits"


def _limit_puts(miserver: _MiServer) -> list[dict[str, Any]]:
    return [r for r in miserver.seen if r["path"] == PATH]


def _call(live, args: dict) -> tuple[Any, _MiServer]:
    miserver = _MiServer()
    call, _ = live(miserver)
    return call("miserver_request_probe_limits", {"model_id": 7, "reason": "r", **args}), miserver


# ── the body miServer receives ───────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("args,expected", [
    # pct, stated
    ({"decode_budget": {"value": 5, "unit": "pct"}, "prefill_budget": {"value": 7.5, "unit": "pct"}},
     {"decode_budget": {"value": 5.0, "unit": "pct"}, "prefill_budget": {"value": 7.5, "unit": "pct"}}),
    # pct is the DEFAULT unit, and it is SENT (miServer requires it)
    ({"decode_budget": {"value": 4}},
     {"decode_budget": {"value": 4.0, "unit": "pct"}}),
    # ms, an absolute override
    ({"decode_budget": {"value": 0.8, "unit": "ms"}, "prefill_budget": {"value": 1.5, "unit": "ms"}},
     {"decode_budget": {"value": 0.8, "unit": "ms"}, "prefill_budget": {"value": 1.5, "unit": "ms"}}),
    # mixed units, plus the probe backstop
    ({"max_armed": 32, "decode_budget": {"value": 5}, "prefill_budget": {"value": 2, "unit": "ms"}},
     {"max_armed": 32, "decode_budget": {"value": 5.0, "unit": "pct"},
      "prefill_budget": {"value": 2.0, "unit": "ms"}}),
])
def test_an_override_sends_exactly_miserver_body_once(live, args, expected):
    result, miserver = _call(live, args)
    # 045 FR-7 adds `approval` beside the relayed body; every relayed key is unchanged.
    assert {k: v for k, v in result.items() if k not in ("approval", "next_steps")} == {
        **_pending("probe_limits_write"), "decide_in": "miserver"}
    # 045 follow-up (2026-11-24): the poll comes first in `next_steps`, with miServer's id.
    assert result["next_steps"][0]["tool"] == "miserver_get_approval_status"
    assert result["next_steps"][0]["args"] == {"approval_id": result["approval_id"]}
    assert result["approval"]["decide_in"] == "miserver"
    puts = _limit_puts(miserver)
    assert len(puts) == 1 and puts[0]["method"] == "PUT"
    assert puts[0]["body"] == {"reason": "r", "set_by": "mistudio-agent", "source": "operator", **expected}


def test_every_budget_sent_carries_its_unit(live):
    _, miserver = _call(live, {"decode_budget": {"value": 5}, "prefill_budget": {"value": 5}})
    body = _limit_puts(miserver)[0]["body"]
    for half in ("decode_budget", "prefill_budget"):
        assert set(body[half]) == {"value", "unit"} and body[half]["unit"] == "pct"


def test_approving_the_recommendation_sends_the_digest_and_no_values(live):
    result, miserver = _call(live, {"recommendation_digest": DIGEST})
    assert result["decide_in"] == "miserver"
    puts = _limit_puts(miserver)
    assert len(puts) == 1
    assert puts[0]["body"] == {"reason": "r", "set_by": "mistudio-agent", "source": "recommended-approved",
                               "recommendation_digest": DIGEST}


def test_values_beside_an_approval_are_refused_and_nothing_is_sent(live):
    result, miserver = _call(live, {"recommendation_digest": DIGEST, "decode_budget": {"value": 5}})
    assert result["refused"] == "approval_takes_no_values" and result["fields"] == ["decode_budget"]
    assert miserver.seen == []


# ── max_behaviours is gone ────────────────────────────────────────────────────────────────────────────────────────


def _schema(tool: str) -> dict[str, Any]:
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(auth_token="x" * 32), stdio=True)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    return tools[tool].input_schema


def test_max_behaviours_is_not_a_parameter_in_the_live_registry(live):
    live(_MiServer())  # deployed categories + a miServer URL
    schema = _schema("miserver_request_probe_limits")
    props = schema["properties"]
    assert "max_behaviours" not in props
    assert {"decode_budget", "prefill_budget", "max_armed", "recommendation_digest"} <= set(props)
    assert "pct" in json.dumps(schema) and "ms" in json.dumps(schema)


def test_a_max_behaviours_sent_anyway_never_reaches_miserver(live):
    """It used to be dropped in silence and the rest sent; since the server-wide unknown-argument refusal (MCP gap
    2026-11-10) the call is REFUSED, naming it, and nothing at all is sent."""
    from mcp.server.mcpserver.exceptions import ToolError

    miserver = _MiServer()
    call, _ = live(miserver)
    with pytest.raises(ToolError, match="unknown argument: max_behaviours"):
        call("miserver_request_probe_limits",
             {"model_id": 7, "reason": "r", "max_behaviours": 3, "decode_budget": {"value": 5}})
    assert _limit_puts(miserver) == [] and miserver.seen == []


# ── arguments refused before anything is sent ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("budget", [
    {"value": 5, "unit": "percent"},       # not a unit miServer knows
    {"value": 0, "unit": "pct"},           # below 0.01
    {"value": 101, "unit": "pct"},         # above 100 %
    {"value": 11, "unit": "ms"},           # above miServer's 10 ms
    {"value": 5, "unit": "pct", "x": 1},   # miServer forbids extra fields
])
def test_a_budget_miserver_would_refuse_is_refused_here_and_nothing_is_sent(live, budget):
    miserver = _MiServer()
    call, _ = live(miserver)
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="validation error") as caught:
        call("miserver_request_probe_limits", {"model_id": 7, "reason": "r", "decode_budget": budget})
    assert "decode_budget" in str(caught.value)
    assert miserver.seen == []


def test_the_ranges_match_miserver():
    """The local range is miServer's `_check_budget` range (cross-repo, read from the sibling when present)."""
    import os
    import re
    from pathlib import Path

    from src.mcp_server.tools.miserver_behaviours import BUDGET_RANGE

    assert BUDGET_RANGE == {"pct": (0.01, 100.0), "ms": (0.01, 10.0)}
    from tests.support.sibling_repos import miserver_path, miserver_path_or_none, miserver_repo

    repo = miserver_repo()
    source = miserver_path_or_none(repo, "services", "probe_limits.py")
    if source is None or not source.exists():
        if require_cross_repo():
            pytest.fail(f"miServer checkout not found at {repo}; set MISERVER_REPO")
        pytest.skip(f"CROSS-REPO, SKIPPED LOUDLY: no miServer checkout at {repo} (set MISERVER_REPO)")
    text = source.read_text()
    match = re.search(r'lo, hi = \(([\d.]+), ([\d.]+)\) if unit == "ms" else \(([\d.]+), ([\d.]+)\)', text)
    assert match, "miServer's _check_budget range line moved; re-read it and update BUDGET_RANGE"
    ms_lo, ms_hi, pct_lo, pct_hi = (float(g) for g in match.groups())
    assert BUDGET_RANGE == {"pct": (pct_lo, pct_hi), "ms": (ms_lo, ms_hi)}
    assert 'unit: Literal["ms", "pct"]' in (
        miserver_path(repo, "api", "routes", "management", "model_probe_capacity.py")).read_text()


# ── the reads relay the v1.25 fields unchanged ────────────────────────────────────────────────────────────────────

RECOMMENDATION = {
    "model_id": 7,
    "values": {"max_behaviours": None, "decode_budget_pct": 5.0, "prefill_budget_pct": 5.0,
               "decode_budget_ms": 1.134, "prefill_budget_ms": None},
    "budget": {"decode_ms_per_pass": 1.134, "prefill_ms_per_1k": 0.5, "headroom": 0.9},
    "capacity": {"reference_sets": 3, "reference_set_probes": 27, "dense_probes": 40, "binding": "decode",
                 "dense_binding": "decode", "upper_bound": False},
    "recommendation_digest": DIGEST,
}
CAPACITY = {
    "measured": {"id": 6, "decode_ms_per_token": 22.68},
    "configured": {"max_behaviours": 2, "decode_budget": {"value": 0.36, "unit": "ms"},
                   "source": "recommended-approved"},
    "effective": {"max_behaviours": None, "decode_ms_per_pass": 1.134,
                  "sources": {"max_behaviours": "not enforced (D83: the cost budget decides)",
                              "decode": "global (5% of 22.68 ms); the 0.36 ms approved from a recommendation "
                                        "before D83 is superseded"},
                  "approval": {"source": "recommended-approved", "superseded": True}},
    "implied": {"dense": {"n": 3}},
    "armed": None,
}


@pytest.mark.parametrize("tool,args,path,data", [
    ("miserver_get_probe_limit_recommendation", {"model_id": "7"}, "/api/models/7/probe-limits/recommendation",
     RECOMMENDATION),
    ("miserver_probe_capacity", {"model_id": 7}, "/api/models/7/probe-capacity", CAPACITY),
])
def test_the_reads_relay_the_v1_25_fields_unchanged(live, tool, args, path, data):
    seen: list[str] = []

    def miserver(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        seen.append(request.url.path)
        return httpx.Response(200, json={"success": True, "data": data, "error": None})

    call, _ = live(miserver)
    result = call(tool, args)
    assert seen == [path]
    # 046: the fake is completed with every field miServer always sends; the v1.25 fields arrive unchanged.
    assert _carries(result, data), (result, data)


def _carries(actual, expected) -> bool:
    """Every value of `expected` is in `actual`, at every depth (`actual` may carry more)."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and _carries(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _carries(a, e) for a, e in zip(actual, expected, strict=True))
    return actual == expected
