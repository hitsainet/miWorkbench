"""The `miserver_*` relays are typed from miServer's vendored answer contract (046 FR-046.26, FTASKS 2.9 and 3.3).

- Every miServer route a REGISTERED tool calls (the one route walker) is a key of the vendored contract's
  `responses` or `excluded`, apart from the listed exception: `GET /openapi.json`, which `miserver_recalibrate_probe`
  reads as a capability probe (its answer is never relayed). A relay calling a route miServer does not declare is red.
- Every miServer relay is typed (none is generic: miServer's contract types all its routes).
- REAL ANSWERS (FR-046.26 AC): `tests/fixtures/miserver_answers_recorded_044.json` holds one answer per route that
  miServer 044's own route tests produced (`tests/unit/api/test_relayed_routes_answer.py`, through its live app, each
  checked by miServer's validator against the same contract revision). Each relay that answers a recorded route
  unchanged is driven through the SDK client with that answer in its real envelope: the client must accept it, and
  the answer must be the envelope's `data`. One relay per miServer category at least.

Controls (0xcc/reviews/046_controls_2026-11-25.md):
  plant a walker route absent from the contract → test_a_planted_route_absent_from_the_contract_is_named (built in)
  M28 (skip the envelope unwrap)                → test_a_relay_validates_miServers_recorded_answer red, every envelope relay
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest

from tests.support import mcp_drive
from tests.support.live_registry import registered_tools

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "miserver_answers_recorded_044.json"
#: Relayed calls that are not contract routes, with the reason (FR-046.26).
NOT_CONTRACT_ROUTES = {("GET", "/openapi.json"): "miserver_recalibrate_probe reads miServer's OpenAPI as a capability "
                                                 "probe (which fields the recalibrate route takes); never relayed"}


@functools.lru_cache(maxsize=1)
def _contract_keys() -> dict[tuple[str, str], str]:
    from scripts.generate_output_schemas import MISERVER_CONTRACT
    from src.mcp_server.route_map import normalise

    contract = json.loads(MISERVER_CONTRACT.read_text(encoding="utf-8"))
    return {(k.split(" ", 1)[0], normalise(k.split(" ", 1)[1])): k
            for section in ("responses", "excluded") for k in contract[section]}


def _relayed(registered: set[str]) -> dict[tuple[str, str], set[str]]:
    from src.mcp_server.route_map import normalise, tool_routes

    out: dict[tuple[str, str], set[str]] = {}
    for tool in registered:
        for product, method, path in tool_routes().get(tool, ()):
            if product == "miserver":
                out.setdefault((method, normalise(path)), set()).add(tool)
    return out


def _absent(relayed: dict) -> dict:
    return {k: sorted(v) for k, v in relayed.items() if k not in _contract_keys() and k not in NOT_CONTRACT_ROUTES}


def test_every_relayed_route_is_in_the_vendored_contract(monkeypatch):
    relayed = _relayed(registered_tools(monkeypatch))
    assert not _absent(relayed), _absent(relayed)
    assert set(NOT_CONTRACT_ROUTES) <= set(relayed), "a listed exception no tool calls any more: remove it"
    assert len(relayed) >= 100, len(relayed)


def test_a_planted_route_absent_from_the_contract_is_named(monkeypatch):
    """The route test's own control: a relay that starts calling an undeclared route is named."""
    relayed = _relayed(registered_tools(monkeypatch))
    planted = {**relayed, ("GET", "/api/not-a-route/{}"): {"miserver_planted"}}
    assert _absent(planted) == {("GET", "/api/not-a-route/{}"): ["miserver_planted"]}


def test_every_miServer_relay_is_typed():
    from src.mcp_server.tool_classes import OUTPUT_SCHEMAS_PATH

    document = json.loads(OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8"))
    relays = [n for n in [*document["tools"], *document["untyped"]] if n.startswith("miserver_")]
    assert len(relays) >= 95, len(relays)
    assert not [n for n in relays if n in document["untyped"]]


# ── real answers ─────────────────────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _recorded() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _cases() -> dict[str, tuple[str, int, Any]]:
    """`{tool: (contract key, status, recorded body)}` for each relay that answers a recorded route unchanged."""
    from scripts.generate_output_schemas import live_tools
    from src.mcp_server.route_map import normalise, returns_unchanged

    recorded = {(k.split(" ", 1)[0], normalise(k.split(" ", 1)[1])): (k, v)
                for k, v in _recorded()["answers"].items()}
    out = {}
    for name, tool in sorted(live_tools().items()):
        if not name.startswith("miserver_"):
            continue
        routes, _ = returns_unchanged(tool.fn)
        if not routes or len(routes) != 1:
            continue
        (_product, method, path), = routes
        hit = recorded.get((method, normalise(path)))
        if hit is not None and str(hit[1]["status"]).startswith("2"):
            out[name] = (hit[0], int(hit[1]["status"]), hit[1]["body"])
    return out


def test_the_recorded_answers_name_their_source_and_cover_every_category():
    source = _recorded()["source"]
    assert source["repo"] == "miServer" and source["commit"] and "test_relayed_routes_answer" in source["recorded_by"]
    cases = _cases()
    servers = mcp_drive.build_every_category()
    by_category = {c: {t.name for t in mcp_drive.wire_tools(m)} for c, m in servers.items() if c.startswith("miserver_")}
    covered = sorted(c for c, tools in by_category.items() if tools & set(cases))
    assert len(covered) >= 5 and len(cases) >= 40, (covered, len(cases))


def test_the_recorded_answers_carry_no_workstation_path():
    """Review round 1, finding 9: the fixture ships in the public mirror, so a recorded value naming the recording
    machine's user (`/tmp/pytest-of-<user>/…`, `/home/<user>/…`) is scrubbed to a neutral path."""
    import re

    text = FIXTURE.read_text(encoding="utf-8")
    found = sorted(set(re.findall(r"/tmp/pytest-of-[^/\"]+|/home/[^/\"]+|/Users/[^/\"]+", text)))
    assert not found, found


@pytest.fixture(scope="module")
def world():
    """ONE backend and every category's server for the module; each case resets the backend."""
    with pytest.MonkeyPatch.context() as mp:
        backend = mcp_drive.RouteBackend()
        with mcp_drive.patched_routes(mp, backend):
            servers = mcp_drive.build_every_category()
            owner = {t.name: m for m in servers.values() for t in m._tool_manager.list_tools()}  # noqa: SLF001
            wire = {t.name: t for m in servers.values() for t in mcp_drive.wire_tools(m)}
            yield backend, owner, wire
            backend.unmapped.clear()


@pytest.mark.parametrize("name", sorted(_cases()))
def test_a_relay_validates_miServers_recorded_answer(name, world):
    backend, owner, wire = world
    key, status, body = _cases()[name]
    method, path = key.split(" ", 1)
    backend.routes.clear(), backend.calls.clear(), backend.unmapped.clear()
    backend.answer(method, path, body, status=status)
    result = mcp_drive.call(owner[name], name, mcp_drive.sample_arguments(wire[name]))  # raises on a refusal
    assert not backend.unmapped, sorted(backend.unmapped)
    assert not result.is_error, result.content[0].text[:300]
    data = body["data"] if isinstance(body, dict) and "success" in body else body
    expected = data if isinstance(data, dict) else {"result": data}
    answer = {k: v for k, v in result.structured_content.items() if k not in ("next_steps", "approval")}
    assert answer == expected
    assert len(backend.calls_to(method, path)) == 1
