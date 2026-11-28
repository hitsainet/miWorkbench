"""Coverage of miWorkbench's answer contract is set equality over the live tables (046 FR-046.3, .4, .9, .15; TD10;
FTASKS 6.4).

- The keys of `responses`, `excluded` and `not_covered` EQUAL every operation of the live app in its schema.
- The covered set (`responses` and `excluded`) EQUALS the routes the live registry's tools call (the one walker) plus
  `SIBLING_ROUTES`; no covered key is in `not_covered`, and the walker reads every call.
- `SIBLING_ROUTES` holds every miWorkbench route miAssist records (cross-repo).
- Every `excluded` entry's media type is the route's own, never JSON.
- Every `responses` entry is typed (FR-046.4), except a SHRINKING allowance: `ALLOWANCE` must equal the untyped set
  exactly, and may only lose routes from the set recorded when the contract was first written. Each typing commit of
  FTASKS 8.1 to 8.10 removes its router's routes here; 8.16 leaves it empty.
- No `additionalProperties: false` anywhere; `$defs` holds exactly the reached definitions; the counts equal a
  recount; no component name is module-qualified (TD10's pin, empty).

Controls: M9 (the writer skips `/api/v1/circuits`), M10 (`SIBLING_ROUTES` emptied), M14 (every component kept), and a
scratch tool calling a `not_covered` route (FR-046.3 NC2). Record: 0xcc/reviews/046_controls_<date>.md.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from src.api import contract_tools as tools
from src.api.answers_contract import (
    SIBLING_ROUTES,
    answers_contract,
    covered_keys,
    live_tool_names,
    operations,
)
from tests.support.cross_repo import require_cross_repo
from tests.support.sibling_repos import miassist_repo

START = Path(__file__).resolve().parents[1] / "fixtures" / "answers_untyped_start_2026-11-26.json"
#: TD10: component names that are module-qualified (`src__schemas__x__Model`); none on 2026-11-26.
QUALIFIED_COMPONENTS: frozenset[str] = frozenset()
#: The untyped covered routes still allowed (FR-046.4). Only ever shrinks; empty at FTASKS 8.16.
ALLOWANCE: frozenset[str] = frozenset()  # FTASKS 8.16: every covered route typed (2026-11-27)


@functools.lru_cache(maxsize=1)
def _doc() -> dict:
    """The contract GENERATED now (the writer's own output; `test_answers_contract.py` pins the committed file to it),
    so a writer that drops a route or keeps every component is red here, not only at the pin."""
    return answers_contract()


@functools.lru_cache(maxsize=1)
def _covered() -> frozenset[str]:
    return covered_keys(live_tool_names())


def _keys(doc: dict, *sections: str) -> set[str]:
    return {k for s in sections for k in doc[s]}


def test_the_sections_are_every_operation_of_the_live_app():
    from src.main import app

    live = {f"{m} {p}" for m, p, _r in operations(app)}
    doc = _doc()
    found = [k for s in ("responses", "excluded", "not_covered") for k in doc[s]]
    assert len(found) == len(set(found)), "a key is in two sections"
    assert set(found) == live, {"missing": sorted(live - set(found)), "extra": sorted(set(found) - live)}


def test_the_covered_set_is_what_the_tools_and_siblings_call():
    doc = _doc()
    covered = _keys(doc, "responses", "excluded")
    want = _covered()
    assert covered == want, {"called, not covered": sorted(want - covered),
                             "covered, not called": sorted(covered - want)}
    assert not want & set(doc["not_covered"]), sorted(want & set(doc["not_covered"]))
    assert all(v == {"reason": "no tool or sibling calls it"} for v in doc["not_covered"].values())


def test_the_walker_reads_every_call():
    from src.mcp_server.route_map import unreadable_calls

    assert unreadable_calls() == []


def test_a_tool_calling_an_uncovered_route_is_named(monkeypatch):
    """FR-046.3 NC2, built in: a planted tool route outside the covered set lands in the covered set, so the
    comparison with the committed `not_covered` names it."""
    from src.mcp_server import route_map

    planted = next(iter(sorted(_doc()["not_covered"])))
    method, path = planted.split(" ", 1)
    real = route_map.tool_routes()
    monkeypatch.setattr(route_map, "tool_routes",
                        lambda: {**real, "scratch_tool": frozenset({("miworkbench", method, path[len("/api/v1"):])})})
    covered = covered_keys([*live_tool_names(), "scratch_tool"])
    assert planted in covered and planted in _doc()["not_covered"]


def test_the_health_route_miassist_calls_is_covered():
    """FR-046.3: `GET /api/health` is called by miAssist, not by a tool, so only `SIBLING_ROUTES` covers it."""
    assert ("GET", "/api/health") in SIBLING_ROUTES
    assert "GET /api/health" in _keys(_doc(), "responses", "excluded")
    assert covered_keys([]) == {"GET /api/health"}  # with no tool at all, the sibling's route is still covered


def test_sibling_routes_hold_every_route_miassist_records():
    """FR-046.3: every miWorkbench route miAssist's playbook fakes answer is covered (cross-repo)."""
    path = miassist_repo() / "tests" / "fixtures" / "openapi" / "responses.json"
    if not path.is_file():
        message = f"miAssist's recorded routes are not at {path}"
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    recorded = set(json.loads(path.read_text(encoding="utf-8"))["apps"]["miworkbench"]["routes"])
    assert "GET /api/health" in recorded
    covered = _keys(_doc(), "responses", "excluded")
    assert recorded <= covered, sorted(recorded - covered)
    assert {f"{m} {p}" for m, p in SIBLING_ROUTES} <= covered


def test_excluded_routes_carry_their_own_media_type():
    """FR-046.9: a covered route with no JSON body is excluded with its declared media type (`null` for a 204)."""
    from src.main import app

    by_key = {f"{m} {p}": r for m, p, r in operations(app)}
    for key, entry in _doc()["excluded"].items():
        assert entry["media_type"] != "application/json", key
        route = by_key[key]
        if entry["media_type"] is None:
            assert route.status_code == 204 and entry["reason"] == "no content", key
        else:
            assert entry["reason"], key
        # every declared 2xx, each with its own media type (null: no body); the row's media is its first status's
        # (046-2 review round 1: the extraction delete's 202 text note was recorded as "no content")
        declared = sorted(str(s) for s in {route.status_code or 200, *(route.responses or {})} if 200 <= int(s) < 300)
        assert sorted(entry["statuses"]) == declared, (key, entry["statuses"], declared)
        assert entry["statuses"][sorted(entry["statuses"])[0]] == entry["media_type"], key
        assert "application/json" not in entry["statuses"].values(), key
    assert _doc()["excluded"]["DELETE /api/v1/extractions/{extraction_id}"]["statuses"] == {
        "202": "text/plain", "204": None}


def test_every_covered_answer_is_typed_but_the_shrinking_allowance():
    """FR-046.4: an untyped `responses` entry fails unless it is still in the allowance, which equals the untyped set
    exactly (a route typed must leave it; a route newly untyped is red)."""
    doc = _doc()
    ext = frozenset(doc["external"])
    untyped = {k for k, e in doc["responses"].items()
               if not all(tools.is_typed(s, doc["$defs"], ext) for st, s in e.items() if st[:1].isdigit())}
    assert untyped == ALLOWANCE, {"untyped, not allowed": sorted(untyped - ALLOWANCE),
                                  "typed now, still allowed (remove them)": sorted(ALLOWANCE - untyped)}


def test_the_allowance_only_shrinks():
    start = set(json.loads(START.read_text(encoding="utf-8"))["routes"])
    assert ALLOWANCE <= start, sorted(ALLOWANCE - start)


def test_no_object_is_closed():
    """SH7: the published contract allows extra keys everywhere."""
    closed = [s for s in tools._walk(_doc()) if s.get("additionalProperties") is False]
    assert not closed, closed[:3]


def test_defs_are_exactly_the_reached_definitions():
    """FR-046.15: no definition is unreferenced, and every reference resolves."""
    doc = _doc()
    reached = tools.reachable_defs([doc["responses"], doc["errors"]], doc["$defs"])
    assert set(reached) == set(doc["$defs"]), sorted(set(doc["$defs"]) - set(reached))


def test_the_counts_equal_a_recount():
    doc = _doc()
    ext = frozenset(doc["external"])
    typed = sum(1 for e in doc["responses"].values()
                if all(tools.is_typed(s, doc["$defs"], ext) for st, s in e.items() if st[:1].isdigit()))
    assert doc["coverage"] == {"typed": typed, "unions": doc["coverage"]["unions"], "excluded": len(doc["excluded"]),
                               "not_covered": len(doc["not_covered"]), "untyped": len(doc["responses"]) - typed}
    unions = sum(1 for e in doc["responses"].values() for st, s in e.items()
                 if st[:1].isdigit() and len([m for m in tools._deref(s, doc["$defs"]).get("anyOf", tools._deref(
                     s, doc["$defs"]).get("oneOf", [s])) if tools._deref(m, doc["$defs"]).get("type") != "null"]) > 1)
    assert doc["coverage"]["unions"] == unions


def test_no_component_name_is_module_qualified():
    """TD10: a module-qualified component name (Pydantic's collision fallback) would rename on a refactor; none is."""
    qualified = {n for n in _doc()["$defs"] if "__" in n.replace("__history", "")}
    assert qualified == QUALIFIED_COMPONENTS, sorted(qualified)
