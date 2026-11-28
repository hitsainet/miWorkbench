"""The T1 routes are typed by documentation only (046 FTASKS 8.16; FR-046.5 as amended 2026-11-26, the serializer
decision): each keeps the serializer it had on `origin/main`, and its contract entry is a typed model.

- Every one of the 146 T1 routes (`0xcc/reviews/046_endpoint_ast.json`) serves with the `response_model` and
  `status_code` it had at `origin/main` `f3943c72c` (`tests/fixtures/answers_t1_served_models_f3943c72c.json`, read
  from that tree's live app), except the two listed below, each of which cannot change a byte.
- Every T1 route's contract entry names a typed 2xx model (the coverage test's allowance is empty; this checks the
  T1 set by name, so a T1 route moved to `excluded` or `not_covered` is red too, unless it is a bodiless route).

The bodies are pinned separately: `0xcc/reviews/046_endpoint_ast_check.py --check` (146 of 146 equal).

Controls (record `0xcc/reviews/046_controls_2026-11-25.md`): give one T1 route a `response_model` (the
documentation model enforced, which WOULD drop undeclared keys) → the served-model test red; drop one T1 route's
`responses=` → the documented test red.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "backend" / "tests" / "fixtures" / "answers_t1_served_models_f3943c72c.json"

#: Served-model changes that cannot change a byte, each with its reason.
EXCEPTIONS = {
    # The endpoint returns its own `Response` (a 204, or a 202 text note), so FastAPI serializes nothing; the
    # decorator now states the 204 for the contract's `excluded` entry (FR-046.9).
    "DELETE /api/v1/extractions/{extraction_id}": {"response_model": None, "status_code": 204},
}


def _name(tp) -> str | None:
    if tp is None:
        return None
    if isinstance(tp, type):
        return f"{tp.__module__}.{tp.__qualname__}"
    return repr(tp).replace("typing.", "")


def _served() -> dict:
    from fastapi.routing import APIRoute, iter_route_contexts

    from src.main import app

    out = {}
    for ctx in iter_route_contexts(app.routes):
        route = ctx.original_route
        if not isinstance(route, APIRoute) or not ctx.include_in_schema:
            continue
        for method in sorted(ctx.methods or ()):
            key = f"{method} {re.sub(r'{(\w+):\w+}', r'{\1}', str(ctx.path))}"
            out[key] = {"response_model": _name(route.response_model), "status_code": route.status_code}
    return out


def _t1() -> set[str]:
    """The 146 T1 keys: the AST record's where this checkout has it, else the fixture's own (the public mirror strips
    `0xcc/`, and its tree must still pass; `test_the_fixture_names_the_ast_records_routes` ties the two privately)."""
    record = REPO / "0xcc" / "reviews" / "046_endpoint_ast.json"
    if record.is_file():
        return set(json.loads(record.read_text()))
    return set(json.loads(RECORD.read_text()))


def test_the_fixture_names_the_ast_records_routes():
    record = REPO / "0xcc" / "reviews" / "046_endpoint_ast.json"
    if not record.is_file():
        pytest.skip("no 0xcc/ in this checkout (the public mirror): the fixture's 146 keys stand alone")
    assert set(json.loads(RECORD.read_text())) == set(json.loads(record.read_text()))
    assert len(json.loads(RECORD.read_text())) == 146


def test_every_t1_route_keeps_the_serializer_it_had():
    recorded = json.loads(RECORD.read_text())
    assert set(recorded) == _t1(), sorted(set(recorded) ^ _t1())
    served = _served()
    moved = {k: {"main": recorded[k], "now": served.get(k)} for k in sorted(recorded)
             if served.get(k) != EXCEPTIONS.get(k, recorded[k])}
    assert not moved, moved
    assert all(recorded[k] != v for k, v in EXCEPTIONS.items()), "an exception no longer differs: remove it"


def test_every_t1_route_documents_a_typed_answer():
    from src.api import contract_tools as tools
    from src.api.answers_contract import answers_contract

    doc = answers_contract()  # generated now from the live app (the pin test holds the committed file to it)
    ext = frozenset(doc["external"])
    untyped = []
    for key in sorted(_t1()):
        if key in doc["excluded"]:
            # only a T1 route with no JSON answer is excluded: no body, or the extraction delete's 202 text note
            assert "application/json" not in doc["excluded"][key]["statuses"].values(), key
            continue
        entry = doc["responses"].get(key)
        statuses = [s for st, s in (entry or {}).items() if st[:1] == "2"]
        if not statuses or not all(tools.is_typed(s, doc["$defs"], ext) for s in statuses):
            untyped.append(key)
    assert not untyped, untyped
