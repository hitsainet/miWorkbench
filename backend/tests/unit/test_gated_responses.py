"""Covered gated routes declare the pending `202` (046 FR-046.7, MW13, TD1; FTASKS 8.11 and 7.2).

- **The guard:** the set of covered routes whose SERVED operation declares a `202` reaching `PendingApprovalAnswer`
  EQUALS `gated_routes(app)` ∩ covered (41 on 2026-11-26).
- **The status:** a route whose own success is `202` declares the union of its own answer and the pending one there
  (`oneOf` for a typed model; the untyped stand-in `anyOf` until its model lands); any other route declares the
  pending answer alone. So the `status_code=` each decorator passes is the route's own.
- **Each covered gated route, called as an agent** (`X-MiStudio-Agent` outside `MCP_POD_AGENTS`, IC-4) with a body its
  own parsing accepts: `202`, and the answer middleware CHECKED that answer against the `202` entry, through the
  gate's exception handler (FTID §16's unverified item: `scope["endpoint"]` is set there). The operator call, the
  route's own status, is each router's HTTP case (FTASKS 9.2).
- **7.2's proof:** with the `202` entry taken out of a scratch copy of the contract, the same agent call is red.

Controls: M6 (one decorator's `gated_responses` removed), M7 (`gated_responses` returns `{}`), M8 (a bare `Union` for
an own-`202` route), M5 (the middleware skips exception-handler answers).
"""
from __future__ import annotations

import copy
import functools
import json

import httpx
import pytest

from tests.support.gate_fixtures import DEFERRED, DESIGN, sample_request

AGENT = {"X-MiStudio-Agent": "miassist"}
PENDING = "#/components/schemas/PendingApprovalAnswer"


def _app():
    from src.main import app

    return app


@functools.lru_cache(maxsize=1)
def _covered() -> frozenset[str]:
    from src.api.answers_contract import covered_keys, live_tool_names

    return covered_keys(live_tool_names())


def _covered_gated() -> dict[str, str]:
    from src.core.approval_gate import gated_routes

    return {f"{m} {p}": a for (m, p), a in gated_routes(_app()).items() if f"{m} {p}" in _covered()}


def _members(schema: dict, components: dict) -> list[str]:
    """The component names a response schema reaches at its top: itself, or its union's members."""
    ref = schema.get("$ref", "")
    if ref.startswith("#/components/schemas/"):
        target = components[ref.rsplit("/", 1)[1]]
        if "oneOf" in target or "anyOf" in target:
            return [m.get("$ref", json.dumps(m, sort_keys=True)) for m in target.get("oneOf", target.get("anyOf"))]
        return [ref]
    return [json.dumps(schema, sort_keys=True)]


def _declared_202s() -> dict[str, list[str]]:
    spec = _app().openapi()
    components = spec["components"]["schemas"]
    out = {}
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            content = ((op.get("responses") or {}).get("202") or {}).get("content", {}).get("application/json")
            if content:
                out[f"{method.upper()} {path}"] = _members(content["schema"], components)
    return out


def test_the_covered_gated_routes_are_exactly_those_declaring_the_pending_202():
    declaring = {k for k, members in _declared_202s().items() if PENDING in members}
    gated = set(_covered_gated())
    assert declaring == gated, {"gated, no pending 202": sorted(gated - declaring),
                                "pending 202, not a covered gated route": sorted(declaring - gated)}
    assert len(gated) >= 41, len(gated)


def test_each_route_declares_the_pending_answer_at_its_own_status():
    from src.api.answers_contract import operations

    routes = {f"{m} {p}": r for m, p, r in operations(_app())}
    declared = _declared_202s()
    wrong = {}
    for key in _covered_gated():
        own_202 = (routes[key].status_code or 200) == 202
        members = declared[key]
        if own_202 and not (len(members) == 2 and PENDING in members):
            wrong[key] = ("own 202: the union of its answer and the pending one", members)
        if not own_202 and members != [PENDING]:
            wrong[key] = ("the pending answer alone", members)
    assert not wrong, wrong


def test_a_typed_own_202_union_is_exclusive():
    """FR-046.7: where a route's own success is also `202`, the declaration is a `oneOf` told apart by the required
    `approval_request_id`: a pending body is not the route's own answer, and the reverse (M8)."""
    from jsonschema import Draft202012Validator

    from src.core.approval_gate import PendingApproval, gated_responses, pending_approval_body
    from src.schemas.model import ModelResponse

    union = gated_responses(ModelResponse, status_code=202)[202]["model"]
    schema = union.model_json_schema()
    assert "oneOf" in schema and len(schema["oneOf"]) == 2
    pending = pending_approval_body(PendingApproval("apr_1", "download_model", {"_method": "POST", "_route": "/x"}))
    own = ModelResponse(id="m_1", name="m", architecture="llama", params_count=1, quantization="FP16",
                        status="ready", created_at="2026-11-26T00:00:00Z",
                        updated_at="2026-11-26T00:00:00Z").model_dump(mode="json")
    validator = Draft202012Validator(schema)
    assert validator.is_valid(pending) and validator.is_valid(own)
    defs = schema["$defs"]
    branch = {name: Draft202012Validator({"$ref": f"#/$defs/{name}", "$defs": defs})
              for name in ("PendingApprovalAnswer", "ModelResponse")}
    assert branch["PendingApprovalAnswer"].is_valid(pending) and not branch["ModelResponse"].is_valid(pending)
    assert branch["ModelResponse"].is_valid(own) and not branch["PendingApprovalAnswer"].is_valid(own)


def _row_of(key: str):
    method, path = key.split(" ", 1)
    rows = [r for r, (template, _a) in DESIGN.items() if template == path and r not in DEFERRED]
    return rows[0] if rows else None


def _gated_keys() -> list[str]:
    """The covered gated keys, from the committed contract (`x-approval-action`), so parametrizing needs no app."""
    from src.api.answers_contract import DEFAULT_PATH

    responses = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))["responses"]
    return sorted(k for k, entry in responses.items() if "x-approval-action" in entry)


GATED_KEYS = _gated_keys()


@pytest.fixture
def no_dispatch(monkeypatch):
    """Every dispatch primitive raises: a gated agent call must reach none of them."""
    from celery.app.task import Task
    from fastapi import BackgroundTasks

    def refuse(*args, **kwargs):
        raise AssertionError("a dispatch primitive was reached on a gated agent call")

    monkeypatch.setattr(Task, "apply_async", refuse)
    monkeypatch.setattr(Task, "delay", refuse)
    monkeypatch.setattr(BackgroundTasks, "add_task", refuse)


async def _agent_call(key: str) -> httpx.Response:
    row = _row_of(key)
    if row is None:
        pytest.fail(f"{key} has no row in tests/support/gate_fixtures.DESIGN")
    req = sample_request(_app(), row)
    transport = httpx.ASGITransport(app=_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.post(req["path"], json=req["body"], params=req["query"], headers=AGENT)


@pytest.mark.parametrize("key", GATED_KEYS)
async def test_an_agent_call_answers_the_checked_pending_202(key, no_dispatch, async_session):
    import tests.support.answer_middleware as middleware

    before = middleware.CHECKED.get(key, 0)
    response = await _agent_call(key)
    assert response.status_code == 202, (key, response.status_code, response.text[:300])
    assert response.json()["status"] == "pending_approval"
    assert middleware.CHECKED.get(key, 0) == before + 1, f"{key}: the middleware did not check the gate's 202"


async def test_without_the_202_entry_the_agent_call_is_red(no_dispatch, async_session, monkeypatch):
    """FTASKS 7.2: the gate's handler-built `202` is checked against the contract's `202` entry; a scratch copy of the
    contract without that entry turns the same call red."""
    import tests.support.answer_middleware as middleware

    key = "POST /api/v1/labeling/panel"
    doc = copy.deepcopy(middleware.contract().contract)
    doc["responses"][key].pop("202")
    monkeypatch.setitem(middleware._STATE, "contract", middleware.Contract(doc))  # noqa: SLF001
    middleware.FAILURES.clear()
    response = await _agent_call(key)
    assert response.status_code == 202
    failures, middleware.FAILURES[:] = list(middleware.FAILURES), []
    assert [f.key for f in failures] == [key] and "status 202 is not declared" in str(failures[0]), failures
