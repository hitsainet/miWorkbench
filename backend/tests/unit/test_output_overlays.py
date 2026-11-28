"""The declared overlays (`scripts/output_overlays.py`; 046 WT7, MW5, FR-046.20).

- **Set equality, both ways:** the tools that build or change their answer (`route_map.returns_unchanged` is `None`)
  EQUAL the overlay rows plus the tools with a fixed reason (the guide, the miForge import, miDataworks' relays and
  `register_probe_dataset`). A tool that starts building its answer without a row is red; so is a stale row.
- **Behaviour:** every row's tool is driven through the SDK client with EVERY route it calls answering a full,
  contract-shaped body (miServer's management routes in their envelope), so the real wrapper code runs. The answer
  must validate against the advertised schema (the client raises otherwise), and every key it carries must be
  explained: the base route's own, the result language's, the shared legacy keys, or a field of the row's wrapper or
  variants. A wrapper that stops declaring a key the tool adds is red. Where the tool answers its base route, the
  wrapper's required fields must be there too.
- `miserver_get_model`, driven with a downloading row that reports no progress through miServer's real envelope,
  carries `progress_note` (a latent defect fixed in 046 P2: the note was never added).

Controls (0xcc/reviews/046_controls_2026-11-25.md):
  M26  delete the `miserver_get_model` row                     → test_every_building_tool_has_a_row_and_no_row_is_stale red
  M27  delete `progress_note` from the `ProgressNote` wrapper   → test_each_overlay_explains_every_key_of_its_answer red
"""
from __future__ import annotations

import functools
import json
from typing import Any

import pytest

from tests.support import mcp_drive
from tests.support.answer_samples import sample
from tests.support.miserver_fakes import approval_actions, envelope

RESULT_LANGUAGE = {"approval", "next_steps", "next"}


@functools.lru_cache(maxsize=1)
def _sources():
    from scripts.generate_output_schemas import miserver_source, native_source

    return {"miworkbench": native_source(), "miserver": miserver_source()}


@functools.lru_cache(maxsize=1)
def _tools():
    from scripts.generate_output_schemas import live_tools

    return live_tools()


@functools.lru_cache(maxsize=1)
def _schemas():
    from src.mcp_server.tool_classes import load_output_schemas

    return load_output_schemas()


def _overlays():
    from scripts.output_overlays import OVERLAYS

    return OVERLAYS


def test_every_building_tool_has_a_row_and_no_row_is_stale():
    from scripts.generate_output_schemas import FIXED_REASONS
    from src.mcp_server.route_map import returns_unchanged, tool_routes

    building = {n for n, t in _tools().items() if returns_unchanged(t.fn)[0] is None}
    fixed = set(FIXED_REASONS) | {n for n in _tools() if any(p == "dataworks" for p, _m, _x in tool_routes()[n])}
    rows = set(_overlays())
    assert building - fixed == rows, {"building, no row": sorted(building - fixed - rows),
                                      "row, not building": sorted(rows - (building - fixed))}
    assert len(rows) >= 60, len(rows)


def test_each_row_names_real_routes_and_code():
    """A row's base routes are the tool's own (the walker sees it call them), and its `code` imports."""
    import importlib

    from src.mcp_server.route_map import normalise, tool_routes

    for name, row in _overlays().items():
        called = {f"{m} {'/api/v1' if p == 'miworkbench' else ''}{normalise(x)}" for p, m, x in tool_routes()[name]}
        assert set(row.base) <= called, (name, sorted(set(row.base) - called))
        if not row.code.startswith("tool:"):
            module, function = row.code.split(":")
            assert callable(getattr(importlib.import_module(module), function)), row.code
        else:
            assert row.code == f"tool:{name}", row.code


# ── behaviour ────────────────────────────────────────────────────────────────────────────────────────────────


def _root(product: str) -> dict:
    source = _sources()[product]
    return {"$defs": source.defs}


def _route_body(product: str, method: str, path: str) -> tuple[int, Any]:
    """A full, contract-shaped answer of one route as it travels (miServer's envelope included)."""
    if product not in _sources():
        return 200, {"k": "v"}  # a product with no published contract (miForge): any object
    source = _sources()[product]
    if source.key(method, path) in source.empty:
        return 204, ""  # no body (the client answers `{}`)
    if source.key(method, path) in source.excluded and path.endswith("/content"):
        return 200, '{"custom_id": "a", "response": {}}\n{"custom_id": "b", "response": {}}\n'  # JSON Lines
    entry = source.entry(method, path) or {}
    statuses = sorted(s for s in entry if s != "202") or sorted(entry)
    if not statuses:
        return 200, {"k": "v"}  # a route with no typed answer (046-2 types it): any object
    schema = entry[statuses[0]]
    if product == "miserver" and schema.get("$ref", "").startswith("#/$defs/"):
        target = source.defs[schema["$ref"].split("/")[-1]]
        if {"success", "data"} <= set(target.get("properties", {})):
            data = target["properties"]["data"]
            options = [a for a in data.get("anyOf", [data]) if a.get("type") != "null"]
            return int(statuses[0]), envelope(sample(options[0], _root(product)) if options else None)
    if product == "miserver" and schema.get("$ref", "").startswith("http"):
        from scripts.generate_output_schemas import _documents

        document = json.loads(_documents()[schema["$ref"]].read_text())
        return int(statuses[0]), sample(document, document)
    body = sample(schema, _root(product))
    return int(statuses[0]), {"k": "v"} if body in (None, {}) else body


#: Row-specific answers that reach the wrapper's keys (route → (status, body)).
def _overrides(name: str) -> dict[tuple[str, str, str], tuple[int, Any]]:
    if name == "miserver_get_model":
        status, body = _route_body("miserver", "GET", "/api/models/{}")
        row = {**body["data"], "status": "downloading", "download_progress": None}
        return {("miserver", "GET", "/api/models/{}"): (status, envelope(row))}
    if name == "miserver_delete_circuit":  # the serving check cannot read: the delete runs and says so
        return {("miserver", "GET", "/api/circuits/active"): (500, {"detail": "down"})}
    return {}


_DRIVE_EXTRA = {"miserver_delete_circuit": {"acknowledge_serving": False}}


@pytest.fixture(scope="module")
def world():
    """ONE backend and every category's server for the module (building them per row is most of the cost); each row
    resets the backend's routes and records."""
    with pytest.MonkeyPatch.context() as mp:
        backend = mcp_drive.RouteBackend()
        with mcp_drive.patched_routes(mp, backend):
            servers = mcp_drive.build_every_category()
            owner = {t.name: m for m in servers.values() for t in m._tool_manager.list_tools()}  # noqa: SLF001
            wire = {t.name: t for m in servers.values() for t in mcp_drive.wire_tools(m)}
            yield backend, owner, wire
            backend.unmapped.clear()  # each row asserts its own


def _drive_one(name: str, row, world) -> tuple[Any, dict]:
    from scripts.generate_output_schemas import _key
    from src.mcp_server.route_map import tool_routes

    backend, owner, wire = world
    backend.routes.clear(), backend.calls.clear(), backend.unmapped.clear()
    backend.answer("GET", "/api/approvals/actions", envelope(approval_actions(_all_actions())))
    bases: dict[str, Any] = {}
    overrides = _overrides(name)
    for product, method, path in sorted(tool_routes()[name]):
        if product == "dataworks":
            continue
        status, body = overrides.get((product, method, path)) or _route_body(product, method, path)
        if (product, method, path) not in overrides and product == "miserver" and method == "GET" \
                and path.endswith("/circuits/active"):
            body = envelope([])  # nothing is serving
        full = ("/api/v1" if product == "miworkbench" else "") + path
        headers = {"content-type": "application/jsonl"} if isinstance(body, str) and body else None
        backend.answer(method, full, body, status=status, headers=headers)
        bases[_key(product, method, path)] = body.get("data") if isinstance(body, dict) and "success" in body else body
    args = {**row.drive_args, **_DRIVE_EXTRA.get(name, {})}
    if any(p == "miworkbench" and m == "POST" and x == "/mcp/approvals" for p, m, x in tool_routes()[name]):
        args.setdefault("approval_request_id", "apr_1")
    arguments = {**mcp_drive.sample_arguments(wire[name]), **args}
    # The steering tools' in-flight guardrail is module state: a task an earlier test started and never polled
    # holds a slot here, and this drive would read the guardrail's refusal instead of its route's answer.
    from src.mcp_server.tools import steering as steering_tools

    steering_tools._inflight.clear()  # noqa: SLF001
    result = mcp_drive.call(owner[name], name, arguments)
    assert not backend.unmapped, (name, sorted(set(backend.unmapped)))
    return result, bases


def _all_actions() -> list[str]:
    from src.mcp_server.tools.miserver_approvals import RELAYED_ACTIONS

    return sorted({*RELAYED_ACTIONS.values(), "model_context_write"})


def _fields(model) -> set[str]:
    return set(model.model_fields) if model is not None else set()


def _required(model) -> set[str]:
    return {k for k, f in model.model_fields.items() if f.is_required()} if model is not None else set()


@pytest.mark.parametrize("name", sorted(__import__("scripts.output_overlays", fromlist=["OVERLAYS"]).OVERLAYS))
def test_each_overlay_explains_every_key_of_its_answer(name, world):
    from scripts.generate_output_schemas import SHARED

    row = _overlays()[name]
    result, bases = _drive_one(name, row, world)  # the SDK client raises on an answer its schema refuses
    assert not result.is_error, (name, result.content[0].text[:400])
    answer = result.structured_content
    base_keys: set[str] = set()
    reached = None
    for key in row.base:
        body = bases.get(key)
        if isinstance(body, dict):
            base_keys |= set(body)
            if row.nest is None and set(body) <= set(answer) and all(answer[k] == body[k] for k in body):
                reached = key
        if row.nest is not None and answer.get(row.nest) == body:
            reached = key
        if isinstance(body, list) or body is None:
            if answer.get("result", ...) == body:
                reached = key
        if body == "" and not set(answer) - RESULT_LANGUAGE:
            reached = key  # a 204, answered as `{}`
    if row.nest is not None:
        base_keys = {row.nest}
    explained = base_keys | RESULT_LANGUAGE | set(SHARED) | _fields(row.wrapper) | {"result", "message", "error"}
    for model in row.variants:
        explained |= _fields(model)
    unexplained = sorted(set(answer) - explained)
    assert not unexplained, (name, unexplained, sorted(answer))
    if row.base and not row.variants:
        assert reached is not None, (name, "the drive did not reach the base route's answer", sorted(answer))
    if reached is not None and row.wrapper is not None:
        assert _required(row.wrapper) <= set(answer), (name, sorted(_required(row.wrapper) - set(answer)))
        # review round 1, finding 4 (O6, O7): the schema REQUIRES what the wrapper always adds (and, for a nested
        # answer, the body's key): the same answer without one of them is refused.
        from tests.unit.test_mcp_typed_output_schemas import ALTERNATIVE_KEYS, _validator

        schema = _schemas()[name] if name in _schemas() else None
        if schema is not None and not ALTERNATIVE_KEYS & set(answer):
            for key in sorted(_required(row.wrapper) | ({row.nest} if row.nest else set())):
                cut = {k: v for k, v in answer.items() if k != key}
                assert list(_validator(schema).iter_errors(cut)), (name, f"accepts the answer without {key!r}")
    if row.alternatives_only and reached is not None:
        # review round 1, finding 6: an alternatives-only row adds nothing but the result language to its base answer
        # (a list, scalar or null body is structured as `result`)
        assert not set(answer) - base_keys - RESULT_LANGUAGE - {"result"}, (name, sorted(set(answer) - base_keys))
    if not row.base:  # a wholly built answer: one of its variants, whole
        assert any(_required(m) <= set(answer) for m in row.variants), (name, sorted(answer))


def test_the_model_note_reaches_the_agent_through_the_real_envelope(world):
    """046 P2: `miserver_get_model` read `data` off an answer the client had already unwrapped, so `progress_note`
    was never added on a real miServer; only a fake answering `{"data": …}` with no envelope reached the wrapper."""
    result, _ = _drive_one("miserver_get_model", _overlays()["miserver_get_model"], world)
    assert "UNKNOWN, not zero" in result.structured_content.get("progress_note", ""), result.structured_content
