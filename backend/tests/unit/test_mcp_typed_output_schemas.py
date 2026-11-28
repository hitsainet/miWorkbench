"""Typed output schemas generated from contracts (045 follow-up RL-1; 046 FR-046.18 to FR-046.23, FR-046.27).

A tool advertises the answer of the routes it relays, typed from each route's contract entry (miWorkbench's live
OpenAPI through the writer's transforms until 046-2; miServer's vendored `miserver-answers-v1.json`), as alternatives
at the root beside the shared non-error answers (MW4). `required` means "always sent" (SH7). Every other tool keeps
the generic envelope and is listed in the file's `untyped` with a reason from the closed list (FR-046.27).

- The committed file is what the generator writes now (from the live registry, the OpenAPI and the vendored contract).
- Membership is the registry's, both ways; every miServer relay is typed; coverage and sources are stated and recount.
- Every typed schema: `required` on the answer, the alternatives present, `next` never constrained, no closed object
  outside the result language's own `approval` and `next_steps`, every reference local and resolved, no unused
  definition, no Pydantic `title`; a pending, a relayed pending (where the tool's routes answer one), an
  `unavailable`, a `refused` and an argument `error` answer each validate.
- The BUILT server advertises them, read through the SDK client (`tools/list`, as an agent reads it).
- Honesty, through the SDK client (which validates `structuredContent` against the advertised schema): each tool that
  answers its route unchanged, driven with the route's full body (an envelope for miServer's management routes, which
  the client unwraps), answers exactly that body; the same body with one REQUIRED field removed, or one scalar of the
  wrong type, is refused.

Controls (0xcc/reviews/046_controls_2026-11-25.md; 045's are in 045_followup_controls_2026-11-24.md):
  M23  the generator drops `required` from the answer part      → test_a_removed_required_field_is_refused red
  M24  the generator drops the `unavailable` alternative        → test_every_schema_is_strict_where_guaranteed… red
  M25  the generator constrains `next`                          → test_every_schema_is_strict_where_guaranteed… red
  M28  the generator skips miServer's envelope unwrap           → test_a_typed_tool_answers_its_routes_body_unchanged red
  M29  the nullable empty variant becomes `{"type": "object"}`  → test_a_nullable_route_answers_the_empty_object red
  M31  `apply_output_schemas` ignores the typed file            → test_the_server_advertises_the_typed_schemas red
  the approval `status` enum widened to any string              → test_an_approval_status_outside_the_enum_is_refused red
  `mismatch` dropped from the enum                              → test_the_tool_side_approval_status_is_the_enum… red
"""
from __future__ import annotations

import copy
import functools
import json
from typing import Any

import jsonschema
import pytest
from referencing import Registry

from tests.support import mcp_drive
from tests.support.answer_samples import sample

PENDING = {
    "approval_request_id": "apr_1", "status": "pending_approval", "hint": "h",
    "approval": {"id": "apr_1", "status": "pending", "decide_in": "miworkbench", "then": "call_again_with",
                 "summary": "x", "call_again_with": {"tool": "t", "args": {}}},
    "next_steps": [{"tool": "get_approval_status", "args": {"approval_request_id": "apr_1"}, "why": "poll"}],
}
RECALL_REFUSED = {"approval_request_id": "apr_1", "status": "not_found", "hint": "h"}
RELAYED_PENDING = {"approval_id": "apr_" + "0" * 32, "status": "pending", "action": "model_load", "decide_in": "miserver",
                   "summary": {"Model": "m"}, "expires_at": "2026-11-25T00:00:00Z", "hint": "h",
                   "request_digest": "sha256:" + "0" * 64}
UNAVAILABLE = {"unavailable": "miworkbench", "reason": "down"}
REFUSED = {"refused": "x", "reason": {"detail": "y"}}
ERROR = {"error": "bad_argument", "message": "m"}
STEP = {"tool": "get_task_status", "args": {}, "why": "see"}
NEXTS = ("list_models", STEP, [STEP], None, {"anything": 1})
ALTERNATIVE_KEYS = {"unavailable", "refused", "error", "approval_request_id", "approval_id"}


def _no_fetch(uri: str):
    raise LookupError(f"a schema reference left the document: {uri}")


def _validator(schema: dict) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(schema, registry=Registry(retrieve=_no_fetch))


@functools.lru_cache(maxsize=1)
def _document() -> dict[str, Any]:
    from src.mcp_server.tool_classes import OUTPUT_SCHEMAS_PATH

    return json.loads(OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8"))


def _typed() -> list[str]:
    return sorted(_document()["tools"])


@functools.lru_cache(maxsize=1)
def _wire() -> dict[str, Any]:
    """`tools/list` of ONE server with every category (the deployed shape), through the SDK client."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    mcp, _ = build_server(MCPSettings(tool_categories=",".join(sorted(all_categories())), allow_anonymous=True,
                                      _env_file=None, **mcp_drive.placeholder_overrides()), stdio=True)
    return {t.name: t for t in mcp_drive.wire_tools(mcp)}


@functools.lru_cache(maxsize=1)
def _unchanged() -> dict[str, frozenset]:
    """Tools whose every return is one route's body (`route_map.returns_unchanged`), typed."""
    from scripts.generate_output_schemas import live_tools
    from src.mcp_server.route_map import returns_unchanged

    tools = live_tools()
    out = {}
    for name in _typed():
        routes, _ = returns_unchanged(tools[name].fn)
        if routes is not None:
            out[name] = frozenset(routes)
    return out


def _walk(node: Any, path: tuple = ()):
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, path + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, path + (index,))


@functools.lru_cache(maxsize=1)
def _alternatives() -> list[dict]:
    from scripts.generate_output_schemas import _relayed_pending, alternatives
    from src.mcp_server.result_language import load_contract

    return alternatives(load_contract(), _relayed_pending())


def _answer_parts(schema: dict) -> list[dict]:
    """The route parts of the root `anyOf`: everything but the shared alternatives."""
    return [p for p in schema["anyOf"] if p not in _alternatives()]


def _is_pending_part(part: dict) -> bool:
    """A gated route's pending `202` (`PendingApprovalAnswer`, 046 FTASKS 8.11), inlined beside its own answer."""
    return {"approval_request_id", "retry_with_header"} <= set(part.get("required") or [])  # `approval` is root-level


def _body_part(schema: dict) -> dict:
    from scripts.generate_output_schemas import EMPTY_OBJECT

    parts = _answer_parts(schema)
    return next((p for p in parts if p != EMPTY_OBJECT and p.get("properties", {}).get("result") != {"type": "null"}
                 and not _is_pending_part(p)), parts[0])


def _body(name: str) -> Any:
    """The full body of the tool's route (every property filled), as the TOOL answers it."""
    from scripts.generate_output_schemas import SHARED

    schema = _document()["tools"][name]
    part = copy.deepcopy(_body_part(schema))
    props = part.get("properties", {})
    if set(props) == {"result"} and part.get("required") == ["result"]:
        return {"result": sample(props["result"], schema)}  # a list, scalar or null answer
    if props:
        part["properties"] = {k: v for k, v in props.items() if k not in SHARED or k in part.get("required", [])}
    return route_complete(name, sample(part, schema))


def route_complete(name: str, body: Any) -> Any:
    """`body` with every key its miWorkbench route always sends filled in, as the route sends it (046 P7). The tool
    schema leaves `next` unconstrained (A30) and so out of a sample, while a route that always sends a `next` (the
    nonsense-text set's next step, say) declares it required in the contract, which the fake check holds a fake to."""
    from tests.support.fake_clients import workbench_answer

    from tests.support.fake_clients import _contract, _entry

    if not isinstance(body, dict) or set(body) == {"result"}:
        return body
    defs = _contract()["$defs"]
    for product, method, path in sorted(_unchanged().get(name, ())):
        full = path if path.startswith("/api/") else "/api/v1" + path
        entry = _entry(method, full) if product == "miworkbench" else None
        for schema in (entry or {}).values():
            ref = schema.get("$ref", "") if isinstance(schema, dict) else ""
            target = defs.get(ref.split("/")[-1], {}) if ref.startswith("#/$defs/") else {}
            if target.get("properties") and set(body) <= set(target["properties"]):
                return workbench_answer(method, full, body)  # the route whose answer the body is
    return body


# ── the generated file ──────────────────────────────────────────────────────────────────────────────────────────


def test_the_committed_schemas_are_the_generated_ones():
    from scripts.generate_output_schemas import SCHEMAS_PATH, document_text

    assert SCHEMAS_PATH.read_text(encoding="utf-8") == document_text(), (
        "output_schemas.json is stale or was edited by hand: cd backend && python -m scripts.generate_output_schemas "
        "--write")


def test_every_registered_tool_is_typed_or_listed_with_its_reason():
    from scripts.generate_output_schemas import MISERVER_CONTRACT, NATIVE_CONTRACT, R_DATAWORKS, REASONS

    document = _document()
    typed, untyped = set(document["tools"]), set(document["untyped"])
    assert not typed & untyped, sorted(typed & untyped)
    live = set(_wire())
    assert typed | untyped == live, {"registered, not in the file": sorted(live - typed - untyped),
                                     "in the file, not registered": sorted((typed | untyped) - live)}
    assert len(typed) >= 180, len(typed)  # a generator that typed nothing would pass the rest
    unlisted = {n: e["reason"] for n, e in document["untyped"].items() if e["reason"] not in REASONS}
    assert not unlisted, unlisted
    # Every miServer relay is typed from the vendored contract (FR-046.26); the miDataworks ones wait (BR-6).
    assert not sorted(n for n in untyped if n.startswith("miserver_"))
    assert all(document["untyped"][n]["reason"] == R_DATAWORKS for n in untyped if n.startswith("dataworks_"))
    # 046-1's interim reason (a native route untyped in the live OpenAPI) is retired at 046-2 (FTASKS 10.1): no native
    # tool waits; only the guide, the miForge call and miDataworks' relays stay generic, besides `result: none`.
    assert {e["reason"] for e in document["untyped"].values()} <= set(REASONS)
    assert not [n for n, e in document["untyped"].items() if "046-2" in e["reason"]]
    assert document["coverage"] == {"typed": len(typed), "generic": len(untyped)}
    vendored = json.loads(MISERVER_CONTRACT.read_text(encoding="utf-8"))
    native = json.loads(NATIVE_CONTRACT.read_text(encoding="utf-8"))
    assert document["sources"] == {MISERVER_CONTRACT.name: vendored["content_sha256"],
                                   NATIVE_CONTRACT.name: native["content_sha256"]}


def test_the_generic_dataworks_set_is_pinned():
    """FTASKS 2.7 / M37: until 046-3 vendors miDataworks' contract, exactly the 19 `dataworks_*` relays and
    `register_probe_dataset` (whose answer carries miDataworks' manifest read) are generic for that reason."""
    from scripts.generate_output_schemas import R_DATAWORKS

    document = _document()
    waiting = sorted(n for n, e in document["untyped"].items() if e["reason"] == R_DATAWORKS)
    relays = sorted(n for n in _wire() if n.startswith("dataworks_"))
    assert len(relays) == 19, relays
    assert waiting == sorted([*relays, "register_probe_dataset"])


@pytest.mark.parametrize("name", _typed())
def test_every_schema_is_strict_where_guaranteed_and_never_constrains_next(name):
    """FR-046.19, FR-046.22: the shape of every typed schema, and the non-error answers each validates."""
    from src.mcp_server.result_language import load_contract
    from src.mcp_server.tool_classes import output_schema

    own = _document()["tools"][name]
    assert own["type"] == "object" and isinstance(own["anyOf"], list), name
    parts = _answer_parts(own)
    assert parts, name
    # a part requires a field, or is the empty object a no-body route answers (`maxProperties: 0`, finding 7)
    assert any(p.get("required") or "$ref" in p or p.get("maxProperties") == 0 for p in parts), \
        f"{name}: no answer part requires a field"
    for part in parts:
        assert "next" not in part.get("properties", {}), f"{name} constrains the legacy next"
    defs = own.get("$defs", {})
    used = set()
    for path, node in _walk(own):
        if not isinstance(node, dict):
            continue
        inside_rl = path[:2] in (("properties", "approval"), ("properties", "next_steps"))
        inside_document = len(path) >= 2 and path[0] == "$defs" and "$id" in defs.get(path[1], {})
        if not inside_rl and not inside_document:
            assert node.get("additionalProperties") is not False, f"{name} closes an object at {path}"
        if path and path[-1] not in ("properties", "$defs", "patternProperties"):
            assert not isinstance(node.get("title"), str), f"{name} keeps a generated title at {path}"
        ref = node.get("$ref")
        if isinstance(ref, str) and not inside_document:
            assert ref.startswith("#/$defs/") and ref.split("/")[-1] in defs, f"{name}: unresolved {ref}"
            if not (len(path) >= 2 and path[0] == "$defs" and path[1] == ref.split("/")[-1]):
                used.add(ref.split("/")[-1])
    assert set(defs) == used, {"unused": sorted(set(defs) - used), "missing": sorted(used - set(defs))}
    advertised = output_schema(name, load_contract())
    assert advertised == own
    validator = _validator(advertised)
    answers = [PENDING, RECALL_REFUSED, UNAVAILABLE, REFUSED, ERROR]
    if any(p.get("allOf") == [{"$ref": "#/$defs/ApprovalPending"}] for p in own["anyOf"]):
        answers.append(RELAYED_PENDING)
    for answer in answers:
        errors = list(validator.iter_errors(answer))
        assert not errors, (name, answer, errors[0].message[:300])
    body = _body(name)
    for nxt in NEXTS if body != {} else ():  # a no-body route (the empty object, finding 7) never carries a `next`
        candidate = {**body, "next": nxt} if isinstance(body, dict) else body
        assert not list(validator.iter_errors(candidate)), (name, "next", nxt)


def _status_literals() -> set[str]:
    """Every string literal a tool module writes as `status` in a dict that also carries `approval_request_id`."""
    import ast
    import pathlib

    found: set[str] = set()
    tools = pathlib.Path(__file__).resolve().parents[2] / "src" / "mcp_server" / "tools"
    for path in sorted(tools.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Dict):
                continue
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            if "approval_request_id" in keys and "status" in keys:
                value = dict(zip([getattr(k, "value", None) for k in node.keys], node.values, strict=True))["status"]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    found.add(value.value)
    return found


def test_the_tool_side_approval_status_is_the_enum_the_code_sends():
    """Operator decision 2026-11-25 (046 question 4): the tool-side approval answer's `status` is an enum of what the
    code can send, never any string: every literal the tool modules write beside `approval_request_id`, and every
    stored request status a refused re-call's 409 passes through (`ApprovalStatus`)."""
    from scripts.generate_output_schemas import tool_approval_statuses
    from src.models.agent_approval import ApprovalStatus

    literals = _status_literals()
    assert {"pending_approval", "not_found", "mismatch"} <= literals, literals  # the scan reads the tool modules
    assert literals <= set(tool_approval_statuses()), sorted(literals - set(tool_approval_statuses()))
    assert {s.value for s in ApprovalStatus} <= set(tool_approval_statuses())
    alternative = next(a for a in _alternatives() if "approval_request_id" in a.get("required", []))
    assert alternative["properties"]["status"] == {"enum": list(tool_approval_statuses())}


@pytest.mark.parametrize("name", _typed())
def test_an_approval_status_outside_the_enum_is_refused(name):
    """The enum bites: a tool-side approval answer with a status the code never sends validates for no typed tool."""
    validator = _validator(_document()["tools"][name])
    assert list(validator.iter_errors({"approval_request_id": "apr_1", "status": "queued", "hint": "h"})), name
    assert not list(validator.iter_errors(RECALL_REFUSED)), name


def test_the_relayed_pending_is_offered_where_a_route_answers_it():
    """FR-046.26: miServer's pending answer is an alternative exactly for the relays of its gated routes."""
    document = _document()
    offering = sorted(n for n, s in document["tools"].items()
                      if any(p.get("allOf") == [{"$ref": "#/$defs/ApprovalPending"}] for p in s["anyOf"]))
    from src.mcp_server.tools.miserver_approvals import RELAYED_ACTIONS

    assert set(RELAYED_ACTIONS) <= set(offering), sorted(set(RELAYED_ACTIONS) - set(offering))
    assert all(n.startswith("miserver_") for n in offering), offering


def test_each_answer_field_keeps_its_description():
    """WT6 drops Pydantic's titles and the result language's repeated descriptions, never a field's own."""
    document = _document()
    described = sum(1 for s in document["tools"].values() for _p, n in _walk(s.get("$defs", {}))
                    if isinstance(n, dict) and isinstance(n.get("description"), str))
    assert described > 1000, described
    from src.mcp_server.result_language import load_contract
    from src.mcp_server.tool_classes import result_language_properties

    rl = result_language_properties(load_contract())
    assert not [p for p, n in _walk(rl) if isinstance(n, dict) and isinstance(n.get("description"), str)]


# ── the built server advertises them ────────────────────────────────────────────────────────────────────────────


def test_the_server_advertises_the_typed_schemas():
    from src.mcp_server.result_language import load_contract
    from src.mcp_server.tool_classes import TOOL_CLASSES, envelope_schema, output_schema

    contract = load_contract()
    envelope = envelope_schema("object", contract)
    wire = _wire()
    wrong = sorted(n for n, t in wire.items() if t.output_schema != output_schema(n, contract))
    assert not wrong, wrong[:10]
    advertised_typed = {n for n, t in wire.items() if t.output_schema != envelope_schema(TOOL_CLASSES[n].result,
                                                                                         contract)}
    assert advertised_typed == set(_typed()), {"typed, not advertised": sorted(set(_typed()) - advertised_typed),
                                               "advertised, not typed": sorted(advertised_typed - set(_typed()))}
    for name in _typed():
        schema = wire[name].output_schema
        assert schema == _document()["tools"][name], name
        assert schema["properties"]["approval"] == envelope["properties"]["approval"], name
        assert schema["properties"]["next_steps"] == envelope["properties"]["next_steps"], name


def test_a_stale_schema_file_stops_the_build():
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.tool_classes import apply_output_schemas

    with pytest.raises(RuntimeError, match="no_such_tool"):
        apply_output_schemas(MCPServer("t"), typed={"no_such_tool": {"type": "object", "properties": {}}})


def test_a_missing_schema_file_stops_the_build(monkeypatch, tmp_path):
    from src.mcp_server import tool_classes

    monkeypatch.setattr(tool_classes, "OUTPUT_SCHEMAS_PATH", tmp_path / "absent.json")
    tool_classes.load_output_schemas.cache_clear()
    try:
        with pytest.raises(FileNotFoundError):
            mcp_drive.build_category("read")
    finally:
        monkeypatch.undo()
        tool_classes.load_output_schemas.cache_clear()


# ── honesty: the route's own body, through the SDK client ───────────────────────────────────────────────────────


def _wire_body(name: str, body: Any) -> Any:
    """What the backend sends for `body`: miServer's management routes wrap it in their envelope (the client unwraps
    it, `miserver_client.py:221`); `/v1`, `raw_get` and miWorkbench's routes send it as it is."""
    from scripts.generate_output_schemas import miserver_source

    answer = body["result"] if isinstance(body, dict) and set(body) == {"result"} else body
    routes = _unchanged()[name]
    if not any(p == "miserver" for p, _m, _x in routes):
        return answer
    from tests.support.miserver_fakes import envelope_for

    source = miserver_source()
    keys = {(k.split(" ", 1)[0], _normalise_key(k)): k for k in source_keys()}
    for _product, method, path in routes:
        for status, schema in (source.entry(method, path) or {}).items():
            ref = schema.get("$ref", "")
            target = source.defs.get(ref.split("/")[-1], {}) if ref.startswith("#/$defs/") else {}
            if {"success", "data"} <= set(target.get("properties", {})):
                return envelope_for(keys[(method, path)], answer, status)
    return answer


def _normalise_key(key: str) -> str:
    from src.mcp_server.route_map import normalise

    return normalise(key.split(" ", 1)[1])


@functools.lru_cache(maxsize=1)
def source_keys() -> tuple[str, ...]:
    from scripts.generate_output_schemas import MISERVER_CONTRACT

    return tuple(json.loads(MISERVER_CONTRACT.read_text(encoding="utf-8"))["responses"])


def _drive(monkeypatch, cases: dict[str, Any], off_contract: str | None = None) -> dict[str, Any]:
    out = {}
    with mcp_drive.patched_http(monkeypatch, off_contract=off_contract) as backend:
        servers = mcp_drive.build_every_category()
        owner = {t.name: m for m in servers.values() for t in m._tool_manager.list_tools()}  # noqa: SLF001
        wire = {t.name: t for m in servers.values() for t in mcp_drive.wire_tools(m)}
        for name, sent in cases.items():
            backend.body, backend.requests[:] = sent, []
            try:
                out[name] = (mcp_drive.call(owner[name], name, mcp_drive.sample_arguments(wire[name])),
                             [r for r in backend.requests if not r.url.path.endswith("/health")])
            except Exception as exc:  # noqa: BLE001 - the client refusing an answer is what some cases expect
                out[name] = (exc, [])
    return out


def test_a_typed_tool_answers_its_routes_body_unchanged(monkeypatch):
    """Every tool that answers its route unchanged, driven with the route's full body: the SDK client accepts the
    answer and `structuredContent` is exactly that body (a list as `{"result": [...]}`), from ONE request."""
    bodies = {name: _body(name) for name in _unchanged()}
    driven = _drive(monkeypatch, {n: _wire_body(n, b) for n, b in bodies.items()})
    failures = {}
    for name, (result, sent) in driven.items():
        if isinstance(result, Exception):
            failures[name] = repr(result)[:300]
        elif result.is_error:
            failures[name] = result.content[0].text[:300]
        else:
            answer = {k: v for k, v in result.structured_content.items() if k not in ("next_steps", "approval")}
            if answer != bodies[name] or len(sent) != 1:
                failures[name] = {"requests": len(sent), "answer_keys": sorted(answer)[:8]}
    assert not failures, failures
    assert len(bodies) >= 120, len(bodies)
    assert sum(1 for n in bodies if n.startswith("miserver_")) >= 55


def _single_part(name: str) -> bool:
    """One answer part (with at most the empty object or `{"result": null}` beside it): another part cannot accept
    what this one refuses."""
    from scripts.generate_output_schemas import EMPTY_OBJECT

    parts = [p for p in _answer_parts(_document()["tools"][name])
             if p != EMPTY_OBJECT and p.get("properties", {}).get("result") != {"type": "null"}
             and not _is_pending_part(p)]
    return len(parts) == 1


def _droppable(name: str, body: dict) -> str | None:
    part = _body_part(_document()["tools"][name])
    if ALTERNATIVE_KEYS & set(body) or not _single_part(name):
        return None
    return next((k for k in sorted(part.get("required", [])) if k in body and k != "result"), None)


def test_a_removed_required_field_is_refused(monkeypatch):
    """FR-046.19 AC: the same full body with one REQUIRED field removed is refused by the SDK client."""
    cases = {}
    for name in _unchanged():
        body = _body(name)
        if isinstance(body, dict) and (key := _droppable(name, body)) is not None:
            cases[name] = (key, {k: v for k, v in body.items() if k != key})
    assert len(cases) >= 100, len(cases)
    driven = _drive(monkeypatch, {n: _wire_body(n, b) for n, (_k, b) in cases.items()},
                    off_contract="a required field removed, so the SDK client must refuse the answer")
    accepted = sorted((n, cases[n][0]) for n, (r, _s) in driven.items() if not isinstance(r, Exception))
    assert not accepted, accepted[:10]


def _a_scalar_property(name: str) -> tuple[str, Any] | None:
    """A top-level property of the route's own body with one concrete scalar type, and a value of another type."""
    from scripts.generate_output_schemas import SHARED

    part = _body_part(_document()["tools"][name])
    for key, sub in sorted(part.get("properties", {}).items()):
        if key in SHARED or key == "result" or not isinstance(sub, dict):
            continue
        if sub.get("type") in ("string", "integer", "number", "boolean") and "anyOf" not in sub:
            return key, {"wrong": True}
    return None


def test_a_wrong_typed_field_is_refused(monkeypatch):
    """The typing bites: for every unchanged object tool with a scalar property, the same body with that property of
    the wrong type is refused by the SDK client (an envelope-only schema would accept it)."""
    cases = {n: c for n in _unchanged() if isinstance(_body(n), dict) and _single_part(n)
             and not ALTERNATIVE_KEYS & set(_body(n)) and (c := _a_scalar_property(n)) is not None}
    assert len(cases) >= 70, len(cases)
    driven = _drive(monkeypatch, {n: _wire_body(n, {**_body(n), k: w}) for n, (k, w) in cases.items()},
                    off_contract="one field of the wrong type, so the SDK client must refuse the answer")
    accepted = sorted((n, cases[n][0]) for n, (r, _s) in driven.items() if not isinstance(r, Exception))
    assert not accepted, accepted[:10]


# ── the generator's own rules, on made-up routes ────────────────────────────────────────────────────────────────

COMPONENTS = {
    "Item": {"type": "object", "properties": {"id": {"type": "string"}, "n": {"type": "integer"},
                                              "child": {"$ref": "#/components/schemas/Child"}},
             "required": ["id"], "additionalProperties": False},
    "Child": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
    "WithNext": {"type": "object", "properties": {"id": {"type": "string"}, "next": {"type": "string"},
                                                  "result": {"type": "integer"}}},
}


def _native(*operations):
    """A made-up contract in the writer's form (`answers_contract`: references under `#/$defs/`, `required` every
    always-sent property, objects open), read by the generator's native source (FTASKS 10.1)."""
    import copy

    from scripts.generate_output_schemas import native_source
    from src.api import contract_tools

    def rewrite(node):
        return json.loads(json.dumps(node).replace("#/components/schemas/", "#/$defs/"))

    defs = rewrite(copy.deepcopy(COMPONENTS))
    responses = {f"GET {path}": {"200": rewrite(schema)} for path, schema in operations}
    for part in (defs, responses):
        contract_tools.always_sent_required(part)
        contract_tools.open_objects(part)
    return native_source(contract={"responses": responses, "excluded": {}, "$defs": defs, "external": {}})


def _parts(source, path):
    from scripts.generate_output_schemas import parts_of

    entry = source.entry("GET", path[len("/api/v1"):])
    return [p for s in entry.values() for p in parts_of(s, source, envelope=False, embedded={})]


def test_an_object_route_requires_every_field_and_a_list_route_its_result():
    """WT5: a response model sends every field, so `required` is every property; a list answers `{"result": [...]}`."""
    source = _native(("/api/v1/item", {"$ref": "#/components/schemas/Item"}),
                     ("/api/v1/items", {"type": "array", "items": {"$ref": "#/components/schemas/Item"}}))
    [part] = _parts(source, "/api/v1/item")
    assert part["required"] == ["child", "id", "n"]
    assert part["properties"]["child"] == {"$ref": "#/$defs/Child"}
    assert source.defs["Item"].get("additionalProperties") is None  # opened (SH7)
    [listed] = _parts(source, "/api/v1/items")
    assert listed == {"type": "object", "required": ["result"],
                      "properties": {"result": {"type": "array", "items": {"$ref": "#/$defs/Item"}}}}


def test_a_routes_own_next_is_never_constrained_and_its_result_stays_open():
    source = _native(("/api/v1/w", {"$ref": "#/components/schemas/WithNext"}))
    [part] = _parts(source, "/api/v1/w")
    assert "next" not in part["properties"] and "next" not in part["required"]
    assert part["properties"]["result"] == {}


def test_a_nullable_route_answers_the_empty_object():
    """MW6, FR-046.21: miWorkbench's client answers `{}` for `null` (`client.py:117`): `{}` and a full model
    validate, `{"x": 1}` without the model's fields does not."""
    source = _native(("/api/v1/maybe", {"anyOf": [{"$ref": "#/components/schemas/Child"}, {"type": "null"}]}))
    parts = _parts(source, "/api/v1/maybe")
    schema = {"anyOf": parts, "$defs": source.defs}
    validator = _validator(schema)
    assert not list(validator.iter_errors({}))
    assert not list(validator.iter_errors({"x": 1.5}))
    assert list(validator.iter_errors({"y": 1})), "an object with none of the model's fields must not validate"


def test_miservers_envelope_is_unwrapped_and_a_null_data_answers_result_null():
    """FR-046.26: the management envelope's `data` is what the relay answers; `null` data is `{"result": null}`."""
    from scripts.generate_output_schemas import Source, parts_of

    defs = {"Row": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            "Env": {"type": "object", "required": ["data", "error", "success"],
                    "properties": {"success": {"type": "boolean"}, "error": {},
                                   "data": {"anyOf": [{"$ref": "#/$defs/Row"}, {"type": "null"}]}}}}
    source = Source("miserver", {("GET", "/api/row"): {"200": {"$ref": "#/$defs/Env"}}}, defs, set(), {})
    parts = parts_of({"$ref": "#/$defs/Env"}, source, envelope=True, embedded={})
    assert parts[0]["required"] == ["id"] and "success" not in parts[0]["properties"]
    assert {"type": "object", "properties": {"result": {"type": "null"}}, "required": ["result"]} in parts


@pytest.mark.parametrize(("schema", "says"), [
    ({}, "no response model"),
    ({"type": "object", "additionalProperties": True}, "untyped object"),
    ({"type": "array", "items": {}}, "untyped list"),
])
def test_an_untyped_body_is_refused(schema, says):
    from scripts.generate_output_schemas import _Untyped, parts_of

    source = _native(("/api/v1/x", schema))
    with pytest.raises(_Untyped, match=says):
        parts_of(schema, source, envelope=False, embedded={})


def test_a_change_in_the_committed_contract_reaches_the_tool_schema():
    """The wiring of FTASKS 10.1: a field planted in the contract file's definition of a route's answer appears in the
    schema of the tool that answers that route, and the same tool built from the unchanged contract has no such
    field. (A generator still reading the live OpenAPI would ignore the file, and this is red.)"""
    from scripts.generate_output_schemas import NATIVE_CONTRACT, live_tools, miserver_source, native_source, tool_schema
    from scripts.output_overlays import OVERLAYS
    from src.mcp_server.result_language import load_contract
    from src.mcp_server.tool_classes import result_language_properties

    planted = json.loads(NATIVE_CONTRACT.read_text(encoding="utf-8"))
    answer = planted["$defs"]["CircuitDeletedAnswer"]
    answer["properties"]["planted_by_the_test"] = {"type": "integer"}
    answer["required"] = sorted([*answer["required"], "planted_by_the_test"])
    result_language = load_contract()
    fields = result_language_properties(result_language)
    fn = live_tools()["delete_circuit"].fn
    built = {}
    for label, native in (("planted", native_source(contract=planted)), ("committed", native_source())):
        sources = {"miworkbench": native, "miserver": miserver_source()}
        schema, _route, reason = tool_schema("delete_circuit", fn, sources, OVERLAYS, result_language, fields)
        assert schema is not None, reason
        built[label] = json.dumps(schema)
    assert "planted_by_the_test" in built["planted"]
    assert "planted_by_the_test" not in built["committed"]


@pytest.mark.parametrize("flag", ["response_model_exclude_none", "response_model_exclude_unset",
                                  "response_model_exclude_defaults"])
def test_a_route_that_leaves_fields_out_stops_the_contract_writer(flag):
    """`required` means always sent, which a response model breaks with `response_model_exclude_*`. Since 046-2's
    native switch (FTASKS 10.1) the generator reads the committed contract, so the guard is the WRITER's: it refuses a
    covered route that sets one, naming it (review round 1, finding 1, carried from the generator's old guard)."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    from src.api.answers_contract import CoverageError, contract_from

    class Row(BaseModel):
        a: int | None = None

    app = FastAPI()

    @app.get("/api/v1/fine", response_model=Row)
    async def fine(q: int = 0):  # a parameter, so the app declares the 422 the writer requires
        return Row()

    @app.get("/api/v1/trims", response_model=Row, **{flag: True})
    async def trims():
        return Row()

    contract_from(app, app.openapi(), frozenset({"GET /api/v1/fine"}))  # nothing flagged
    with pytest.raises(CoverageError, match=f"/api/v1/trims.*{flag}"):
        contract_from(app, app.openapi(), frozenset({"GET /api/v1/fine", "GET /api/v1/trims"}))


def test_a_document_the_contract_pins_by_another_sha_stops_the_generator():
    """046-2 review round 1, finding 4: the generator embeds a document schema only when its bytes are the ones the
    source contract's `external` pins; a contract pinning another sha256 is a GenerationError, never a silent embed."""
    from scripts.generate_output_schemas import (GenerationError, NATIVE_CONTRACT, live_tools, miserver_source,
                                                 native_source, tool_schema)
    from scripts.output_overlays import OVERLAYS
    from src.mcp_server.result_language import load_contract
    from src.mcp_server.tool_classes import result_language_properties

    contract = json.loads(NATIVE_CONTRACT.read_text(encoding="utf-8"))
    circuit = next(sid for sid, row in contract["external"].items() if row["file"].endswith("circuit-definition-v1.json"))
    result_language = load_contract()
    fields = result_language_properties(result_language)
    fn = live_tools()["export_circuit_definition"].fn
    sources = {"miworkbench": native_source(contract=contract), "miserver": miserver_source()}
    schema, _route, _reason = tool_schema("export_circuit_definition", fn, sources, OVERLAYS, result_language, fields)
    assert schema is not None and "circuit-definition-v1.json" in json.dumps(schema)  # the control: it embeds
    contract["external"][circuit] = {**contract["external"][circuit], "sha256": "0" * 64}
    sources = {"miworkbench": native_source(contract=contract), "miserver": miserver_source()}
    with pytest.raises(GenerationError, match="circuit-definition-v1.json"):
        tool_schema("export_circuit_definition", fn, sources, OVERLAYS, result_language, fields)


def test_the_native_source_is_the_committed_contract():
    """FTASKS 10.1: the native tools' answers come from `docs/schemas/miworkbench-answers-v1.json`, whole: every
    covered route's 2xx entries, its `$defs` and `external`, and the file and digest named in `sources`."""
    from scripts.generate_output_schemas import NATIVE_CONTRACT, native_source
    from src.mcp_server.route_map import normalise

    contract = json.loads(NATIVE_CONTRACT.read_text(encoding="utf-8"))
    source = native_source()
    assert source.defs == contract["$defs"] and source.external == contract["external"]
    want = {(k.split(" ", 1)[0], normalise(k.split(" ", 1)[1])): {s: v for s, v in e.items() if s[:1] == "2"}
            for k, e in contract["responses"].items()}
    assert source.entries == want
    # no body at all on some status: an excluded row's null status, or a JSON route's `x-no-body` (review round 1)
    assert source.empty == {(k.split(" ", 1)[0], normalise(k.split(" ", 1)[1])) for k, r in contract["excluded"].items()
                            if None in r["statuses"].values()} | {
        (k.split(" ", 1)[0], normalise(k.split(" ", 1)[1])) for k, e in contract["responses"].items()
        if e.get("x-no-body")}
    document = _document()
    assert document["sources"][NATIVE_CONTRACT.name] == contract["content_sha256"]


def test_a_relayed_pending_without_decide_in_is_refused():
    """review round 1, finding 4 (O5): the relay always adds `decide_in` to miServer's pending answer
    (`miserver_approvals.relayed`), so the alternative requires it."""
    offering = [n for n, sch in _document()["tools"].items()
                if any(p.get("allOf") == [{"$ref": "#/$defs/ApprovalPending"}] for p in sch["anyOf"])]
    assert len(offering) >= 8, offering
    bare = {k: v for k, v in RELAYED_PENDING.items() if k != "decide_in"}
    for name in offering:
        validator = _validator(_document()["tools"][name])
        assert not list(validator.iter_errors(RELAYED_PENDING)), name
        assert list(validator.iter_errors(bare)), f"{name} accepts a relayed pending with no decide_in"


def test_every_answer_part_types_the_shared_legacy_keys():
    """review round 1, finding 4 (O11): beside a route's own fields, each shared legacy key is allowed with ITS type
    (`SHARED`), and a key the route declares keeps the route's type."""
    from scripts.generate_output_schemas import EMPTY_OBJECT, SHARED

    checked = own = 0
    for name in _unchanged():
        for part in _answer_parts(_document()["tools"][name]):
            props = part.get("properties") or {}
            if part == EMPTY_OBJECT or set(props) == {"result"} or "$ref" in part:
                continue  # the empty answer, a list/scalar/null under `result`, an embedded document
            missing = sorted(set(SHARED) - set(props))
            assert not missing, (name, missing)
            checked += 1
            own += sum(1 for key in SHARED if props[key] != SHARED[key])
    assert checked >= 100, checked
    assert own >= 1, "no route's own declaration of a shared key survived: the shared type overwrote it"


def test_a_shared_key_the_route_declares_keeps_the_routes_type():
    """O11, round 2 (control R2h survived the live test alone): the route is its field's authority."""
    from scripts.generate_output_schemas import SHARED, parts_of

    contract = {"responses": {"GET /api/v1/row": {"200": {"$ref": "#/$defs/Row"}}}, "excluded": {}, "external": {},
                "$defs": {"Row": {"type": "object", "required": ["status"],
                                  "properties": {"status": {"enum": ["done", "failed"]}}}}}
    from scripts.generate_output_schemas import native_source

    source = native_source(contract=contract)
    [part] = parts_of(source.entry("GET", "/row")["200"], source, envelope=False, embedded={})
    row = source.defs["Row"] if "$ref" in part else part
    props = part.get("properties") or row["properties"]
    assert props["status"] == {"enum": ["done", "failed"]}, props["status"]
    assert "status" in SHARED


#: The three tools whose only route answers no body (a 204): review round 1, finding 7.
NO_BODY_TOOLS = {"delete_judge_prompt": ("DELETE", "/api/v1/probe-monitors/judge-prompts/{}", {"prompt_id": "jp_1"}),
                 "delete_model": ("DELETE", "/api/v1/models/{}", {"model_id": "m_1", "acknowledge_permanent": True}),
                 "delete_probe_training_run": ("DELETE", "/api/v1/probe-monitors/runs/{}",
                                               {"run_id": "pmr_1", "acknowledge_permanent": True})}


def test_a_route_with_no_body_is_the_empty_object_not_untyped():
    """Review round 1, finding 7: a route that answers no body at all (204) is answered `{}` by miWorkbench's client
    (`client.py:117`), so it is typed as the empty object; a 2xx with a non-JSON body stays excluded."""
    from scripts.generate_output_schemas import native_source

    # the writer's form: both routes are `excluded`, the 204 with no media type (FR-046.9)
    contract = {"responses": {}, "$defs": {}, "external": {},
                "excluded": {"DELETE /api/v1/gone/{gone_id}": {"media_type": None, "reason": "no content"},
                             "GET /api/v1/text": {"media_type": "text/plain", "reason": "a text answer"}}}
    source = native_source(contract=contract)
    assert source.empty == {source.key("DELETE", "/gone/{gone_id}")}
    assert source.key("GET", "/text") in source.excluded and source.key("GET", "/text") not in source.empty


def test_the_no_body_tools_are_typed_and_answer_the_empty_object(monkeypatch):
    """Finding 7, live: the three delete tools are typed with the empty object, driven against a 204 they answer `{}`
    through the SDK client from one DELETE, and a body with a field does not validate their answer part."""
    from scripts.generate_output_schemas import EMPTY_OBJECT

    document = _document()
    for name in NO_BODY_TOOLS:
        assert name not in document["untyped"], (name, document["untyped"].get(name))
        schema = document["tools"][name]
        assert _answer_parts(schema) == [EMPTY_OBJECT], (name, _answer_parts(schema))
        assert list(_validator({"anyOf": _answer_parts(schema)}).iter_errors({"deleted": True}))
    with mcp_drive.patched_routes(monkeypatch) as backend:
        for method, path, _arguments in NO_BODY_TOOLS.values():
            backend.answer(method, path, b"", status=204)
        servers = mcp_drive.build_every_category()
        owner = {t.name: m for m in servers.values() for t in m._tool_manager.list_tools()}  # noqa: SLF001
        for name, (method, path, arguments) in NO_BODY_TOOLS.items():
            result = mcp_drive.call(owner[name], name, arguments)
            assert not result.is_error, (name, result.content[0].text[:300])
            assert result.structured_content == {}, (name, result.structured_content)
            assert len(backend.calls_to(method, path)) == 1, name



def test_a_json_route_with_a_no_body_status_is_also_the_empty_object():
    """Review round 1, finding 6: a JSON route that also answers a status with no body (`x-no-body`, a 204) is in the
    native source's empty set, so its tool is typed with the `{}` variant beside the model."""
    from scripts.generate_output_schemas import native_source

    contract = {"responses": {"DELETE /api/v1/thing/{thing_id}": {"200": {"$ref": "#/$defs/Row"}, "x-no-body": ["204"]},
                              "GET /api/v1/thing/{thing_id}": {"200": {"$ref": "#/$defs/Row"}}},
                "excluded": {}, "external": {},
                "$defs": {"Row": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}}
    source = native_source(contract=contract)
    assert source.empty == {source.key("DELETE", "/thing/{thing_id}")}


def test_an_own_202_union_needs_the_routes_model():
    """Review round 1, finding 6: the untyped stand-in is gone; a non-model would give every route one shared name."""
    from src.core.approval_gate import gated_responses
    from src.schemas.answer_fields import PendingApprovalAnswer

    with pytest.raises(TypeError, match="Pydantic model"):
        gated_responses(dict, status_code=202)
    assert gated_responses(PendingApprovalAnswer, status_code=202)[202]["model"].__name__ == "PendingApprovalAnswerOrPending"


def test_a_no_body_status_beside_a_json_answer_gives_the_tool_its_empty_variant():
    """Round 2 (R11 survived): `tool_schema` adds the `{}` variant for a route's `x-no-body` status, not only the source's
    empty set: `delete_circuit` with its route given a 204 too answers its model or `{}`."""
    from scripts.generate_output_schemas import (EMPTY_OBJECT, NATIVE_CONTRACT, live_tools, miserver_source,
                                                 native_source, tool_schema)
    from scripts.output_overlays import OVERLAYS
    from src.mcp_server.result_language import load_contract
    from src.mcp_server.tool_classes import result_language_properties

    contract = json.loads(NATIVE_CONTRACT.read_text(encoding="utf-8"))
    result_language = load_contract()
    fields = result_language_properties(result_language)
    fn = live_tools()["delete_circuit"].fn

    def parts(doc):
        sources = {"miworkbench": native_source(contract=doc), "miserver": miserver_source()}
        schema, _route, _reason = tool_schema("delete_circuit", fn, sources, OVERLAYS, result_language, fields)
        return _answer_parts(schema)

    assert EMPTY_OBJECT not in parts(contract)
    contract["responses"]["DELETE /api/v1/circuits/{circuit_id}"]["x-no-body"] = ["204"]
    assert EMPTY_OBJECT in parts(contract)
