"""No advertised output schema rejects a naive datetime (hotfix fix/046-1-datetime-format, 2026-11-26).

The production MCP client (JavaScript, ajv with formats) asserts `format`. `date-time` and `time` need an RFC 3339
offset, and the backend serialises naive datetimes without one (`2026-11-26T10:15:00.123456`, a SQLAlchemy `DateTime`
column through a Pydantic response model). After 046-1 typed 187 tools, every one of them declared `format: date-time`
somewhere, and `list_cluster_profiles` failed its schema on EVERY row in production ("data/data/0/created_at must match
format date-time"), so the client threw the whole answer away. The Python SDK client this suite drives does not check
formats, so the suite stayed green.

The fix strips those formats from what a tool ADVERTISES (`tool_classes.without_offset_formats`); no route's served bytes
change. These guards check answers the way the production client does, with `STRICT`, a format checker that follows
ajv-formats' full mode and fails closed on a format it does not know:

1. the live registry: no tool's `outputSchema` on the wire (typed or the generic envelope) carries a format that a naive
   datetime fails, and every format it does carry is one `STRICT` checks;
2. a real answer: a `ClusterProfile` ROW in the test database, served by `GET /api/v1/cluster-profiles` through its own
   response model, answered to `list_cluster_profiles` and `get_cluster_profile`, validates against their advertised
   schemas (and the served value has no offset, so the case is the production one, not one that agrees by
   construction);
3. the sweep: every typed tool that answers its route unchanged, driven with its route's full body where EVERY field the
   contract declares `date-time` or `time` carries the naive value the backend's serialiser produces, validates;
4. miServer's relayed pending with a naive `expires_at` validates (the root `approval.expires_at`, which the tool copies
   from it, was `date-time`; miServer's own `ApprovalPending.expires_at` declares no format).
"""
from __future__ import annotations

import copy
import functools
import json
import re
from datetime import date, datetime, time
from typing import Any
from uuid import UUID

import jsonschema
import pytest
from pydantic import TypeAdapter
from referencing import Registry

from tests.support import mcp_drive
from tests.support.answer_samples import sample

#: What the backend sends for a naive datetime: Pydantic's JSON form of `datetime.utcnow()`-style values, the default
#: of every `DateTime` column without `timezone=True` (e.g. `src/models/cluster_profile.py`).
NAIVE_DATETIME = TypeAdapter(datetime).dump_python(datetime(2026, 11, 26, 10, 15, 0, 123456), mode="json")
NAIVE_TIME = TypeAdapter(time).dump_python(time(10, 15, 0, 123456), mode="json")
#: The value each declared format carries in a sweep body: what the backend's serialiser sends.
SENT = {"date-time": NAIVE_DATETIME, "time": NAIVE_TIME,
        "date": TypeAdapter(date).dump_python(date(2026, 11, 26), mode="json"),
        "uuid": TypeAdapter(UUID).dump_python(UUID(int=0x1234), mode="json")}

# ajv-formats `full` mode (strictTimeZone, the default since ajv-formats 3): `date-time` and `time` need an offset.
_DATE = re.compile(r"^(\d\d\d\d)-(\d\d)-(\d\d)$")
_TIME = re.compile(r"^(\d\d):(\d\d):(\d\d(?:\.\d+)?)(z|[+-]\d\d(?::?\d\d)?)$", re.IGNORECASE)
_UUID = re.compile(r"^(?:urn:uuid:)?[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$", re.IGNORECASE)


def _date(value: str) -> bool:
    found = _DATE.match(value)
    return bool(found) and 1 <= int(found[2]) <= 12 and 1 <= int(found[3]) <= 31


def _time(value: str) -> bool:
    found = _TIME.match(value)
    return bool(found) and int(found[1]) <= 23 and int(found[2]) <= 59 and float(found[3]) < 61


def _date_time(value: str) -> bool:
    parts = re.split(r"t|\s", value, maxsplit=1, flags=re.IGNORECASE)
    return len(parts) == 2 and _date(parts[0]) and _time(parts[1])


_CHECKS = {"date-time": _date_time, "time": _time, "date": _date, "uuid": lambda v: bool(_UUID.match(v))}

STRICT = jsonschema.FormatChecker(formats=())
for _name, _check in _CHECKS.items():
    STRICT.checks(_name)(lambda v, _c=_check: not isinstance(v, str) or _c(v))


def _no_fetch(uri: str):
    raise LookupError(f"a schema reference left the document: {uri}")


def _strict_errors(schema: dict, value: Any) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema, registry=Registry(retrieve=_no_fetch), format_checker=STRICT)
    out = []
    for error in validator.iter_errors(value):
        deepest = max([error, *error.context], key=lambda e: len(e.absolute_path))  # an anyOf names its branches
        out.append(f"{'/'.join(map(str, deepest.absolute_path))}: {deepest.message[:160]}")
    return out[:5]


def _walk(node: Any, path: tuple = ()):
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, path + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, path + (index,))


def _formats(schema: Any) -> list[tuple[str, str]]:
    """`(path, format)` for every `format` KEYWORD (a property named `format` is skipped)."""
    out = []
    for path, node in _walk(schema):
        if isinstance(node, dict) and isinstance(node.get("format"), str) and (not path or path[-1] != "properties"):
            out.append(("/".join(map(str, path)), node["format"]))
    return out


@functools.lru_cache(maxsize=1)
def _wire() -> dict[str, Any]:
    """`tools/list` of ONE server with every category (the deployed shape), through the SDK client."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    mcp, _ = build_server(MCPSettings(tool_categories=",".join(sorted(all_categories())), allow_anonymous=True,
                                      _env_file=None, **mcp_drive.placeholder_overrides()), stdio=True)
    return {t.name: t for t in mcp_drive.wire_tools(mcp)}


def _advertised(name: str) -> dict:
    schema = _wire()[name].output_schema
    assert schema, f"{name} advertises no output schema"
    return schema


# ── 1. the live registry ────────────────────────────────────────────────────────────────────────────────────────


def test_no_advertised_schema_rejects_a_naive_datetime():
    """Every tool on the wire: no `format` a naive datetime fails, and no format `STRICT` cannot check (an unknown
    format would pass here and could still fail in the production client)."""
    from src.mcp_server.tool_classes import load_output_schemas

    rejecting, unknown, with_schema = {}, {}, 0
    for name, tool in _wire().items():
        if not tool.output_schema:
            continue
        with_schema += 1
        for path, fmt in _formats(tool.output_schema):
            if fmt not in _CHECKS:
                unknown.setdefault(name, []).append(f"{path}: {fmt}")
            elif fmt in ("date-time", "time"):  # STRICT refuses the naive value for both (the next test)
                rejecting.setdefault(name, []).append(f"{path}: {fmt}")
    assert not rejecting, f"{len(rejecting)} tools advertise a format a naive datetime fails: {dict(list(rejecting.items())[:5])}"
    assert not unknown, f"formats STRICT does not check (add an ajv-equivalent check): {unknown}"
    assert with_schema >= len(load_output_schemas()) >= 180, (with_schema, len(load_output_schemas()))


def test_the_strict_checker_is_the_production_clients():
    """`STRICT` refuses what ajv refuses: the naive value fails `date-time` and `time`; offsets pass."""
    assert not STRICT.conforms(NAIVE_DATETIME, "date-time") and not STRICT.conforms(NAIVE_TIME, "time")
    assert STRICT.conforms(NAIVE_DATETIME + "Z", "date-time") and STRICT.conforms("2026-11-26T10:15:00+05:30",
                                                                                   "date-time")
    assert STRICT.conforms("10:15:00Z", "time") and STRICT.conforms("2026-11-26", "date")
    assert "T" in NAIVE_DATETIME and not NAIVE_DATETIME.endswith(("Z", "z")) and "+" not in NAIVE_DATETIME


# ── 2. a real answer: a database row through the route's own serializer ──────────────────────────────────────────


async def _call(name: str, arguments: dict, backend: mcp_drive.RouteBackend, monkeypatch):
    from mcp import Client

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    with mcp_drive.patched_routes(monkeypatch, backend):
        mcp, _ = build_server(MCPSettings(tool_categories="profiles", allow_anonymous=True, _env_file=None,
                                          **mcp_drive.placeholder_overrides()), stdio=True)
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments)


async def test_list_cluster_profiles_answers_a_real_row_the_strict_client_accepts(client, async_session, monkeypatch):
    """The production failure, reproduced from a row: `ClusterProfile.created_at` is a naive `DateTime`
    (`datetime.utcnow`), served by `GET /cluster-profiles` through `ClusterProfileListResponse`."""
    from src.models.cluster_profile import ClusterProfile

    row = ClusterProfile(name="Fear cluster", members=[{"feature_idx": 1, "strength": 1.0}], schema_version="1")
    async_session.add(row)
    await async_session.commit()

    served = await client.get("/api/v1/cluster-profiles")
    assert served.status_code == 200, served.text[:300]
    listed = served.json()
    stamp = listed["data"][0]["created_at"]
    assert not STRICT.conforms(stamp, "date-time"), f"the route now sends an offset ({stamp}): update this guard"

    one = await client.get(f"/api/v1/cluster-profiles/{row.id}")
    assert one.status_code == 200, one.text[:300]

    backend = mcp_drive.RouteBackend()
    backend.answer("GET", "/api/v1/cluster-profiles", listed)
    backend.answer("GET", "/api/v1/cluster-profiles/{}", one.json())
    for name, arguments, answer in (("list_cluster_profiles", {}, listed),
                                    ("get_cluster_profile", {"profile_id": row.id}, one.json())):
        result = await _call(name, arguments, backend, monkeypatch)
        assert not result.is_error, result.content[0].text[:300]
        body = {k: v for k, v in result.structured_content.items() if k not in ("next_steps", "approval")}
        assert body == answer, name  # the route's bytes, unchanged (no offset added on the way)
        assert not _strict_errors(_advertised(name), result.structured_content), (
            name, _strict_errors(_advertised(name), result.structured_content))


# ── 3. the sweep: every unchanged-answer typed tool, a naive value in every datetime field ───────────────────────


@functools.lru_cache(maxsize=1)
def _declared() -> dict[str, Any]:
    """The tool schemas WITH the source contracts' formats (the generator, its strip turned off): where each
    datetime field is. Each formatted string becomes the value the backend sends (`SENT`): naive for `date-time` and
    `time`.

    Since the native switch (046 FTASKS 10.1) miWorkbench's tools are generated from its answer contract, whose writer
    drops the offset formats too (`contract_tools.drop_offset_formats`); so the native source here is the writer's
    contract of the live app with that drop turned off, which is where the routes' datetime fields are declared."""
    import scripts.generate_output_schemas as generator
    from src.api import answers_contract, contract_tools

    real, real_drop = generator.without_offset_formats, contract_tools.drop_offset_formats
    generator.without_offset_formats = lambda schema: schema
    contract_tools.drop_offset_formats = lambda node: node
    try:
        native = generator.native_source(contract=answers_contract.answers_contract())
        tools = generator.generate(native=native)["tools"]
    finally:
        generator.without_offset_formats, contract_tools.drop_offset_formats = real, real_drop

    def naive(node: Any) -> Any:
        if isinstance(node, list):
            return [naive(v) for v in node]
        if not isinstance(node, dict):
            return node
        if isinstance(node.get("format"), str) and node["format"] in SENT and node.get("type") in ("string", None):
            return {"const": SENT[node["format"]]}
        return {k: naive(v) for k, v in node.items()}

    return {name: naive(schema) for name, schema in tools.items()}


def test_every_typed_tool_answers_a_naive_datetime_the_strict_client_accepts(monkeypatch):
    from scripts.generate_output_schemas import SHARED
    from tests.unit.test_mcp_typed_output_schemas import _body_part, _drive, _unchanged, _wire_body, route_complete

    declared = _declared()
    bodies = {}
    for name in _unchanged():
        schema = declared[name]
        part = copy.deepcopy(_body_part(schema))
        props = part.get("properties", {})
        if set(props) == {"result"} and part.get("required") == ["result"]:
            body = {"result": sample(props["result"], schema)}
        else:
            if props:
                part["properties"] = {k: v for k, v in props.items() if k not in SHARED or k in part.get("required", [])}
            body = route_complete(name, sample(part, schema))
        if NAIVE_DATETIME in json.dumps(body) or NAIVE_TIME in json.dumps(body):
            bodies[name] = body
    driven = _drive(monkeypatch, {n: _wire_body(n, b) for n, b in bodies.items()})
    failures = {}
    for name, (result, _sent) in driven.items():
        if isinstance(result, Exception):
            failures[name] = repr(result)[:200]
        elif result.is_error:
            failures[name] = result.content[0].text[:200]
        else:
            errors = _strict_errors(_advertised(name), result.structured_content)
            if errors:
                failures[name] = errors
    assert not failures, f"{len(failures)} of {len(bodies)} tools: {dict(list(failures.items())[:5])}"
    assert len(bodies) >= 40, len(bodies)  # every unchanged-answer tool with a datetime field (44, 2026-11-26)


# ── 4. miServer's relayed pending ────────────────────────────────────────────────────────────────────────────────


def test_a_relayed_pending_with_a_naive_expiry_is_accepted(monkeypatch):
    """`miserver_load_model` relaying miServer's 202 pending with a naive `expires_at`: the root `approval.expires_at`
    (`result_language_properties`), which the tool copies from it, accepts it."""
    from tests.support.miserver_fakes import approval_actions, envelope

    pending = {"approval_id": "apr_" + "0" * 32, "status": "pending", "action": "model_load", "summary": {"model": "m"},
               "request_digest": "sha256:" + "a" * 64, "expires_at": NAIVE_DATETIME, "hint": "h"}
    backend = mcp_drive.RouteBackend()
    backend.answer("GET", "/api/approvals/actions", envelope(approval_actions(["model_load"])))
    backend.answer("POST", "/api/models/{}/load", {"success": True, "data": pending, "error": None}, status=202,
                   headers={"X-miLLM-Approval": "pending"})
    name = "miserver_load_model"
    with mcp_drive.patched_routes(monkeypatch, backend):
        result = mcp_drive.call(mcp_drive.build_category("miserver_models"), name, {"model_id": 7})
    assert not result.is_error, result.content[0].text[:300]
    answer = result.structured_content
    assert answer["expires_at"] == NAIVE_DATETIME and answer["approval"]["expires_at"] == NAIVE_DATETIME, answer
    assert not _strict_errors(_advertised(name), answer), _strict_errors(_advertised(name), answer)


@pytest.mark.parametrize("kind", ["object", "list"])
def test_the_generic_envelope_accepts_a_naive_approval_expiry(kind):
    """The generic envelope (every untyped tool, the miDataworks relays among them) carries the same root
    `approval.expires_at`."""
    from src.mcp_server.result_language import load_contract
    from src.mcp_server.tool_classes import envelope_schema

    schema = envelope_schema(kind, load_contract())
    answer = {"approval": {"id": "apr_1", "status": "pending", "decide_in": "midataworks", "then": "server_runs",
                           "summary": "x", "expires_at": NAIVE_DATETIME}}
    if kind == "list":
        answer["result"] = []
    assert schema["properties"]["approval"], schema
    assert not _strict_errors(schema, answer), _strict_errors(schema, answer)
