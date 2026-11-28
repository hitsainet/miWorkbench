"""045 FR-4: every structured tool advertises an envelope output schema and answers with structured content that
the SDK's own client accepts, in every non-error result path; `content` is what it was before 045.

The SDK client validates `structuredContent` against the tool's `outputSchema` and RAISES when it is missing or does
not conform (`mcp/client/session.py:1118-1153`), so "no call raises" is the assertion. Each tool is driven over the
wire with the backend answering `{}`, `[]`, a 503 (miServer's tools turn it into `{"refused": …}`) and with every
product down (`{"unavailable": …}`); a tool that refuses the generated arguments answers an error result, which the
client does not validate, so the tests also require that most tools answered without an error.

Controls (record `0xcc/reviews/045_implementation_controls_2026-11-16.md`):
  drop structured content for dicts           → test_every_structured_tool_validates red (client raises)
  return a scalar unwrapped                   → test_a_scalar_or_none_answer_is_structured red
  wrap dicts in {"result": …} in `content`     → test_content_is_unchanged red
  apply_output_schemas before normalise_results → test_mcp_list_results.py red
"""
from __future__ import annotations

import pytest

from tests.support import mcp_drive

SCENARIOS = ("object", "list", "refused_503", "down")
#: 046: a TYPED tool's schema requires its route's fields, so `{}` and `[]` are not its route's answer (its client
#: refuses them, which is the typing biting); and a miServer relay's fake must be in miServer's contract shape (the
#: fake check, WT3). Both are driven with their routes' full bodies in test_mcp_typed_output_schemas.py,
#: test_relay_output_schemas.py and test_output_overlays.py. Here, the generic envelope is checked on every OTHER
#: tool, and every tool still answers the 503 and the outage scenarios.
BODY_SCENARIOS = ("object", "list")
#: Why these drives are off-contract (046 FTASKS 7.3): FR-4 checks the generic envelope of EVERY tool with ONE body
#: (`{}`, `[]`) for every route, which no single route's contract describes; the typed tools and the relays are driven
#: with their routes' own full bodies in test_mcp_typed_output_schemas.py and test_relay_output_schemas.py.
ONE_BODY = "the 045 FR-4 envelope check answers every route with one body ({} or [])"
#: The generic tools driven with one body must mostly answer without an error (the harness reaches them). 046 P7
#: types routes, so the generic set shrinks (160 native at origin/main f3943c72c); the floor is a share of it.
REACHED_SHARE = 0.75
#: Generic tools stay until FTASKS 10.1 (the `dataworks_*` relays and the accepted reasons stay after it).
MIN_GENERIC = 20


def _generic(name: str) -> bool:
    from src.mcp_server.tool_classes import load_output_schemas

    return name not in load_output_schemas() and not name.startswith("miserver_")


def _drive(servers, backend, scenario: str) -> dict:
    backend.down = scenario == "down"
    backend.status = 503 if scenario == "refused_503" else None  # None: each route's own 2xx
    backend.body = {"list": [], "object": {}}.get(scenario, {"detail": "busy"})
    out = {}
    for mcp in servers.values():
        for tool in mcp_drive.wire_tools(mcp):
            if scenario in BODY_SCENARIOS and not _generic(tool.name):
                continue
            try:
                out[tool.name] = mcp_drive.call(mcp, tool.name, mcp_drive.sample_arguments(tool))
            except Exception as exc:  # noqa: BLE001 — the client's validation failure is the finding
                out[tool.name] = exc
    backend.down, backend.status = False, None
    return out


@pytest.fixture(scope="module")
def driven():
    with pytest.MonkeyPatch.context() as mp:
        with mcp_drive.patched_http(mp, off_contract=ONE_BODY) as backend:
            servers = mcp_drive.build_every_category()
            schemas = {t.name: t.output_schema for m in servers.values() for t in mcp_drive.wire_tools(m)}
            return {"schemas": schemas, **{s: _drive(servers, backend, s) for s in SCENARIOS}}


def test_every_structured_tool_advertises_an_envelope(driven):
    from src.mcp_server.result_language import load_contract
    from src.mcp_server.tool_classes import TOOL_CLASSES, envelope_schema, load_output_schemas, output_schema

    typed = load_output_schemas()
    for name, schema in driven["schemas"].items():
        # Phase B (FTDD §6.1): the envelope carries the contract's `approval` and `next_steps`, inlined. The 045
        # follow-up (RL-1) types a tool that answers one route's body unchanged; its schema keeps those two.
        assert schema == output_schema(name, load_contract()), name
        if name not in typed:
            assert schema == envelope_schema(TOOL_CLASSES[name].result, load_contract()), name
            assert "$ref" not in str(schema), name
    assert all(s is not None for s in driven["schemas"].values())


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_structured_tool_validates(driven, scenario):
    results = driven[scenario]
    raised = {n: repr(r)[:200] for n, r in results.items() if isinstance(r, Exception)}
    assert not raised, f"{len(raised)} tools' answers fail their own output schema: {dict(list(raised.items())[:5])}"
    ok = [n for n, r in results.items() if not r.is_error]
    assert all(r.structured_content is not None for n, r in results.items() if not r.is_error)
    # The harness must REACH the tools: most answer without an error in the healthy scenarios. Only the generic tools
    # are driven with one body, and 046 P7 shrinks that set as it types routes, so the floor is a share of it.
    if scenario in BODY_SCENARIOS:
        assert len(results) >= MIN_GENERIC, f"only {len(results)} generic tools driven"
        assert len(ok) >= REACHED_SHARE * len(results), f"only {len(ok)} of {len(results)} tools answered without an error"
    if scenario == "down":
        unavailable = [n for n, r in results.items() if not r.is_error and "unavailable" in r.structured_content]
        assert len(unavailable) > 100, len(unavailable)
    if scenario == "refused_503":
        refused = [n for n, r in results.items() if not r.is_error and "refused" in r.structured_content]
        assert len(refused) >= 5, refused


def test_a_list_answer_is_structured_as_result(driven):
    lists = [r for r in driven["list"].values() if not r.is_error and "result" in r.structured_content]
    assert len(lists) >= REACHED_SHARE * len(driven["list"]), (len(lists), len(driven["list"]))
    assert all(isinstance(r.structured_content["result"], list) for r in lists)


def test_a_dict_answer_is_structured_as_itself(driven):
    import json

    dicts = [r for r in driven["object"].values()
             if not r.is_error and len(r.content) == 1 and r.content[0].text.startswith("{")]
    assert len(dicts) >= REACHED_SHARE * len(driven["object"]), (len(dicts), len(driven["object"]))
    for r in dicts:
        assert r.structured_content == json.loads(r.content[0].text)


@pytest.mark.parametrize("value", ["a plain string", 3, True, None])
def test_a_scalar_or_none_answer_is_structured(value):
    """A30, R1A-21: a structured tool that answers a scalar or None still carries structured content."""
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.server import normalise_results
    from src.mcp_server.tool_classes import apply_output_schemas

    mcp = MCPServer("t")

    @mcp.tool(name="get_model")  # a real row, so the table applies
    async def stub(model_id: str) -> object:
        return value

    normalise_results(mcp)
    apply_output_schemas(mcp, typed={})  # the generic envelope (046 types `get_model` from its route)
    result = mcp_drive.call(mcp, "get_model", {"model_id": "m"})
    assert not result.is_error
    assert result.structured_content == {"result": value}


def test_content_is_unchanged(monkeypatch):
    """FR-4: `content` is byte-for-byte what a server built without FR-4 answers."""
    from src.mcp_server import server as server_module
    from src.mcp_server import tool_classes

    def pre_045(value):
        """What the SDK made of an answer before 045: a list through `list_result`, anything else through its own
        unstructured conversion (`FuncMetadata.convert_result` with no output schema)."""
        from mcp.server.mcpserver.utilities.func_metadata import _convert_to_content
        from mcp.types import CallToolResult

        if server_module._is_plain_list(value):
            return server_module.list_result(value)
        if isinstance(value, CallToolResult):
            return value
        return CallToolResult(content=list(_convert_to_content(value)))

    with mcp_drive.patched_http(monkeypatch, off_contract=ONE_BODY) as backend:
        now = mcp_drive.build_every_category()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(server_module, "structured_result", pre_045)
            mp.setattr(tool_classes, "apply_output_schemas", lambda mcp, contract=None: 0)
            before = mcp_drive.build_every_category()
        compared = {"generic": 0, "typed": 0}
        generic_tools: set[str] = set()
        for body in (mcp_drive.EMPTY_OBJECT, mcp_drive.EMPTY_LIST):
            backend.body = body
            for category, mcp in now.items():
                for tool in mcp_drive.wire_tools(mcp):
                    if not _generic(tool.name):  # 046: `{}` / `[]` is not a typed tool's or a relay's answer
                        continue
                    args = mcp_drive.sample_arguments(tool)
                    a = mcp_drive.call(mcp, tool.name, args)
                    b = mcp_drive.call(before[category], tool.name, args)
                    assert [c.model_dump() for c in a.content] == [c.model_dump() for c in b.content], tool.name
                    compared["generic"] += 1
                    generic_tools.add(tool.name)
        # 046 review round 1, finding 8: the typed tools and the relays, each driven with its route's FULL body (what
        # the typed schema accepts), so FR-4's comparison covers them again rather than skipping them.
        from tests.unit.test_mcp_typed_output_schemas import _body, _unchanged, _wire_body

        for category, mcp in now.items():
            for tool in mcp_drive.wire_tools(mcp):
                if tool.name not in _unchanged():
                    continue
                backend.body = _wire_body(tool.name, _body(tool.name))
                args = mcp_drive.sample_arguments(tool)
                a = mcp_drive.call(mcp, tool.name, args)
                b = mcp_drive.call(before[category], tool.name, args)
                assert not a.is_error, (tool.name, a.content[0].text[:300])
                assert [c.model_dump() for c in a.content] == [c.model_dump() for c in b.content], tool.name
                compared["typed"] += 1
    # Every generic tool is compared under both bodies, and the typed tools that answer a route unchanged once each.
    # 046 P7 moves tools from the first set to the second as it types routes, so the floor is on the two together.
    assert compared["generic"] == 2 * len(generic_tools) and len(generic_tools) >= MIN_GENERIC, compared
    assert compared["typed"] >= 120, compared  # 134 typed tools answered a route unchanged on 2026-11-25
    assert len(generic_tools) + compared["typed"] > 280, (len(generic_tools), compared)
