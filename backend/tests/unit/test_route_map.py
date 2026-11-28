"""The one route walker, `src/mcp_server/route_map.py` (046 WT1, FTASKS 1.6).

Every consumer (the generated MCP contract, the output-schema generator, the route-parity guards and, from 046-2, the
answer contract's coverage) reads this walker, so what it misses is invisible everywhere at once. These tests pin it
against the LIVE registry (membership never comes from the walk; memory derive-from-the-registry-not-the-source):

- every registered tool is a walked `@mcp.tool` function, and every one but the guide calls a route;
- no client call has a path the walk cannot read (an unread call is a route no guard can see);
- the three steering tools reach their routes only THROUGH `_submit`, and the two WSP tools only through
  `_submit_approved`: helper-following is what finds them (the old contract walker found only `_submit*` calls by
  name, and missed every other helper; 046 P1 regenerated 14 contract rows that gained routes);
- `returns_unchanged` (the generator's classifier, moved here) on made-up tools, relays included.

Controls (0xcc/reviews/046_controls_2026-11-25.md):
  M11  `resolve` stops following helpers     → test_the_steering_and_wsp_routes_are_found_through_their_helpers red
  M36  `contract.py` walks on its own, not following helpers → test_mcp_contract_generated.py red (the steering rows)
"""
from __future__ import annotations

import pytest

from src.mcp_server.route_map import normalise, returns_unchanged, tool_routes, unreadable_calls
from tests.support.live_registry import registered_tools


def test_every_registered_tool_is_walked_and_calls_a_route(monkeypatch):
    registered = registered_tools(monkeypatch)
    walked = tool_routes()
    assert len(registered) > 300, len(registered)  # a registry that built nothing would pass the rest
    assert registered <= set(walked), sorted(registered - set(walked))
    silent = sorted(name for name in registered if not walked[name])
    assert silent == ["miworkbench_howto"], silent


def test_no_client_call_is_unreadable():
    assert unreadable_calls() == []


@pytest.mark.parametrize(("tool", "route"), [
    ("steer_sweep", ("miworkbench", "POST", "/steering/async/sweep")),
    ("steer_compare", ("miworkbench", "POST", "/steering/async/compare")),
    ("steer_combined", ("miworkbench", "POST", "/steering/async/combined")),
    ("capture_wsp_heldout", ("miworkbench", "POST", "/wsp/captures")),
    ("import_wsp_candidate", ("miworkbench", "POST", "/wsp/candidates/import")),
])
def test_the_steering_and_wsp_routes_are_found_through_their_helpers(tool, route):
    """`steer_*` call `_submit(name, "/steering/async/…", body)`, the WSP tools `_submit_approved(name, route, …)`:
    the path is a HELPER's parameter, read from the tool's call site."""
    assert route in tool_routes()[tool], sorted(tool_routes()[tool])


def test_every_product_is_walked():
    products = {product for routes in tool_routes().values() for product, _m, _p in routes}
    assert products == {"miworkbench", "miserver", "dataworks", "miforge"}, products


def test_normalise():
    assert normalise("/a/{x}/b?q=1") == "/a/{}/b"
    assert normalise("/models/{repo_id:path}/") == "/models/{}"
    assert normalise("/") == "/"


# ── returns_unchanged (the generator's classifier) ─────────────────────────────────────────────────────────────


def test_returns_unchanged_reads_every_return_of_the_tool_itself():
    async def plain(client, x):
        return await client.get(f"/things/{x}")

    async def gated(client, x):
        if x:
            return {"refused": "no"}
        return await client.get("/things")

    async def nested(client):
        def helper():
            return {"not": "the tool's"}
        return await client.post("/things", json_body={})

    async def nothing(client):
        await client.get("/things")

    assert returns_unchanged(plain) == ({("miworkbench", "GET", "/things/{}")}, "")
    assert returns_unchanged(gated)[0] is None and "changes its answer" in returns_unchanged(gated)[1]
    assert returns_unchanged(nested) == ({("miworkbench", "POST", "/things")}, "")
    assert returns_unchanged(nothing) == (None, "the tool returns no route's body")


def test_returns_unchanged_names_the_relayed_product():
    async def relay(miserver, x):
        return await miserver.get(f"/api/probes/{x}")

    async def raw(miserver, x):
        return await miserver.raw_get(f"/api/clusters/{x}/export")

    async def dataworks(dw):
        return await dw.get("/datasets")

    async def full(miserver):
        return await miserver.get("/api/models", full=True)  # a MiServerResponse, not the body

    async def forge_runs(forge):
        return await forge.get("/api/v1/runs")

    async def other(stranger):
        return await stranger.get("/api/v1/runs")

    assert returns_unchanged(relay) == ({("miserver", "GET", "/api/probes/{}")}, "")
    assert returns_unchanged(raw) == ({("miserver", "GET", "/api/clusters/{}/export")}, "")
    assert returns_unchanged(dataworks) == ({("dataworks", "GET", "/datasets")}, "")
    assert returns_unchanged(full)[0] is None
    assert returns_unchanged(forge_runs) == ({("miforge", "GET", "/api/v1/runs")}, "")
    assert returns_unchanged(other)[0] is None


def test_a_route_call_on_an_unknown_receiver_is_reported(tmp_path, monkeypatch):
    """review round 1, finding 5: a verb with a route path on a receiver the walk does not know is unreadable, never
    silently skipped (the miForge run-list calls were invisible until `forge` was mapped)."""
    from src.mcp_server import route_map

    planted = tmp_path / "planted.py"
    planted.write_text(
        "def register(mcp):\n"
        "    @mcp.tool()\n"
        "    async def planted_tool():\n"
        "        return await stranger.get('/api/v1/hidden')\n")
    routes, unreadable = route_map._module_routes(planted)  # noqa: SLF001
    assert routes == {"planted_tool": set()}
    assert unreadable == ["planted.py:4 (receiver stranger)"]
