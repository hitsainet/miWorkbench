"""The LIVE tool registry, joined to the one route walker (`src/mcp_server/route_map.py`; 046 WT1).

Membership comes from the authority: `registered_tools()` builds the REAL server with every category enabled and
reads `list_tools()` — the same call an agent makes. `route_map` only ENRICHES a registered tool with the routes it
calls (`list_tools()` cannot report endpoints); it never decides what is registered. A tool whose registration line
is deleted therefore drops out of the covered set and its routes go uncovered, which is the red the parity guards
exist for (memory: derive-from-the-registry-not-the-source, source-scrape-guards-fail-open).

Moved from `tests/support/mcp_route_map.py` (deleted in 046 P1), whose walker moved into `src/mcp_server/route_map.py`.
"""
from __future__ import annotations

import asyncio

from src.mcp_server.route_map import normalise, tool_routes, unreadable_calls  # noqa: F401  (re-exported)


def registered_tools(monkeypatch) -> set[str]:
    """Every tool on the REAL built server with EVERY category enabled and every product's URL
    set — the set the agent sees, not the set the source declares."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    for spec in PRODUCTS:
        monkeypatch.setenv(spec.url_env, f"http://{spec.name}.test")
    settings = MCPSettings(
        tool_categories=",".join(sorted(all_categories())), allow_anonymous=True
    )
    mcp, _client = build_server(settings, stdio=True)
    return {t.name for t in asyncio.run(mcp.list_tools())}


def called_routes(registered: set[str], product: str, prefix: str = "") -> dict[tuple[str, str], set[str]]:
    """`{(METHOD, prefix + normalised path): {tools}}` for REGISTERED tools calling `product`."""
    routes = tool_routes()
    out: dict[tuple[str, str], set[str]] = {}
    for tool in registered:
        for prod, method, path in routes.get(tool, ()):
            if prod == product:
                out.setdefault((method, normalise(prefix + path)), set()).add(tool)
    return out
