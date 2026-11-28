"""Drive MCP tools against scripted HTTP answers, for the result-language tests (045 FR-7 – FR-9).

`mcp_drive` answers every request with one body; these tests need a different answer per route (a filing, then a
consume, then the job). A `Router` holds `(method, path regex) → answer` rules and records every request, so a test
asserts the payload a route received and how many times it was called (memory `mutation-survivors-cluster-on-callers`).
Every MCP-side `httpx.AsyncClient` (miWorkbench, miServer, miDataworks, the health gate) is pointed at it, and `/health`
answers healthy unless a rule says otherwise. No test here reaches a network, a database or Celery.
"""
from __future__ import annotations

import asyncio
import json
import re
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

import httpx

from tests.support.mcp_drive import placeholder_overrides


class Router:
    def __init__(self) -> None:
        self.rules: list[tuple[str, re.Pattern, Callable[[httpx.Request], httpx.Response]]] = []
        self.requests: list[httpx.Request] = []

    def on(self, method: str, path: str, status: Optional[int] = None, body: Any = None,
           headers: Optional[dict[str, str]] = None) -> "Router":
        """Answer `method` on a path matching the regex `path` (searched, so a suffix is enough). With no `status`, the
        route's own 2xx from its contract (`contract_fakes.own_status`, 046 FTASKS 7.3)."""
        def answer(request: httpx.Request) -> httpx.Response:
            from tests.support.contract_fakes import own_status, product_of

            code = status if status is not None else own_status(product_of(request.url.host), request.method,
                                                                request.url.path, body)
            return httpx.Response(code, json=body, headers=headers or {})
        self.rules.insert(0, (method.upper(), re.compile(path), answer))
        return self

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for method, pattern, answer in self.rules:
            if request.method == method and pattern.search(request.url.path):
                # An off-contract source is still completed (its route's always-sent fields beneath its own); it is
                # only not judged (`mcp_drive.OffContractTransport`).
                return _as_miserver_sends_it(request, answer(request))
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "healthy"})
        return httpx.Response(200, json={})

    def calls(self, method: str, path: str) -> list[httpx.Request]:
        pattern = re.compile(path)
        return [r for r in self.requests if r.method == method.upper() and pattern.search(r.url.path)]

    @staticmethod
    def body(request: httpx.Request) -> Any:
        return json.loads(request.content or b"null")


def _as_miserver_sends_it(request: httpx.Request, response: httpx.Response) -> httpx.Response:
    """A rule's answer to miServer, as miServer sends it (`miserver_fakes.as_sent`, 046 WT3)."""
    from tests.support.miserver_fakes import complete_response

    return complete_response(request, response)


@contextmanager
def routed(monkeypatch, off_contract: Optional[str] = None) -> Iterator[Router]:
    """`off_contract=<why>`: this test's sources deliberately answer outside their contracts (a backend's own
    `approval` or `next_steps`, which miServer's contract does not declare), to prove the result language repairs or
    drops them (`mcp_drive.OffContractTransport`). Its answers are then not completed or judged."""
    from src.mcp_server import client as wb_client
    from src.mcp_server import dataworks_client, health_gate, miserver_client
    from tests.support.mcp_drive import transport_for

    router = Router()
    router.off_contract = off_contract
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport_for(router.handler, off_contract)
        return real(*args, **kwargs)

    for module in (wb_client, miserver_client, dataworks_client, health_gate):
        monkeypatch.setattr(module.httpx, "AsyncClient", factory)
    yield router


def build(categories: str, **settings: Any):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, _client = build_server(
        MCPSettings(tool_categories=categories, allow_anonymous=True, _env_file=None,
                    **{**placeholder_overrides(), **settings}),
        stdio=True,
    )
    return mcp


def call(mcp, name: str, arguments: dict):
    """`tools/call` through the SDK in-memory client (which validates structured content against the output
    schema and raises when it does not conform)."""
    from mcp import Client

    async def go():
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(go())


def raw_tool_fn(mcp, name: str):
    """The tool's own function, under the 045 wrappers (`speak_result_language`, `normalise_results`) but ABOVE any
    decorator the tool module applied itself (`gated`): what the tool answered before phase B."""
    fn = mcp._tool_manager.get_tool(name).fn  # noqa: SLF001
    while getattr(fn, "speaks_result_language", False) or getattr(fn, "normalises_list_results", False):
        fn = fn.__wrapped__
    return fn


def unwrapped(mcp, name: str, arguments: dict) -> Any:
    """What the tool answers without the result-language wrapper, for the same arguments (defaults filled the way
    the SDK fills them)."""
    tool = mcp._tool_manager.get_tool(name)  # noqa: SLF001
    validated = tool.fn_metadata.validate_arguments(arguments)
    fn = raw_tool_fn(mcp, name)
    return asyncio.run(fn(**validated))


def contract() -> dict:
    from src.mcp_server.result_language import load_contract

    return load_contract()


def validator(definition: str):
    """A draft 2020-12 validator for one `$def` of the vendored contract."""
    import jsonschema

    c = contract()
    return jsonschema.Draft202012Validator({"$defs": c["$defs"], "$ref": f"#/$defs/{definition}"})
