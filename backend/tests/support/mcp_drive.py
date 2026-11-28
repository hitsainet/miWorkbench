"""Drive every MCP tool through the SDK's own in-memory client (045 FR-1 – FR-4).

`test_tool_classes.py` and `test_mcp_output_schemas.py` need the LIVE registry of every category, built the way
`howto._all_tools` builds it (one server per category, placeholder product URLs, stdio, anonymous), and
`test_mcp_output_schemas.py` needs to CALL each tool and let the SDK client validate the answer against the tool's
advertised output schema. Both read the wire, never the table alone.

The HTTP side is a `httpx.MockTransport` whose answer the caller sets (`Backend.body`), patched into every product
client (miWorkbench, miServer, miDataworks) and the health gate, as `test_miserver_next_reaches_agent.py` does. A
tool is called with arguments generated from its advertised `inputSchema` (required fields only), so a tool that
refuses those arguments answers an error result, which the SDK client does not validate; the callers count how many
answered without an error so a harness that reached nothing cannot pass.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import httpx

#: A body for each of the four shapes the backend can answer (FPRD FR-4).
EMPTY_OBJECT: Any = {}
EMPTY_LIST: Any = []


class Backend:
    """The mock backend every client talks to. `body` is what a non-health request answers (200, JSON)."""

    def __init__(self) -> None:
        self.body: Any = {}
        #: `None`: each route's own 2xx from its contract (`contract_fakes.own_status`, 046 FTASKS 7.3).
        self.status: Optional[int] = None
        self.down = False
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        from tests.support.contract_fakes import own_status, product_of

        self.requests.append(request)
        if request.url.path.endswith("/health"):
            if self.down:
                return httpx.Response(503, json={"status": "down"})
            return httpx.Response(200, json={"status": "healthy"})
        if self.down:
            raise httpx.ConnectError("backend down (mcp_drive)")
        status = self.status if self.status is not None else own_status(
            product_of(request.url.host), request.method, request.url.path, self.body)
        return httpx.Response(status, json=self.body)


class OffContractTransport(httpx.AsyncBaseTransport):
    """A fake that DELIBERATELY answers outside its route's contract, to prove a check downstream refuses or
    repairs it (a wrong-typed field the SDK client must refuse, a backend `approval` the result language must
    repair). It is not an `httpx.MockTransport`, so the fake check (`contract_fakes.py`, 046 WT3) does not judge it:
    the test states, by choosing it and naming `reason`, that the answer is not one the backend can give. Every
    other fake goes through `MockTransport` and is judged."""

    def __init__(self, handler, reason: str) -> None:
        if not reason or not reason.strip():
            raise ValueError("an off-contract fake must say why its answer is outside the contract")
        self.handler, self.reason = handler, reason

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()  # as MockTransport does: a handler may read a streamed (multipart) body
        response = self.handler(request)
        await response.aread()
        return response


def transport_for(handler, off_contract: "str | None" = None) -> httpx.AsyncBaseTransport:
    return OffContractTransport(handler, off_contract) if off_contract is not None else httpx.MockTransport(handler)


@contextmanager
def patched_http(monkeypatch, off_contract: "str | None" = None) -> Iterator[Backend]:
    """Route every MCP-side `httpx.AsyncClient` to one `Backend`. `off_contract=<why>` marks a test whose answers
    are deliberately outside their contracts (`OffContractTransport`)."""
    from src.mcp_server import client as wb_client
    from src.mcp_server import dataworks_client, health_gate, miserver_client

    backend = Backend()
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport_for(backend.handler, off_contract)
        return real(*args, **kwargs)

    for module in (wb_client, miserver_client, dataworks_client, health_gate):
        monkeypatch.setattr(module.httpx, "AsyncClient", factory)
    yield backend


class RouteBackend:
    """A mock backend that answers PER ROUTE (046 FTDD §6.1, FTID §11), for tests that must keep their payload and
    call-count assertions: `routes[(METHOD, normalised path)] = (status, body, headers)`, keyed by the path the request
    carries (miWorkbench's under `/api/v1`, miServer's as its contract keys them). Every request is recorded as
    `(METHOD, path, json body)` in `calls`.

    `/health` reads answer healthy (the health gate's probe). AN UNMAPPED ROUTE NEVER ANSWERS `{}`: it answers 599 and
    is recorded in `unmapped`, and `patched_routes` fails the test on exit naming it (a fake that answers anything for
    any route is how a suite agrees with itself by construction)."""

    UNMAPPED_STATUS = 599

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], tuple[int, Any, dict[str, str]]] = {}
        self.calls: list[tuple[str, str, Any]] = []
        self.requests: list[httpx.Request] = []  # each request whole (its query parameters, its headers)
        self.unmapped: list[str] = []
        self.down = False

    def answer(self, method: str, path: str, body: Any, status: Optional[int] = None,
               headers: dict[str, str] | None = None) -> "RouteBackend":
        from src.mcp_server.route_map import normalise

        self.routes[(method.upper(), normalise(path))] = (status, body, dict(headers or {}))
        return self

    def handler(self, request: httpx.Request) -> httpx.Response:
        import json


        try:
            sent = json.loads(request.content) if request.content else None
        except ValueError:
            sent = request.content
        self.calls.append((request.method, request.url.path, sent))
        self.requests.append(request)
        if request.url.path.endswith("/health"):
            if self.down:
                return httpx.Response(503, json={"status": "down"})
            return httpx.Response(200, json={"status": "healthy"})
        if self.down:
            raise httpx.ConnectError("backend down (mcp_drive.RouteBackend)")
        key = self.match(request.method, request.url.path)
        if key is None:
            self.unmapped.append(f"{request.method} {request.url.path}")
            return httpx.Response(self.UNMAPPED_STATUS,
                                  json={"detail": f"unmapped route {request.method} {request.url.path}"})
        status, body, headers = self.routes[key]
        if status is None:  # the route's own 2xx from its contract (046 FTASKS 7.3)
            from tests.support.contract_fakes import own_status, product_of

            status = own_status(product_of(request.url.host), request.method, request.url.path, body)
        if isinstance(body, (str, bytes)):
            return httpx.Response(status, content=body if isinstance(body, bytes) else body.encode(),
                                  headers=headers)
        return json_response(status, body, headers)

    def match(self, method: str, path: str) -> "tuple[str, str] | None":
        """The registered `(METHOD, template)` a request path matches: `{}` is one path segment; the template with the
        fewest parameters wins (`/models/active` before `/models/{}`)."""
        from src.mcp_server.route_map import normalise

        segments = normalise(path).split("/")
        found = []
        for m, template in self.routes:
            parts = template.split("/")
            if m == method.upper() and len(parts) == len(segments) and all(
                    t == "{}" or t == p for t, p in zip(parts, segments, strict=True)):
                found.append((parts.count("{}"), template))
        return (method.upper(), min(found)[1]) if found else None

    def calls_to(self, method: str, path: str) -> list[Any]:
        """The JSON bodies sent to one route, in order (the payload and call-count assertions read this)."""
        from src.mcp_server.route_map import normalise

        template = normalise(path)
        return [body for m, p, body in self.calls if m == method.upper() and self.match(m, p) == (m, template)]


@contextmanager
def patched_routes(monkeypatch, backend: "RouteBackend | None" = None) -> Iterator[RouteBackend]:
    """Route every MCP-side `httpx.AsyncClient` to one `RouteBackend`; on exit, fail naming any unmapped route."""
    from src.mcp_server import client as wb_client
    from src.mcp_server import dataworks_client, health_gate, miserver_client

    backend = backend or RouteBackend()
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(backend.handler)
        return real(*args, **kwargs)

    for module in (wb_client, miserver_client, dataworks_client, health_gate):
        monkeypatch.setattr(module.httpx, "AsyncClient", factory)
    yield backend
    assert not backend.unmapped, f"requests to routes the fake does not answer: {sorted(set(backend.unmapped))}"


def placeholder_overrides() -> dict[str, str]:
    from src.mcp_server.tools import EXTERNAL_PRODUCTS

    return {p.spec.override_field: f"http://{p.spec.name}.invalid" for p in EXTERNAL_PRODUCTS}


def build_category(category: str):
    """One server with one category, as `howto._all_tools` builds it."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, _client = build_server(
        MCPSettings(tool_categories=category, allow_anonymous=True, _env_file=None,
                    **placeholder_overrides()),
        stdio=True,
    )
    return mcp


def build_every_category() -> dict[str, Any]:
    from src.mcp_server.tools import all_categories

    return {category: build_category(category) for category in all_categories()}


def run(coro):
    return asyncio.run(coro)


def wire_tools(mcp) -> list:
    """`tools/list` as a client reads it (the SDK in-memory client)."""
    from mcp import Client

    async def go():
        async with Client(mcp) as client:
            return (await client.list_tools()).tools

    return run(go())


def _sample(schema: dict, defs: dict) -> Any:
    if not isinstance(schema, dict):
        return "x"
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return _sample(defs.get(ref.split("/")[-1], {}), defs)
    for key in ("anyOf", "oneOf"):
        options = [s for s in schema.get(key) or [] if s.get("type") != "null"]
        if options:
            return _sample(options[0], defs)
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "string":
        if "pattern" in schema:  # 046 P7: an id or digest argument (`code_sha256`) the tool would otherwise refuse
            import re

            from tests.support.answer_samples import _PATTERN_CANDIDATES

            found = next((c for c in _PATTERN_CANDIDATES if re.search(schema["pattern"], c)), None)
            if found is not None:
                return found
        return "x" * max(1, int(schema.get("minLength", 1)))
    if kind == "integer":
        return int(max(schema.get("minimum", 1), schema.get("exclusiveMinimum", 0) + 1))
    if kind == "number":
        low = schema.get("minimum", schema.get("exclusiveMinimum", 0))
        high = schema.get("maximum", schema.get("exclusiveMaximum", low + 1))
        return (low + high) / 2
    if kind == "boolean":
        return False
    if kind == "array":
        return [_sample(schema.get("items", {}), defs) for _ in range(int(schema.get("minItems", 0)))]
    if kind == "object":
        props = schema.get("properties") or {}
        return {k: _sample(props[k], defs) for k in schema.get("required", []) if k in props}
    return "x"


def sample_arguments(tool) -> dict:
    """Required arguments only, each the first value its schema admits."""
    schema = tool.input_schema or {}
    defs = schema.get("$defs") or {}
    props = schema.get("properties") or {}
    return {name: _sample(props[name], defs) for name in schema.get("required", []) if name in props}


def call(mcp, name: str, arguments: dict):
    """`tools/call` through the SDK client, which validates structured content against the output schema and
    RAISES when it does not conform (`mcp/client/session.py:1118-1153`)."""
    from mcp import Client

    async def go():
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments)

    return run(go())


class RecordedVerb:
    """A REAL client verb that records each call (`calls`, as `unittest.mock.call` objects), for tests that assert
    a tool's payload and call count. The call still goes through the client and its transport, so the answer is a
    route fake's (judged by the fake check), never a value a mock returned (046 FTASKS 3.4)."""

    def __init__(self, method) -> None:
        self._method = method
        self.calls: list = []

    async def __call__(self, *args, **kwargs):
        from unittest.mock import _Call

        self.calls.append(_Call((args, kwargs), two=True))  # what a mock's `call_args` is: (args, kwargs)
        return await self._method(*args, **kwargs)

    @property
    def await_count(self) -> int:
        return len(self.calls)

    @property
    def call_args(self):
        return self.calls[-1] if self.calls else None

    @property
    def await_args(self):
        return self.call_args

    @property
    def await_args_list(self) -> list:
        return list(self.calls)

    def assert_not_called(self) -> None:
        assert not self.calls, f"expected no call, got {self.calls}"

    def assert_awaited_once(self) -> None:
        assert len(self.calls) == 1, f"expected one call, got {self.calls}"

    def assert_awaited_once_with(self, *args, **kwargs) -> None:
        from unittest.mock import call

        assert self.calls == [call(*args, **kwargs)], f"expected one {call(*args, **kwargs)}, got {self.calls}"



CLIENT_VERBS = ("get", "post", "put", "patch", "delete", "raw_get", "request_full")


def recorded_client(client, backend: "RouteBackend", base_url: str, off_contract: "str | None" = None):
    """`client` (a real `MiServerClient`, `MiWorkbenchClient` or `DataworksClient`) on `backend` over a
    `MockTransport` (or, with `off_contract=<why>`, the `OffContractTransport`), each verb a `RecordedVerb`."""
    client._http = httpx.AsyncClient(base_url=base_url,  # noqa: SLF001
                                     transport=transport_for(backend.handler, off_contract))
    for verb in CLIENT_VERBS:
        if hasattr(client, verb):
            setattr(client, verb, RecordedVerb(getattr(client, verb)))
    return client


class ContractBackend(RouteBackend):
    """A `RouteBackend` for miWorkbench whose UNMAPPED routes answer their contract's neutral full body (046 FTASKS
    7.5): the route's own 2xx, every field it always sends filled in, nothing else. For a migrated test that asserts a
    tool's payload and call count and does not care what the route answered: the answer is still one the route can
    give, and the fake check judges it. A route the contract does not cover still answers 599 and fails the test.
    `answer(...)` maps a route to a scripted body as in `RouteBackend`."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health") or self.down or self.match(request.method, request.url.path):
            return super().handler(request)
        from tests.support.fake_clients import _entry, workbench_answer

        if _entry(request.method, request.url.path) is None:
            return super().handler(request)  # not a covered route: 599, named on exit
        self.calls.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        self.requests.append(request)
        from tests.support.answer_samples import sample
        from tests.support.contract_fakes import own_status
        from tests.support.fake_clients import _contract

        status = own_status("miworkbench", request.method, request.url.path)
        schema = (_entry(request.method, request.url.path) or {}).get(str(status))
        body = sample(schema, {"$defs": _contract()["$defs"]}, prefer_null=True) if schema else {}
        if isinstance(body, dict):
            body = workbench_answer(request.method, request.url.path, body)
        return json_response(status, body)


def json_response(status: int, body: Any, headers: Any = None) -> httpx.Response:
    """A JSON answer. `None` is the JSON `null` a nullable route sends (`httpx.Response(json=None)` sends NO body,
    which the fake check rightly refuses on a JSON route: 046-2 review round 2, finding 1); a 204 stays bodiless."""
    if body is None and status != 204:
        return httpx.Response(status, content=b"null", headers={"content-type": "application/json", **(headers or {})})
    return httpx.Response(status, json=body, headers=headers)


def recorded_workbench(backend: "RouteBackend | None" = None):
    """A REAL `MiWorkbenchClient` over `backend` (default: a `ContractBackend`), each verb a `RecordedVerb`: the
    migration of a mock client whose test reads `client.<verb>.call_args` (046 FTASKS 7.5). Returns
    `(client, backend)`."""
    from src.mcp_server.client import MiWorkbenchClient

    backend = backend if backend is not None else ContractBackend()
    base = "http://miworkbench.invalid"
    client = recorded_client(MiWorkbenchClient(base), backend, f"{base}/api/v1")
    return client, backend
