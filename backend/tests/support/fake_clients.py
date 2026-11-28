"""Plain-object client fakes, put behind a transport (046 FTASKS 7.5, FTDD §6.4).

Many MCP tests fake miWorkbench with a small object whose `get`/`post`/... record the call and return a body. Handed
to a tool as the client, such an answer never crosses a transport, so the fake check (`contract_fakes.py`) never
judges it, and a fixture can agree with the tool by construction. `workbench_over(fake)` keeps the test's object, its
recorded calls and its answers, and puts a REAL `MiWorkbenchClient` in front of it: each call the tool makes goes out
through an `httpx.MockTransport`, whose handler hands the call (with its original Python arguments, not their query
string form) to the fake's verb and answers its return value as JSON. So every answer is judged against the route's
contract entry like any other fake's, and the real client parses, unwraps and refuses it as it would in production.

- The answer's status is the route's own 2xx from the committed contract (one declared status; or, for a route that
  declares the pending `202` beside its own, `202` for a pending body and the own status otherwise). A route the
  contract does not cover answers `200`.
- A fake that raises `BackendError(status, detail)` answers that status with `{"detail": detail}`, which the real client
  turns back into the same `BackendError`; status `0` is a refused connection ("unreachable").
- `complete=True` fills each answer's missing REQUIRED fields from the contract entry (`answer_samples.complete`),
  for a fake that scripts only the fields its test reads; the fake's own keys and values are kept, so an undeclared
  key or a wrong type still fails.
"""
from __future__ import annotations

import contextvars
import functools
import json
from typing import Any, Optional

import httpx

_CALL: contextvars.ContextVar = contextvars.ContextVar("fake_clients_call", default=None)
BASE = "http://miworkbench.invalid"


@functools.lru_cache(maxsize=1)
def _contract() -> dict[str, Any]:
    from src.api.answers_contract import DEFAULT_PATH

    return json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))


def _entry(method: str, path: str) -> Optional[dict[str, Any]]:
    from tests.support.contract_fakes import _key_for

    found = _key_for("miworkbench", method, path)
    if found is None or found[0] != "responses":
        return None
    return _contract()["responses"][found[1]]


def _status(method: str, path: str, value: Any) -> int:
    from tests.support.contract_fakes import own_status

    return own_status("miworkbench", method, path, value)


def _completed(method: str, path: str, status: int, value: Any) -> Any:
    from tests.support.answer_samples import complete

    entry = _entry(method, path) or {}
    schema = entry.get(str(status))
    if not schema or not isinstance(value, (dict, list)):
        return value
    contract = _contract()
    return complete(schema, value, {"$defs": contract["$defs"]})


def workbench_over(fake: Any, *, complete: bool = False) -> Any:
    """A real `MiWorkbenchClient` whose transport answers every request from `fake`'s verb (see the module text)."""
    from src.mcp_server.client import BackendError, MiWorkbenchClient

    async def handle(request: httpx.Request) -> httpx.Response:
        call = _CALL.get()
        if call is None:
            raise AssertionError(f"{request.method} {request.url.path}: a request the fake adapter did not start")
        verb, args, kwargs = call
        try:
            value = await getattr(fake, verb)(*args, **kwargs)
        except BackendError as error:
            if not error.status_code:  # a refused connection: the real client words it again, so pass its cause only
                cause = str(error.detail).rsplit("unreachable: ", 1)[-1]
                raise httpx.ConnectError(cause, request=request) from error
            body = {"detail": error.detail}
            return httpx.Response(error.status_code, json=body, headers=error.headers or None)
        path = request.url.path
        status = _status(request.method, path, value)
        if complete:
            value = _completed(request.method, path, status, value)
        return httpx.Response(status, json=value)

    class Over(MiWorkbenchClient):
        def __init__(self) -> None:
            super().__init__(BASE)
            headers = dict(self._http.headers)
            self._http = httpx.AsyncClient(base_url=f"{BASE}/api/v1", transport=httpx.MockTransport(handle),
                                           headers=headers)
            self.fake = fake

        async def _via(self, verb: str, args: tuple, kwargs: dict, run: Any) -> Any:
            token = _CALL.set((verb, args, kwargs))
            try:
                return await run()
            finally:
                _CALL.reset(token)

        # Each verb hands the fake EXACTLY the arguments the tool passed (a test's payload assertions read them).
        async def get(self, path: str, *args: Any, **kwargs: Any) -> Any:
            return await self._via("get", (path, *args), kwargs, lambda: super(Over, self).get(path, *args, **kwargs))

        async def post(self, path: str, *args: Any, **kwargs: Any) -> Any:
            return await self._via("post", (path, *args), kwargs, lambda: super(Over, self).post(path, *args, **kwargs))

        async def patch(self, path: str, *args: Any, **kwargs: Any) -> Any:
            return await self._via("patch", (path, *args), kwargs,
                                   lambda: super(Over, self).patch(path, *args, **kwargs))

        async def put(self, path: str, *args: Any, **kwargs: Any) -> Any:
            return await self._via("put", (path, *args), kwargs, lambda: super(Over, self).put(path, *args, **kwargs))

        async def delete(self, path: str, *args: Any, **kwargs: Any) -> Any:
            return await self._via("delete", (path, *args), kwargs,
                                   lambda: super(Over, self).delete(path, *args, **kwargs))

        async def request(self, method: str, path: str, **kwargs: Any) -> Any:
            if _CALL.get() is None:  # a tool calling `request` itself: the fake answers through its `request`
                return await self._via("request", (method, path), kwargs,
                                       lambda: super(Over, self).request(method, path, **kwargs))
            return await super().request(method, path, **kwargs)

    return Over()


def workbench_answer(method: str, path: str, body: Any = None) -> Any:
    """A miWorkbench route's answer as it sends it (046 FTASKS 7.3): `body`'s own fields, every other field its
    contract entry always sends filled in (at its own status). For a fake that scripts only what its test reads."""
    value = {} if body is None else body
    status = _status(method.upper(), path, value)
    return _completed(method.upper(), path, status, value)


MISERVER_BASE = "http://miserver.invalid"


def miserver_over(fake: Any) -> Any:
    """A real `MiServerClient` whose transport answers every request from `fake`'s verb (046 FTASKS 7.5), as
    `workbench_over` does for miWorkbench. The fake answers what the client's verb returns: the envelope's `data`, or
    (for a `full=True` call) a `MiServerResponse`; the handler sends it AS MISERVER SENDS IT, a management route's
    `data` in its envelope with every field its contract always sends filled in (`miserver_fakes.as_sent`), so the
    fake check judges it and the real client unwraps it. A raised `BackendError` answers its status with miServer's
    refusal envelope."""
    import httpx

    from src.mcp_server.client import BackendError
    from src.mcp_server.miserver_client import MiServerClient, MiServerResponse
    from tests.support.contract_fakes import own_status
    from tests.support.miserver_fakes import complete_response

    async def handle(request: httpx.Request) -> httpx.Response:
        call = _CALL.get()
        if call is None:
            raise AssertionError(f"{request.method} {request.url.path}: a request the fake adapter did not start")
        verb, args, kwargs = call
        try:
            value = await getattr(fake, verb)(*args, **kwargs)
        except BackendError as error:
            if not error.status_code:  # a refused connection: the real client words it again, so pass its cause only
                cause = str(error.detail).rsplit("unreachable: ", 1)[-1]
                raise httpx.ConnectError(cause, request=request) from error
            detail = error.detail if isinstance(error.detail, dict) else {"code": "ERROR", "message": str(error.detail)}
            body = {"success": False, "data": None, "error": {"code": detail.get("code", "ERROR"),
                                                               "message": detail.get("message", ""), **detail}}
            return httpx.Response(error.status_code, json=body, headers=error.headers or None)
        headers: dict = {}
        if isinstance(value, MiServerResponse):
            status, data, headers = value.status, value.body, dict(value.headers or {})
        else:
            status, data = own_status("miserver", request.method, request.url.path, value), value
        path = request.url.path
        sent = data if path.startswith("/v1/") else {"success": True, "data": data, "error": None}
        return complete_response(request, httpx.Response(status, json=sent, headers=headers, request=request))

    class Over(MiServerClient):
        def __init__(self) -> None:
            super().__init__(MISERVER_BASE)
            self._http = httpx.AsyncClient(base_url=MISERVER_BASE, transport=httpx.MockTransport(handle),
                                           headers=dict(self._http.headers))
            self.fake = fake

        async def _via(self, verb: str, args: tuple, kwargs: dict, run: Any) -> Any:
            token = _CALL.set((verb, args, kwargs))
            try:
                return await run()
            finally:
                _CALL.reset(token)

        async def get(self, path: str, *args: Any, **kwargs: Any) -> Any:
            return await self._via("get", (path, *args), kwargs, lambda: super(Over, self).get(path, *args, **kwargs))

        async def post(self, path: str, *args: Any, **kwargs: Any) -> Any:
            return await self._via("post", (path, *args), kwargs, lambda: super(Over, self).post(path, *args, **kwargs))

        async def put(self, path: str, *args: Any, **kwargs: Any) -> Any:
            return await self._via("put", (path, *args), kwargs, lambda: super(Over, self).put(path, *args, **kwargs))

        async def delete(self, path: str, *args: Any, **kwargs: Any) -> Any:
            return await self._via("delete", (path, *args), kwargs,
                                   lambda: super(Over, self).delete(path, *args, **kwargs))

    return Over()
