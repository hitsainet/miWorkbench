"""The dynamic backstop's machinery (045 FR-14, FR-18; FTDD §16.4, §18; CHAIN_PLAN A27, A40, A42, A43; FTASKS 21.9, 22.1).

Static analysis cannot see every dispatch (a `getattr`, a registry lookup), so the backstop CALLS the live routes and
watches what they reach:

* `install_primitives(monkeypatch)` replaces every primitive of both categories with a recorder that raises
  `PrimitiveReached`: the GPU starts (`dispatch_gpu_task`, `gpu_delay`, `logit_lens_lease`, `ensure_steering_worker`,
  `ensure_model_loaded`, `celery.app.task.Task.delay` / `.apply_async`, `Celery.send_task`,
  `BackgroundTasks.add_task`) and the outside paid calls (`ask_judge`, `_client_for`), wherever a module bound them
  (a `from x import f` copy is replaced too, by identity). At the HTTP transports (httpx, httpx2, requests) a request
  to miServer's `/api/models/{id}/load`, `/v1/chat/completions` or `/v1/completions` is a primitive; any other
  outbound request is recorded and refused. `subprocess.Popen` and `os.kill` are refused (a `celery` worker spawn is a primitive; any other command is recorded and fails as
  not found). Nothing is queued for
  real: no message reaches the shared local Redis.
* `blocked_network(monkeypatch)`: no name resolves except the local database / Redis hosts, so no test reaches the
  network whatever a route does (OVN-mcp-workbench-31: one run once reached the real Hub).

`call(app, method, path, ...)` sends one request through `httpx.ASGITransport` (a different class from the patched
transports) with `raise_app_exceptions=False`, so a primitive's raise is a 500 and the recorder holds what was reached.
"""
from __future__ import annotations

import re
import socket
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import httpx

GPU_FUNCTIONS = {
    "dispatch_gpu_task": "src.services.gpu_dispatch",
    "gpu_delay": "src.services.gpu_dispatch",
    "logit_lens_lease": "src.services.logit_lens_service",
    "ensure_steering_worker": "src.api.v1.steering_workers",
    "ensure_model_loaded": "src.utils.miserver_utils",
}
OUTSIDE_FUNCTIONS = {
    "ask_judge": "src.services.probe_monitor_judge",
    "_client_for": "src.services.probe_monitor_judge",
}
#: miServer's model-load and generation paths: a request to one is a GPU start (or a paid generation) on miServer.
MISERVER_PRIMITIVE = re.compile(r"(/api/models/[^/]+/load|/v1/chat/completions|/v1/completions)$")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", None)


def is_hub_host(host: Optional[str]) -> bool:
    host = (host or "").lower()
    return host in ("huggingface.co", "hf.co") or host.endswith((".huggingface.co", ".hf.co"))


class PrimitiveReached(Exception):
    """Raised by every patched primitive; the route answers 500 and the recorder holds the name."""


@dataclass
class Primitives:
    reached: list[str] = field(default_factory=list)
    outbound: list[str] = field(default_factory=list)
    spawned: list[str] = field(default_factory=list)
    #: (method, url, Authorization) of every request to a Hub host made outside huggingface_hub's own client.
    hub_direct: list[tuple] = field(default_factory=list)

    def clear(self) -> None:
        self.reached.clear()
        self.outbound.clear()
        self.spawned.clear()
        self.hub_direct.clear()


def _replace_everywhere(monkeypatch, module_name: str, name: str, replacement: Any) -> int:
    """Replace `module.name` and every `src.*` module's binding of the SAME object. Returns how many."""
    import importlib

    module = importlib.import_module(module_name)
    original = getattr(module, name)
    count = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not (mod_name == module_name or mod_name.startswith("src.")):
            continue
        try:
            bound = vars(mod).get(name)
        except TypeError:
            continue
        if bound is original:
            monkeypatch.setattr(mod, name, replacement)
            count += 1
    return count


def install_primitives(monkeypatch, *, gpu: Optional[Iterable[str]] = None,
                       outside: Optional[Iterable[str]] = None) -> Primitives:
    """Every primitive records and raises. `gpu` / `outside` narrow the function primitives (controls only)."""
    import inspect
    import subprocess

    import requests.adapters
    from celery import Celery
    from celery.app.task import Task
    from fastapi import BackgroundTasks

    rec = Primitives()

    def recorder(label: str, *, is_async: bool = False):
        if is_async:
            async def _araise(*args, **kwargs):
                rec.reached.append(label)
                raise PrimitiveReached(label)
            return _araise

        def _raise(*args, **kwargs):
            rec.reached.append(label)
            raise PrimitiveReached(label)
        return _raise

    functions = {**{k: v for k, v in GPU_FUNCTIONS.items() if gpu is None or k in gpu},
                 **{k: v for k, v in OUTSIDE_FUNCTIONS.items() if outside is None or k in outside}}
    import importlib

    for name, module_name in functions.items():
        fn = getattr(importlib.import_module(module_name), name)
        n = _replace_everywhere(monkeypatch, module_name, name,
                                recorder(name, is_async=inspect.iscoroutinefunction(fn)))
        assert n >= 1, f"{module_name}.{name} was not replaced"

    def task_recorder(kind: str):
        def _raise(self, *args, **kwargs):
            rec.reached.append(f"{kind}:{getattr(self, 'name', '?')}")
            raise PrimitiveReached(kind)
        return _raise

    monkeypatch.setattr(Task, "delay", task_recorder("Task.delay"))
    monkeypatch.setattr(Task, "apply_async", task_recorder("Task.apply_async"))

    def send_task(self, name, *args, **kwargs):
        rec.reached.append(f"Celery.send_task:{name}")
        raise PrimitiveReached("Celery.send_task")

    monkeypatch.setattr(Celery, "send_task", send_task)

    def add_task(self, func, *args, **kwargs):
        rec.reached.append(f"BackgroundTasks.add_task:{getattr(func, '__name__', func)}")
        raise PrimitiveReached("BackgroundTasks.add_task")

    monkeypatch.setattr(BackgroundTasks, "add_task", add_task)

    def popen(self, args, *a, **k):
        line = args if isinstance(args, str) else " ".join(map(str, args))
        if "celery" in line:  # a worker spawn is a GPU start
            rec.reached.append(f"subprocess.Popen:{line[:80]}")
            raise PrimitiveReached("subprocess.Popen")
        rec.spawned.append(line[:120])
        raise FileNotFoundError(f"backstop: no subprocess ({line[:40]})")

    monkeypatch.setattr(subprocess.Popen, "__init__", popen)

    import os

    def no_kill(pid, sig, *a):  # a route that signals a process (exit-mode's worker kill) signals nothing here
        rec.spawned.append(f"kill:{pid}:{sig}")
        raise ProcessLookupError(pid)

    monkeypatch.setattr(os, "kill", no_kill)
    monkeypatch.setattr(os, "killpg", no_kill)

    def _outbound(request) -> None:
        url = str(request.url)
        path = httpx.URL(url).path
        if MISERVER_PRIMITIVE.search(path):
            rec.reached.append(f"miserver:{path}")
            raise PrimitiveReached("miserver")
        rec.outbound.append(url)
        if is_hub_host(httpx.URL(url).host):  # a Hub request made beside huggingface_hub's own client
            rec.hub_direct.append((request.method, url, request.headers.get("authorization")))

    def _sync(self, request):
        _outbound(request)
        raise httpx.ConnectError("backstop: no outbound request", request=request)

    async def _async(self, request):
        _outbound(request)
        raise httpx.ConnectError("backstop: no outbound request", request=request)

    def _requests(self, request, *a, **k):
        _outbound(request)
        raise requests.exceptions.ConnectionError("backstop: no outbound request")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _sync)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _async)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", _requests)
    try:
        import httpx2

        def _sync2(self, request):
            _outbound(request)
            raise httpx2.ConnectError("backstop: no outbound request", request=request)

        async def _async2(self, request):
            _outbound(request)
            raise httpx2.ConnectError("backstop: no outbound request", request=request)

        monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", _sync2)
        monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", _async2)
    except ImportError:  # pragma: no cover - huggingface_hub 1.x
        pass
    return rec


def blocked_network(monkeypatch) -> None:
    """No name resolves except the local database / Redis hosts: nothing leaves this machine."""
    real = socket.getaddrinfo

    def guarded(host, *args, **kwargs):
        if host not in LOCAL_HOSTS:
            raise OSError(f"blocked by the backstop: no network to {host!r}")
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded)


async def call(app, method: str, path: str, *, query: Optional[dict] = None, body: Any = None,
               headers: Optional[dict] = None, files: Optional[dict] = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=60) as c:
        kwargs: dict[str, Any] = {"params": query or None, "headers": headers or {}}
        if files is not None:
            kwargs["files"] = files
            if isinstance(body, dict):
                kwargs["data"] = body
        elif body is not None:
            kwargs["json"] = body
        return await c.request(method, path, **kwargs)


class HandlerProbe:
    """Records which route handlers were ENTERED (`sys.monitoring` PY_START on each endpoint's own code object, in
    every thread), so the backstop can tell a handler that ran from a request refused before it (a validation 422,
    a dependency): a refusal before the handler is not a pass (A40)."""

    TOOL_IDS = (4, 3)

    def __init__(self, routes: Iterable[Any]):
        import inspect

        self.codes: dict[Any, Any] = {}
        for route in routes:
            endpoint = getattr(route, "endpoint", None)
            if endpoint is not None:
                self.codes[inspect.unwrap(endpoint).__code__] = route
        self.entered: set[Any] = set()
        self.tool = next((t for t in self.TOOL_IDS if sys.monitoring.get_tool(t) is None), None)
        assert self.tool is not None, "no free sys.monitoring tool id"

    def __enter__(self) -> "HandlerProbe":
        mon = sys.monitoring
        mon.use_tool_id(self.tool, "backstop")
        mon.register_callback(self.tool, mon.events.PY_START, self._started)
        for code in self.codes:
            mon.set_local_events(self.tool, code, mon.events.PY_START)
        return self

    def _started(self, code, offset):
        self.entered.add(code)

    def entered_route(self, route) -> bool:
        import inspect

        return inspect.unwrap(route.endpoint).__code__ in self.entered

    def __exit__(self, *exc) -> None:
        mon = sys.monitoring
        for code in self.codes:
            mon.set_local_events(self.tool, code, 0)
        mon.register_callback(self.tool, mon.events.PY_START, None)
        mon.free_tool_id(self.tool)


# ---------------------------------------------------------------------------------------------------------------------
# The sweep: one world, every listed route, in a fixed order (reads, then writes, then destructive verbs and deletes,
# so a route that removes a row runs after the routes that read it).
# ---------------------------------------------------------------------------------------------------------------------

_LATE = ("cancel", "delete", "archive", "purge", "prune", "remove", "reset", "dismiss", "exit", "restart", "revoke",
         "deny", "cleanup")
METHOD_ORDER = {"GET": 0, "HEAD": 0, "POST": 1, "PUT": 1, "PATCH": 1, "DELETE": 3}


def order_key(method: str, path: str) -> tuple:
    late = any(word in path.rsplit("/", 1)[-1] for word in _LATE) or "redownload" in path
    return (METHOD_ORDER.get(method, 2) + (1 if late and method != "DELETE" else 0), path, method)


@dataclass
class Outcome:
    method: str
    path: str
    variant: int
    status: Any
    entered: bool
    reached: list
    spawned: list
    hub: list
    text: str


async def sweep(app, routes: Iterable[tuple[str, str, Any]], requests_for, rec: Primitives, *,
                headers: dict, hub=None) -> list[Outcome]:
    """Send each `(method, template, route)` its fixture request(s) (`requests_for(method, template)` → a list of
    `world_request` dicts) with `headers`; record, per request, whether the route's handler was entered, the
    primitives it reached, and the Hub requests it made (`hub`: a `hub_spy.FakeHub`, plus any direct request to a
    Hub host). One `Outcome` per request (`variant` is its index)."""
    routes = sorted(routes, key=lambda r: order_key(r[0], r[1]))
    out: list[Outcome] = []
    with HandlerProbe([r for _, _, r in routes]) as probe:
        for method, template, route in routes:
            for variant, req in enumerate(requests_for(method, template)):
                rec.clear()
                probe.entered.clear()
                seen_before = len(hub.requests) if hub is not None else 0
                try:
                    response = await call(app, method, req["path"], query=req["query"], body=req["body"],
                                          headers=headers)
                    status, text = response.status_code, response.text[:300]
                except Exception as exc:  # noqa: BLE001 - a route that raises is recorded, not fatal to the sweep
                    status, text = "raised", repr(exc)[:300]
                hub_seen = [(r.method, r.path, r.authorization) for r in (hub.requests[seen_before:] if hub else [])]
                out.append(Outcome(method, template, variant, status, probe.entered_route(route), list(rec.reached),
                                   list(rec.spawned), hub_seen + list(rec.hub_direct), text))
    return out
