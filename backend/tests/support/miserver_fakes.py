"""miServer answers in the shapes its contract declares (`vendored/miserver-answers-v1.json`), for tests that fake it.

`approval_actions(names)` is `GET /api/approvals/actions`'s `data`: the contract's `ApprovalActions`,
`{"actions": [ApprovalActionEntry, …]}`, every field the entry always sends. Since 046 WT10 `_action_names` reads only
this shape and refuses any other (a bare list of names used to pass, which is how a fake and the client agreed with
each other while neither matched miServer)."""
from __future__ import annotations

from typing import Any, Iterable


def approval_actions(names: Iterable[str]) -> dict[str, Any]:
    return {"actions": [{"action": name, "batch_kind": None, "description": f"{name} (a test action)", "label": name,
                         "routes": [], "settings_change": False} for name in names]}


def envelope(data: Any) -> dict[str, Any]:
    """miServer's management envelope around `data` (the client unwraps it, `miserver_client.py:221`)."""
    return {"success": True, "data": data, "error": None}


def _contract() -> dict[str, Any]:
    import functools
    import json
    from pathlib import Path

    @functools.lru_cache(maxsize=1)
    def load() -> dict[str, Any]:
        path = Path(__file__).resolve().parents[2] / "src" / "mcp_server" / "vendored" / "miserver-answers-v1.json"
        return json.loads(path.read_text(encoding="utf-8"))

    return load()


def data_of(key: str, http_status: str = "200", **overrides: Any) -> Any:
    """A complete `data` for one of miServer's management routes (`key` is the contract's `"<METHOD> <path>"`):
    every field the contract says the route always sends, filled from its definition, then `overrides` on top.
    A fake built this way passes the fake check (`tests/support/contract_fakes.py`) where a hand-written partial
    body does not."""
    from tests.support.answer_samples import sample

    contract = _contract()
    schema = contract["responses"][key][http_status]
    target = contract["$defs"][schema["$ref"].split("/")[-1]]
    if "anyOf" in target:  # a named union (a route's answer or its pending answer): the first non-pending one
        target = next(contract["$defs"][a["$ref"].split("/")[-1]] for a in target["anyOf"]
                      if "ApprovalPending" not in a.get("$ref", ""))
    data = target["properties"]["data"]
    options = [a for a in data.get("anyOf", [data]) if a.get("type") != "null"]
    value = sample(options[0], contract) if options else None
    if isinstance(value, dict):
        value = {**value, **overrides}
    return value


def answer(key: str, http_status: str = "200", **overrides: Any) -> dict[str, Any]:
    """`data_of(...)` in miServer's envelope, as the route sends it."""
    return envelope(data_of(key, http_status, **overrides))


def pending_data(**overrides: Any) -> dict[str, Any]:
    """miServer's pending answer's `data` (the contract's `ApprovalPending`), every field filled, `overrides` on top."""
    from tests.support.answer_samples import sample

    return {**sample({"$ref": "#/$defs/ApprovalPending"}, _contract()), **overrides}


MISERVER_HOST = "miserver.invalid"


def _deep_complete(full: Any, body: Any) -> Any:
    """`body` with every key of `full` it lacks added, at every depth (a list's items each completed by `full`'s
    first item). `body`'s own values always win, so a test's fields, and its mistakes, reach the fake check."""
    if isinstance(full, dict) and isinstance(body, dict):
        return {**{k: v for k, v in full.items() if k not in body},
                **{k: _deep_complete(full.get(k), v) if k in full else v for k, v in body.items()}}
    if isinstance(full, list) and isinstance(body, list) and full:
        return [_deep_complete(full[0], item) for item in body]
    return body


def complete_response(request: Any, response: Any, product: Any = None) -> Any:
    """A fake's 2xx answer to one of miServer's routes, AS MISERVER SENDS IT (046 WT3): its management routes in
    their envelope, and every field the contract says the route always sends filled in beneath the fake's own (at
    every depth). Tests script the fields they care about; the real route never answers a bare or partial body, and
    the fake check (`contract_fakes.py`) refuses one. A non-2xx answer, a non-JSON one, a route the contract does not
    list, a host other than miServer's placeholder and a body already in the envelope pass unchanged; the fake check
    then judges what passed (a wrong type or an undeclared key is still the test's to fix)."""
    import httpx

    from tests.support.answer_samples import complete, sample
    from tests.support.contract_fakes import _key_for

    if not 200 <= response.status_code < 300:
        return response
    if product is None:
        product = "miserver" if request.url.host == MISERVER_HOST else "miworkbench"
    if product != "miserver":
        return _complete_native(request, response) if request.url.path.startswith("/api/v1/") else response
    try:
        body = response.json()
    except ValueError:
        return response
    found = _key_for("miserver", request.method, request.url.path)
    if found is None or found[0] != "responses":
        return response
    contract = _contract()
    status = str(response.status_code)
    schema = contract["responses"][found[1]].get(status)
    if schema is None:
        return response
    ref = schema.get("$ref", "")
    target = contract["$defs"].get(ref.split("/")[-1], {}) if ref.startswith("#/$defs/") else {}
    members = [contract["$defs"][a["$ref"].split("/")[-1]] for a in target.get("anyOf", []) if "$ref" in a]
    envelopes = [t for t in [target, *members] if {"success", "data"} <= set(t.get("properties", {}))]
    flagged = str(response.headers.get("x-millm-approval", "")).strip().lower() == "pending"
    if envelopes:
        if isinstance(body, dict) and "success" in body:
            data = body.get("data")
            outer = {k: v for k, v in body.items() if k != "data"}
        else:
            data, outer = body, {}
        if flagged and isinstance(data, dict):
            data = pending_data(**data)
        elif isinstance(data, (dict, list)):
            chosen = next((e for e in envelopes if "ApprovalPending" not in json_text(e)), envelopes[0])
            data = complete(chosen["properties"]["data"], data, contract)
        full = {**sample({k: v for k, v in envelopes[0].items() if k != "examples"}, contract, prefer_null=True),
                **outer, "data": data}
        full["success"] = True
        full.setdefault("error", None)
        if outer.get("error", None) is None:
            full["error"] = None
        completed = full
    elif isinstance(body, (dict, list)) and not str(ref).startswith("http"):
        completed = complete(schema, body, contract)
    else:
        return response
    headers = {k: v for k, v in response.headers.items() if k.lower() not in ("content-length", "content-encoding")}
    return httpx.Response(response.status_code, json=completed, headers=headers, request=request)


def json_text(node: Any) -> str:
    import json

    return json.dumps(node)


def as_sent(handler: Any, product: str = "miserver") -> Any:
    """A fake handler (whatever host it is mounted on) whose answers are completed as `product` sends them
    (`complete_response`): miServer's by default, or `"miworkbench"` for miWorkbench's own routes."""

    def wrapped(request: Any) -> Any:
        return complete_response(request, handler(request), product=product)

    return wrapped


def _native_source() -> Any:
    """miWorkbench's committed answer contract (046-2, FTASKS 7.3), as `(entries, defs)`: `entries[(METHOD, normalised
    path)] = {status: schema}`, the same contract the fake check judges a fake answer against."""
    import functools
    import json

    @functools.lru_cache(maxsize=1)
    def load() -> Any:
        from types import SimpleNamespace

        from src.api.answers_contract import DEFAULT_PATH
        from src.mcp_server.route_map import normalise

        contract = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
        entries = {}
        for key, entry in contract["responses"].items():
            method, path = key.split(" ", 1)
            entries[(method, normalise(path))] = {s: v for s, v in entry.items() if s[:1].isdigit()}
        return SimpleNamespace(entries=entries, defs=contract["$defs"])

    return load()


def _complete_native(request: Any, response: Any) -> Any:
    """miWorkbench's own route: completed from its committed answer contract (046-2), so a TYPED native tool's fake
    carries the fields its schema requires. A route whose answer is untyped there passes unchanged."""
    import httpx

    from tests.support.answer_samples import complete

    try:
        body = response.json()
    except ValueError:
        return response
    source = _native_source()
    entry = _match(source.entries, request.method, request.url.path)
    schema = (entry or {}).get(str(response.status_code))
    if not schema or not isinstance(body, (dict, list)):
        return response
    completed = complete(schema, body, {"$defs": source.defs})
    headers = {k: v for k, v in response.headers.items() if k.lower() not in ("content-length", "content-encoding")}
    return httpx.Response(response.status_code, json=completed, headers=headers, request=request)


def _match(entries: dict, method: str, path: str) -> Any:
    """The entry of the template a concrete path matches (`{}` is one segment; the fewest parameters win)."""
    from src.mcp_server.route_map import normalise

    segments = normalise(path).split("/")
    best = None
    for (m, template), entry in entries.items():
        parts = template.split("/")
        if m == method and len(parts) == len(segments) and all(t == "{}" or t == p for t, p in zip(parts, segments, strict=True)):
            if best is None or parts.count("{}") < best[0]:
                best = (parts.count("{}"), entry)
    return best[1] if best else None


def envelope_for(key: str, data: Any, http_status: str = "200") -> dict[str, Any]:
    """`data` in the envelope `key`'s route sends, every other envelope field filled (a page's `meta`, …)."""
    from tests.support.answer_samples import sample

    contract = _contract()
    ref = contract["responses"][key][http_status]["$ref"]
    target = contract["$defs"][ref.split("/")[-1]]
    if "anyOf" in target:
        target = next(contract["$defs"][a["$ref"].split("/")[-1]] for a in target["anyOf"]
                      if "ApprovalPending" not in a.get("$ref", ""))
    outer = sample({k: v for k, v in target.items() if k != "examples"}, contract, prefer_null=True)
    return {**outer, "success": True, "data": data, "error": None}


def complete_def(name: str, body: Any) -> Any:
    """`body` completed by the contract definition `name` (its always-sent fields filled beneath body's own)."""

    from tests.support.answer_samples import complete

    return complete({"$ref": "#/$defs/" + name}, body, _contract())
