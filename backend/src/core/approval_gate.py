"""The REST approval gate: GPU starts, outside paid calls and Hub publishes wait for the operator when an agent other
than the MCP pod calls them (045 FR-13 – FR-16; CHAIN_PLAN A12, A15, A29, A37 – A41; FTDD §16.3).

One dependency, `require_consumed_approval(action)`, attached per route (`dependencies=[...]` on the decorator; no
handler body changes, A40):

* a caller WITHOUT `X-MiStudio-Agent` (the UI, the operator's scripts) is unchanged (FR-15);
* the MCP pod (`mcp`, A15) is exempt: its gates stay in the MCP layer — unless the route lists it in
  `refuse_without_id` (044's routes: 403 `approval_required`);
* any other agent value (`agent:miassist`, `miassist`, …) with no approval id: its query and body are parsed by the
  route's own rules, undeclared keys refused (422 `unknown_body_field` / `unknown_query_field`, A38), then ONE pending
  approval is filed in its own committed session and the call answers 202 the approval-state header `pending` — nothing
  is queued, the handler never runs;
* a call WITH an id (the approval-request header, or `approval_request_id` in the raw body), whoever sends it:
  the approval is consumed in its own committed session BEFORE the handler runs (A40), only for this action and the
  exact canonical payload (which binds `_method` and `_route`, A38); if the handler then raises, the approval is
  marked `failed` in a fresh session and never re-opened (A29).

044 (WSP) designs the first version of this dependency; 045 builds it off `main` because 044 had not landed when 045-A0
was built (NEEDS N-1): the two converge at the rebase.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError

from .approval_payload import (APPROVAL_ID_KEY, APPROVAL_REQUEST_HEADER, RESERVED_KEYS, CollidingPayloadKey,
                               canonical_payload)

#: The header an agent names itself in (the MCP pod's client sends `mcp`; `mcp_server/client.py`).
AGENT_HEADER = "X-MiStudio-Agent"
#: The header the 202 carries.
APPROVAL_STATE_HEADER = "X-MiStudio-Approval"  # p9: keep, a 045 wire header fixed by CHAIN_PLAN A37 and pinned by miAssist


def _mcp_pod_agents() -> frozenset[str]:
    """`{AGENT_NAME}`, read from the MCP pod's own client constant so the exemption cannot drift from what the pod
    sends (A15; FTASKS 22.2). The client module imports only `httpx` and the MCP SDK."""
    from src.mcp_server.client import AGENT_NAME

    return frozenset({AGENT_NAME})


MCP_POD_AGENTS = _mcp_pod_agents()


class PendingApproval(Exception):
    """Raised by the dependency once a request is filed; its handler (registered in `main.py`) answers 202."""

    def __init__(self, approval_id: str, action: str, payload: dict[str, Any]):
        super().__init__(approval_id)
        self.approval_id = approval_id
        self.action = action
        self.payload = payload


def pending_approval_body(exc: PendingApproval) -> dict[str, Any]:
    """The 202's body: today's MCP spelling plus the A6 `approval` object, valid against miAssist's
    `mistudio.result-language/v1` `approval` $def (checked by the phase B vendored-contract test). A REST caller re-calls
    the same request with the id in a header, which `call_again` ({tool, args}) cannot express, so `call_again_with`
    is omitted (FPRD §7 (a): optional; the summary names the re-call) and the header is given beside it."""
    route = f"{exc.payload.get('_method')} {exc.payload.get('_route')}"
    return {
        "approval_request_id": exc.approval_id,
        "status": "pending_approval",
        "action": exc.action,
        "retry_with_header": {APPROVAL_REQUEST_HEADER: exc.approval_id},
        "approval": {
            "id": exc.approval_id,
            "status": "pending",
            "decide_in": "miworkbench",
            "then": "call_again_with",
            "action": exc.action,
            "summary": (f"{exc.action}: {route}. Once the operator approves it in miWorkbench, send the same request "
                        f"again with the header {APPROVAL_REQUEST_HEADER}: {exc.approval_id}.")[:500],
        },
    }


async def pending_approval_handler(request: Request, exc: PendingApproval):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=202, content=pending_approval_body(exc),
                        headers={APPROVAL_STATE_HEADER: "pending"})


@dataclass
class ParsedParams:
    """The route's own parse of a call: query values by parameter name, and the body as the handler will see it."""

    query: dict[str, Any] = field(default_factory=dict)
    body: Any = None


async def read_json_body(request: Request) -> Any:
    """The raw JSON body (Starlette caches it on the request, so the handler reads the same bytes), `None` if empty.
    A body that is not JSON is the route's own 422 (it would refuse it)."""
    raw = await request.body()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise RequestValidationError([{"type": "json_invalid", "loc": ("body",), "msg": "body is not JSON",
                                       "input": None}]) from exc


@dataclass
class _Flat:
    query_params: list[Any]
    body_params: list[Any]


@dataclass
class RouteContext:
    """What the gate needs of the matched route: its FULL path template (prefixes included), its dependant and
    whether its body fields are embedded. FastAPI 0.159 keeps included routers unflattened (`_IncludedRouter`), so
    `scope["route"].path` is the router-local path; the effective route context carries the full one."""

    path: str
    dependant: Any
    embed_body_fields: bool


def route_context(request: Request) -> RouteContext:
    effective = (request.scope.get("fastapi") or {}).get("effective_route_context")
    if effective is not None:
        return RouteContext(effective.path, effective.dependant, bool(getattr(effective, "_embed_body_fields", False)))
    route = request.scope["route"]
    return RouteContext(route.path, route.dependant, bool(getattr(route, "_embed_body_fields", False)))


def _flat(route: Any) -> _Flat:
    """The route's query and body parameters across its whole dependency tree, each name once (FastAPI 0.159 keeps
    its own flattening private)."""
    query: dict[str, Any] = {}
    body: dict[str, Any] = {}
    stack = [route.dependant]
    while stack:
        current = stack.pop()
        for param in current.query_params:
            query.setdefault(param.name, param)
        for param in current.body_params:
            body.setdefault(param.name, param)
        stack.extend(reversed(current.dependencies))
    return _Flat(list(query.values()), list(body.values()))


def body_param_name(route: Any) -> Optional[str]:
    params = _flat(route).body_params
    return params[0].name if len(params) == 1 else None


def _embed(route: Any) -> bool:
    if isinstance(route, RouteContext):
        return route.embed_body_fields
    if hasattr(route, "_embed_body_fields"):
        return bool(route._embed_body_fields)  # noqa: SLF001
    return len(_flat(route).body_params) > 1


async def parse_with_route(route: Any, request: Request, raw: Any) -> ParsedParams:
    """FastAPI's own parameter solving over the route's dependant (A41): the arms read the values the handler will
    see, and a call the route would refuse is the route's own 422, raised before anything is filed or consumed."""
    from fastapi.dependencies.utils import request_body_to_args, request_params_to_args

    flat = _flat(route)
    query, errors = request_params_to_args(flat.query_params, request.query_params)
    body: Any = None
    if flat.body_params:
        body_without_id = raw
        if isinstance(raw, dict) and APPROVAL_ID_KEY in raw:
            body_without_id = {k: v for k, v in raw.items() if k != APPROVAL_ID_KEY}
        values, body_errors = await request_body_to_args(flat.body_params, body_without_id, _embed(route))
        errors = [*errors, *body_errors]
        body = values[flat.body_params[0].name] if len(flat.body_params) == 1 and not _embed(route) else values
    if errors:
        raise RequestValidationError(errors)
    return ParsedParams(query=query, body=body)


def _declared_keys(annotation: Any) -> Optional[dict[str, Any]]:
    """`{wire name: nested annotation}` for a pydantic model, else `None` (a body that is not a model)."""
    from pydantic import BaseModel

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        out = {}
        for name, info in annotation.model_fields.items():
            out[info.alias or name] = info.annotation
            out[name] = info.annotation
        return out
    return None


def _models_in(annotation: Any) -> list[Any]:
    """The pydantic models an annotation can hold (Optional, list, Union …), for the nested walk."""
    import typing

    from pydantic import BaseModel

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    out = []
    for arg in typing.get_args(annotation):
        out += _models_in(arg)
    return out


def _undeclared(value: Any, annotation: Any, path: str) -> Optional[str]:
    """The dotted path of the first key no model declares, at any depth, or `None`."""
    if isinstance(value, list):
        for i, item in enumerate(value):
            hit = _undeclared(item, annotation, f"{path}[{i}]")
            if hit:
                return hit
        return None
    if not isinstance(value, dict):
        return None
    models = _models_in(annotation)
    if not models:
        return None  # a free-form dict field (`config: dict`) declares no keys
    declared: dict[str, Any] = {}
    for model in models:
        declared.update(_declared_keys(model) or {})
    for key, sub in value.items():
        if key not in declared:
            return f"{path}.{key}" if path else key
        hit = _undeclared(sub, declared[key], f"{path}.{key}" if path else key)
        if hit:
            return hit
    return None


def refuse_undeclared(route: Any, request: Request, raw: Any) -> None:
    """For an agent caller (A38): a query parameter or body key the route does not declare is refused before filing
    or consuming, so a card never shows a key the route would silently ignore."""
    flat = _flat(route)
    declared_query = {p.alias for p in flat.query_params} | {p.name for p in flat.query_params}
    for key in request.query_params.keys():
        if key not in declared_query:
            raise HTTPException(422, {"code": "unknown_query_field", "field": key})
    if isinstance(raw, dict):
        raw = {k: v for k, v in raw.items() if k != APPROVAL_ID_KEY}
    if not flat.body_params and raw not in (None, {}):
        # R1 review fix (OR1-045a0-1), 2026-11-16: a route that declares no body ignores one, so a body key would
        # be a key the card shows and the route never reads.
        field = sorted(raw)[0] if isinstance(raw, dict) else "body"
        raise HTTPException(422, {"code": "unknown_body_field", "field": field})
    if not flat.body_params or raw is None:
        return
    if _embed(route):
        names = {p.alias: p for p in flat.body_params} | {p.name: p for p in flat.body_params}
        if not isinstance(raw, dict):
            return
        for key, sub in raw.items():
            if key not in names:
                raise HTTPException(422, {"code": "unknown_body_field", "field": key})
            hit = _undeclared(sub, names[key].field_info.annotation, key)
            if hit:
                raise HTTPException(422, {"code": "unknown_body_field", "field": hit})
        return
    hit = _undeclared(raw, flat.body_params[0].field_info.annotation, "")
    if hit:
        raise HTTPException(422, {"code": "unknown_body_field", "field": hit})


#: Body keys that carry a Hugging Face token. An agent other than the MCP pod may not send one to a gated route
#: (FR-18; V-9): it is refused before anything is filed or consumed.
AGENT_TOKEN_KEYS = ("token", "access_token", "hf_token")


def refuse_agent_token(raw: Any) -> None:
    """V-9 (FR-13, FR-18): an agent's own Hub token is `422 agent_token_not_accepted`, BEFORE filing or consuming, so
    no approval is created or spent on a call that would push or read with a token the operator never saw."""
    if not isinstance(raw, dict):
        return
    for key in AGENT_TOKEN_KEYS:
        if raw.get(key) not in (None, ""):
            raise HTTPException(422, {"code": "agent_token_not_accepted", "field": key})


Arm = Callable[[ParsedParams], bool]


def RECUT_WINDOWS_ARM(parsed: ParsedParams) -> bool:  # noqa: N802 — a named constant
    """Row 25 (A41): a recalibration is a GPU start only with `recut_windows` true — the PARSED bool, so every
    spelling FastAPI reads as true is gated; armed unless it parsed to False."""
    return parsed.query.get("recut_windows") is not False


_NOT_RESUME = frozenset({"pause", "stop", "stop_and_finalize"})


def RESUME_ARM(parsed: ParsedParams) -> bool:  # noqa: N802
    """Row 35 (A41): training control starts GPU work only on `resume`; armed unless the parsed action is
    definitely another one."""
    action = getattr(parsed.body, "action", None)
    return str(getattr(action, "value", action)) not in _NOT_RESUME


def require_consumed_approval(action: str, *, arm: Optional[Arm] = None,
                              exempt_agents: frozenset[str] = MCP_POD_AGENTS,
                              refuse_without_id: frozenset[str] = frozenset()) -> Any:
    """The dependency for one gated route (FTDD §16.3). `arm`, when given, says on the PARSED call whether this call
    is gated at all (fail closed: an arm that raises gates)."""

    async def dependency(request: Request):
        from src.api.v1.endpoints import mcp_approvals

        agent = request.headers.get(AGENT_HEADER)
        header_id = request.headers.get(APPROVAL_REQUEST_HEADER)
        if header_id is None and (agent is None or (agent in exempt_agents and agent not in refuse_without_id)):
            # FR-15, A15: a caller without the agent header, and the MCP pod, are exactly as before: nothing is read,
            # parsed or filed here (a form upload or any body the route reads its own way is untouched).
            yield
            return
        route = route_context(request)
        raw = await read_json_body(request)
        request_id = header_id or (raw.get(APPROVAL_ID_KEY) if isinstance(raw, dict) else None)
        parsed = await parse_with_route(route, request, raw)
        gated_agent = agent is not None and agent not in exempt_agents
        if gated_agent:
            refuse_agent_token(raw)
            refuse_undeclared(route, request, raw)
        if arm is not None:
            try:
                armed = bool(arm(parsed))
            except Exception:  # noqa: BLE001 — an arm that cannot decide gates (fail closed)
                armed = True
            if not armed:
                if request_id is not None:
                    raise HTTPException(422, {"code": "approval_not_applicable", "action": action})
                yield
                return
        try:
            payload = canonical_payload(request.method, route.path, request.path_params,
                                        _query_for_payload(request, parsed), raw, body_param=body_param_name(route))
        except CollidingPayloadKey as exc:  # OR1-045a0-10: refused before anything is filed or consumed
            raise HTTPException(422, {"code": "colliding_field", "field": exc.key}) from exc
        if request_id is None:
            if agent is None or (agent in exempt_agents and agent not in refuse_without_id):
                yield
                return
            if agent in refuse_without_id:
                raise HTTPException(403, {"code": "approval_required", "action": action})
            row = await mcp_approvals.file_request(action, payload, requested_by=mcp_approvals.requested_by_of(request))
            raise PendingApproval(row.id, action, payload)
        await _consume_and_commit(request_id, action, payload)
        try:
            yield
        except BaseException as exc:
            await mcp_approvals.mark_failed(request_id, reason=f"{type(exc).__name__}: {exc}"[:2000])
            raise

    dependency.approval_action = action  # type: ignore[attr-defined]
    dependency.is_require_consumed_approval = True  # type: ignore[attr-defined]
    return Depends(dependency)


def _query_for_payload(request: Request, parsed: ParsedParams) -> dict[str, Any]:
    """Only the query values the caller SENT, as the route parsed them (a default the caller did not send is not part
    of what is approved), JSON-safe."""
    import pydantic_core

    sent = set(request.query_params.keys())
    return {k: pydantic_core.to_jsonable_python(v) for k, v in parsed.query.items() if k in sent}


async def _consume_and_commit(request_id: str, action: str, payload: dict[str, Any]) -> None:
    """The consume, in its OWN session, committed before the handler runs (A40). A payload that differs (another
    route, another body) is `422 mismatch`."""
    from src.api.v1.endpoints import mcp_approvals
    from src.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as own:
        try:
            await mcp_approvals.consume(own, request_id, action, payload)
        except HTTPException as exc:
            if exc.status_code == 422:
                raise HTTPException(422, {"code": "mismatch", "action": action, "detail": exc.detail}) from exc
            raise


def iter_routes(app: Any):
    """Every `(full path template, APIRoute)` of an app, through included routers (FastAPI 0.159 keeps them as
    `_IncludedRouter`; memory `app-routes-is-not-a-route-list`)."""
    from fastapi.routing import APIRoute

    def walk(routes, prefix):
        for route in routes:
            if isinstance(route, APIRoute):
                yield prefix + route.path, route
            elif hasattr(route, "original_router"):
                context = getattr(route, "include_context", None)
                yield from walk(route.original_router.routes, prefix + (getattr(context, "prefix", "") or ""))

    yield from walk(getattr(app, "routes", []), "")


def gated_routes(app: Any) -> dict[tuple[str, str], str]:
    """`{(METHOD, full path template): action}` for every live route carrying this dependency (FR-14; the vectors' and
    the queue filing's lookup)."""
    out: dict[tuple[str, str], str] = {}
    for path, route in iter_routes(app):
        for dep in _flat_dependencies(route.dependant):
            if getattr(dep.call, "is_require_consumed_approval", False):
                for method in sorted(route.methods or ()):
                    out[(method, path)] = dep.call.approval_action
    return out


def _flat_dependencies(dependant: Any):
    stack = list(dependant.dependencies)
    while stack:
        dep = stack.pop()
        yield dep
        stack.extend(dep.dependencies)


async def validate_queue_filing(request: Request, action: str, payload: Any) -> None:
    """A43 (V-1; FTDD §16.3 "The queue filing is validated too"): an agent other than the MCP pod that files a
    caller-executed action through `POST /mcp/approvals` names the route in the payload's `_method` and `_route`; the
    filing is checked as that route's own dependency checks a call, BEFORE any row is written:
    an unknown route, or one whose action is not `action` → `422 mismatch`; an agent token → `422
    agent_token_not_accepted`; the route's own parse → its FastAPI 422; an undeclared key → `422 unknown_body_field` /
    `unknown_query_field`. The MCP pod's filings and header-less (operator) filings are not touched."""
    from urllib.parse import urlencode

    from starlette.requests import Request as StarletteRequest

    agent = request.headers.get(AGENT_HEADER)
    if agent is None or agent in MCP_POD_AGENTS:
        return
    if not isinstance(payload, dict):
        raise HTTPException(422, {"code": "mismatch", "action": action, "detail": "the payload is not an object"})
    method, template = payload.get("_method"), payload.get("_route")
    routes = {(m, path): route for path, route in iter_routes(request.app) for m in (route.methods or ())}
    gated = gated_routes(request.app)
    if (method, template) not in gated or gated[(method, template)] != action:
        raise HTTPException(422, {"code": "mismatch", "action": action,
                                  "detail": f"{method} {template} is not a route gated as {action}"})
    route = routes[(method, template)]
    flat = _flat(route)
    path_names = {p.name for p in route.dependant.path_params}
    query_names = {p.alias for p in flat.query_params} | {p.name for p in flat.query_params}
    rest = {k: v for k, v in payload.items() if k not in RESERVED_KEYS}
    path_params = {k: rest.pop(k) for k in list(rest) if k in path_names}
    query = {k: rest.pop(k) for k in list(rest) if k in query_names}
    name = body_param_name(route)
    # R1 review fix (OR1-045a0-1), 2026-11-16: a payload with no body keys is the canonical form of BOTH an empty
    # object body `{}` and no body at all, so both are tried, `{}` first: a route whose body has a default
    # (`request: X = X()`) or is required refuses `None` and accepts `{}`; an optional body `X | None` may accept only
    # `None`. `rest or None` refused 10 of the 56 published vectors. A route with no body gets any leftover keys as
    # a body, which `refuse_undeclared` refuses.
    candidates: list[Any] = [rest] if rest else ([{}, None] if flat.body_params else [None])
    body: Any = candidates[0]  # `rest` itself when it has keys
    if name and not _embed(route) and _declared_keys(flat.body_params[0].field_info.annotation) is None:
        body = rest.pop(name, None)  # a non-object body (a list) is keyed by its parameter
        if rest:
            raise HTTPException(422, {"code": "unknown_body_field", "field": sorted(rest)[0]})
        candidates = [body]
    concrete = template
    for key, value in path_params.items():
        concrete = concrete.replace("{" + key + "}", str(value))
    fake = StarletteRequest({"type": "http", "method": method, "path": concrete, "headers": [],
                             "query_string": urlencode(query, doseq=True).encode()})
    refuse_agent_token(body)
    for i, body in enumerate(candidates):
        try:
            await parse_with_route(route, fake, body)
            break
        except RequestValidationError:
            if i == len(candidates) - 1:
                raise
    refuse_undeclared(route, fake, body)
