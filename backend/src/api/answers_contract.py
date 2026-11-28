"""miWorkbench's published answer contract, DERIVED from the live route table (046, BRD-MIS-SHAPES-001; FTDD §6.2).

`docs/schemas/miworkbench-answers-v1.json` describes every answer a COVERED route serves: the routes a registered MCP
tool calls (the one route walker, `src/mcp_server/route_map.py`) plus the routes a sibling calls (`SIBLING_ROUTES`).
For each covered route and 2xx status it holds the JSON body schema (`responses`, with `x-approval-action` on gated
routes); the covered routes that answer no JSON (`excluded`, with their media type); every other operation
(`not_covered`, with the reason); the two error envelopes (`errors`); the document schemas the answers reference
(`external`); and coverage counts. It is layout for layout miServer's (`miserver-answers-v1.json`, MW9), so miAssist
reads one layout for every app.

It is generated from `src.main.app`'s OpenAPI document, never written by hand, and pinned by
`tests/unit/test_answers_contract.py`. Regenerate with `MIWORKBENCH_WRITE_CONTRACTS=1` on that test; the writer
refuses a breaking change (FR-046.13), which needs `miworkbench-answers-v2.json` beside v1.

Transforms, in order (FTDD §6.2): references rewritten to `#/$defs/`; the siblings FastAPI merges beside a documented
model's `$ref` removed (the 2026-11-26 serializer decision: a documented-only route keeps its served model, and
FastAPI deep-merges that model's `type`, `title` and `additionalProperties` into the documented entry); every
always-sent field `required` (SH7, TD4); the history-form fields (WT9); no `additionalProperties: false` (SH7);
external references resolved to `docs/schemas/` files (SH8); only reachable definitions kept (FR-046.15); coverage
counted; the digest computed (TD11).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from src.api import contract_tools as tools
from src.api.contract_tools import render

CONTRACT_FILENAME = "miworkbench-answers-v1.json"
CONTRACT_ID = "https://hitsai.net/schemas/" + CONTRACT_FILENAME
KIND = "miworkbench.answer-contract/v1"
SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "docs" / "schemas"
DEFAULT_PATH = SCHEMAS_DIR / CONTRACT_FILENAME
#: Routes a sibling app calls, beside the MCP tools (FR-046.3). Checked against miAssist's recorded routes by
#: `tests/unit/test_answers_contract_coverage.py`.
SIBLING_ROUTES = frozenset({("GET", "/api/health")})
API_PREFIX = "/api/v1"
NOT_COVERED_REASON = "no tool or sibling calls it"
NO_CONTENT_REASON = "no content"
_OPENAPI_REF = "#/components/schemas/"
_JSON = "application/json"
_DEFAULT_DESCRIPTION = "Successful Response"
_EXCLUDE_FLAGS = ("response_model_exclude_none", "response_model_exclude_unset", "response_model_exclude_defaults")
#: What FastAPI deep-merges beside a `responses={...: {"model": X}}` `$ref` from the route's own served model.
#: What FastAPI merges from a served `Dict` or `List[Dict]` model next to a documentation model's `$ref`.
_MERGED_SIBLINGS = ("additionalProperties", "anyOf", "items", "title", "type")
HISTORY_EXCEPTION = (
    "The one exception to 'required means always sent' (SH7): a field holding JSON an earlier release stored is "
    "checked in its HISTORY FORM (WT9): it keeps its types but has no `required` at any depth, because a stored "
    "value can predate a field."
)
_DESCRIPTION = (
    "GENERATED from miWorkbench's registered routes (their response declarations, via FastAPI's OpenAPI document); "
    "do not edit by hand. Regenerate with MIWORKBENCH_WRITE_CONTRACTS=1 on tests/unit/test_answers_contract.py; the "
    "writer refuses a breaking change, which needs miworkbench-answers-v2.json beside this file. `responses` holds "
    "the routes an MCP tool or a sibling app calls (the covered routes), keyed '<METHOD> <path>', each mapping a 2xx "
    "status to that answer's JSON body schema; `excluded`, the covered routes with no JSON body; `not_covered`, "
    "every other operation. `required` lists every field a route always sends; extra keys are allowed (SH7). "
    "Datetimes are ISO 8601 strings that may carry no offset (a naive value is UTC), so no field declares "
    "`format: date-time` or `format: time`. "
)


class CoverageError(ValueError):
    """The route table and the tools disagree, or a covered route cannot honestly be described (FTID §12)."""


class BreakingAnswerChange(Exception):
    """The writer met a change FR-046.13 classifies as breaking (or one no row names)."""

    def __init__(self, changes: list[tools.Change]) -> None:
        self.changes = changes
        lines = "\n".join(f"  {c}" for c in changes)
        super().__init__(f"{len(changes)} breaking change(s) to {CONTRACT_FILENAME}:\n{lines}\n"
                         "a breaking change needs miworkbench-answers-v2.json beside v1")


def _path_key(path: str) -> str:
    """The OpenAPI form of a route path: converter suffixes dropped (`{repo_id:path}` -> `{repo_id}`)."""
    return re.sub(r"\{(\w+):\w+\}", r"{\1}", path)


def _normalise(path: str) -> str:
    from src.mcp_server.route_map import normalise

    return normalise(path)


def operations(app: Any) -> list[tuple[str, str, Any]]:
    """`(METHOD, OpenAPI path, APIRoute)` for every operation of `app` in its schema, sorted. FastAPI 0.159 wraps
    included routers, so `app.routes` alone lists none of them (memory `app-routes-is-not-a-route-list`)."""
    from fastapi.routing import APIRoute, iter_route_contexts

    out = []
    for ctx in iter_route_contexts(app.routes):
        route = ctx.original_route
        if not isinstance(route, APIRoute) or not ctx.include_in_schema:
            continue
        for method in sorted(ctx.methods or ()):
            out.append((method, _path_key(str(ctx.path)), route))
    return sorted(out, key=lambda o: (o[1], o[0]))


def tool_route_keys(tool_names: Iterable[str]) -> dict[str, set[tuple[str, str]]]:
    """`{tool: {(METHOD, normalised full path)}}`: the miWorkbench routes each named tool calls (the one walker)."""
    from src.mcp_server.route_map import tool_routes

    walked = tool_routes()
    out: dict[str, set[tuple[str, str]]] = {}
    for name in sorted(set(tool_names)):
        out[name] = {(m, _normalise(API_PREFIX + p)) for product, m, p in walked.get(name, ()) if product == "miworkbench"}
    return out


def covered_keys(tool_names: Iterable[str], app: Any = None) -> frozenset[str]:
    """FR-046.3: the covered set, `<METHOD> <full path template>`, computed from the walker and `SIBLING_ROUTES`.

    A tool route that matches no operation, or matches two, is a `CoverageError` naming it."""
    if app is None:
        from src.main import app
    by_normal: dict[tuple[str, str], list[str]] = {}
    for method, path, _route in operations(app):
        by_normal.setdefault((method, _normalise(path)), []).append(path)
    covered: set[str] = set()
    unknown: list[str] = []
    for tool, routes in tool_route_keys(tool_names).items():
        for method, normal in sorted(routes):
            paths = by_normal.get((method, normal), [])
            if len(paths) != 1:
                unknown.append(f"{tool}: {method} {normal} matches {paths or 'no operation'}")
                continue
            covered.add(f"{method} {paths[0]}")
    for method, path in SIBLING_ROUTES:
        if len(by_normal.get((method, _normalise(path)), [])) != 1:
            unknown.append(f"SIBLING_ROUTES: {method} {path} matches no operation")
        covered.add(f"{method} {path}")
    if unknown:
        raise CoverageError("tool routes that are not one live operation:\n  " + "\n  ".join(unknown))
    return frozenset(covered)


def _rewrite(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: ("#/$defs/" + v[len(_OPENAPI_REF):] if k == "$ref" and isinstance(v, str)
                    and v.startswith(_OPENAPI_REF) else _rewrite(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [_rewrite(v) for v in node]
    return node


def _entry_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """A route's 2xx body schema as the contract states it: a documented model's `$ref` without the served model's
    merged siblings (the serializer decision of 2026-11-26)."""
    if isinstance(schema.get("$ref"), str):
        return {k: v for k, v in schema.items() if k not in _MERGED_SIBLINGS}
    return schema


def _media_reason(response: dict[str, Any], media: str, content: dict[str, Any]) -> str:
    reason = content.get("x-reason")
    if isinstance(reason, str) and reason:
        return reason
    text = response.get("description")
    if isinstance(text, str) and text and text != _DEFAULT_DESCRIPTION:
        return text
    return f"the route answers {media}, not JSON"


def _external_files() -> dict[str, tuple[str, str]]:
    """Every document schema in `docs/schemas/` by its `$id`: (file, sha256)."""
    out: dict[str, tuple[str, str]] = {}
    for path in sorted(SCHEMAS_DIR.glob("*.json")):
        if path.name == CONTRACT_FILENAME:
            continue
        raw = path.read_bytes()
        try:
            doc = json.loads(raw)
        except ValueError:
            continue
        sid = doc.get("$id") if isinstance(doc, dict) else None
        if isinstance(sid, str):
            out[sid] = (f"docs/schemas/{path.name}", hashlib.sha256(raw).hexdigest())
    return out


#: FR-046.10: FastAPI's refusal envelope, declared once. `detail` is a string, an object (the 61 `detail={...}` sites)
#: or a list.
HTTP_ERROR = {
    "type": "object",
    "description": "Every refusal: FastAPI's `HTTPException` envelope. `detail` is a string, an object or a list.",
    "properties": {"detail": {"anyOf": [{"type": "string"}, {"type": "object"}, {"type": "array"}]}},
    "required": ["detail"],
}


def _members(schema: Any, defs: dict[str, Any]) -> list[Any]:
    resolved = tools._deref(schema, defs)
    for key in ("anyOf", "oneOf"):
        if isinstance(resolved, dict) and key in resolved:
            return [m for member in resolved[key] for m in _members(member, defs)]
    return [] if isinstance(resolved, dict) and resolved.get("type") == "null" else [schema]


def contract_from(app: Any, spec: dict[str, Any], covered: frozenset[str]) -> dict[str, Any]:
    """The contract body (no `revision`, no `content_sha256`) of `app`'s route table, read from the OpenAPI document
    `spec` (normally `app.openapi()`; the live check passes the pod's), for the `covered` keys."""
    from src.core.approval_gate import gated_routes

    spec = copy.deepcopy(spec)  # FastAPI caches the served document on the app: never transform it in place
    components: dict[str, Any] = spec.get("components", {}).get("schemas", {})
    gated = {f"{m} {_path_key(p)}": a for (m, p), a in gated_routes(app).items()}
    responses: dict[str, Any] = {}
    excluded: dict[str, Any] = {}
    not_covered: dict[str, Any] = {}
    seen: set[str] = set()
    for method, path, route in operations(app):
        key = f"{method} {path}"
        seen.add(key)
        if key not in covered:
            not_covered[key] = {"reason": NOT_COVERED_REASON}
            continue
        flags = [f for f in _EXCLUDE_FLAGS if getattr(route, f, False)]
        if flags:
            raise CoverageError(f"{key} sets {flags}: 'always sent' would be false (FR-046.12)")
        operation = spec["paths"][path][method.lower()]
        entry: dict[str, Any] = {}
        other: dict[str, Optional[str]] = {}
        for status, response in sorted(operation.get("responses", {}).items()):
            if not status.startswith("2"):
                continue
            content = response.get("content") or {}
            if not content:
                other[status] = None
            for media, body in sorted(content.items()):
                if media == _JSON:
                    entry[status] = _entry_schema(_rewrite(body.get("schema", {})))
                else:
                    other[status] = media
                    entry.setdefault("x-variants", {})[media] = _media_reason(response, media, body)
        if entry and any(k[:1].isdigit() for k in entry):
            no_body = sorted(st for st, media in other.items() if media is None)
            if no_body:
                entry["x-no-body"] = no_body  # a status the route answers with no content at all (FR-046.9)
            if key in gated:
                entry["x-approval-action"] = gated[key]
            responses[key] = dict(sorted(entry.items()))
        elif other:
            status = sorted(other)[0]
            media = other[status]
            excluded[key] = ({"media_type": None, "reason": NO_CONTENT_REASON} if media is None
                             else {"media_type": media, "reason": entry.get("x-variants", {}).get(media, "")})
            # Each declared 2xx with ITS media type (null: no body at all), so a route answering a 204 or a 202 text
            # note is checked per status (046-2 review round 1: one row's media described only the first status).
            excluded[key]["statuses"] = dict(sorted(other.items()))
        else:
            raise CoverageError(f"{key} declares no 2xx answer at all")
    missing = sorted(covered - seen)
    if missing:
        raise CoverageError(f"covered keys that are no live operation: {missing}")

    defs = {name: _rewrite(body) for name, body in components.items()}
    errors = {"http": copy.deepcopy(HTTP_ERROR), "validation": {"$ref": "#/$defs/HTTPValidationError"}}
    if "HTTPValidationError" not in defs:
        raise CoverageError("the app declares no HTTPValidationError: errors.validation has no schema")

    # FastAPI's own 422 item sends `ctx` only for a constrained field (a `missing` error has none), so it is marked
    # may-be-absent before the always-sent rule; `input` is always sent (046 P7 finding, 2026-11-27).
    if isinstance((defs.get("ValidationError") or {}).get("properties", {}).get("ctx"), dict):
        defs["ValidationError"]["properties"]["ctx"][tools.MAY_BE_ABSENT] = True
    tools.always_sent_required(defs)
    tools.always_sent_required(responses)
    tools.always_sent_required(errors)
    history = tools.apply_history_forms(defs)
    tools.open_objects(defs)
    tools.open_objects(responses)
    tools.drop_offset_formats(defs)  # the routes send naive datetimes (P7 finding, 2026-11-26)
    tools.drop_offset_formats(responses)

    known = _external_files()
    external: dict[str, Any] = {}
    for schema in tools._walk([responses, defs]):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("https://"):
            if ref not in known:
                raise CoverageError(f"unknown external $ref {ref!r}: no docs/schemas/*.json has that $id")
            for sibling in _MERGED_SIBLINGS:
                schema.pop(sibling, None)

    kept = tools.reachable_defs([responses, errors], defs)
    # `external` names only the documents an answer reaches (046-2 review round 1: an unreachable component's
    # reference was pinned too)
    for schema in tools._walk([responses, errors, kept]):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("https://"):
            file, sha = known[ref]
            external[ref] = {"file": file, "sha256": sha}
    ext = frozenset(external)
    typed = sum(1 for entry in responses.values()
                if all(tools.is_typed(s, kept, ext) for k, s in entry.items() if k[:1].isdigit()))
    unions = sum(1 for entry in responses.values() for k, s in entry.items()
                 if k[:1].isdigit() and len(_members(s, kept)) > 1)
    coverage = {"typed": typed, "unions": unions, "excluded": len(excluded), "not_covered": len(not_covered),
                "untyped": len(responses) - typed}
    description = _DESCRIPTION + HISTORY_EXCEPTION
    if history:
        description += " History-form fields: " + ", ".join(history) + "."
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": CONTRACT_ID,
        "kind": KIND,
        "title": "miWorkbench answers",
        "description": description,
        "responses": dict(sorted(responses.items())),
        "excluded": dict(sorted(excluded.items())),
        "not_covered": dict(sorted(not_covered.items())),
        "errors": errors,
        "external": dict(sorted(external.items())),
        "coverage": coverage,
        "$defs": kept,
    }


def live_tool_names() -> list[str]:
    """The live registry's tool names: one server with every category, the deployed shape (the generator's)."""
    from scripts.generate_output_schemas import live_tools

    return sorted(live_tools())


def answers_contract(tool_names: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """The contract body of this process's app. `tool_names` defaults to the live registry."""
    from src.main import app

    names = live_tool_names() if tool_names is None else sorted(tool_names)
    return contract_from(app, app.openapi(), covered_keys(names, app))


def body_of(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k not in tools.UNDIGESTED}


def finished(body: dict[str, Any], revision: int) -> dict[str, Any]:
    """`body` with its `revision` and `content_sha256` (TD11)."""
    doc = dict(body)
    doc["revision"] = revision
    doc["content_sha256"] = tools.content_digest(body)
    return doc


def resolver(*docs: dict[str, Any]) -> tools.Resolver:
    """Reads each document schema the documents' `external` names, by sha256, from this checkout's tree and history."""
    files = {sid: row["file"] for doc in docs for sid, row in (doc.get("external") or {}).items()}
    return tools.git_resolver(SCHEMAS_DIR.parents[1], files)


def write_contract(path: Path = DEFAULT_PATH, body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Write `path` from `body` (default: this process's contract), refusing a breaking change (FR-046.13).

    Equal bodies: nothing is written. All changes additive: `revision` + 1. No committed file: revision 1. Any breaking
    change: `BreakingAnswerChange`, and the committed bytes are left as they were."""
    body = answers_contract() if body is None else body_of(body)
    if path.is_file():
        old: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if body_of(old) == body:
            return old
        bad = tools.breaking(tools.classify(body_of(old), body, resolver(old, body)))
        if bad:
            raise BreakingAnswerChange(bad)
        doc = finished(body, int(old.get("revision", 0)) + 1)
    else:
        doc = finished(body, 1)
    path.write_text(render(doc), encoding="utf-8")
    return doc


def problems(doc: dict[str, Any]) -> list[str]:
    """What is wrong with a committed contract on its own (FR-046.14): a hand edit leaves `content_sha256` stale."""
    out = []
    revision = doc.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        out.append(f"revision {revision!r} is not a positive integer")
    if doc.get("content_sha256") != tools.content_digest(doc):
        out.append("content_sha256 does not match the content: was it edited by hand? Regenerate it with the writer "
                   "(MIWORKBENCH_WRITE_CONTRACTS=1)")
    return out
