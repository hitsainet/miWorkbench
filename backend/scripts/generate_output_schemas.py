"""Generate `backend/src/mcp_server/output_schemas.json`: each tool's `outputSchema`, generated from CONTRACTS (046
FR-046.18 to FR-046.27; MW4 to MW6, MW15, MW16; WT5 to WT7). The 045 follow-up (debt RL-1) typed 69 native tools
permissively from the live OpenAPI; 046 types every tool it can from the contract of each route it answers.

For each tool of the LIVE registry (one server with every category, the deployed shape):

1. **Routes.** `src/mcp_server/route_map.tool_routes()[tool]`, by product (the one route walker, WT1).
2. **The answer part.** A tool whose every `return` is one route's body unchanged (`route_map.returns_unchanged`)
   answers those routes' 2xx entries; any other tool needs its row in `scripts/output_overlays.py` (WT7). Entries come
   from the product's contract:
   - miWorkbench: the committed `docs/schemas/miworkbench-answers-v1.json` (046 FTASKS 10.1, FR-046.18), the
     contract the writer (`src/api/answers_contract.py`) wrote and its pin test holds to the live routes: `required`
     means always sent, history-form fields carry no `required` beneath them, objects are open. (Until 046-2 it was
     the live OpenAPI through the writer's transforms, WT5, and a tool whose route was untyped waited with an interim
     reason, now retired);
   - miServer: the vendored `src/mcp_server/vendored/miserver-answers-v1.json` (FR-046.25). The management envelope is
     unwrapped to its `data`, as `MiServerClient` does (`miserver_client.py:221`); a `data` of `null` answers
     `{"result": null}` (`server.structured_result`); `/v1` and `raw_get` answers carry no envelope;
   - a list becomes `{"result": [...]}` with `result` required; a scalar `{"result": <scalar>}`;
   - a nullable miWorkbench route answers `{}` for `null` (`client.py:117`), so it is `anyOf` [the model, the empty
     object `{"type": "object", "maxProperties": 0}`] (MW6, FR-046.21);
   - several routes or statuses become `anyOf` of their parts;
   - a reference to a document schema (a circuit or cluster definition) is embedded from `docs/schemas/`, after
     checking its sha256 against the contract's `external`, PER PRODUCT (`_pinned_document`): each product's answers
     embed the bytes ITS contract pins. While miServer still pins an older revision of a document this repo publishes
     (producer first), miServer's tools embed the pinned bytes from `docs/schemas/pinned/` under that copy's file name
     and the native tools the current document under its own; one tool that would need both stops the build.
3. **The overlay** (`OVERLAYS`): the wrapper's keys merged beside the body, or the body nested under a key, plus the
   tool's own built variants.
4. **The shared keys** (`SHARED`): each legacy key is allowed and optional beside the answer; `next` is NEVER
   constrained (045 A30): a route's own `next` is dropped, and nothing declares one.
5. **The alternatives** (MW4), each typed from its authority: miWorkbench's tool-side approval answer
   (`approval_request_id` and `status`), miServer's relayed pending (`ApprovalPending` from the vendored contract,
   plus `decide_in`; only where the tool's routes answer it), `unavailable` (the result language's), `refused` and the
   argument `error`. The root is `{"type": "object", "properties": {approval, next_steps}, "anyOf": [...]}`.
6. **`approval` and `next_steps`** at the root, from the vendored result language through
   `tool_classes.result_language_properties` (the generic envelope's own definition).
7. **Trim (WT6, BR-7):** local `#/$defs/` only, the reached definitions only, Pydantic's generated `title` strings
   dropped, and the result language's descriptions dropped (`result_language_properties`); each answer field keeps its
   own `description`.
   Last, every `format: date-time` and `format: time` is removed, embedded documents included
   (`tool_classes.without_offset_formats`, hotfix 2026-11-26): the backend serialises naive datetimes with no offset,
   and a client that asserts formats rejects the whole answer over one such field.
8. **Coverage (FR-046.27).** A tool that cannot be typed stays on the generic envelope only with a reason from
   `REASONS`, the closed list, and the file states `coverage` and the `sources` it was built from.

No `required` is relaxed and no object is closed: `required` means "always sent" (SH7), and extra keys stay allowed.

Regenerate, or check (exits 1 on a difference), from `backend/` with the test environment (`DATABASE_URL`,
`SECRET_KEY`, …) exported, since the live tool registry is built from the backend's modules::

    python -m scripts.generate_output_schemas --write
    python -m scripts.generate_output_schemas --check

`tests/unit/test_mcp_typed_output_schemas.py` regenerates it and fails on a difference.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

from src.mcp_server.route_map import normalise as _normalise
from src.mcp_server.route_map import returns_unchanged, tool_routes
from src.mcp_server.tool_classes import without_offset_formats

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
SCHEMAS_PATH = BACKEND / "src" / "mcp_server" / "output_schemas.json"
VENDORED = BACKEND / "src" / "mcp_server" / "vendored"
MISERVER_CONTRACT = VENDORED / "miserver-answers-v1.json"
DOC_SCHEMAS = REPO / "docs" / "schemas"
#: The bytes a consumer's contract still pins after this repo published a newer revision (`_pinned_bytes`).
PINNED_DOCUMENTS = DOC_SCHEMAS / "pinned"
#: miWorkbench's own answer contract (046-2), the native tools' source (FTASKS 10.1).
NATIVE_CONTRACT = DOC_SCHEMAS / "miworkbench-answers-v1.json"
REF = "#/components/schemas/"
DEFS = "#/$defs/"
API_PREFIX = "/api/v1"
VERSION = 2

#: Keys a tool's answer may carry beside the route's own (the legacy pending, refusal and unavailable keys), all
#: optional. `approval` and `next_steps` are the result language's, at the root.
SHARED: dict[str, dict[str, Any]] = {
    "action": {"type": "string"},
    "approval_id": {"type": "string"},
    "approval_request_id": {"type": "string"},
    "decide_in": {"type": "string"},
    "expires_at": {"type": ["string", "null"]},
    "hint": {"type": "string"},
    "reason": {},
    "refused": {"type": "string"},
    "status": {"type": "string"},
    "summary": {"type": ["string", "object"]},  # miServer's pending summary is a map of card lines
    "unavailable": {"type": "string"},
}
#: The result language's keys (`tool_classes.RESULT_LANGUAGE_KEYS`): a route's own property of that name is dropped
#: from the answer part, because `result_language.conform` repairs or removes it before the answer leaves the tool.
CONTRACT_KEYS = ("approval", "next_steps")

#: FR-046.27's closed list of reasons a tool stays on the generic envelope. 046-1's interim reason (a native tool
#: whose route was untyped in the live OpenAPI) is retired with 10.1: every covered route is typed in the contract.
R_NONE = "no structured result"
R_FORGE = "calls miForge, which publishes after its 020 (D2)"
R_DATAWORKS = "waits for miDataworks' answer contract (BR-6)"
R_GUIDE = "the guide"
REASONS = (R_NONE, R_FORGE, R_DATAWORKS, R_GUIDE)
#: The tools whose reason is fixed by the requirement, not derived.
FIXED_REASONS: dict[str, str] = {
    "miworkbench_howto": R_GUIDE,
    "import_wsp_candidate": R_FORGE,
    "register_probe_dataset": R_DATAWORKS,
}


class GenerationError(RuntimeError):
    """The contracts or the overlay table cannot honestly produce a tool's schema."""


# ── small JSON-schema transforms ───────────────────────────────────────────────────────────────────────────────


def _walk_refs(node: Any, root: bool = True) -> set[str]:
    """The local `#/$defs/` names `node` references. An embedded document schema (a dict with its own `$id`) is its
    own resource: its references resolve inside it, so the walk does not enter it."""
    found: set[str] = set()
    if isinstance(node, dict):
        if not root and "$id" in node:
            return found
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str) and value.startswith(DEFS):
                found.add(value[len(DEFS):])
            else:
                found |= _walk_refs(value, False)
    elif isinstance(node, list):
        for value in node:
            found |= _walk_refs(value, False)
    return found


def _rewrite(node: Any, old: str, new: str) -> Any:
    if isinstance(node, dict):
        return {k: (new + v[len(old):] if k == "$ref" and isinstance(v, str) and v.startswith(old)
                    else _rewrite(v, old, new)) for k, v in node.items()}
    if isinstance(node, list):
        return [_rewrite(v, old, new) for v in node]
    return node


def _open(node: Any) -> Any:
    """`additionalProperties: false` removed everywhere (SH7: extra keys are allowed); schema-valued maps kept."""
    if isinstance(node, dict):
        return {k: _open(v) for k, v in node.items() if not (k == "additionalProperties" and v is False)}
    if isinstance(node, list):
        return [_open(v) for v in node]
    return node


#: Keywords whose value is a map of NAME → schema: the names are not keywords.
_NAMED = ("properties", "$defs", "patternProperties", "dependentSchemas")


def _untitled(node: Any) -> Any:
    """Every `title` KEYWORD whose value is a string removed (Pydantic generates them; WT6). A property NAMED `title`
    is kept."""
    if isinstance(node, list):
        return [_untitled(v) for v in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue
        if key in _NAMED and isinstance(value, dict):
            out[key] = {name: _untitled(sub) for name, sub in value.items()}
        else:
            out[key] = _untitled(value)
    return out


# ── the sources ──────────────────────────────────────────────────────────────────────────────────────────────


class Source:
    """One product's answers: `entries[(METHOD, normalised path)] = {status: schema}` and `defs`."""

    def __init__(self, product: str, entries: dict, defs: dict, excluded: set, external: dict,
                 file: Optional[str] = None, sha: Optional[str] = None, empty: Optional[set] = None) -> None:
        self.product, self.entries, self.defs, self.excluded = product, entries, defs, excluded
        self.external, self.file, self.sha = external, file, sha
        #: routes answering no body at all (a 204), which `MiWorkbenchClient.request` answers as `{}`
        self.empty = empty or set()
        self._resolved: Optional[dict[str, Any]] = None

    def resolved_defs(self) -> dict[str, Any]:
        """`defs` with every document reference embedded at THIS product's pin (`_embed_externals`), plus the embedded
        documents themselves under their `$defs` keys. Computed once per source."""
        if self._resolved is None:
            embedded: dict[str, Any] = {}
            resolved = _embed_externals(self.defs, self, embedded)
            self._resolved = {**resolved, **embedded}
        return self._resolved

    def entry(self, method: str, path: str) -> Optional[dict[str, Any]]:
        """The entry of a route as a TOOL names it: miWorkbench's tools call paths under `/api/v1`."""
        return self.entries.get(self.key(method, path))

    def key(self, method: str, path: str) -> tuple[str, str]:
        prefix = API_PREFIX if self.product == "miworkbench" else ""
        return method, _normalise(prefix + path)


def native_source(path: Path = NATIVE_CONTRACT, contract: Optional[dict[str, Any]] = None) -> Source:
    """miWorkbench's answers from its committed answer contract (046 FTASKS 10.1): each covered route's 2xx entries,
    the contract's `$defs` and `external`, and its `excluded` routes (a `media_type` of null answers no body, which
    `MiWorkbenchClient.request` answers as `{}`). `contract` is for the generator's own tests (a made-up contract)."""
    if contract is None:
        contract = json.loads(path.read_text(encoding="utf-8"))
    entries = {}
    excluded, empty = set(), set()
    for key, statuses in contract["responses"].items():
        method, route = key.split(" ", 1)
        entries[(method, _normalise(route))] = {s: v for s, v in statuses.items() if s[:1] == "2"}
        if statuses.get("x-no-body"):  # a JSON route that also answers a status with no body (a 204): `{}` too
            empty.add((method, _normalise(route)))
    for key, row in contract.get("excluded", {}).items():
        method, route = key.split(" ", 1)
        excluded.add((method, _normalise(route)))
        # a status the route answers with no body at all, which the client answers as `{}`; a status with a non-JSON
        # body is no answer a tool gives (the client raises on it)
        if None in row.get("statuses", {"": row.get("media_type")}).values():
            empty.add((method, _normalise(route)))
    return Source("miworkbench", entries, contract["$defs"], excluded, contract.get("external", {}),
                  file=path.name, sha=contract.get("content_sha256"), empty=empty)


def miserver_source(path: Path = MISERVER_CONTRACT) -> Source:
    raw = path.read_text(encoding="utf-8")
    contract = json.loads(raw)
    entries = {}
    for key, statuses in contract["responses"].items():
        method, route = key.split(" ", 1)
        entries[(method, _normalise(route))] = {s: v for s, v in statuses.items() if s[:1] == "2"}
    excluded = {(k.split(" ", 1)[0], _normalise(k.split(" ", 1)[1])) for k in contract.get("excluded", {})}
    return Source("miserver", entries, contract["$defs"], excluded, contract.get("external", {}),
                  file=path.name, sha=contract["content_sha256"])


# ── one route's answer as schema parts ───────────────────────────────────────────────────────────────────────


class _Untyped(Exception):
    pass


def _is_pending(schema: dict[str, Any], source: Source) -> bool:
    """A miServer schema whose `data` (or itself) is `ApprovalPending`."""
    ref = schema.get("$ref", "")
    if ref == DEFS + "ApprovalPending":
        return True
    if ref.startswith(DEFS):
        target = source.defs[ref[len(DEFS):]]
        if "anyOf" in target:
            return any(_is_pending(alt, source) for alt in target["anyOf"])
        data = (target.get("properties") or {}).get("data")
        if data is not None and "success" in (target.get("properties") or {}):
            return any(_is_pending(alt, source) for alt in data.get("anyOf", [data]))
    return False


def _object_part(props: dict[str, Any], required: list[str], with_shared: bool = True) -> dict[str, Any]:
    props = copy.deepcopy(props)
    for key in ("next", *CONTRACT_KEYS):
        props.pop(key, None)  # A30: `next` is never constrained; the result language owns the other two
    required = [r for r in required if r in props]
    if "result" in props and with_shared:
        props["result"] = {}  # an object route's own `result` stays unconstrained, as before (a non-dict answer)
    if with_shared:
        for key, shared in SHARED.items():
            # A key the route (or the variant) declares keeps ITS type: the route is that field's authority, and the
            # legacy pending, refusal and unavailable shapes are the alternatives'. (045 widened it to `anyOf` [own,
            # shared], which let a not-found variant's `status: "not_found"` accept any string; 046 question 4.)
            props.setdefault(key, shared)
    part: dict[str, Any] = {"type": "object", "properties": dict(sorted(props.items()))}
    if required:
        part["required"] = sorted(required)
    return part


def _result(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": {"result": schema}, "required": ["result"]}


EMPTY_OBJECT = {"type": "object", "maxProperties": 0}


def parts_of(schema: dict[str, Any], source: Source, *, envelope: bool, embedded: dict[str, Any],
             with_shared: bool = True) -> list[dict[str, Any]]:
    """The answer parts for one 2xx body `schema` of `source`. Raises `_Untyped` with the reason."""
    if not schema:
        raise _Untyped("the route declares no response model (its body is untyped)")
    ref = schema.get("$ref")
    if isinstance(ref, str) and not ref.startswith(DEFS):
        return [_external(ref, source, embedded)]
    if isinstance(ref, str):
        target = source.defs[ref[len(DEFS):]]
        props = target.get("properties") or {}
        if envelope and "success" in props and "data" in props:
            return _unwrap(props["data"], source, embedded, with_shared)
        if "anyOf" in target or "oneOf" in target:  # a named union (it may hold `null` too)
            union = {"anyOf": target.get("anyOf", target.get("oneOf"))}
            return parts_of(union, source, envelope=envelope, embedded=embedded, with_shared=with_shared)
        if target.get("type") == "object" and props:
            return [_object_part(props, target.get("required") or [], with_shared)]
        if target.get("type") == "array":
            return [_result(schema)]
        raise _Untyped("the route's body is an untyped object or list (a dict, Any or a Response)")
    if "anyOf" in schema or "oneOf" in schema:
        alts = schema.get("anyOf", schema.get("oneOf"))
        non_null = [a for a in alts if a.get("type") != "null"]
        out = [p for alt in non_null if not _is_pending(alt, source)
               for p in parts_of(alt, source, envelope=envelope, embedded=embedded, with_shared=with_shared)]
        if len(non_null) < len(alts):  # nullable: miWorkbench's client answers {} for null (MW6)
            out.append(EMPTY_OBJECT if source.product == "miworkbench" else _result({"type": "null"}))
        return out
    kind = schema.get("type")
    if kind == "array":
        items = schema.get("items") or {}
        if not items:
            raise _Untyped("the route's body is an untyped list")
        return [_result(schema)]
    if kind in ("string", "integer", "number", "boolean"):
        return [_result(schema)]
    if kind == "object" and schema.get("properties"):
        return [_object_part(schema["properties"], schema.get("required") or [], with_shared)]
    raise _Untyped("the route's body is an untyped object or list (a dict, Any or a Response)")


def _unwrap(data: dict[str, Any], source: Source, embedded: dict, with_shared: bool) -> list[dict[str, Any]]:
    """The envelope's `data`: its non-null alternatives, plus `{"result": null}` for a `null` data."""
    alts = data.get("anyOf", [data])
    out: list[dict[str, Any]] = []
    for alt in alts:
        if alt.get("type") == "null":
            out.append(_result({"type": "null"}))
        elif not _is_pending(alt, source):
            out += parts_of(alt, source, envelope=False, embedded=embedded, with_shared=with_shared)
    return out


def _external(ref: str, source: Source, embedded: dict[str, Any]) -> dict[str, Any]:
    """A document schema as an answer: embedded (`_embed_externals` resolves it)."""
    return {"$ref": ref}


@functools.lru_cache(maxsize=1)
def _documents() -> dict[str, Path]:
    """`{$id: path}` of the document schemas in `docs/schemas/` (the files the contracts' `external` names)."""
    out: dict[str, Path] = {}
    for path in sorted(DOC_SCHEMAS.glob("*.json")):
        try:
            sid = json.loads(path.read_text(encoding="utf-8")).get("$id")
        except (ValueError, AttributeError):
            continue
        if isinstance(sid, str):
            out[sid] = path
    return out


def _pinned_document(ref: str, path: Path, source: Source) -> tuple[str, bytes]:
    """`(the $defs key, the bytes)` of the document `ref` as `source`'s contract pins it (its `external` sha256).

    Normally `path`, this repo's own `docs/schemas/` file, under its own name. A PRODUCER moves first (contracts rule
    9): when miWorkbench publishes a new revision of a document a consumer's answer contract pins, that consumer still
    answers the old document until it re-vendors, so the old bytes are kept under `docs/schemas/pinned/`
    (content-addressed: only a file whose sha256 is the pin is used) for exactly that window, and are embedded under
    THAT file's name, so a product answering the current revision and one answering the pinned one never share a key.
    `test_pinned_documents.py` fails on a copy no pin names any more. A pin no file matches stops the build."""
    current = path.read_bytes()
    row = source.external.get(ref)
    if row is None or hashlib.sha256(current).hexdigest() == row["sha256"]:
        return path.name, current
    for copy_path in sorted(PINNED_DOCUMENTS.glob("*.json")):
        data = copy_path.read_bytes()
        if hashlib.sha256(data).hexdigest() == row["sha256"]:
            return copy_path.name, data
    raise GenerationError(f"{path} is not the document {source.product}'s contract pins ({ref}), and no file in "
                          f"{PINNED_DOCUMENTS} has those bytes")


def _pinned_bytes(ref: str, path: Path, source: Source) -> bytes:
    """The bytes of `ref` that `source`'s contract pins (`_pinned_document`)."""
    return _pinned_document(ref, path, source)[1]


def _embed_externals(node: Any, source: Source, embedded: dict[str, Any]) -> Any:
    """Every `$ref` to a document schema's `$id` in `node`, an answer part or definition of `source`, rewritten to
    `#/$defs/<key>`, with the document embedded in `embedded[key]` at the bytes `source`'s contract pins (it keeps its
    own `$id`, so its own references resolve inside it). An `$id` with no file in `docs/schemas/` is an error. Nothing
    is ever fetched."""
    if isinstance(node, list):
        return [_embed_externals(v, source, embedded) for v in node]
    if not isinstance(node, dict):
        return node
    if "$id" in node:
        return node  # an embedded document: its references are its own
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "$ref" and isinstance(value, str) and not value.startswith("#"):
            path = _documents().get(value)
            if path is None:
                raise GenerationError(f"a reference to a document schema with no file in docs/schemas/: {value}")
            name, data = _pinned_document(value, path, source)
            document = json.loads(data)
            document.pop("$schema", None)
            embedded[name] = _untitled(_open(document))
            out[key] = DEFS + name
        else:
            out[key] = _embed_externals(value, source, embedded)
    return out


def _external_refs(node: Any) -> list[str]:
    """Document references left unresolved in `node` (outside embedded documents): none may reach the file."""
    if isinstance(node, list):
        return [r for v in node for r in _external_refs(v)]
    if not isinstance(node, dict) or "$id" in node:
        return []
    return [v for k, v in node.items() if k == "$ref" and isinstance(v, str) and not v.startswith("#")] + [
        r for k, v in node.items() if k != "$ref" for r in _external_refs(v)]


def _one_revision_per_document(schema: dict[str, Any], tool: str) -> None:
    """Two embedded revisions of one document (`$id`) in one tool's schema would be two resources with one identifier:
    a tool answering both products' copies during a producer-first window stops the build instead."""
    seen: dict[str, str] = {}
    for name, sub in (schema.get("$defs") or {}).items():
        sid = sub.get("$id") if isinstance(sub, dict) else None
        if isinstance(sid, str):
            if sid in seen:
                raise GenerationError(f"{tool}: embeds two revisions of {sid} ({seen[sid]}, {name}); one tool cannot "
                                      "answer both until the consumer re-vendors")
            seen[sid] = name


# ── alternatives ─────────────────────────────────────────────────────────────────────────────────────────────


def alternatives(result_language: dict[str, Any], relayed_pending: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """The non-error shapes every tool may answer besides its resource (MW4), typed from their authorities."""
    unavailable = copy.deepcopy(result_language["$defs"]["unavailable"])
    unavailable.pop("description", None)
    out: list[dict[str, Any]] = [
        # miWorkbench's tool-side approval answer: a pending request, or a re-call the route refused (status
        # `not_found`, `mismatch`, `consumed`, …: `steering._refused_recall`, `approvals` and the probe gates).
        {"type": "object", "required": ["approval_request_id", "status"],
         "properties": {"approval_request_id": {"type": "string"}, "status": {"enum": list(tool_approval_statuses())},
                        "hint": {"type": "string"}}},
        _open(unavailable),
        {"type": "object", "required": ["refused"], "properties": {"refused": {"type": "string"}, "reason": {}}},
        {"type": "object", "required": ["error"], "properties": {"error": {"type": "string"},
                                                                 "message": {"type": "string"}}},
    ]
    if relayed_pending is not None:
        out.insert(1, relayed_pending)
    return out


#: The statuses a miWorkbench tool's own approval answer carries beside `approval_request_id`, besides the stored
#: request's (operator decision 2026-11-25, 046 question 4: an enum, never any string): the first call's
#: `pending_approval`, and a re-call the route refused, `not_found` (404) or `mismatch` (422; `steering`,
#: `probe_monitors`, `wsp`, `approvals`). A refused re-call's 409 passes the stored request's own status through.
TOOL_APPROVAL_ANSWERS = ("pending_approval", "not_found", "mismatch")


def tool_approval_statuses() -> tuple[str, ...]:
    """`TOOL_APPROVAL_ANSWERS` plus every status a stored approval request can hold (`ApprovalStatus`, the table's
    own enum: a 409 on a re-call names it), sorted. `test_mcp_typed_output_schemas.py` pins every literal the tool
    modules write beside `approval_request_id` to this set."""
    from src.models.agent_approval import ApprovalStatus

    return tuple(sorted({*TOOL_APPROVAL_ANSWERS, *(s.value for s in ApprovalStatus)}))


def _relayed_pending() -> dict[str, Any]:
    """miServer's pending answer as the relay returns it: `ApprovalPending` plus `decide_in`."""
    return {"type": "object", "allOf": [{"$ref": DEFS + "ApprovalPending"}],
            "required": ["decide_in"], "properties": {"decide_in": {"type": "string"}}}


# ── overlays ─────────────────────────────────────────────────────────────────────────────────────────────────


def _model_schema(model: Any, defs: dict[str, Any]) -> dict[str, Any]:
    schema = model.model_json_schema(mode="serialization", ref_template=DEFS + "{model}")
    for name, sub in schema.pop("$defs", {}).items():
        defs.setdefault(name, sub)
    return schema


def _with_wrapper(parts: list[dict[str, Any]], wrapper: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for part in parts:
        if part == EMPTY_OBJECT or part.get("type") != "object" or "properties" not in part:
            out.append(part)
            continue
        merged = copy.deepcopy(part)
        merged["properties"] = dict(sorted({**merged["properties"], **wrapper.get("properties", {})}.items()))
        required = sorted(set(merged.get("required", [])) | set(wrapper.get("required", [])))
        if required:
            merged["required"] = required
        out.append(merged)
    return out


def _raw(part: dict[str, Any]) -> dict[str, Any]:
    """The body a part describes, before it is structured: `{"result": X}` (a list, scalar or null) is X."""
    if set(part.get("properties", {})) == {"result"} and part.get("required") == ["result"]:
        return part["properties"]["result"]
    return part


def _nested(bodies: list[dict[str, Any]], key: str, wrapper: dict[str, Any]) -> dict[str, Any]:
    raw = [_raw(b) for b in bodies]
    body = raw[0] if len(raw) == 1 else {"anyOf": raw}
    return {"type": "object", "properties": dict(sorted({key: body, **wrapper.get("properties", {})}.items())),
            "required": sorted({key, *wrapper.get("required", [])})}


# ── the document ─────────────────────────────────────────────────────────────────────────────────────────────


def live_tools() -> dict[str, Any]:
    """`{name: Tool}` of ONE server with every category, the deployed shape (`test_tool_classes.py` builds it so)."""
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import EXTERNAL_PRODUCTS, all_categories

    overrides = {p.spec.override_field: f"http://{p.spec.name}.invalid" for p in EXTERNAL_PRODUCTS}
    mcp, _ = build_server(MCPSettings(tool_categories=",".join(sorted(all_categories())), allow_anonymous=True,
                                      _env_file=None, **overrides), stdio=True)
    return {t.name: t for t in mcp._tool_manager.list_tools()}  # noqa: SLF001


def _key(product: str, method: str, path: str) -> str:
    return f"{method} {API_PREFIX}{path}" if product == "miworkbench" else f"{method} {path}"


def _split(key: str) -> tuple[str, str, str]:
    method, path = key.split(" ", 1)
    if path.startswith(API_PREFIX + "/"):
        return "miworkbench", method, path[len(API_PREFIX):]
    return "miserver", method, path


def _trimmed(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """`schema` with the reached definitions only, every title dropped (WT6)."""
    reached: dict[str, Any] = {}
    todo = _walk_refs(schema)
    while todo:
        name = todo.pop()
        if name in reached:
            continue
        if name not in defs:
            raise GenerationError(f"an unresolved reference: #/$defs/{name}")
        reached[name] = defs[name]
        if "$id" not in defs[name]:  # an embedded document resolves its own references
            todo |= _walk_refs(defs[name]) - set(reached)
    out = dict(schema)
    if reached:
        out["$defs"] = dict(sorted(reached.items()))
    return _untitled(out)


def _merge_defs(into: dict[str, Any], new: dict[str, Any], tool: str) -> None:
    for name, sub in new.items():
        if name in into and into[name] != sub:
            raise GenerationError(f"{tool}: two products define {name} differently")
        into[name] = sub


def tool_schema(name: str, fn: Any, sources: dict[str, Source], overlays: dict[str, Any],
                result_language: dict[str, Any], fields: dict[str, Any]) -> tuple[Optional[dict[str, Any]], str, str]:
    """`(schema, route, "")` or `(None, route, reason)` for one tool."""
    routes = tool_routes().get(name, frozenset())
    products = {p for p, _m, _x in routes}
    route_text = "; ".join(sorted(_key(p, m, _normalise(x)) for p, m, x in routes)) or None
    if name in FIXED_REASONS:
        return None, route_text or "", FIXED_REASONS[name]
    if "dataworks" in products:
        return None, route_text or "", R_DATAWORKS
    unchanged, _why = returns_unchanged(fn)
    overlay = overlays.get(name)
    if unchanged is None and overlay is None:
        raise GenerationError(f"{name} builds or changes its answer and has no row in scripts/output_overlays.py")
    if unchanged is not None and overlay is not None:
        raise GenerationError(f"{name} answers its routes unchanged; its overlay row is stale")
    base = (sorted(_key(p, m, x) for p, m, x in unchanged) if unchanged is not None else list(overlay.base))
    defs: dict[str, Any] = {}
    embedded: dict[str, Any] = {}
    bodies: list[dict[str, Any]] = []
    nested = overlay is not None and overlay.nest is not None
    try:
        for key in base:
            product, method, path = _split(key)
            source = sources[product]
            entry = source.entry(method, path)
            if entry is None:
                if source.key(method, path) in source.empty:
                    # review round 1, finding 7: no body at all, which `MiWorkbenchClient.request` answers as `{}`
                    if EMPTY_OBJECT not in bodies:
                        bodies.append(EMPTY_OBJECT)
                    continue
                if source.key(method, path) in source.excluded:
                    raise _Untyped(f"{key} has no JSON answer")
                raise GenerationError(f"{name}: {key} is not in {product}'s contract")
            for status in sorted(entry):
                for part in parts_of(entry[status], source, envelope=product == "miserver", embedded=embedded,
                                     with_shared=not nested):
                    part = _embed_externals(part, source, embedded)
                    if part not in bodies:
                        bodies.append(part)
            if source.key(method, path) in source.empty and EMPTY_OBJECT not in bodies:
                bodies.append(EMPTY_OBJECT)  # its `x-no-body` status, which the client answers as `{}`
            _merge_defs(defs, source.resolved_defs(), name)
    except _Untyped as why:
        # Every covered route is typed in both contracts (046-2), so an untyped answer is a defect, never a reason.
        raise GenerationError(f"{name}: {why}") from None
    if overlay is not None:
        wrapper = _model_schema(overlay.wrapper, defs) if overlay.wrapper is not None else {}
        if nested:
            bodies = [_nested(bodies, overlay.nest, wrapper)]
        elif wrapper:
            bodies = _with_wrapper(bodies, wrapper)
        for model in overlay.variants:
            variant = _model_schema(model, defs)
            if overlay.items_of is not None and overlay.items_of in variant.get("properties", {}):
                item_parts = [b["properties"]["result"]["items"] for b in bodies
                              if "result" in b.get("properties", {}) and "items" in b["properties"]["result"]]
                extra = _model_schema(overlay.item_wrapper, defs) if overlay.item_wrapper is not None else {}
                if item_parts:
                    items = item_parts[0]
                    if extra:
                        items = {"allOf": [items], "properties": extra.get("properties", {})}
                    variant["properties"][overlay.items_of] = {"type": "array", "items": items}
            bodies.append(_object_part(variant.get("properties", {}), variant.get("required", [])))
        if overlay.items_of is not None:
            bodies = [b for b in bodies if overlay.items_of in b.get("properties", {})]  # the route's list is not answered
    if not bodies:
        raise GenerationError(f"{name}: no answer part")
    pending = None
    if any(_is_pending(s, sources["miserver"]) for p, m, x in routes if p == "miserver"
           for s in (sources["miserver"].entry(m, x) or {}).values()):
        pending = _relayed_pending()
        _merge_defs(defs, sources["miserver"].resolved_defs(), name)
    schema: dict[str, Any] = {"type": "object", "properties": copy.deepcopy(fields),
                              "anyOf": [*bodies, *alternatives(result_language, pending)]}
    _merge_defs(defs, embedded, name)
    schema = _trimmed(schema, defs)
    left = _external_refs(schema)
    if left:
        raise GenerationError(f"{name}: a document reference no product's answer resolved: {sorted(set(left))}")
    _one_revision_per_document(schema, name)
    return without_offset_formats(schema), route_text or "", ""


def generate(tools: Optional[dict[str, Any]] = None, native: Optional[Source] = None,
             miserver: Optional[Source] = None) -> dict[str, Any]:
    from scripts.output_overlays import OVERLAYS
    from src.mcp_server.result_language import load_contract
    from src.mcp_server.tool_classes import TOOL_CLASSES, result_language_properties

    tools = live_tools() if tools is None else tools
    sources = {"miworkbench": native or native_source(), "miserver": miserver or miserver_source()}
    result_language = load_contract()
    fields = result_language_properties(result_language)
    typed: dict[str, Any] = {}
    untyped: dict[str, Any] = {}
    stale = sorted(set(OVERLAYS) - set(tools))
    if stale:
        raise GenerationError(f"overlay rows for tools that are not registered: {stale}")
    for name in sorted(tools):
        if TOOL_CLASSES[name].result == "none":
            untyped[name] = {"route": None, "reason": R_NONE}
            continue
        schema, route, reason = tool_schema(name, tools[name].fn, sources, OVERLAYS, result_language, fields)
        if schema is None:
            untyped[name] = {"route": route or None, "reason": reason}
        else:
            typed[name] = schema
    return {
        "version": VERSION,
        "coverage": {"typed": len(typed), "generic": len(untyped)},
        "sources": {source.file: source.sha for source in sources.values()},
        "tools": typed,
        "untyped": untyped,
    }


def document_text(**kwargs: Any) -> str:
    return json.dumps(generate(**kwargs), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    if argv[1:] == ["--write"]:
        SCHEMAS_PATH.write_text(document_text(), encoding="utf-8")
        return 0
    if argv[1:] == ["--check"]:
        current = SCHEMAS_PATH.read_text(encoding="utf-8") if SCHEMAS_PATH.exists() else ""
        if current != document_text():
            print(f"{SCHEMAS_PATH} is stale or was edited by hand: run with --write", file=sys.stderr)
            return 1
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
