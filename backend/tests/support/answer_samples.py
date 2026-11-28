"""A full sample answer for a JSON schema: EVERY property filled, to depth 8, the first non-null option of a union
(046 FR-046.19's "each typed tool answers its route's full body"; FTDD §6.6's base body for an overlay).

The schema is a generated tool schema (`output_schemas.json`) or a contract entry, with its `$defs`. An embedded
document schema (a definition with its own `$id`, e.g. `circuit-definition-v1.json`) resolves its own references,
so the sampler keeps the current resource while it descends.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Optional

MAX_DEPTH = 16

#: Strings tried, in order, against a `pattern` (the contracts' patterns are ids, digests and keys).
_PATTERN_CANDIDATES = ("s", "ab", "a-b", "0" * 64, "sha256:" + "0" * 64, "apr_" + "0" * 32, "pm_0123456789ab",
                       "2026-11-25T00:00:00Z", "a.b", "A", "1", "s_s", "stack", "max", "/a",
                       "mistudio.probe-identity/1")
#: A string for each `format` the published documents use (the answer middleware checks formats as ajv does).
_FORMATS = {"date-time": "2026-11-25T00:00:00Z", "date": "2026-11-25", "time": "00:00:00Z",
            "uuid": "00000000-0000-4000-8000-000000000000", "uri": "https://example.invalid/a"}


@functools.lru_cache(maxsize=1)
def _published() -> dict[str, dict[str, Any]]:
    """`$id` → document, for every schema miWorkbench publishes under `docs/schemas/` (046 FR-046.11: an answer that
    IS one of them is a `$ref` to its `$id`, which a sample resolves to the document itself)."""
    root = Path(__file__).resolve().parents[3] / "docs" / "schemas"
    out = {}
    for path in sorted(root.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(document, dict) and isinstance(document.get("$id"), str):
            out[document["$id"]] = document
    return out


def published_sample(filename: str) -> dict[str, Any]:
    """A whole document of one schema miWorkbench publishes (`docs/schemas/<filename>`), every field filled and valid
    in the strict form (formats and patterns included): what a route documented as that `$ref` answers."""
    root = Path(__file__).resolve().parents[3] / "docs" / "schemas"
    schema = json.loads((root / filename).read_text(encoding="utf-8"))
    return sample(schema, schema)


def sample(schema: Any, root: dict[str, Any], depth: int = 0, resource: Optional[dict[str, Any]] = None,
           prefer_null: bool = False) -> Any:
    """A value for `schema`, every property filled. `root` holds the `$defs` the references name. `prefer_null`: a
    nullable value is `None` and an optional property is left out (the most neutral completion of a fake:
    `miserver_fakes.complete_response`)."""
    resource = root if resource is None else resource
    if not isinstance(schema, dict) or depth > MAX_DEPTH:
        return None
    if "$id" in schema and schema is not root:
        resource = schema
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        target = (resource.get("$defs") or {}).get(ref.split("/")[-1])
        if target is None:
            target = (root.get("$defs") or {})[ref.split("/")[-1]]
        if "$id" in target:
            resource = target
        merged = {**target, **{k: v for k, v in schema.items() if k != "$ref"}}
        return sample(merged, root, depth, resource, prefer_null)
    if isinstance(ref, str) and ref == "#":
        return sample(resource, root, depth + 1, resource, prefer_null)
    if isinstance(ref, str) and ref in _published():  # a published document, by its `$id`
        document = _published()[ref]
        return sample(document, root, depth + 1, document, prefer_null)
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return next((v for v in schema["enum"] if v is not None), schema["enum"][0])
    for key in ("anyOf", "oneOf"):
        if prefer_null and any(isinstance(s, dict) and s.get("type") == "null" for s in schema.get(key) or []):
            return None
        options = [s for s in schema.get(key) or [] if isinstance(s, dict) and s.get("type") != "null"
                   and s.get("maxProperties") != 0]
        if options:
            rest = {k: v for k, v in schema.items() if k not in ("anyOf", "oneOf")}
            base = sample(rest, root, depth, resource, prefer_null) if rest.get("properties") else None
            chosen = sample(options[0], root, depth, resource, prefer_null)
            if isinstance(base, dict) and isinstance(chosen, dict):
                return {**base, **chosen}
            return chosen
    if schema.get("allOf"):
        out: Any = None
        for part in schema["allOf"]:
            if "if" in part and "properties" not in part:  # a condition: a view, not a value
                continue
            value = sample(part, root, depth, resource, prefer_null)
            if isinstance(value, dict):
                out = {**(out or {}), **value}
            elif value is not None:
                out = value
        rest = {k: v for k, v in schema.items() if k != "allOf"}
        if rest.get("properties"):
            extra = sample(rest, root, depth, resource, prefer_null)
            if isinstance(extra, dict):
                out = {**(out if isinstance(out, dict) else {}), **extra}
        return out
    kind = schema.get("type")
    if isinstance(kind, list):
        if prefer_null and "null" in kind:
            return None
        kind = next((k for k in kind if k != "null"), None)
    if kind == "string":
        if schema.get("format") in _FORMATS:
            return _FORMATS[schema["format"]]
        if "pattern" in schema:
            import re

            found = next((c for c in _PATTERN_CANDIDATES if re.search(schema["pattern"], c)), None)
            return found if found is not None else "s"  # the validation then names the pattern
        return "s" * max(1, int(schema.get("minLength", 1)))
    if kind == "integer":
        return int(max(schema.get("minimum", 1), schema.get("exclusiveMinimum", 0) + 1))
    if kind == "number":
        low = schema.get("minimum", schema.get("exclusiveMinimum", 0))
        high = schema.get("maximum", schema.get("exclusiveMaximum", low + 2))
        return (low + high) / 2
    if kind == "boolean":
        return True
    if kind == "array" and schema.get("prefixItems"):  # a tuple
        return [sample(sub, root, depth + 1, resource, prefer_null) for sub in schema["prefixItems"]]
    if kind == "array":
        item = sample(schema.get("items", {}), root, depth + 1, resource, prefer_null)
        count = max(1, int(schema.get("minItems", 1)))
        return [] if item is None else [item] * count
    if kind == "object" or "properties" in schema:
        out = {}
        for key, sub in (schema.get("properties") or {}).items():
            if prefer_null and key not in (schema.get("required") or []):
                continue  # the neutral completion fills only what the route always sends
            value = sample(sub, root, depth + 1, resource, prefer_null)
            if value is not None or key in (schema.get("required") or []):
                out[key] = value
        if not out and int(schema.get("minProperties", 0)) > 0:
            extra = schema.get("additionalProperties")
            for pattern, sub in (schema.get("patternProperties") or {}).items():
                import re

                key = next((c for c in _PATTERN_CANDIDATES if re.search(pattern, c)), "k")
                out[key] = sample(sub, root, depth + 1, resource, prefer_null)
                break
            else:
                if isinstance(extra, dict):
                    out["k"] = sample(extra, root, depth + 1, resource, prefer_null)
        return out
    return None


def _resolve(schema: Any, root: dict[str, Any], resource: Optional[dict[str, Any]]) -> tuple[Any, Any]:
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str) and schema["$ref"].startswith("#/$defs/"):
        name = schema["$ref"].split("/")[-1]
        target = ((resource or {}).get("$defs") or {}).get(name) or (root.get("$defs") or {})[name]
        if "$id" in target:
            resource = target
        schema = {**target, **{k: v for k, v in schema.items() if k != "$ref"}}
    return schema, resource


def complete(schema: Any, body: Any, root: dict[str, Any], resource: Optional[dict[str, Any]] = None,
             depth: int = 0) -> Any:
    """`body` with every REQUIRED field its schema declares and it lacks filled in (neutrally: `sample(...,
    prefer_null=True)`), at every depth; `body`'s own values always win, so its mistakes still reach a check."""
    schema, resource = _resolve(schema, root, resource)
    if not isinstance(schema, dict) or depth > MAX_DEPTH:
        return body
    if isinstance(body, dict):
        options = [schema]
        for key in ("anyOf", "oneOf"):
            options = [o for o in schema.get(key) or [] if isinstance(o, dict)] or options
        every = [_resolve(o, root, resource) for o in options]
        resolved = [(o, r) for o, r in every if isinstance(o, dict) and isinstance(o.get("properties"), dict)]
        untyped = [o for o, _r in every if isinstance(o, dict) and not isinstance(o.get("properties"), dict)
                   and o.get("type") in (None, "object")]
        if untyped and len(options) > 1:
            # a union with an untyped member (046: a gated route's own answer not typed yet, beside the pending one):
            # a typed variant is the body's only when the body already carries most of what it requires
            resolved = [(o, r) for o, r in resolved
                        if len(set(body) & set(o.get("required") or [])) * 2 > len(o.get("required") or [])]
            if not resolved:
                return body
        if resolved:
            # the union variant the body is: the one sharing most of its keys
            option, res = max(resolved, key=lambda pair: len(set(body) & set(pair[0]["properties"])))
            props = option["properties"]
            out = dict(body)
            for key in option.get("required") or []:
                if key not in out and key in props:
                    out[key] = sample(props[key], root, depth + 1, res, prefer_null=True)
            for key, value in body.items():
                if key in props:
                    out[key] = complete(props[key], value, root, res, depth + 1)
            return out
        if schema.get("allOf"):
            out = dict(body)
            for part in schema["allOf"]:
                if "if" in part and "properties" not in part:
                    continue
                out = complete(part, out, root, resource, depth)
            return out
        return body
    if isinstance(body, list):
        items = schema.get("items")
        if items is None:
            for key in ("anyOf", "oneOf"):
                arrays = [o for o in schema.get(key) or [] if isinstance(o, dict) and _resolve(o, root, resource)[0].get("type") == "array"]
                if arrays:
                    items = _resolve(arrays[0], root, resource)[0].get("items")
        return [complete(items, item, root, resource, depth + 1) for item in body] if items else body
    return body
