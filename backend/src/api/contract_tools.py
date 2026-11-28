"""Pure transforms of answer-contract documents (046 FTID §3.2; miServer TD12's shape). STANDARD LIBRARY ONLY (IC-2):
the contract writer (`src/api/answers_contract.py`), the generator, the test-only answer checks (the middleware and the
fake check) and the cross-repo consumer test all use them, and that last one must run where only the standard
library and pytest are installed.

Ported from miServer's `miserver/api/schemas/contract_tools.py` (044, origin/main `4a09a469`) so both apps judge a
contract the same way. Differences, each from 046's FPRD:

- `is_typed` (FR-046.4) has no envelope case (miWorkbench answers carry none), and `Any` at a field is untyped: a
  typed object may hold declared maps (`dict[str, X]`, values free), never a bare `Any` field;
- the classifier knows the third section, `not_covered` (FR-046.13): `not_covered` -> `responses` (or `excluded`)
  is additive, a route newly called by a tool; every move OUT of `responses` is breaking; anything unlisted is
  breaking (fails closed).
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

OPEN_KEY = "additionalProperties"
#: TD4: a field a route sometimes leaves out (`answer_fields.may_be_absent()`); never `required` in the contract.
MAY_BE_ABSENT = "x-may-be-absent"
#: WT9: a field holding JSON an earlier release wrote; beneath it the contract keeps types and drops every `required`.
HISTORY_FORM = "x-history-form"
DEFS_REF = "#/$defs/"
HISTORY_SUFFIX = "__history"
#: The two top-level keys a digest leaves out (TD11): the writer sets both.
UNDIGESTED = ("revision", "content_sha256")
#: Keys that only annotate a schema: a schema made of these alone accepts anything (Pydantic's `Any`).
_ANNOTATIONS = frozenset({"title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly",
                          "$comment"})


def render(doc: dict[str, Any]) -> str:
    """The committed byte form (FR-046.1): sorted keys, two-space indent, `ensure_ascii=False`, a trailing newline."""
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    """Every dict below `node`, depth first (the node itself included)."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def always_sent_required(node: Any) -> None:
    """SH7 / TD4, in place: every object schema with `properties` requires every property not marked
    `x-may-be-absent`, and the marker is removed. An object whose list comes out empty carries no `required`."""
    for schema in list(_walk(node)):
        props = schema.get("properties")
        if not isinstance(props, dict):
            continue
        required = []
        for name, prop in props.items():
            if isinstance(prop, dict) and prop.pop(MAY_BE_ABSENT, False):
                continue
            required.append(name)
        if required:
            schema["required"] = sorted(required)
        else:
            schema.pop("required", None)


def open_objects(node: Any) -> Any:
    """SH7, in place: delete every `additionalProperties: false`. A schema-valued `additionalProperties` (a map) stays."""
    for schema in _walk(node):
        if schema.get(OPEN_KEY) is False:
            del schema[OPEN_KEY]
    return node


#: The `format` values a contract never declares (046 P7 finding, 2026-11-26; the live regression fixed on 046-1 by
#: `fix/046-1-datetime-format`). Both need an RFC 3339 offset, and the routes serialise naive datetimes without one
#: (`2026-11-26T10:15:00.123456`, a `DateTime` column through a Pydantic model): a consumer that checks formats would
#: refuse every such answer. The value stays a string.
OFFSET_FORMATS = frozenset({"date-time", "time"})


def drop_offset_formats(node: Any) -> int:
    """In place: delete every `format` keyword in `OFFSET_FORMATS` (a property NAMED `format` is a schema, not a
    string, so it stays). Returns how many."""
    count = 0
    for schema in _walk(node):
        if isinstance(schema.get("format"), str) and schema["format"] in OFFSET_FORMATS:
            del schema["format"]
            count += 1
    return count


def strict_form(node: Any) -> Any:
    """TD5 / SH13, on a COPY: every object with `properties` refuses keys it does not declare (`additionalProperties:
    false`), which is how a check catches a fake, an answer or a model that carries an undeclared key (MW3).

    Exactly miServer's rule (`miserver/api/schemas/contract_tools.strict_form`), so both apps judge a contract the same
    way: maps (only `additionalProperties`) are left alone; a schema that DECLARES its extras (`additionalProperties`
    already present) keeps them; a condition (`if`/`then`/`else`) and an `allOf` member are partial views of one
    object, so they are not closed themselves (their own nested objects are)."""
    out = copy.deepcopy(node)

    def close(value: Any, conditional: bool) -> None:
        if isinstance(value, dict):
            if not conditional and isinstance(value.get("properties"), dict) and OPEN_KEY not in value:
                value[OPEN_KEY] = False
            for key, sub in value.items():
                if key == "properties" and isinstance(sub, dict):
                    for prop in sub.values():
                        close(prop, False)
                elif key == "allOf" and isinstance(sub, list):
                    for member in sub:
                        close(member, True)
                else:
                    close(sub, key in ("if", "then", "else"))
        elif isinstance(value, list):
            for item in value:
                close(item, False)

    close(out, False)
    return out


def _refs(node: Any) -> list[str]:
    return [s["$ref"][len(DEFS_REF):] for s in _walk(node)
            if isinstance(s.get("$ref"), str) and s["$ref"].startswith(DEFS_REF)]


def reachable_defs(roots: list[Any], defs: dict[str, Any]) -> dict[str, Any]:
    """FR-046.15: only the definitions a `#/$defs/<name>` reference reaches from `roots`, sorted by name."""
    seen: dict[str, Any] = {}
    pending = sorted({name for root in roots for name in _refs(root)})
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen[name] = defs[name]
        pending.extend(sorted(set(_refs(defs[name])) - set(seen)))
    return dict(sorted(seen.items()))


def history_form(node: Any, defs: dict[str, Any]) -> Any:
    """WT9: a copy of `node` with no `required` at any depth, whose every reference points at a `<Name>__history` copy
    of its definition (made in `defs`, likewise without `required`). The live definitions keep theirs. `__history` is
    not a suffix Pydantic produces, so a copy's name cannot collide with a component's."""
    out = copy.deepcopy(node)

    def strip(schema: Any) -> None:
        for s in _walk(schema):
            s.pop("required", None)
            ref = s.get("$ref")
            if isinstance(ref, str) and ref.startswith(DEFS_REF):
                name = ref[len(DEFS_REF):]
                if not name.endswith(HISTORY_SUFFIX):
                    target = name + HISTORY_SUFFIX
                    s["$ref"] = DEFS_REF + target
                    if target not in defs:
                        defs[target] = copy.deepcopy(defs[name])
                        strip(defs[target])

    strip(out)
    return out


def apply_history_forms(defs: dict[str, Any]) -> list[str]:
    """In place: every property marked `x-history-form` is replaced by its `history_form`, and the marker removed.
    Returns the marked properties as `<Definition>.<property>`, sorted (the writer states them in `description`)."""
    marked = []
    for name in sorted(defs):
        props = defs[name].get("properties") if isinstance(defs[name], dict) else None
        if not isinstance(props, dict):
            continue
        for prop, schema in sorted(props.items()):
            if isinstance(schema, dict) and schema.pop(HISTORY_FORM, False):
                props[prop] = history_form(schema, defs)
                marked.append(f"{name}.{prop}")
    return marked


def _deref(schema: Any, defs: dict[str, Any]) -> Any:
    seen = 0
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str) and schema["$ref"].startswith(DEFS_REF):
        schema = defs.get(schema["$ref"][len(DEFS_REF):], {})
        seen += 1
        if seen > 50:
            return {}
    return schema


def is_any(schema: Any) -> bool:
    """Pydantic's `Any`: a schema that constrains nothing (no keys, or annotations only)."""
    return isinstance(schema, dict) and not (set(schema) - _ANNOTATIONS)


def _field_typed(schema: Any, defs: dict[str, Any], external: frozenset[str] | set[str], seen: set[str],
                 depth: int) -> bool:
    """A FIELD's schema (FR-046.4): anything but a bare `Any`; inside it, an object's fields and a list's items are
    held to the same rule. A map's values (`additionalProperties`) are free: `dict[str, Any]` is a declared map."""
    if depth > 40:
        return True
    if is_any(schema):
        return False
    if not isinstance(schema, dict):
        return True
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if not ref.startswith(DEFS_REF):
            return ref in external
        name = ref[len(DEFS_REF):]
        if name in seen:
            return True
        return _field_typed(defs.get(name, {}), defs, external, seen | {name}, depth + 1)
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            members = [m for m in schema[key] if _deref(m, defs).get("type") != "null"]
            if not all(_field_typed(m, defs, external, seen, depth + 1) for m in members):
                return False
    props = schema.get("properties")
    if isinstance(props, dict):
        if not all(_field_typed(p, defs, external, seen, depth + 1) for p in props.values()):
            return False
    if schema.get("type") == "array" or "items" in schema:
        items = schema.get("items")
        if items is None or not _field_typed(items, defs, external, seen, depth + 1):
            return False
    return True


def is_typed(schema: Any, defs: dict[str, Any], external: frozenset[str] | set[str] = frozenset(),
             _depth: int = 0) -> bool:
    """FR-046.4. Typed: an object with named properties (none of them a bare `Any`, at any depth); an array of typed
    items; an `anyOf`/`oneOf` whose non-null members are all typed; an external `$ref` listed in `external`. Not
    typed: `{}`, an object without properties, a top-level map, an array of untyped items, anything else."""
    if _depth > 40 or not isinstance(schema, dict) or not schema:
        return False
    ref = schema.get("$ref")
    if isinstance(ref, str) and not ref.startswith(DEFS_REF):
        return ref in external
    resolved = _deref(schema, defs)
    if not isinstance(resolved, dict) or not resolved:
        return False
    for key in ("anyOf", "oneOf"):
        if key in resolved:
            members = [m for m in resolved[key] if _deref(m, defs).get("type") != "null"]
            return bool(members) and all(is_typed(m, defs, external, _depth + 1) for m in members)
    props = resolved.get("properties")
    if isinstance(props, dict) and props:
        return _field_typed(schema, defs, external, set(), 0)
    if resolved.get("type") == "array":
        return is_typed(resolved.get("items") or {}, defs, external, _depth + 1)
    return False


def content_digest(doc: dict[str, Any]) -> str:
    """TD11 / FR-046.14: sha256 of the rendered document with `revision` and `content_sha256` left out."""
    body = {k: v for k, v in doc.items() if k not in UNDIGESTED}
    return hashlib.sha256(render(body).encode("utf-8")).hexdigest()


# ── the classifier (FR-046.13: SH12's table, miServer FR-044.16 row for row, plus the section moves) ─────────────

#: The table, by row name. Every row not marked additive is breaking, and so is anything no row names.
ADDITIVE_ROWS = frozenset({
    "new route",
    "new status code",
    "new optional field",
    "new always-sent field",
    "descriptive",
    "vendored schema revised (additive)",
    "newly covered",  # FR-046.13: `not_covered` -> `responses` or `excluded`
})
UNLISTED = "unlisted change"
_DESCRIPTIVE = ("description", "title", "$comment")
_RETYPING = ("type", "format")
#: Keys derived from others and not compared on their own: `coverage` (from the sections), and the two the writer sets.
_DERIVED_TOP = ("coverage", "revision", "content_sha256")
SECTIONS = ("responses", "excluded", "not_covered")


@dataclass(frozen=True)
class Change:
    path: str
    row: str

    @property
    def breaking(self) -> bool:
        return self.row not in ADDITIVE_ROWS

    def __str__(self) -> str:
        return f"{self.row}: {self.path}"


#: `resolve(sid, sha256)` -> the vendored document with that `$id` whose bytes have that sha256, or None.
Resolver = Callable[[str, str], "dict[str, Any] | None"]


class _Classifier:
    def __init__(self, old: dict[str, Any], new: dict[str, Any], resolve: Resolver | None = None) -> None:
        self.old_defs: dict[str, Any] = old.get("$defs", {}) or {}
        self.new_defs: dict[str, Any] = new.get("$defs", {}) or {}
        self.changes: list[Change] = []
        self.compared: set[tuple[str, str]] = set()
        self.resolve = resolve

    def external(self, a: dict[str, Any], b: dict[str, Any]) -> None:
        """An entry removed is breaking, a new one additive (its `$ref` site is classified where it appears), a moved
        file unlisted. The SAME `$id` and file with a new sha256 (a re-vendor) is classified by running this
        classifier on the two documents: additive passes as one `vendored schema revised (additive)` change, anything
        else is breaking. A copy that cannot be read fails closed."""
        for sid in sorted(set(a) | set(b)):
            path = f"external.{sid}"
            if sid not in b:
                self.add(path, "removed")
            elif sid not in a:
                self.add(path, "descriptive")
            elif a[sid] != b[sid]:
                if a[sid].get("file") != b[sid].get("file"):
                    self.add(f"{path}.file", UNLISTED)
                    continue
                old_doc = self.resolve(sid, a[sid]["sha256"]) if self.resolve else None
                new_doc = self.resolve(sid, b[sid]["sha256"]) if self.resolve else None
                if old_doc is None or new_doc is None:
                    self.add(f"{path}.sha256", UNLISTED)
                    continue
                inner = _Classifier(old_doc, new_doc)
                inner.schema(_without_defs(old_doc), _without_defs(new_doc), path)
                inner.sweep()
                bad = [c for c in inner.changes if c.breaking]
                if bad:
                    self.changes.extend(bad)
                else:
                    self.add(f"{path}.sha256", "vendored schema revised (additive)")

    def sweep(self) -> None:
        """Compare every definition both sides hold that nothing compared yet (miServer 044 R1-4)."""
        for name in sorted(set(self.old_defs) & set(self.new_defs)):
            if (name, name) not in self.compared:
                self.compared.add((name, name))
                self.schema(self.old_defs[name], self.new_defs[name], f"$defs.{name}")

    def add(self, path: str, row: str) -> None:
        self.changes.append(Change(path, row))

    def schema(self, old: Any, new: Any, path: str) -> None:
        if old == new:
            self._follow(new)
            return
        if not isinstance(old, dict) or not isinstance(new, dict):
            self.add(path, UNLISTED)
            return
        old_ref, new_ref = old.get("$ref"), new.get("$ref")
        if old_ref != new_ref:
            self.add(f"{path}.$ref", "retyped")
        elif isinstance(old_ref, str) and old_ref.startswith(DEFS_REF):
            name = old_ref[len(DEFS_REF):]
            if (name, name) not in self.compared:
                self.compared.add((name, name))
                if name not in self.new_defs:
                    self.add(f"$defs.{name}", "removed")
                elif name in self.old_defs:
                    self.schema(self.old_defs[name], self.new_defs[name], f"$defs.{name}")
        for key in sorted(set(old) | set(new)):
            if key == "$ref":
                continue
            a, b = old.get(key), new.get(key)
            sub = f"{path}.{key}"
            if a == b:
                self._follow(b)
                continue
            if key in _DESCRIPTIVE:
                self.add(sub, "descriptive")
            elif key in _RETYPING:
                self.add(sub, "retyped")
            elif key == "properties":
                self.properties(a or {}, b or {}, new, sub)
            elif key == "required":
                self.required(set(a or ()), set(b or ()), old.get("properties") or {}, new.get("properties") or {},
                              path)
            elif key == "enum":
                self.add(sub, "enum widened" if set(map(_key, b or ())) - set(map(_key, a or ())) else UNLISTED)
            elif key == "const":
                self.add(sub, "enum widened")
            elif (key == "items" or (key == OPEN_KEY and isinstance(a, dict) and isinstance(b, dict))
                  or (key in ("if", "then", "else") and isinstance(a, dict) and isinstance(b, dict))):
                self.schema(a, b, sub)
            elif key in ("anyOf", "oneOf", "allOf") and isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                for i, (x, y) in enumerate(zip(a, b, strict=True)):
                    self.schema(x, y, f"{sub}[{i}]")
            else:
                self.add(sub, UNLISTED)

    def _follow(self, node: Any) -> None:
        """Compare, once each, the definitions `node` references: equal text can reference changed content."""
        for name in sorted(set(_refs(node))):
            if (name, name) in self.compared:
                continue
            self.compared.add((name, name))
            if name not in self.new_defs:
                self.add(f"$defs.{name}", "removed")
            elif name in self.old_defs:
                self.schema(self.old_defs[name], self.new_defs[name], f"$defs.{name}")

    def properties(self, a: dict[str, Any], b: dict[str, Any], new: dict[str, Any], path: str) -> None:
        new_required = set(new.get("required") or ())
        for name in sorted(set(a) | set(b)):
            sub = f"{path}.{name}"
            if name not in b:
                self.add(sub, "removed")
            elif name not in a:
                self.add(sub, "new always-sent field" if name in new_required else "new optional field")
                self._follow(b[name])
            else:
                self.schema(a[name], b[name], sub)

    def required(self, a: set[str], b: set[str], old_props: dict[str, Any], new_props: dict[str, Any],
                 path: str) -> None:
        for name in sorted(b - a):
            if name in old_props:
                self.add(f"{path}.required.{name}", "made required")
        for name in sorted(a - b):
            if name in new_props:
                self.add(f"{path}.required.{name}", "sometimes absent")

    def routes(self, old: dict[str, Any], new: dict[str, Any]) -> None:
        old_s = {s: old.get(s, {}) or {} for s in SECTIONS}
        new_s = {s: new.get(s, {}) or {} for s in SECTIONS}
        keys = sorted(set().union(*old_s.values(), *new_s.values()))
        for key in keys:
            was = next((s for s in SECTIONS if key in old_s[s]), None)
            now = next((s for s in SECTIONS if key in new_s[s]), None)
            if was is None:
                self.add(f"{now}.{key}", "new route")
            elif now is None:
                self.add(f"{was}.{key}", "removed")
            elif was != now:
                if was == "not_covered":
                    self.add(f"{now}.{key}", "newly covered")
                    if now == "responses":
                        self._follow(new_s[now][key])
                elif was == "responses":
                    self.add(f"{now}.{key}", f"moved out of responses (to {now})")
                else:
                    self.add(f"{now}.{key}", f"moved between {was} and {now}")
            elif now == "responses":
                self.statuses(old_s[now][key], new_s[now][key], f"responses.{key}")
            elif old_s[now][key] != new_s[now][key]:
                a, b = old_s[now][key], new_s[now][key]
                only_reason = {k: v for k, v in a.items() if k != "reason"} == {k: v for k, v in b.items()
                                                                                  if k != "reason"}
                self.add(f"{now}.{key}", "descriptive" if only_reason else UNLISTED)

    def statuses(self, a: dict[str, Any], b: dict[str, Any], path: str) -> None:
        for status in sorted(set(a) | set(b)):
            sub = f"{path}.{status}"
            if not status[:1].isdigit():
                if a.get(status) != b.get(status):
                    self.add(sub, UNLISTED)
                continue
            if status not in b:
                self.add(sub, "removed")
            elif status not in a:
                self.add(sub, "new status code")
                self._follow(b[status])
            else:
                self.schema(a[status], b[status], sub)

    def top(self, old: dict[str, Any], new: dict[str, Any]) -> None:
        self.routes(old, new)
        for key in sorted(set(old) | set(new)):
            if key in (*SECTIONS, "$defs") or key in _DERIVED_TOP:
                continue
            a, b = old.get(key), new.get(key)
            if a == b:
                if key == "errors":
                    self._follow(b)
                continue
            if key in _DESCRIPTIVE:
                self.add(key, "descriptive")
            elif key == "external" and isinstance(a, dict) and isinstance(b, dict):
                self.external(a, b)
            elif key == "errors" and isinstance(a, dict) and isinstance(b, dict):
                for name in sorted(set(a) | set(b)):
                    if name not in b:
                        self.add(f"errors.{name}", "removed")
                    elif name not in a:
                        self.add(f"errors.{name}", UNLISTED)
                    else:
                        self.schema(a[name], b[name], f"errors.{name}")
            else:
                self.add(key, UNLISTED)
        self.sweep()
        for name in sorted(set(self.new_defs) - set(self.old_defs)):
            self.add(f"$defs.{name}", "descriptive")
        for name in sorted(set(self.old_defs) - set(self.new_defs)):
            if not any(c.path == f"$defs.{name}" for c in self.changes):
                self.add(f"$defs.{name}", "removed")


def _without_defs(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k != "$defs"}


def git_resolver(root: Any, files: dict[str, str]) -> Resolver:
    """A `Resolver` over a git checkout: `files` maps each `$id` to its path under `root`; the bytes are looked for in
    the working tree, then in every commit that touched the file, by sha256 (the `git` command only)."""
    import subprocess
    from pathlib import Path

    def resolve(sid: str, sha: str) -> "dict[str, Any] | None":
        rel = files.get(sid)
        if rel is None:
            return None
        here = Path(root) / rel
        if here.is_file() and hashlib.sha256(here.read_bytes()).hexdigest() == sha:
            found: dict[str, Any] = json.loads(here.read_bytes())
            return found
        try:
            revs = subprocess.run(["git", "rev-list", "--all", "--", rel], cwd=root, capture_output=True, text=True,
                                  check=True, timeout=60).stdout.split()
        except (OSError, subprocess.SubprocessError):
            revs = []
        for rev in revs:
            shown = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=root, capture_output=True, timeout=60)
            if shown.returncode == 0 and hashlib.sha256(shown.stdout).hexdigest() == sha:
                shown_doc: dict[str, Any] = json.loads(shown.stdout)
                return shown_doc
        return None

    return resolve


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def classify(old: dict[str, Any], new: dict[str, Any], resolve: Resolver | None = None) -> list[Change]:
    """Every difference between two contract documents, each mapped onto a row of FR-046.13's table. A difference no
    row names is `unlisted change`, which is breaking (fails closed). A rename is a removal plus an addition."""
    run = _Classifier(old, new, resolve)
    run.top(old, new)
    return run.changes


def breaking(changes: list[Change]) -> list[Change]:
    return [c for c in changes if c.breaking]
