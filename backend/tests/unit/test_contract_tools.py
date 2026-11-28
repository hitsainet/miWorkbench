"""The answer contract's pure transforms, on small synthetic schemas (046 FTASKS 6.1, FTID §3.2).

Ported from miServer's `tests/unit/api/test_contract_tools.py` (044) with miWorkbench's differences: `is_typed` has no
envelope case and refuses `Any` at a field (FR-046.4); the classifier knows `not_covered` (FR-046.13, in
`test_answers_contract_classifier.py`); and the module imports the standard library only (IC-2), checked here in a
process that cannot import anything else.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from src.api import contract_tools as tools

MODULE = Path(tools.__file__)


def test_always_sent_required_lists_every_unmarked_property_and_drops_the_marker():
    defs = {
        "A": {"type": "object", "properties": {"b": {"type": "integer"}, "a": {"type": "string", tools.MAY_BE_ABSENT: True},
                                               "c": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
              "required": ["b"]},
        "Empty": {"type": "object", "properties": {"x": {tools.MAY_BE_ABSENT: True}}, "required": ["x"]},
        "Map": {"type": "object", "additionalProperties": {"type": "integer"}},
    }
    tools.always_sent_required(defs)
    assert defs["A"]["required"] == ["b", "c"]
    assert tools.MAY_BE_ABSENT not in defs["A"]["properties"]["a"]
    assert "required" not in defs["Empty"]
    assert "required" not in defs["Map"]


def test_open_objects_removes_closed_objects_and_keeps_maps():
    node = {"a": {"type": "object", "properties": {}, "additionalProperties": False},
            "m": {"type": "object", "additionalProperties": {"type": "string"}},
            "t": {"type": "object", "additionalProperties": True}}
    tools.open_objects(node)
    assert "additionalProperties" not in node["a"]
    assert node["m"]["additionalProperties"] == {"type": "string"}
    assert node["t"]["additionalProperties"] is True


def test_strict_form_closes_objects_on_a_copy_and_leaves_maps_and_conditions():
    node = {"type": "object", "properties": {"x": {"type": "object", "properties": {"y": {}}}},
            "allOf": [{"if": {"properties": {"k": {"const": 1}}}, "then": {"properties": {"x": {}}}}],
            "$defs": {"M": {"type": "object", "additionalProperties": {"type": "integer"}}}}
    strict = tools.strict_form(node)
    assert strict["additionalProperties"] is False
    assert strict["properties"]["x"]["additionalProperties"] is False
    assert strict["$defs"]["M"]["additionalProperties"] == {"type": "integer"}
    assert "additionalProperties" not in strict["allOf"][0]["if"]
    assert "additionalProperties" not in strict["allOf"][0]["then"]
    assert "additionalProperties" not in node


def test_reachable_defs_follows_references_and_drops_the_rest():
    defs = {"A": {"$ref": "#/$defs/B"}, "B": {"properties": {"c": {"$ref": "#/$defs/C"}}}, "C": {},
            "Lost": {"$ref": "#/$defs/A"}}
    assert list(tools.reachable_defs([{"x": {"$ref": "#/$defs/A"}}], defs)) == ["A", "B", "C"]


def test_history_form_copies_definitions_without_required():
    defs = {"M": {"type": "object", "properties": {"n": {"$ref": "#/$defs/N"}}, "required": ["n"]},
            "N": {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}}
    out = tools.history_form({"$ref": "#/$defs/M"}, defs)
    assert out == {"$ref": "#/$defs/M__history"}
    assert "required" not in defs["M__history"] and "required" not in defs["N__history"]
    assert defs["M__history"]["properties"]["n"] == {"$ref": "#/$defs/N__history"}
    assert defs["M"]["required"] == ["n"] and defs["N"]["required"] == ["v"]
    assert defs["N__history"]["properties"]["v"] == {"type": "integer"}


def test_apply_history_forms_replaces_only_the_marked_field():
    defs = {"Run": {"type": "object", "required": ["id", "result"],
                    "properties": {"id": {"type": "string"},
                                   "result": {"$ref": "#/$defs/Out", tools.HISTORY_FORM: True}}},
            "Out": {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}}
    assert tools.apply_history_forms(defs) == ["Run.result"]
    assert defs["Run"]["properties"]["result"] == {"$ref": "#/$defs/Out__history"}
    assert defs["Run"]["required"] == ["id", "result"]  # the field itself is still always sent
    assert defs["Out"]["required"] == ["v"] and "required" not in defs["Out__history"]


def test_is_typed_follows_fr_046_4():
    defs = {
        "Obj": {"type": "object", "properties": {"a": {"type": "string"}}},
        "WithMap": {"type": "object", "properties": {"config": {"type": "object", "additionalProperties": True}}},
        "WithAny": {"type": "object", "properties": {"a": {"type": "string"}, "blob": {"title": "Blob"}}},
        "DeepAny": {"type": "object", "properties": {"inner": {"$ref": "#/$defs/WithAny"}}},
        "AnyList": {"type": "object", "properties": {"rows": {"type": "array", "items": {}}}},
        "Map": {"type": "object", "additionalProperties": True},
        "Loop": {"type": "object", "properties": {"next": {"anyOf": [{"$ref": "#/$defs/Loop"}, {"type": "null"}]}}},
    }
    ext = {"https://hitsai.net/schemas/x.json"}
    assert tools.is_typed({"$ref": "#/$defs/Obj"}, defs)
    assert tools.is_typed({"$ref": "#/$defs/WithMap"}, defs)  # a declared map at a field
    assert tools.is_typed({"$ref": "#/$defs/Loop"}, defs)  # a recursive model terminates
    assert tools.is_typed({"type": "array", "items": {"$ref": "#/$defs/Obj"}}, defs)
    assert tools.is_typed({"anyOf": [{"$ref": "#/$defs/Obj"}, {"type": "null"}]}, defs)
    assert tools.is_typed({"$ref": "https://hitsai.net/schemas/x.json"}, defs, ext)
    assert not tools.is_typed({"$ref": "https://hitsai.net/schemas/y.json"}, defs, ext)
    assert not tools.is_typed({}, defs)
    assert not tools.is_typed({"$ref": "#/$defs/Map"}, defs)  # a top-level map
    assert not tools.is_typed({"$ref": "#/$defs/WithAny"}, defs)  # `Any` at a field
    assert not tools.is_typed({"$ref": "#/$defs/DeepAny"}, defs)  # ... at any depth
    assert not tools.is_typed({"$ref": "#/$defs/AnyList"}, defs)  # a list of `Any` at a field
    assert not tools.is_typed({"type": "array", "items": {}}, defs)
    assert not tools.is_typed({"anyOf": [{"$ref": "#/$defs/Obj"}, {}]}, defs)


def test_content_digest_ignores_revision_and_its_own_field():
    doc = {"a": 1, "revision": 3, "content_sha256": "x"}
    assert tools.content_digest(doc) == tools.content_digest({"a": 1})
    assert tools.content_digest(doc) != tools.content_digest({"a": 2})


def test_render_is_sorted_two_space_unicode_with_a_newline():
    assert tools.render({"b": "é", "a": [1]}) == '{\n  "a": [\n    1\n  ],\n  "b": "é"\n}\n'


def test_strict_form_closes_the_members_of_unions_and_tuples_but_not_of_allof():
    node = {"anyOf": [{"type": "object", "properties": {"a": {}}}], "oneOf": [{"type": "object", "properties": {"b": {}}}],
            "allOf": [{"type": "object", "properties": {"c": {"type": "object", "properties": {"d": {}}}}}],
            "prefixItems": [{"type": "object", "properties": {"e": {}}}]}
    strict = tools.strict_form(node)
    for key in ("anyOf", "oneOf", "prefixItems"):
        assert strict[key][0]["additionalProperties"] is False, key
    assert "additionalProperties" not in strict["allOf"][0]
    assert strict["allOf"][0]["properties"]["c"]["additionalProperties"] is False


def test_the_module_imports_the_standard_library_only():
    """IC-2: every import is a standard-library module, and the module imports in an isolated interpreter (`-I`, no
    site-packages) with nothing but its own file on the path."""
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    imported = set()
    for node in ast.walk(ast.parse(MODULE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= stdlib, sorted(imported - stdlib)
    code = (f"import importlib.util as u, sys; s = u.spec_from_file_location('t', {str(MODULE)!r}); "
            "m = u.module_from_spec(s); sys.modules['t'] = m; s.loader.exec_module(m); print(m.content_digest({'a': 1}))")
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert len(proc.stdout.strip()) == 64
