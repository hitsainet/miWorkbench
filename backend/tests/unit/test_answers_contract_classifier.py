"""The writer classifies every difference and refuses a breaking one (046 FR-046.13, SH3, SH12, TD11; FTASKS 6.5).

Ported from miServer's `tests/unit/api/test_answers_contract_classifier.py` (044, `4a09a469`): one synthetic
before-and-after pair per row of the table (miServer FR-044.16, row for row), plus an unlisted change, which must come
out breaking (the classifier fails closed). miWorkbench adds the third section, `not_covered`: `not_covered` ->
`responses` or `excluded` is additive (`newly covered`), any move out of `responses` is breaking, and every other move
is unlisted, so breaking. Then the writer on a planted file: a removal is refused and leaves the committed bytes as
they were; an always-sent new field raises `revision` by exactly 1.

Controls: M15, twice (a removal called additive; an unlisted change called additive).
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from src.api import contract_tools as tools
from src.api.answers_contract import BreakingAnswerChange, write_contract

ENVELOPE = {"$ref": "#/$defs/Env"}


def _base() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://hitsai.net/schemas/miworkbench-answers-v1.json",
        "kind": "miworkbench.answer-contract/v1",
        "title": "t",
        "description": "d",
        "responses": {"GET /a": {"200": copy.deepcopy(ENVELOPE)}},
        "excluded": {
            "GET /m": {
                "media_type": "text/plain",
                "reason": "text",
                "x-schema": {"$ref": "#/$defs/Fixed"},
            }
        },
        "not_covered": {"GET /n": {"reason": "no tool or sibling calls it"}},
        "errors": {"http": {"$ref": "#/$defs/Err"}},
        "external": {},
        "coverage": {"typed": 1, "unions": 0, "excluded": 1, "not_covered": 1, "untyped": 0},
        "$defs": {
            "Env": {
                "type": "object",
                "properties": {"data": {"$ref": "#/$defs/Item"}},
                "required": ["data"],
            },
            "Item": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "s": {"type": "string", "format": "date-time"},
                    "k": {"type": "string", "enum": ["a", "b"]},
                    "c": {"const": "x"},
                    "xs": {"type": "array", "items": {"type": "integer"}},
                    "opt": {"type": "string"},
                    "u": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "m": {"type": "object", "additionalProperties": {"type": "integer"}},
                },
                "allOf": [
                    {
                        "if": {"properties": {"c": {"const": "x"}}},
                        "then": {"properties": {"n": {"type": "integer"}}},
                    }
                ],
                "required": ["c", "k", "m", "n", "s", "u", "xs"],
            },
            "Other": {
                "type": "object",
                "properties": {"z": {"type": "integer"}},
                "required": ["z"],
            },
            "Err": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            # Reached only through an `excluded` entry's `x-` key (a path the walk does not follow).
            "Fixed": {
                "type": "object",
                "properties": {"z": {"type": "integer"}},
                "required": ["z"],
            },
        },
    }


def _item(doc: dict) -> dict:
    return doc["$defs"]["Item"]


def _rows(old: dict, new: dict) -> set[tuple[str, bool]]:
    return {(c.row, c.breaking) for c in tools.classify(old, new)}


# Each case: (row, edit of the new document, breaking?)
CASES = [
    ("new route", lambda d: d["responses"].__setitem__("GET /b", copy.deepcopy(ENVELOPE)), False),
    (
        "new route",
        lambda d: d["excluded"].__setitem__("GET /t", {"media_type": "text/csv", "reason": "r"}),
        False,
    ),
    (
        "new status code",
        lambda d: d["responses"]["GET /a"].__setitem__("202", copy.deepcopy(ENVELOPE)),
        False,
    ),
    (
        "new optional field",
        lambda d: _item(d)["properties"].__setitem__("o2", {"type": "string"}),
        False,
    ),
    (
        "new always-sent field",
        lambda d: (
            _item(d)["properties"].__setitem__("a2", {"type": "integer"}),
            _item(d)["required"].append("a2"),
        ),
        False,
    ),
    ("descriptive", lambda d: _item(d).__setitem__("description", "now described"), False),
    ("descriptive", lambda d: d["$defs"].__setitem__("Fresh", {"type": "object"}), False),
    ("descriptive", lambda d: d.__setitem__("title", "another title"), False),
    ("removed", lambda d: d["responses"].pop("GET /a"), True),
    ("removed", lambda d: d["responses"]["GET /a"].pop("200"), True),
    ("removed", lambda d: _item(d)["properties"].pop("opt"), True),
    ("retyped", lambda d: _item(d)["properties"]["n"].__setitem__("type", "string"), True),
    (
        "retyped",
        lambda d: d["$defs"]["Env"]["properties"].__setitem__("data", {"$ref": "#/$defs/Other"}),
        True,
    ),
    ("retyped", lambda d: _item(d)["properties"]["s"].__setitem__("format", "date"), True),
    (
        "retyped",
        lambda d: _item(d)["properties"]["xs"]["items"].__setitem__("type", "string"),
        True,
    ),
    ("made required", lambda d: _item(d)["required"].append("opt"), True),
    ("enum widened", lambda d: _item(d)["properties"]["k"]["enum"].append("c"), True),
    ("enum widened", lambda d: _item(d)["properties"]["c"].__setitem__("const", "y"), True),
    ("sometimes absent", lambda d: _item(d)["required"].remove("n"), True),
    (
        "moved out of responses (to excluded)",
        lambda d: (
            d["excluded"].__setitem__("GET /a", {"media_type": "text/plain", "reason": "r"})
            or d["responses"].pop("GET /a")
        ),
        True,
    ),
    # FR-046.13's section moves.
    (
        "moved out of responses (to not_covered)",
        lambda d: d["not_covered"].__setitem__("GET /a", {"reason": "r"}) or d["responses"].pop("GET /a"),
        True,
    ),
    (
        "newly covered",
        lambda d: d["responses"].__setitem__("GET /n", copy.deepcopy(ENVELOPE)) or d["not_covered"].pop("GET /n"),
        False,
    ),
    (
        "newly covered",
        lambda d: d["excluded"].__setitem__("GET /n", {"media_type": None, "reason": "no content"})
        or d["not_covered"].pop("GET /n"),
        False,
    ),
    (
        "moved between excluded and not_covered",
        lambda d: d["not_covered"].__setitem__("GET /m", {"reason": "r"}) or d["excluded"].pop("GET /m"),
        True,
    ),
    ("new route", lambda d: d["not_covered"].__setitem__("GET /n2", {"reason": "r"}), False),
    ("removed", lambda d: d["not_covered"].pop("GET /n"), True),
    ("descriptive", lambda d: d["not_covered"]["GET /n"].__setitem__("reason", "another reason"), False),
    (tools.UNLISTED, lambda d: d["not_covered"]["GET /n"].__setitem__("x-note", 1), True),
    (tools.UNLISTED, lambda d: _item(d)["properties"]["n"].__setitem__("minimum", 0), True),
    # 044 review round 1 (R1-7): one case per path the classifier walks, each a breaking difference
    # that a dropped branch would turn into "no change" or additive.
    (tools.UNLISTED, lambda d: d["excluded"]["GET /m"].__setitem__("media_type", "text/csv"), True),
    (
        tools.UNLISTED,
        lambda d: d["responses"]["GET /a"].__setitem__("x-approval-action", "start_labeling"),
        True,
    ),
    (tools.UNLISTED, lambda d: d.__setitem__("kind", "miworkbench.answer-contract/v2"), True),
    (
        tools.UNLISTED,
        lambda d: d.__setitem__("$schema", "http://json-schema.org/draft-07/schema#"),
        True,
    ),
    (tools.UNLISTED, lambda d: d["errors"].__setitem__("other", {"$ref": "#/$defs/Err"}), True),
    ("removed", lambda d: d["errors"].pop("http"), True),
    ("removed", lambda d: d["$defs"].pop("Other"), True),
    (tools.UNLISTED, lambda d: _item(d)["properties"]["k"].__setitem__("enum", ["a"]), True),
    (tools.UNLISTED, lambda d: _item(d)["properties"].__setitem__("n", True), True),
    (
        "retyped",
        lambda d: _item(d)["properties"]["u"]["anyOf"][0].__setitem__("type", "string"),
        True,
    ),
    (
        "retyped",
        lambda d: _item(d)["properties"]["m"]["additionalProperties"].__setitem__("type", "string"),
        True,
    ),
    (
        "retyped",
        lambda d: _item(d)["allOf"][0]["then"]["properties"]["n"].__setitem__("type", "string"),
        True,
    ),
    ("removed", lambda d: d["$defs"]["Err"]["properties"].pop("code"), True),
    ("removed", lambda d: d["$defs"]["Fixed"]["properties"].pop("z"), True),
]


@pytest.mark.parametrize(
    ("row", "edit", "is_breaking"), CASES, ids=[f"{i}-{c[0]}" for i, c in enumerate(CASES)]
)
def test_each_row_of_the_table(row, edit, is_breaking):
    old = _base()
    new = _base()
    edit(new)
    changes = tools.classify(old, new)
    assert (row, is_breaking) in {(c.row, c.breaking) for c in changes}, changes
    assert bool(tools.breaking(changes)) is is_breaking, changes


def test_a_rename_is_a_removal_plus_an_addition():
    old, new = _base(), _base()
    _item(new)["properties"]["optional"] = _item(new)["properties"].pop("opt")
    rows = _rows(old, new)
    assert ("removed", True) in rows and ("new optional field", False) in rows


def test_equal_documents_have_no_change():
    assert tools.classify(_base(), _base()) == []


def test_a_change_inside_a_referenced_definition_is_seen_through_equal_text():
    """The route's own schema text is equal (`$ref` Env); the change is two references down."""
    old, new = _base(), _base()
    _item(new)["properties"].pop("n")
    assert ("removed", True) in _rows(old, new)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _body() -> dict:
    body = _base()
    return body


def test_the_writer_writes_revision_1_then_refuses_a_removal(tmp_path):
    path = tmp_path / "miworkbench-answers-v1.json"
    first = write_contract(path, _body())
    assert first["revision"] == 1
    assert first["content_sha256"] == tools.content_digest(first)
    before = _sha(path)
    smaller = _body()
    _item(smaller)["properties"].pop("opt")
    with pytest.raises(BreakingAnswerChange, match="miworkbench-answers-v2.json"):
        write_contract(path, smaller)
    assert _sha(path) == before


def test_an_always_sent_new_field_raises_the_revision_by_exactly_one(tmp_path):
    path = tmp_path / "miworkbench-answers-v1.json"
    write_contract(path, _body())
    bigger = _body()
    _item(bigger)["properties"]["a2"] = {"type": "integer"}
    _item(bigger)["required"].append("a2")
    doc = write_contract(path, bigger)
    assert doc["revision"] == 2
    assert json.loads(path.read_text())["revision"] == 2
    # Equal again: nothing is written, the revision stays.
    assert write_contract(path, bigger)["revision"] == 2


# ── the re-vendor row (miServer FR-044.16, suite decision 2026-11-25): same `$id`, new sha256 in `external` ─

SID = "https://hitsai.net/schemas/doc-v1.json"
DOC_V1 = {
    "$id": SID,
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
    "$defs": {"Inner": {"type": "object", "properties": {"x": {"type": "integer"}}}},
}


def _doc_sha(doc: dict) -> str:
    return hashlib.sha256(tools.render(doc).encode()).hexdigest()


def _with_external(doc: dict) -> dict:
    contract = _base()
    contract["external"] = {SID: {"file": "docs/schemas/doc-v1.json", "sha256": _doc_sha(doc)}}
    return contract


def _resolver(*docs: dict):
    known = {_doc_sha(d): d for d in docs}
    return lambda sid, sha: known.get(sha) if sid == SID else None


def test_an_additive_re_vendor_passes_as_one_change():
    new = copy.deepcopy(DOC_V1)
    new["properties"]["b"] = {"type": "integer"}  # a new optional field in the vendored schema
    new["$defs"]["Inner"]["description"] = "described"
    changes = tools.classify(_with_external(DOC_V1), _with_external(new), _resolver(DOC_V1, new))
    assert [(c.row, c.breaking) for c in changes] == [("vendored schema revised (additive)", False)]


def test_a_re_vendor_that_changes_only_a_comment_is_additive():
    new = {**copy.deepcopy(DOC_V1), "$comment": "a note"}
    changes = tools.classify(_with_external(DOC_V1), _with_external(new), _resolver(DOC_V1, new))
    assert [(c.row, c.breaking) for c in changes] == [("vendored schema revised (additive)", False)]


def test_a_breaking_re_vendor_is_breaking_and_names_what_broke():
    new = copy.deepcopy(DOC_V1)
    new["$defs"]["Inner"]["properties"].pop("x")  # removed two levels down, under $defs
    changes = tools.classify(_with_external(DOC_V1), _with_external(new), _resolver(DOC_V1, new))
    assert [(c.row, c.path) for c in tools.breaking(changes)] == [
        ("removed", "$defs.Inner.properties.x")
    ]


def test_a_re_vendor_whose_copies_cannot_be_read_fails_closed():
    new = copy.deepcopy(DOC_V1)
    new["properties"]["b"] = {"type": "integer"}
    for resolve in (None, _resolver(new)):  # no resolver; the OLD bytes not found
        changes = tools.classify(_with_external(DOC_V1), _with_external(new), resolve)
        assert [(c.row, c.breaking) for c in changes] == [(tools.UNLISTED, True)]


def test_external_entries_added_removed_and_moved():
    old, new = _with_external(DOC_V1), _with_external(DOC_V1)
    new["external"]["https://x/new.json"] = {"file": "docs/schemas/new.json", "sha256": "0"}
    assert [(c.row, c.breaking) for c in tools.classify(old, new)] == [("descriptive", False)]
    assert [(c.row, c.breaking) for c in tools.classify(new, old)] == [("removed", True)]
    moved = _with_external(DOC_V1)
    moved["external"][SID]["file"] = "docs/schemas/elsewhere.json"
    assert [(c.row, c.breaking) for c in tools.classify(old, moved)] == [(tools.UNLISTED, True)]


def test_the_git_resolver_finds_old_bytes_in_history(tmp_path):
    import subprocess

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    target = tmp_path / "docs" / "schemas" / "doc-v1.json"
    target.parent.mkdir(parents=True)
    target.write_text(tools.render(DOC_V1))
    git("add", "-A")
    git("-c", "user.email=t@x", "-c", "user.name=t", "commit", "-qm", "v1")
    new = copy.deepcopy(DOC_V1)
    new["properties"]["b"] = {"type": "integer"}
    target.write_text(tools.render(new))  # re-vendored in the working tree, not committed
    resolve = tools.git_resolver(tmp_path, {SID: "docs/schemas/doc-v1.json"})
    assert resolve(SID, _doc_sha(DOC_V1)) == DOC_V1  # from history
    assert resolve(SID, _doc_sha(new)) == new  # from the working tree
    assert resolve(SID, "0" * 64) is None
    assert resolve("https://x/unknown.json", _doc_sha(DOC_V1)) is None


def test_the_writer_accepts_an_additive_re_vendor_and_refuses_a_breaking_one(tmp_path, monkeypatch):
    import src.api.answers_contract as gen

    path = tmp_path / "miworkbench-answers-v1.json"
    write_contract(path, _with_external(DOC_V1))
    additive = copy.deepcopy(DOC_V1)
    additive["properties"]["b"] = {"type": "integer"}
    breaking_doc = copy.deepcopy(DOC_V1)
    breaking_doc["properties"]["a"]["type"] = "integer"
    monkeypatch.setattr(gen, "resolver", lambda *docs: _resolver(DOC_V1, additive, breaking_doc))
    assert write_contract(path, _with_external(additive))["revision"] == 2
    before = _sha_file(path)
    with pytest.raises(BreakingAnswerChange, match="retyped"):
        write_contract(path, _with_external(breaking_doc))
    assert _sha_file(path) == before


def _sha_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
