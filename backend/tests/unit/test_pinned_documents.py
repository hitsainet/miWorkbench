"""The document bytes a consumer's contract pins, after this repo published a newer revision (ENH-011 X-2).

`scripts/generate_output_schemas.py` embeds a document schema (a circuit or cluster definition) into a miServer tool's
output schema only at the bytes miServer's answer contract pins in its `external` (sha256). miWorkbench PRODUCES those
documents and publishes first (contracts rule 9), so between this repo's push and miServer's re-vendor the pin names
the previous revision. Those bytes are kept, content-addressed, under `docs/schemas/pinned/`, and only for that window:

1. every pin of the vendored miServer contract resolves to bytes (this repo's file or a pinned copy);
2. every pinned copy is still pinned, under its own `$id`, and is NOT this repo's current file (else it is dead: delete
   it with the re-vendor of `miserver-answers-v1.json`); its name carries the first 12 hex of its sha256;
3. the generator refuses a pin no file matches;
4. the embedding is PER PRODUCT (046-2's rebase onto ENH-011 X-2, 2026-11-28): since 046-2 miWorkbench's own answer
   contract pins the documents too, at the bytes it publishes now, so in every producer-first window the two contracts
   pin one document differently. Each product's tools embed what ITS contract pins: the native tools the current file
   under its name, miServer's tools the pinned copy under the copy's name; one tool needing both stops the build.
   (X-2's rule was "two contracts pinning one document at different bytes stop the build", which made every such
   window a build stop once 046-2's contract existed.)

Controls: 0xcc/reviews/ENH-011_X-2_controls_2026-11-27.md; the per-product rule, 0xcc/reviews/046_controls_2026-11-27.md
(§ the rebase onto 0b64145e6).
"""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts import generate_output_schemas as gen


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copies() -> list:
    return sorted(gen.PINNED_DOCUMENTS.glob("*.json"))


def test_every_pin_of_miservers_contract_resolves_to_bytes():
    source = gen.miserver_source()
    assert source.external, "the vendored miServer contract pins no document (it pinned four on 2026-11-27)"
    for ref, row in source.external.items():
        path = gen._documents().get(ref)  # noqa: SLF001
        assert path is not None, f"no file in docs/schemas/ has $id {ref}"
        data = gen._pinned_bytes(ref, path, source)  # noqa: SLF001
        assert _sha(data) == row["sha256"], ref
        assert json.loads(data)["$id"] == ref


def test_every_pinned_copy_is_still_pinned_and_not_the_current_file():
    pins = {row["sha256"]: ref for ref, row in gen.miserver_source().external.items()}
    for copy_path in _copies():
        data = copy_path.read_bytes()
        sha = _sha(data)
        assert sha in pins, (f"{copy_path.name} is pinned by no contract any more: delete it (miServer re-vendored "
                             "and miserver-answers-v1.json was re-vendored here)")
        ref = json.loads(data)["$id"]
        assert ref == pins[sha], (copy_path.name, ref, pins[sha])
        current = gen._documents()[ref]  # noqa: SLF001
        assert current.read_bytes() != data, f"{copy_path.name} is the current {current.name}: delete the copy"
        assert copy_path.name == f"{current.stem}.{sha[:12]}.json", copy_path.name


def _source(external: dict) -> gen.Source:
    return gen.Source("miserver", {}, {}, set(), external)


def test_a_pin_no_file_matches_stops_the_build(monkeypatch, tmp_path):
    ref, path = next(iter(sorted(gen._documents().items())))  # noqa: SLF001
    monkeypatch.setattr(gen, "PINNED_DOCUMENTS", tmp_path)  # no copies
    with pytest.raises(gen.GenerationError, match="is not the document miserver's contract pins"):
        gen._pinned_bytes(ref, path, _source({ref: {"sha256": "0" * 64}}))  # noqa: SLF001


def test_a_matching_pinned_copy_is_used_and_the_current_file_otherwise(monkeypatch, tmp_path):
    ref, path = next(iter(sorted(gen._documents().items())))  # noqa: SLF001
    older = json.dumps({"$id": ref, "older": True}).encode()
    (tmp_path / "older.json").write_bytes(older)
    monkeypatch.setattr(gen, "PINNED_DOCUMENTS", tmp_path)
    got = gen._pinned_document(ref, path, _source({ref: {"sha256": _sha(older)}}))  # noqa: SLF001
    assert got == ("older.json", older)
    current = path.read_bytes()
    assert gen._pinned_document(ref, path, _source({ref: {"sha256": _sha(current)}})) == (path.name, current)  # noqa: SLF001
    assert gen._pinned_document(ref, path, _source({})) == (path.name, current)  # noqa: SLF001  (unpinned)


def test_two_products_pinning_one_document_differently_each_embed_their_own(monkeypatch, tmp_path):
    """The per-product rule through the generator's own `tool_schema`, on the live registry's export tools: miServer's
    contract made to pin an OLDER circuit definition (a planted copy), miWorkbench's left at the current file. The
    native export embeds the current document under its file name; miServer's export embeds the planted bytes under
    the copy's name; neither sees the other's."""
    from scripts.output_overlays import OVERLAYS
    from src.mcp_server.result_language import load_contract
    from src.mcp_server.tool_classes import result_language_properties

    ref = "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/circuit-definition-v1.json"
    current_path = gen._documents()[ref]  # noqa: SLF001
    older = json.loads(current_path.read_text(encoding="utf-8"))
    older["description"] = "an older revision, planted by the test"
    older_bytes = json.dumps(older, indent=2).encode()
    copy_name = f"{current_path.stem}.{_sha(older_bytes)[:12]}.json"
    (tmp_path / copy_name).write_bytes(older_bytes)
    monkeypatch.setattr(gen, "PINNED_DOCUMENTS", tmp_path)
    native, miserver = gen.native_source(), gen.miserver_source()
    assert native.external[ref]["sha256"] == _sha(current_path.read_bytes()), "miWorkbench's contract pins the file"
    miserver.external = {**miserver.external, ref: {**miserver.external[ref], "sha256": _sha(older_bytes)}}
    sources = {"miworkbench": native, "miserver": miserver}
    tools = gen.live_tools()
    language = load_contract()
    fields = result_language_properties(language)

    def defs_of(tool: str) -> dict:
        schema, _route, reason = gen.tool_schema(tool, tools[tool].fn, sources, OVERLAYS, language, fields)
        assert schema is not None, reason
        return {k: v for k, v in schema["$defs"].items() if v.get("$id") == ref}

    mine = defs_of("export_circuit_definition")
    theirs = defs_of("miserver_export_circuit")
    assert list(mine) == [current_path.name]
    assert "an older revision" not in json.dumps(mine)
    assert list(theirs) == [copy_name]
    assert theirs[copy_name]["description"] == "an older revision, planted by the test"


def test_one_tool_needing_two_revisions_of_a_document_stops_the_build():
    ref = "https://example.invalid/doc.json"
    schema = {"$defs": {"doc.json": {"$id": ref}, "doc.0123456789ab.json": {"$id": ref}, "Other": {"type": "object"}}}
    with pytest.raises(gen.GenerationError, match="embeds two revisions of"):
        gen._one_revision_per_document(schema, "a_tool")  # noqa: SLF001
    gen._one_revision_per_document({"$defs": {"doc.json": {"$id": ref}}}, "a_tool")  # noqa: SLF001


def test_a_document_reference_left_unresolved_stops_the_build():
    assert gen._external_refs({"anyOf": [{"$ref": "https://x/y.json"}, {"$ref": "#/$defs/A"}]}) == ["https://x/y.json"]  # noqa: SLF001
    assert gen._external_refs({"$defs": {"d": {"$id": "https://x/d.json", "$ref": "https://x/e.json"}}}) == []  # noqa: SLF001


def test_tool_schema_refuses_a_document_reference_no_product_resolved(monkeypatch):
    """The CALLER of the unresolved-reference check, through `tool_schema` on the live registry: with the per-product
    rewrite of the answer parts made inert, `export_circuit_definition`'s reference to the circuit definition reaches
    the end unresolved, and the build stops instead of publishing a schema whose `$ref` points at a URL."""
    from scripts.output_overlays import OVERLAYS
    from src.mcp_server.result_language import load_contract
    from src.mcp_server.tool_classes import result_language_properties

    real = gen._embed_externals  # noqa: SLF001

    def inert_for_parts(node, source, embedded):
        if isinstance(node, dict) and "$ref" in node and not str(node["$ref"]).startswith("#"):
            return node  # an answer part that IS a document reference: left as it came
        return real(node, source, embedded)

    monkeypatch.setattr(gen, "_embed_externals", inert_for_parts)
    sources = {"miworkbench": gen.native_source(), "miserver": gen.miserver_source()}
    tools = gen.live_tools()
    language = load_contract()
    with pytest.raises(gen.GenerationError, match="no product's answer resolved"):
        gen.tool_schema("export_circuit_definition", tools["export_circuit_definition"].fn, sources, OVERLAYS,
                        language, result_language_properties(language))
