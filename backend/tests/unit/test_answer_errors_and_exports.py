"""Exports reference miWorkbench's own documents, and the error envelopes are declared once (046 FTASKS 8.13, 8.14;
FR-046.10, FR-046.11).

- The routes that answer a published document unchanged are each a `$ref` to that document's `$id`, and nothing
  else: the circuit, cluster and probe definitions and the J-lens artifact document. The verdict export WRAPS its
  document (`{definition, definition_sha256, bytes}`: FTASKS 8.13's unverified item, settled by reading
  `services/behaviour_verdict.export`), so only its `definition` field references `behaviour-verdict-v1.json`.
- `external` lists exactly the referenced documents, each with its file, its `$id` and the sha256 of its bytes now.
  A changed byte in a referenced schema is reported (checked here on a copy of `docs/schemas/`).
- `errors.http` accepts a string, an object or a list `detail` (61 refusals send a dict); `errors.validation` is
  FastAPI's 422 body. Neither is repeated per route.

That a real export validates against its document is the 9.1 cases' (each export route answers through the answer
middleware, which loads every `external` document).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASE = "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/"
EXPORTS = {
    "GET /api/v1/circuits/{circuit_id}/export": BASE + "circuit-definition-v1.json",
    "GET /api/v1/cluster-profiles/{profile_id}/export": BASE + "cluster-definition-v1.json",
    "GET /api/v1/probe-monitors/probes/{probe_id}/definition": BASE + "probe-definition-v1.json",
    "GET /api/v1/jlens/artifacts/{slug}/document": BASE + "jspace-contracts-v1.json",
}
VERDICT_ID = "https://hitsai.net/schemas/mistudio/behaviour-verdict-v1.json"


def _contract() -> dict:
    return json.loads((REPO / "docs" / "schemas" / "miworkbench-answers-v1.json").read_text(encoding="utf-8"))


def _problems(contract: dict, root: Path) -> list[str]:
    """What is wrong with `contract`'s `external` against the document schemas under `root`."""
    out = []
    for sid, row in sorted(contract["external"].items()):
        path = root / row["file"]
        if not path.is_file():
            out.append(f"{sid}: {row['file']} is missing")
            continue
        raw = path.read_bytes()
        if json.loads(raw).get("$id") != sid:
            out.append(f"{sid}: {row['file']} has another $id")
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            out.append(f"{sid}: {row['file']} changed (sha256)")
    return out


def test_each_document_export_is_a_ref_to_its_id_and_nothing_else():
    contract = _contract()
    for key, sid in EXPORTS.items():
        assert contract["responses"][key] == {"200": {"$ref": sid}}, (key, contract["responses"][key])


def test_the_verdict_export_wraps_its_document():
    contract = _contract()
    entry = contract["responses"]["POST /api/v1/probe-monitors/behaviours/{key}/behaviour-verdict/export"]
    wrapper = contract["$defs"][entry["200"]["$ref"].split("/")[-1]]
    assert set(wrapper["required"]) == {"definition", "definition_sha256", "bytes"}, wrapper["required"]
    assert wrapper["properties"]["definition"]["$ref"] == VERDICT_ID, wrapper["properties"]["definition"]
    assert set(wrapper["properties"]["definition"]) <= {"$ref", "description"}, wrapper["properties"]["definition"]
    assert wrapper["properties"]["bytes"]["type"] == "integer"  # the length of the canonical bytes


def test_external_lists_exactly_the_referenced_documents_with_their_sha256():
    contract = _contract()
    assert set(contract["external"]) == set(EXPORTS.values()) | {VERDICT_ID}, sorted(contract["external"])
    assert not _problems(contract, REPO), _problems(contract, REPO)


def test_a_changed_byte_in_a_referenced_schema_is_reported(tmp_path):
    """FR-046.11's NC, built in: one byte of a referenced schema changed in a copy of the tree."""
    contract = _contract()
    shutil.copytree(REPO / "docs" / "schemas", tmp_path / "docs" / "schemas")
    target = tmp_path / "docs" / "schemas" / "probe-definition-v1.json"
    target.write_bytes(target.read_bytes().replace(b"Portable probe monitor", b"Portable probe Monitor", 1))
    assert _problems(contract, tmp_path) == [f"{BASE}probe-definition-v1.json: "
                                             "docs/schemas/probe-definition-v1.json changed (sha256)"]


def _errors_validator(name: str, contract: dict):
    from jsonschema import Draft202012Validator

    return Draft202012Validator({**contract["errors"][name], "$defs": contract["$defs"]})


def test_a_dict_detail_refusal_and_a_422_validate_against_their_envelopes():
    contract = _contract()
    http = _errors_validator("http", contract)
    for detail in ("Circuit not found", {"code": "model_not_downloaded", "message": "m", "model_id": "m_1"},
                   [{"code": "a"}]):
        assert not list(http.iter_errors({"detail": detail})), detail
    assert list(http.iter_errors({"detail": 3})) and list(http.iter_errors({}))
    validation = _errors_validator("validation", contract)
    # FastAPI's own 422 bodies (0.159, Pydantic 2.14): a missing field carries no `ctx`, a constrained one does.
    for body in ({"detail": [{"type": "missing", "loc": ["body", "name"], "msg": "Field required", "input": {}}]},
                 {"detail": [{"type": "greater_than_equal", "loc": ["body", "n"], "input": 0, "ctx": {"ge": 1},
                              "msg": "Input should be greater than or equal to 1"}]}):
        assert not list(validation.iter_errors(body)), list(validation.iter_errors(body))
    assert set(contract["errors"]) == {"http", "validation"}
    assert '"detail"' not in json.dumps(contract["responses"]), "an error envelope repeated per route"


def test_detail_declared_as_a_string_only_would_refuse_the_dict_refusals():
    """FR-046.10's NC, built in: the envelope narrowed to a string `detail` refuses a dict refusal."""
    from jsonschema import Draft202012Validator

    narrowed = json.loads(json.dumps(_contract()["errors"]["http"]))
    narrowed["properties"]["detail"] = {"type": "string"}
    assert list(Draft202012Validator(narrowed).iter_errors({"detail": {"code": "x"}}))
