"""Report identity (Feature 040 S2 FR-25; 040-T17; FTASKS 11.1)."""
from __future__ import annotations

import copy

import pytest

from src.services.detector_identity import REPORT_ALGORITHM, identity, jcs_bytes, round6

BASE = {"report_algorithm": REPORT_ALGORITHM, "behaviour_key": "k-a", "behaviour_version": 2, "model_id": "m_1",
        "dataset_identity": {"kind": "view_set", "views": [{"id": "pmd_1", "rows": 10, "snapshots": {"c": "a" * 64}}]},
        "probe": {"id": "pm_1", "decision_revision": 3}, "cited": {"evaluations": ["pme_1"]},
        "rule_set_digest": None, "settings": {"calibration_floor_negatives": 2000}}


def test_identical_inputs_identical_digest_whatever_the_key_order():
    reordered = dict(reversed(list(copy.deepcopy(BASE).items())))
    assert identity(BASE) == identity(reordered)


@pytest.mark.parametrize("path,value", [
    (("report_algorithm",), REPORT_ALGORITHM + 1), (("behaviour_version",), 3), (("model_id",), "m_2"),
    (("dataset_identity", "views", 0, "rows"), 11), (("probe", "decision_revision"), 4),
    (("cited", "evaluations", 0), "pme_2"), (("rule_set_digest",), "b" * 64),
    (("settings", "calibration_floor_negatives"), 5000),
])
def test_each_input_class_moves_the_digest(path, value):
    moved = copy.deepcopy(BASE)
    target = moved
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    assert identity(moved) != identity(BASE)


def test_rfc8785_numbers_and_rounding():
    assert jcs_bytes({"a": 1.0, "b": 1e21}) == b'{"a":1,"b":1e+21}'
    assert round6(0.1234567) == 0.123457 and round6(-0.0000001) == 0.0 and round6(None) is None


def test_derived_from_moves_neither_digest():
    """ENH-010 (044 task 9.2, CHAIN_PLAN P25): a probe definition's `provenance.derived_from` — its weight-space
    programming lineage — moves neither the detector digest nor the decision digest; `schemas/probe_identity.py`
    needed no change (`provenance` is excluded at the root). Mutation control (R1A-W16): make `detector_payload` also
    emit `{"provenance": {"derived_from": …}}` when set → this test and only the new identity case go red."""
    from src.schemas.probe_definition import ProbeDefinitionV1
    from src.schemas.probe_identity import document_identity
    from tests.unit.test_probe_identity_cases import _residual

    base = _residual()
    derived = copy.deepcopy(base)
    derived["provenance"]["derived_from"] = {
        "parent_probe_id": "pm_parent", "parent_detector_digest": "e" * 64, "patch_sha256": "f" * 64,
        "provider": "weightspace", "provider_version": "0.3.0",
        "verdict": {"value": "improved", "run_sha256": "9" * 64},
    }
    ProbeDefinitionV1.model_validate(derived)          # the field is part of the contract, not an extra
    assert document_identity(derived) == document_identity(base)
    other = copy.deepcopy(derived)
    other["provenance"]["derived_from"]["verdict"]["value"] = "regressed"
    other["provenance"]["derived_from"]["provider_version"] = "0.4.0"
    assert document_identity(other) == document_identity(base)


def test_a_pre_044_definition_still_validates_and_keeps_its_bytes():
    """ENH-010 (044 task 9.1): a definition exported before `derived_from` existed validates against the new contract,
    and a non-derived definition serialises WITHOUT the key (`exclude_if`), so its bytes — and every consumer whose
    vendored `Provenance` forbids extra keys — are unchanged."""
    import json
    from pathlib import Path

    import jsonschema

    from src.schemas.probe_definition import ProbeDefinitionV1

    root = Path(__file__).resolve().parents[3] / "docs" / "schemas"
    published = json.loads((root / "probe-definition-v1.json").read_text())
    cases = json.loads((root / "probe-identity-cases.json").read_text())
    pre_044 = next(c["document"] for c in cases["cases"] if c["name"] == "residual")
    assert "derived_from" not in pre_044["provenance"]
    jsonschema.validate(pre_044, published)
    parsed = ProbeDefinitionV1.model_validate(pre_044)
    assert "derived_from" not in parsed.model_dump(mode="json")["provenance"]
