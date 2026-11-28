"""ENH-010 FU-7a C2: the `supplement` view role (2026-11-25) — extra PRACTICE rows for a WSP bundle, negatives only in
this revision, scored as a negatives-only set, never counted towards a rung, and captured with the held-out sets.

Negative controls (`0xcc/reviews/fu7a_c1c2_2026-11-25.md`): let a supplement map a positive, or score it under the
`eval` preparation → red.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError


def _create(**over):
    from src.schemas.probe_monitor import ProbeDatasetCreate

    body = {"name": "unseen-generator negatives", "dataset_id": "d", "input_column": "text", "label_column": "kind",
            "label_mapping": {"masked": "negative", "lookalike": "negative"}, "role": "supplement",
            "behaviour": "dlp-ssn-us"}
    return ProbeDatasetCreate(**{**body, **over})


def test_a_supplement_of_negatives_registers_with_its_behaviour():
    view = _create()
    assert view.role == "supplement" and view.behaviour == "dlp-ssn-us"


def test_a_supplement_mapping_a_positive_is_refused():
    with pytest.raises(ValidationError, match="supplement set must not map any value to 'positive'"):
        _create(label_mapping={"masked": "negative", "digits": "positive"})


def test_a_supplement_with_no_negative_is_refused():
    with pytest.raises(ValidationError, match="needs at least one value mapped to 'negative'"):
        _create(label_mapping={"masked": "excluded"})


def test_a_supplement_takes_no_distribution():
    with pytest.raises(ValidationError, match="distribution is meaningless"):
        _create(distribution="out_of_distribution")


def test_the_view_needs_the_negatives_floor():
    from src.services.probe_monitor_service import ProbeDatasetRefused, _floor, build_view

    floor = _floor()
    with pytest.raises(ProbeDatasetRefused, match="practice supplement needs at least"):
        build_view(["x"] * (floor - 1), ["masked"] * (floor - 1), {"masked": "negative"}, role="supplement")
    built = build_view(["row %d" % i for i in range(floor)], ["masked"] * floor, {"masked": "negative"},
                       role="supplement")
    assert len(built.examples) == floor


def test_a_supplement_is_scored_as_a_negatives_only_set():
    from src.services.probe_monitor_run import preparation_role

    assert preparation_role(SimpleNamespace(role="supplement")) == "calibration"
    assert preparation_role(SimpleNamespace(role="eval")) == "eval"


def test_a_supplement_evaluation_never_counts_towards_the_rung():
    from src.services.probe_monitor_rung import rung_eligibility_of

    verdict = rung_eligibility_of("supplement", None, None)
    assert verdict.counts is False and verdict.note == "role supplement"


def test_the_capture_and_the_bundle_agree_on_the_role():
    from src.services.wsp_bundle import HELDOUT_CAPTURE_ROLES, SUPPLEMENT_ROLE
    from src.services.wsp_capture import CAPTURE_ROLES, HELDOUT_ROLES

    assert SUPPLEMENT_ROLE in CAPTURE_ROLES and SUPPLEMENT_ROLE not in HELDOUT_ROLES
    assert SUPPLEMENT_ROLE not in HELDOUT_CAPTURE_ROLES
    assert len(SUPPLEMENT_ROLE) <= 16, "wsp_activation_captures.role and probe_monitor_datasets.role are VARCHAR(16)"
