"""ENH-010 044 task 7.1: the scoring contract's request and response models (FTDD §5.7.1) — `extra="forbid"` at every
level, no aliases, the head-label grammar, and the whole refusal table (R3B-16)."""
from __future__ import annotations

import base64
import copy
import hashlib

import pytest
from pydantic import BaseModel, ValidationError

from src.schemas import wsp_scoring as ws

H = "a" * 64


def request(**overrides):
    body = {
        "kind": "miworkbench.wsp-score-request/v1", "bundle_id": "wsb_0123456789ab", "bundle_sha256": H,
        "run_ref": {"miforge_run_id": "wsp_0123456789ab", "patch_sha256": H, "max_changed_entries": None},
        "heldout_ref": {"owner": "miworkbench", "sets": [
            {"name": "ood", "view_id": "v1", "role": "eval", "rows": 10, "row_ids_sha256": H},
            {"name": "cal", "view_id": "v2", "role": "calibration", "rows": 5, "row_ids_sha256": H}]},
        "heads": [{"label": "parent", "tensors": None},
                  {"label": "candidate", "tensors": [{"name": "weight", "dtype": "float32", "shape": [4],
                                                       "data_b64": base64.b64encode(b"\0" * 16).decode(),
                                                       "sha256": hashlib.sha256(b"\0" * 16).hexdigest()}]}],
        "primary_sets": ["ood"],
    }
    body.update(overrides)
    return body


def _models(root):
    seen, todo = set(), [root]
    while todo:
        model = todo.pop()
        if model in seen:
            continue
        seen.add(model)
        for field in model.model_fields.values():
            for arg in _types(field.annotation):
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    todo.append(arg)
    return seen


def _types(annotation):
    args = getattr(annotation, "__args__", None)
    if not args:
        return [annotation]
    return [t for a in args for t in _types(a)]


@pytest.mark.parametrize("root", [ws.WspScoreRequest, ws.WspScoreResponse])
def test_every_model_forbids_extra_and_has_no_alias(root):
    models = _models(root)
    assert len(models) >= 5
    for model in models:
        assert model.model_config.get("extra") == "forbid", model.__name__
        assert all(f.alias is None and f.serialization_alias is None and f.validation_alias is None
                   for f in model.model_fields.values()), model.__name__


def test_a_valid_request_round_trips_with_its_own_field_names():
    """The opt-in `report_identical_to_parent` (operator decision 2026-11-27) defaults to false and round-trips."""
    parsed = ws.WspScoreRequest.model_validate(request())
    assert parsed.model_dump(mode="json") == {**request(), "report_identical_to_parent": False}
    asked = {**request(), "report_identical_to_parent": True}
    assert ws.WspScoreRequest.model_validate(asked).model_dump(mode="json") == asked


def test_an_extra_key_is_refused_at_depth():
    body = request()
    body["heldout_ref"]["sets"][0]["text"] = "leak"
    with pytest.raises(ValidationError):
        ws.WspScoreRequest.model_validate(body)


@pytest.mark.parametrize("label", ["parent", "candidate", "control", "control:0", "control:12",
                                   *[f"baseline:{k}" for k in ws.BASELINE_KINDS]])
def test_every_documented_label_is_accepted(label):
    ws.ScoreHead.model_validate({"label": label})


@pytest.mark.parametrize("label", ["baseline:svm", "control:01", "control:-1", "Parent", "candidate2", "control:"])
def test_an_undocumented_label_is_refused(label):
    with pytest.raises(ValidationError):
        ws.ScoreHead.model_validate({"label": label})


def test_tensor_bearing_labels():
    assert [ws.carries_tensors(x) for x in ("parent", "candidate", "control", "control:3", "baseline:lda")] == [
        False, True, True, True, False]
    # `control_index` left with R2 review fix OR2-044a-L4 (no phase-2 caller); 044-C's run binding (7.2a,
    # `control_index_out_of_range`) adds it with its caller.


def test_duplicate_labels_and_unknown_primary_sets_are_refused():
    body = request()
    body["heads"].append({"label": "parent"})
    with pytest.raises(ValidationError):
        ws.WspScoreRequest.model_validate(body)
    with pytest.raises(ValidationError):
        ws.WspScoreRequest.model_validate(request(primary_sets=["nope"]))


def test_non_base64_tensor_bytes_are_refused():
    body = request()
    body["heads"][1]["tensors"][0]["data_b64"] = "not base64!"
    with pytest.raises(ValidationError):
        ws.WspScoreRequest.model_validate(body)


def test_a_non_finite_bar_is_refused():
    with pytest.raises(ValidationError):
        ws.Bar.model_validate({"threshold": float("inf"), "source": "calibration", "n_negatives": 1,
                               "realised_fpr": 0.0, "target_fpr": 0.01})


def test_the_refusal_table_is_whole():
    """FTDD §5.3's scorer codes, phase 3 included (R3B-16); the 409 exhaustion is its own status, never 429."""
    required = {"heldout_ref_mismatch", "parent_tensors_not_accepted", "head_shape_mismatch", "too_many_heads",
                "capture_not_ready", "capture_parity_failed", "candidate_outside_patch_limits",
                "heldout_set_head_budget_exhausted", "scorer_busy", "operator_or_service_only",
                "bundle_kind_unsupported", "candidate_not_the_registered_patch",
                "control_not_the_seeded_permutation", "control_index_out_of_range"}
    assert required <= set(ws.REFUSALS)
    assert ws.refusal_status("heldout_set_head_budget_exhausted") == 409
    assert ws.refusal_status("scorer_busy") == 429
    assert ws.refusal_status("operator_or_service_only") == 403
    # 503 only for `score_timeout` (3.5's reader, OVN-wsp-workbench-40): the call answered nothing and recorded nothing.
    assert {status for status, _ in ws.REFUSALS.values()} <= {403, 404, 409, 422, 429, 503}
    assert [c for c, (status, _) in ws.REFUSALS.items() if status == 503] == ["score_timeout"]


def test_the_response_keys_are_exactly_the_documented_ones():
    """No field beyond FTDD §5.7.1's (7.4's leak test pins the served answer; this pins the model)."""
    assert set(ws.WspScoreResponse.model_fields) == {"kind", "bundle_sha256", "heldout_ref_sha256", "function",
                                                     "sets", "scored_at", "call_id"}
    # `identical_to_parent`: operator decision 2026-11-27, present only when the request asks for it.
    assert set(ws.SetResult.model_fields) == {"name", "view_id", "role", "row_ids_sha256", "scored_row_ids_sha256",
                                              "rows", "excluded", "labels", "scores", "bars", "metrics",
                                              "identical_to_parent"}
    assert set(ws.Metrics.model_fields) == {"auroc", "recall_at_bar", "false_alarms_at_bar", "negatives",
                                            "positives"}
