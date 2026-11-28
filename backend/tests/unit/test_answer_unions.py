"""Unions, nullables and bodiless routes in miWorkbench's answer contract (046 FTASKS 8.12; FR-046.8, FR-046.9).

Each answer is checked as the test middleware checks a served one (`answer_middleware.Contract`: the strict form,
formats as the production client reads them), against the COMMITTED contract:

- `POST /steering/cluster-allocation` answers one of two models, and each full answer is exactly one of them;
- each of the 5 nullable routes accepts `null` and its model's full answer, and nothing else at the top level;
- the 4 bodiless covered routes are `excluded` with no media type ("no content"); the HTTP half (each answers its
  204 with no body) is the 9.1 cases'.

Controls (record `0xcc/reviews/046_controls_2026-11-25.md`): declare the allocation route as the single-layer
model only → the multi-layer case red; drop `null` from one nullable route → its null case red.
"""
from __future__ import annotations

import json

import pytest

ALLOCATION = "POST /api/v1/steering/cluster-allocation"
#: FR-046.8: the routes that answer JSON `null` when there is nothing to show (four typed before 046, plus the
#: replication report typed in P7).
NULLABLE = (
    "GET /api/v1/features/{feature_id}/label/enhanced/latest",
    "GET /api/v1/features/{feature_id}/nlp-analysis",
    "GET /api/v1/jlens/artifacts/{slug}/band-report",
    "GET /api/v1/jlens/artifacts/{slug}/gate",
    "GET /api/v1/jlens/reports/replication",
)
#: FR-046.9's three, plus the extraction delete (a 204, or a 202 text note).
BODILESS = (
    "DELETE /api/v1/models/{model_id}",
    "DELETE /api/v1/probe-monitors/runs/{run_id}",
    "DELETE /api/v1/probe-monitors/judge-prompts/{prompt_id}",
    "DELETE /api/v1/extractions/{extraction_id}",
)


def _contract():
    from tests.support.answer_middleware import Contract
    from src.api.answers_contract import DEFAULT_PATH

    return Contract(json.loads(DEFAULT_PATH.read_text(encoding="utf-8")))


def _full(contract, ref: str):
    from tests.support.answer_samples import sample

    return sample({"$ref": ref}, contract.contract)


def _entry(contract, key: str) -> dict:
    """A route's 200 entry, through a named root model (`ReplicationReportOrNull`: FR-046.8's nullable form since
    046-2 review round 1, finding 2) to the union it names."""
    schema = contract.contract["responses"][key]["200"]
    if set(schema) == {"$ref"} and schema["$ref"].startswith("#/$defs/"):
        named = contract.contract["$defs"][schema["$ref"][len("#/$defs/"):]]
        if "anyOf" in named or "oneOf" in named:
            return named
    return schema


def _members(contract, key: str) -> list[str]:
    schema = _entry(contract, key)
    return [m["$ref"] for m in schema.get("anyOf") or schema.get("oneOf") or [] if "$ref" in m]


def _fits(contract, ref: str, body) -> bool:
    """Whether `body` is a valid answer of the one definition `ref`, in the strict form."""
    from jsonschema import Draft202012Validator

    from tests.support.strict_formats import STRICT

    validator = Draft202012Validator({"$ref": contract.contract["$id"] + ref}, registry=contract.registry,
                                     format_checker=STRICT)
    return not list(validator.iter_errors(body))


def test_the_allocation_route_answers_either_model_and_each_answer_is_exactly_one():
    contract = _contract()
    members = _members(contract, ALLOCATION)
    assert sorted(m.split("/")[-1] for m in members) == ["ClusterAllocationResponse", "MultiLayerAllocationResponse"]
    for ref in members:
        body = _full(contract, ref)
        assert not contract.check_json(ALLOCATION, 200, body), (ref, contract.check_json(ALLOCATION, 200, body))
        assert [m for m in members if _fits(contract, m, body)] == [ref], ref


@pytest.mark.parametrize("key", NULLABLE)
def test_a_nullable_route_answers_null_or_its_model(key):
    contract = _contract()
    members = _members(contract, key)
    nulls = [m for m in _entry(contract, key).get("anyOf") or [] if m.get("type") == "null"]
    assert len(members) == 1 and len(nulls) == 1, contract.contract["responses"][key]
    assert not contract.check_json(key, 200, None), contract.check_json(key, 200, None)
    body = _full(contract, members[0])
    assert not contract.check_json(key, 200, body), contract.check_json(key, 200, body)
    assert contract.check_json(key, 200, "x") and contract.check_json(key, 200, [])


@pytest.mark.parametrize("key", BODILESS)
def test_a_bodiless_route_is_excluded_with_no_media_type(key):
    contract = _contract()
    assert key not in contract.contract["responses"], key
    row = contract.contract["excluded"][key]
    if key == "DELETE /api/v1/extractions/{extraction_id}":  # a 204, or a queued 202's text note (review round 1)
        assert row["statuses"] == {"202": "text/plain", "204": None}, row
    else:
        assert row == {"media_type": None, "reason": "no content", "statuses": {"204": None}}, row
    assert not contract.check_other(key, 204, None)
    assert contract.check_other(key, 204, "application/json", 2), "a JSON body from a bodiless route must fail"
