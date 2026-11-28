"""The shared answer-model pieces (046 FTASKS 6.2, FTID §3.4).

- `PendingApprovalAnswer` describes `pending_approval_body` field for field, at both levels, so the model cannot drift
  from the body the gate's `202` sends.
- The two markers reach the OpenAPI schema and change nothing a model validates or dumps.
"""
from __future__ import annotations

from pydantic import BaseModel

from src.api.contract_tools import HISTORY_FORM, MAY_BE_ABSENT
from src.core.approval_gate import PendingApproval, pending_approval_body
from src.schemas.answer_fields import (
    PendingApprovalAnswer,
    PendingApprovalObject,
    history_form,
    may_be_absent,
)


def _body() -> dict:
    return pending_approval_body(PendingApproval("apr_1", "start_labeling",
                                                 {"_method": "POST", "_route": "/api/v1/labeling/trials"}))


def test_the_pending_model_has_exactly_the_bodys_keys():
    body = _body()
    assert set(body) == set(PendingApprovalAnswer.model_fields)
    assert set(body["approval"]) == set(PendingApprovalObject.model_fields)


def test_the_pending_body_validates_and_round_trips():
    body = _body()
    assert PendingApprovalAnswer.model_validate(body).model_dump(mode="json") == body


def test_the_markers_reach_the_schema_and_nothing_else():
    class M(BaseModel):
        a: str
        b: int | None = may_be_absent(description="sometimes left out")
        c: dict = history_form(description="stored by an earlier release")

    schema = M.model_json_schema(mode="serialization")
    assert schema["properties"]["b"][MAY_BE_ABSENT] is True
    assert schema["properties"]["c"][HISTORY_FORM] is True
    assert MAY_BE_ABSENT not in schema["properties"]["a"]
    assert M(a="x", c={}).model_dump() == {"a": "x", "b": None, "c": {}}
