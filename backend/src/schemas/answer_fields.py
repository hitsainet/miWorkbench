"""Shared pieces of miWorkbench's answer models (046 FTID §3.4, FPRD FR-046.6 and FR-046.7).

The two markers change only what the OpenAPI document, and so the published answer contract
(`docs/schemas/`, `src/api/answers_contract.py`), says about a field. Neither changes what a route serves:

- `may_be_absent()` (TD4): a field some answers leave out. The contract does not list it in `required`, because
  "required" means "always sent" (SH7).
- `history_form()` (WT9): a field holding JSON an earlier release wrote and stored. Beneath it the contract keeps every
  type but drops every `required`, since a stored value may predate a field. The contract's `description` states this
  as the one exception to "required means always sent".

`PendingApprovalAnswer` is the approval gate's `202` body (`core/approval_gate.pending_approval_body`), field for field;
a test builds one and requires the body's keys to EQUAL the model's fields, so the two cannot drift.
"""
from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field
from typing_extensions import TypeAliasType

from ..api.contract_tools import HISTORY_FORM, MAY_BE_ABSENT


#: Any JSON value whose shape is the data's or a stored record's own (a dataset cell, a raw label, a conversation as
#: stored, a stored field's old or new value). Recursive, so it is typed at every depth (`contract_tools.is_typed`
#: refuses a bare `Any`).
JsonCell = TypeAliasType(
    "JsonCell", Union[str, float, bool, dict[str, Any], list["JsonCell"], None]
)


def may_be_absent(**kwargs: Any) -> Any:
    """A field some answers leave out: default `None`, never `required` in the contract (TD4)."""
    extra = dict(kwargs.pop("json_schema_extra", None) or {})
    extra[MAY_BE_ABSENT] = True
    kwargs.setdefault("default", None)
    return Field(json_schema_extra=extra, **kwargs)


def history_form(**kwargs: Any) -> Any:
    """A field whose stored JSON an earlier release may have written: checked without `required` beneath it (WT9)."""
    extra = dict(kwargs.pop("json_schema_extra", None) or {})
    extra[HISTORY_FORM] = True
    return Field(json_schema_extra=extra, **kwargs)


class PendingApprovalObject(BaseModel):
    """The suite result language's `approval` object as the gate's `202` carries it (A6): the operator decides in
    miWorkbench, and the caller sends the same request again with the approval header."""

    id: str = Field(description="The approval request's id (the same as `approval_request_id`)")
    status: Literal["pending"]
    decide_in: Literal["miworkbench"]
    then: Literal["call_again_with"]
    action: str = Field(description="The gated action this request waits for")
    summary: str = Field(description="One line naming the route and the header to re-send with (at most 500 characters)")


class PendingApprovalAnswer(BaseModel):
    """The `202` a gated route answers an agent's call that carries no consumed approval (`pending_approval_body`)."""

    approval_request_id: str = Field(description="The approval request filed for this call")
    status: Literal["pending_approval"]
    action: str = Field(description="The gated action")
    retry_with_header: dict[str, str] = Field(description="The header (name: value) to send the same request again with")
    approval: PendingApprovalObject
