"""Stored answers in miWorkbench's answer contract: the history form (046 FTASKS 8.15; WT9, FTDD §6.7).

A field holding JSON that an earlier release wrote and stored is declared with `answer_fields.history_form()`. The
contract keeps its types but drops every `required` beneath it (`contract_tools.history_form`), because a stored value
can predate a field, and its `description` names it as the one exception to "required means always sent".

The fields, after reading each candidate FTDD §6.7 names (record `0xcc/reviews/046_controls_2026-11-25.md` §P7):

- typed records read back from a row's JSON column, whose keys grew over releases: the probe dataset view's seven
  (`keyword_filter`, `counts`, `dataset_version`, `positive_labels_confirmed`, `column_snapshots`, `rule_columns`,
  `evidence_spans`) and the summary's `counts`; a blend report's `resolved` (row and list); the stored replication
  report's `results`;
- declared maps read back from storage (no `required` beneath them, so already their own history form; marked so the
  description names every stored field of the candidates): the steering result (the Celery result backend), a
  validation or calibration manifest's `payload`, the stored band report's `boundaries` and `profiles`, and the
  J-lens intervention `records`.
- Not stored JSON, so not marked: the J-lens gate (each field rebuilt with a default, `jlens.gate`) and the
  task-queue lookup (`table`, `row_id`, `status` columns).

Each field has two cases, checked as the test middleware checks a served answer (`answer_middleware.Contract`: the
strict form, formats as the production client reads them) against the COMMITTED contract: an old-shaped stored value
validates, and a wrong-typed one fails. A set-equality test ties this table to the contract's own list.

Control M13 (record): `history_form` keeps `required` and the contract is regenerated → every typed old-shaped case red.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Callable

import pytest


def _drop(*keys: str) -> Callable[[Any], Any]:
    def edit(value: Any) -> Any:
        for key in keys:
            value.pop(key)
        return value
    return edit


def _set(key: str, wrong: Any) -> Callable[[Any], Any]:
    def edit(value: Any) -> Any:
        value[key] = wrong
        return value
    return edit


def _each_entry(edit: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """`edit` applied to every value of a map, or every item of a list."""
    def apply(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: edit(v) for k, v in value.items()}
        return [edit(v) for v in value]
    return apply


def _value(v: Any) -> Callable[[Any], Any]:
    return lambda _full: copy.deepcopy(v)


#: `<Definition>.<field>` → (old-shaped stored value, wrong-typed value), each built from the field's full sample.
#: The old-shaped value leaves out keys a later release added (the dataset-version record's `sensitivity`, `wsp_role`
#: and `wsp_set_kind` arrived on 2026-11-16, `6c0b6ef27`; the rest are keys the current writer sends).
HISTORY: dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    "BlendReportAnswer.resolved": (_drop("flags", "seed"), _set("folds", "five")),
    "BlendReportRow.resolved": (_drop("flags", "seed"), _set("folds", "five")),
    "ProbeDatasetAnswer.column_snapshots": (_each_entry(_drop("n_values", "dtype")),
                                            _each_entry(_set("n_rows", "ten"))),
    "ProbeDatasetAnswer.counts": (_drop("kinds", "unparseable"), _set("positive", "3")),
    "ProbeDatasetAnswer.dataset_version": (_drop("sensitivity", "wsp_role", "wsp_set_kind"),
                                           _set("manifest_rows", "12")),
    "ProbeDatasetAnswer.evidence_spans": (_drop("length_spread", "row_chars"), _set("positives", "4")),
    "ProbeDatasetAnswer.keyword_filter": (_drop("case_sensitive"), _set("terms", "word")),
    "ProbeDatasetAnswer.positive_labels_confirmed": (_drop("origin"), _set("version", "one")),
    "ProbeDatasetAnswer.rule_columns": (_drop("rule_set_digest"), _set("rule_veto", 3)),
    "ProbeDatasetSummaryAnswer.counts": (_drop("kinds", "unparseable"), _set("positive", "3")),
    "ReplicationReportAnswer.results": (_each_entry(_drop("ablation_kl", "swap_success_rate", "coverage")),
                                        _each_entry(_set("ablation_kl", "high"))),
    "SteeringResultResponse.result": (_value({"comparison_id": "cmp_1", "results": [{"feature_idx": 3}]}),
                                      _value("completed")),
    "ValidationManifestAnswer.payload": (_value({"kind": "edge_validation", "edges": [{"source": "a"}]}),
                                         _value(["edge_validation"])),
    "BandReportResponse.boundaries": (_value({"early_end": 4}), _value({"early_end": "four"})),
    "BandReportResponse.profiles": (_value([{"layer": 0}]), _value([0])),
    "InterventionRecordsResponse.records": (_value([{"primitive": "swap"}]), _value(["swap"])),
}

#: The typed ones: their history form is a `__history` copy of a definition that keeps its `required` elsewhere.
TYPED = sorted(k for k in HISTORY if k.split(".")[0] not in {
    "SteeringResultResponse", "ValidationManifestAnswer", "BandReportResponse", "InterventionRecordsResponse"})


def _contract():
    from src.api.answers_contract import DEFAULT_PATH
    from tests.support.answer_middleware import Contract

    return Contract(json.loads(DEFAULT_PATH.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def contract():
    return _contract()


def _errors(contract, definition: str, body: Any) -> list[str]:
    from jsonschema import Draft202012Validator

    from tests.support.strict_formats import STRICT

    validator = Draft202012Validator({"$ref": contract.contract["$id"] + "#/$defs/" + definition},
                                     registry=contract.registry, format_checker=STRICT)
    return [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(body)]


def _body(contract, field: str, edit: Callable[[Any], Any]) -> tuple[str, Any]:
    """The definition's full sample answer, with `field` replaced by `edit` of its own full sample."""
    from tests.support.answer_samples import sample

    definition, prop = field.split(".")
    body = sample({"$ref": "#/$defs/" + definition}, contract.contract)
    full = body.get(prop)
    schema = contract.contract["$defs"][definition]["properties"][prop]
    if full is None or full == {} or full == []:  # a nullable or open field: sample its non-null form
        full = sample(schema, contract.contract)
    if full == {}:  # a map of records (`column_snapshots`): one entry, sampled from the map's value schema
        values = next((s["additionalProperties"] for s in [schema, *schema.get("anyOf", [])]
                       if isinstance(s.get("additionalProperties"), dict)), None)
        if values:
            full = {"text": sample(values, contract.contract)}
    body[prop] = edit(copy.deepcopy(full))
    return definition, body


def _listed(contract) -> list[str]:
    from src.api.answers_contract import HISTORY_EXCEPTION

    description = contract.contract["description"]
    assert HISTORY_EXCEPTION in description
    tail = description.split("History-form fields: ", 1)[1]
    return tail.rstrip(".").split(", ")


def test_the_table_is_the_contracts_own_list(contract):
    """Every field the contract lists has its two cases here, and nothing here is unlisted."""
    assert sorted(_listed(contract)) == sorted(HISTORY)


def test_the_description_states_the_exception_and_names_every_field(contract):
    from src.api.answers_contract import HISTORY_EXCEPTION

    description = contract.contract["description"]
    assert "The one exception to 'required means always sent'" in HISTORY_EXCEPTION
    assert "has no `required` at any depth" in HISTORY_EXCEPTION
    assert description.endswith(HISTORY_EXCEPTION + " History-form fields: " + ", ".join(sorted(HISTORY)) + ".")


@pytest.mark.parametrize("field", TYPED)
def test_a_typed_stored_field_points_at_a_history_copy(contract, field):
    definition, prop = field.split(".")
    schema = json.dumps(contract.contract["$defs"][definition]["properties"][prop])
    refs = [part.split('"')[0] for part in schema.split("#/$defs/")[1:]]
    assert refs and all(r.endswith("__history") for r in refs), (field, refs)
    for ref in refs:
        assert "required" not in json.dumps(contract.contract["$defs"][ref]), ref


@pytest.mark.parametrize("field", sorted(HISTORY))
def test_an_old_shaped_stored_value_validates(contract, field):
    old, _wrong = HISTORY[field]
    definition, body = _body(contract, field, old)
    assert _errors(contract, definition, body) == [], (field, body[field.split(".")[1]])


@pytest.mark.parametrize("field", sorted(HISTORY))
def test_a_wrong_typed_stored_value_fails(contract, field):
    _old, wrong = HISTORY[field]
    definition, body = _body(contract, field, wrong)
    assert _errors(contract, definition, body), (field, body[field.split(".")[1]])


def test_the_full_samples_validate(contract):
    """The base the cases edit is itself a valid answer, so a red above is the edit's."""
    for field in sorted(HISTORY):
        definition, body = _body(contract, field, lambda v: v)
        assert _errors(contract, definition, body) == [], field
