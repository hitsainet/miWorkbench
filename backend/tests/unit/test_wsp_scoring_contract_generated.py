"""ENH-010 044 task 7.6: `docs/schemas/miworkbench-wsp-scoring-v1.json` is what `scoring_contract()` generates from the
live models and the whole refusal table (the miServer pattern). Regenerate: `python -m
tests.unit.test_wsp_scoring_contract_generated --write` from `backend/`.

Control (0xcc/reviews/044_implementation_controls_2026-11-16.md): add an optional response field without regenerating
→ `test_the_committed_contract_is_the_generated_one` red.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.schemas import wsp_scoring

REPO = Path(__file__).resolve().parents[3]
CONTRACT = REPO / wsp_scoring.CONTRACT_PATH


def test_the_committed_contract_is_the_generated_one():
    assert CONTRACT.read_text(encoding="utf-8") == wsp_scoring.contract_text(), (
        "regenerate: python -m tests.unit.test_wsp_scoring_contract_generated --write")


def test_refusal_table_is_complete():
    """Every code the scorer can answer is in the table (the scorer's `ScoreRefused` refuses any other code at
    raise time), and the published table is exactly that registry, phase-3 and phase-4 codes included."""
    from src.services.wsp_score import ScoreRefused

    published = json.loads(CONTRACT.read_text())["refusals"]
    assert set(published) == set(wsp_scoring.REFUSALS)
    assert {c: v["status"] for c, v in published.items()} == {c: s for c, (s, _) in wsp_scoring.REFUSALS.items()}
    for code in ("candidate_not_the_registered_patch", "control_not_the_seeded_permutation",
                 "control_index_out_of_range", "heldout_set_head_budget_exhausted", "scorer_busy"):
        assert code in published
    assert published["heldout_set_head_budget_exhausted"]["status"] == 409 and published["scorer_busy"]["status"] == 429
    try:
        ScoreRefused("not_a_published_code", "x")
    except KeyError:
        pass
    else:
        raise AssertionError("a refusal outside the published table must not be raisable")


def test_the_contract_has_no_alias_and_forbids_extra():
    contract = json.loads(CONTRACT.read_text())
    for side in ("request", "response"):
        schema = contract[side]
        assert schema["additionalProperties"] is False
        for definition in schema.get("$defs", {}).values():
            if definition.get("type") == "object" and "properties" in definition:
                assert definition.get("additionalProperties") is False, definition.get("title")


if __name__ == "__main__" and "--write" in sys.argv:
    CONTRACT.write_text(wsp_scoring.contract_text(), encoding="utf-8")
    print(f"wrote {CONTRACT}")
