"""The P3 MCP cut, held on the live registry (042 FR-042.15; FTDD §6.6; FTID §3.9; FTASKS 12.3, 12.28 – 12.31, 18.12).

`scripts/rebrand/mcp_cut.py` wrote `tests/support/mcp_rename_map.json` from the LIVE registry of the pre-cut tree and
applied it at P3; the script was deleted at ENH-009 P7 (FTASKS 18.12) and the map stays as history. This file holds
the cut to that map, reading the live registry of THIS tree (every category on, every product URL set —
`next_calls.live_input_schemas()`), never a module import:

(a) no tool is named `millm_*`; (b) the `miserver_*` names equal the map's values exactly; (c) the total equals  (p9: keep, retired-name test)
the map's; (d) no name is registered twice and no old name beside its new one; (e) `EXPECTED_CALLS` and
`_CALLER_ASSERTION_EXEMPT` carry only new names with the payloads they had before the cut. (The script's own tests,
and the cut-time check that no test file's assertion count dropped, went with the script.)

Controls (`0xcc/reviews/042_implementation_controls_*.md`): K-12.29 drop `miserver_sensing` from
`MISERVER_CATEGORY_MODULES` → (b) red; K-12.30 register one module twice → (d) red; K-12.31 leave one function
`millm_<x>` unrenamed → (a) red.  (p9: keep, retired-name test)
"""
from __future__ import annotations

import pytest

from tests.support import mcp_cut_history as history
from tests.support.next_calls import live_input_schemas


@pytest.fixture(scope="module")
def cut_map() -> dict:
    assert history.MAP_PATH.is_file(), "tests/support/mcp_rename_map.json (the P3 cut's map, kept as history) is missing"
    return history.load_map()


# ── 12.3: the cut, read from the live registry ─────────────────────────────────────────────────
def test_no_millm_tool_registered():  # p9: keep, retired-name test
    live = set(live_input_schemas())
    assert live, "the live registry is empty: the check would read nothing"
    assert sorted(n for n in live if n.startswith("millm_")) == []
    assert "mistudio_howto" not in live and "miworkbench_howto" in live


def test_miserver_names_equal_map_values(cut_map):
    live = {n for n in live_input_schemas() if n.startswith("miserver_")}
    expected = {v for v in cut_map["tools"].values() if v.startswith("miserver_")}
    assert expected, "the map renames no millm_* tool: the check would read nothing"
    # A miserver_* tool added after the cut is a NEW name (test_tools_added_after_the_cut_are_new_names).
    added = {n for n in history.ADDED_AFTER_THE_CUT if n.startswith("miserver_")}
    assert live == expected | added, (sorted((expected | added) - live), sorted(live - expected - added))


def test_total_equals_map_total(cut_map):
    live = set(live_input_schemas())
    assert set(history.ADDED_AFTER_THE_CUT) <= live, "a tool recorded as added after the cut is not registered"
    assert len(live) == cut_map["total"] + len(history.ADDED_AFTER_THE_CUT)


def test_tools_added_after_the_cut_are_new_names(cut_map):
    """An addition is a NEW name: never one the cut renamed from or to (that would hide a rename behind it)."""
    named = set(cut_map["tools"]) | set(cut_map["tools"].values())
    assert sorted(set(history.ADDED_AFTER_THE_CUT) & named) == []


def test_no_duplicates_and_no_old_beside_new(cut_map):
    names = history.registered_names_with_duplicates()
    assert len(set(names)) == cut_map["total"] + len(history.ADDED_AFTER_THE_CUT), (
        "the counting build registered a different set")
    assert history.duplicate_problems(names, cut_map) == []


def test_expected_calls_new_names_same_payloads_and_counts(cut_map):
    import tests.unit.test_reachability as reach

    tools = cut_map["tools"]
    before = cut_map["expected_calls"]
    assert before, "the map recorded no payloads: the check would read nothing"
    assert not [k for k in reach.EXPECTED_CALLS if k.startswith("millm_")]
    assert not [k for k in reach._CALLER_ASSERTION_EXEMPT if k.startswith("millm_")]
    assert len(set(reach.EXPECTED_CALLS) - set(history.ADDED_AFTER_THE_CUT)) == len(before), (
        "a payload-asserted tool was added or dropped by the cut")
    # The one intended payload change (the actor id, FR-042.16) is read back through the map's record of it.
    renames = cut_map["payload_renames"]
    assert renames == {"agent:mistudio-mcp": "agent:miworkbench-mcp"}
    changed = [old for old, d in before.items()
               if history.digest(reach.EXPECTED_CALLS.get(tools.get(old, old)), renames) != d]
    assert changed == [], f"payloads changed by the cut: {changed}"
    # A tool added after the cut may be exempt with its reason (it is asserted in its own file), as it may be in
    # EXPECTED_CALLS above; the cut's own exemptions are unchanged.
    assert sorted(set(reach._CALLER_ASSERTION_EXEMPT) - set(history.ADDED_AFTER_THE_CUT)) == sorted(
        tools.get(o, o) for o in cut_map["caller_exempt"])
    live = set(live_input_schemas())
    assert live == set(reach.EXPECTED_CALLS) | set(reach._CALLER_ASSERTION_EXEMPT)


