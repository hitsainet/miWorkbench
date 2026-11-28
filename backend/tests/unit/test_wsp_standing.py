"""miForge's FU-7b standing rule, as miWorkbench's import reads it (`src/services/wsp_standing.py`, 2026-11-25).

miWorkbench does not import miForge's code, so the rule is its own copy. It is held to miForge's ONE case table
(`frontend/src/components/panels/wsp/standing.cases.json`, the table miForge's backend and run page are both held to)
through `MIFORGE_REPO` (default: the `miForge` sibling of this checkout). Absent, the comparison skips loudly; under
`MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1` (or the suite-wide switch) it fails. The in-repo cases below run everywhere,
the public mirror included.
"""
from __future__ import annotations

import json

import pytest

from src.services import wsp_standing
from tests.support.cross_repo import REQUIRE_NEW, require_cross_repo
from tests.support.sibling_repos import miforge_repo

CASES = ("frontend", "src", "components", "panels", "wsp", "standing.cases.json")


def _sets(*entries):
    return [{"name": n, "primary": p, "value": v, "heldout_uses": u} for n, p, v, u in entries]


@pytest.mark.parametrize(("verdict", "standing", "confirmed"), [
    ({"value": "improved", "per_set": _sets(("a", True, "improved", 1))}, "improved", ["a"]),
    ({"value": "improved", "per_set": _sets(("a", True, "improved", 18))}, "improved_unconfirmed", []),
    # A fresh NON-primary set (everyday text) guards false alarms; it does not confirm the gain.
    ({"value": "improved", "per_set": _sets(("a", True, "improved", 3), ("e", False, "improved", 1))},
     "improved_unconfirmed", []),
    # A fresh primary set that did not read improved confirms nothing.
    ({"value": "improved", "per_set": _sets(("a", True, "not_significant", 1), ("b", True, "improved", 4))},
     "improved_unconfirmed", []),
    ({"value": "improved", "per_set": _sets(("a", True, "improved", True))}, "improved_unconfirmed", []),
    ({"value": "improved", "per_set": _sets(("a", True, "improved", 1.0))}, "improved", ["a"]),
    ({"value": "not_significant", "per_set": _sets(("a", True, "improved", 1))}, "not_significant", []),
])
def test_the_rule(verdict, standing, confirmed):
    assert wsp_standing.standing(verdict) == standing
    assert wsp_standing.confirmed_by(verdict) == confirmed


def test_it_agrees_with_miforges_case_table():
    path = miforge_repo().joinpath(*CASES)
    if not path.is_file():
        message = (f"miForge's standing case table is not at {path} (set MIFORGE_REPO to a miForge checkout). "
                   f"Skipping is expected on the public mirror; set {REQUIRE_NEW}=1 to make this a failure.")
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    table = json.loads(path.read_text("utf-8"))["cases"]
    assert len(table) >= 10, "the table holds too few cases: it is not miForge's"
    for case in table:
        assert wsp_standing.standing(case["document"]) == case["standing"], case["case"]
        assert wsp_standing.confirmed_by(case["document"]) == case["confirmed_by"], case["case"]
