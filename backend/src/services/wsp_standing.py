"""A WSP verdict's standing, read from the run document's own fields (miForge 018 FU-7b, operator decision
2026-11-24; applied at miWorkbench's import 2026-11-25).

A held-out set scored again and again stops being held out. So miForge reads an `improved` verdict two ways:

- **`improved`**: at least one PRIMARY held-out set reads `improved` in this verdict AND was scored for the first time
  by it (its `heldout_uses` after the evaluation is 1);
- **`improved_unconfirmed`**: every primary set that read `improved` had been scored before. Do not act on it (import
  it, build on it, call the probe better) without a confirmation on a held-out set never scored before.

Every other value stands as it is. Only the verdict's `value` and each `per_set` entry's `primary`, `value` and
`heldout_uses` are read, all in `miforge-wsp-run-v1.json`, so no contract field is involved.

This is miWorkbench's own copy of miForge's rule (`backend/src/wsp/standing.py` there): apps never import each other.
`tests/unit/test_wsp_standing.py` holds it to miForge's case table (`standing.cases.json`) through `MIFORGE_REPO`, so a
change to the rule in miForge goes red here.
"""
from __future__ import annotations

from typing import Any, Dict, List

UNCONFIRMED = "improved_unconfirmed"


def confirmed_by(verdict: Dict[str, Any]) -> List[str]:
    """The primary sets that confirm an `improved` verdict (read `improved`, `heldout_uses == 1`); empty otherwise."""
    if verdict.get("value") != "improved":
        return []
    out = []
    for entry in verdict.get("per_set") or []:
        uses = entry.get("heldout_uses")
        # A count, never a bool (True == 1 in Python); 1.0 is 1, as JSON may carry it.
        fresh = isinstance(uses, (int, float)) and not isinstance(uses, bool) and uses == 1
        if entry.get("primary") is True and entry.get("value") == "improved" and fresh:
            out.append(str(entry.get("name")))
    return out


def standing(verdict: Dict[str, Any]) -> str:
    """`improved` only when confirmed, else `improved_unconfirmed`; any other value as it is."""
    value = str(verdict.get("value"))
    if value != "improved":
        return value
    return "improved" if confirmed_by(verdict) else UNCONFIRMED
