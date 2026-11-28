"""The per-router HTTP-case modules, together, cover every covered route (046 FTASKS 9.3; FR-046.17 as amended).

Each `test_<router>_answers.py` beside this file states its `ROUTER` and its `CASES`. Their union must EQUAL the
committed contract's covered set (`responses` and `excluded`), each router must have exactly one module, and no two
modules may share a key. A router with no module is named.

Control M34 (record): one router's module deleted → this test red, naming the router.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.support import covered_route_cases as crc

HERE = Path(__file__).resolve().parent


def _modules() -> dict[str, set[str]]:
    """module file name → (its ROUTER, the keys of its CASES), each module loaded from its file."""
    out = {}
    for path in sorted(HERE.glob("test_*_answers.py")):
        spec = importlib.util.spec_from_file_location(f"_covered_union_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out[path.name] = (module.ROUTER, {c.key for c in module.CASES})
    return out


def test_every_router_has_one_module_and_their_union_is_the_covered_set():
    modules = _modules()
    by_router: dict[str, list[str]] = {}
    for name, (router, _keys) in modules.items():
        by_router.setdefault(router, []).append(name)
    assert not [r for r, names in by_router.items() if len(names) > 1], by_router
    for name, (router, _keys) in modules.items():
        assert name == f"test_{Path(router).stem}_answers.py", (name, router)

    routers = crc.routers()
    missing = sorted(set(routers) - set(by_router))
    assert not missing, f"routers with covered routes and no HTTP-case module: {missing}"
    assert not sorted(set(by_router) - set(routers)), "a module for a router with no covered route"

    union: set[str] = set()
    for name, (_router, keys) in sorted(modules.items()):
        assert not (union & keys), f"{name} repeats keys of another module: {sorted(union & keys)}"
        union |= keys
    # Both sides come from the authority: the contract file's covered set, and the walker's (the live tools' routes
    # plus the sibling routes), which does not read the file (review round 1, finding 5).
    for label, covered in (("contract", set(crc.covered_keys())), ("walker", set(crc.walker_covered_keys()))):
        assert union == covered, (f"{label}: covered keys no module runs: {sorted(covered - union)}; "
                                  f"keys outside the covered set: {sorted(union - covered)}")
    assert len(modules) == 30  # the 29 routers of FPRD §6 plus main.py (FTASKS 9.2)


#: The empty-list ratchet's recorded size (review round 1, finding 8): it may only shrink. Lower these as rows are
#: seeded; raising them is red.
EMPTY_LIST_KEYS, EMPTY_LIST_PATHS = 85, 165


def test_the_empty_list_ratchet_names_only_covered_json_routes_and_only_shrinks():
    """Round 2, finding 5: each recorded key is a covered JSON route, so its own operator case compares it exactly
    (a key no case observes cannot hide there), and the recorded totals never grow."""
    import json

    recorded = json.loads(crc.EMPTY_LISTS_PATH.read_text(encoding="utf-8"))
    responses = set(crc._contract()["responses"])
    assert not sorted(set(recorded) - responses), sorted(set(recorded) - responses)
    assert all(paths and paths == sorted(set(paths)) for paths in recorded.values()), "sorted, distinct, non-empty"
    assert len(recorded) <= EMPTY_LIST_KEYS, len(recorded)
    assert sum(len(p) for p in recorded.values()) <= EMPTY_LIST_PATHS, sum(len(p) for p in recorded.values())
