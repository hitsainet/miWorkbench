"""045 FR-17: proxied tools carry their source app's hints, compared only where the two tools do the same thing.

A `dataworks_*` relay's `readOnlyHint` and `destructiveHint` must equal those of the miDataworks tool of the same
name in miDataworks' published classification (`docs/mcp-annotations.json`, version 2, miDataworks 022) ONLY when the
relay's call set equals the source tool's `calls` set (A30 as settled in round 2; R1A-19; R2A-6). Otherwise the relay
is classified by A13 on its own calls (FR-3's rules, `test_tool_classes.py`), and its row's `why` names both sets.
The relay's calls are its `test_reachability.EXPECTED_CALLS` entry; a concrete segment there matches a template's
`{name}` segment by segment. The set of compared relays is pinned, so a comparison that silently stops is red.

Absent miDataworks (the public mirror) → loud skip; absent under the require-cross-repo switch → fail.

Controls (record `0xcc/reviews/045_implementation_controls_2026-11-16.md`):
  flip `dataworks_delete_version` to write              → red
  flip `dataworks_list_datasets` (equal sets) to write   → red
  give `dataworks_get_version_manifest` its source's hint → red (subset: classified by its own GET)
  replace set equality with "one call in common"         → the compared-set pin red
  skip the comparison step                               → the compared-set pin red
"""
from __future__ import annotations

import json

import pytest

from tests.support.cross_repo import require_cross_repo
from tests.support.sibling_repos import midataworks_repo

#: The relays whose call set equals the source's, so whose hints are compared (pinned, R2A-6).
COMPARED = frozenset({
    "dataworks_build_version_files", "dataworks_cancel_job", "dataworks_delete_version",
    "dataworks_get_detector_send", "dataworks_get_detector_set", "dataworks_get_probe_results",
    "dataworks_get_version", "dataworks_job_status", "dataworks_list_datasets", "dataworks_list_detector_sets",
    "dataworks_publish_version", "dataworks_refresh_probe_results", "dataworks_resume_detector_send",
    "dataworks_start_label_run", "dataworks_reread_behaviour_definition",
    "dataworks_get_approval_status",
})


def _annotations() -> dict:
    path = midataworks_repo() / "docs" / "mcp-annotations.json"
    if not path.is_file():
        message = (f"miDataworks' published classification {path} is absent (set MIDATAWORKS_REPO); "
                   "FR-17 compared nothing")
        if require_cross_repo():
            pytest.fail(message + " and the require-cross-repo switch is on")
        pytest.skip(message)
    data = json.loads(path.read_text())
    assert data.get("version") == 2, f"{path} is version {data.get('version')}, FR-17 needs version 2 (with calls)"
    return data["tools"]


def _matches(concrete: str, template: str) -> bool:
    a, b = concrete.strip("/").split("/"), template.strip("/").split("/")
    return len(a) == len(b) and all(x == y or (y.startswith("{") and y.endswith("}")) for x, y in zip(a, b))


def same_calls(relay: set[tuple[str, str]], source: set[tuple[str, str]]) -> bool:
    """Set equality under template matching: every relay call is one of the source's, and every source call is
    made by the relay. Never "one call in common" (that would mark a pure-GET relay write)."""
    if not relay or not source:
        return False
    covered = {s for s in source for r in relay if r[0] == s[0] and _matches(r[1], s[1])}
    every_relay = all(any(r[0] == s[0] and _matches(r[1], s[1]) for s in source) for r in relay)
    return every_relay and covered == source


def _relay_calls(name: str) -> set[tuple[str, str]]:
    from tests.unit.test_reachability import EXPECTED_CALLS

    method, path, _kwargs, _payload = EXPECTED_CALLS[name]
    return {(method, path)}


def compared_relays(source_tools: dict) -> dict[str, bool]:
    from src.mcp_server.tool_classes import TOOL_CLASSES

    out = {}
    for name in sorted(n for n in TOOL_CLASSES if n.startswith("dataworks_")):
        src = source_tools.get(name)
        if src is None:
            continue
        source = {(c["method"], c["path"]) for c in src.get("calls") or []}
        out[name] = same_calls(_relay_calls(name), source)
    return out


def test_dataworks_relays_match_midataworks():
    from src.mcp_server.tool_classes import TOOL_CLASSES

    source_tools = _annotations()
    compared = compared_relays(source_tools)
    assert len(compared) == 19, sorted(compared)
    wrong = []
    for name, equal in compared.items():
        row = TOOL_CLASSES[name]
        if equal:
            src = source_tools[name]
            mine = (row.kind == "read", row.kind == "destructive")
            if mine != (src["readOnlyHint"], src["destructiveHint"]):
                wrong.append((name, mine, (src["readOnlyHint"], src["destructiveHint"])))
    assert not wrong, f"relays whose hints differ from their source's: {wrong}"
    assert frozenset(n for n, eq in compared.items() if eq) == COMPARED, (
        "the set of compared relays moved", sorted(set(n for n, eq in compared.items() if eq) ^ COMPARED))


def test_a_subset_relay_is_classified_by_its_own_calls():
    """R2A-6: `dataworks_get_version_manifest` makes one GET; its source also builds files when none exist. The call
    sets differ, so the relay is read-only by its own call."""
    from src.mcp_server.tool_classes import TOOL_CLASSES

    source_tools = _annotations()
    assert not compared_relays(source_tools)["dataworks_get_version_manifest"]
    assert TOOL_CLASSES["dataworks_get_version_manifest"].kind == "read"


def test_the_matcher_is_set_equality_not_overlap():
    a = {("GET", "/api/v1/versions/v1/handoff-manifest")}
    b = {("GET", "/api/v1/versions/{version_id}/handoff-manifest"),
         ("POST", "/api/v1/versions/{version_id}/publish-builds")}
    assert not same_calls(a, b)
    assert same_calls(a, {("GET", "/api/v1/versions/{version_id}/handoff-manifest")})


def test_miserver_rows_follow_the_rules():
    """miServer publishes no table: its 98 rows follow A13 (FR-3's rules, checked for every row in
    `test_tool_classes.py`); here, that every `miserver_*` row exists and the gated ones are not read-only."""
    from src.mcp_server.tool_classes import TOOL_CLASSES

    rows = {n: r for n, r in TOOL_CLASSES.items() if n.startswith("miserver_")}
    assert len(rows) == 98
    assert all(r.kind != "read" for r in rows.values() if r.approval)
    assert rows["miserver_generate"].kind == "write" and rows["miserver_delete_model"].kind == "destructive"
