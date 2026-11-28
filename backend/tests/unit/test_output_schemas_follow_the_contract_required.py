"""A native tool's output schema requires no more than miWorkbench's answer contract says its route always sends
(046 P7; FR-046.19's `required` means "always sent", SH7).

Until FTASKS 10.1 the generator types native tools from the live OpenAPI (WT5), not from the committed contract, so it
must apply the writer's own rules: a field marked `x-may-be-absent` is not required, and a stored-answer field
(`x-history-form`) is checked with no `required` beneath it. The generator once required every property, so a tool
whose route left a may-be-absent key out (`create_behaviour`'s warnings' `next[].needs`) had its real answer refused
by the SDK client; that is the production MCP client's check too.

Control (record `0xcc/reviews/046_controls_2026-11-25.md`): put back "every property required" in
`scripts/generate_output_schemas.native_source` and regenerate → both tests red.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _tools() -> dict:
    return json.loads((REPO / "backend" / "src" / "mcp_server" / "output_schemas.json").read_text())["tools"]


def _contract_defs() -> dict:
    return json.loads((REPO / "docs" / "schemas" / "miworkbench-answers-v1.json").read_text())["$defs"]


def test_no_writer_marker_reaches_a_tool_schema():
    leaked = sorted(name for name, schema in _tools().items()
                    if "x-may-be-absent" in json.dumps(schema) or "x-history-form" in json.dumps(schema))
    assert not leaked, leaked


def test_a_tool_never_requires_a_field_the_contract_says_may_be_absent():
    contract = _contract_defs()
    over: list[str] = []
    compared = 0
    for tool, schema in _tools().items():
        for name, definition in (schema.get("$defs") or {}).items():
            theirs = contract.get(name)
            if not isinstance(theirs, dict) or not isinstance(definition, dict):
                continue
            if set(theirs.get("properties") or {}) != set(definition.get("properties") or {}):
                continue  # not the same model (a relay's definition of the same name, say)
            compared += 1
            extra = set(definition.get("required") or []) - set(theirs.get("required") or [])
            if extra:
                over.append(f"{tool}: {name} requires {sorted(extra)}")
    assert compared > 200, compared  # the harness reaches the native typed tools' definitions
    assert not over, over[:10]
