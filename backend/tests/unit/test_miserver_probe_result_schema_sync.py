"""`docs/schemas/probe-result-v1.json` is miServer's `mi.probe.result/v1`, vendored BYTE-IDENTICALLY
(miServer 038 FR-38.28, FTASKS 6.1; HANDOFF_PLAN §6 step 4).

miServer OWNS this schema: it is generated there from `millm/api/schemas/probe_result.py` and published
at `docs/schemas/probe-result-v1.json`. miWorkbench is a consumer — its MCP tools pass the contract
objects through (`miserver_generate`, `miserver_score_chat`, `miserver_score_probes`, `miserver_probe_events`,
`miserver_get_probe_event`), and `test_mcp_miserver_probe_results.py` validates its miServer stub's objects
against THIS copy, so the stub cannot drift from the contract unnoticed.

NEVER REGENERATE OR HAND-EDIT IT HERE. When miServer moves the schema, copy its file across unchanged
(`cp ../miServer/docs/schemas/probe-result-v1.json docs/schemas/`) in the same release. The pin is red
in whichever repo moves second, by design: from miServer's S1 push until this copy lands it was red on
any machine with miServer checked out beside this repo.

Skips LOUDLY when the sibling is absent (the public mirror has no sibling), and FAILS under
`MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1` — an unverified pin is not a verified one.

Controls (038 S3-M, recorded in `0xcc/reviews/038_comiStudio_controls_2026-10-23.md`):  (p9: keep, a record's name)
  V1 append one byte to the vendored copy → `test_byte_identical_to_miservers_published_schema` red
  V2 point the sibling at an empty directory with the env flag set → red, not skipped
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.support.sibling_repos import REPO_ROOT, miserver_repo
from tests.support.cross_repo import require_cross_repo

VENDORED = REPO_ROOT / "docs" / "schemas" / "probe-result-v1.json"
#: The contract's identity, stated once so a re-vendor of a DIFFERENT document is caught even on
#: the public mirror, where the byte pin skips.
TITLE = "mi.probe.result/v1"
SCHEMA_ID = "https://hitsai.net/schemas/millm/probe-result-v1.json"


def _theirs() -> Path:
    return miserver_repo() / "docs" / "schemas" / "probe-result-v1.json"


def test_byte_identical_to_miservers_published_schema():
    theirs = _theirs()
    if not theirs.exists():
        if require_cross_repo():
            pytest.fail(
                f"{theirs} is absent and MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1. miServer publishes "
                "probe-result-v1.json (038 FR-38.28); set MISERVER_REPO to its checkout."
            )
        pytest.skip(f"miServer's published schema is not present ({theirs}); set MISERVER_REPO to check it")
    assert VENDORED.read_bytes() == theirs.read_bytes(), (
        "docs/schemas/probe-result-v1.json differs from miServer's published copy. miServer owns it: copy "
        "its file across byte-identically in the same release (never regenerate or edit it here)."
    )


def test_the_vendored_copy_is_the_probe_result_contract():
    """Runs everywhere, the mirror included: the file is present and is the document it claims."""
    assert VENDORED.is_file(), f"vendored schema missing: {VENDORED}"
    schema = json.loads(VENDORED.read_text(encoding="utf-8"))
    assert schema["title"] == TITLE
    assert schema["$id"] == SCHEMA_ID
    assert schema["discriminator"]["propertyName"] == "kind"
    # ENH-006 G19 (miServer contract v1.26): `kind: "behaviour"`, the served behaviour verdict.
    assert set(schema["discriminator"]["mapping"]) == {"probe", "rule", "behaviour"}
    # The fields the miWorkbench tools promise an agent reaches it unchanged (FR-38.27).
    props = schema["$defs"]["ProbeResult"]["properties"]
    for field in ("result_id", "trace_id", "outcome", "reason", "decision", "detector_digest",
                  "persistence", "window", "verdict"):
        assert field in props, f"{field} is not in the vendored ProbeResult"
