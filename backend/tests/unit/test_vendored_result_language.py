"""miAssist's `mistudio.result-language/v1` is VENDORED byte-identical, never generated (045 FR-6; CHAIN_PLAN A6).

miAssist OWNS `docs/schemas/mistudio-result-language-v1.json` (published at miAssist `b1f30e7`, "publish  (p9: keep, the suite contract's file name)
mistudio.result-language/v1 and mistudio.playbook/v1"). miWorkbench carries two copies, both copied from miAssist:

* `docs/schemas/…` — the vendored contract, compared with miAssist's file;
* `backend/src/mcp_server/vendored/…` — the PACKAGE copy the MCP pod reads as data (not `src/schemas/vendored/`: the
  MCP pod must not import the backend schema package; memory `watch-test-only-pushes-too`).

Three states for the cross-repo comparison, through `MIASSIST_REPO` (default: the `miAssist` sibling):
present → the bytes must match; absent (or without the file) → a LOUD skip naming the path and the switch; absent with
`MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1` (or `MISTUDIO_…`) → fail.  (p9: keep, the suite-wide switch)

⚠ ON THE PUBLIC MIRROR THE CROSS-REPO TEST SKIPS BY DESIGN (no miAssist checkout there); the in-repo identity test
still runs, and `sync-to-clean.yml` keeps `docs/schemas/`.

Controls (045 FTASKS 7.3; 0xcc/reviews/045_implementation_controls_2026-11-16.md):
  change one byte of the package copy → test_the_package_copy_equals_the_docs_copy and
    test_it_is_byte_identical_to_miAssist red;
  `MIASSIST_REPO` at an empty directory with the switch → test_an_absent_sibling_FAILS_under_the_flag asserts the
    failure path (and the live test fails rather than skips).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tests.support.cross_repo import REQUIRE_NEW, REQUIRE_SWITCH, require_cross_repo
from tests.support.sibling_repos import miassist_repo

REPO = pathlib.Path(__file__).resolve().parents[3]
BACKEND = REPO / "backend"
NAME = "mistudio-result-language-v1.json"
DOCS_COPY = REPO / "docs" / "schemas" / NAME
PACKAGE_COPY = BACKEND / "src" / "mcp_server" / "vendored" / NAME
FLAG = REQUIRE_NEW
#: The miAssist commit the copies were taken from (045 FTASKS 0.2, 7.1).
SOURCE_COMMIT = "b1f30e7"


def _theirs(root: pathlib.Path) -> pathlib.Path:
    return root / "docs" / "schemas" / NAME


def _compare_with_sibling(root: pathlib.Path) -> None:
    theirs = _theirs(root)
    if not theirs.is_file():
        message = (
            f"miAssist's {NAME} is not at {theirs} (set MIASSIST_REPO to a miAssist checkout). Skipping is expected "
            f"on the public mirror; set {FLAG}=1 to make this a failure."
        )
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    assert PACKAGE_COPY.read_bytes() == theirs.read_bytes(), (
        f"miWorkbench's vendored {NAME} differs from miAssist's. Copy miAssist's file FROM {theirs} INTO {DOCS_COPY} "
        f"and {PACKAGE_COPY} — never edit or regenerate it here; miAssist owns the contract."
    )


def test_the_package_copy_equals_the_docs_copy():
    assert DOCS_COPY.is_file() and PACKAGE_COPY.is_file()
    assert DOCS_COPY.read_bytes() == PACKAGE_COPY.read_bytes()
    assert json.loads(PACKAGE_COPY.read_text())["title"] == "miStudio result language, v1"


def test_it_is_byte_identical_to_miAssist():
    # Skips on the public mirror by design (no sibling there) — see the module docstring.
    _compare_with_sibling(miassist_repo())


def test_an_absent_sibling_SKIPS_loudly(tmp_path, monkeypatch):
    for switch in REQUIRE_SWITCH:
        monkeypatch.delenv(switch, raising=False)
    with pytest.raises(pytest.skip.Exception) as skipped:
        _compare_with_sibling(tmp_path)
    assert str(_theirs(tmp_path)) in str(skipped.value) and FLAG in str(skipped.value)


def test_an_absent_sibling_FAILS_under_the_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(pytest.fail.Exception):
        _compare_with_sibling(tmp_path)


def test_the_mirror_keeps_docs_schemas():
    workflow = (REPO / ".github" / "workflows" / "sync-to-clean.yml").read_text()
    assert "! -name schemas" in workflow
    assert "for p in docs/schemas" in workflow
