"""miServer's answer contract, `miserver.answer-contract/v1`, is VENDORED byte-identical, never generated (046
FR-046.25, MW12, BR-5).

miServer OWNS `docs/schemas/miserver-answers-v1.json`. miWorkbench carries ONE copy,
`backend/src/mcp_server/vendored/miserver-answers-v1.json` (the path miServer's FR-044.17 `CONSUMERS["miWorkbench"]`
names, TD13). There is no `docs/schemas/` copy: the file is an input of `scripts/generate_output_schemas.py` only, and
the MCP pod does not read it.

TAKEN FROM: miServer `origin/main` at commit `05672754` (revision 2, `content_sha256` 49e883aa…; ENH-011 X-2, pushed
2026-11-28 05:39 ET), re-vendored 2026-11-28 with 046-2's rebase. Revision 2 moves only `external`'s circuit and cluster
definition pins to the bytes miWorkbench publishes since X-2 (`7fad8daa…`, `93d7abf8…`), so `docs/schemas/pinned/`
was deleted with it. History: revision 1 from `origin/main` `4a09a469` (044-1, train S1, live 2026-11-25 22:42 ET),
first copied from branch `feat/044-typed-answers` at `a8ff9281` (046 FTASKS 0.6: byte-identical, `a9178d8a…`).

Two checks:
* the cross-repo pin, through `MISERVER_REPO` (default: the `miServer` sibling), in three states: present → the bytes
  must match; absent → a LOUD skip naming the path and the switch; absent with `MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1`
  (or `MISTUDIO_…`) → fail.  (p9: keep, the suite-wide switch)
* the OFFLINE digest: the copy's own `content_sha256`, recomputed as miServer's TD11 defines it (sha256 of the rendered
  document without `revision` and `content_sha256`; sorted keys, two-space indent, `ensure_ascii=False`, a trailing
  newline). It needs no sibling, so the public mirror still proves the copy is intact.

Controls (046 FTASKS 2.1, M32; 0xcc/reviews/046_controls_2026-11-25.md):
  edit one byte of the vendored copy             → test_the_offline_digest_matches and test_it_is_byte_identical_to_miServer red
  remove one field from a planted sibling copy   → test_a_sibling_copy_that_differs_fails (built in)
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from tests.support.cross_repo import REQUIRE_NEW, REQUIRE_SWITCH, require_cross_repo
from tests.support.sibling_repos import miserver_repo

BACKEND = pathlib.Path(__file__).resolve().parents[2]
NAME = "miserver-answers-v1.json"
COPY = BACKEND / "src" / "mcp_server" / "vendored" / NAME
FLAG = REQUIRE_NEW
#: The miServer commit and ref the copy is pinned to (046 FTASKS 2.1; 0.6: 044-1; 2026-11-28: X-2's revision 2).
SOURCE_COMMIT = "05672754"
SOURCE_REF = "origin/main"
UNDIGESTED = ("revision", "content_sha256")


def _render(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _digest(doc: dict) -> str:
    return hashlib.sha256(_render({k: v for k, v in doc.items() if k not in UNDIGESTED}).encode("utf-8")).hexdigest()


def _theirs(root: pathlib.Path) -> pathlib.Path:
    return root / "docs" / "schemas" / NAME


def _compare_with_sibling(root: pathlib.Path) -> None:
    theirs = _theirs(root)
    if not theirs.is_file():
        message = (
            f"miServer's {NAME} is not at {theirs} (set MISERVER_REPO to a miServer checkout; this copy was taken "
            f"from {SOURCE_REF} at {SOURCE_COMMIT}). Skipping is expected on the public mirror; set {FLAG}=1 to make "
            "this a failure."
        )
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    assert COPY.read_bytes() == theirs.read_bytes(), (
        f"miWorkbench's vendored {NAME} differs from miServer's at {theirs}. Copy miServer's file INTO {COPY}, update "
        "SOURCE_COMMIT here, and regenerate output_schemas.json (python -m scripts.generate_output_schemas --write): "
        "never edit the copy by hand; miServer owns the contract."
    )


def test_the_copy_is_the_contract_it_claims():
    doc = json.loads(COPY.read_text(encoding="utf-8"))
    assert doc["$id"] == "https://hitsai.net/schemas/" + NAME
    assert doc["kind"] == "miserver.answer-contract/v1"
    assert doc["revision"] >= 1
    assert COPY.read_text(encoding="utf-8") == _render(doc), "the copy is not in the committed byte form"


def test_the_offline_digest_matches():
    """Needs no sibling: on the public mirror this is what proves the copy is intact."""
    doc = json.loads(COPY.read_text(encoding="utf-8"))
    assert _digest(doc) == doc["content_sha256"], (
        f"{COPY}'s content no longer matches its own content_sha256: it was edited by hand. Re-copy miServer's file.")


def test_it_is_byte_identical_to_miServer():
    # Skips on the public mirror by design (no sibling there) — see the module docstring.
    _compare_with_sibling(miserver_repo())


def test_an_absent_sibling_SKIPS_loudly(tmp_path, monkeypatch):
    for switch in REQUIRE_SWITCH:
        monkeypatch.delenv(switch, raising=False)
    with pytest.raises(pytest.skip.Exception) as skipped:
        _compare_with_sibling(tmp_path)
    assert str(_theirs(tmp_path)) in str(skipped.value) and FLAG in str(skipped.value)
    assert SOURCE_COMMIT in str(skipped.value)


def test_an_absent_sibling_FAILS_under_the_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(pytest.fail.Exception):
        _compare_with_sibling(tmp_path)


def test_a_sibling_copy_that_differs_fails(tmp_path):
    """FR-046.25 NC1, built in: a planted sibling copy with one field removed fails the pin."""
    doc = json.loads(COPY.read_text(encoding="utf-8"))
    key = sorted(doc["responses"])[0]
    del doc["responses"][key]
    planted = _theirs(tmp_path)
    planted.parent.mkdir(parents=True)
    planted.write_text(_render(doc), encoding="utf-8")
    with pytest.raises(AssertionError, match="differs from miServer's"):
        _compare_with_sibling(tmp_path)


def test_the_mirror_keeps_the_vendored_copy():
    """The copy is under `backend/src/`, which the mirror ships; the strip step must not name it."""
    workflow = (BACKEND.parent / ".github" / "workflows" / "sync-to-clean.yml").read_text(encoding="utf-8")
    assert "mcp_server/vendored" not in workflow and NAME not in workflow


def test_the_copy_is_the_one_on_miServers_main_at_its_source_commit():
    """046 FTASKS 0.6: the pinned commit's file, read from miServer's own history, is these bytes. Under the switch an
    absent sibling (or a sibling that lacks the commit) fails; otherwise it skips loudly."""
    import subprocess

    root = miserver_repo()
    shown = subprocess.run(["git", "-C", str(root), "show", f"{SOURCE_COMMIT}:docs/schemas/{NAME}"],
                           capture_output=True, timeout=60)
    if shown.returncode != 0:
        message = f"miServer at {root} cannot show {SOURCE_COMMIT}:docs/schemas/{NAME} (set MISERVER_REPO)"
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    assert shown.stdout == COPY.read_bytes(), f"the vendored copy is not miServer's file at {SOURCE_COMMIT}"
