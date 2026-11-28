"""The stripped-tree guard (`tests/support/stripped_tree.py`, W2a 2026-11-25), held to real commits and to the sync's
own workflow.

A check that reads a file the mirror sync deletes skips LOUDLY on the mirror's stripped snapshot and nowhere else. These
tests pin both halves: the predicate says "mirror" for exactly the commit shape the sync builds (a parentless commit
with both trailers), and a missing file in any other tree FAILS. Breaking the predicate either way goes red here:
always-true turns `test_this_checkout_is_not_the_mirror` red on the private tree; always-false leaves the mirror
simulation's guarded reads failing (T1's mirror job).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.support import stripped_tree as st

SHA = "a" * 40


def _repo(tmp: Path, *, parent: bool, source: bool, private_ci: bool) -> Path:
    root = tmp / "repo"
    root.mkdir()
    env = ["-c", "user.name=t", "-c", "user.email=t@t.invalid"]
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "f").write_text("x\n")
    subprocess.run(["git", "add", "f"], cwd=root, check=True)
    if parent:
        subprocess.run(["git", *env, "commit", "-q", "-m", "first"], cwd=root, check=True)
        (root / "f").write_text("y\n")
        subprocess.run(["git", "add", "f"], cwd=root, check=True)
    trailers = []
    if source:
        trailers += ["--trailer", f"Source-Commit: {SHA}"]
    if private_ci:
        trailers += ["--trailer", "Private-CI: passed"]
    subprocess.run(["git", *env, "commit", "-q", "-m", "snapshot", *trailers], cwd=root, check=True)
    return root


@pytest.mark.parametrize(("parent", "source", "private_ci", "expected"), [
    (False, True, True, True),     # the sync's orphan snapshot
    (True, True, True, False),     # trailers, but a commit with a parent
    (False, True, False, False),   # orphan, Source-Commit only
    (False, False, True, False),   # orphan, Private-CI only
    (False, False, False, False),  # a plain orphan commit
    (True, False, False, False),   # an ordinary private commit
])
def test_the_predicate_reads_the_commit_shape(tmp_path, parent, source, private_ci, expected):
    assert st.mirror_snapshot(_repo(tmp_path, parent=parent, source=source, private_ci=private_ci)) is expected


def test_no_repository_is_not_the_mirror(tmp_path):
    assert st.mirror_snapshot(tmp_path) is False


def test_this_checkout_is_not_the_mirror():
    """The wiring control's private half: in this repository (and its CI) nothing may skip as "stripped"."""
    if st.mirror_snapshot():
        pytest.skip("LOUD SKIP: this run IS on the mirror's stripped snapshot")
    assert (st.REPO_ROOT / "0xcc").is_dir() and (st.REPO_ROOT / "CLAUDE.md").is_file()
    assert st.mirror_snapshot() is False


def test_the_removal_rules_are_the_workflows():
    removed, kept = st.removal_rules()
    assert {"0xcc", ".claude", "CLAUDE.md", "backups", "scripts", ".understand-anything"} <= removed
    assert kept == {"schemas", "mcp-contract.md", "legacy-identifiers.public.md"}


@pytest.mark.parametrize(("rel", "stripped"), [
    ("0xcc/tasks/003_FTASKS|SAE_Training.md", True),
    ("CLAUDE.md", True),
    ("scripts/k8s-helpers.sh", True),
    ("docs/miWorkbench_Manual.md", True),
    ("docs/legacy-identifiers.md", True),
    ("docs/legacy-identifiers.public.md", False),
    ("docs/schemas/miforge-wsp-run-v1.json", False),
    ("docs/mcp-contract.md", False),
    ("README.md", False),
    ("manual/docs/core-workflow/sae-training.md", False),
    ("backend/src/main.py", False),
])
def test_what_the_sync_strips(rel, stripped):
    assert st.stripped_by_sync(st.REPO_ROOT / rel) is stripped


def test_a_present_path_passes(tmp_path):
    st.require_unstripped(st.REPO_ROOT / "README.md")


def test_an_absent_stripped_path_skips_only_on_the_mirror(monkeypatch):
    absent = st.REPO_ROOT / "0xcc" / "no-such-file.md"
    monkeypatch.setattr(st, "mirror_snapshot", lambda root=st.REPO_ROOT: True)
    with pytest.raises(pytest.skip.Exception, match="LOUD SKIP"):
        st.require_unstripped(absent)
    monkeypatch.setattr(st, "mirror_snapshot", lambda root=st.REPO_ROOT: False)
    _fails(lambda: st.require_unstripped(absent), "not the mirror")


def test_a_path_the_sync_keeps_fails_even_on_the_mirror(monkeypatch):
    monkeypatch.setattr(st, "mirror_snapshot", lambda root=st.REPO_ROOT: True)
    _fails(lambda: st.require_unstripped(st.REPO_ROOT / "docs" / "schemas" / "no-such-schema.json"), "missing")


def _fails(call, words):
    """A FAILURE, never a skip: `pytest.raises(pytest.fail.Exception)` lets a skip escape, and the test then reads as
    skipped (control S3, 2026-11-25: a guard that skipped everywhere passed this file)."""
    with pytest.raises(BaseException) as caught:
        call()
    assert caught.type is pytest.fail.Exception, f"expected a failure, got {caught.type.__name__}"
    assert words in str(caught.value)
