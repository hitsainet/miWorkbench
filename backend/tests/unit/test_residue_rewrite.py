"""The P9 rewrite tool (043 FTASKS 1.6; FTID §3.3, §12). Deleted with the tool at P9 close (FTASKS 10.4).

Every case runs the REAL tool on a temporary git tree that carries this repository's committed export, so the
allowances are the guard's own (nothing here re-types a register fact).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.rebrand import residue_rewrite as rw
from tests.support import legacy_export as le

OLD_STUDIO = "mi" "Studio"
OLD_LLM = "mi" "LLM"


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "tree"
    (root / "docs").mkdir(parents=True)
    for p in le.paths():
        if p.exists():
            shutil.copy(p, root / "docs" / p.name)
    for rel, text in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "fixture"]):
        subprocess.run(["git", "-C", str(root), *args], check=True)
    return root


def _commit(root: Path) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "x"],
                   check=True)


FILES = {
    "backend/src/a.py": f'X = 1  # {OLD_LLM} serves this\nY = "mi' 'studio.probe-definition/v1"  # frozen\n',
    "backend/tests/unit/test_mi" "llm_thing.py": "def test_x():\n    pass\n",
}


def test_review_maps_and_leaves_what_the_export_allows(tmp_path):
    root = _tree(tmp_path, FILES)
    rows = rw.review("comments", root)
    assert [(r.action, r.path, r.row, r.before, r.after) for r in rows] == [
        ("apply", "backend/src/a.py", 1, OLD_LLM, "miServer")]
    # DR-01's kind on line 2 is code, and the identifiers class allows it: no row at all.
    assert rw.review("identifiers", root) == []


def test_apply_rewrites_only_the_reviewed_span(tmp_path):
    root = _tree(tmp_path, FILES)
    counts = rw.apply(rw.review("comments", root), root)
    assert counts == {"applied": 1}
    text = (root / "backend/src/a.py").read_text()
    assert text.startswith("X = 1  # miServer serves this\n")
    assert '"mi' 'studio.probe-definition/v1"' in text  # untouched


def test_apply_refuses_a_dirty_tree(tmp_path):
    root = _tree(tmp_path, FILES)
    rows = rw.review("comments", root)
    (root / "backend/src/a.py").write_text("changed\n")
    with pytest.raises(rw.Refused, match="uncommitted"):
        rw.apply(rows, root)


def test_apply_skips_a_line_that_changed_since_review(tmp_path):
    root = _tree(tmp_path, FILES)
    rows = rw.review("comments", root)
    (root / "backend/src/a.py").write_text(f"X = 2  # {OLD_LLM} serves this\n")
    _commit(root)
    assert rw.apply(rows, root) == {"changed_line": 1}
    assert OLD_LLM in (root / "backend/src/a.py").read_text()


def test_apply_refuses_records_and_applied_migrations(tmp_path):
    root = _tree(tmp_path, FILES)
    for path in ("0xcc/notes.md", "backend/alembic/versions/a1.py", ".claude/context/s.md"):
        row = rw.Row("apply", "comments", path, 1, 2, OLD_LLM, "miServer", OLD_LLM, f"# {OLD_LLM}")
        with pytest.raises(rw.Refused, match="never rewritten"):
            rw.apply([row], root)


def test_apply_skips_a_span_the_export_allows(tmp_path):
    """A review row edited by hand to point at an allowed span (DR-01's kind) is not applied."""
    root = _tree(tmp_path, FILES)
    line = (root / "backend/src/a.py").read_text().split("\n")[1]
    row = rw.Row("apply", "identifiers", "backend/src/a.py", 2, line.index("mi"), "mistudio", "miworkbench",
                 "mistudio.probe-definition", line)
    assert rw.apply([row], root) == {"allowed": 1}


def test_restrict_and_keep_and_unmapped(tmp_path):
    root = _tree(tmp_path, {"backend/src/b.py": f"# {OLD_STUDIO} one\n# renamed from {OLD_STUDIO}\n# mi" "STUDIO odd\n"})
    keep = __import__("re").compile("renamed from")
    rows = rw.review("comments", root, restrict={("backend/src/b.py", 1, 2), ("backend/src/b.py", 2, 15)}, keep=keep)
    assert [r.action for r in rows] == ["apply", "keep", "unmapped"]
    rows = rw.review("comments", root, restrict=set())
    assert [r.action for r in rows][:2] == ["restrict", "restrict"]


def test_file_names_are_renamed_with_git_mv(tmp_path):
    root = _tree(tmp_path, FILES)
    rows = [r for r in rw.review("test_text", root) if r.action == "filename"]
    assert [(r.path, r.after) for r in rows] == [("backend/tests/unit/test_mi" "llm_thing.py",
                                                  "backend/tests/unit/test_miserver_thing.py")]
    rows = [rw.Row("apply", *list(r.__dict__.values())[1:]) for r in rows]
    assert rw.apply(rows, root) == {"renamed": 1}
    assert (root / "backend/tests/unit/test_miserver_thing.py").exists()


def test_review_file_round_trip(tmp_path):
    root = _tree(tmp_path, FILES)
    rows = rw.review("comments", root)
    out = tmp_path / "r.tsv"
    rw.write_review(rows, out)
    assert rw.read_review(out) == rows
    out.write_text("not\ta\treview\n")
    with pytest.raises(rw.Refused, match="not a review file"):
        rw.read_review(out)
