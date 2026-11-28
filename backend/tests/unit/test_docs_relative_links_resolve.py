"""Every relative Markdown link under `docs/` and in each README resolves to a tracked file (043 FR-043.12;
FTASKS 4.4). ENH-009 P9 renamed `docs/miStudio_Manual.md`; a link left on the old name would read as a working  (p9: keep, brand-guard plant or rule)
reference while pointing nowhere. Links are read from the tracked files (`git ls-files`), never listed here.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath

from tests.support import legacy_export as le

ROOT = le.REPO_ROOT
#: `[text](target)` with no scheme, no anchor-only and no mail link; the target's `#fragment` is dropped.
LINK = re.compile(r"\[[^\]]*\]\((?!https?:|mailto:|#|<)([^)\s]+)\)")
FENCE = re.compile(r"^\s*(```|~~~)")


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    return {p for p in out.split("\0") if p}


def _docs(tracked: set[str]) -> list[str]:
    return sorted(p for p in tracked if p.endswith(".md") and not p.startswith(("0xcc/", ".claude/"))
                  and (p.startswith("docs/") or PurePosixPath(p).name.startswith("README")))


def broken_links(root: Path = ROOT, tracked: set[str] | None = None) -> list[str]:
    tracked = _tracked() if tracked is None else tracked
    dirs = {str(PurePosixPath(p).parent) for p in tracked}
    bad = []
    for rel in _docs(tracked):
        fenced = False
        for n, line in enumerate((root / rel).read_text(encoding="utf-8").split("\n"), 1):
            if FENCE.match(line):
                fenced = not fenced
                continue
            if fenced:
                continue
            for m in LINK.finditer(line):
                target = m.group(1).split("#", 1)[0]
                if not target:
                    continue
                path = PurePosixPath(rel).parent / target
                norm = str(PurePosixPath(*[x for x in _normalise(path.parts)])) if path.parts else ""
                if norm not in tracked and norm.rstrip("/") not in dirs:
                    bad.append(f"{rel}:{n}: {m.group(1)}")
    return bad


def _normalise(parts: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for p in parts:
        if p in (".", ""):
            continue
        if p == "..":
            if out:
                out.pop()
            continue
        out.append(p)
    return out


def test_docs_relative_links_resolve():
    assert _docs(_tracked()), "no document discovered: a link check over nothing asserts nothing"
    assert broken_links() == []


def test_a_link_to_a_missing_file_is_reported(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("See [the manual](mi" "Studio_Manual.md) and [b](b.md).\n")
    (tmp_path / "docs" / "b.md").write_text("ok\n")
    assert broken_links(tmp_path, {"docs/a.md", "docs/b.md"}) == ["docs/a.md:1: mi" "Studio_Manual.md"]
