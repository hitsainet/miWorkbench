"""`0xcc/NAMES.md` keys the old names in dated records to the new ones (043 FR-043.19; P9-D9; FTASKS 9.3).

Records keep the names of their date and are never rewritten, so the note is how a reader maps them. The
names checked are the export's own repo records (old and new), never typed here.
"""
from __future__ import annotations

import pytest

from tests.support import legacy_export as le

NOTE = le.REPO_ROOT / "0xcc" / "NAMES.md"


def test_names_note_exists_and_names_both_renames():
    if not (le.REPO_ROOT / "0xcc").is_dir():
        pytest.skip("mirror-shaped tree: the sync strips 0xcc/, the note's home (CF-2)")
    assert NOTE.is_file(), "0xcc/NAMES.md is missing: dated records have no key to the old names"
    text = NOTE.read_text(encoding="utf-8")
    export = le.load()
    repos = [r for r in export.repos() if not r["old"].endswith("/miSuite")]
    assert repos, "the export lists no repo record"
    for rec in repos:
        old_name, new_name = rec["old"].split("/", 1)[1], rec["new"].split("/", 1)[1]
        assert old_name in text and new_name in text, (old_name, new_name)
    assert "legacy-identifiers.md" in text, "the note does not point at the frozen-value export"
