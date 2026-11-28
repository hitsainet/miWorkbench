"""Read every `.py` file under a directory for a source scan, skipping a file deleted between the listing and the read.

Under xdist another worker writes and deletes transient test files under `tests/`: `test_suite_reports_its_totals.py`
writes `tests/unit/test_zz_totals_probe_<pid>.py` and removes it after its inner run. A scan that listed `tests/` and
then read each file raised `FileNotFoundError` whenever it listed that probe and read it after the deletion (the
`test_no_process_wide_legacy_env_writes` red in shard 5 of the T0 and T1 of `fix/wsp-baselines-supplement`,
2026-11-27/28). A file that is gone when it is read is not part of the tree (a committed file does not vanish during a
run), so it is skipped; every other `OSError` still raises.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def python_sources(root: Path) -> List[Tuple[Path, str]]:
    """`(path, text)` for every `*.py` under `root`, sorted by path; a file deleted after the listing is left out."""
    out = []
    for path in sorted(root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        out.append((path, text))
    return out
