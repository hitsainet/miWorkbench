"""Is this checkout the public mirror's sync-stripped tree?  (W2a, 2026-11-25)

`sync-to-clean.yml` publishes ONE parentless commit to `hitsainet/miWorkbench`, from which its "Remove excluded files and
folders" step has deleted the private record trees (`0xcc/`, `.claude/`, `CLAUDE.md`, `docs/` but its public files,
`scripts/`, ...), and stamps it with the `Source-Commit: <sha>` and `Private-CI: passed` trailers ("Build the orphan
snapshot"). A test that reads a file the sync deletes cannot run there. It may skip there, LOUDLY, and only there: the
private repository's CI is the gate, and a file missing from a private tree is a defect that must stay red. The absence
of a file is not proof of the mirror (a deleted file looks the same); the commit shape is, together with the sync's own
removal list.

- :func:`mirror_snapshot`: HEAD is a parentless commit carrying both trailers (miForge's predicate,
  `backend/tests/support/mirror_snapshot.py` there).
- :func:`stripped_by_sync`: the path is one the sync deletes, read from the workflow's own step (the source of truth),
  never from a list kept here.
- :func:`require_unstripped`: the guard a test calls before it reads such a path. Present: nothing happens. Absent on the
  stripped tree: a loud skip naming the path. Absent anywhere else: a failure.

W1's T1 (2026-11-25) found 18 failures and 3 errors in a faithful mirror simulation, in four files that read what the
sync strips; this module is their fix (`tests/unit/test_stripped_tree.py` holds it to real commits).
"""
from __future__ import annotations

import re
import subprocess
from functools import cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "sync-to-clean.yml"
STRIP_STEP = "Remove excluded files and folders"
_SOURCE_COMMIT = re.compile(r"^Source-Commit: [0-9a-f]{40}$", re.M)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except (FileNotFoundError, NotADirectoryError):
        return None


def mirror_snapshot(root: Path = REPO_ROOT) -> bool:
    """True only when `root`'s HEAD is a parentless commit carrying both sync trailers. A commit of this repository
    has a parent and no such trailers; a shallow clone of it still lacks the trailers. No git, or no repository:
    False."""
    parents = _git(root, "rev-list", "--parents", "-n", "1", "HEAD")
    trailers = _git(root, "log", "-1", "--format=%(trailers:only,unfold)", "HEAD")
    if parents is None or trailers is None or parents.returncode or trailers.returncode:
        return False
    if len(parents.stdout.split()) != 1:
        return False
    lines = trailers.stdout.splitlines()
    return "Private-CI: passed" in lines and bool(_SOURCE_COMMIT.search(trailers.stdout))


@cache
def removal_rules(workflow: Path = WORKFLOW) -> tuple[frozenset[str], frozenset[str]]:
    """`(removed, docs_kept)` read from the sync's strip step: every path its `rm -rf`/`rm -f` lines delete (relative,
    no trailing slash), and the `docs/` entries its `find docs/ ... ! -name X` keeps. Refuses a step it cannot read,
    so a reworded step fails here instead of making every guard skip or fail open."""
    import yaml

    jobs = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]
    steps = [s for job in jobs.values() for s in job.get("steps", []) if s.get("name") == STRIP_STEP]
    if len(steps) != 1:
        raise AssertionError(f"{workflow.name}: expected one {STRIP_STEP!r} step, found {len(steps)}")
    run = steps[0]["run"].replace("\\\n", " ")
    removed = {m.group(1).rstrip("/") for m in re.finditer(r"^\s*rm -r?f\s+([\w./-]+)\s*$", run, re.M)}
    finds = re.findall(r"find docs/ [^\n]*", run)
    if len(finds) != 1 or len(removed) < 4:
        raise AssertionError(f"{workflow.name}'s {STRIP_STEP!r} step no longer reads as this module expects: "
                             f"removed {sorted(removed)}, docs finds {finds}")
    kept = frozenset(re.findall(r"! -name ([\w.-]+)", finds[0]))
    if "schemas" not in kept:
        raise AssertionError(f"the docs/ keep-list reads {sorted(kept)}: the parse is wrong")
    return frozenset(removed), kept


def stripped_by_sync(path: Path, root: Path = REPO_ROOT) -> bool:
    """Whether the sync deletes `path` (a path inside `root`)."""
    rel = Path(path).resolve().relative_to(root.resolve()).as_posix()
    removed, docs_kept = removal_rules(root / ".github" / "workflows" / "sync-to-clean.yml")
    if any(rel == r or rel.startswith(r + "/") for r in removed):
        return True
    parts = rel.split("/")
    return parts[0] == "docs" and len(parts) > 1 and parts[1] not in docs_kept


def require_unstripped(path: Path, root: Path = REPO_ROOT) -> None:
    """Call before reading `path`. Present: returns. Absent on the mirror's stripped tree, and a path the sync deletes:
    skips LOUDLY. Absent anywhere else (this private repository, a clone, a wrong path): fails."""
    path = Path(path)
    if path.exists():
        return
    if stripped_by_sync(path, root) and mirror_snapshot(root):
        pytest.skip(f"LOUD SKIP: {path.relative_to(root)} is deleted by sync-to-clean.yml's {STRIP_STEP!r} step and this "
                    "is the mirror's stripped snapshot; the check runs in the private repository, whose CI gates the "
                    "sync")
    pytest.fail(f"{path} is missing, and this tree is not the mirror's stripped snapshot (or the sync keeps that path): "
                "the check would read nothing")
