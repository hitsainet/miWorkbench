"""ENH-010 G5 §2a (P44, P66; R2B-5, R3B-3, R3B-23): miWorkbench's REAL paged scores answer through miDataworks' REAL
pipeline, in miDataworks' venv: `select_served_rows` (021 20.1, when present) → `join_rows` → `accept_source`.

Why it exists. miDataworks' `accept_source` recomputes the AUROC over the rows the scores route serves and refuses
anything outside 5e-4 of `reported_auroc`. G5 adds `reported_auroc_untruncated` and a per-row `truncated` flag and
must not change `reported_auroc`; the fixture's truncated rows move the AUROC by far more than 5e-4, so a wrong pairing
(all rows against the untruncated figure, or the reverse) cannot pass.

Shapes over time (R3B-23). Before 021 20.1 lands, miDataworks has no `select_served_rows`: the "unfiltered" shape runs
(`join_rows` + `accept_source` over every served row against the all-rows `reported_auroc`). When the function exists
the "filtered" shape runs too (its rows against its own reported figure). Once 021-P3/P4 is on miDataworks `main`,
`MIDATAWORKS_FILTER_COMMIT` records that commit and a checkout at or after it WITHOUT the function fails here, so a
revert of the filter is caught. Set 2026-11-17 when 021-P4 landed (ledger 1.9).

Loud skip without the sibling or its interpreter; fails under `MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1`.
Negative controls (in-test, every run): the all-rows rows against `reported_auroc_untruncated` are REFUSED, and the
untruncated rows against `reported_auroc` are REFUSED — the same pairing errors the §2a controls name.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.support.cross_repo import require_cross_repo
from tests.support.sibling_repos import midataworks_repo
from tests.unit.test_probe_scores_route import G5_MASK, _all_pages, _seed_g5, data_dir  # noqa: F401 (fixture)

pytestmark = pytest.mark.asyncio

#: The miDataworks commit that put `select_served_rows` on `main` (021-P3 or 021-P4). None until it lands; set it then.
MIDATAWORKS_FILTER_COMMIT = "8e08ac8e53e98a9e6f618db0714b389a0d89bd16"  # miDataworks 021-P4, pushed 2026-11-17

_SCRIPT = r'''
import json, sys
payload = json.load(open(sys.argv[1]))
from src.services.detector_sets import mistudio_scores as ms
from src.services.detector_sets.paired import accept_source

def run(rows, reported):
    joined = ms.join_rows(rows, keys=payload["keys"], occurrences=payload["occurrences"], values=payload["values"],
                          pairs=None, mapping=payload["mapping"])
    if isinstance(joined, ms.Refusal):
        return {"refused": joined.reason}
    scores, labels, _groups, _keys = joined
    verdict = accept_source("mistudio", scores, labels, reported_auroc=reported, reported_ci=None)
    return {"accepted": bool(verdict.accepted), "auroc": verdict.auroc, "reason": verdict.reason, "rows": len(scores)}

served = payload["served"]
untruncated_rows = [r for r in served["rows"] if not r["truncated"]]
out = {
    "has_filter": hasattr(ms, "select_served_rows"),
    "unfiltered": run(served["rows"], served["reported_auroc"]),
    "control_all_rows_vs_untruncated_figure": run(served["rows"], served["reported_auroc_untruncated"]),
    "control_untruncated_rows_vs_all_rows_figure": run(untruncated_rows, served["reported_auroc"]),
}
if out["has_filter"]:
    selected = ms.select_served_rows(served)
    if isinstance(selected, ms.Refusal):
        out["filtered"] = {"refused": selected.reason}
    else:
        # 021 20.1 returns a `Selected(rows, reported_auroc, excluded_truncated)` dataclass; G5 §2a wrote it as a
        # (rows, reported_auroc, excluded) tuple. Read either, by name when it has names (OVN-wsp-workbench-16).
        if hasattr(selected, "rows"):
            rows, reported, excluded = selected.rows, selected.reported_auroc, selected.excluded_truncated
        else:
            rows, reported, excluded = selected
        out["filtered"] = {**run(rows, reported), "excluded": excluded}
print("@@G5@@" + json.dumps(out))
'''


def _absent(reason: str) -> None:
    if require_cross_repo():
        pytest.fail("MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1 but the G5 accept_source check cannot run: " + reason)
    pytest.skip("CROSS-REPO, SKIPPED LOUDLY: " + reason)


def _midataworks():
    repo = midataworks_repo()
    python = Path(os.environ.get("MIDATAWORKS_PYTHON") or repo / "backend" / ".venv" / "bin" / "python")
    if not (repo / "backend" / "src" / "services" / "detector_sets" / "paired.py").is_file():
        _absent(f"no miDataworks checkout at {repo} (set MIDATAWORKS_REPO)")
    if not python.is_file():
        _absent(f"no miDataworks interpreter at {python} (set MIDATAWORKS_PYTHON)")
    return repo, python


def _is_at_or_after(repo: Path, commit: str) -> bool | None:
    """True when `repo`'s HEAD contains `commit`, False when it does not, None when that cannot be told.

    R4 review fix (C044a-L3), 2026-11-17: any non-zero exit of `git merge-base --is-ancestor` used to read "not
    after", and an exit of 128 ("not a repository", "unknown commit") is not an answer. A `git archive` export, the
    suite's own sibling convention for reviews, made the pin pass with the filter reverted. Only exit 0 and exit 1
    are answers; and only from a repository whose top level IS `repo` (a copy inside another repository would answer
    for that one)."""
    top = subprocess.run(["git", "-C", str(repo), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != Path(repo).resolve():
        return None
    proc = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"],
                          capture_output=True, text=True)
    return {0: True, 1: False}.get(proc.returncode)


async def test_the_real_scores_answer_passes_accept_source(client, async_session, tmp_path):
    repo, python = _midataworks()
    seeded = await _seed_g5(async_session)
    rows, last, _pages = await _all_pages(client, seeded["probe"], seeded["evaluation"].id, limit=7)
    served = {**last, "rows": rows}
    assert [bool(r["truncated"]) for r in rows] == [bool(m) for m in G5_MASK]
    assert abs(served["reported_auroc"] - served["reported_auroc_untruncated"]) > 5e-4, (
        "fixture too weak: the truncated rows must move the AUROC by more than accept_source's tolerance")
    # The published split, as miDataworks holds it: one row per split position, its label value at that position.
    positions = [int(r["position"]) for r in rows]
    width = max(positions) + 1
    values = ["0"] * width
    for r in rows:
        values[int(r["position"])] = str(int(r["label"]))
    payload = {"served": served, "keys": [f"k{i}" for i in range(width)], "occurrences": [0] * width,
               "values": values, "mapping": {"1": "positive", "0": "negative"}}
    path = tmp_path / "served.json"
    path.write_text(json.dumps(payload))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_URL", "PYTHONPATH"))}
    proc = subprocess.run([str(python), "-B", "-c", _SCRIPT, str(path)], cwd=repo / "backend", env=env,
                          capture_output=True, text=True, timeout=240)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@G5@@")]
    if proc.returncode != 0 or not lines:
        pytest.fail(f"miDataworks' pipeline did not run (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    out = json.loads(lines[0][len("@@G5@@"):])

    # The shape every miDataworks checkout has today: every served row against the all-rows figure — accepted.
    assert out["unfiltered"].get("accepted") is True, out["unfiltered"]
    assert out["unfiltered"]["rows"] == len(rows)
    # The two pairing errors §2a names, refused every run.
    assert out["control_all_rows_vs_untruncated_figure"].get("accepted") is False, out
    assert out["control_untruncated_rows_vs_all_rows_figure"].get("accepted") is False, out
    if out["has_filter"]:
        assert out["filtered"].get("accepted") is True, out["filtered"]
        assert out["filtered"]["rows"] == len(rows) - sum(G5_MASK)
        assert out["filtered"]["excluded"] == sum(G5_MASK), out["filtered"]
    elif MIDATAWORKS_FILTER_COMMIT is not None:
        after = _is_at_or_after(repo, MIDATAWORKS_FILTER_COMMIT)
        if after is None:
            # No history to tell a checkout from before the filter from a revert of it (C044a-L3).
            _absent(f"miDataworks at {repo} has no select_served_rows and is not a git repository that can say whether "
                    f"it is at or after {MIDATAWORKS_FILTER_COMMIT} (the 021 filter): a revert cannot be told from an "
                    f"older checkout. Use a clone, or a checkout with the filter")
        elif after:
            pytest.fail(f"miDataworks at {repo} is at or after {MIDATAWORKS_FILTER_COMMIT} (the 021 filter) but has "
                        f"no select_served_rows: the filter was reverted")
