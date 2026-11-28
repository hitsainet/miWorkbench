"""The mirror's image build trusts THIS repository's CI (operator directive, 2026-10-20).

Before: every push ran the suite here, the sync published a snapshot straight away, and the
public mirror's image build waited for the mirror to run the same suite again (or, in some
repos, for nothing at all). After:

1. PRIVATE: `sync-to-clean.yml`'s `gate` job runs `.github/scripts/private_ci_gate.py`, which
   refuses unless every required CI workflow passed for the exact commit (Backend Tests
   including its `all-shards` aggregator). The `sync` job `needs` the gate, checks out the
   commit the gate passed, and stamps the snapshot with `Source-Commit:` and
   `Private-CI: passed` trailers.
2. MIRROR: `docker-images.yml`'s `build` needs a `private-ci` job that requires those
   trailers on the snapshot, and nothing in it waits on the mirror's own test runs.
3. ORDER: the sync job is serialised, and a branch sync never moves the mirror back to an
   older commit than the one it already holds.

Every assertion reads the parsed YAML or RUNS the step's script; none greps the text.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

yaml = pytest.importorskip("yaml")

# ── per-repository facts ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]


def _private_names() -> set[str]:
    """The private repository's names, from the export's `repo` records (042 TPD-27, CF-13): its old and
    new name until ENH-009 P1 closes, the new one after (CF-17). Never spelled here."""
    from tests.support import legacy_export as le
    from tests.support import workflow_gates as wg

    export = le.load()
    if export.is_public and any(n.startswith("repo:sha256:") for n in export.withheld):
        pytest.skip("the private repository's repo records are withheld from the public export: "
                    + ", ".join(n for n in export.withheld if n.startswith("repo:")))
    private_owner, _public = wg.private_and_public_owner(wg.load_workflows(), export)
    return wg.private_names(export, private_owner)


PRIVATE: set[str] = set()  # filled on first use by `_private()` (a module-level skip would hide every test)


def _private() -> set[str]:
    if not PRIVATE:
        PRIVATE.update(_private_names())
    return PRIVATE
# Workflows that run as CI here but deliberately do not gate the mirror, with the reason.
EXEMPT: dict[str, str] = {}
# ─────────────────────────────────────────────────────────────────────────────────────

WORKFLOWS = REPO_ROOT / ".github" / "workflows"
GATE_SCRIPT = REPO_ROOT / ".github" / "scripts" / "private_ci_gate.py"


def _load(name: str) -> dict:
    doc = yaml.safe_load((WORKFLOWS / name).read_text())
    doc["on"] = doc.pop(True, doc.get("on"))  # PyYAML reads the bare key `on` as boolean True
    return doc


def _runs_here(job: dict) -> bool:
    """False for a job that can only run on the public mirror."""
    cond = str(job.get("if", ""))
    return not (
        "github.repository" in cond and "hitsainet/" in cond and "Onegaishimas/" not in cond
    )


def _private_ci_workflows() -> set[str]:
    """Every workflow that is CI for main here: runs on a push to main AND on pull requests,
    with at least one job that runs in this repository."""
    out = set()
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = _load(path.name)
        on = doc["on"] if isinstance(doc["on"], dict) else dict.fromkeys(doc["on"])
        if "pull_request" not in on or "push" not in on:
            continue
        branches = (on["push"] or {}).get("branches")
        if branches is not None and "main" not in branches:
            continue
        if any(_runs_here(j) for j in doc["jobs"].values()):
            out.add(path.name)
    return out - set(EXEMPT)


def _step(job: dict, name: str) -> dict:
    return next(s for s in job["steps"] if s.get("name") == name)


def _bash(script: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    full = {**os.environ, **env}
    return subprocess.run(
        ["bash", "-e", "-c", script], cwd=cwd, env=full, capture_output=True, text=True, check=False
    )


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


# ── 1. the private gate ──────────────────────────────────────────────────────────────


def test_the_gate_requires_every_ci_workflow_that_runs_here():
    sync = _load("sync-to-clean.yml")
    gate = sync["jobs"]["gate"]
    required = {w.strip() for w in gate["env"]["REQUIRED_WORKFLOWS"].split() if w.strip()}
    assert required == _private_ci_workflows(), (
        "REQUIRED_WORKFLOWS must name every CI workflow that runs on main here (or EXEMPT it "
        "with a reason): a CI workflow left out can be red while its code reaches the mirror"
    )
    assert "backend-tests.yml" in required
    assert gate["permissions"].get("actions") == "read", "the gate lists runs and jobs"
    run = _step(gate, "Require private CI to have passed for this commit")
    assert run["run"].strip() == "python .github/scripts/private_ci_gate.py"
    assert run["env"]["SHA"] == "${{ github.sha }}"
    assert gate["outputs"]["sha"] == "${{ steps.gate.outputs.sha }}" and run["id"] == "gate"


def test_the_gate_and_the_sync_run_only_in_the_private_repository():
    """Both jobs that hold the mirror token or decide what reaches it run in the private repository only,
    under every name it has (FR-042.12): a gate naming only the old name stops publishing at the P1
    rename; one naming the mirror would let the public repository publish to itself."""
    import re

    jobs = _load("sync-to-clean.yml")["jobs"]
    for name in ("gate", "sync"):
        cond = str(jobs[name].get("if", ""))
        named = set(re.findall(r"github\.repository == '([^']+)'", cond))
        assert named == _private(), f"sync-to-clean.yml {name}: {cond!r} names {named}, expected {_private()}"


def test_the_required_ci_runs_only_in_the_private_repository():
    """The required CI workflows run every job here and none on the mirror (whose image build trusts the
    `Private-CI: passed` trailer, not its own runs)."""
    import re

    gate = _load("sync-to-clean.yml")["jobs"]["gate"]
    required = {w.strip() for w in gate["env"]["REQUIRED_WORKFLOWS"].split() if w.strip()}
    for wf in sorted(required):
        for job_id, job in _load(wf)["jobs"].items():
            cond = str(job.get("if", ""))
            named = set(re.findall(r"github\.repository == '([^']+)'", cond))
            assert named == _private(), f"{wf} {job_id}: {cond!r}"


def test_nothing_reaches_the_mirror_except_through_the_gate():
    jobs = _load("sync-to-clean.yml")["jobs"]
    pushers = {k for k, j in jobs.items() if "HITSAINET_SYNC" in json.dumps(j)}
    assert pushers == {"sync"}, f"only the gated job may hold the mirror token: {pushers}"
    needs = jobs["sync"]["needs"]
    assert needs == "gate" or "gate" in needs
    assert "always()" not in str(
        jobs["sync"].get("if", "")
    ), "always() would run the sync over a failed gate"


def test_the_sync_publishes_the_commit_the_gate_passed():
    sync = _load("sync-to-clean.yml")["jobs"]["sync"]
    checkout = next(
        s for s in sync["steps"] if str(s.get("uses", "")).startswith("actions/checkout")
    )
    assert checkout["with"]["ref"] == "${{ needs.gate.outputs.sha }}"
    snap = _step(sync, "Build the orphan snapshot")
    assert snap["env"]["SOURCE_SHA"] == "${{ needs.gate.outputs.sha }}"
    for step in sync["steps"]:
        assert "GITHUB_SHA" not in step.get("run", ""), (
            f"{step.get('name')}: GITHUB_SHA names the triggering commit, "
            "not necessarily the gated one"
        )


def test_the_sync_is_serialised_and_never_moves_the_mirror_backwards():
    sync = _load("sync-to-clean.yml")["jobs"]["sync"]
    conc = sync["concurrency"]
    assert conc["group"] and conc["cancel-in-progress"] is False
    names = [s.get("name") for s in sync["steps"]]
    order = names.index("Never move the mirror backwards")
    assert sync["steps"][order]["id"] == "order"
    for step in sync["steps"][order + 1 :]:
        assert "steps.order.outputs.publish != 'false'" in str(step.get("if", "")), (
            f"{step.get('name')} runs even when the ordering check said "
            "an older commit must not publish"
        )


# ── 2. the snapshot's trailers and the mirror's trust check, end to end ─────────────


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "src"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "VERSION").write_text("0.5.0\n")
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.5.0"\n')
    (repo / "README.md").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "source")
    return repo


def _mirror_check() -> str:
    job = _load("docker-images.yml")["jobs"]["private-ci"]
    return _step(job, "Require the private repo's CI verdict")["run"]


def test_a_snapshot_the_sync_builds_passes_the_mirror_check(source_repo):
    source_sha = _git(source_repo, "rev-parse", "HEAD")
    snap = _step(_load("sync-to-clean.yml")["jobs"]["sync"], "Build the orphan snapshot")
    built = _bash(snap["run"], source_repo, {"SOURCE_SHA": source_sha})
    assert built.returncode == 0, built.stdout + built.stderr
    head = _git(source_repo, "rev-parse", "HEAD")
    trailers = _git(source_repo, "log", "-1", "--format=%(trailers:only,unfold)")
    assert f"Source-Commit: {source_sha}" in trailers and "Private-CI: passed" in trailers

    ok = _bash(_mirror_check(), source_repo, {"GITHUB_SHA": head})
    assert ok.returncode == 0, ok.stdout + ok.stderr


@pytest.mark.parametrize(
    "message",
    [
        "snapshot with no trailers",
        "snapshot\n\nSource-Commit: " + "a" * 40 + "\nPrivate-CI: failed",
        "snapshot\n\nPrivate-CI: passed",
        "snapshot\n\nSource-Commit: abc1234\nPrivate-CI: passed",
    ],
)
def test_the_mirror_refuses_a_commit_without_the_verdict(source_repo, message):
    _git(
        source_repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        message,
    )
    head = _git(source_repo, "rev-parse", "HEAD")
    res = _bash(_mirror_check(), source_repo, {"GITHUB_SHA": head})
    assert res.returncode != 0, f"accepted {message!r}: {res.stdout}"


def test_the_mirror_build_trusts_the_trailer_not_its_own_test_runs():
    doc = _load("docker-images.yml")
    jobs = doc["jobs"]
    trust = jobs["private-ci"]
    assert not _runs_here(trust), "the trust check belongs to the mirror"
    build = jobs["build"]
    needs = build["needs"] if isinstance(build["needs"], list) else [build["needs"]]
    assert "private-ci" in needs
    assert "always()" not in str(
        build.get("if", "")
    ), "always() would build over a failed trust check"
    for name, job in jobs.items():
        perms = job.get("permissions") or {}
        assert (
            perms.get("actions") is None
        ), f"{name} can read workflow runs - nothing here should wait on tests"
        for step in job.get("steps", []):
            run = step.get("run", "")
            assert (
                "/actions/runs" not in run and "Backend Tests" not in run
            ), f"{name}/{step.get('name')} still waits on the mirror's own test runs"


# ── 3. the shards and their aggregator ───────────────────────────────────────────────


def _shard_jobs(doc: dict) -> dict[str, dict]:
    return {
        k: j
        for k, j in doc["jobs"].items()
        if "shard" in ((j.get("strategy") or {}).get("matrix") or {})
    }


def test_every_shard_job_runs_one_slice_of_n_and_the_aggregator_needs_them_all():
    doc = _load("backend-tests.yml")
    shards = _shard_jobs(doc)
    assert shards, "Backend Tests is not sharded"
    for name, job in shards.items():
        values = job["strategy"]["matrix"]["shard"]
        n = len(values)
        assert values == list(range(1, n + 1)), f"{name}: shards must be 1..{n}, got {values}"
        assert (
            job["env"]["CI_SHARD"] == f"${{{{ matrix.shard }}}}/{n}"
        ), f"{name}: CI_SHARD disagrees with the matrix"
        assert job["strategy"].get("fail-fast") is False
    agg = doc["jobs"]["all-shards"]
    needs = agg["needs"] if isinstance(agg["needs"], list) else [agg["needs"]]
    assert set(needs) == set(shards), "the aggregator must need every shard job"
    assert "always()" in str(
        agg.get("if", "")
    ), "a skipped aggregator reads as neither pass nor fail"


@pytest.mark.parametrize(
    "results, ok",
    [
        ({"a": "success", "b": "success"}, True),
        ({"a": "success", "b": "failure"}, False),
        ({"a": "success", "b": "cancelled"}, False),
        ({"a": "skipped"}, False),
        ({}, False),
    ],
)
def test_the_aggregator_passes_only_when_every_shard_passed(tmp_path, results, ok):
    agg = _load("backend-tests.yml")["jobs"]["all-shards"]
    step = _step(agg, "Require every shard to have passed")
    needs = json.dumps({k: {"result": v, "outputs": {}} for k, v in results.items()})
    res = _bash(step["run"], tmp_path, {"RESULTS": needs})
    assert (res.returncode == 0) is ok, res.stdout + res.stderr


# ── 4. the gate script's decisions ───────────────────────────────────────────────────


def _gate_module():
    spec = importlib.util.spec_from_file_location("private_ci_gate", GATE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Api:
    """A fake of the three GitHub endpoints the gate reads."""

    def __init__(self, by_sha=None, history=None, jobs=None):
        self.by_sha = by_sha or []  # list of successive answers for head_sha=<sha>
        self.history = history or []
        self.jobs = jobs or {}
        self.calls = 0

    def __call__(self, path: str):
        if "/jobs?" in path:
            run_id = int(path.split("/runs/")[1].split("/")[0])
            return {"jobs": self.jobs.get(run_id, [])}
        if "head_sha=" in path:
            answer = self.by_sha[min(self.calls, len(self.by_sha) - 1)] if self.by_sha else []
            self.calls += 1
            return {"workflow_runs": answer}
        return {"workflow_runs": self.history}


def _run(i, sha, status="completed", conclusion="success", event="push"):
    return {
        "id": i,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
        "event": event,
        "created_at": f"2026-10-20T00:00:{i:02d}Z",
        "run_attempt": 1,
    }


@pytest.fixture
def history_repo(tmp_path):
    """main: A (tests passed) -> B (docs only) -> C (touches the tested path)."""
    repo = tmp_path / "repo"
    (repo / ".github" / "workflows").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".github" / "workflows" / "filtered.yml").write_text(
        "name: F\non:\n  push:\n    branches: [main]\n    paths: ['src/**']\n"
        "jobs:\n  t: {runs-on: x, steps: []}\n"
    )
    (repo / ".github" / "workflows" / "sharded.yml").write_text(
        "name: S\non:\n  push:\n    branches: [main]\njobs:\n  t: {runs-on: x, steps: []}\n"
        "  all-shards:\n    name: all shards\n    needs: [t]\n    runs-on: x\n    steps: []\n"
    )
    (repo / "src").mkdir()
    shas = []
    for name, path in (("A", "src/a.py"), ("B", "docs.md"), ("C", "src/c.py")):
        (repo / path).write_text(name)
        _git(repo, "add", "-A")
        _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", name)
        shas.append(_git(repo, "rev-parse", "HEAD"))
    return SimpleNamespace(repo=repo, a=shas[0], b=shas[1], c=shas[2])


def _gate(mod, repo, sha, api):
    return mod.Gate(
        "o/r",
        sha,
        repo,
        get=api,
        sleep=lambda s: None,
        clock=iter(range(0, 10**6, 60)).__next__,
        timeout=600,
        grace=120,
        interval=30,
    )


def test_gate_passes_a_green_run_for_the_commit(history_repo):
    mod = _gate_module()
    h = history_repo
    ok, msg = _gate(mod, h.repo, h.c, _Api(by_sha=[[_run(1, h.c)]])).check("filtered.yml")
    assert ok, msg


def test_gate_waits_for_a_running_run_then_passes(history_repo):
    mod = _gate_module()
    h = history_repo
    api = _Api(
        by_sha=[
            [_run(1, h.c, "in_progress", None)],
            [_run(1, h.c, "in_progress", None)],
            [_run(1, h.c)],
        ]
    )
    ok, msg = _gate(mod, h.repo, h.c, api).check("filtered.yml")
    assert ok and api.calls == 3, msg


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_gate_refuses_a_red_run_for_the_commit(history_repo, conclusion):
    mod = _gate_module()
    h = history_repo
    ok, msg = _gate(mod, h.repo, h.c, _Api(by_sha=[[_run(1, h.c, conclusion=conclusion)]])).check(
        "filtered.yml"
    )
    assert not ok and conclusion in msg


def test_gate_ignores_a_pull_request_run(history_repo):
    mod = _gate_module()
    h = history_repo
    ok, _ = _gate(mod, h.repo, h.c, _Api(by_sha=[[_run(1, h.c, event="pull_request")]])).check(
        "filtered.yml"
    )
    assert not ok


def test_gate_requires_the_all_shards_job(history_repo):
    mod = _gate_module()
    h = history_repo
    run = _run(7, h.c)
    no_agg = _Api(by_sha=[[run]], jobs={7: [{"name": "t", "conclusion": "success"}]})
    assert not _gate(mod, h.repo, h.c, no_agg).check("sharded.yml")[0]
    red_agg = _Api(by_sha=[[run]], jobs={7: [{"name": "all shards", "conclusion": "failure"}]})
    assert not _gate(mod, h.repo, h.c, red_agg).check("sharded.yml")[0]
    green = _Api(by_sha=[[run]], jobs={7: [{"name": "all shards", "conclusion": "success"}]})
    assert _gate(mod, h.repo, h.c, green).check("sharded.yml")[0]


def test_gate_waits_for_an_aggregator_the_jobs_api_has_not_reported_yet(history_repo):
    """2026-10-22: a run read completed/success while its `all shards` job had no conclusion yet,
    and the gate refused a green commit. An unconcluded aggregator is waited for."""
    mod = _gate_module()
    h = history_repo
    run = _run(7, h.c)
    answers = iter(
        [
            [{"name": "all shards", "conclusion": None}],
            [{"name": "all shards", "conclusion": None}],
            [{"name": "all shards", "conclusion": "success"}],
        ]
    )
    api = _Api(by_sha=[[run]])
    api.jobs = type("Lagging", (), {"get": lambda self, rid, default=None: next(answers)})()
    ok, msg = _gate(mod, h.repo, h.c, api).check("sharded.yml")
    assert ok, msg


def test_gate_still_refuses_an_aggregator_that_never_reports(history_repo):
    mod = _gate_module()
    h = history_repo
    run = _run(7, h.c)
    api = _Api(by_sha=[[run]], jobs={7: [{"name": "all shards", "conclusion": None}]})
    ok, msg = _gate(mod, h.repo, h.c, api).check("sharded.yml")
    assert not ok and "timed out" in msg


def test_gate_keeps_waiting_when_a_listed_run_drops_out_of_one_answer(history_repo):
    """2026-10-22: a run listed in progress was missing from the next answer, and the gate refused
    the commit as having no run. A run once seen is asked for again."""
    mod = _gate_module()
    h = history_repo
    api = _Api(by_sha=[[_run(1, h.c, "in_progress", None)], [], [], [_run(1, h.c)]])
    ok, msg = _gate(mod, h.repo, h.c, api).check("filtered.yml")
    assert ok and api.calls == 4, msg


def test_gate_falls_back_only_when_the_tested_paths_are_unchanged(history_repo):
    mod = _gate_module()
    h = history_repo
    # B touched only docs: no run of the filtered workflow, and A passed with identical src/.
    ok, msg = _gate(mod, h.repo, h.b, _Api(history=[_run(1, h.a)])).check("filtered.yml")
    assert ok and h.a in msg, msg
    # C changed src/ since A: A's verdict does not cover it.
    ok, _ = _gate(mod, h.repo, h.c, _Api(history=[_run(1, h.a)])).check("filtered.yml")
    assert not ok
    # A newer commit that is NOT an ancestor never vouches for an older one - even when, as
    # B here, its tested paths are identical: only the ancestry rule refuses this one.
    ok, _ = _gate(mod, h.repo, h.a, _Api(history=[_run(2, h.b)])).check("filtered.yml")
    assert not ok


def test_gate_refuses_a_missing_run_for_an_unfiltered_workflow(history_repo):
    mod = _gate_module()
    h = history_repo
    api = _Api(history=[_run(1, h.b)], jobs={1: [{"name": "all shards", "conclusion": "success"}]})
    ok, msg = _gate(mod, h.repo, h.c, api).check("sharded.yml")
    assert not ok and "no path filter" in msg


def test_gate_refuses_an_empty_requirement(monkeypatch):
    mod = _gate_module()
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("SHA", "a" * 40)
    monkeypatch.setenv("REQUIRED_WORKFLOWS", "  \n")
    assert mod.main() == 1


# ── 5. the gitops sync waits for the same gate (ENH-011 WB-1, #13) ─────────────────────────────────────────────
#
# `sync-main-to-gitops-pilot.yml` merged and converged `origin/main` the moment a push landed, with no `needs:`: a red
# commit's manifests reached ArgoCD's branch while its CI was still running. It now needs the same `gate` the mirror
# sync has, and merges, converges and orders by the commit that gate passed. Each test drives the parsed steps.

GITOPS = "sync-main-to-gitops-pilot.yml"
GATE_STEP = "Require private CI to have passed for this commit"
#: The Image-Updater-owned pin files the converge keeps (the old app's until it is deleted, then the new one's).
PIN_FILES = ("k8s/base/.argocd-source-mistudio.yaml", "k8s/base/.argocd-source-miworkbench.yaml")
_IF_CLAUSE = __import__("re").compile(r"^steps\.([\w-]+)\.outputs\.([\w-]+)\s*(!=|==)\s*'([^']*)'$")


def _gitops_jobs() -> dict:
    return _load(GITOPS)["jobs"]


def test_the_gitops_sync_needs_the_private_ci_gate():
    """(a) The sync job needs a job that runs `private_ci_gate.py` for this commit with Backend Tests required."""
    jobs = _gitops_jobs()
    sync = jobs["sync"]
    needs = sync.get("needs")
    needs = [needs] if isinstance(needs, str) else list(needs or [])
    assert needs == ["gate"], f"the gitops sync must need the gate, and only it: {needs}"
    assert "always()" not in str(sync.get("if", "")), "always() would run the sync over a failed gate"
    gate = jobs["gate"]
    run = _step(gate, GATE_STEP)
    assert run["run"].strip() == "python .github/scripts/private_ci_gate.py" and run["id"] == "gate"
    assert run["env"]["SHA"] == "${{ github.sha }}"
    assert gate["outputs"]["sha"] == "${{ steps.gate.outputs.sha }}"
    assert gate["permissions"].get("actions") == "read", "the gate lists runs and jobs"
    required = {w.strip() for w in gate["env"]["REQUIRED_WORKFLOWS"].split() if w.strip()}
    assert "backend-tests.yml" in required and required == _private_ci_workflows()
    mirror = _load("sync-to-clean.yml")["jobs"]["gate"]
    assert gate["env"]["REQUIRED_WORKFLOWS"] == mirror["env"]["REQUIRED_WORKFLOWS"], "the two gates must agree"
    checkout = next(s for s in gate["steps"] if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout["with"] == {"fetch-depth": 0, "ref": "${{ github.sha }}"}, "the path-filter fallback reads history"
    assert sync["env"]["GATED_SHA"] == "${{ needs.gate.outputs.sha }}"
    assert sync["permissions"] == {"contents": "write"} and (_load(GITOPS).get("permissions") or {}).get(
        "contents") == "read", "only the job that pushes may write"


def test_the_gitops_gate_and_sync_run_only_in_the_private_repository():
    import re

    for name, job in _gitops_jobs().items():
        named = set(re.findall(r"github\.repository == '([^']+)'", str(job.get("if", ""))))
        assert named == _private(), f"{GITOPS} {name}: {job.get('if')!r} names {named}, expected {_private()}"


def _fixture_git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True,
                          text=True).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    _fixture_git(repo, "add", "-A")
    _fixture_git(repo, "commit", "-q", "-m", message)
    return _fixture_git(repo, "rev-parse", "HEAD")


#: A pin file the sync never names: it exists only on gitops/pilot, so the converge (which overwrites the files main
#: has and deletes nothing) leaves it alone. The neuronpedia Application's pin lives this way on the real branch.
PILOT_ONLY = "k8s/neuronpedia/base/.argocd-source-neuronpedia.yaml"


@pytest.fixture
def gitops_remote(tmp_path):
    """A bare `origin` with `main` M1 -> M2 -> M3 and `gitops/pilot` = M1 plus an Image Updater pin commit and drift.

    `PILOT_ONLY` exists only on gitops/pilot, as every pin does on the real branch; the two named pin files also have a
    copy on main, so only the converge's exclusions keep them. M2 is the commit the gate passed; M3 is newer and
    NOT gated (its CI is still running, or red). `dead` is a commit off M1 that is no longer on main (a force-push
    dropped it) whose gate finished late."""
    work = tmp_path / "work"
    work.mkdir()
    _fixture_git(work, "init", "-q", "-b", "main")
    # The two named pin files are given a copy on main too, so the converge's exclusions are what keeps the pins.
    on_main = {p: f"# {p} as main has it\n" for p in PIN_FILES}
    m1 = _commit(work, {"app.txt": "m1\n", "k8s/base/backend.yaml": "image: m1\n", **on_main}, "M1")
    _fixture_git(work, "checkout", "-q", "-b", "gitops/pilot")
    pinned = {p: f"# image pin of {p}, written by Image Updater\n" for p in (*PIN_FILES, PILOT_ONLY)}
    _commit(work, {**pinned, "k8s/base/backend.yaml": "image: m1\n# pilot-only drift\n"},
            "build: automatic update of the image pins")
    _fixture_git(work, "checkout", "-q", "-b", "dead", m1)
    dead = _commit(work, {"app.txt": "dead\n"}, "a commit main no longer has")
    _fixture_git(work, "checkout", "-q", "main")
    m2 = _commit(work, {"app.txt": "m2\n", "k8s/base/backend.yaml": "image: m2\n"}, "M2 (gated)")
    m3 = _commit(work, {"app.txt": "m3\n", "new-in-m3.txt": "ungated\n"}, "M3 (not gated)")
    origin = tmp_path / "origin.git"
    _fixture_git(tmp_path, "clone", "-q", "--bare", str(work), str(origin))
    return SimpleNamespace(tmp=tmp_path, origin=origin, m1=m1, m2=m2, m3=m3, dead=dead, pinned=pinned)


def _evaluate_if(expr: str, outputs: dict[str, dict[str, str]]) -> bool:
    """The step conditions this workflow uses: `steps.<id>.outputs.<key> != '<v>'` (or `==`), joined by `&&`.
    Anything else fails the test rather than being guessed at."""
    expr = expr.strip()
    if expr.startswith("${{") and expr.endswith("}}"):
        expr = expr[3:-2].strip()
    result = True
    for clause in expr.split("&&"):
        m = _IF_CLAUSE.match(clause.strip())
        assert m, f"the test cannot evaluate the step condition {clause.strip()!r}"
        got = outputs.get(m.group(1), {}).get(m.group(2), "")
        result = result and ((got != m.group(4)) if m.group(3) == "!=" else (got == m.group(4)))
    return result


def _run_gitops_sync(remote, gated_sha: str):
    """Every `run` step of the real sync job, in order and under its own `if`, in a clone checked out at
    `gitops/pilot` the way `actions/checkout` leaves it. Returns what each step printed and output."""
    clone = remote.tmp / "pilot"
    _fixture_git(remote.tmp, "clone", "-q", str(remote.origin), str(clone))
    _fixture_git(clone, "checkout", "-q", "gitops/pilot")
    env = {"GATED_SHA": gated_sha, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    outputs: dict[str, dict[str, str]] = {}
    ran: list[tuple[str, subprocess.CompletedProcess]] = []
    for step in _gitops_jobs()["sync"]["steps"]:
        if "run" not in step:
            continue
        if "if" in step and not _evaluate_if(str(step["if"]), outputs):
            continue
        out_file = remote.tmp / f"out-{len(ran)}"
        out_file.write_text("")
        res = subprocess.run(["bash", "-eo", "pipefail", "-c", step["run"]], cwd=clone,
                             env={**os.environ, **env, "GITHUB_OUTPUT": str(out_file)},
                             capture_output=True, text=True, check=False)
        ran.append((step.get("name"), res))
        assert res.returncode == 0, f"{step.get('name')} failed:\n{res.stdout}\n{res.stderr}"
        if step.get("id"):
            outputs[step["id"]] = dict(line.split("=", 1) for line in out_file.read_text().splitlines() if "=" in line)
    return SimpleNamespace(ran=ran, outputs=outputs,
                           pilot=_fixture_git(remote.origin, "rev-parse", "refs/heads/gitops/pilot"))


def _tree_without_pins(repo: Path, rev: str) -> dict[str, str]:
    names = _fixture_git(repo, "ls-tree", "-r", "--name-only", rev).splitlines()
    return {n: _fixture_git(repo, "show", f"{rev}:{n}") for n in names if n not in (*PIN_FILES, PILOT_ONLY)}


def test_the_gitops_sync_merges_and_converges_the_gated_commit_never_main(gitops_remote):
    """(b) main has moved past the gated commit (M3, not gated): the deploy branch gets exactly M2's files (the drift
    on a file main has is converged away), and every pin file keeps the Image Updater's content."""
    r = gitops_remote
    before = _fixture_git(r.origin, "rev-parse", "refs/heads/gitops/pilot")
    done = _run_gitops_sync(r, r.m2)
    assert done.pilot != before, "nothing was pushed: the test proves nothing"
    assert _tree_without_pins(r.origin, done.pilot) == _tree_without_pins(r.origin, r.m2), (
        "gitops/pilot must hold exactly the gated commit's files")
    for pin, text in r.pinned.items():
        assert _fixture_git(r.origin, "show", f"{done.pilot}:{pin}") == text.strip(), f"{pin} was converged away"
    assert subprocess.run(["git", "merge-base", "--is-ancestor", r.m3, done.pilot], cwd=r.origin).returncode == 1, (
        "the ungated M3 reached the deploy branch")


def test_no_gitops_step_merges_or_checks_out_main():
    """(b) statically: every merge, checkout and ancestry test names the gated commit, never `origin/main`."""
    import re

    lines = [s for step in _gitops_jobs()["sync"]["steps"] for s in re.sub(r"\\\n\s*", " ", step.get("run", ""))
             .splitlines() if s.strip() and not s.strip().startswith("#")]
    acting = [s.strip() for s in lines if re.search(r"\bgit (merge|checkout)\b", s) and "merge-base --all" not in s]
    assert acting, "no merge or checkout found: the scan reads nothing"
    for s in acting:
        assert "origin/main" not in s, f"merges or converges main's head, not the gated commit: {s}"
        assert "GATED_SHA" in s, f"does not name the gated commit: {s}"


@pytest.mark.parametrize("case", ["older", "off_main"])
def test_the_gitops_sync_never_moves_the_deploy_branch_backwards(gitops_remote, case):
    """(c) gitops/pilot already holds M3 (a later push whose gate finished first). A gate that passes afterwards for
    M2 (older: already contained) or for a commit no longer on main changes nothing, and says why."""
    r = gitops_remote
    first = _run_gitops_sync(r, r.m3)
    assert first.pilot != r.m3 and _tree_without_pins(r.origin, first.pilot) == _tree_without_pins(r.origin, r.m3)
    # A fresh clone, as the next run's checkout is.
    import shutil

    shutil.rmtree(r.tmp / "pilot")
    later = _run_gitops_sync(r, r.m2 if case == "older" else r.dead)
    assert later.pilot == first.pilot, f"{case}: the deploy branch moved"
    names = [n for n, _ in later.ran]
    if case == "older":
        assert later.outputs["skip_check"]["skip"] == "true", later.outputs
    else:
        assert later.outputs["order"]["publish"] == "false", later.outputs
        said = next(res.stdout for n, res in later.ran if n == "Never move gitops/pilot backwards")
        assert "::notice::" in said and r.dead in said, said
    assert not any(n and n.startswith(("Merge", "Converge", "Push")) for n in names), names


def test_every_gitops_step_after_the_order_check_obeys_it():
    sync = _gitops_jobs()["sync"]
    names = [s.get("name") for s in sync["steps"]]
    order = names.index("Never move gitops/pilot backwards")
    assert sync["steps"][order]["id"] == "order" and names.index(
        "Skip if gitops/pilot already contains the gated commit") < order
    for step in sync["steps"][order + 1:]:
        cond = str(step.get("if", ""))
        assert "steps.order.outputs.publish != 'false'" in cond and "steps.skip_check.outputs.skip != 'true'" in cond, (
            f"{step.get('name')} runs even when the gated commit must not move the deploy branch")
    assert sync["concurrency"]["cancel-in-progress"] is False, "syncs run one at a time, in order"


# ── 6. the mirror's manual dispatch is checked (ENH-011 WB-1, N5) ───────────────────────────────────────────────

DISPATCH = "Dispatch the mirror's manual deploy"


def _dispatch(tmp_path, code: str, exit_code: int = 0):
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    calls = tmp_path / "curl-calls"
    (fake / "curl").write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >> {calls}\nprintf "%s" "{code}"\nexit {exit_code}\n')
    (fake / "curl").chmod(0o755)
    step = _step(_load("sync-to-clean.yml")["jobs"]["sync"], DISPATCH)
    res = subprocess.run(["bash", "-eo", "pipefail", "-c", step["run"]], cwd=tmp_path, capture_output=True, text=True,
                         env={**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "MIRROR": "owner/mirror",
                              "SYNC_TOKEN": "sync-token-value"}, check=False)
    return res, (calls.read_text().splitlines() if calls.exists() else [])


@pytest.mark.parametrize("code,exit_code,ok", [
    ("204", 0, True), ("200", 0, False), ("403", 0, False), ("404", 0, False), ("422", 0, False), ("500", 0, False),
    ("000", 7, False),
])
def test_a_refused_manual_dispatch_fails_the_mirror_sync(tmp_path, code, exit_code, ok):
    """(d) `deploy-manual` is dispatched once, at the mirror, with `ref: main`; anything but 204 fails the step."""
    res, calls = _dispatch(tmp_path, code, exit_code)
    assert (res.returncode == 0) is ok, f"HTTP {code}: rc {res.returncode}\n{res.stdout}\n{res.stderr}"
    assert len(calls) == 1, calls
    assert "https://api.github.com/repos/owner/mirror/actions/workflows/deploy-manual.yml/dispatches" in calls[0]
    assert '{"ref":"main"}' in calls[0] and "%{http_code}" in calls[0]
    if not ok:
        assert f"HTTP {code}" in res.stdout + res.stderr


def test_the_manual_dispatch_runs_after_the_leak_check():
    """A failed dispatch must not skip `Verify the published mirror`: a step after a failure does not run."""
    sync = _load("sync-to-clean.yml")["jobs"]["sync"]
    names = [s.get("name") for s in sync["steps"]]
    assert names.index(DISPATCH) > names.index("Verify the published mirror")
    assert "SYNC_TOKEN" in _step(sync, DISPATCH)["env"]
    assert not any("dispatches" in s.get("run", "") for s in sync["steps"] if s.get("name") != DISPATCH), (
        "a second, unchecked dispatch")
