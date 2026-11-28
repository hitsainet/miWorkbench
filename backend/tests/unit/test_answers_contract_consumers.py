"""Each consumer's copy of miWorkbench's answer contract is the committed file, or one additive revision behind it
(046 FR-046.14, SH3; FTASKS 6.6). Ported from miServer's `tests/unit/test_answers_contract_consumers.py` (044).

`CONSUMERS` names each app that vendors `docs/schemas/miworkbench-answers-v1.json`, with its path and state:
* `pending`: the consumer has not vendored it yet; an absent copy passes (046-2: miAssist's 006-3 vendors it);
* `vendored`: an absent copy fails.

A present copy is compared with the committed contract through `contract_tools.classify` (the writer's own table):
equal passes; an additive difference passes and says the consumer must re-vendor; a breaking difference under the
same major fails, because a breaking change needs `miworkbench-answers-v2.json`. Without the sibling the test skips
loudly; under the cross-repo switch (`tests/support/cross_repo.py`) it fails instead.

Control: M17 (the comparison skips the major check: the planted breaking copy goes green).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from src.api import contract_tools
from src.api.answers_contract import CONTRACT_FILENAME
from tests.support.cross_repo import require_cross_repo
from tests.support.sibling_repos import miassist_repo

ROOT = Path(__file__).resolve().parents[3]
COMMITTED = ROOT / "docs" / "schemas" / CONTRACT_FILENAME

#: app -> (its copy's path in its repository, state). miAssist's 006 FTID §3.1 vendors under `docs/schemas/`.
CONSUMERS = {
    # The path follows `CONTRACT_FILENAME`, so a v2 file is compared with the v2 copy (miServer 044 R2-3).
    "miAssist": ("docs/schemas/" + CONTRACT_FILENAME, "pending"),
}
_REPOS = {"miAssist": miassist_repo}


def require_or_skip(message: str) -> None:
    if require_cross_repo():
        pytest.fail(message)
    pytest.skip(message)


def compare(committed: dict, copy: dict | None, state: str) -> tuple[bool, str]:
    """(passes, message) for one consumer's copy."""
    if copy is None:
        if state == "pending":
            return True, "not vendored yet (pending)"
        return False, "the consumer's copy is missing, and it is vendored"
    if copy.get("$id") != committed["$id"]:
        # 044 R1-6: the vendored file is THIS major's name, so its `$id` must be this major's too; a
        # copy under another `$id` is a corrupted or mis-vendored file, not one to skip.
        return False, f"the copy's $id is {copy.get('$id')!r}, not {committed['$id']!r}"
    files = {sid: row["file"] for doc in (copy, committed) for sid, row in (doc.get("external") or {}).items()}
    changes = contract_tools.classify(_body(copy), _body(committed),
                                      contract_tools.git_resolver(ROOT, files))
    if not changes:
        return True, "equal"
    bad = contract_tools.breaking(changes)
    if bad:
        return False, (
            "breaking under the same major (a breaking change needs miworkbench-answers-v2.json):\n"
            + "\n".join(f"  {c}" for c in bad)
        )
    return True, (
        f"additive: the consumer must re-vendor (its revision {copy.get('revision')}, "
        f"committed {committed.get('revision')})"
    )


def _body(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in contract_tools.UNDIGESTED}


@pytest.mark.parametrize("app", sorted(CONSUMERS))
def test_each_consumers_copy_is_the_committed_contract_or_behind_it_additively(app: str):
    path, state = CONSUMERS[app]
    repo = _REPOS[app]()
    if not repo.is_dir():
        require_or_skip(
            f"{app} is not checked out beside this repository ({repo}); set its _REPO path"
        )
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    vendored = repo / path
    copy = json.loads(vendored.read_text(encoding="utf-8")) if vendored.is_file() else None
    ok, message = compare(committed, copy, state)
    assert ok, f"{app} ({vendored}): {message}"
    if message.startswith("additive"):
        warnings.warn(f"{app}: {message}", stacklevel=1)


# ── the comparison itself, on planted copies (tmp_path; no sibling needed) ───────────────────────────


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


def _planted(tmp_path, doc: dict | None) -> dict | None:
    if doc is None:
        return None
    path = tmp_path / CONTRACT_FILENAME
    path.write_text(contract_tools.render(doc), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def test_an_equal_copy_passes(committed, tmp_path):
    assert compare(committed, _planted(tmp_path, committed), "vendored") == (True, "equal")


def test_a_copy_one_additive_revision_behind_passes_and_says_re_vendor(committed, tmp_path):
    behind = json.loads(json.dumps(committed))
    behind["responses"].pop(sorted(behind["responses"])[0])  # the committed file ADDS this route
    behind["revision"] = committed["revision"] - 1 if committed["revision"] > 1 else 0
    ok, message = compare(committed, _planted(tmp_path, behind), "vendored")
    assert ok and "re-vendor" in message, message


def test_a_breaking_copy_under_the_same_major_fails(committed, tmp_path):
    ahead = json.loads(json.dumps(committed))
    ahead["responses"]["GET /api/a-route-the-server-dropped"] = {"200": {"type": "object"}}
    ok, message = compare(committed, _planted(tmp_path, ahead), "vendored")
    assert not ok and "breaking" in message, message


def test_a_copy_under_another_id_fails(committed, tmp_path):
    other = {
        **json.loads(json.dumps(committed)),
        "$id": "https://hitsai.net/schemas/elsewhere.json",
    }
    ok, message = compare(committed, _planted(tmp_path, other), "vendored")
    assert not ok and "$id" in message, message


def test_an_absent_copy_passes_only_while_pending(committed):
    assert compare(committed, None, "pending")[0] is True
    assert compare(committed, None, "vendored")[0] is False


def test_no_sibling_skips_loudly_or_fails_under_the_switch(monkeypatch, tmp_path):
    import tests.unit.test_answers_contract_consumers as me

    from tests.support.cross_repo import REQUIRE_NEW, REQUIRE_OLD

    monkeypatch.setitem(me._REPOS, "miAssist", lambda: tmp_path / "absent")
    for switch in (REQUIRE_NEW, REQUIRE_OLD):  # the switch's names, from the one module that holds them
        monkeypatch.delenv(switch, raising=False)
    with pytest.raises(pytest.skip.Exception):
        me.test_each_consumers_copy_is_the_committed_contract_or_behind_it_additively("miAssist")
    monkeypatch.setenv(REQUIRE_OLD, "1")
    with pytest.raises(pytest.fail.Exception):
        me.test_each_consumers_copy_is_the_committed_contract_or_behind_it_additively("miAssist")


def test_each_consumer_path_names_this_majors_file():
    """miServer 044 R2-3: a consumer's copy is the file of the major this repository publishes."""
    for app, (path, _state) in CONSUMERS.items():
        assert Path(path).name == CONTRACT_FILENAME, app
