"""miForge's three WSP schemas are VENDORED byte-identical, never generated (044 tasks 2.1 – 2.3; FR-2, FR-8).

miForge OWNS `docs/schemas/miforge-wsp-{bundle,patch,run}-v1.json` (016 task 11.3, push 016-P1a). miWorkbench
carries two copies of each, both copied from miForge and never regenerated here:

* `docs/schemas/…` — the vendored contract, compared with miForge's file;
* `backend/src/schemas/vendored/…` — the PACKAGE copy the image reads (the build context is `backend/`, so
  `docs/` never reaches the image).

Three states for the cross-repo comparison, through `MIFORGE_REPO` (default: the `miForge` sibling of this
checkout, `tests/support/sibling_repos.py::miforge_repo`):

* present → the bytes must match;
* absent (or present without the file) → a LOUD skip naming the missing path and the flag;
* absent with `MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1` → fail.

⚠ ON THE PUBLIC MIRROR THE CROSS-REPO TEST SKIPS BY DESIGN: miForge is not checked out there. The in-repo
identity test still runs there.

Push order: miForge first (016-P1a), then miWorkbench (044-V); the vendored bytes came from miForge commit
`1a84f6a` (branch `wsp/016-P1a`, 2026-11-16). Re-vendored 2026-11-19 (miForge F-2): `miforge-wsp-run-v1.json`
from miForge branch `wsp/016-P2-main`, whose `SetVerdict.truncated_fired` is a map (head label -> count); push
miForge's 016-P2 first, then this branch. Re-vendored 2026-11-25 (FU-7a C2): `miforge-wsp-bundle-v1.json` from miForge
branch `feat/fu7a-c2`, whose `Practice.supplements` (validation-only practice rows, an additive revision within v1)
this app writes; push miForge's C2 first, then this branch (the byte-identity test is red in whichever moves second).

Re-vendored 2026-11-25 (miForge's run-contract revision 2, branch `feat/run-contract-r2`): `FrozenRequest.python`
(018 FU-8) and the document's `submitted_check` (018 FU-2), both optional. Push miForge first, then this branch.
`test_the_run_contract_is_the_recorded_revision` pins the bytes here too, because this repo's CI has no miForge
checkout and the byte-identity test above skips there.

Re-vendored 2026-11-26 (miForge F3 `2b31b2f`, operator decision 2026-11-26): all three files gain an absolute `$id`
(`https://hitsai.net/schemas/miforge-wsp-{bundle,patch,run}-v1.json`); every `$ref` in them is document-local
(`#/$defs/...`), so validation reads the same. Push order: miForge F3 first, then this train (W2c).

Re-vendored 2026-11-28 (miForge's run-contract revision 4, branch `fix/wsp-window-read-not-resident` `7ce80f9e`):
`Head.identical_to_parent` (optional boolean, on a `baseline:<kind>` head: its `scores_sha256` equals the parent's on
the set, miForge fix 8), and revision-4 descriptions of `submitted_check` for an inspect that spent no forward pass.
The revision-3 copy closed `Head`, so every verdict snapshot miForge writes after it deploys was refused on import
(`test_wsp_import.py::test_a_rev4_snapshot_with_identical_to_parent_and_a_bundle_zeroed_parent_imports`). The bundle
and patch contracts are unchanged on that branch. Push order: miForge first, then this branch, right after it (no
candidate evaluated in between).
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

import pytest

from tests.support.cross_repo import REQUIRE_NEW, REQUIRE_SWITCH, require_cross_repo
from tests.support.sibling_repos import MIFORGE_CANDIDATES, miforge_repo

REPO = pathlib.Path(__file__).resolve().parents[3]
BACKEND = REPO / "backend"
NAMES = ("miforge-wsp-bundle-v1.json", "miforge-wsp-patch-v1.json", "miforge-wsp-run-v1.json")
FLAG = REQUIRE_NEW


def _docs(name: str) -> pathlib.Path:
    return REPO / "docs" / "schemas" / name


def _package(name: str) -> pathlib.Path:
    return BACKEND / "src" / "schemas" / "vendored" / name


def _theirs(root: pathlib.Path, name: str) -> pathlib.Path:
    return root / "docs" / "schemas" / name


def _compare_with_sibling(root: pathlib.Path, name: str) -> None:
    theirs = _theirs(root, name)
    if not theirs.is_file():
        message = (
            f"miForge's {name} is not at {theirs} (set MIFORGE_REPO to a miForge checkout). Skipping is "
            f"expected on the public mirror; set {FLAG}=1 to make this a failure."
        )
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    assert _package(name).read_bytes() == theirs.read_bytes(), (
        f"miWorkbench's vendored {name} differs from miForge's. Copy miForge's file FROM {theirs} INTO "
        f"{_docs(name)} and {_package(name)} — never edit or regenerate it here; miForge owns the contract."
    )


@pytest.mark.parametrize("name", NAMES)
def test_the_package_copy_equals_the_docs_copy(name):
    assert _docs(name).is_file() and _package(name).is_file(), name
    assert _docs(name).read_bytes() == _package(name).read_bytes(), (
        "the image reads the package copy and the contract is the docs copy — the same bytes, both from miForge"
    )


@pytest.mark.parametrize("name", NAMES)
def test_each_copy_is_a_json_schema(name):
    from jsonschema.validators import validator_for

    schema = json.loads(_package(name).read_text(encoding="utf-8"))
    validator_for(schema).check_schema(schema)


@pytest.mark.parametrize("name", NAMES)
def test_it_is_byte_identical_to_miForge(name):
    # Skips on the public mirror by design (no sibling there) — see the module docstring.
    _compare_with_sibling(miforge_repo(), name)


#: `miforge-wsp-run-v1.json`'s sha256, revision 4 (miForge `fix/wsp-window-read-not-resident` `7ce80f9e`, 2026-11-28:
#: `Head.identical_to_parent`); revision 3 (F3 `2b31b2f`, the absolute `$id`) was `476d34efeef0…`, revision 2
#: `5a8870118fcd…`, revision 1 `a3fc40105d09…`. It moves only with a re-vendor from miForge, in miForge's push order.
RUN_CONTRACT_SHA256 = "05fc289aa9420d1388a1646f9df3dc739a03cb1e4fd058e28636a81ab8e4a5bb"


def test_the_run_contract_is_the_recorded_revision():
    for path in (_docs(NAMES[2]), _package(NAMES[2])):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == RUN_CONTRACT_SHA256, path


def test_miforge_repo_reads_its_variable(monkeypatch, tmp_path):
    assert MIFORGE_CANDIDATES == (("miForge", "MIFORGE_REPO"),)
    monkeypatch.setenv("MIFORGE_REPO", str(tmp_path))
    assert miforge_repo() == tmp_path


def test_an_absent_sibling_SKIPS_loudly(tmp_path, monkeypatch):
    for _switch in REQUIRE_SWITCH:
        monkeypatch.delenv(_switch, raising=False)
    with pytest.raises(pytest.skip.Exception) as skipped:
        _compare_with_sibling(tmp_path, NAMES[0])
    assert str(_theirs(tmp_path, NAMES[0])) in str(skipped.value) and FLAG in str(skipped.value)


def test_an_absent_sibling_FAILS_under_the_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(pytest.fail.Exception):
        _compare_with_sibling(tmp_path, NAMES[0])


def test_a_differing_byte_fails_and_says_copy_FROM_miForge(tmp_path):
    theirs = _theirs(tmp_path, NAMES[1])
    theirs.parent.mkdir(parents=True)
    theirs.write_bytes(_package(NAMES[1]).read_bytes() + b" ")
    with pytest.raises(AssertionError, match="FROM"):
        _compare_with_sibling(tmp_path, NAMES[1])


def test_nothing_in_src_WRITES_any_copy():
    """Vendored, never generated: no `write_text`/`write_bytes`/`dump`/`open(..., "w")` in a module of
    `backend/src` that names one of the files."""
    offenders = []
    for path in sorted((BACKEND / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if not any(name in text for name in NAMES):
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            called = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if called in ("write_text", "write_bytes", "dump"):
                offenders.append(f"{path.relative_to(BACKEND)}:{node.lineno}")
            if called == "open" and any(
                isinstance(a, ast.Constant) and isinstance(a.value, str) and "w" in a.value for a in node.args[1:]
            ):
                offenders.append(f"{path.relative_to(BACKEND)}:{node.lineno}")
    assert not offenders, offenders


def test_a_run_document_naming_miwsp_engine_is_accepted_by_the_real_importer():
    """miForge 018 (FR-018.21): miForge's own engine is a provider id like any other. `Provider.id` is a free string
    of at most 64 characters, so a run document from `miwsp_engine` 0.1.0 passes miWorkbench's real import validator
    (the package copy the image reads), with the provider named in the request, the lineage and the patch's
    `created_by`. A future enum on `Provider.id` would refuse every engine run; this test goes red first."""
    from types import SimpleNamespace

    from src.services.wsp_import import run_sha256_of, validate_run_document
    from src.services.wsp_patch import jcs_sha256
    from tests.support.wsp_candidate import run_document

    bundle_sha = "b" * 64
    bundle = SimpleNamespace(
        id="wsb_engine_case",
        bundle_sha256=bundle_sha,
        manifest={"heldout_ref": {"sets": [{"name": "heldout", "role": "eval", "rows": 16}]}},
    )
    manifest = {"content_sha256": "c" * 64, "parent_patch_sha256": None, "patch_sha256": "d" * 64}
    document = run_document(bundle, manifest, run_id="wsp_000000000018")
    engine = {"id": "miwsp_engine", "version": "0.1.0"}
    document["request"]["provider"] = dict(engine)
    document["request_sha256"] = jcs_sha256(document["request"])
    document["verdict"]["request_sha256"] = document["request_sha256"]
    document["lineage"].update(provider=engine["id"], provider_version=engine["version"])
    document["run_sha256"] = run_sha256_of(document)

    accepted = validate_run_document(document)

    assert accepted["request"]["provider"] == engine
    assert (accepted["lineage"]["provider"], accepted["lineage"]["provider_version"]) == ("miwsp_engine", "0.1.0")
