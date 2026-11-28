"""miForge 018 FU-4 (2026-11-25): what `import_wsp_candidate` reads from miForge, held to miForge's OWN code.

The tool resolves a candidate id (`wspc_…`) through miForge's API (`tools/wsp.py::resolve_candidate`); its unit tests
(`test_wsp_tools_registered.py`, the FU-4 section) serve those reads from a fake. This file checks the fake's picture
against the miForge checkout `tests/support/sibling_repos.py` finds (never a copy), so the two sides cannot drift
apart unseen (the 044a R3 lesson: each side tested against its own picture of the other):

* every path the tool reads is a GET route of miForge's live `/api/v1` router (its OpenAPI paths);
* the patch files the tool asks for are the names miForge serves (`PATCH_FILES`);
* the summary fields the tool reads are the keys miForge's own `_summary` builds;
* a miForge refusal is the envelope the tool parses (`{"error": {"code", "message", ...}}`), and an unknown candidate
  is a 4xx there, which the tool relays by its code;
* miForge's candidate ids match the tool's `candidate_id` pattern.

miForge is loaded only for this test (its top-level packages put on `sys.path` and taken out of `sys.modules` again).
Loud skip without miForge; fails under `MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1` (or the suite-wide switch).
"""
from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tests.support.cross_repo import REQUIRE_NEW, require_cross_repo
from tests.support.sibling_repos import miforge_repo


@contextmanager
def miforge():
    src = miforge_repo() / "backend" / "src"
    if not (src / "api" / "v1" / "endpoints" / "wsp_candidates.py").is_file():
        message = (f"miForge's candidate routes are not at {src} (set MIFORGE_REPO to a miForge checkout with 016 "
                   f"phase 2). Skipping is expected on the public mirror; set {REQUIRE_NEW}=1 to make this a failure.")
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    tops = {p.name for p in src.iterdir() if (p / "__init__.py").is_file()} | {p.stem for p in src.glob("*.py")}
    saved = {name: module for name, module in sys.modules.items() if name.split(".")[0] in tops}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(src))
    try:
        from fastapi import FastAPI

        import api.app as app_module
        import api.v1.endpoints.wsp_candidates as candidates
        import core.errors as errors
        import wsp.runs as runs
        from api.v1.router import api_v1_router

        app = FastAPI()
        app.include_router(api_v1_router)  # as `api.app.create_app` mounts it
        yield SimpleNamespace(paths=app.openapi()["paths"], candidates=candidates, errors=errors, runs=runs,
                              status_for=app_module.status_for)
    finally:
        sys.path.remove(str(src))
        for name in [n for n in sys.modules if n.split(".")[0] in tops]:
            del sys.modules[name]
        sys.modules.update(saved)


def _template(path: str) -> str:
    """The tool's concrete patch file path → miForge's route template (`patch/{name}`)."""
    return re.sub(r"/patch/[^/]+$", "/patch/{name}", path)


def test_every_path_the_tool_reads_is_a_miforge_get_route():
    from src.mcp_server.tools import wsp

    reads = [wsp.FORGE_SUMMARY, *wsp.FORGE_DOCUMENTS.values()]
    assert len(reads) == 4
    with miforge() as forge:
        for path in reads:
            route = "/api/v1" + _template(path)
            assert route in forge.paths, (route, sorted(p for p in forge.paths if "/wsp/candidates" in p))
            assert "get" in forge.paths[route], (route, sorted(forge.paths[route]))


def test_the_patch_files_the_tool_asks_for_are_the_ones_miforge_serves():
    from src.mcp_server.tools import wsp

    asked = {wsp.FORGE_DOCUMENTS[k].rsplit("/", 1)[1] for k in ("patch_manifest", "edit_npz_b64")}
    with miforge() as forge:
        assert asked == set(forge.candidates.PATCH_FILES), (asked, sorted(forge.candidates.PATCH_FILES))


def test_the_summary_fields_the_tool_reads_are_miforges():
    """`resolve_candidate` and `candidate_mismatch` read these keys (and `verdict.value`) from the summary."""
    row = SimpleNamespace(candidate_id="wspc_0123456789ab", run_id="wsp_0123456789ab", parent_kind="bundle",
                          parent_ref="b" * 64, bundle_base_sha256="b" * 64, patch_sha256="c" * 64,
                          changed_entries=3, tensors={"tensors": ["w"]}, created_at=None)
    with miforge() as forge:
        # miForge FU-7b (a66639d): the summary's verdict carries the standing, read from the stored document.
        verdict = SimpleNamespace(verdict_id="wsv_1", value="regressed", document={"value": "regressed", "per_set": []})
        summary = forge.candidates._summary(row, verdict)
        unverdicted = forge.candidates._summary(row, None)
    assert {"candidate_id", "run_id", "bundle_base_sha256", "patch_sha256", "verdict"} <= set(summary)
    assert summary["verdict"]["value"] == "regressed"
    assert summary["verdict"]["standing"] == "regressed"  # FU-7b's reading, beside the value the tool reads
    assert summary["candidate_id"] == row.candidate_id and summary["run_id"] == row.run_id
    assert summary["bundle_base_sha256"] == row.bundle_base_sha256 and summary["patch_sha256"] == row.patch_sha256
    assert unverdicted["verdict"] is None  # what `run_not_evaluated` reads


def test_an_unknown_candidate_is_miforges_4xx_refusal_in_the_envelope_the_tool_parses():
    with miforge() as forge:
        envelope = forge.errors.refuse("wsp_candidate_not_found", "No WSP candidate.").envelope()
        status = forge.status_for("wsp_candidate_not_found")
    assert 400 <= status < 500, status
    assert envelope["error"]["code"] == "wsp_candidate_not_found" and "message" in envelope["error"]


def test_miforges_candidate_ids_match_the_tools_pattern():
    from src.mcp_server.tools import wsp

    with miforge() as forge:
        ids = {forge.runs.new_candidate_id() for _ in range(32)}
    assert all(re.fullmatch(wsp.CANDIDATE_ID_PATTERN, i) for i in ids), sorted(ids)[:4]
