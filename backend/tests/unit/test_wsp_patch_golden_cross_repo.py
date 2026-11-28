"""ENH-010 044 task 8.1: miForge's own golden patches (`docs/schemas/examples/wsp-patch/{sparse,lowrank,dense}/`,
016 task 2.11a) re-apply bit-exactly through miWorkbench's `wsp_patch.apply_patch`, so the two implementations of
the published patch rules agree. Loud skip without miForge; fails under `MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1`."""
from __future__ import annotations

import json

import pytest

from src.services import wsp_patch as wp
from tests.support.cross_repo import REQUIRE_NEW, require_cross_repo
from tests.support.sibling_repos import miforge_repo

#: Golden folders. `lowrank_rank2` (miForge `3fcb7c8`, asked in NEEDS 7) has rank 2 with cancelling terms, so a
#: fused `base + a @ b` and the term-by-term sum differ in their bits: the rank-1 `lowrank` golden cannot see that.
#: `lowrank_rank2`'s terms turned out NOT to cancel (NEEDS 8), so `lowrank_cancel` (miForge `150af53`, wsp/016-P1a) is
#: the golden a fused reconstruction fails (044-C8.1-NC-fused-golden-cancel).
FORMS = ("sparse", "lowrank", "lowrank_rank2", "lowrank_cancel", "dense")


def _golden(form: str):
    folder = miforge_repo() / "docs" / "schemas" / "examples" / "wsp-patch" / form
    if not (folder / "patch.json").is_file():
        message = (f"miForge's golden patch is not at {folder} (set MIFORGE_REPO to a miForge checkout). Skipping is "
                   f"expected on the public mirror; set {REQUIRE_NEW}=1 to make this a failure.")
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    return folder


@pytest.mark.parametrize("form", FORMS)
def test_each_golden_reapplies_bit_exactly(form):
    folder = _golden(form)
    manifest = json.loads((folder / "patch.json").read_text())
    expected = json.loads((folder / "expected.json").read_text())
    base = wp.read_safetensors((folder / "base.safetensors").read_bytes())
    applied, edited = wp.apply_patch(manifest, (folder / "edit.npz").read_bytes(), base)
    assert applied["tensors"][0]["form"] == expected["form"] == form.split("_")[0]
    assert {name: edited[name].sha256 for name in expected["edited_sha256"]} == expected["edited_sha256"]
    assert {edited[name].dtype for name in expected["edited_sha256"]} == {expected["dtype"]}
