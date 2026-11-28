"""ENH-010 (044 task 9.5, P13/P25; R1A-W13): a DERIVED probe definition imports into miServer — its own vendored
schema, its own model and its own digest, in its own venv.

Until miServer re-vendors `probe-definition-v1.json` (043 7.10, push 043-3), its `Provenance` refuses
`derived_from` (`additionalProperties: false`) — the SAFE direction: no derived definition is sent to miServer before
then. So this test is a STRICT expected failure whose condition is COMPUTED: miServer's vendored `Provenance` lacks
`derived_from`. When miServer moves, the condition is false and the test must pass; a pass while the condition is true
fails the suite (`strict=True`), so a miServer that silently started accepting unknown keys is caught too.

Loud skip without the sibling or its interpreter; fails under `MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1`.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from src.schemas.probe_identity import document_identity
from tests.support.cross_repo import require_cross_repo
from tests.support.sibling_repos import miserver_package, miserver_repo
from tests.unit.test_probe_identity_cases import _residual

DERIVED_FROM = {
    "parent_probe_id": "pm_parent", "parent_detector_digest": "e" * 64, "patch_sha256": "f" * 64,
    "provider": "weightspace", "provider_version": "0.3.0",
    "verdict": {"value": "improved", "run_sha256": "9" * 64},
}

_SCRIPT = r'''
import json, sys
import jsonschema
pkg = sys.argv[2]
document = json.load(open(sys.argv[1]))
schema = json.load(open("docs/schemas/probe-definition-v1.json"))
out = {}
try:
    jsonschema.validate(document, schema); out["schema"] = "valid"
except jsonschema.ValidationError as exc:
    out["schema"] = "refused: " + exc.message[:300]
model = __import__(pkg + ".api.schemas.probe", fromlist=["ProbeDefinitionV1"])
try:
    parsed = model.ProbeDefinitionV1.model_validate(document)
    dumped = parsed.model_dump(mode="json")
    out["model"] = "valid"
    out["round_trip_keeps_lineage"] = dumped.get("provenance", {}).get("derived_from") == document["provenance"]["derived_from"]
except Exception as exc:
    out["model"] = "refused: " + str(exc)[:300]
digest = __import__(pkg + ".services.probe_digest", fromlist=["detector_digest", "decision_digest"])
out["detector_digest"] = digest.detector_digest(document)
out["decision_digest"] = digest.decision_digest(document.get("decision"))
print("@@DERIVED@@" + json.dumps(out))
'''


def _absent(reason: str) -> None:
    if require_cross_repo():
        pytest.fail("MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1 but the derived-definition import check cannot run: "
                    + reason)
    pytest.skip("CROSS-REPO, SKIPPED LOUDLY: " + reason)


def _miserver_lacks_derived_from() -> bool:
    """The xfail condition, computed from miServer's vendored schema (True also when the sibling is absent: the test
    then skips before the condition matters)."""
    path = miserver_repo() / "docs" / "schemas" / "probe-definition-v1.json"
    if not path.is_file():
        return True
    provenance = json.loads(path.read_text())["$defs"]["Provenance"]["properties"]
    return "derived_from" not in provenance


@pytest.mark.xfail(_miserver_lacks_derived_from(), strict=True,
                   reason="miServer has not re-vendored probe-definition-v1.json with derived_from (043 7.10, 043-3)")
def test_a_derived_definition_imports_into_miserver(tmp_path):
    repo = miserver_repo()
    pkg = miserver_package(repo)
    python = Path(os.environ.get("MISERVER_PYTHON") or repo / "venv" / "bin" / "python")
    if pkg is None:
        _absent(f"no miServer checkout at {repo} (set MISERVER_REPO)")
    if not python.is_file():
        _absent(f"no miServer interpreter at {python} (set MISERVER_PYTHON)")
    document = copy.deepcopy(_residual())
    document["provenance"]["derived_from"] = copy.deepcopy(DERIVED_FROM)
    path = tmp_path / "derived.probe.json"
    path.write_text(json.dumps(document))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_URL", "PYTHONPATH"))}
    proc = subprocess.run([str(python), "-B", "-c", _SCRIPT, str(path), pkg], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=240)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@DERIVED@@")]
    if proc.returncode != 0 or not lines:
        pytest.fail(f"miServer's check did not run (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    out = json.loads(lines[0][len("@@DERIVED@@"):])
    assert out["schema"] == "valid", out
    assert out["model"] == "valid", out
    assert out["round_trip_keeps_lineage"] is True, "miServer's hand mirror dropped derived_from on re-export"
    ours = document_identity(document)
    assert (out["detector_digest"], out["decision_digest"]) == ours, "the two repos disagree about the digests"
