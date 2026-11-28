"""Exported circuit and cluster definitions write their provenance datetimes as UTC WITH an offset (hotfix
fix/046-1-followups, operator decision (a), 2026-11-26).

The published `docs/schemas/cluster-definition-v1.json` and `circuit-definition-v1.json` declare
`provenance.created_at` and `provenance.exported_at` `format: date-time` (RFC 3339: an offset is required). The export
wrote `created_at` from a naive column (`datetime.utcnow` defaults), so a consumer checking formats refused the
document. `DefinitionProvenance` now gives a naive value its UTC offset.

1. A real export (rows in the test database, `GET /api/v1/cluster-profiles/{id}/export` and
   `GET /api/v1/circuits/{id}/export`, the route's own serializer) validates against the PUBLISHED schema with
   `tests/support/strict_formats.STRICT` (ajv-formats' full mode), and `created_at` is the row's instant in UTC.
2. Cross-repo: miServer's importer (its own models, `ClusterDefinitionV1` / `CircuitDefinitionV1.model_validate`, the
   parse its import routes run, in its own venv) accepts the new form AND the old naive form (stored documents
   exist), reading the same instant from both. Loud skip without the sibling; fails under the require switch.

Control: `_utc_with_offset` returns the value unchanged → test 1 red for both kinds.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jsonschema
import pytest

from tests.support.cross_repo import require_cross_repo
from tests.support.sibling_repos import miserver_package, miserver_repo
from tests.support.strict_formats import STRICT

DOCS = Path(__file__).resolve().parents[3] / "docs" / "schemas"
STAMP = datetime(2026, 11, 26, 21, 24, 31, 123456)  # naive, as the column stores it


async def _exports(client, session) -> dict[str, dict]:
    from src.models.circuit import Circuit
    from src.models.cluster_profile import ClusterProfile

    profile = ClusterProfile(name="utc profile", members=[{"feature_idx": 1, "strength": 1.0}], schema_version="1",
                             created_at=STAMP, updated_at=STAMP)
    circuit = Circuit(id="crc_utc", name="utc circuit", created_at=STAMP, updated_at=STAMP,
                      members=[{"layer": 0, "member_kind": "feature_ref", "feature": {"feature_idx": 0, "strength": 1.0}}],
                      edges=[], saes=[{"layer": 0, "sae_id": "sae_utc"}], model_id="m_utc",
                      model_hf_id="fixture-org/fixture-model")
    session.add_all([profile, circuit])
    await session.commit()
    out = {}
    for kind, path in (("cluster", f"/api/v1/cluster-profiles/{profile.id}/export"),
                       ("circuit", "/api/v1/circuits/crc_utc/export")):
        response = await client.get(path)
        assert response.status_code == 200, (kind, response.text[:300])
        out[kind] = response.json()
    return out


def _strict_errors(document: dict, schema_file: str) -> list[str]:
    schema = json.loads((DOCS / schema_file).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=STRICT)
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}" for e in validator.iter_errors(document)]


async def test_a_real_export_validates_against_its_published_schema_strictly(client, async_session):
    exports = await _exports(client, async_session)
    for kind, schema_file in (("cluster", "cluster-definition-v1.json"), ("circuit", "circuit-definition-v1.json")):
        document = exports[kind]
        assert not _strict_errors(document, schema_file), (kind, _strict_errors(document, schema_file))
        created = document["provenance"]["created_at"]
        assert datetime.fromisoformat(created) == STAMP.replace(tzinfo=timezone.utc), (kind, created)
        assert STRICT.conforms(document["provenance"]["exported_at"], "date-time"), document["provenance"]


def test_a_naive_value_gains_the_utc_offset_and_an_aware_one_is_kept():
    from src.schemas.cluster_profile import DefinitionProvenance

    dumped = DefinitionProvenance(created_at=STAMP).model_dump(mode="json")["created_at"]
    assert dumped == "2026-11-26T21:24:31.123456Z", dumped
    plus_two = datetime(2026, 11, 26, 23, 24, 31, tzinfo=timezone(timedelta(hours=2)))
    assert DefinitionProvenance(exported_at=plus_two).exported_at == plus_two  # its own offset is kept


_SCRIPT = r'''
import json, sys
pkg = sys.argv[2]
docs = json.load(open(sys.argv[1]))
cluster = __import__(pkg + ".api.schemas.cluster", fromlist=["ClusterDefinitionV1"])
circuit = __import__(pkg + ".api.schemas.circuit", fromlist=["CircuitDefinitionV1"])
models = {"cluster": cluster.ClusterDefinitionV1, "circuit": circuit.CircuitDefinitionV1}
out = {}
for name, document in docs.items():
    kind = name.split(":")[0]
    try:
        parsed = models[kind].model_validate(document)
        stamp = parsed.provenance.created_at
        out[name] = {"parsed": True, "created_at_utc": stamp.isoformat(),
                     "aware": stamp.tzinfo is not None}
    except Exception as exc:
        out[name] = {"parsed": False, "error": str(exc)[:400]}
print("@@IMPORT@@" + json.dumps(out))
'''


def _absent(reason: str) -> None:
    if require_cross_repo():
        pytest.fail("the cross-repo switch is set but miServer's import check cannot run: " + reason)
    pytest.skip("CROSS-REPO, SKIPPED LOUDLY: " + reason)


async def test_miservers_importer_accepts_the_new_form_and_the_old_naive_form(client, async_session, tmp_path):
    repo = miserver_repo()
    pkg = miserver_package(repo)
    if pkg is None:
        _absent(f"no miServer checkout at {repo} (set MISERVER_REPO)")
    python = Path(os.environ.get("MISERVER_PYTHON") or repo / "venv" / "bin" / "python")
    if not python.is_file():
        _absent(f"no miServer interpreter at {python} (set MISERVER_PYTHON)")
    exports = await _exports(client, async_session)
    documents = {}
    for kind, document in exports.items():
        documents[f"{kind}:new"] = document
        old = copy.deepcopy(document)
        old["provenance"]["created_at"] = STAMP.isoformat()  # what a stored, earlier export carries
        old["provenance"]["exported_at"] = STAMP.isoformat()
        documents[f"{kind}:old"] = old
    path = tmp_path / "definitions.json"
    path.write_text(json.dumps(documents))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_URL", "PYTHONPATH"))}
    proc = subprocess.run([str(python), "-B", "-c", _SCRIPT, str(path), pkg], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=240)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@IMPORT@@")]
    if proc.returncode != 0 or not lines:
        pytest.fail(f"miServer's import check did not run (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    out = json.loads(lines[0][len("@@IMPORT@@"):])
    assert set(out) == set(documents)
    for name, row in out.items():
        assert row["parsed"] is True, (name, row)
        read = datetime.fromisoformat(row["created_at_utc"])
        expected = STAMP.replace(tzinfo=timezone.utc) if row["aware"] else STAMP
        assert read == expected, (name, row)
    assert out["cluster:new"]["aware"] and out["circuit:new"]["aware"], out
