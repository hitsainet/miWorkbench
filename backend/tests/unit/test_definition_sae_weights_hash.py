"""ENH-011 X-2 (#12): an exported circuit or cluster names the hash of the SAE weights it was built on.

`DefinitionSAERef` carried `mistudio_sae_id`, `layer`, `hook_type`, `n_features`, `d_model` and `source_hint`, and no
hash of the weights. A consumer binding by those fields can bind a circuit or a cluster to a DIFFERENT SAE of the same
shape (another training run, another release of a repository), and nothing in the document can tell. The probe
definition already carries `sae.weights_sha256` (`probe_definition_builder._sae_weights_sha256`); the circuit and cluster
definitions now carry the same hash, from the same function, as the optional `weights_sha256` (planner default E11-6:
additive and optional; a consumer compares it at bind as a warning in its first release).

Every export is read through its REAL route and its MCP tool (the live registry with the deployed categories, over
ASGI to the app on the test database), and validated against the PUBLISHED schema strictly:

1. a cluster export (`GET /cluster-profiles/{id}/export`, `export_cluster_definition`), a bundle
   (`POST /cluster-profiles/export-bundle`), a circuit export (`GET /circuits/{id}/export`,
   `export_circuit_definition`) and its slices (`POST /circuits/{id}/export-slices`, `export_circuit_slices`) each
   carry the hash of THEIR SAE's file. The fixture is one repository published per layer with the ordering trap of
   `test_probe_sae_hash_is_the_right_layer.py` (`layer_10` sorts before `layer_9`), so a hash read from the wrong layer,
   or one SAE's hash given to every layer, is red;
2. an SAE whose weights cannot be hashed (no local copy) still exports, without the field: the field is optional, and
   a refusal would break exports that work today. A value carried in by an import is kept when no local SAE answers
   for it; a stored value is replaced by the local file's hash when one does;
3. both published schemas accept the old form (no field) and the new one, and refuse a malformed hash;
4. cross-repo: miServer's importer (its own models, in its own venv) parses the new form, and the new exports validate
   against the schema files miServer vendors today, so the producer-first push breaks no import there.

Controls: 0xcc/reviews/ENH-011_X-2_controls_2026-11-27.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import jsonschema
import pytest
import yaml

from tests.support.cross_repo import require_cross_repo
from tests.support.sibling_repos import miserver_package, miserver_repo
from tests.support.strict_formats import STRICT

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs" / "schemas"
SCHEMAS = {"cluster": "cluster-definition-v1.json", "circuit": "circuit-definition-v1.json"}
LAYERS = (9, 10)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sae_files(tmp_path, monkeypatch):
    """One repository published per layer, downloaded once: `local_path` is the repository, `hf_filepath` the layer.

    ⚠ `layer_10` sorts before `layer_9`, and each layer's bytes differ, so the fixture cannot agree by construction:
    the first file of the tree, or one SAE's hash reused for another layer, is a wrong answer here.
    `data_dir` is patched, not `resolve_data_path`, so the resolution production performs is the one tested."""
    from src.core.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    digests = {}
    for layer, payload in ((9, b"NINE" * 4096), (10, b"TEN!" * 4096), (11, b"ELVN" * 4096)):
        path = tmp_path / "saes" / "x2_repo" / f"layer_{layer}" / "sae_weights.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        digests[layer] = hashlib.sha256(payload).hexdigest()
    assert sorted(["layer_9", "layer_10"])[0] == "layer_10", "the fixture lost the ordering trap it exists for"
    assert len(set(digests.values())) == len(digests)
    return digests


async def _seed_saes(session) -> dict[int, object]:
    from src.models.external_sae import ExternalSAE

    rows = {}
    for layer in LAYERS:
        rows[layer] = ExternalSAE(
            id=f"sae_x2_l{layer}", name=f"x2 layer {layer}", source="huggingface", status="ready",
            hf_repo_id="fixture-org/x2-saes", hf_filepath=f"layer_{layer}", local_path="saes/x2_repo",
            model_name="fixture-org/fixture-model", layer=layer, hook_type="residual", n_features=64, d_model=8)
    # An SAE with no local copy: its hash cannot be computed, and its definitions must still export.
    rows["remote"] = ExternalSAE(
        id="sae_x2_remote", name="x2 remote", source="huggingface", status="pending",
        hf_repo_id="fixture-org/x2-saes", hf_filepath="layer_11", local_path=None,
        model_name="fixture-org/fixture-model", layer=11, hook_type="residual", n_features=64, d_model=8)
    session.add_all(rows.values())
    await session.commit()
    return rows


async def _seed_profiles(session) -> dict[object, str]:
    from src.models.cluster_profile import ClusterProfile

    ids = {}
    for key, sae_id in ((9, "sae_x2_l9"), (10, "sae_x2_l10"), ("remote", "sae_x2_remote")):
        profile = ClusterProfile(name=f"x2 profile {key}", sae_id=sae_id, schema_version="1",
                                 members=[{"feature_idx": 3, "strength": 1.0}])
        session.add(profile)
        await session.flush()
        ids[key] = profile.id
    await session.commit()
    return ids


def _circuit_members(layers) -> list[dict]:
    return [{"layer": layer, "member_kind": "feature_ref", "feature": {"feature_idx": layer, "strength": 1.0}}
            for layer in layers]


async def _seed_circuit(session, circuit_id: str, saes: list[dict]) -> None:
    from src.models.circuit import Circuit

    session.add(Circuit(id=circuit_id, name=f"x2 {circuit_id}", saes=saes,
                        members=_circuit_members([s["layer"] for s in saes]), edges=[], model_id="m_x2",
                        model_hf_id="fixture-org/fixture-model"))
    await session.commit()


def _strict_errors(document: dict, schema: dict) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema, format_checker=STRICT)
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}" for e in validator.iter_errors(document)]


def _published(kind: str) -> dict:
    return json.loads((DOCS / SCHEMAS[kind]).read_text(encoding="utf-8"))


def _assert_valid(document: dict, kind: str) -> None:
    errors = _strict_errors(document, _published(kind))
    assert not errors, (kind, errors)


# ── 1. every export names its own SAE's hash ─────────────────────────────────


async def test_a_cluster_export_names_the_hash_of_its_saes_weights(client, async_session, sae_files):
    from src.services.probe_definition_builder import _sae_weights_sha256

    rows = await _seed_saes(async_session)
    profiles = await _seed_profiles(async_session)
    for layer in LAYERS:
        response = await client.get(f"/api/v1/cluster-profiles/{profiles[layer]}/export")
        assert response.status_code == 200, response.text[:300]
        document = response.json()
        _assert_valid(document, "cluster")
        assert document["sae"]["mistudio_sae_id"] == f"sae_x2_l{layer}"
        # The file's own bytes, and the function the probe builder pins its definitions with: the same value.
        assert document["sae"]["weights_sha256"] == sae_files[layer] == _sae_weights_sha256(rows[layer]), document["sae"]


async def test_a_bundle_names_each_definitions_own_hash(client, async_session, sae_files):
    await _seed_saes(async_session)
    profiles = await _seed_profiles(async_session)
    response = await client.post("/api/v1/cluster-profiles/export-bundle",
                                 json={"ids": [profiles[10], profiles[9]]})
    assert response.status_code == 200, response.text[:300]
    bundle = response.json()
    _assert_valid(bundle, "cluster")
    got = [(d["sae"]["mistudio_sae_id"], d["sae"].get("weights_sha256")) for d in bundle["definitions"]]
    assert got == [("sae_x2_l10", sae_files[10]), ("sae_x2_l9", sae_files[9])], got


async def test_a_circuit_export_names_each_layers_own_hash(client, async_session, sae_files):
    await _seed_saes(async_session)
    await _seed_circuit(async_session, "crc_x2_two", [{"layer": 9, "sae_id": "sae_x2_l9"},
                                                      {"layer": 10, "sae_id": "sae_x2_l10"}])
    response = await client.get("/api/v1/circuits/crc_x2_two/export")
    assert response.status_code == 200, response.text[:300]
    document = response.json()
    _assert_valid(document, "circuit")
    got = {s["layer"]: (s["mistudio_sae_id"], s.get("weights_sha256")) for s in document["saes"]}
    assert got == {9: ("sae_x2_l9", sae_files[9]), 10: ("sae_x2_l10", sae_files[10])}, got


async def test_each_circuit_slice_names_its_layers_hash(client, async_session, sae_files):
    await _seed_saes(async_session)
    await _seed_circuit(async_session, "crc_x2_slices", [{"layer": 9, "sae_id": "sae_x2_l9"},
                                                         {"layer": 10, "sae_id": "sae_x2_l10"}])
    response = await client.post("/api/v1/circuits/crc_x2_slices/export-slices")
    assert response.status_code == 200, response.text[:300]
    slices = response.json()["slices"]
    for document in slices:
        _assert_valid(document, "cluster")
    got = sorted((s["sae"]["layer"], s["sae"].get("weights_sha256")) for s in slices)
    assert got == [(9, sae_files[9]), (10, sae_files[10])], got


# ── the MCP tools, through the live registry ─────────────────────────────────


def _deployed_categories() -> str:
    """MCP_TOOL_CATEGORIES exactly as the manifest ArgoCD deploys sets it."""
    doc = next(d for d in yaml.safe_load_all((REPO / "k8s/base/mcp.yaml").read_text())
               if d and d.get("kind") == "Deployment")
    env = doc["spec"]["template"]["spec"]["containers"][0]["env"]
    return next(e["value"] for e in env if e["name"] == "MCP_TOOL_CATEGORIES")


@pytest.fixture
def mcp_server(client, monkeypatch):
    """`build_server()` with the deployed categories, its miWorkbench client pointed at the app over ASGI. `client`
    is requested first, so the app's database dependency is the test session's."""
    from src.main import app
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    for var in ("MILLM_API_URL", "MISERVER_API_URL", "MIDATAWORKS_API_URL"):
        monkeypatch.delenv(var, raising=False)
    settings = MCPSettings(tool_categories=_deployed_categories(), allow_anonymous=True, _env_file=None)
    mcp, backend = build_server(settings, stdio=True)
    requests: list[tuple[str, str]] = []

    async def record(request):
        requests.append((request.method, request.url.path))

    backend._http = httpx.AsyncClient(  # noqa: SLF001 — the tools hold this client
        transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1", event_hooks={"request": [record]})
    return SimpleNamespace(mcp=mcp, requests=requests)


async def test_the_mcp_export_tools_answer_the_hash(client, async_session, sae_files, mcp_server):
    from mcp import Client

    await _seed_saes(async_session)
    profiles = await _seed_profiles(async_session)
    await _seed_circuit(async_session, "crc_x2_mcp", [{"layer": 9, "sae_id": "sae_x2_l9"},
                                                      {"layer": 10, "sae_id": "sae_x2_l10"}])
    async with Client(mcp_server.mcp) as session:
        listed = {t.name for t in (await session.list_tools()).tools}
        assert {"export_cluster_definition", "export_circuit_definition", "export_circuit_slices"} <= listed

        async def call(tool, args):
            result = await session.call_tool(tool, args)
            assert not result.is_error, (tool, result.content)
            return json.loads(result.content[0].text)

        cluster = await call("export_cluster_definition", {"profile_id": profiles[9]})
        circuit = await call("export_circuit_definition", {"circuit_id": "crc_x2_mcp"})
        sliced = await call("export_circuit_slices", {"circuit_id": "crc_x2_mcp"})

    assert cluster["sae"]["weights_sha256"] == sae_files[9], cluster["sae"]
    assert {s["layer"]: s.get("weights_sha256") for s in circuit["saes"]} == {9: sae_files[9], 10: sae_files[10]}
    assert sorted((s["sae"]["layer"], s["sae"].get("weights_sha256")) for s in sliced["slices"]) == [
        (9, sae_files[9]), (10, sae_files[10])]
    # One request per call, to the export route each tool names.
    exports = [r for r in mcp_server.requests if "export" in r[1]]
    assert exports == [("GET", f"/api/v1/cluster-profiles/{profiles[9]}/export"),
                       ("GET", "/api/v1/circuits/crc_x2_mcp/export"),
                       ("POST", "/api/v1/circuits/crc_x2_mcp/export-slices")], exports


# ── 2. what cannot be hashed, and what was carried in ────────────────────────


async def test_an_sae_with_no_local_copy_still_exports_without_the_field(client, async_session, sae_files):
    await _seed_saes(async_session)
    profiles = await _seed_profiles(async_session)
    response = await client.get(f"/api/v1/cluster-profiles/{profiles['remote']}/export")
    assert response.status_code == 200, response.text[:300]
    document = response.json()
    _assert_valid(document, "cluster")
    assert document["sae"]["mistudio_sae_id"] == "sae_x2_remote"
    assert "weights_sha256" not in document["sae"], document["sae"]


async def test_an_imported_hash_is_kept_when_no_local_sae_answers_for_it(client, async_session, sae_files):
    """The import route stores the SAE refs as the document gave them; an id this instance does not hold cannot be
    re-hashed here, so the hash the source vouched for travels on (the round trip miServer's half also keeps)."""
    await _seed_saes(async_session)
    foreign = "a" * 64
    document = {
        "kind": "mistudio.circuit-definition", "schema_version": "1", "name": "x2 imported",
        "saes": [{"mistudio_sae_id": "sae_elsewhere", "layer": 9, "weights_sha256": foreign},
                 {"mistudio_sae_id": "sae_x2_l10", "layer": 10, "weights_sha256": "b" * 64}],
        "members": _circuit_members(LAYERS), "edges": [],
    }
    imported = await client.post("/api/v1/circuits/import", json=document)
    assert imported.status_code == 201, imported.text[:300]
    circuit_id = imported.json()["id"]
    exported = await client.get(f"/api/v1/circuits/{circuit_id}/export")
    assert exported.status_code == 200, exported.text[:300]
    _assert_valid(exported.json(), "circuit")
    got = {s["layer"]: s.get("weights_sha256") for s in exported.json()["saes"]}
    # Layer 9 names an SAE this instance does not hold: the carried hash. Layer 10 names a local SAE: its file's hash
    # replaces the stored one, which named other bytes.
    assert got == {9: foreign, 10: sae_files[10]}, got


async def test_a_circuit_naming_an_sae_this_instance_lacks_exports_without_the_field(client, async_session,
                                                                                    sae_files):
    await _seed_circuit(async_session, "crc_x2_absent", [{"layer": 9, "sae_id": "mistudio--sae-elsewhere--layer_9"}])
    response = await client.get("/api/v1/circuits/crc_x2_absent/export")
    assert response.status_code == 200, response.text[:300]
    _assert_valid(response.json(), "circuit")
    assert response.json()["saes"][0].get("weights_sha256") is None, response.json()["saes"]


# ── 3. the published schemas: old form, new form, malformed ──────────────────


def _minimal(kind: str, sae: dict) -> dict:
    if kind == "cluster":
        return {"kind": "mistudio.cluster-definition", "schema_version": "1", "name": "x2",
                "sae": sae, "members": [{"feature_idx": 1, "strength": 1.0}]}
    return {"kind": "mistudio.circuit-definition", "schema_version": "1", "name": "x2",
            "saes": [{**sae, "layer": 9}], "members": _circuit_members([9])}


OLD_SAE = {"mistudio_sae_id": "sae_x2", "layer": 9, "hook_type": "residual", "n_features": 64, "d_model": 8}


@pytest.mark.parametrize("kind", sorted(SCHEMAS))
def test_the_published_schema_accepts_the_old_form_and_the_new(kind):
    schema = _published(kind)
    assert not _strict_errors(_minimal(kind, dict(OLD_SAE)), schema), "a pre-X-2 document no longer validates"
    assert not _strict_errors(_minimal(kind, {**OLD_SAE, "weights_sha256": "0123456789abcdef" * 4}), schema)
    ref = schema["$defs"]["DefinitionSAERef"]
    assert "weights_sha256" in ref["properties"] and "weights_sha256" not in ref.get("required", []), ref


@pytest.mark.parametrize("kind", sorted(SCHEMAS))
@pytest.mark.parametrize("bad", ["ABCDEF" * 10 + "ABCD", "0" * 63, "0" * 65, "sha256:" + "0" * 64, 7])
def test_the_published_schema_refuses_a_malformed_hash(kind, bad):
    assert _strict_errors(_minimal(kind, {**OLD_SAE, "weights_sha256": bad}), _published(kind)), bad


def test_the_contract_model_reads_the_old_form_and_refuses_a_malformed_hash():
    from pydantic import ValidationError

    from src.schemas.cluster_profile import DefinitionSAERef

    assert DefinitionSAERef.model_validate(dict(OLD_SAE)).weights_sha256 is None
    with pytest.raises(ValidationError):
        DefinitionSAERef.model_validate({**OLD_SAE, "weights_sha256": "0" * 63})


# ── 4. cross-repo: what miServer runs today ──────────────────────────────────


_SCRIPT = r'''
import json, sys
pkg = sys.argv[2]
docs = json.load(open(sys.argv[1]))
cluster = __import__(pkg + ".api.schemas.cluster", fromlist=["ClusterDefinitionV1"])
circuit = __import__(pkg + ".api.schemas.circuit", fromlist=["CircuitDefinitionV1"])
models = {"cluster": cluster.ClusterDefinitionV1, "circuit": circuit.CircuitDefinitionV1}
out = {}
for name, document in docs.items():
    try:
        models[name.split(":")[0]].model_validate(document)
        out[name] = {"parsed": True}
    except Exception as exc:
        out[name] = {"parsed": False, "error": str(exc)[:400]}
print("@@IMPORT@@" + json.dumps(out))
'''


def _absent(reason: str) -> None:
    if require_cross_repo():
        pytest.fail("the cross-repo switch is set but miServer's import check cannot run: " + reason)
    pytest.skip("CROSS-REPO, SKIPPED LOUDLY: " + reason)


async def test_miserver_parses_the_new_form_and_its_vendored_schemas_accept_it(client, async_session, sae_files,
                                                                               tmp_path):
    repo = miserver_repo()
    pkg = miserver_package(repo)
    if pkg is None:
        _absent(f"no miServer checkout at {repo} (set MISERVER_REPO)")
    python = Path(os.environ.get("MISERVER_PYTHON") or repo / "venv" / "bin" / "python")
    if not python.is_file():
        _absent(f"no miServer interpreter at {python} (set MISERVER_PYTHON)")
    await _seed_saes(async_session)
    profiles = await _seed_profiles(async_session)
    await _seed_circuit(async_session, "crc_x2_xrepo", [{"layer": 9, "sae_id": "sae_x2_l9"},
                                                        {"layer": 10, "sae_id": "sae_x2_l10"}])
    cluster = (await client.get(f"/api/v1/cluster-profiles/{profiles[9]}/export")).json()
    circuit = (await client.get("/api/v1/circuits/crc_x2_xrepo/export")).json()
    sliced = (await client.post("/api/v1/circuits/crc_x2_xrepo/export-slices")).json()["slices"][0]
    assert cluster["sae"]["weights_sha256"] == sae_files[9]  # the new form, or this proves nothing about it
    documents = {"cluster:new": cluster, "circuit:new": circuit, "cluster:slice": sliced}

    # The schema files miServer vendors today (pre-X-2 until its half re-vendors; the new bytes after).
    for name, document in documents.items():
        theirs = json.loads((repo / "docs" / "schemas" / SCHEMAS[name.split(":")[0]]).read_text(encoding="utf-8"))
        assert not _strict_errors(document, theirs), (name, _strict_errors(document, theirs))

    path = tmp_path / "definitions.json"
    path.write_text(json.dumps(documents))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABASE_URL", "PYTHONPATH"))}
    proc = subprocess.run([str(python), "-B", "-c", _SCRIPT, str(path), pkg], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=240)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@IMPORT@@")]
    if proc.returncode != 0 or not lines:
        pytest.fail(f"miServer's import check did not run (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    out = json.loads(lines[0][len("@@IMPORT@@"):])
    assert out == {name: {"parsed": True} for name in documents}, out
