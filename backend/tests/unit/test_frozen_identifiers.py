"""Frozen identifiers keep their values, proven through the code that reads or writes them (042 FR-042.7; T-05).

ENH-009 renames miStudio to miWorkbench and miLLM to miServer, but some names are FROZEN: they are stored,  (p9: keep, frozen value named by its exercise)
signed, hashed, matched by a sibling or by an agent, so renaming them would break data or callers that
already exist (`docs/legacy-identifiers.md`, the census export; register `deferred-renames.md`). This file
holds EXACTLY one exercise per entry the export names for this repository (FTID §11.1; TI-7), and
`test_frozen_identifiers_coverage.py` holds the bookkeeping: every entry's `exercise` resolves to a function
here, and every function here is named by an entry.

The rules each exercise follows:
* **Values come from the export**, never re-typed: `_entry(id)["literals"]` and `["pattern"]`. Where census
  lists no literal for an entry (DR-01, DR-02, DR-07, DR-13, …: values built at runtime), the exercise drives
  the real code, takes the value it PRODUCED, and requires the entry's `pattern` to match it.
* **Through the real reader or writer** (a route, a client request captured by `httpx.MockTransport`, a
  serializer, the migration's own seed function, the live MCP registry, the parsed manifest, the workflow's
  own shell step run by bash). A grep of source text would match a comment (the miLLM P0 review's finding).  (p9: keep, frozen value named by its exercise)
* **Mirror:** DR-19 and DR-29 are withheld from the public export; `legacy_export.require` skips them BY NAME
  there. Nothing else here skips (DR-20's cross-repo half fails under the require flag, as every cross-repo
  check does).

Per-entry production mutations (FTASKS 6.21.1 – 6.21.27) are recorded in
`0xcc/reviews/042_implementation_controls_<date>.md`.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import hmac
import json
import os
import re
import subprocess
import tempfile
import typing
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
import yaml
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tests.support import legacy_export
from tests.support.cross_repo import require_cross_repo
from tests.support.real_broker import real_broker  # noqa: F401  (fixture)
from tests.support.sibling_repos import REPO_ROOT, both_packages, miserver_package, miserver_repo
# Fixtures reused, not copied (FTDD T-05: "the existing exclusion test, imported"). Imported names are pytest
# fixtures or plain helpers; no test function is imported (that would add an unnamed test to this file).
from tests.unit.test_circuit_gpu_lock import CARDS, RTX_UUID, TI_UUID, pg  # noqa: F401
from tests.unit.test_miserver_approvals_relay import RELAYED, _MiServer, live  # noqa: F401
from tests.unit.test_probe_spot_check import _body, _fake, _setup, queued, sync_db, wire  # noqa: F401
from tests.support.miserver_fakes import as_sent

BACKEND = REPO_ROOT / "backend"
SRC = BACKEND / "src"
SCHEMAS = REPO_ROOT / "docs" / "schemas"
API = "/api/v1"


# ── export access ───────────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _export() -> legacy_export.Export:
    return legacy_export.load()


def _entry(entry_id: str) -> dict[str, Any]:
    export = _export()
    legacy_export.require(export, "entry", entry_id)
    return export.entry(entry_id)


def _pattern(entry_id: str) -> re.Pattern[str]:
    return re.compile(_entry(entry_id)["pattern"])


def _literals(entry_id: str) -> list[str]:
    return list(_entry(entry_id)["literals"])


def _quoted(value: str) -> str:
    """How a value sits in code: several patterns anchor on the surrounding quote."""
    return json.dumps(value)


# ── real-code readers shared by several exercises ───────────────────────────────────────────────────
def _src_strings(root: Path = SRC) -> list[tuple[Path, int, str]]:
    """Every string constant under `root` (`.py`), docstrings EXCLUDED, comments never seen (AST)."""
    out: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                out.append((path, node.lineno, node.value))
    assert out, f"no string constant found under {root}: the AST walk sees nothing"
    return out


@lru_cache(maxsize=1)
def _live_tools() -> dict[str, dict[str, Any]]:
    """`{name: {"description", "schema"}}` from the REAL built MCP server, every category on, every product
    URL set (the `tests/support/next_calls.py` recipe), so a tool text is read as an agent reads it."""
    from concurrent.futures import ThreadPoolExecutor

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.products import PRODUCTS
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    mp = pytest.MonkeyPatch()
    try:
        for spec in PRODUCTS:
            mp.setenv(spec.url_env, f"http://{spec.name}.test")
        settings = MCPSettings(tool_categories=",".join(sorted(all_categories())), allow_anonymous=True)
        mcp, _client = build_server(settings, stdio=True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            tools = pool.submit(asyncio.run, mcp.list_tools()).result()
    finally:
        mp.undo()
    out = {t.name: {"description": t.description or "",
                    "schema": dict(getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {})}
           for t in tools}
    assert len(out) > 100, f"the live registry lists only {len(out)} tools"
    return out


def _k8s_docs() -> list[dict[str, Any]]:
    """The objects `k8s/base/kustomization.yaml` deploys, parsed (its `resources:` list, in order)."""
    base = REPO_ROOT / "k8s" / "base"
    kustomization = yaml.safe_load((base / "kustomization.yaml").read_text())
    docs: list[dict[str, Any]] = []
    for resource in kustomization["resources"]:
        docs += [d for d in yaml.safe_load_all((base / resource).read_text()) if d]
    assert docs, "the kustomization renders nothing"
    return docs


def _workflow(name: str) -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / ".github" / "workflows" / name).read_text())


# ── DR-01 kinds ─────────────────────────────────────────────────────────────────────────────────────
#: Every `mistudio.*` kind this repository's code names, by what this repository does with it. A kind the  # p9: keep, frozen value named by its exercise
#: code names that is in neither set fails (a new kind must be placed), and every kind here must still be named.
IMPORTED_KINDS = {
    "mistudio.circuit-definition",      # POST /circuits/import (api/v1/endpoints/circuits.py)
    "mistudio.cluster-definition",      # POST /cluster-profiles/import
    "mistudio.cluster-bundle",          # POST /cluster-profiles/import
    "mistudio.judge-prompt/v1",         # POST /probe-monitors/judge-prompts/import
    "mistudio.jlens-artifact/v1",       # restore / validate_jlens_document
}
EXPORTED_KINDS = {
    "mistudio.probe-definition/v1", "mistudio.probe-manifest/v1", "mistudio.nonsense-text/v1",
    "mistudio.spot-check-disagreement/v1", "mistudio.jlens-readout/v1", "mistudio.jspace-watchlist/v1",
    "mistudio.workspace-annotation/v1",
    # Report and identity kinds this repository writes into stored documents and digests:
    "mistudio.probe-identity/1", "mistudio.detector-success-report/1", "mistudio.rules-evaluation/1",
    "mistudio.blend-report/1", "mistudio.suggested-blend/1", "mistudio.verified-dataset-version/1",
    # ENH-006 G19 (2026-11-08): the behaviour verdict definition and its shared cases, owned here, vendored by miServer.
    "mistudio.behaviour-verdict/v1", "mistudio.behaviour-verdict-cases/1",
}


@pytest.fixture
def rt_api(async_engine, monkeypatch):
    """The app through FastAPI's TestClient on the unit database (`test_circuit_contract_roundtrip` shape)."""
    import torch
    from fastapi.testclient import TestClient

    from src.core.database import get_db
    from src.main import app

    dec, enc = torch.eye(4, 8), torch.eye(8, 4)
    monkeypatch.setattr("src.services.steering_service.load_sae_weights_cpu",
                        lambda path, *, d_model=None, n_features=None, architecture=None: (dec, enc))
    maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async def _override():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db] = _override
    yield TestClient(app, raise_server_exceptions=False), maker
    app.dependency_overrides.pop(get_db, None)


def _renamed_kind(kind: str) -> str:
    return "miworkbench." + kind.split(".", 1)[1]


@pytest.mark.asyncio
async def test_dr01_document_kinds_import_and_export(rt_api, tmp_path):
    from src.services.probe_definition_publisher import merge_manifest
    from tests.unit.test_circuit_contract_roundtrip import _built_circuit_id

    pattern = _pattern("DR-01")
    named = {v for _p, _l, v in _src_strings() if v.startswith("mistudio.") and pattern.search(_quoted(v))}
    assert named, "no mistudio.* kind found in src/: the discovery sees nothing"
    unplaced = named - IMPORTED_KINDS - EXPORTED_KINDS
    assert not unplaced, f"kinds the code names that this test does not place as import or export: {unplaced}"
    assert (IMPORTED_KINDS | EXPORTED_KINDS) <= named, (IMPORTED_KINDS | EXPORTED_KINDS) - named

    client, maker = rt_api
    async with maker() as db:
        cid = await _built_circuit_id(db, client, tmp_path)
    # Export, then import the circuit through the real route; the renamed kind is refused.
    doc = client.get(f"{API}/circuits/{cid}/export").json()
    assert doc["kind"] in IMPORTED_KINDS and pattern.search(_quoted(doc["kind"])), doc["kind"]
    assert client.post(f"{API}/circuits/import", json=doc).status_code == 201
    renamed = _renamed_kind(doc["kind"])
    refused = client.post(f"{API}/circuits/import", json={**doc, "kind": renamed})
    assert refused.status_code == 422, refused.text
    # Refused by the ROUTE's own kind check (it names both kinds), not only by the model's Literal behind it.
    detail = refused.json()["detail"]
    assert isinstance(detail, str) and renamed in detail and doc["kind"] in detail, detail

    # Its slices are cluster definitions; the cluster import takes the frozen kind and refuses the renamed one.
    slices = client.post(f"{API}/circuits/{cid}/export-slices").json()["slices"]
    piece = slices[0]
    assert piece["kind"] in IMPORTED_KINDS and pattern.search(_quoted(piece["kind"]))
    ok = client.post(f"{API}/cluster-profiles/import", json={"payload": piece}).json()["results"][0]
    assert ok["status"] == "imported", ok
    bad = client.post(f"{API}/cluster-profiles/import",
                      json={"payload": {**piece, "kind": _renamed_kind(piece["kind"])}})
    assert bad.status_code != 200 or bad.json()["results"][0]["status"] != "imported", bad.text

    # A kind this repository only writes: the manifest the publisher merges.
    manifest = merge_manifest(None, {"name": "a"})
    assert manifest["kind"] in EXPORTED_KINDS and pattern.search(_quoted(manifest["kind"]))

    # The probe definition, as the builder writes it (043 acceptance 2026-11-26 F-2, control B4; FR-042.7). Its kind
    # is a CONSTANT that is only the model's DEFAULT, beside a `Literal` that names the frozen value a second time.
    # Changing `PROBE_DEFINITION_KIND` alone left the Literal (which the discovery above reads) and every route
    # green, while `probe_definition_builder.build`, which never passes `kind`, would write the renamed kind. So:
    # construct it the builder's way (no `kind`), write it through the ONE serialiser, read the kind it wrote, and
    # read the file back the way a consumer validates it.
    from src.schemas.probe_definition import ProbeDefinitionV1
    from src.services.probe_definition_builder import serialise
    from tests.unit.test_probe_definition import definition

    built = serialise(ProbeDefinitionV1(**{k: v for k, v in definition().items() if k != "kind"}))
    written = json.loads(built)["kind"]
    assert written in EXPORTED_KINDS and pattern.search(_quoted(written)), written
    assert ProbeDefinitionV1.model_validate_json(built).kind == written

    # The same hole, for every document model whose kind is a default beside a one-value Literal: the default the
    # model writes must be the value its Literal accepts, and a kind this test places.
    defaults = _literal_kind_defaults()
    assert ("ProbeDefinitionV1", "kind") in {(m, f) for m, f, _d, _a in defaults}, "the walk missed the probe model"
    documents = [d for d in defaults if any(str(v).startswith(("mistudio.", "miworkbench.")) for v in (d[2], *d[3]))]
    assert len(documents) >= 8, f"the walk found {len(documents)} document kinds: it reads almost nothing"
    for model, name, default, accepted in documents:
        assert accepted == (default,), f"{model}.{name}: writes {default!r}, accepts {accepted!r}"
        assert default in IMPORTED_KINDS | EXPORTED_KINDS and pattern.search(_quoted(default)), \
            f"{model}.{name}: writes {default!r}"


def _literal_kind_defaults() -> list[tuple[str, str, str, tuple]]:
    """`(model, field, default, Literal values)` for every pydantic model under `src.schemas` whose `kind` or `format`
    field is a `Literal` with a string default: read from the IMPORTED models, never from source text."""
    import importlib
    import pkgutil

    import src.schemas
    from pydantic import BaseModel

    out: dict[tuple[str, str], tuple[str, str, str, tuple]] = {}
    for info in pkgutil.iter_modules(src.schemas.__path__, "src.schemas."):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if not (isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == module.__name__):
                continue
            for name in ("kind", "format"):
                field = obj.model_fields.get(name)
                if field is None or typing.get_origin(field.annotation) is not typing.Literal:
                    continue
                if isinstance(field.default, str):
                    out[(obj.__name__, name)] = (obj.__name__, name, field.default, typing.get_args(field.annotation))
    return sorted(out.values())


# ── DR-02 millm.* kinds ─────────────────────────────────────────────────────────────────────────────  # p9: keep, frozen value named by its exercise
def test_dr02_millm_kind_read_through_rule_signals():  # p9: keep, frozen value named by its exercise
    from src.rule_signals import document as rule_document
    from src.rule_signals import vector_digests

    pattern = _pattern("DR-02")
    path = rule_document.REDUCTION_CASES_PATH
    raw = path.read_bytes()
    parsed = json.loads(raw)
    assert pattern.search(parsed["format"]), f"the vendored cases file's format is {parsed['format']!r}"
    # The engine's verdicts are pinned by this exact file (its digest enters the engine digest).
    assert vector_digests()["rule-text-reduction-cases.json"] == hashlib.sha256(raw).hexdigest()
    # Byte-identical to miServer's published copy (the owner's file name and kind, DR-02).
    twin = miserver_repo() / "docs" / "schemas" / path.name
    if not twin.is_file():
        message = f"CROSS-REPO CHECK SKIPPED LOUDLY: miServer's {twin} is not here"
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    assert twin.read_bytes() == raw, f"{path} differs from miServer's {twin}"


# ── DR-03 probe identity ────────────────────────────────────────────────────────────────────────────
def test_dr03_probe_identity_digests_recompute():
    from src.schemas import probe_identity

    (label,) = _literals("DR-03")
    cases = json.loads((SCHEMAS / "probe-identity-cases.json").read_text(encoding="utf-8"))
    assert cases["algorithm"] == label
    assert len(cases["cases"]) > 10
    for case in cases["cases"]:
        detector, decision = probe_identity.document_identity(case["document"])
        assert (detector, decision) == (case["detector_digest"], case["decision_digest"]), case["name"]
        framed = json.loads(probe_identity.canonical_bytes("detector", probe_identity.detector_payload(case["document"])))
        assert framed["algorithm"] == label, case["name"]


# ── DR-04 schema $ids (a name, not behaviour) ───────────────────────────────────────────────────────
def test_dr04_schema_ids_equal_register():
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    pattern = _pattern("DR-04")
    literals = set(_literals("DR-04"))
    schemas = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(SCHEMAS.glob("*.json"))}
    ids = {s["$id"]: name for name, s in schemas.items() if isinstance(s, dict) and isinstance(s.get("$id"), str)}
    frozen_ids = {i for i in ids if pattern.search(i)}
    assert frozen_ids, "no $id in docs/schemas matches DR-04"
    # Every frozen-shaped $id is a register literal, and every literal is a served $id (scheme stripped).
    assert {i.split("://", 1)[1] for i in frozen_ids} == literals
    registry = Registry().with_resources(
        [(i, Resource.from_contents(schemas[n], default_specification=DRAFT202012)) for i, n in ids.items()])
    resolver = registry.resolver()
    for literal in literals:
        uri = f"https://{literal}"
        resolved = resolver.lookup(uri).contents
        assert resolved["$id"] == uri
        assert resolved is schemas[ids[uri]]
        if "raw.githubusercontent.com" in literal:
            assert ids[uri] == literal.rsplit("/", 1)[1], "the $id must name its own file"


# ── DR-06 Hugging Face tags; DR-44 library_name ─────────────────────────────────────────────────────
def _front_matter(text: str) -> tuple[dict[str, Any], list[str]]:
    assert text.startswith("---\n"), text[:80]
    block = text.split("---\n", 2)[1]
    return yaml.safe_load(block), block.splitlines()


def _cards() -> dict[str, str]:
    from src.services.probe_definition_publisher import manifest_entry, render_readme, render_repo_readme
    from tests.unit.test_probe_definition_publisher import _definition

    definition = _definition()
    entry = manifest_entry(definition, "probe-a.probe.json")
    return {"render_readme": render_readme(definition, {"probes": [entry]}),
            "render_repo_readme": render_repo_readme({"probes": [entry]}, {"probe-a.probe.json": definition})}


def test_dr06_hub_tags_in_built_readme():
    pattern = _pattern("DR-06")
    literals = set(_literals("DR-06"))
    assert literals
    in_cards: set[str] = set()
    for writer, card in _cards().items():
        tags = set(_front_matter(card)[0]["tags"])
        frozen = {t for t in tags if pattern.search(t)}
        assert frozen and frozen <= literals, (writer, tags)
        in_cards |= frozen
    # The literals the cards do not carry are the tags miServer's hub search filters on, which this
    # repository names to agents in its live tool texts (e.g. the cluster-definition tag).
    text = "\n".join(t["description"] + json.dumps(t["schema"]) for t in _live_tools().values())
    for literal in sorted(literals - in_cards):
        assert literal in text, f"{literal} is neither in a built card nor in any live tool text"


def test_dr44_library_name_in_both_card_writers():
    (literal,) = _literals("DR-44")
    key, value = (s.strip() for s in literal.split(":", 1))
    for writer, card in _cards().items():
        front, lines = _front_matter(card)
        assert front[key] == value, (writer, front.get(key))
        assert literal in lines, writer


# ── DR-07 Neuronpedia id ────────────────────────────────────────────────────────────────────────────
def test_dr07_neuronpedia_id_from_export_builder():
    from src.models.external_sae import ExternalSAE
    from src.services.neuronpedia_export_service import NeuronpediaExportService

    sae = ExternalSAE(id="sae_x", name="x", source="local", status="ready", layer=12, n_features=16384,
                      d_model=2048, hook_type="residual")
    metadata = NeuronpediaExportService()._generate_metadata_json(sae, "gpt2")  # noqa: SLF001
    assert _pattern("DR-07").search(metadata["neuronpedia_id"]), metadata["neuronpedia_id"]
    assert metadata["neuronpedia_id"].startswith(f"{metadata['model_id']}/12-")


# ── DR-08 X-miLLM-* headers ─────────────────────────────────────────────────────────────────────────  # p9: keep, frozen value named by its exercise
def test_dr08_x_millm_agent_on_every_client_path(monkeypatch):  # p9: keep, frozen value named by its exercise
    from tests.unit.test_mcp_miserver_probe_results import CHAT, MiServerStub, _call, _server
    from tests.unit.test_miserver_client_agent_header import CALLS, _client

    literals = {h.lower(): h for h in _literals("DR-08")}
    (agent_header,) = [h for h in literals.values() if h.lower().endswith("-agent")]
    # The agent header goes out on every way a request leaves miServer's client, under the frozen name.
    assert len(CALLS) >= 10
    for name, call in sorted(CALLS.items()):
        client, transport = _client()
        asyncio.run(call(client))
        assert len(transport.seen) == 1, (name, transport.seen)
        assert transport.seen[0][2].get(agent_header.lower()), f"{name} sent no {agent_header}"
    # What a real tool call sends to miServer: every x-millm-* header on the wire is a frozen literal.  # p9: keep, frozen value named by its exercise
    stub = MiServerStub(armed=True)
    _call(_server(stub, monkeypatch), "miserver_generate", dict(CHAT, max_tokens=8))
    (sent,) = stub.requests
    on_wire = set(sent["headers"])
    assert len(on_wire) >= 3 and agent_header.lower() in on_wire, on_wire
    assert on_wire <= set(literals), f"sent headers the register does not freeze: {on_wire - set(literals)}"


# ── DR-09 X-MiStudio-Agent ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_dr09_x_mistudio_agent_header(client, monkeypatch):  # p9: keep, frozen value named by its exercise
    from src.api.v1.endpoints import mcp_approvals
    from src.mcp_server.client import MiWorkbenchClient

    (header,) = _literals("DR-09")
    monkeypatch.setattr(mcp_approvals, "emit_progress", lambda **_: None)
    # Purge is the operator's alone: the route refuses the header by this name.
    refused = await client.get(f"{API}/probe-monitors/dataset-purges", headers={header: "mcp"})
    assert refused.status_code == 403 and refused.json()["detail"]["code"] == "purge_operator_only"
    assert (await client.get(f"{API}/probe-monitors/dataset-purges")).status_code == 200
    # The approvals route records who asked from it.
    body = {"tool_name": "start_probe_training_run", "payload": {"model_id": 7, "train_dataset_id": "pmd_1"}}
    filed = (await client.post(f"{API}/mcp/approvals", json=body, headers={header: "mcp"})).json()
    assert filed["requested_by"] == "mcp", filed
    # The MCP server's backend client sends it on every request.
    from tests.support.mcp_drive import OffContractTransport

    seen: list[httpx.Request] = []
    mcp_client = MiWorkbenchClient("http://backend.test")
    mcp_client._http = httpx.AsyncClient(  # noqa: SLF001
        base_url="http://backend.test",
        headers=dict(mcp_client._http.headers),  # noqa: SLF001
        transport=OffContractTransport(lambda r: (seen.append(r), httpx.Response(200, json={}))[1],
                                       reason="only the request's header is read here; the answer is a placeholder"))
    await mcp_client.get("/api/v1/models")
    await mcp_client.post("/api/v1/x", json={})
    assert len(seen) == 2 and all(r.headers.get(header) == "mcp" for r in seen), [dict(r.headers) for r in seen]


# ── DR-11 internal API label ────────────────────────────────────────────────────────────────────────
def test_dr11_internal_api_label():
    from fastapi.testclient import TestClient

    from src.core.config import Settings, settings
    from src.main import app

    (label,) = _literals("DR-11")
    key = "a-shared-secret-key-of-at-least-32-chars"
    api_side = Settings(secret_key=key, _env_file=None)
    worker_side = Settings(secret_key=key, _env_file=None)
    expected = hmac.new(key.encode(), label.encode(), hashlib.sha256).hexdigest()
    assert api_side.internal_api_secret == worker_side.internal_api_secret == expected
    # The running app accepts the token derived under the frozen label, and refuses any other label's.
    good = hmac.new(settings.secret_key.encode(), label.encode(), hashlib.sha256).hexdigest()
    other = hmac.new(settings.secret_key.encode(), b"miworkbench-internal-api-v1", hashlib.sha256).hexdigest()
    with TestClient(app) as http:
        body = {"channel": "c", "event": "e", "data": {}}
        assert http.post("/api/internal/ws/emit", json=body, headers={"X-Internal-Token": good}).status_code == 200
        assert http.post("/api/internal/ws/emit", json=body, headers={"X-Internal-Token": other}).status_code == 403


# ── DR-12 circuit GPU lock key ──────────────────────────────────────────────────────────────────────
def test_dr12_circuit_gpu_lock_key(pg):
    from sqlalchemy import text

    from src.services import circuit_gpu_lock as L
    from src.services.circuit_capture_service import CircuitCaptureService
    from src.services.gpu_placement import normalise_uuid

    (prefix,) = _literals("DR-12")

    def frozen_key(uuid: str) -> int:  # the key a process built before the rename computes
        return int(hashlib.sha256(f"{prefix}:{normalise_uuid(uuid)}".encode()).hexdigest()[:15], 16)

    assert L.lock_key(TI_UUID) == frozen_key(TI_UUID) and L.lock_key(RTX_UUID) == frozen_key(RTX_UUID)
    holder = pg()
    CircuitCaptureService.assert_no_active_gpu_run(holder, gpu_request=TI_UUID)  # transaction left open
    other = pg()
    taken = other.execute(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": frozen_key(TI_UUID)}).scalar()
    assert taken is False, "a holder of the card's frozen key was not excluded"
    assert other.execute(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": frozen_key(RTX_UUID)}).scalar()


# ── DR-13 Redis keys ────────────────────────────────────────────────────────────────────────────────
def test_dr13_redis_keys_round_trip(real_broker):  # noqa: F811
    from src.services import gpu_queue_state, gpu_worker_queues
    from src.workers import task_completion, task_liveness

    client, _url = real_broker
    pattern = _pattern("DR-13")

    def keys() -> set[str]:
        return {k.decode() for k in client.keys("*")}

    assert task_liveness.stamp("t-1", client=client) is True
    written = keys()
    assert len(written) == 1 and all(pattern.search(_quoted(k)) for k in written), written
    # A mark another process wrote under the same key (e.g. one still running the pre-rename image) is seen.
    (key,) = written
    client.delete(key)
    client.set(key, "1", ex=60)
    assert task_liveness.has_live_stamp("t-1", client=client) is True

    task_completion.record_completed("t-2", "x.task", client=client)
    done = keys() - written
    assert len(done) == 1 and all(pattern.search(_quoted(k)) for k in done), done
    assert task_completion.completed_at("t-2", client=client)

    gpu_worker_queues.mark_waiting("t-3", client=client)
    assert gpu_worker_queues.is_waiting_for_gpu("t-3", client=client) is True
    gpu_worker_queues.park_job("x.task", [], {}, {"task_id": "t-4"}, due_at=1.0, client=client)
    queued = keys() - written - done
    assert queued and all(pattern.search(_quoted(k)) for k in queued), queued
    for constant in (gpu_queue_state.FIRST_SEEN_KEY, gpu_worker_queues.WAITING_KEY):
        assert pattern.search(_quoted(constant)), constant


# ── DR-14 columns ───────────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_dr14_frozen_columns(client, sync_db, wire, queued):  # noqa: F811
    from sqlalchemy import create_engine, inspect

    from src.models.probe_spot_check import ProbeSpotCheckItem
    from tests.support.test_databases import migrated_url

    literals = set(_literals("DR-14"))
    # The spot-check route stores and serves the probe pair under the frozen names.
    probe, prompt = await _setup(client, sync_db)
    wire(_fake(probe))
    started = await client.post(f"{API}/probe-monitors/spot-checks", json=_body(prompt))
    assert started.status_code == 202, started.text
    got = (await client.get(f"{API}/probe-monitors/spot-checks/{started.json()['id']}")).json()
    assert set(got["sample_spec"]["primary"]) == literals, got["sample_spec"]["primary"]
    for pair in got["sample_spec"]["compared"]:
        assert set(pair) == literals
    # The column under one of them exists in the ORM and in the MIGRATED schema.
    column_names = {c for c in literals if c in ProbeSpotCheckItem.__table__.c}
    assert column_names, "no frozen column on the ORM model"
    engine = create_engine(migrated_url())
    try:
        migrated = {c["name"] for c in inspect(engine).get_columns(ProbeSpotCheckItem.__tablename__)}
    finally:
        engine.dispose()
    assert column_names <= migrated


# ── DR-16 seeded labeling template ──────────────────────────────────────────────────────────────────
def test_dr16_seeded_labeling_template(monkeypatch):
    import importlib.util

    from sqlalchemy import create_engine, text

    from tests.support.test_databases import migrated_url

    (template_id,) = [x for x in _literals("DR-16") if x.startswith("lpt_")]
    (seeded_name,) = [x for x in _literals("DR-16") if not x.startswith("lpt_")]
    path = BACKEND / "alembic" / "versions" / "c458030cd014_seed_system_labeling_templates.py"
    spec = importlib.util.spec_from_file_location("seed_templates_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    inserted: list[dict[str, Any]] = []

    class _Op:
        def bulk_insert(self, _table, rows):
            inserted.extend(rows)

    monkeypatch.setattr(migration, "op", _Op())
    migration.upgrade()
    seeded = {row["id"]: row for row in inserted}
    assert template_id in seeded, sorted(seeded)
    internal = json.loads((SRC / "data" / "templates" / "mistudio_internal.json").read_text())
    assert seeded[template_id]["name"] == internal["name"] == seeded_name
    # P9 (DR-16 widened): two later revisions find the row by this name (parsed, never a text search).
    versions = BACKEND / "alembic" / "versions"
    rename = ast.parse((versions / "c25e7a41d9b3_rename_internal_labeling_template.py").read_text())
    assigned = {n.targets[0].id: n.value.value for n in rename.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
    assert assigned["SEEDED_NAME"] == seeded_name, assigned.get("SEEDED_NAME")
    improve = ast.parse((versions / "n1o2p3q4r5s6_improve_default_labeling_template.py").read_text())
    wheres = [n.value for n in ast.walk(improve) if isinstance(n, ast.Constant) and isinstance(n.value, str)
              and "WHERE name = " in n.value]
    assert wheres and all(f"WHERE name = '{seeded_name}'" in w for w in wheres), wheres
    # And on a database migrated to head, the row is there.
    engine = create_engine(migrated_url())
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT id FROM labeling_prompt_templates WHERE id = :i"),
                               {"i": template_id}).first()
    finally:
        engine.dispose()
    assert row is not None, f"{template_id} is not seeded on the migrated database"


# ── DR-17 actor ids ─────────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_dr17_both_actor_ids_are_agents(client):
    from tests.unit.test_judge_prompt_library import NEW_PROMPT

    (old_actor,) = _literals("DR-17")
    new_actor = "agent:miworkbench-mcp"
    for n, actor in enumerate((old_actor, new_actor)):
        made = await client.post(f"{API}/probe-monitors/judge-prompts",
                                 json={**NEW_PROMPT, "name": f"dr17-{n}", "actor": actor})
        assert made.status_code == 201, made.text
        assert (made.json()["source"], made.json()["created_by"]) == ("agent", actor)


# ── DR-19 database names (presence; withheld from the public export) ───────────────────────────────
def test_dr19_database_names_present():
    from tests.support import test_databases

    pattern = _pattern("DR-19")
    job = _workflow("backend-tests.yml")["jobs"]["test"]
    service_env = job["services"]["postgres"]["env"]
    for key in ("POSTGRES_DB", "POSTGRES_USER"):
        line = f"{key}: {service_env[key]}"
        assert pattern.search(line), line
    assert pattern.search(job["env"]["DATABASE_URL"]), job["env"]["DATABASE_URL"]
    assert pattern.search(f"postgresql://x@h/{test_databases.TEST_DATABASE}")
    assert pattern.search(f"postgresql://x@h/{test_databases.DEV_DATABASE}")
    # P9 widening (GAP-13): CI's schema-check database and the restore drill's scratch database.
    assert pattern.search(job["env"]["SCHEMA_CHECK_DATABASE_URL"]), job["env"]["SCHEMA_CHECK_DATABASE_URL"]
    drill = [ln.strip() for ln in (REPO_ROOT / "k8s" / "base" / "db-backup.yaml").read_text().splitlines()
             if "createdb " in ln]
    assert drill and all(pattern.search(ln) for ln in drill), drill


# ── DR-20 miServer's package (scheduled P8) ─────────────────────────────────────────────────────────
def test_dr20_miserver_package_resolves_either(tmp_path):
    pattern = _pattern("DR-20")
    trees = {}
    for name, pkgs in {"old": ["millm"], "new": ["miserver"], "both": ["millm", "miserver"], "none": []}.items():
        root = tmp_path / name
        for pkg in pkgs:
            (root / pkg).mkdir(parents=True)
            (root / pkg / "main.py").write_text("")
        root.mkdir(exist_ok=True)
        trees[name] = root
    assert miserver_package(trees["old"]) == "millm"
    assert pattern.search(_quoted(f"{miserver_package(trees['old'])}/main.py"))
    assert miserver_package(trees["new"]) == "miserver"
    assert miserver_package(trees["both"]) == "miserver" and both_packages(trees["both"])
    assert miserver_package(trees["none"]) is None
    repo = miserver_repo()
    if not repo.is_dir():
        message = f"CROSS-REPO CHECK SKIPPED LOUDLY: miServer's checkout {repo} is not here"
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    assert miserver_package(repo) in ("millm", "miserver"), repo


# ── DR-25 is the Vitest exercise (frontend/src/test/frozenStorageKeys.test.ts) ─────────────────────


# ── DR-27 Docker Hub repositories ───────────────────────────────────────────────────────────────────
def test_dr27_old_image_repos_still_pushed(tmp_path):
    """DR-27 (frozen: Docker Hub cannot rename a repository; the old repositories keep their images and are never
    deleted). The NAME is the register's exercise id, which predates P7: from ENH-009 P1 to P7 each image was also
    pushed to its old repository; P7 stopped that push (FTASKS 18.10, `test_legacy_retired.py::test_single_image_push`).
    What this pins now: no repository the build pushes, or the change detector reads, matches the DR-27 pattern
    (both read `.github/scripts/image_repos.sh`); each image goes to its `hitsai/miworkbench-*` twin. (The export's
    `literals` for DR-27 are the spellings still present in this tree, so they shrink as old comments go; the
    pattern is what this checks.) Flagged to S001: rename the exercise in the register."""
    pattern = _pattern("DR-27")
    wf = _workflow("docker-images.yml")
    images = wf["jobs"]["build"]["strategy"]["matrix"]
    names = ["backend", "frontend", "rule-runner"]
    pushed = set()
    step = next(s for s in wf["jobs"]["build"]["steps"] if s.get("name") == "Resolve image build parameters")
    assert step.get("env", {}).get("IMAGE") == "${{ matrix.name }}", step.get("env")
    for name in names:
        out = tmp_path / f"out-{name}"
        subprocess.run(["bash", "-euc", step["run"]], cwd=REPO_ROOT, check=True,
                       env={**os.environ, "IMAGE": name, "GITHUB_OUTPUT": str(out)})
        values = dict(line.split("=", 1) for line in out.read_text().splitlines())
        pushed.update(values["repos"].split())
    assert all(pattern.search(f"hitsai/mistudio-{n}") for n in names), "the DR-27 pattern no longer names the old repos"
    assert {p for p in pushed if pattern.search(p)} == set(), f"an old repository is still pushed: {pushed}"
    assert pushed == {f"hitsai/miworkbench-{n}" for n in names}, pushed
    # The change detector names the same repositories (its `repo_for`, one line over the shared function).
    detect = next(s["run"] for s in wf["jobs"]["detect"]["steps"] if "repo_for()" in (s.get("run") or ""))
    prelude = [ln.strip() for ln in detect.splitlines()
               if ln.strip().startswith(("source .github/scripts/image_repos.sh", "repo_for() {"))]
    assert len(prelude) == 2, prelude
    detected = set()
    for n in names:
        detected.update(subprocess.run(["bash", "-c", "\n".join(prelude) + f'\nrepo_for "{n}"'], cwd=REPO_ROOT,
                                       capture_output=True, text=True, check=True).stdout.split())
    assert detected == pushed, detected
    assert images, "the build matrix is empty"


# ── DR-28 host paths (scheduled P2/P3) ──────────────────────────────────────────────────────────────
#: P0 deployed the register's paths as `DirectoryOrCreate`. FTASKS 13.5 (P3, FTASKS 6.8 note) moves them to
#: `/data/miworkbench/…` (decision MOVE, a same-device rename) and types every one `type: Directory`, so a skipped
#: move refuses to mount instead of starting an empty database. If S001's P3 rehearsal (runbook 3.05) keeps the old
#: paths instead, `MOVED_FROM`/`MOVED_TO` become identical and the old paths are asserted here.
EXPECTED_HOST_PATH_TYPE = "Directory"
MOVED_FROM, MOVED_TO = "/data/mistudio/", "/data/miworkbench/"


def test_dr28_host_paths_are_directory_typed():
    pattern = _pattern("DR-28")
    found = {}
    for doc in _k8s_docs():
        spec = (doc.get("spec") or {}).get("template", {}).get("spec") or doc.get("spec") or {}
        for vol in spec.get("volumes") or []:
            host = vol.get("hostPath")
            if host:
                found[host["path"]] = host.get("type")
    # Each register path, read at its P3 home: the register keeps the old spelling (it names what was deployed).
    data = {p: t for p, t in found.items() if pattern.search(p.replace(MOVED_TO, MOVED_FROM))}
    assert data, f"no hostPath under a DR-28 path: {found}"
    assert data == {p: EXPECTED_HOST_PATH_TYPE for p in data}, data
    assert {p.rsplit("/", 1)[1] for p in data} >= {"data", "postgres", "redis"}, data
    assert all(p.startswith(MOVED_TO) for p in data), f"a data path was not moved with the others: {data}"


# ── DR-29 old hosts (scheduled P7; withheld from the public export) ────────────────────────────────
def test_dr29_old_hosts_served_until_retired():
    pattern = _pattern("DR-29")
    export = _export()
    served: set[str] = set()
    services: set[tuple[str, str]] = set()
    for doc in _k8s_docs():
        if doc.get("kind") == "Ingress":
            served |= {r["host"] for r in doc["spec"].get("rules", []) if r.get("host")}
        if doc.get("kind") == "Service":
            services.add((doc["metadata"]["name"], doc["metadata"].get("namespace", "")))
    assert export.hosts(), "the export lists no host record"
    for h in export.hosts():  # every old public-style host is a DR-29 name, retired or not
        if h["old"].endswith((".hitsai.local", ".hitsai.net")):
            assert pattern.search(h["old"]), h["old"]
    # Served while `transition`. Since ENH-009 P7 every host record is `retired` and the Ingress drops them with
    # group G5; `test_legacy_retired.py::test_old_hosts_not_served_and_origins_gone` holds that inverse.
    hosts = [h for h in export.hosts() if h["status"] == "transition"]
    for h in hosts:
        old = h["old"]
        if old.endswith((".hitsai.local", ".hitsai.net")):
            assert old in served, f"old host {old} is no longer on any Ingress (DR-29 retires at P7)"
        else:  # in-cluster DNS forms of a Service
            name, namespace = old.split(".")[:2]
            assert (name, namespace) in services, f"{old}: no Service {name} in {namespace}"
    if hosts:
        assert {lit for lit in _literals("DR-29") if lit in served}, "no DR-29 literal is served"


# ── DR-30 Hugging Face account (a name) ─────────────────────────────────────────────────────────────
def test_dr30_hf_account_named():
    pattern = _pattern("DR-30")
    named = {name for name, t in _live_tools().items()
             if pattern.search(t["description"]) or pattern.search(json.dumps(t["schema"]))
             or pattern.search(json.dumps(t["schema"]).replace('\\"', '"'))}
    assert named, "no live MCP tool names the Hugging Face account (DR-30)"


# ── DR-31 document ids in front-door files (names) ──────────────────────────────────────────────────
def test_dr31_document_ids_in_front_door_files():
    pattern = _pattern("DR-31")

    def documented(path: str) -> bool:
        out = subprocess.run(["git", "check-attr", "linguist-documentation", "--", path], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout
        return out.strip().endswith(": set")

    attrs = REPO_ROOT / ".gitattributes"
    named = [line.split()[0] for line in attrs.read_text().splitlines()
             if line.strip() and not line.startswith("#") and pattern.search(line.split()[0])]
    assert named, ".gitattributes names no DR-31 document"
    for path in named:
        assert documented(path), f".gitattributes names {path} but git does not read it as documentation"
    # The other direction: every tracked DR-31 front-door document (the project PRD and ADR, by their
    # frozen file names) is the one .gitattributes marks. The mirror strips 0xcc/, so there are none there.
    tracked = [p for p in subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True,
                                         check=True).stdout.decode().split("\0")
               if p and p.rsplit("/", 1)[-1].startswith("000_") and pattern.search(p.rsplit("/", 1)[-1])]
    if (REPO_ROOT / "0xcc").is_dir():
        assert len(tracked) >= 2, f"the project PRD and ADR are not tracked under their DR-31 names: {tracked}"
    for path in tracked:
        assert documented(path), f"{path}: .gitattributes no longer marks it (its DR-31 name changed?)"
        assert path in named, f"{path} is tracked but .gitattributes does not name it"


# ── DR-41 compose volume names ──────────────────────────────────────────────────────────────────────
def test_dr41_compose_volume_names():
    pattern = re.compile(_entry("DR-41")["pattern"], re.MULTILINE)
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    volumes = compose["volumes"]
    named = {key: v["name"] for key, v in volumes.items() if isinstance(v, dict) and v.get("name")}
    assert len(named) >= 4, named
    for key, value in named.items():
        assert pattern.search(f"    name: {value}\n"), (key, value)
    used = {m.split(":", 1)[0] for svc in compose["services"].values() for m in svc.get("volumes", [])
            if isinstance(m, str) and not m.startswith((".", "/", "~"))}
    assert used and used <= set(volumes), used - set(volumes)


# ── DR-42 dump prefix ───────────────────────────────────────────────────────────────────────────────
def test_dr42_prune_glob_matches_dump_name(tmp_path):
    pattern = _pattern("DR-42")
    (cron,) = [d for d in _k8s_docs() if d.get("kind") == "CronJob"]
    container = cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]
    script = "\n".join(container.get("command", []) + container.get("args", []))
    code = [ln.strip() for ln in script.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    (target,) = [ln for ln in code if ln.startswith("target=")]
    globs = sorted({m for ln in code for m in re.findall(r'"\$BACKUP_DIR"/([^\s|)]+\.dump)', ln)})
    assert globs, "no dump glob found in the backup script"
    assert pattern.search(target) and all(pattern.search(g) for g in globs), (target, globs)
    backup = tmp_path / "b"
    backup.mkdir()
    made = subprocess.run(["bash", "-c", f'BACKUP_DIR="{backup}"; stamp=20261108-033000; {target}; '
                                         'touch "$target"; basename "$target"'],
                          capture_output=True, text=True, check=True).stdout.strip()
    for glob in globs:
        listed = subprocess.run(["bash", "-c", f'BACKUP_DIR="{backup}"; ls -1 "$BACKUP_DIR"/{glob}'],
                                capture_output=True, text=True).stdout.split()
        assert [Path(p).name for p in listed] == [made], (glob, listed, made)


# ── DR-45 agent-facing codes ────────────────────────────────────────────────────────────────────────
def test_dr45_agent_facing_codes(live):  # noqa: F811
    from src.mcp_server.client import BackendError
    from src.mcp_server.miserver_client import MiServerClient

    pattern = _pattern("DR-45")
    # Three literals since the register's P4 rows: the moved-approval code and the relay's two refusal
    # codes. Selected by role, never re-typed.
    literals = _literals("DR-45")
    (moved,) = [x for x in literals if x.startswith("approval_moved")]
    (no_queue,) = [x for x in literals if x.endswith("_has_no_approval_queue")]
    (refused_code,) = [x for x in literals if x.endswith("_refused")]
    assert sorted(literals) == sorted([moved, no_queue, refused_code]) and all(pattern.search(x) for x in literals)

    # A miServer error with no `code` surfaces the frozen fallback code through BOTH client paths
    # (before `live` patches httpx's client class for the tool calls below).
    def envelope(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"success": False, "error": {"message": "boom"}})

    codes = []
    client = MiServerClient("http://millm.test")
    client._http = httpx.AsyncClient(base_url="http://millm.test", transport=httpx.MockTransport(as_sent(envelope)))  # noqa: SLF001
    for path_call in (lambda: client.get("/api/models"), lambda: client.raw_get("/api/clusters/c/export")):
        with pytest.raises(BackendError) as info:
            asyncio.run(path_call())
        codes.append(info.value.detail["code"])
    assert len(codes) == 2 and len(set(codes)) == 1 and pattern.search(_quoted(codes[0])), codes

    # One built server; this miServer's approval queue gates telemetry writes and NOT model loads, and it
    # refuses the telemetry change it is sent.
    class Refusing(_MiServer):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if (request.method, request.url.path) == ("PUT", "/api/environment"):
                self.seen.append({"method": "PUT", "path": "/api/environment"})
                return httpx.Response(422, json={"detail": "TELEMETRY_ENDPOINT carries a credential"})
            return super().__call__(request)

    miserver = Refusing(actions=["telemetry_write"])
    call, mistudio_seen = live(miserver)  # p9: keep, frozen value named by its exercise

    # The deprecated approval id is refused with the frozen code through the real tool; nothing is sent.
    args = dict(RELAYED["miserver_load_model"][0], approval_request_id="ap_old")
    result = call("miserver_load_model", args)
    assert result["refused"] == moved and miserver.seen == [] and mistudio_seen == []  # p9: keep, frozen value named by its exercise

    # An action miServer's queue does not gate: refused with the frozen code after ONE read of the action
    # list, and the load itself is never sent.
    result = call("miserver_load_model", dict(RELAYED["miserver_load_model"][0]))
    assert result["refused"] == no_queue, result
    assert [(r["method"], r["path"]) for r in miserver.seen] == [("GET", "/api/approvals/actions")], miserver.seen

    # miServer's own refusal of a relayed change comes back under the frozen code, with its status.
    result = call("miserver_set_telemetry", {"values": {"TELEMETRY_TRACES_ENABLED": True}})
    assert result["refused"] == refused_code and result["status_code"] == 422, result
    assert (miserver.seen[-1]["method"], miserver.seen[-1]["path"]) == ("PUT", "/api/environment")
    assert mistudio_seen == []  # p9: keep, frozen value named by its exercise


# ── DR-46 probe verdicts key ────────────────────────────────────────────────────────────────────────
def test_dr46_probe_verdicts_reach_tool_result(monkeypatch):
    from tests.unit.test_mcp_miserver_probe_results import CHAT, FIXTURE, MiServerStub, _call, _server

    (key,) = _literals("DR-46")
    verdicts = FIXTURE["millm_probe_verdicts"]
    assert verdicts, "the fixture has no verdicts"

    class Stub(MiServerStub):
        def chat_body(self):
            body = super().chat_body()
            body.pop(key, None)
            body[key] = verdicts  # the body field under the FROZEN name, as miServer sends it
            return body

    result = _call(_server(Stub(armed=True), monkeypatch), "miserver_generate", dict(CHAT, max_tokens=8))
    assert result[key] == verdicts, f"the tool result does not carry {key} at the top level"
    assert result["result"][key] == verdicts


# ── 2026-11-11 (P4 register work): entries the P4 enforce dry run found with no record ───────────────
class _Sibling:
    """miServer at the HTTP layer for the `live` fixture: `routes` maps `(method, path)` to a response;
    the gate's health probe always answers. Every other request is recorded and answered 404."""

    def __init__(self, routes: dict[tuple[str, str], httpx.Response]):
        self.routes = routes
        self.seen: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key == ("GET", "/api/health"):
            return httpx.Response(200, json={"status": "ok"})
        self.seen.append(key)
        return self.routes.get(key, httpx.Response(404, json={"detail": "Not Found"}))


def _matches(pattern: re.Pattern[str], text: str) -> list[str]:
    return [m.group(0) for m in pattern.finditer(text)]


def _bare_names() -> tuple[str, str]:
    """`(this app's, the server's)` bare old names: the two alternatives of DR-53's pattern, each of
    which the pattern must match as a whole quoted value."""
    pattern = _pattern("DR-53")
    bare = sorted({m for alt in re.findall(r"\(([^()]+)\)", pattern.pattern) for m in alt.split("|")})
    assert len(bare) == 2 and all(pattern.fullmatch(_quoted(v)) for v in bare), bare
    own = next(v for v in bare if "studio" in v)
    return own, next(v for v in bare if v != own)


# ── DR-34 the `millm` block of miServer's batch answers ─────────────────────────────────────────────  # p9: keep, frozen value named by its exercise
def test_dr34_batch_tools_name_and_carry_the_millm_block(live):  # noqa: F811  # p9: keep, frozen value named by its exercise
    """The three batch tools tell an agent where miServer's own block is (`millm.waiting_reason` on a
    batch, `millm: {headers}` on a result row), and the real tools hand that block through untouched."""
    pattern = re.compile(_entry("DR-34")["pattern"], re.MULTILINE)
    tools = _live_tools()
    named = {name: _matches(pattern, tools[name]["description"])
             for name in ("miserver_submit_batch", "miserver_batch_status", "miserver_batch_results")}
    assert all(named.values()), f"a batch tool no longer names the frozen block: {named}"
    block = {m.split(".")[0].split(":")[0] for hits in named.values() for m in hits}
    assert len(block) == 1, block
    (key,) = block
    fields = {m.split(".", 1)[1] for hits in named.values() for m in hits if "." in m}
    assert fields == {"waiting_reason"}, fields

    batch = {"id": "batch_1", "object": "batch", "status": "validating", key: {"waiting_reason": "queued"}}
    row = {"id": "r1", "custom_id": "c1", "error": None,
           "response": {"status_code": 200, "body": {"ok": True}, key: {"headers": {"x-millm-seed": "7"}}}}
    sibling = _Sibling({
        ("GET", "/v1/batches/batch_1"): httpx.Response(200, json=batch),
        ("GET", "/v1/files/file_1/content"): httpx.Response(
            200, content=(json.dumps(row) + "\n").encode(), headers={"content-type": "application/jsonl"}),
    })
    call, _mistudio_seen = live(sibling)  # p9: keep, frozen value named by its exercise
    status = call("miserver_batch_status", {"batch_id": "batch_1"})
    # 046: miServer's block arrives under its frozen key (the fake is completed with the block's other always-sent
    # fields, as miServer's contract declares them).
    assert status["result"][key]["waiting_reason"] == "queued", status
    page = call("miserver_batch_results", {"file_id": "file_1"})
    assert page["total"] == 1 and page["rows"][0]["response"][key] == row["response"][key], page
    assert sibling.seen == [("GET", "/v1/batches/batch_1"), ("GET", "/v1/files/file_1/content")]


# ── DR-49 miServer's per-card memory field ──────────────────────────────────────────────────────────
PREFLIGHT_CARD = "GPU-00000000-0000-0000-0000-0000000000aa"


def _preflight(live, off_contract: Optional[str] = None) -> Any:  # noqa: F811
    """`ask(card)`: the real `gpu_preflight` on the built server, with miServer answering that one card.
    `live` is built ONCE per test (a second build would wrap the first one's transport)."""
    sibling = _Sibling({("GET", "/api/health/detailed"): httpx.Response(200, json={
        "model_loaded": True, "model_name": "m", "lease": None, "model_placement": None})})
    call, _mistudio_seen = live(sibling, off_contract)  # p9: keep, frozen value named by its exercise

    def ask(card: dict[str, Any]) -> dict[str, Any]:
        sibling.seen.clear()
        sibling.routes[("GET", "/api/health/gpus")] = httpx.Response(200, json={"read_at": "x", "cards": [card]})
        out = call("gpu_preflight", {"gpu": PREFLIGHT_CARD})
        assert sibling.seen == [("GET", "/api/health/detailed"), ("GET", "/api/health/gpus")], sibling.seen
        return out

    return ask


def test_dr49_gpu_preflight_reads_millm_allocated_mb(live):  # noqa: F811
    """`gpu_preflight` decides "miServer's model is on this card" from the field miServer serves under its
    frozen name. The same card with the number under a renamed field reads as NOT resident."""
    pattern = _pattern("DR-49")
    # The record lists every frozen field of the pattern found in the tree; this test is about the one the GPU
    # preflight reads. It unpacked exactly one until 046's vendored miServer contract added `millm_reserved_mb`
    # (train W2b, 2026-11-26).
    field = "millm_allocated_mb"
    assert field in _literals("DR-49"), _literals("DR-49")
    assert pattern.fullmatch(field), field
    _own, server = _bare_names()  # the answer's key for miServer's block (DR-53)
    card = {"smi_index": 0, "uuid": PREFLIGHT_CARD.lower(), "torch_measured": True, "engine_memory": None}
    ask = _preflight(live, off_contract="the same card under a RENAMED field, which miServer's contract does not "
                                        "declare: the point is that it is not read as the frozen one")
    out = ask({**card, field: 15000})
    assert out[server]["card"][field] == 15000
    assert out[server]["resident_on_this_card"] is True, out[server]
    assert out["verdict"] == "busy" and any("miServer" in r for r in out["busy_reasons"]), out
    renamed = field.replace(server, "miserver")
    assert renamed != field and not pattern.search(renamed)
    moved = ask({**card, renamed: 15000})
    assert moved[server]["resident_on_this_card"] is False, "a renamed field was read as the frozen one"


# ── DR-51 `mistudio_sae_id`, `mistudio_model_id` in exchanged documents ──────────────────────────────
def test_dr51_sae_and_model_id_fields_in_definitions():
    """A circuit definition is WRITTEN with both frozen field names (the service's own validation path,
    then the contract model's serializer), a document that carries them is READ back with its values, and
    the published schema names them. Under a renamed key the value is not read."""
    from src.schemas.circuit_definition import CircuitDefinitionV1
    from src.services.circuit_service import CircuitService

    pattern = _pattern("DR-51")
    literals = _literals("DR-51")
    (sae_key,) = [x for x in literals if "_sae_" in x]
    (model_key,) = [x for x in literals if "_model_" in x]
    assert all(pattern.fullmatch(x) for x in (sae_key, model_key))
    member = {"layer": 12, "member_kind": "feature_ref", "feature": {"feature_idx": 1, "strength": 1.0}}

    built = CircuitService._validate(  # noqa: SLF001
        name="t", narrative=None, saes=[{"layer": 12, "sae_id": "sae_abc"}], members=[member], edges=[], budget=None)
    written = built.model_dump(mode="json")
    assert written["saes"][0][sae_key] == "sae_abc", written["saes"]
    assert model_key in written["model"], written["model"]

    stored = {"kind": written["kind"], "schema_version": written["schema_version"], "name": "t",
              "model": {"hf_id": "org/m", model_key: "m_1"}, "saes": [{"layer": 12, sae_key: "sae_abc"}],
              "members": [member]}
    read = CircuitDefinitionV1.model_validate(stored).model_dump(mode="json")
    assert read["saes"][0][sae_key] == "sae_abc" and read["model"][model_key] == "m_1", read

    def renamed(key: str) -> str:
        return key.replace(_bare_names()[0], "miworkbench")

    moved = dict(stored, model={"hf_id": "org/m", renamed(model_key): "m_1"},
                 saes=[{"layer": 12, renamed(sae_key): "sae_abc"}])
    lost = CircuitDefinitionV1.model_validate(moved).model_dump(mode="json")
    assert lost["saes"][0][sae_key] is None and lost["model"][model_key] is None, lost

    schema = json.loads((SCHEMAS / "circuit-definition-v1.json").read_text())
    defs = schema.get("$defs") or schema.get("definitions") or {}
    assert sae_key in defs["DefinitionSAERef"]["properties"], sorted(defs["DefinitionSAERef"]["properties"])
    assert model_key in defs["DefinitionModelRef"]["properties"]


# ── DR-52 this server's own answer fields `millm_headers`, `millm_copies` ───────────────────────────
def test_dr52_tool_answers_carry_millm_headers_and_copies(monkeypatch, sync_db, tmp_path):  # noqa: F811
    """`millm_headers`: what a miServer inference tool adds beside miServer's body, through the real built
    server. `millm_copies`: what a committed re-cut reports about miServer's imported copies."""
    from src.services import miserver_read
    from tests.unit.test_mcp_miserver_probe_results import CHAT, MiServerStub, _call, _server
    from tests.unit.test_probe_decision import make_rows
    from tests.unit.test_probe_recalibration import _recut

    pattern = _pattern("DR-52")
    literals = _literals("DR-52")
    (headers_key,) = [x for x in literals if x.endswith("_headers")]
    (copies_key,) = [x for x in literals if x.endswith("_copies")]
    assert all(pattern.fullmatch(x) for x in (headers_key, copies_key))

    answer = _call(_server(MiServerStub(armed=False), monkeypatch), "miserver_generate", dict(CHAT, max_tokens=8))
    assert answer[headers_key] == {"steering": "none"}, answer.get(headers_key)
    assert answer["result"]["choices"][0]["message"]["content"] == "hello"
    assert not [k for k in answer if k != headers_key and pattern.search(k)], sorted(answer)

    rows = make_rows(sync_db, tmp_path)
    monkeypatch.setattr(miserver_read, "imported_copies", lambda pid: ["pr_a", "pr_b"])
    out = _recut(sync_db, rows.probe.id, 0.02)
    assert out[copies_key] == ["pr_a", "pr_b"], sorted(out)
    assert [c["arguments"]["probe_id"] for c in out["next"] if c["tool"] == "miserver_recalibrate_probe"] \
        == ["pr_a", "pr_b"]


# ── DR-53 bare `mistudio` / `millm` values and keys ─────────────────────────────────────────────────  # p9: keep, frozen value named by its exercise
def test_dr53_bare_values_and_answer_keys(live, tmp_path):  # noqa: F811
    """The whole quoted value, never display text: the stored SAE format (detected from a real checkpoint,
    stored through the enum), the default corpus segment of a published lens path, the Celery app and
    Sentinel master names, the sibling stems, and the two product keys of `gpu_preflight`'s answer."""
    import torch
    from safetensors.torch import save_file

    from src.core import legacy_names
    from src.core.celery_app import celery_app
    from src.models.external_sae import SAEFormat
    from src.services import jlens_acquire_service
    from src.services.sae_converter import SAEConverterService

    pattern = _pattern("DR-53")
    own, sibling = _bare_names()
    bare = [own, sibling]

    def frozen(value: Any) -> bool:
        return isinstance(value, str) and bool(pattern.fullmatch(_quoted(value)))

    save_file({"model.encoder.weight": torch.zeros(4, 2), "model.encoder.bias": torch.zeros(4),
               "model.decoder.weight": torch.zeros(2, 4), "model.decoder_bias": torch.zeros(2)},
              str(tmp_path / "checkpoint.safetensors"))
    detected = SAEConverterService.detect_format(str(tmp_path))
    assert detected == own and frozen(detected), detected
    assert SAEFormat(detected) is SAEFormat.MISTUDIO, "the detected format is not a stored format value"  # p9: keep, frozen value named by its exercise

    path = jlens_acquire_service.published_path("org/some-model")
    assert path.rstrip("/").split("/")[-1] == own and "/jlens/" in path, path

    assert celery_app.main == own
    assert celery_app.conf.result_backend_transport_options["master_name"] == own
    assert {stems[-1] for stems in legacy_names.SIBLING_STEMS.values()} == set(bare)

    answer = _preflight(live)({"smi_index": 0, "uuid": PREFLIGHT_CARD.lower(), "engine_memory": None})
    products = {k for k in answer if isinstance(answer[k], dict)}
    assert {k for k in products if frozen(k)} == {own, sibling}, sorted(answer)
    assert answer[sibling]["configured"] is True and "active_jobs" in answer[own]


# ── DR-54 the transition Service in the old namespace ───────────────────────────────────────────────
def test_dr54_transition_service_keeps_the_old_namespace():
    """Until P7 the old in-cluster address answered through an ExternalName Service keeping the old name in the
    old namespace, which the AppProject admitted. P7 removed both (FTASKS 18.9; the inverse is
    `test_legacy_retired.py::test_externalname_and_old_destination_gone`). What DR-54 still holds: the sync
    workflow leaves the old application's image pin file alone. Read from the rendered base, the parsed
    AppProject and the workflow's own step."""
    from tests.support import k8s_render

    pattern = re.compile(_entry("DR-54")["pattern"], re.MULTILINE)
    aliases = [d for d in k8s_render.render()
               if d.get("kind") == "Service" and d["spec"].get("type") == "ExternalName"]
    assert aliases == [], [k8s_render.key(a) for a in aliases]
    namespace = "mistudio"
    assert pattern.search(f"  namespace: {namespace}\n"), "the DR-54 pattern no longer names the old namespace"

    projects = [d for d in yaml.safe_load_all((REPO_ROOT / "k8s" / "argocd" / "miworkbench-app.yaml").read_text())
                if d and d.get("kind") == "AppProject"]
    assert len(projects) == 1
    admitted = [d["namespace"] for d in projects[0]["spec"]["destinations"]]
    assert admitted == ["miworkbench"], admitted

    steps = [s.get("run", "") for job in _workflow("sync-main-to-gitops-pilot.yml")["jobs"].values()
             for s in job["steps"]]
    excluded = sorted({m for run in steps for m in re.findall(r"':\(exclude\)([^']+)'", run)})
    pins = [p for p in excluded if pattern.search(p)]
    assert pins == [f"k8s/base/.argocd-source-{namespace}.yaml"], excluded


# ── ENH-009 P9 wave 1 (043 FTASKS 2.3 - 2.6; suite exercises-2026-11-14.md §2): the registered gaps ──────
# ── DR-32 the producer name `millm` in the vendored probe-result schema ───────────────────────────────  # p9: keep, frozen value named by its exercise
def test_dr32_vendored_probe_result_producer_name():
    """The vendored `probe-result-v1.json` (miServer owns it) names its own producer: `producer.name` has a
    `const` branch and a `default`, both the frozen value, and a result naming it validates. (The schema also
    admits other producers by pattern, so a renamed value would still validate: the pin is the const/default,
    which a re-vendor of a renamed producer changes.)"""
    import jsonschema

    pattern = _pattern("DR-32")
    schema = json.loads((SCHEMAS / "probe-result-v1.json").read_text())
    name = schema["$defs"]["Producer"]["properties"]["name"]
    consts = [b["const"] for b in name.get("anyOf", [name]) if "const" in b]
    assert len(consts) == 1 and pattern.search(f'"const": "{consts[0]}"'), name
    assert pattern.search(f'"default": "{name["default"]}"') and name["default"] == consts[0], name
    validator = jsonschema.Draft202012Validator({"$ref": "#/$defs/Producer", "$defs": schema["$defs"]})
    assert not list(validator.iter_errors({"name": consts[0], "version": "1.0.0"}))


# ── DR-56 the settings HKDF salt ───────────────────────────────────────────────────────────────────────
#: Made ONCE under the literal salt with SETTINGS_ENCRYPTION_KEY below (043, 2026-11-14). Never regenerate it:
#: it is the stored secret a renamed salt can no longer read.
_DR56_KEY = "p9-dr56-fixed-test-key"
_DR56_CIPHERTEXT = "VKFh1s1lAvFTrWsHkKGZWNMXCs+KK2r+HYiurxGxPw+NqK4tf2MFy2aMLNB+yw=="


def test_dr56_settings_secret_decrypts_under_frozen_salt(monkeypatch):
    from src.core import encryption

    (salt,) = _literals("DR-56")
    assert _pattern("DR-56").fullmatch(salt)
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", _DR56_KEY)
    monkeypatch.setattr(encryption, "_derived_key", None)
    assert encryption.decrypt_value(_DR56_CIPHERTEXT) == "frozen-salt-secret"
    assert encryption.decrypt_value(encryption.encrypt_value("round trip")) == "round trip"


# ── DR-57 the built-in judge's fingerprint sentinel ───────────────────────────────────────────────────
#: The built-in judge's fingerprint, pinned 2026-11-14 (stored in every labeling row it judged).
_DR57_FINGERPRINT = "a75a0c57b26b0a567dfd776cd7eb98043817862fea9d8f0004c26715929f3597"


def test_dr57_builtin_judge_fingerprint_is_pinned():
    from src.services.labeling_fingerprint import prompt_fingerprint, prompt_fingerprint_fields

    (sentinel,) = _literals("DR-57")
    assert prompt_fingerprint_fields(None) == {"__template__": sentinel}
    assert prompt_fingerprint(None) == _DR57_FINGERPRINT


# ── DR-58 the document field `mistudio_version` ───────────────────────────────────────────────────────
def test_dr58_definitions_carry_mistudio_version():
    """The probe, cluster and circuit contract models carry the field in their provenance, the published
    probe schema lists it under `additionalProperties: false`, and the probe model refuses the field under
    a renamed key (so a document written by a renamed producer is refused, not silently emptied)."""
    from pydantic import ValidationError

    from src.schemas.circuit_definition import CircuitDefinitionV1
    from src.schemas.cluster_profile import DefinitionProvenance
    from src.schemas.probe_definition import Provenance

    (field,) = _literals("DR-58")
    pattern = _pattern("DR-58")
    assert pattern.fullmatch(field)
    assert Provenance(**{field: "1.2.3"}).model_dump()[field] == "1.2.3"
    with pytest.raises(ValidationError):
        Provenance(**{field.replace(_bare_names()[0], "miworkbench"): "1.2.3"})
    assert DefinitionProvenance(**{field: "1.2.3"}).model_dump()[field] == "1.2.3"
    circuit_provenance = CircuitDefinitionV1.model_fields["provenance"].annotation
    circuit_provenance = typing.get_args(circuit_provenance)[0] if typing.get_args(circuit_provenance) \
        else circuit_provenance
    assert field in circuit_provenance.model_fields, circuit_provenance
    schema = json.loads((SCHEMAS / "probe-definition-v1.json").read_text())
    prov = schema["$defs"]["Provenance"]
    assert field in prov["properties"] and prov.get("additionalProperties") is False
    for name in ("probe-definition-v1.json", "cluster-definition-v1.json", "circuit-definition-v1.json"):
        text = (SCHEMAS / name).read_text()
        assert f'"{field}"' in json.dumps(json.loads(text)) and pattern.search(text), name


# ── DR-59 the spot-check disagreement `source` ─────────────────────────────────────────────────────────
def test_dr59_spot_check_source_is_millm_session():  # p9: keep, frozen value named by its exercise
    from src.schemas.spot_check import DisagreementRow

    (value,) = _literals("DR-59")
    assert DisagreementRow.model_fields["source"].default == value
    schema = json.loads((SCHEMAS / "spot-check-disagreement-v1.json").read_text())
    blobs = json.dumps(schema)
    assert f'"const": "{value}"' in blobs, "the published schema no longer pins the source"


# ── DR-60 the refusal code `millm_unavailable` ─────────────────────────────────────────────────────────
def test_dr60_spot_check_refuses_with_millm_unavailable(monkeypatch):
    from src.services import miserver_read, probe_spot_check

    (code,) = _literals("DR-60")
    monkeypatch.setattr(miserver_read, "spot_check_availability",
                        lambda: {"available": False, "reason": "miServer is not connected", "url": None})
    with pytest.raises(probe_spot_check.SpotCheckRefused) as refused:
        asyncio.run(probe_spot_check.start(None, {"behaviour": "b"}, "agent:test"))
    assert (refused.value.status, refused.value.code) == (503, code)


# ── DR-61 the stored template_type `mistudio_context` ─────────────────────────────────────────────────
def test_dr61_stored_mistudio_context_template_formats():
    """A stored template whose type is the literal (and one with no type: the default) formats through the
    context formatter, differently from `anthropic_logit`; the seeded system template stores the literal."""
    from src.services.openai_labeling_service import OpenAILabelingService

    (value,) = _literals("DR-61")
    svc = OpenAILabelingService.__new__(OpenAILabelingService)
    examples = [{"prefix_tokens": ["a"], "prime_token": "b", "suffix_tokens": ["c"], "max_activation": 1.0,
                 "tokens": ["a", "b", "c"], "activations": [0.0, 1.0, 0.0]}]
    by_type = {t: svc._format_examples_block(examples, {"template_type": t} if t else {}, "f1")  # noqa: SLF001
               for t in (value, None, "anthropic_logit")}
    assert by_type[value] == by_type[None] != by_type["anthropic_logit"]
    seeded = json.loads((SRC / "data" / "templates" / "mistudio_internal.json").read_text())
    assert seeded["template_type"] == value


# ── DR-62 the SAE cfg.json keys ────────────────────────────────────────────────────────────────────────
def test_dr62_finalize_reads_mistudio_checkpoint_step(tmp_path, monkeypatch):
    from src.ml.community_format import CommunityStandardConfig
    from src.services import training_finalize_service as finalize

    literals = _literals("DR-62")
    (step_key,) = [x for x in literals if x.endswith("_step")]
    (training_key,) = [x for x in literals if x.endswith("_training_id")]
    cfg = CommunityStandardConfig.from_training_hyperparams(
        {"hidden_dim": 4, "latent_dim": 8, "total_steps": 100}, "org/m", 0, training_id="tr_1",
        checkpoint_step=100).to_dict()
    assert cfg[step_key] == 100 and cfg[training_key] == "tr_1", sorted(cfg)
    root = tmp_path / "community_format"
    (root / "layer_0_residual").mkdir(parents=True)
    (root / "layer_0_residual" / "cfg.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(finalize, "community_format_dir", lambda training_id: root)
    assert finalize.full_length_export_step("tr_1", 100) == 100
    renamed = {k.replace(_bare_names()[0], "miworkbench"): v for k, v in cfg.items()}
    (root / "layer_0_residual" / "cfg.json").write_text(json.dumps(renamed))
    assert finalize.full_length_export_step("tr_1", 100) is None


# ── DR-63 the Neuronpedia explanation method prefix ────────────────────────────────────────────────────
def test_dr63_neuronpedia_method_prefix():
    from types import SimpleNamespace

    from src.services.neuronpedia_export_service import NeuronpediaExportService

    feature = SimpleNamespace(neuron_index=7, name="a label", label_source="llm", interpretability_score=None,
                              labeled_at=None)
    exported = NeuronpediaExportService()._generate_explanations_json([feature])  # noqa: SLF001
    (method,) = [e["method"] for x in exported["explanations"] for e in x["explanations"]]
    assert _pattern("DR-63").fullmatch(method) and method.endswith("_llm"), method


# ── DR-64 the rules-evaluation agreement keys ──────────────────────────────────────────────────────────
def test_dr64_rules_evaluation_agreement_keys():
    import numpy as np

    from src.services.rules_evaluation import agreement

    literals = _literals("DR-64")
    (only,) = [x for x in literals if x.endswith("_only")]
    out = agreement(np.array([True, True, False]), np.array([True, False, False]))
    assert out[only] == 1 and set(out) == {"both", "neither", only, "producer_only"}, out
    assert [x for x in literals if x.endswith("_producer_computed")], literals


# ── DR-65 the probe-limit `set_by` ─────────────────────────────────────────────────────────────────────
def test_dr65_probe_limit_request_sends_set_by(live):  # noqa: F811
    from tests.unit.test_miserver_probe_limits_pct import _call, _limit_puts

    (value,) = _literals("DR-65")
    _, server = _call(live, {"decode_budget": {"value": 5}})
    puts = _limit_puts(server)
    assert len(puts) == 1 and puts[0]["body"]["set_by"] == value, puts


# ── DR-66 the image's OS user ──────────────────────────────────────────────────────────────────────────
def test_dr66_image_user_and_entrypoint_identity(tmp_path):
    """The parsed Dockerfile creates the group and the user and copies with their ownership; the real
    entrypoint, run in bash with `setpriv` recorded, switches to that user. No uid is pinned (`useradd -r`:
    the base image's first free system id); pinning one is DR-66's retirement step 1."""
    from tests.unit.test_gpu_workers_entrypoint import _start

    pattern = _pattern("DR-66")
    lines = [ln.strip() for ln in (BACKEND / "Dockerfile").read_text().splitlines()
             if ln.strip().startswith(("RUN", "COPY"))]
    makes = [ln for ln in lines if "useradd" in ln]
    assert len(makes) == 1 and pattern.search(makes[0]) and "--uid" not in makes[0] and " -u " not in makes[0], makes
    user = makes[0].split()[-1]  # `useradd … <name>`: the name the image creates
    assert user == _bare_names()[0], makes
    assert any(f"--chown={user}:{user}" in ln for ln in lines if ln.startswith("COPY"))
    command = _start("celery-worker", tmp_path)
    assert f"--reuid={user} --regid={user}" in command and f" USER={user} LOGNAME={user} " in command, command
    assert pattern.search(command), command


# ── DR-67 the migration-safety logger ──────────────────────────────────────────────────────────────────
def test_dr67_migration_safety_logger_name():
    from src.db import migration_safety

    (name,) = _literals("DR-67")
    assert migration_safety.logger.name == name and name.startswith("alembic.")


# ── DR-68 titles derived from frozen field names ───────────────────────────────────────────────────────
def test_dr68_derived_titles_in_published_schemas():
    from src.schemas.circuit_definition import CircuitDefinitionV1
    from src.schemas.probe_definition import ProbeDefinitionV1
    from src.schemas.spot_check import DisagreementRow

    pattern = _pattern("DR-68")
    found: dict[str, set[str]] = {}
    for name in ("circuit-definition-v1.json", "cluster-definition-v1.json", "probe-definition-v1.json",
                 "spot-check-disagreement-v1.json"):
        found[name] = {m.group(0) for m in pattern.finditer((SCHEMAS / name).read_text())}
        assert found[name], f"{name} carries no title DR-68 names"
    for model, name in ((CircuitDefinitionV1, "circuit-definition-v1.json"),
                        (ProbeDefinitionV1, "probe-definition-v1.json"),
                        (DisagreementRow, "spot-check-disagreement-v1.json")):
        generated = {m.group(0) for m in pattern.finditer(json.dumps(model.model_json_schema(), indent=2))}
        assert generated == found[name], (name, generated, found[name])


# ── DR-69 the training-state format tag ────────────────────────────────────────────────────────────────
def test_dr69_training_state_written_before_the_rename_resumes(tmp_path):
    """A state file written with the export's literal (not `TRAINING_STATE_FORMAT`, which would move with a
    rename) loads through the real reader; any other tag is refused. Writer and reader share the constant, so
    a round trip alone would stay green over a rename."""
    import torch

    from src.services import checkpoint_service as cs

    (tag,) = _literals("DR-69")
    good = tmp_path / "good"
    good.mkdir()
    torch.save({"format": tag, "version": cs.TRAINING_STATE_VERSION, "step": 7}, cs.training_state_path(good))
    assert cs.load_training_state(good)["step"] == 7
    bad = tmp_path / "bad"
    bad.mkdir()
    torch.save({"format": tag.replace(_bare_names()[0], "miworkbench"), "version": cs.TRAINING_STATE_VERSION},
               cs.training_state_path(bad))
    with pytest.raises(ValueError, match="not a miWorkbench training state"):
        cs.load_training_state(bad)


# ── DR-70 the SAE metadata key the conversion script writes ───────────────────────────────────────────
def test_dr70_sae_migration_marks_migrated_from_mistudio():
    """The script imports `core.*`/`models.*` with `src/` on `sys.path` (a second copy of the ORM), so the
    one `sae.sae_metadata[<constant>] = True` assignment in `migrate_sae` is read from its AST (parsed, so a
    comment cannot satisfy it)."""
    (key,) = _literals("DR-70")
    tree = ast.parse((BACKEND / "scripts" / "migrate_saes_to_community_format.py").read_text())
    (fn,) = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "migrate_sae"]
    keys = [n.targets[0].slice.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Subscript) and isinstance(n.targets[0].value, ast.Attribute)
            and n.targets[0].value.attr == "sae_metadata" and isinstance(n.targets[0].slice, ast.Constant)
            and isinstance(n.value, ast.Constant) and n.value.value is True]
    assert keys == [key], keys
