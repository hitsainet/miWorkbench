"""ENH-011 X-1, Q1 under OQ-3 answer B: a `layer_meaning_changed` artefact is refused for serving and arming.

An SAE, probe or circuit recorded on a composite model before the fix names layer indices into the stack the old
first-match resolver chose (a vision encoder), so after the fix its numbers mean another layer. Migration
`c7d2e4f9a1b3` flags each one; every path that hands the artefact toward miServer refuses it with 409
`layer_meaning_changed`, naming the old and the new stack. Nothing is deleted.

Each refusal is asserted through the LIVE app (the route and the exception handler registered in `main.py`), with
the exact detail and with the export never reached, so removing either the check or the handler goes red.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from src.core.database import get_db
from src.main import app
from src.models.circuit import Circuit
from src.models.cluster_profile import ClusterProfile
from src.models.external_sae import ExternalSAE
from src.models.probe_monitor import ProbeMonitor
from src.services import layer_meaning
from src.services.layer_meaning import CODE, LayerMeaningChanged, layer_meaning_handler
from tests.support.test_databases import sync_test_url

FLAG = {
    "old_layers_path": "model.vision_tower.encoder.layers",
    "new_layers_path": "model.language_model.layers",
    "model_type": "gemma3",
    "flagged_at": "2026-11-26T17:00:00Z",
    "flagged_by": "migration c7d2e4f9a1b3",
    "reason": "recorded before ENH-011 X-1",
}


def _expected_detail(kind, artefact_id, via=None):
    exc = LayerMeaningChanged(kind, artefact_id, FLAG, via=via)
    return {"code": CODE, "message": str(exc), "kind": kind, "id": artefact_id,
            "old_layers_path": FLAG["old_layers_path"], "new_layers_path": FLAG["new_layers_path"]}


class TestTheRefusal:

    def test_a_flagged_row_is_refused_with_both_paths(self):
        row = SimpleNamespace(id="pm_x", layer_meaning_changed=FLAG)
        with pytest.raises(LayerMeaningChanged) as refused:
            layer_meaning.refuse_if_changed("probe", row)
        message = str(refused.value)
        assert "probe pm_x" in message
        assert "'model.vision_tower.encoder.layers'" in message
        assert "'model.language_model.layers'" in message
        assert "never deleted" in message
        assert refused.value.status == 409
        assert refused.value.detail() == _expected_detail("probe", "pm_x")

    @pytest.mark.parametrize("value", [None, {}, "yes", ["x"], True])
    def test_anything_but_a_non_empty_dict_is_not_a_flag(self, value):
        layer_meaning.refuse_if_changed("probe", SimpleNamespace(id="pm_x", layer_meaning_changed=value))
        layer_meaning.refuse_if_changed("probe", None)

    def test_an_unknown_old_path_still_refuses(self):
        flag = dict(FLAG, old_layers_path=None)
        with pytest.raises(LayerMeaningChanged, match="a stack the old resolver chose"):
            layer_meaning.refuse_if_changed("SAE", SimpleNamespace(id="sae_x", layer_meaning_changed=flag))

    def test_each_artefact_table_carries_the_column(self):
        for model in (ExternalSAE, ProbeMonitor, Circuit):
            column = model.__table__.columns["layer_meaning_changed"]
            assert column.nullable is True


class TestTheHandlerIsRegisteredOnTheLiveApp:

    def test_the_app_maps_the_refusal_to_its_handler(self):
        assert app.exception_handlers.get(LayerMeaningChanged) is layer_meaning_handler


class TestTheProbeBuilderGateRefuses:
    """The build runs as a task, so the gate in the builder refuses too, whoever dispatched it."""

    def _run(self):
        return SimpleNamespace(id="pmr_x", status="completed")

    def _probe(self, **kw):
        base = {"id": "pm_x", "variant": "dense", "weights_path": "w.pt", "rung": 2, "layer_meaning_changed": None}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_a_flagged_probe_refuses_409_before_the_rung_question(self):
        from src.services.probe_definition_builder import ProbeExportRefused, check_export_gate

        with pytest.raises(ProbeExportRefused) as refused:
            check_export_gate(self._probe(layer_meaning_changed=FLAG, rung=0), self._run(),
                              acknowledge_below_rung2={"reason": "accepted on thin evidence"})
        assert refused.value.status == 409
        assert str(refused.value) == str(LayerMeaningChanged("probe", "pm_x", FLAG))

    def test_a_probe_reading_a_flagged_sae_refuses(self):
        from src.services.probe_definition_builder import ProbeExportRefused, check_export_gate

        sae = SimpleNamespace(id="sae_x", hf_repo_id="org/sae", hf_filepath="l1", layer_meaning_changed=FLAG)
        with pytest.raises(ProbeExportRefused) as refused:
            check_export_gate(self._probe(variant="sae", sae_id="sae_x"), self._run(), sae_row=sae)
        assert refused.value.status == 409
        assert str(refused.value) == str(LayerMeaningChanged("SAE", "sae_x", FLAG, via="probe pm_x"))

    def test_an_unflagged_probe_passes_the_gate(self):
        from src.services.probe_definition_builder import check_export_gate

        assert check_export_gate(self._probe(), self._run()) is None


class _Result:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _Session:
    """Answers every `execute` with one row and counts the calls."""

    def __init__(self, row):
        self.row = row
        self.calls = 0

    async def execute(self, *args, **kwargs):
        self.calls += 1
        return _Result(self.row)


@pytest.fixture
def client_with():
    sessions = []

    def make(row):
        session = _Session(row)
        sessions.append(session)

        async def _db():
            yield session

        app.dependency_overrides[get_db] = _db
        return TestClient(app, raise_server_exceptions=False), session

    try:
        yield make
    finally:
        app.dependency_overrides.pop(get_db, None)


class TestTheProbeRoutesRefuse:

    def _probe(self, flag=FLAG):
        return ProbeMonitor(id="pm_flagged", run_id="pmr_x", layer=1, rule="mean", variant="dense",
                            definition_path="probes/pm_flagged.probe.json", weights_path="w.pt",
                            layer_meaning_changed=flag)

    @pytest.mark.parametrize("method,path", [
        ("get", "/api/v1/probe-monitors/probes/pm_flagged/definition"),
        ("get", "/api/v1/probe-monitors/probes/pm_flagged/export"),
        ("post", "/api/v1/probe-monitors/probes/pm_flagged/publish"),
        ("post", "/api/v1/probe-monitors/probes/pm_flagged/definition"),
    ])
    def test_every_route_toward_miserver_refuses_a_flagged_probe(self, client_with, method, path):
        client, session = client_with(self._probe())
        body = {"repo_id": "org/probe", "token": "hf_x"} if path.endswith("publish") else {}
        with patch("src.api.v1.endpoints.probe_monitors.refuse_precision_mismatch",
                   new=AsyncMock()) as precision, \
             patch("src.workers.probe_monitor_tasks.publish_probe_definition") as publish, \
             patch("src.api.v1.endpoints.probe_monitors._huggingface_rejects", return_value=None):
            response = getattr(client, method)(path, json=body) if method == "post" else client.get(path)
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == _expected_detail("probe", "pm_flagged")
        assert session.calls == 1, "the probe is read once and nothing after the refusal"
        assert precision.await_count == 0
        assert publish.delay.call_count == 0

    def test_an_unflagged_probe_is_not_refused_by_this_check(self, client_with, tmp_path):
        probe = self._probe(flag=None)
        probe.definition_path = str(tmp_path / "absent.probe.json")
        client, _ = client_with(probe)
        response = client.get("/api/v1/probe-monitors/probes/pm_flagged/definition")
        assert response.status_code == 409
        assert response.json()["detail"] == (
            f"probe pm_flagged records a definition at {probe.definition_path} and the file is gone; build again")


def _circuit(flag):
    from datetime import datetime

    return Circuit(
        id="crc_flagged", name="Flagged", granularity="feature",
        saes=[{"mistudio_sae_id": "sae_l1", "layer": 1, "n_features": 8}],
        members=[{"layer": 1, "member_kind": "feature_ref", "feature": {"feature_idx": 1, "strength": 0.5}}],
        edges=[], budget=None, faithfulness=None, rung=2, promoted=False, model_id="m_gemma3",
        created_at=datetime(2026, 11, 26), updated_at=datetime(2026, 11, 26), layer_meaning_changed=flag,
    )


class TestTheCircuitRoutesRefuse:

    @pytest.mark.parametrize("method,path", [
        ("get", "/api/v1/circuits/crc_flagged/export"),
        ("post", "/api/v1/circuits/crc_flagged/export-slices"),
    ])
    def test_a_flagged_circuit_is_not_exported(self, method, path):
        client = TestClient(app, raise_server_exceptions=False)
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit(FLAG))), \
             patch("src.api.v1.endpoints.circuits.CircuitService.to_definition") as to_definition:
            response = getattr(client, method)(path)
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == _expected_detail("circuit", "crc_flagged")
        assert to_definition.call_count == 0, "the definition must not be built for a refused circuit"

    def test_an_unflagged_circuit_exports(self):
        client = TestClient(app, raise_server_exceptions=False)
        hashes = AsyncMock(return_value=None)  # the SAE weights hash reads the database (ENH-011 X-2); none here
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit(None))), \
             patch("src.api.v1.endpoints.circuits.CircuitService.attach_weight_hashes", new=hashes):
            response = client.get("/api/v1/circuits/crc_flagged/export")
        assert response.status_code == 200, response.text
        assert hashes.await_count == 1
        assert response.json()["name"] == "Flagged"


class TestTheClusterRoutesRefuseThroughTheirSAE:

    def _profile(self):
        return ClusterProfile(id="cp_flagged", name="Flagged", sae_id="sae_flagged", members=[])

    def _sae(self, flag):
        return ExternalSAE(id="sae_flagged", name="s", source="trained", layer=1, hook_type="residual",
                           n_features=8, d_model=32, layer_meaning_changed=flag)

    @pytest.mark.parametrize("method,path,body", [
        ("get", "/api/v1/cluster-profiles/cp_flagged/export", None),
        ("post", "/api/v1/cluster-profiles/export-bundle", {"ids": ["cp_flagged"]}),
    ])
    def test_a_profile_on_a_flagged_sae_is_not_exported(self, client_with, method, path, body):
        client, session = client_with(self._sae(FLAG))
        with patch("src.api.v1.endpoints.cluster_profiles.ClusterProfileService.get",
                   new=AsyncMock(return_value=self._profile())), \
             patch("src.services.cluster_profile_service.ClusterProfileService._enrich_members",
                   new=AsyncMock(return_value=[])) as enrich:
            response = client.post(path, json=body) if method == "post" else client.get(path)
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == _expected_detail("SAE", "sae_flagged", via="cluster profile cp_flagged")
        assert session.calls == 1, "the SAE is read once"
        assert enrich.await_count == 0, "the definition must not be assembled"


# ── the migration ────────────────────────────────────────────────────────────

BACKEND = pathlib.Path(__file__).resolve().parents[2]
REVISION_FILE = BACKEND / "alembic" / "versions" / "c7d2e4f9a1b3_layer_meaning_changed_flag.py"
STAND_INS = ("probe_monitors", "probe_monitor_runs", "circuits", "external_saes", "models")


def _revision():
    spec = importlib.util.spec_from_file_location("rev_c7d2e4f9a1b3", REVISION_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scratch_engine():
    url = sync_test_url()
    server, name = url.rsplit("/", 1)
    name = name.split("?")[0] + "_x1_layer_meaning"
    admin = create_engine(server + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
                conn.execute(text('CREATE DATABASE "' + name + '"'))
    finally:
        admin.dispose()
    return create_engine(server + "/" + name)


MODELS = {
    "m_gemma3": {"model_type": "gemma3", "text_tower": "text_config", "towers": {"text_config": {}, "vision_config": {}}},
    "m_qwen": {"model_type": "qwen2_5_vl", "text_tower": "text_config"},
    "m_unified": {"model_type": "gemma4_unified", "text_tower": "text_config"},
    "m_mystery": {"model_type": "mystery_vl", "text_tower": "text_config"},
    "m_llama": {"model_type": "llama", "num_hidden_layers": 32},
}


@pytest.fixture
def scratch():
    engine = _scratch_engine()
    with engine.begin() as c:
        for t in STAND_INS:
            c.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
        c.execute(text("CREATE TABLE models (id VARCHAR(255) PRIMARY KEY, architecture_config JSONB)"))
        c.execute(text("CREATE TABLE external_saes (id VARCHAR(255) PRIMARY KEY, model_id VARCHAR(255))"))
        c.execute(text("CREATE TABLE circuits (id VARCHAR(36) PRIMARY KEY, model_id VARCHAR(255))"))
        c.execute(text("CREATE TABLE probe_monitor_runs (id VARCHAR(36) PRIMARY KEY, model_id VARCHAR(255))"))
        c.execute(text("CREATE TABLE probe_monitors (id VARCHAR(36) PRIMARY KEY, run_id VARCHAR(36))"))
        for model_id, config in MODELS.items():
            c.execute(text("INSERT INTO models VALUES (:i, CAST(:c AS JSONB))"), {"i": model_id, "c": json.dumps(config)})
            c.execute(text("INSERT INTO external_saes VALUES (:i, :m)"), {"i": f"sae_{model_id}", "m": model_id})
            c.execute(text("INSERT INTO circuits VALUES (:i, :m)"), {"i": f"crc_{model_id}", "m": model_id})
            c.execute(text("INSERT INTO probe_monitor_runs VALUES (:i, :m)"), {"i": f"pmr_{model_id}", "m": model_id})
            c.execute(text("INSERT INTO probe_monitors VALUES (:i, :r)"), {"i": f"pm_{model_id}", "r": f"pmr_{model_id}"})
        c.execute(text("INSERT INTO models VALUES ('m_null', NULL)"))
        c.execute(text("INSERT INTO external_saes VALUES ('sae_orphan', NULL)"))
    try:
        yield engine
    finally:
        with engine.begin() as c:
            for t in STAND_INS:
                c.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
        engine.dispose()


def _apply(conn, step):
    with Operations.context(MigrationContext.configure(conn)):
        step()


def _flags(conn, table):
    rows = conn.execute(text(f"SELECT id, layer_meaning_changed FROM {table} ORDER BY id")).all()
    return {row[0]: row[1] for row in rows}


class TestTheMigration:

    def test_it_flags_exactly_the_rows_on_composite_models_whose_answer_changed(self, scratch):
        revision = _revision()
        with scratch.begin() as conn:
            _apply(conn, revision.upgrade)
            flagged = {t: {k: v for k, v in _flags(conn, t).items() if v} for t in
                       ("external_saes", "circuits", "probe_monitors")}
        expected_ids = {"m_gemma3", "m_qwen", "m_mystery"}
        assert set(flagged["external_saes"]) == {f"sae_{m}" for m in expected_ids}
        assert set(flagged["circuits"]) == {f"crc_{m}" for m in expected_ids}
        assert set(flagged["probe_monitors"]) == {f"pm_{m}" for m in expected_ids}
        gemma = flagged["probe_monitors"]["pm_m_gemma3"]
        assert (gemma["old_layers_path"], gemma["new_layers_path"], gemma["model_type"], gemma["flagged_by"]) == (
            "model.vision_tower.encoder.layers", "model.language_model.layers", "gemma3", "migration c7d2e4f9a1b3")
        assert gemma["flagged_at"].endswith("Z") and gemma["reason"] == revision.REASON
        qwen = flagged["circuits"]["crc_m_qwen"]
        assert (qwen["old_layers_path"], qwen["new_layers_path"]) == ("model.visual.blocks", "model.language_model.layers")
        mystery = flagged["external_saes"]["sae_m_mystery"]
        assert mystery["old_layers_path"] is None and mystery["new_layers_path"] is None, (
            "an unknown composite family is refused, never given an invented path")

    def test_the_frozen_paths_agree_with_the_cases_file_it_was_taken_from(self):
        from tests.support.layer_cases import load_cases

        revision = _revision()
        cases = {c["family"]: c for c in load_cases()}
        for family, (old, new) in revision.OLD_AND_NEW.items():
            assert (cases[family]["vision_layers_path"], cases[family]["text_layers_path"]) == (old, new)
        for family in revision.UNCHANGED:
            assert "vision_layers_path" not in cases[family]

    def test_it_goes_down_and_up_again(self, scratch):
        revision = _revision()
        with scratch.begin() as conn:
            _apply(conn, revision.upgrade)
        with scratch.begin() as conn:
            _apply(conn, revision.downgrade)
            for table in revision.TABLES:
                assert "layer_meaning_changed" not in {c["name"] for c in sa.inspect(conn).get_columns(table)}
            assert conn.execute(text("SELECT count(*) FROM probe_monitors")).scalar() == len(MODELS), "nothing is deleted"
        with scratch.begin() as conn:
            _apply(conn, revision.upgrade)
            assert sum(1 for v in _flags(conn, "circuits").values() if v) == 3

    def test_it_is_the_single_head_after_its_parent(self):
        revision = _revision()
        assert (revision.revision, revision.down_revision) == ("c7d2e4f9a1b3", "5e3c9a1d7b42")

    def test_it_runs_through_asyncpg_as_alembic_runs_it(self, scratch):
        """`alembic/env.py` migrates through asyncpg, which cannot type an untyped bind inside jsonb_build_object.

        The replays above use psycopg2 and passed while `alembic upgrade head` failed on a fresh database with
        IndeterminateDatatypeError (2026-11-26); the CAST of `:reason` is what this pins.
        """
        import asyncio

        from sqlalchemy.ext.asyncio import create_async_engine

        revision = _revision()
        url = scratch.url.set(drivername="postgresql+asyncpg")

        async def run():
            engine = create_async_engine(url)
            try:
                async with engine.begin() as conn:
                    await conn.run_sync(lambda sync: _apply(sync, revision.upgrade))
                async with engine.connect() as conn:
                    rows = await conn.execute(text(
                        "SELECT count(*) FROM probe_monitors WHERE layer_meaning_changed IS NOT NULL"))
                    return rows.scalar()
            finally:
                await engine.dispose()

        assert asyncio.run(run()) == 3
