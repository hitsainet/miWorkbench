"""ENH-010 044 task 13.1: `<044b>`'s `wsp_experiments` table and `probe_monitors.wsp_experiment_id` (FTDD §4.2, §4.3).

Revision `408803691ec6` on `a698f0b14a0b`, ported from the unpushed `wsp/044-c` (`c0d8c8192`; suite plan B5,
2026-11-24). The migration's shape against the ORM is pinned by `test_schema_guards.py` and
`test_orm_matches_migrated_schema.py` (the schema-guard database) and `test_alembic_single_head.py`; this file pins the
BEHAVIOUR of the keys the two tables share, which form a cycle (an experiment names the probe it imported; that probe
names its experiment), the one-run-one-experiment rule, and that the revision upgrades, downgrades and upgrades again.

Controls (`0xcc/reviews/044c_port_2026-11-24.md`): `imported_probe_id` ON DELETE CASCADE instead of SET NULL →
`test_deleting_the_imported_probe_keeps_the_experiment` red; drop the unique constraint →
`test_one_miforge_run_is_one_experiment` red; drop `wsp_experiment_id`'s SET NULL →
`test_deleting_the_experiment_unlinks_the_derived_probe` red; `probe_id` or `bundle_id` back to CASCADE or NOT NULL →
the two keeps-the-experiment-unlinked tests red (operator decision 2026-11-24, B5 Q2); the migration's downgrade leaving the
column →
`test_the_migration_round_trips` red.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from src.models.probe_monitor import ProbeMonitor
from src.models.wsp import WspBundle, WspExperiment
from tests.support.wsp_world import seed_probe, sync_db  # noqa: F401 (fixture)


def _experiment(db, world, **fields) -> WspExperiment:
    bundle = WspBundle(probe_id=world.probe_id, status="ready")
    db.add(bundle)
    db.flush()
    row = WspExperiment(probe_id=world.probe_id, bundle_id=bundle.id, kind="repair", **fields)
    db.add(row)
    db.commit()
    return row


def _derived_probe(db, world, experiment_id) -> ProbeMonitor:
    probe = ProbeMonitor(run_id=world.run_id, layer=1, rule="mean", derived_from={"kind": "wsp"},
                         wsp_experiment_id=experiment_id)
    db.add(probe)
    db.commit()
    return probe


def test_defaults_and_id_shape(sync_db):
    row = _experiment(sync_db, seed_probe(sync_db))
    assert row.id.startswith("wsx_") and (row.mode, row.status) == ("session", "importing_bundle")
    assert row.miforge_run_id is None and row.verdict_id is None and row.imported_probe_id is None


def test_one_miforge_run_is_one_experiment(sync_db):
    world = seed_probe(sync_db)
    _experiment(sync_db, world, miforge_run_id="run_1")
    with pytest.raises(IntegrityError):
        _experiment(sync_db, world, miforge_run_id="run_1")
    sync_db.rollback()
    # Many experiments may wait for their run id (NULL is not a duplicate).
    _experiment(sync_db, world)
    _experiment(sync_db, world)


def test_deleting_the_imported_probe_keeps_the_experiment(sync_db):
    world = seed_probe(sync_db)
    row = _experiment(sync_db, world, miforge_run_id="run_2")
    derived = _derived_probe(sync_db, world, row.id)
    row.imported_probe_id = derived.id
    sync_db.commit()
    sync_db.delete(derived)
    sync_db.commit()
    sync_db.expire_all()
    kept = sync_db.get(WspExperiment, row.id)
    assert kept is not None and kept.imported_probe_id is None and kept.miforge_run_id == "run_2"


def test_deleting_the_experiment_unlinks_the_derived_probe(sync_db):
    world = seed_probe(sync_db)
    row = _experiment(sync_db, world)
    derived = _derived_probe(sync_db, world, row.id)
    sync_db.delete(row)
    sync_db.commit()
    sync_db.expire_all()
    kept = sync_db.get(ProbeMonitor, derived.id)
    assert kept is not None and kept.wsp_experiment_id is None


def test_deleting_the_parent_probe_keeps_the_experiment_unlinked(sync_db):
    """Operator decision 2026-11-24 (B5 Q2): the row outlives its probe — and the bundle, which cascades from the probe
    — with both links NULL and miForge's run id and the verdict id intact, as score calls keep their record."""
    world = seed_probe(sync_db)
    row_id = _experiment(sync_db, world, miforge_run_id="run_p", verdict_id="vdt_p").id
    sync_db.delete(sync_db.get(ProbeMonitor, world.probe_id))
    sync_db.commit()
    sync_db.expunge_all()
    kept = sync_db.get(WspExperiment, row_id)
    assert kept is not None
    assert (kept.probe_id, kept.bundle_id) == (None, None)
    assert (kept.miforge_run_id, kept.verdict_id) == ("run_p", "vdt_p")


def test_deleting_the_bundle_keeps_the_experiment_unlinked(sync_db):
    """The same for a bundle deleted on its own: the bundle link is NULL and the probe link stays."""
    world = seed_probe(sync_db)
    row = _experiment(sync_db, world, miforge_run_id="run_b", verdict_id="vdt_b")
    row_id, bundle_id = row.id, row.bundle_id
    sync_db.delete(sync_db.get(WspBundle, bundle_id))
    sync_db.commit()
    sync_db.expunge_all()
    kept = sync_db.get(WspExperiment, row_id)
    assert kept is not None
    assert (kept.probe_id, kept.bundle_id) == (world.probe_id, None)
    assert (kept.miforge_run_id, kept.verdict_id) == ("run_b", "vdt_b")


#: `models.wsp.WspExperiment.status`: mid-flight while the bundle imports into miForge, the start awaits miForge's
#: approval, the run runs, or miWorkbench evaluates it; finished once verdicted, imported, refused or failed.
IN_FLIGHT = ("importing_bundle", "pending_approval", "running", "evaluating")
FINISHED = ("verdicted", "imported", "refused", "failed")


@pytest.mark.parametrize("status", IN_FLIGHT + FINISHED)
def test_the_drain_check_reads_an_open_experiment_as_in_flight(status):
    """`scripts/rebrand/inflight.py` (the read-only drain check before a rollout) counts an open experiment as work in
    flight and a finished one as done. `test_inflight_check.py` only asks that every value be CLASSIFIED; this pins
    WHICH way (control M11: `pending_approval` read as finished stayed green without it)."""
    from scripts.rebrand import inflight

    found = inflight.findings({("wsp_experiments", "status"): {status: 1}}, {})
    assert bool(found) is (status in IN_FLIGHT), (status, found)


REVISION, PARENT = "408803691ec6", "a698f0b14a0b"
BACKEND = Path(__file__).resolve().parents[2]


def test_the_migration_round_trips():
    """`408803691ec6` upgrades, downgrades and upgrades again on a fresh database built by the migrations (named
    after this run's own test database and xdist worker, so parallel runs never share it): the table, its indexes,
    the column and its key appear, disappear and reappear, and the downgrade leaves the parent's schema untouched."""
    from urllib.parse import urlparse

    import psycopg2

    from tests.support.test_databases import sync_test_url

    source = sync_test_url()  # never the dev database (OSD-57)
    parsed = urlparse(source.replace("+psycopg2", "").replace("+asyncpg", ""))
    assert parsed.hostname and parsed.username, f"DATABASE_URL_SYNC unparseable: {source!r}"
    own = (parsed.path or "/mistudio_test").lstrip("/").split("?")[0] or "mistudio_test"
    name = f"{own}_wsx_mig_{os.getenv('PYTEST_XDIST_WORKER', 'solo')}"[:63]
    admin_kw = dict(host=parsed.hostname, port=parsed.port or 5432, user=parsed.username,
                    password=parsed.password or "", dbname="postgres")

    def admin(sql):
        conn = psycopg2.connect(**admin_kw)
        conn.autocommit = True
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
        finally:
            conn.close()

    admin(f'DROP DATABASE IF EXISTS "{name}"')
    admin(f'CREATE DATABASE "{name}"')
    url = (f"postgresql://{parsed.username}:{parsed.password or ''}@{parsed.hostname}:"
           f"{parsed.port or 5432}/{name}")
    env = {**os.environ, "DATABASE_URL_SYNC": url,
           "DATABASE_URL": url.replace("postgresql://", "postgresql+asyncpg://")}

    def alembic(*args):
        done = subprocess.run([sys.executable, "-m", "alembic", *args], cwd=BACKEND, env=env,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]

    def shape():
        engine = sa.create_engine(url)
        try:
            inspector = sa.inspect(engine)
            tables = set(inspector.get_table_names())
            columns = {c["name"] for c in inspector.get_columns("probe_monitors")}
            keys = {fk["name"] for fk in inspector.get_foreign_keys("probe_monitors")}
            present = "wsp_experiments" in tables
            # A unique constraint's backing index is reported by `get_indexes` too: compare it as a constraint.
            indexes = ({i["name"] for i in inspector.get_indexes("wsp_experiments")
                        if not i.get("duplicates_constraint")} if present else set())
            indexes |= ({u["name"] for u in inspector.get_unique_constraints("wsp_experiments")} if present else set())
            with engine.connect() as conn:
                head = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one()
        finally:
            engine.dispose()
        return ("wsp_experiments" in tables, "wsp_experiment_id" in columns,
                "fk_probe_monitors_wsp_experiment_id" in keys, indexes, head)

    added = (True, True, True, {"ix_wsp_experiments_probe_id", "ix_wsp_experiments_bundle_id",
                                "uq_wsp_experiments_miforge_run_id"}, REVISION)
    try:
        alembic("upgrade", REVISION)
        assert shape() == added
        alembic("downgrade", PARENT)
        assert shape() == (False, False, False, set(), PARENT)
        alembic("upgrade", REVISION)
        assert shape() == added
    finally:
        admin(f'DROP DATABASE IF EXISTS "{name}"')
