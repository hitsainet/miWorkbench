"""ENH-010 044 task 3.4: the WSP janitor fails a capture or bundle whose worker vanished; never requeues.

Control (0xcc/reviews/044_implementation_controls_2026-11-16.md): drop the beat entry → `test_the_beat_entry_is_wired`
red; drop the include line → `test_a_worker_registers_the_janitor` red.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.models.wsp import WspActivationCapture, WspBundle
from src.workers import cleanup_stuck_wsp_rows as janitor
from tests.support.wsp_world import add_view, seed_probe, sync_db  # noqa: F401 (fixture)

NOW = datetime.now(timezone.utc)


def _capture(db, world, view_id, status, *, heartbeat=None, created=None):
    row = WspActivationCapture(probe_id=world.probe_id, dataset_id=view_id, role="eval", layer=1, scope="all",
                               status=status, heartbeat_at=heartbeat, created_at=created or NOW)
    db.add(row)
    db.commit()
    return row


def _bundle(db, world, status, *, heartbeat=None, created=None):
    row = WspBundle(probe_id=world.probe_id, status=status, heartbeat_at=heartbeat, created_at=created or NOW)
    db.add(row)
    db.commit()
    return row


def test_stale_running_capture_is_failed(sync_db):
    world = seed_probe(sync_db)
    stale = _capture(sync_db, world, add_view(sync_db, world),
                     "running", heartbeat=NOW - janitor.CAPTURE_STALE - timedelta(seconds=5))
    alive = _capture(sync_db, world, add_view(sync_db, world), "running", heartbeat=NOW - timedelta(minutes=5))
    reaped = janitor.sweep(sync_db, NOW)
    sync_db.refresh(stale), sync_db.refresh(alive)
    assert stale.status == "failed" and stale.error.startswith("worker_lost") and stale.finished_at is not None
    assert alive.status == "running" and reaped["captures"] >= 1


def test_stale_building_bundle_is_failed_and_a_fresh_one_is_spared(sync_db):
    world = seed_probe(sync_db)
    stale = _bundle(sync_db, world, "building", heartbeat=NOW - janitor.BUNDLE_STALE - timedelta(seconds=5))
    fresh = _bundle(sync_db, world, "building", heartbeat=NOW - timedelta(minutes=1))
    janitor.sweep(sync_db, NOW)
    sync_db.refresh(stale), sync_db.refresh(fresh)
    assert (stale.status, fresh.status) == ("failed", "building") and stale.error.startswith("worker_lost")


def test_a_queued_bundle_is_reaped_only_as_never_started(sync_db):
    """R3 review fix (OR3-044a-L2): a `building` bundle with no heartbeat has not started (the solo `wsp` worker is busy
    with another build), so `BUNDLE_STALE` does not apply to it: past `BUNDLE_STALE` it is spared, and only past
    `PENDING_LIMIT` is it failed `never_started`. Control: age a queued bundle from `created_at` again → red."""
    world = seed_probe(sync_db)
    queued = _bundle(sync_db, world, "building", created=NOW - janitor.BUNDLE_STALE - timedelta(minutes=30))
    lost = _bundle(sync_db, world, "building", created=NOW - janitor.PENDING_LIMIT - timedelta(minutes=1))
    janitor.sweep(sync_db, NOW)
    sync_db.refresh(queued), sync_db.refresh(lost)
    assert queued.status == "building" and queued.error is None
    assert lost.status == "failed" and lost.error.startswith("never_started")


def test_a_row_a_task_holds_is_skipped_not_overwritten(sync_db):
    """The janitor decides and writes each row under its lock, skipping a row another transaction holds (a build
    committing `ready`), so it never overwrites a status a task wrote after the janitor read the row. Control: read the
    rows without `FOR UPDATE SKIP LOCKED` → the sweep waits on the held row → lock timeout → red."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    world = seed_probe(sync_db)
    held = _bundle(sync_db, world, "building", heartbeat=NOW - janitor.BUNDLE_STALE - timedelta(minutes=5))
    stale = _bundle(sync_db, world, "building", heartbeat=NOW - janitor.BUNDLE_STALE - timedelta(minutes=5))
    from sqlalchemy import create_engine

    # The janitor's own engine, every connection of it with a lock timeout: without SKIP LOCKED its sweep would wait
    # on the held row for ever; with it the timeout turns that wait into a failure.
    impatient = create_engine(sync_db.get_bind().url, connect_args={"options": "-c lock_timeout=3000"})
    try:
        with Session(bind=sync_db.get_bind()) as task:
            task.execute(text("SELECT 1 FROM wsp_bundles WHERE id = :id FOR UPDATE"), {"id": held.id})
            with Session(bind=impatient) as janitor_session:
                janitor.sweep(janitor_session, NOW)
            task.rollback()
    finally:
        impatient.dispose()
    sync_db.refresh(held), sync_db.refresh(stale)
    assert (held.status, stale.status) == ("building", "failed")


def test_a_pending_capture_is_failed_only_after_the_pending_limit(sync_db):
    world = seed_probe(sync_db)
    lost = _capture(sync_db, world, add_view(sync_db, world), "pending",
                    created=NOW - janitor.PENDING_LIMIT - timedelta(minutes=1))
    queued = _capture(sync_db, world, add_view(sync_db, world), "pending", created=NOW - timedelta(hours=5))
    janitor.sweep(sync_db, NOW)
    sync_db.refresh(lost), sync_db.refresh(queued)
    assert lost.status == "failed" and lost.error.startswith("never_started") and queued.status == "pending"


def test_terminal_rows_are_untouched(sync_db):
    world = seed_probe(sync_db)
    done = _capture(sync_db, world, add_view(sync_db, world), "ready", heartbeat=NOW - timedelta(days=3))
    janitor.sweep(sync_db, NOW)
    sync_db.refresh(done)
    assert done.status == "ready" and done.error is None


def test_the_beat_entry_is_wired():
    from src.core.celery_app import celery_app

    entry = celery_app.conf.beat_schedule["cleanup-stuck-wsp-rows"]
    assert entry["task"] == "cleanup_stuck_wsp_rows" and entry["schedule"] == 600.0
    queue = entry["options"]["queue"]
    assert getattr(queue, "name", queue) == "low_priority" and entry["options"]["expires"] < 600
    assert celery_app.conf.task_routes["cleanup_stuck_wsp_rows"] == {"queue": "low_priority"}


def test_a_worker_registers_the_janitor():
    """A worker registers only the include list's modules; looked up in a fresh interpreter's registry."""
    from tests.unit.test_training_evaluation_task_registered_in_worker import _worker_registry

    assert "cleanup_stuck_wsp_rows" in _worker_registry()
