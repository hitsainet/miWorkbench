"""Reclaim WSP captures and bundles whose worker is gone (ENH-010 044 task 3.4; FTDD §7; memory
`nlp-status-had-no-janitor`, `long-phases-need-a-db-heartbeat`).

A held-out capture (`wsp_activation_captures`, GPU, queue `extraction`) and a bundle build (`wsp_bundles`, CPU, queue
`wsp`) each hold a live status while their task runs. A worker that vanishes (a rollout, an OOM kill) cannot write a
terminal status, so without this sweep the row reads `running` / `building` forever: the capture's unique key
(probe, view, evaluation) then refuses a fresh request and the UI shows work that is not happening.

Swept every ten minutes from beat (`cleanup-stuck-wsp-rows`):

* a STARTED `running` capture or `building` bundle (one whose task has written `heartbeat_at`) whose last heartbeat
  is older than its threshold → `failed` with `error = "worker_lost: …"`. The tasks write `heartbeat_at` when they
  start and between batches and stages, so silence this long means the process is gone, not slow;
* a `pending` capture, or a `building` bundle with no heartbeat yet (still queued), older than `PENDING_LIMIT` →
  `failed` / `never_started` (its message was lost; a queue that is merely long is far shorter than this).

R3 review fix (OR3-044a-L2), 2026-11-17: a queued bundle used to be aged from `created_at` against `BUNDLE_STALE`, so
one waiting on the solo `wsp` worker behind a long build was reaped `worker_lost`, and the late task then built it
`ready` anyway. The bundle task now claims its row at start (`wsp_bundle.claim_building`) and stops, writing nothing,
on a row that is no longer `building`.

⚠ NEVER REQUEUES and never deletes a file: a reaped row stays failed with its reason (the operator asks again); files
under the run directory are reclaimed with the run (FTDD §4.4).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ..core.celery_app import celery_app
from .base_task import DatabaseTask

logger = logging.getLogger(__name__)

#: The capture task heartbeats after every forward batch and every parity step; the longest single step is one
#: batch, far below this. Generous on purpose: reaping a live GPU capture is worse than a late reap.
CAPTURE_STALE = timedelta(minutes=90)
#: The bundle task heartbeats between stages and every 256 MiB of the practice-file hash.
BUNDLE_STALE = timedelta(minutes=60)
#: A capture still `pending`, or a bundle still `building` with no heartbeat, after this long never reached a worker.
PENDING_LIMIT = timedelta(hours=24)

WORDS = {
    "worker_lost": "worker_lost: the worker running this stopped reporting; nothing more will be written. Ask again.",
    "never_started": "never_started: no worker picked this up within 24 hours; its task was lost. Ask again.",
}


def _aware(value):
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _verdict(row, live_status: str, stale: timedelta, now) -> str | None:
    if row.status not in ("pending", live_status):
        return None
    if row.status == "pending" or row.heartbeat_at is None:   # never started: queued, or its message was lost
        created = _aware(row.created_at)
        return "never_started" if created is not None and now - created > PENDING_LIMIT else None
    last = _aware(row.heartbeat_at)
    return "worker_lost" if now - last > stale else None


def sweep(db, now=None) -> dict:
    from ..models.wsp import WspActivationCapture, WspBundle

    now = now or datetime.now(timezone.utc)
    reaped = {"captures": 0, "bundles": 0}
    plan = (
        ("captures", WspActivationCapture, ("pending", "running"), "running", CAPTURE_STALE),
        ("bundles", WspBundle, ("building",), "building", BUNDLE_STALE),
    )
    for key, model, statuses, live_status, stale in plan:
        candidates = [r.id for r in db.query(model.id).filter(model.status.in_(statuses)).all()]
        db.rollback()
        for row_id in candidates:
            # Decided and written under the row's lock (OR3-044a-L2): a row a task holds (its final `ready` commit)
            # is skipped this sweep, never overwritten after the task finishes.
            row = (db.query(model).filter(model.id == row_id, model.status.in_(statuses))
                   .with_for_update(skip_locked=True).populate_existing().first())
            code = None if row is None else _verdict(row, live_status, stale, now)
            if code is None:
                db.rollback()
                continue
            row.status, row.error = "failed", WORDS[code]
            if hasattr(row, "finished_at"):
                row.finished_at = now
            db.commit()
            logger.warning("WSP %s %s reaped: %s", key[:-1], row.id, code)
            reaped[key] += 1
    return reaped


@celery_app.task(bind=True, base=DatabaseTask, name="cleanup_stuck_wsp_rows")
def cleanup_stuck_wsp_rows(self) -> dict:
    with self.get_db() as db:
        return sweep(db)
