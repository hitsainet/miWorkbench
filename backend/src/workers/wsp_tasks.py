"""ENH-010 044 Celery tasks (FTDD §6 `workers/wsp_tasks.py`).

* `capture_wsp_heldout` (queue `extraction`, a GPU job): the held-out capture of FR-1 (`services/wsp_capture.py`).
* `build_wsp_bundle`, `import_wsp_candidate` (queue `wsp`, CPU): with tasks 6.6 and 8.6.

Routes are exact task names in `core/celery_app.py` (test_wsp_worker_manifest.py, test_audit_mutation_pins.py).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core.celery_app import celery_app
from .base_task import DatabaseTask
from .gpu_job import gpu_job

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, base=DatabaseTask, name="src.workers.wsp_tasks.capture_wsp_heldout", max_retries=0)
@gpu_job("probe_monitor_run")
def capture_wsp_heldout(self, probe_id: str, capture_ids: List[str]) -> Dict[str, Any]:
    """Capture each held-out set of `capture_ids` for `probe_id`, then check its parity (FR-1).

    Narrated like an evaluation (2026-11-27): the route's `wsp_heldout_capture` task-queue row moves with the
    rows captured of the rows planned and the set it is on (`jlens_progress.RowNarration`)."""
    from ..services.wsp_capture import run_captures
    from ..services.gpu_job_claim import JobHandoff
    from .jlens_progress import WSP_HELDOUT_CAPTURE, RowNarration

    narration = RowNarration(self)
    narration.start()
    with self.get_db() as db:
        try:
            outcome = run_captures(db, probe_id, list(capture_ids), progress=narration.meter(WSP_HELDOUT_CAPTURE))
        except JobHandoff:
            narration.requeued()
            raise
        except Exception as exc:
            narration.failed(exc)
            raise
    narration.completed()
    return outcome


@celery_app.task(bind=True, base=DatabaseTask, name="src.workers.wsp_tasks.build_wsp_bundle", max_retries=0)
def build_wsp_bundle(self, bundle_id: str, heldout_view_ids: Optional[List[str]] = None,
                     supplement_view_ids: Optional[List[str]] = None,
                     used_heldout: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Build one bundle (FR-2, FR-4): practice rebuild, overlap checks, files, baselines, manifest (CPU). The request's
    named held-out sets and supplements (FU-7a C1, C2), and the held-out uses miForge reported to the route, travel
    with the task; None = every ready held-out set, no supplement, no use known."""
    from ..services.wsp_bundle import build_bundle_row

    with self.get_db() as db:
        return build_bundle_row(db, bundle_id, heldout_view_ids=heldout_view_ids,
                                supplement_view_ids=supplement_view_ids, used_heldout=used_heldout)


@celery_app.task(bind=True, base=DatabaseTask, name="src.workers.wsp_tasks.import_wsp_candidate", max_retries=0)
def import_wsp_candidate(self, run_document: Dict[str, Any], patch_manifest: Dict[str, Any], edit_npz_b64: str,
                         acknowledgement: Any = None) -> Dict[str, Any]:
    """Import one verdicted candidate (FR-8, CPU): every check again, then the probe, P23, bar, evaluations, rung.
    A refusal is the task's answer (`{"refused": code, ...}`), never a retry.

    Its `wsp_candidate_import` task-queue row (opened by the route under this task's id, 2026-11-27) is what
    `get_task_status` answers: `running`, then `completed` naming the NEW probe (`jlens_progress.complete_as`), or
    `failed` with `refused <code>: <message>` or the exception."""
    from ..services import wsp_import
    from . import jlens_progress

    task_id: str = getattr(self.request, "id", None) or ""   # "" = run directly: no row
    has_row = bool(task_id) and jlens_progress.mark_running(task_id, progress=0.0)

    def failed(message: str) -> None:
        if has_row:
            jlens_progress.update_row(task_id, status="failed", error_message=message[:2000])

    with self.get_db() as db:
        try:
            result = wsp_import.import_candidate(db, run_document=run_document, patch_manifest=patch_manifest,
                                                 edit_npz=wsp_import.decode_arrays(edit_npz_b64),
                                                 acknowledgement=acknowledgement)
        except wsp_import.ImportRefused as refused:
            logger.warning("WSP candidate import refused: %s", refused.code)
            failed(f"refused {refused.code}: {refused.message}")
            return {"refused": refused.code, **refused.detail()}
        except Exception as exc:
            failed(f"{type(exc).__name__}: {exc}")
            raise
    if has_row:
        jlens_progress.complete_as(task_id, result["probe_id"])
    return result
