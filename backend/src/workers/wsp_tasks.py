"""ENH-010 044 Celery tasks (FTDD §6 `workers/wsp_tasks.py`).

* `capture_wsp_heldout` (queue `extraction`, a GPU job): the held-out capture of FR-1 (`services/wsp_capture.py`).
* `build_wsp_bundle`, `import_wsp_candidate` (queue `wsp`, CPU): with tasks 6.6 and 8.6.

Routes are exact task names in `core/celery_app.py` (test_wsp_worker_manifest.py, test_audit_mutation_pins.py).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..core.celery_app import celery_app
from .base_task import DatabaseTask
from .gpu_job import gpu_job

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, base=DatabaseTask, name="src.workers.wsp_tasks.capture_wsp_heldout", max_retries=0)
@gpu_job("probe_monitor_run")
def capture_wsp_heldout(self, probe_id: str, capture_ids: List[str]) -> Dict[str, Any]:
    """Capture each held-out set of `capture_ids` for `probe_id`, then check its parity (FR-1)."""
    from ..services.wsp_capture import run_captures

    with self.get_db() as db:
        return run_captures(db, probe_id, list(capture_ids))


@celery_app.task(bind=True, base=DatabaseTask, name="src.workers.wsp_tasks.build_wsp_bundle", max_retries=0)
def build_wsp_bundle(self, bundle_id: str) -> Dict[str, Any]:
    """Build one bundle (FR-2, FR-4): practice rebuild, overlap checks, files, baselines, manifest (CPU)."""
    from ..services.wsp_bundle import build_bundle_row

    with self.get_db() as db:
        return build_bundle_row(db, bundle_id)


@celery_app.task(bind=True, base=DatabaseTask, name="src.workers.wsp_tasks.import_wsp_candidate", max_retries=0)
def import_wsp_candidate(self, run_document: Dict[str, Any], patch_manifest: Dict[str, Any], edit_npz_b64: str,
                         acknowledgement: Any = None) -> Dict[str, Any]:
    """Import one verdicted candidate (FR-8, CPU): every check again, then the probe, P23, bar, evaluations, rung.
    A refusal is the task's answer (`{"refused": code, ...}`), never a retry."""
    from ..services import wsp_import

    with self.get_db() as db:
        try:
            return wsp_import.import_candidate(db, run_document=run_document, patch_manifest=patch_manifest,
                                               edit_npz=wsp_import.decode_arrays(edit_npz_b64),
                                               acknowledgement=acknowledgement)
        except wsp_import.ImportRefused as refused:
            logger.warning("WSP candidate import refused: %s", refused.code)
            return {"refused": refused.code, **refused.detail()}
