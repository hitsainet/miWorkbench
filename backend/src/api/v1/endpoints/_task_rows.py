"""The task-queue row a long GPU task is sent under, so Active Operations and the page that started it can follow it.

The ENH-006 G17 order (`probe_monitors.build_definition`), shared by the probe evaluation and the WSP held-out capture
(2026-11-27, operator: "what is this job? where can I see its progress in a UI?"):

1. `open_task_row` chooses the Celery id and opens the `task_queue` row (`queued`) under it, BEFORE the task is sent,
   so a task that fails in its first milliseconds cannot write its end before the insert commits;
2. the route sends the task itself with `dispatch_gpu_task(..., task_id=celery_id)` — inline, so the route walk and
   the dispatch guards resolve the task at the call site;
3. a send that raises goes through `fail_unsent_row`, so the row never waits for a task that will not come.

A row that cannot be written is a 503 and nothing is sent: the work would run unseen, which is the defect this fixes.
The task moves the row (`jlens_progress.RowNarration`).
"""
from __future__ import annotations

import uuid
from typing import Optional, Tuple

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool


async def open_task_row(task_type: str, entity_id: str, *, what: str, gpu_request: Optional[str] = "auto",
                        entity_type: str = "probe_monitor") -> Tuple[str, str]:
    """`(celery_id, task_queue_id)`: the row is committed before this returns. `gpu_request=None` for a CPU task
    (the WSP import): its row records no GPU request."""
    from ....workers import jlens_progress

    celery_id = str(uuid.uuid4())
    task_queue_id = await run_in_threadpool(
        jlens_progress.open_row, task_type, entity_id, celery_id,
        {"gpu": gpu_request} if gpu_request is not None else None, entity_type,
    )
    if task_queue_id is None:
        raise HTTPException(status_code=503, detail=(
            f"could not record the {what} in the task queue, so it would run with no progress anywhere; nothing "
            "was dispatched. Try again once the database is writable."))
    return celery_id, task_queue_id


async def fail_unsent_row(celery_id: str, exc: BaseException) -> None:
    """The send raised: the row is failed with why, never left `queued`."""
    from ....workers import jlens_progress

    await run_in_threadpool(
        jlens_progress.update_row, celery_id, "failed", None, f"not dispatched: {type(exc).__name__}: {exc}"[:2000])
