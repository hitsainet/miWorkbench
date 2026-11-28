"""Capture a wedged GPU worker's stack, once, without stopping it (ENH-011 WB-2, #9; ENH-006 G12 "Defect B").

Every worker registers faulthandler on SIGUSR2 (`core/celery_app.install_stack_dump_signal`): the signal writes every
thread's stack to the worker's `sys.__stderr__`, which is the `celery-gpu-workers` container's log. The wedge watch
(`workers/gpu_worker_watch.py`) finds a wedged worker, but runs on the CPU janitors' queue in another pod and cannot
signal a process there; the GPU supervisor (PID 1 of `celery-gpu-workers`) knows each child's pid. So:

1. the watch, on a `wedged` reading, writes ONE mark per stuck task (`request_stack_dump`) naming the worker that ran
   it: its Celery hostname before the `@`, which is how the supervisor's `WorkerSpec` names it (`gpu-<index>-<uuid8>`);
2. the supervisor, on every poll, reads the marks of its OWN running children (`StackDumpRelay`), consumes each with
   `GETDEL`, sends `SIGUSR2` once to that child and logs `stack_dump_requested worker=<name> pid=<pid>`.

Reading the stack: `kubectl logs <pod> -c celery-gpu-workers`, from the `stack_dump_requested` line on; faulthandler
writes `Thread 0x… (most recent call first):` blocks, the current thread marked `Current thread`.

NOTHING HERE RESTARTS, KILLS OR STOPS ANYTHING: SIGUSR2 is handled by faulthandler and the worker carries on. A mark
naming no child of this supervisor is left alone and expires (`MARK_TTL_S`).
"""
from __future__ import annotations

import json
import logging
import os
import signal
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: `<prefix><worker name>` → `{"task_id", "at"}`: a stack dump the supervisor should request of that worker.
MARK_PREFIX = "miworkbench:gpu:stack-dump:"
#: How long an unconsumed mark lives: a supervisor polls every second, so this only bounds a mark nothing reads.
MARK_TTL_S = 600
#: `<prefix><task id>`: the stuck task's dump was already requested, so a wedge lasting many passes asks once.
REQUESTED_PREFIX = "miworkbench:gpu:stack-dump-requested:"
REQUESTED_TTL_S = 7 * 24 * 3600


def worker_name(hostname: Optional[str]) -> Optional[str]:
    """`gpu-1-247aa582@pod` → `gpu-1-247aa582`, the supervisor's name for the worker; None when there is none."""
    name = (hostname or "").split("@", 1)[0].strip()
    return name or None


def request_stack_dump(client: Any, worker: str, *, task_id: str, now: float) -> bool:
    """Mark `worker` for one stack dump on behalf of stuck task `task_id`. True when this call wrote the mark;
    False when this task's dump was already requested (one mark per stuck task, however many watch passes)."""
    if not client.set(REQUESTED_PREFIX + str(task_id), worker, nx=True, ex=REQUESTED_TTL_S):
        return False
    client.set(MARK_PREFIX + worker, json.dumps({"task_id": str(task_id), "at": now}), ex=MARK_TTL_S)
    return True


def broker_client() -> Any:
    """The broker's Redis, where the watch writes its marks."""
    import redis

    from ..core.config import settings

    return redis.Redis.from_url(str(settings.celery_broker_url), socket_connect_timeout=1.0, socket_timeout=2.0)


class StackDumpRelay:
    """The supervisor's half: called with every worker's status on each poll (`supervise(on_stack_dump=…)`)."""

    def __init__(self, client_factory: Callable[[], Any] = broker_client) -> None:
        self.client_factory = client_factory
        self._client: Any = None

    def __call__(self, statuses: list) -> None:
        from .gpu_supervisor import RUNNING

        running = [s for s in statuses if s.get("state") == RUNNING and s.get("pid")]
        if not running:
            return
        if self._client is None:
            self._client = self.client_factory()
        for status in running:
            name, pid = status["name"], int(status["pid"])
            if self._client.getdel(MARK_PREFIX + name) is None:
                continue
            os.kill(pid, signal.SIGUSR2)
            logger.warning("stack_dump_requested worker=%s pid=%s (faulthandler writes it to this container's log)",
                           name, pid)
