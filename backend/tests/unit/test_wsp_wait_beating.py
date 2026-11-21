"""`_wait_beating` returns a fit's own failure at once and never spins on it (confirm-044a4 L2, 2026-11-17).

A fit that raises its own `TimeoutError` is a finished future. Without the `future.done()` guard the wait loop read
that error as "keep waiting", and the reviewer's probe counted ~147,000 `beat()` calls a second, each a database
UPDATE and commit, for up to a day. The wait must re-raise the fit's own exception with no heartbeat at all, and a
reap seen in a heartbeat must stop the wait while the fit is still running.

Controls: drop the `future.done()` check → the first test spins and fails; drop the beat call → the reap test fails.
No database: the heartbeat is a recorder.
"""
from __future__ import annotations

import concurrent.futures
import time

import pytest

from src.services import wsp_bundle


def test_a_fits_own_timeout_is_raised_at_once_and_never_beaten_on():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(lambda: (_ for _ in ()).throw(TimeoutError("the fit's own")))
        concurrent.futures.wait([future])
        beats: list[int] = []
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="the fit's own"):
            wsp_bundle._wait_beating(future, 1.0, lambda: beats.append(1))
        assert beats == [], f"a finished fit was beaten on {len(beats)} times"
        assert time.monotonic() - started < 0.5, "the wait did not return at once on a finished fit"
    finally:
        pool.shutdown(wait=True)


def test_a_reap_seen_in_a_heartbeat_stops_the_wait(monkeypatch):
    monkeypatch.setattr(wsp_bundle, "BASELINE_BEAT_SECONDS", 0.05)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    class Reaped(Exception):
        pass

    def beat():
        raise Reaped()

    try:
        future = pool.submit(time.sleep, 0.5)
        started = time.monotonic()
        with pytest.raises(Reaped):
            wsp_bundle._wait_beating(future, 10.0, beat)
        assert time.monotonic() - started < 0.4, "the reap was not seen at the first heartbeat"
    finally:
        pool.shutdown(wait=True)
