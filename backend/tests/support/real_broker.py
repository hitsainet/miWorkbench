"""A real Redis for tests of kombu's own structures, on a database of its own per xdist worker.

`real_broker` yields `(client, url)`. A message put in `unacked` by kombu itself, not a hand-written imitation of it: a
fixture whose fields agree by construction is the usual reason a suite stays green over a real bug.
"""
from __future__ import annotations

import os
import zlib

import pytest

#: Where a NAMED worker's database lies. xdist's `gw<N>` workers take 8 + N (8-11 at the suite's `-n 4`); a named
#: worker (`PYTEST_XDIST_WORKER=<name>`: a SERIAL run on its own test database, `tests/support/xdist_isolation.py`)
#: takes one of these, chosen by a stable hash of its name.
NAMED_WORKER_DATABASES = range(12, 16)


def broker_database(worker: str | None) -> int:
    """The Redis database for this worker: 8 serially, 8 + N for `gw<N>`, and a stable one of
    `NAMED_WORKER_DATABASES` for any other name. This did `int()` on whatever followed `gw`, so every test using the
    fixture errored in setup under a named worker (042 acceptance 2026-11-26, F-5)."""
    if not worker:
        return 8
    number = worker.removeprefix("gw")
    if worker.startswith("gw") and number.isdigit():
        return 8 + int(number)
    return NAMED_WORKER_DATABASES[zlib.crc32(worker.encode()) % len(NAMED_WORKER_DATABASES)]


@pytest.fixture
def real_broker():
    import redis

    base = os.environ.get("REDIS_URL", "redis://localhost:6379/0").rsplit("/", 1)[0]
    url = f"{base}/{broker_database(os.environ.get('PYTEST_XDIST_WORKER'))}"
    client = redis.Redis.from_url(url, socket_connect_timeout=1.0, socket_timeout=5.0)
    try:
        client.ping()
    except redis.RedisError as exc:
        pytest.skip(f"NO REDIS AT {url}: {exc}. These tests read kombu's real structures.")
    client.flushdb()
    yield client, url
    client.flushdb()
