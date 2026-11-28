"""`real_broker` chooses its Redis database from `PYTEST_XDIST_WORKER`, for xdist's `gw<N>` AND for a named worker.

A SERIAL run on a private test database sets `PYTEST_XDIST_WORKER=<name>` (`tests/support/xdist_isolation.py`; memory
"private test DB dies under xdist"). The fixture did `int()` on whatever followed `gw`, so every test using it errored
in setup under such a run (`ValueError: invalid literal for int()`; 042 acceptance 2026-11-26, F-5, where the walk had
to borrow `gw7` to get past it).
"""
from __future__ import annotations

import pytest

from tests.support import real_broker as rb
from tests.support.real_broker import real_broker  # noqa: F401  (fixture)

NAMED = "rbfix-named"


@pytest.mark.parametrize(("worker", "database"), [(None, 8), ("", 8), ("gw0", 8), ("gw1", 9), ("gw3", 11)])
def test_xdist_workers_keep_their_databases(worker, database):
    assert rb.broker_database(worker) == database


@pytest.mark.parametrize("worker", ["mcpux4", NAMED, "gw", "gwx"])
def test_a_named_worker_gets_a_stable_database_of_the_named_range(worker):
    database = rb.broker_database(worker)
    assert database in rb.NAMED_WORKER_DATABASES and database == rb.broker_database(worker)


@pytest.fixture
def _named_worker(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", NAMED)


def test_the_fixture_runs_under_a_named_worker(_named_worker, real_broker):  # noqa: F811
    """The WIRING: the fixture itself, set up under a named worker (requested after the env is set), answers on the
    database `broker_database` chose for that name, and the database it hands out works."""
    client, url = real_broker
    assert int(url.rsplit("/", 1)[1]) == rb.broker_database(NAMED), url
    client.set("real-broker-probe", b"1")
    assert client.get("real-broker-probe") == b"1"
