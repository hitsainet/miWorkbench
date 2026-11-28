"""The covered-route cases' database (046 P8): the schema built once per process, emptied between tests.

The suite's `async_engine` (tests/conftest.py) drops and recreates every table and enum for every test, which costs
6 to 19 s per test on this schema, more than a case itself. These modules only add and read rows, so they share one
schema and start each test with every table emptied (`_empty`). The schema is
rebuilt (`schema_reset.rebuild`, the suite's own) unless its marker (a comment on the first table) carries the
models' `schema_reset.fingerprint()`: a first run, another module's teardown, or a schema an older checkout built
(a missing column) all rebuild, so a test here never runs on a stale or partial schema. The
app's own engine reads the same database.
"""
from __future__ import annotations

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from src.core.config import settings
from src.core.database import Base
from tests.support import schema_reset


def _database_url() -> str:
    url = str(settings.database_url)
    if "postgresql" in url and "test" not in url:
        url = url.rsplit("/", 1)[0] + "/mistudio_test"  # as tests/conftest.py's `async_engine`
    return url


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Overrides tests/conftest.py's for these modules: the models' tables, every one empty."""
    engine = create_async_engine(_database_url(), poolclass=NullPool)
    marker = f'"{Base.metadata.sorted_tables[0].name}"'
    wanted = f"covered-routes schema {schema_reset.fingerprint()}"
    async with engine.begin() as conn:
        rows = await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"))
        same_tables = {r[0] for r in rows} == {t.name for t in Base.metadata.sorted_tables}
        built_as = (await conn.execute(text(f"SELECT obj_description('{marker}'::regclass, 'pg_class')"))).scalar() \
            if same_tables else None
        if built_as != wanted:
            # Not built by this fixture from these models (a first run, another module's teardown, or an older
            # checkout's schema): rebuilt, and marked with the models' fingerprint on a table the rebuild owns.
            await schema_reset.rebuild(conn)
            await conn.execute(text(f"COMMENT ON TABLE {marker} IS '{wanted}'"))
        else:
            await _empty(conn)
    yield engine
    # No teardown drop: the next test here empties the tables, and the suite's own fixture drops and rebuilds them.
    await engine.dispose()


async def _empty(conn) -> None:
    """Every table's rows deleted, dependents first, in one transaction. A `DELETE` per table, not a `TRUNCATE`: a
    `TRUNCATE` gives every table and index a new file (4.5 s for the whole schema here, 0.8 s for the dozen
    append-only tables alone), a `DELETE` of a few rows does not. The append-only tables refuse a row `DELETE` by
    trigger, so their user triggers are off for the transaction's own delete and back on before it commits."""
    await conn.execute(text("SET LOCAL lock_timeout = '30s'"))
    guarded = {r[0] for r in await conn.execute(text(
        "SELECT DISTINCT c.relname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
        "WHERE NOT t.tgisinternal AND c.relnamespace = current_schema()::regnamespace"))}
    for table in reversed(Base.metadata.sorted_tables):
        if table.name in guarded:
            await conn.execute(text(f'ALTER TABLE "{table.name}" DISABLE TRIGGER USER'))
        await conn.execute(table.delete())
        if table.name in guarded:
            await conn.execute(text(f'ALTER TABLE "{table.name}" ENABLE TRIGGER USER'))
