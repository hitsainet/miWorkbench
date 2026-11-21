"""The MIGRATED schema's calibration-capture index covers only current rows (confirm-044a4 L4, 2026-11-17).

Unit databases are built from the ORM, so a migration that created `uq_wsp_capture_calibration` without
`superseded_at IS NULL` passed every unit test; the reviewer re-migrated with the old predicate and the schema guards
stayed green. In production that index would answer every capture request after a recalibration with a 500 (the old
row and its successor are both `calibration` rows). This reads the index definition from the schema CI migrates with
Alembic (`SCHEMA_CHECK_DATABASE_URL`, the backend job's migrate step).

Control: migrate with the round-3 predicate (`WHERE role = 'calibration'`) → red.
"""
from __future__ import annotations

import os

import psycopg2
import pytest


@pytest.fixture
def migrated_dsn():
    dsn = os.getenv("SCHEMA_CHECK_DATABASE_URL")
    if not dsn:
        pytest.skip("SCHEMA_CHECK_DATABASE_URL is not set")
    scheme, separator, rest = dsn.partition("://")
    return f"{scheme.split('+', 1)[0]}{separator}{rest}"


def test_the_calibration_index_excludes_superseded_rows(migrated_dsn):
    connection = psycopg2.connect(migrated_dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT indexdef FROM pg_indexes WHERE indexname = 'uq_wsp_capture_calibration'")
            rows = [r[0] for r in cursor.fetchall()]
    finally:
        connection.close()
    assert len(rows) == 1, f"uq_wsp_capture_calibration not found in the migrated schema: {rows}"
    definition = rows[0].lower()
    assert "unique" in definition and "(probe_id, dataset_id)" in definition, definition
    assert "superseded_at is null" in definition, (
        "the migrated index still covers superseded calibration rows: every recapture after a recalibration would "
        f"violate it ({rows[0]})"
    )
