"""The covered-route fixture rebuilds a schema whose models changed (046 P8; `tests/unit/covered_routes/conftest.py`).

Control M44: `fingerprint` reduced to the table names (what the first version checked). `test_a_new_column_changes_it`
is red, and a rebased tree runs on a schema missing a column (every covered case answered 500).
"""
from sqlalchemy import Column, Integer, MetaData, String, Table

from tests.support import schema_reset


def _metadata(*extra):
    metadata = MetaData()
    Table("t_one", metadata, Column("id", Integer, primary_key=True), Column("name", String(10)), *extra)
    return metadata


def test_it_is_stable():
    assert schema_reset.fingerprint(_metadata()) == schema_reset.fingerprint(_metadata())


def test_a_new_column_changes_it():
    assert schema_reset.fingerprint(_metadata()) != schema_reset.fingerprint(_metadata(Column("flag", Integer)))


def test_a_changed_type_or_index_changes_it():
    from sqlalchemy import Index

    base = schema_reset.fingerprint(_metadata())
    other = MetaData()
    Table("t_one", other, Column("id", Integer, primary_key=True), Column("name", String(20)))
    assert schema_reset.fingerprint(other) != base
    indexed = _metadata()
    Index("ix_t_one_name", indexed.tables["t_one"].c.name)
    assert schema_reset.fingerprint(indexed) != base


def test_the_models_have_one():
    assert len(schema_reset.fingerprint()) == 64
