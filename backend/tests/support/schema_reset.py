"""The test schema's lifecycle, shared by tests/conftest.py's `async_engine` (a fresh schema per test) and the
covered-route cases' (tests/unit/covered_routes/conftest.py: one schema per process, emptied between tests)."""
from __future__ import annotations

from sqlalchemy import text

# PostgreSQL enum types required by models that use create_type=False. These are NOT created by
# Base.metadata.create_all (create_type=False), so the fixture must manage their lifecycle explicitly. Values must
# match the model definitions exactly (label_source_enum includes 'enhanced_llm').
ENUMS = [
    ("export_status", ["pending", "computing", "packaging", "completed", "failed", "cancelled"]),
    ("label_source_enum", ["auto", "user", "llm", "local_llm", "openai", "enhanced_llm", "mcp_agent"]),
    ("analysis_type_enum", ["logit_lens", "correlations", "ablation", "nlp_analysis"]),
    ("extraction_status_enum", ["queued", "loading", "extracting", "saving", "completed", "failed", "cancelled"]),
]


async def drop(conn) -> None:
    """Tables first (they depend on the enums), then the enums, WITHOUT CASCADE: a CASCADE here would drop
    dependent tables out from under drop_all, causing "table does not exist" on the next drop_all/create_all."""
    from src.core.database import Base

    await conn.run_sync(Base.metadata.drop_all)
    for name, _ in ENUMS:
        await conn.execute(text(f"DROP TYPE IF EXISTS {name};"))


async def rebuild(conn) -> None:
    """A clean slate (leftovers of an interrupted teardown dropped first: this prevents the pg_type unique-index race
    on CREATE TYPE), then the enums, then the tables."""
    from src.core.database import Base

    await drop(conn)
    for name, values in ENUMS:
        values_str = ", ".join(f"'{v}'" for v in values)
        await conn.execute(text(f"CREATE TYPE {name} AS ENUM ({values_str});"))
    await conn.run_sync(Base.metadata.create_all)


def fingerprint(metadata=None) -> str:
    """A digest of the schema `rebuild` creates: every table's and index's DDL as PostgreSQL compiles it, and the
    enums. A column, type, default, constraint or index added to a model changes it, so a database built from an
    older checkout of the models never passes for a current one (046 P8: a check of table NAMES alone ran a
    rebased tree on a schema missing `probe_monitors.layer_meaning_changed`, and every case answered 500)."""
    import hashlib
    import json

    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateIndex, CreateTable

    if metadata is None:
        from src.core.database import Base

        metadata = Base.metadata
    dialect = postgresql.dialect()
    parts = [json.dumps(ENUMS)]
    for table in metadata.sorted_tables:
        parts.append(str(CreateTable(table).compile(dialect=dialect)))
        parts.extend(sorted(str(CreateIndex(index).compile(dialect=dialect)) for index in table.indexes))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()
