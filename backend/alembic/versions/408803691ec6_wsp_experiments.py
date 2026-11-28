"""ENH-010 044 <044b>: WSP experiments, and the derived probe's link back to its experiment

Revision ID: 408803691ec6
Revises: a698f0b14a0b
Create Date: 2026-11-24

044 FTDD §4.2 / §4.3 (FR-12, FR-13, FR-14; phase 3; FTASKS 13.1). Ported from the unpushed branch `wsp/044-c`
(`c0d8c8192`, revision `b7d3e5f0a812` on `0e4a1c6b9d27`), whose 78 conflicts with `main` ruled out a merge (suite plan
B5, 2026-11-24). This is a NEW revision on the current head: `b7d3e5f0a812` never reached `main`, and reusing it on
another parent would let a database migrated on the branch claim a schema it does not have.

One new table, `wsp_experiments` (one miForge WSP run started for one bundle: the relayed approval, the run id, the
verdict id miWorkbench's own `evaluate` returned (P71), the cached verdict, use counts and program digest), and one
nullable column, `probe_monitors.wsp_experiment_id` (SET NULL; NULL for every probe not imported from an
experiment). The two foreign keys form a cycle (`wsp_experiments.imported_probe_id` → `probe_monitors`), so the
column's key is added after the table. The table has no writer until the experiment driver (FTASKS 13.2) lands.

Deletes (operator decision, 2026-11-24, B5 Q2): an experiment row OUTLIVES its probe and its bundle. `probe_id` and
`bundle_id` are nullable and `ON DELETE SET NULL`, as `wsp_score_calls` keeps its record since the R2 review: the row
holds miForge's run id and the verdict id the import uses, and the run itself lives on in miForge. (The branch had
CASCADE on both; amended here before any push.)

Purely additive; the downgrade drops exactly what this adds.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "408803691ec6"
down_revision: Union[str, None] = "a698f0b14a0b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "wsp_experiments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("probe_id", sa.String(length=36), nullable=True),
        sa.Column("bundle_id", sa.String(length=36), nullable=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("miforge_run_id", sa.String(length=64), nullable=True),
        sa.Column("verdict_id", sa.String(length=64), nullable=True),
        sa.Column("request_sha256", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("approval", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("verdict", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("heldout_uses", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("program_sha256", sa.String(length=64), nullable=True),
        sa.Column("reruns", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("imported_probe_id", sa.String(length=36), nullable=True),
        sa.Column("import_acknowledgement", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["bundle_id"], ["wsp_bundles.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["imported_probe_id"], ["probe_monitors.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["probe_id"], ["probe_monitors.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("miforge_run_id", name="uq_wsp_experiments_miforge_run_id"),
    )
    op.create_index("ix_wsp_experiments_bundle_id", "wsp_experiments", ["bundle_id"], unique=False)
    op.create_index("ix_wsp_experiments_probe_id", "wsp_experiments", ["probe_id"], unique=False)
    op.add_column("probe_monitors", sa.Column("wsp_experiment_id", sa.String(length=36), nullable=True))
    op.create_foreign_key("fk_probe_monitors_wsp_experiment_id", "probe_monitors", "wsp_experiments",
                          ["wsp_experiment_id"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    op.drop_constraint("fk_probe_monitors_wsp_experiment_id", "probe_monitors", type_="foreignkey")
    op.drop_column("probe_monitors", "wsp_experiment_id")
    op.drop_index("ix_wsp_experiments_probe_id", table_name="wsp_experiments")
    op.drop_index("ix_wsp_experiments_bundle_id", table_name="wsp_experiments")
    op.drop_table("wsp_experiments")
