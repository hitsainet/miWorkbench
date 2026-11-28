"""G5: the truncation mask of an evaluation and of a calibration

Revision ID: a5e1d0c7b2f9
Revises: e045b7c1d2f3
Create Date: 2026-11-16

ENH-010 G5 (``0xcc/tasks/DEBT_TASKS|WSP_G5_Truncated_Rows_2026-11-15.md``, GR-1, GR-6). A row that was cut at
``max_length`` is a shape served traffic never produces (miServer refuses past the context instead of cutting), so it
is neither a negative nor a positive: it is reported and left out of every decided metric.

* ``probe_monitor_evaluations.truncated_path`` — a fifth int8 array beside scores, positions, ``n_scored`` and labels,
  index-for-index with them (1 = the row was truncated).
* ``probe_monitors.calibration_truncated_path`` — the same mask over the calibration negatives
  (``calibration_scores_path``), for calibrations cut after G5.

Purely additive and nullable. NULL means "truncation not recorded" (an evaluation or calibration from before G5), never
"no row was truncated": rung 2 reads such a set as not measured until a CPU backfill writes the mask, and an
arithmetic re-cut of a calibration without the mask uses the full array, so a shipped bar re-cuts bit for bit.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a5e1d0c7b2f9"
down_revision: Union[str, None] = "e045b7c1d2f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "probe_monitor_evaluations",
        sa.Column("truncated_path", sa.String(length=1000), nullable=True),
    )
    op.add_column(
        "probe_monitors",
        sa.Column("calibration_truncated_path", sa.String(length=1000), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("probe_monitors", "calibration_truncated_path")
    op.drop_column("probe_monitor_evaluations", "truncated_path")
