"""ENH-010 044, B5a follow-up: each WSP score call records the arithmetic it was computed with

Revision ID: 5e3c9a1d7b42
Revises: 408803691ec6
Create Date: 2026-11-24

One nullable JSONB column, `wsp_score_calls.arithmetic`: `{"mkl_kernel_family", "torch_version", "scorer_threads",
"numpy_version"}` at score time (`wsp_score.arithmetic_record`). When the import's P23 bit check fails, it compares
this record with the importing host's and refuses `persistence_mismatch_arithmetic_differs`, naming the field
(operator approval 2026-11-24, review record `0xcc/reviews/b5a_import_determinism_2026-11-24.md`, product question 1).

NULL for every call recorded before this revision: such a call keeps the old `persistence_mismatch`. No backfill: the
arithmetic of a past call is not known, and a guess would name a cause that may be false.

Purely additive; the downgrade drops exactly what this adds.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "5e3c9a1d7b42"
down_revision: Union[str, None] = "408803691ec6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("wsp_score_calls",
                  sa.Column("arithmetic", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("wsp_score_calls", "arithmetic")
