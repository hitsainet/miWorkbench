"""ENH-010 044 (R4 review): a ready capture is never reset; a bundle records what it was sealed on

Revision ID: a698f0b14a0b
Revises: d3f8a2c6e0b4
Create Date: 2026-11-17

044-A confirmation review of the round-3 fixes, finding M1 (`0xcc/reviews/044a_r4_review_fixes_2026-11-17.md`).
Round 3 re-armed a stale READY calibration capture IN PLACE (status, parity and files reset under the same id),
because `uq_wsp_capture_calibration` allowed one calibration row per (probe, view). Every earlier bundle names its
captures by id, so a recalibration followed by a capture request rewrote the evidence under every earlier
calibration-basis bundle: unscorable while the new capture ran, for good if it failed parity, and, after a G5 re-cut,
scored on rows its manifest does not name.

* `wsp_activation_captures.superseded_at` (timestamptz) and `superseded_by` (the replacement capture's id, no foreign
  key): a stale ready calibration capture is superseded by a NEW row, never reset.
* `uq_wsp_capture_calibration` covers only the current row: `role = 'calibration' AND superseded_at IS NULL`.
* `wsp_bundles.capture_seals` (JSONB): per capture id, the sha256 of the parity record the bundle was built on.

Additive apart from the rebuilt index. The 044 tables are new in this push (044-A), so no live row exists. The
downgrade restores the old index exactly; it fails, by design, while a superseded calibration row exists beside its
replacement (the old index allows one calibration row per view).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a698f0b14a0b"
down_revision: Union[str, None] = "d3f8a2c6e0b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("wsp_activation_captures", sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("wsp_activation_captures", sa.Column("superseded_by", sa.String(length=36), nullable=True))
    op.add_column("wsp_bundles", sa.Column("capture_seals", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.drop_index("uq_wsp_capture_calibration", table_name="wsp_activation_captures",
                  postgresql_where=sa.text("role = 'calibration'"))
    op.create_index("uq_wsp_capture_calibration", "wsp_activation_captures", ["probe_id", "dataset_id"],
                    unique=True, postgresql_where=sa.text("role = 'calibration' AND superseded_at IS NULL"))


def downgrade() -> None:
    op.drop_index("uq_wsp_capture_calibration", table_name="wsp_activation_captures",
                  postgresql_where=sa.text("role = 'calibration' AND superseded_at IS NULL"))
    op.create_index("uq_wsp_capture_calibration", "wsp_activation_captures", ["probe_id", "dataset_id"],
                    unique=True, postgresql_where=sa.text("role = 'calibration'"))
    op.drop_column("wsp_bundles", "capture_seals")
    op.drop_column("wsp_activation_captures", "superseded_by")
    op.drop_column("wsp_activation_captures", "superseded_at")
