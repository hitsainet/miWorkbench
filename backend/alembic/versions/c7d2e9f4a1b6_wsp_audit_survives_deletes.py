"""ENH-010 044 (R2 review): WSP score calls and the head budget survive deletes; one calibration capture per view

Revision ID: c7d2e9f4a1b6
Revises: 0e4a1c6b9d27
Create Date: 2026-11-17

044-A independent review, round 2 (`0xcc/reviews/044a_r2_review_fixes_2026-11-17.md`):

* OR2-044a-H2. `wsp_score_calls.bundle_id` and `wsp_heldout_row_heads.first_call_id` lose their foreign keys. Both
  were `ON DELETE CASCADE`, and the bundle cascades from the probe and the run, so deleting a run erased every score
  call (FR-5: "every call is recorded") and every row of the per-row head budget (FR-5: "the only bound"): a new run on
  the same views got a fresh 64 heads. The columns keep the minted id they were written with. `SET NULL` would UPDATE
  an append-only row and erase the bundle FR-5 says the record holds; `RESTRICT` would make every scored run
  undeletable, since score calls are never deleted.
* OR2-044a-L6. A unique partial index `uq_wsp_capture_calibration` on `wsp_activation_captures (probe_id, dataset_id)
  WHERE role = 'calibration'`: the existing unique key includes `source_evaluation_id`, which is NULL for a calibration
  capture, and NULLs never collide, so two concurrent capture requests made two calibration rows.
* OR2-044a-M3. `wsp_bundles.scorer_parent_bar` (JSONB, nullable): the parent's bar as the scorer cuts it, shown beside
  the bar as served (FR-1). NULL for a bundle built before this revision ("not recorded").

Additive apart from the two dropped constraints. The 044 tables are new in this push (044-A), so no live row exists
for the index to collide with. The downgrade restores the constraints exactly; it fails, by design, while any score
call names a bundle that is gone or any budget row names a call that is gone (the cascade would have deleted them).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c7d2e9f4a1b6"
down_revision: Union[str, None] = "0e4a1c6b9d27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("wsp_heldout_row_heads_first_call_id_fkey", "wsp_heldout_row_heads", type_="foreignkey")
    op.drop_constraint("wsp_score_calls_bundle_id_fkey", "wsp_score_calls", type_="foreignkey")
    op.create_index("uq_wsp_capture_calibration", "wsp_activation_captures", ["probe_id", "dataset_id"],
                    unique=True, postgresql_where=sa.text("role = 'calibration'"))
    op.add_column("wsp_bundles", sa.Column("scorer_parent_bar", postgresql.JSONB(astext_type=sa.Text()),
                                           nullable=True))


def downgrade() -> None:
    op.drop_column("wsp_bundles", "scorer_parent_bar")
    op.drop_index("uq_wsp_capture_calibration", table_name="wsp_activation_captures",
                  postgresql_where=sa.text("role = 'calibration'"))
    op.create_foreign_key("wsp_score_calls_bundle_id_fkey", "wsp_score_calls", "wsp_bundles", ["bundle_id"], ["id"],
                          ondelete="CASCADE")
    op.create_foreign_key("wsp_heldout_row_heads_first_call_id_fkey", "wsp_heldout_row_heads", "wsp_score_calls",
                          ["first_call_id"], ["id"], ondelete="CASCADE")
