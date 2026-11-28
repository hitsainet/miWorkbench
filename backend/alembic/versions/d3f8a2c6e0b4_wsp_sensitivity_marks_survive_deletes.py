"""ENH-010 044 (R3 review): a WSP sensitivity mark survives the deletion of its dataset

Revision ID: d3f8a2c6e0b4
Revises: c7d2e9f4a1b6
Create Date: 2026-11-17

044-A independent review, round 3 (`0xcc/reviews/044a_r3_review_fixes_2026-11-17.md`), OR3-044a-L1:
`wsp_sensitivity_marks.dataset_id` loses its foreign key to `datasets.id`. It was `ON DELETE CASCADE`, so deleting a
dataset erased every sensitivity decision the operator had made about it. The table is append-only ("an operator's
decision is history"), and the ORM's append-only guard never sees a database cascade: the same defect round 2 fixed for
`wsp_score_calls` (`c7d2e9f4a1b6`). The column keeps the dataset id it was written with. A re-downloaded dataset has a
new id, so it reads `unknown` (fail-closed), as before.

The downgrade restores the constraint exactly. It fails, by design, while any mark names a dataset that is gone (the
cascade would have deleted those marks).
"""
from typing import Sequence, Union

from alembic import op

revision: str = "d3f8a2c6e0b4"
down_revision: Union[str, None] = "c7d2e9f4a1b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("wsp_sensitivity_marks_dataset_id_fkey", "wsp_sensitivity_marks", type_="foreignkey")


def downgrade() -> None:
    op.create_foreign_key("wsp_sensitivity_marks_dataset_id_fkey", "wsp_sensitivity_marks", "datasets",
                          ["dataset_id"], ["id"], ondelete="CASCADE")
