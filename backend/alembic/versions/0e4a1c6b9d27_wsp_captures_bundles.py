"""ENH-010 044 <044a>: WSP captures, bundles, baselines, score calls, the per-row head budget, sensitivity marks

Revision ID: 0e4a1c6b9d27
Revises: a5e1d0c7b2f9
Create Date: 2026-11-16

044 FTDD §4.2 / §4.3 (FR-1, FR-2, FR-4, FR-5, FR-7, FR-8). Six new tables and one nullable column
(`probe_monitors.derived_from`, NULL for every trained probe). Purely additive: nothing existing is rewritten, and the
downgrade drops exactly what this adds. `wsp_sensitivity_marks` and `wsp_score_calls` are append-only by an ORM guard
(`models/wsp.py`), the pattern of the purge tombstones.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0e4a1c6b9d27"
down_revision: Union[str, None] = "a5e1d0c7b2f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('wsp_sensitivity_marks',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('dataset_id', sa.UUID(), nullable=False),
    sa.Column('value', sa.String(length=16), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('decided_by', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_wsp_sensitivity_marks_dataset_id', 'wsp_sensitivity_marks', ['dataset_id'], unique=False)
    op.create_table('wsp_bundles',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('probe_id', sa.String(length=36), nullable=False),
    sa.Column('bundle_kind', sa.String(length=16), nullable=False),
    sa.Column('manifest', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('bundle_sha256', sa.String(length=64), nullable=True),
    sa.Column('sensitivity', sa.String(length=16), nullable=False),
    sa.Column('sensitivity_basis', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('capture_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('dir_path', sa.String(length=1000), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['probe_id'], ['probe_monitors.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('bundle_sha256')
    )
    op.create_index('ix_wsp_bundles_probe_id', 'wsp_bundles', ['probe_id'], unique=False)
    op.create_table('wsp_activation_captures',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('probe_id', sa.String(length=36), nullable=False),
    sa.Column('dataset_id', sa.String(length=36), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('source_evaluation_id', sa.String(length=36), nullable=True),
    sa.Column('layer', sa.Integer(), nullable=False),
    sa.Column('scope', sa.String(length=16), nullable=False),
    sa.Column('rows', sa.Integer(), nullable=True),
    sa.Column('tokens', sa.BigInteger(), nullable=True),
    sa.Column('d_model', sa.Integer(), nullable=True),
    sa.Column('dtype', sa.String(length=8), nullable=True),
    sa.Column('dir_path', sa.String(length=1000), nullable=True),
    sa.Column('row_ids_sha256', sa.String(length=64), nullable=True),
    sa.Column('parity', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('bytes', sa.BigInteger(), nullable=True),
    sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['probe_monitor_datasets.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['probe_id'], ['probe_monitors.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['source_evaluation_id'], ['probe_monitor_evaluations.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('probe_id', 'dataset_id', 'source_evaluation_id', name='uq_wsp_capture_source')
    )
    op.create_index('ix_wsp_activation_captures_probe_id', 'wsp_activation_captures', ['probe_id'], unique=False)
    op.create_table('wsp_baselines',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('bundle_id', sa.String(length=36), nullable=False),
    sa.Column('kind', sa.String(length=32), nullable=False),
    sa.Column('weights_path', sa.String(length=1000), nullable=True),
    sa.Column('bar', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('transform', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('fit', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['bundle_id'], ['wsp_bundles.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('bundle_id', 'kind', name='uq_wsp_baseline_kind')
    )
    op.create_table('wsp_score_calls',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('bundle_id', sa.String(length=36), nullable=False),
    sa.Column('heldout_ref_sha256', sa.String(length=64), nullable=False),
    sa.Column('labels', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('miforge_run_id', sa.String(length=64), nullable=True),
    sa.Column('patch_sha256', sa.String(length=64), nullable=True),
    sa.Column('candidate_tensors_sha256', sa.String(length=64), nullable=True),
    sa.Column('heads', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('caller', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('request_sha256', sa.String(length=64), nullable=True),
    sa.Column('response_sha256', sa.String(length=64), nullable=True),
    sa.Column('candidate_scores_sha256', sa.String(length=64), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['bundle_id'], ['wsp_bundles.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_wsp_score_calls_bundle_id', 'wsp_score_calls', ['bundle_id'], unique=False)
    op.create_index('ix_wsp_score_calls_run_candidate', 'wsp_score_calls', ['miforge_run_id', 'candidate_tensors_sha256'], unique=False)
    op.create_table('wsp_heldout_row_heads',
    sa.Column('model', sa.String(length=255), nullable=False),
    sa.Column('layer', sa.Integer(), nullable=False),
    sa.Column('scope', sa.String(length=16), nullable=False),
    sa.Column('row_content_key', sa.String(length=64), nullable=False),
    sa.Column('head_tensors_sha256', sa.String(length=64), nullable=False),
    sa.Column('label', sa.String(length=32), nullable=False),
    sa.Column('first_call_id', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['first_call_id'], ['wsp_score_calls.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('model', 'layer', 'scope', 'row_content_key', 'head_tensors_sha256')
    )
    op.add_column('probe_monitors', sa.Column('derived_from', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("probe_monitors", "derived_from")
    op.drop_table("wsp_heldout_row_heads")
    op.drop_index("ix_wsp_score_calls_run_candidate", table_name="wsp_score_calls")
    op.drop_index("ix_wsp_score_calls_bundle_id", table_name="wsp_score_calls")
    op.drop_table("wsp_score_calls")
    op.drop_table("wsp_baselines")
    op.drop_index("ix_wsp_activation_captures_probe_id", table_name="wsp_activation_captures")
    op.drop_table("wsp_activation_captures")
    op.drop_index("ix_wsp_bundles_probe_id", table_name="wsp_bundles")
    op.drop_table("wsp_bundles")
    op.drop_index("ix_wsp_sensitivity_marks_dataset_id", table_name="wsp_sensitivity_marks")
    op.drop_table("wsp_sensitivity_marks")
