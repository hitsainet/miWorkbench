"""ENH-011 X-1: flag artefacts recorded at a composite model's vision-stack indices

Revision ID: c7d2e4f9a1b3
Revises: 5e3c9a1d7b42
Create Date: 2026-11-26

One nullable JSONB column, `layer_meaning_changed`, on `external_saes`, `probe_monitors` and `circuits`
(services/layer_meaning.py). Before this release, layer discovery took the FIRST layer stack in the module tree;
on a composite model whose vision tower is registered before its language model (Gemma 3 and 4, LLaVA,
Qwen2.5-VL) that was the vision encoder. An artefact recorded on such a model names indices into that stack, and
after the fix the same index means a text decoder layer. Operator decision Q1 (2026-11-15): mark each such
artefact and refuse it for serving or arming until it is re-validated; delete nothing. OQ-3 (2026-11-26, answer
B): build the flag even though the live inventory found nothing to mark.

THE BACKFILL IS DETERMINISTIC, NOT A GUESS. A row is flagged when its model's `architecture_config` carries
`text_tower` (written by `model_loader.extract_architecture_config` only for a composite config) and its
`model_type` is not one whose old and new answers agree. The paths come from the frozen table below, taken from
`docs/schemas/layer-resolution-cases.json` v1 on transformers 5.20.2. A composite family not in the table is
flagged with `old_layers_path` null: the old resolver's choice on it is not known, and Q1 refuses until
re-validated rather than assuming it was right. `gemma4_unified` has one stack (patches are embedded into the text
decoder), so the old answer was already right and nothing on it is flagged.

Probes are reached through their run's model; SAEs and circuits carry `model_id`. Cluster profiles have no layer
of their own and are refused through their SAE.

On the live database of 2026-11-26 this flags nothing (three models, none composite). The downgrade drops exactly
the three columns, and with them any flag.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "c7d2e4f9a1b3"
down_revision: Union[str, None] = "5e3c9a1d7b42"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ("external_saes", "probe_monitors", "circuits")

#: model_type -> (the stack the old first-match resolver took, the text decoder's stack). Frozen: a migration is
#: history, so it never reads the live cases file.
OLD_AND_NEW = {
    "gemma4": ("model.vision_tower.encoder.layers", "model.language_model.layers"),
    "gemma3": ("model.vision_tower.encoder.layers", "model.language_model.layers"),
    "qwen2_5_vl": ("model.visual.blocks", "model.language_model.layers"),
    "llava": ("model.vision_tower.encoder.layers", "model.language_model.layers"),
}

#: Composite families whose old and new answers agree: never flagged.
UNCHANGED = ("gemma4_unified",)

REASON = (
    "recorded before ENH-011 X-1, when layer discovery took the first layer stack; on this composite model that "
    "was not the text decoder's, so this artefact's layer indices name the old stack"
)

#: Each table, and the join that names its model.
_MODEL_OF = {
    "external_saes": ("models m", "t.model_id = m.id"),
    "circuits": ("models m", "t.model_id = m.id"),
    "probe_monitors": ("probe_monitor_runs r, models m", "t.run_id = r.id AND r.model_id = m.id"),
}


def _paths_sql() -> tuple:
    """SQL CASE expressions for the old and new paths, from OLD_AND_NEW."""
    old_cases = " ".join(f"WHEN '{k}' THEN '{v[0]}'" for k, v in OLD_AND_NEW.items())
    new_cases = " ".join(f"WHEN '{k}' THEN '{v[1]}'" for k, v in OLD_AND_NEW.items())
    model_type = "m.architecture_config->>'model_type'"
    return (f"(CASE {model_type} {old_cases} ELSE NULL END)",
            f"(CASE {model_type} {new_cases} ELSE NULL END)")


def flag_rows(bind) -> dict:
    """Flag every row recorded for a composite model whose old and new answers differ. Returns counts per table."""
    old_sql, new_sql = _paths_sql()
    unchanged = ", ".join(f"'{t}'" for t in UNCHANGED)
    counts = {}
    for table in TABLES:
        from_clause, join = _MODEL_OF[table]
        result = bind.execute(sa.text(f"""
            UPDATE {table} AS t
               SET layer_meaning_changed = jsonb_build_object(
                     'old_layers_path', {old_sql},
                     'new_layers_path', {new_sql},
                     'model_type', m.architecture_config->>'model_type',
                     'flagged_at', to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
                     'flagged_by', 'migration {revision}',
                     'reason', CAST(:reason AS text))
              FROM {from_clause}
             WHERE {join}
               AND m.architecture_config IS NOT NULL
               AND jsonb_typeof(m.architecture_config) = 'object'
               AND m.architecture_config ? 'text_tower'
               AND COALESCE(m.architecture_config->>'model_type', '') NOT IN ({unchanged})
               AND t.layer_meaning_changed IS NULL
        """), {"reason": REASON})
        counts[table] = result.rowcount
    return counts


def upgrade() -> None:
    for table in TABLES:
        op.add_column(table, sa.Column("layer_meaning_changed", postgresql.JSONB(astext_type=sa.Text()),
                                       nullable=True))
    flag_rows(op.get_bind())


def downgrade() -> None:
    for table in reversed(TABLES):
        op.drop_column(table, "layer_meaning_changed")
