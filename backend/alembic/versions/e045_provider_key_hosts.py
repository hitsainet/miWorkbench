"""the stored OpenAI key is bound to the base URLs it was entered for (045 FPRD FR-19; CHAIN_PLAN A39, A43)

Revision ID: e045b7c1d2f3
Revises: e045a1b2c3d4
Create Date: 2026-11-16

Before 045 the stored `openai_api_key` was sent to whatever endpoint a request named. Since 045 a key goes only to
the base URLs saved beside it (`services/provider_keys.py`; the setting `openai_api_key_base_urls`, a JSON list).
This data migration binds the key that already exists to the two hosts the operator entered it for (R3A-2's fix):

* `https://api.openai.com/v1`, the provider default; and
* the stored `openai_compatible_endpoint`, when it is set (plaintext; an encrypted copy cannot be read here and is
  left out, which binds the key more narrowly, never more widely).

Nothing is written when there is no stored key, or when a binding was already saved. DATA ONLY; the downgrade
removes the binding row (a key with no saved list is bound to the same default at read time).
"""

import json

import sqlalchemy as sa
from alembic import op

revision = "e045b7c1d2f3"
down_revision = "e045a1b2c3d4"
branch_labels = None
depends_on = None

#: ⚠ FROZEN, written out (not imported) so the migration means the same thing forever.
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
KEY = "openai_api_key"
BASE_URLS = "openai_api_key_base_urls"
COMPATIBLE = "openai_compatible_endpoint"


def _value(conn, key):
    row = conn.execute(sa.text("SELECT value, is_sensitive FROM app_settings WHERE key = :k"), {"k": key}).first()
    return row


def upgrade() -> None:
    conn = op.get_bind()
    key = _value(conn, KEY)
    if key is None or not (key[0] or "").strip():
        return
    if _value(conn, BASE_URLS) is not None:
        return
    urls = [DEFAULT_OPENAI_BASE_URL]
    compatible = _value(conn, COMPATIBLE)
    if compatible is not None and not compatible[1]:
        value = (compatible[0] or "").strip()
        if value.lower().startswith(("http://", "https://")) and value not in urls:
            urls.append(value)
    conn.execute(
        sa.text(
            "INSERT INTO app_settings (id, key, value, is_sensitive, category) "
            # R1 review fix (OR1-045a0-8), 2026-11-16: the settings page's category, so the page and the migration
            # write one row shape (it was 'credentials'; not yet deployed).
            "VALUES (gen_random_uuid(), :k, :v, false, 'api_keys')"
        ),
        {"k": BASE_URLS, "v": json.dumps(urls)},
    )


def downgrade() -> None:
    # The pre-045 code never reads this row. Removing it loses only a list the operator edited after the upgrade.
    op.get_bind().execute(sa.text("DELETE FROM app_settings WHERE key = :k"), {"k": BASE_URLS})
