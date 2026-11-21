"""steering approvals switch to the caller-executed mechanism (045 FR-13a; CHAIN_PLAN A16 D5, A29)

Revision ID: e045a1b2c3d4
Revises: c25e7a41d9b3
Create Date: 2026-11-16

Before 045 a steering approval was BACKEND-executed: approving it made the backend submit the stored request itself
and store the task it started in `steering_task_id`, so `approved` meant "approved and run". Since 045 steering has
the one caller-executed mechanism (approve marks the row; the agent's re-call is consumed by the steering route), and
an `approved` steering row would read as "approved and unspent": the re-call would run it a SECOND time. So this
data migration keeps history true (FTDD §16.5):

* a steering row `approved` WITH a `steering_task_id` was already run: it becomes `consumed` (`consumed_at` = its
  approval time), its task id kept for the record;
* a steering row still `pending` can no longer be approved into a self-call: it is `expired` with the reason below,
  `decided_by` "system";
* an `approved` steering row WITHOUT a task id cannot be told apart from an unspent approval; the migration refuses
  to run and names them, so the operator decides each one (never guessed).

Every other row, and every steering row in any other state, is untouched. DATA ONLY; the downgrade does nothing
(a consumed or expired request is never re-opened).
"""

from alembic import op
import sqlalchemy as sa

revision = "e045a1b2c3d4"
down_revision = "c25e7a41d9b3"
branch_labels = None
depends_on = None

#: ⚠ FROZEN: the three steering tools, written out (not imported) so the migration means the same thing forever.
STEERING_TOOLS = ("steer_compare", "steer_sweep", "steer_combined")
REASON = "expired: steering approvals are re-called since 045 (A16 D5); ask again"


class UnrunApprovedSteering(RuntimeError):
    """An `approved` steering row with no task id: the switch-over cannot tell whether it ran."""


def refusal_message(orphans) -> str:
    """R3 review fix (OR3-045a0-8), 2026-11-16: the refusal names every row AND the fix, because the API container runs
    this migration at start and exits on failure: without the fix in the log the pod only crash-loops."""
    listed = "; ".join(f"{rid} ({tool}, approved {at})" for rid, tool, at in orphans)
    ids = ", ".join(f"'{rid}'" for rid, _tool, _at in orphans)
    return (
        f"045 steering switch-over REFUSED; nothing was changed. {len(orphans)} approved steering request(s) have no "
        f"steering_task_id, so the switch-over cannot tell whether they ran: {listed}. The API will not start until "
        "each is decided (045 FTASKS push gate, pre-deploy step; OPERATOR item 4). Fix: for a request that did NOT "
        "run, UPDATE agent_approval_requests SET status = 'expired', reason = 'decided by the operator before the "
        f"045 switch-over', decided_by = 'operator' WHERE id IN ({ids}); for one that DID run, SET status = "
        "'consumed', consumed_at = now() instead. Then restart the API pod."
    )


def upgrade() -> None:
    bind = op.get_bind()
    tools = sa.bindparam("tools", expanding=True)
    orphans = bind.execute(
        sa.text(
            "SELECT id, tool_name, resolved_at FROM agent_approval_requests WHERE tool_name IN :tools "
            "AND status = 'approved' AND (steering_task_id IS NULL OR steering_task_id = '') ORDER BY id"
        ).bindparams(tools),
        {"tools": list(STEERING_TOOLS)},
    ).all()
    if orphans:
        raise UnrunApprovedSteering(refusal_message(orphans))
    bind.execute(
        sa.text(
            """
            UPDATE agent_approval_requests
               SET status = 'consumed',
                   consumed_at = COALESCE(consumed_at, resolved_at, now())
             WHERE tool_name IN :tools
               AND status = 'approved'
               AND steering_task_id IS NOT NULL AND steering_task_id <> ''
            """
        ).bindparams(sa.bindparam("tools", expanding=True)),
        {"tools": list(STEERING_TOOLS)},
    )
    bind.execute(
        sa.text(
            """
            UPDATE agent_approval_requests
               SET status = 'expired',
                   reason = :reason,
                   decided_by = 'system',
                   resolved_at = COALESCE(resolved_at, now())
             WHERE tool_name IN :tools
               AND status = 'pending'
            """
        ).bindparams(sa.bindparam("tools", expanding=True)),
        {"reason": REASON, "tools": list(STEERING_TOOLS)},
    )


def downgrade() -> None:
    pass
