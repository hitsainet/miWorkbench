"""045 FR-13a (A16 D5, A29; FTDD §16.5): the steering switch-over keeps history true.

`e045a1b2c3d4` (`e045_steering_switchover.py`), on a scratch database named after this run's test database:
* an `approved` steering row WITH a task id (the backend already ran it) → `consumed`, its task id and decision kept;
* a `pending` steering row → `expired`, reason "… re-called since 045 …", `decided_by` "system";
* every other row (other tools in any state, steering rows already decided) is byte-for-byte unchanged;
* an `approved` steering row WITHOUT a task id → the migration refuses, naming it, and changes nothing;
* the downgrade re-opens nothing; the revision lies on the one line to head.

Controls (0xcc/reviews/045_A0_implementation_controls_2026-11-16.md): expire approved rows too → red; drop the
orphan refusal → red.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import text

from tests.unit.test_e008_drain_migration import BACKEND, TABLE, _create_table, _load, _run, _scratch_engine

REASON = "expired: steering approvals are re-called since 045 (A16 D5); ask again"

#: (tool, status, steering_task_id, expected status or None for "unchanged")
ROWS = [
    ("steer_compare", "approved", "task-1", "consumed"),
    ("steer_sweep", "approved", "task-2", "consumed"),
    ("steer_combined", "pending", None, "expired"),
    ("steer_compare", "pending", None, "expired"),
    # already-decided steering rows are history
    ("steer_compare", "denied", None, None),
    ("steer_sweep", "expired", None, None),
    ("steer_combined", "consumed", "task-3", None),
    # every other tool, in the states the switch-over targets, untouched
    ("start_probe_training_run", "pending", None, None),
    ("start_probe_training_run", "approved", None, None),
    ("code_rule_approve", "approved", None, None),
]


def _module():
    return _load("e045_steering_switchover.py", "e045a1b2c3d4", "c25e7a41d9b3")


def _insert(conn, rows):
    ids = {}
    for tool, status, task, _ in rows:
        rid = str(uuid.uuid4())
        ids[rid] = (tool, status, task)
        conn.execute(text(
            f'INSERT INTO "{TABLE}" (id, tool_name, payload, status, reason, steering_task_id, created_at, '
            "resolved_at) VALUES (:id, :t, CAST(:p AS JSONB), :s, :r, :task, now() - interval '1 hour', "
            "CASE WHEN :s = 'pending' THEN NULL ELSE now() - interval '30 minutes' END)"),
            {"id": rid, "t": tool, "p": json.dumps({"prompt": "x"}), "s": status, "r": "before", "task": task})
    return ids


def _rows(conn):
    return {r.id: dict(r._mapping) for r in conn.execute(text(f'SELECT * FROM "{TABLE}"'))}


def test_the_switchover_keeps_history():
    module = _module()
    engine = _scratch_engine("e045_migrations")
    try:
        with engine.begin() as conn:
            _create_table(conn)
            ids = _insert(conn, ROWS)
            before = _rows(conn)

            _run(conn, module.upgrade)

            after = _rows(conn)
            for rid, (tool, status, task) in ids.items():
                want = next(w for t, s, k, w in ROWS if (t, s, k) == (tool, status, task))
                row = after[rid]
                if want is None:
                    assert row == before[rid], f"{tool}/{status} was changed by the switch-over"
                elif want == "consumed":
                    assert row["status"] == "consumed", (tool, status)
                    assert row["steering_task_id"] == task, "the task id is kept for the record"
                    assert row["consumed_at"] == before[rid]["resolved_at"]
                    assert row["resolved_at"] == before[rid]["resolved_at"]
                else:
                    assert row["status"] == "expired" and row["reason"] == REASON, (tool, status)
                    assert row["decided_by"] == "system" and row["resolved_at"] is not None

            assert len(after) == len(ROWS)
            _run(conn, module.downgrade)
            assert _rows(conn) == after, "the downgrade re-opens nothing"
            conn.execute(text(f'DROP TABLE "{TABLE}" CASCADE'))
    finally:
        engine.dispose()


def test_an_approved_steering_row_without_a_task_id_refuses_the_switchover():
    module = _module()
    engine = _scratch_engine("e045_migrations")
    try:
        with engine.begin() as conn:
            _create_table(conn)
            ids = _insert(conn, [("steer_compare", "approved", None, None), ("steer_sweep", "pending", None, None)])
            before = _rows(conn)
        module = _module()
        with pytest.raises(module.UnrunApprovedSteering) as refused:
            with engine.begin() as conn:
                _run(conn, module.upgrade)
        orphan = next(rid for rid, (_, status, _t) in ids.items() if status == "approved")
        assert orphan in str(refused.value)
        # R3 review fix (OR3-045a0-8), 2026-11-16: the log line the crash-looping API pod shows names the fix too.
        message = str(refused.value)
        assert "nothing was changed" in message and "steer_compare" in message
        assert f"WHERE id IN ('{orphan}')" in message and "status = 'expired'" in message
        assert "status = 'consumed'" in message and "restart the API pod" in message
        with engine.begin() as conn:
            assert _rows(conn) == before, "a refused switch-over changes nothing"
            conn.execute(text(f'DROP TABLE "{TABLE}" CASCADE'))
    finally:
        engine.dispose()


def test_the_switchover_names_exactly_the_steering_tools():
    from src.api.v1.endpoints import mcp_approvals

    assert set(_module().STEERING_TOOLS) == set(mcp_approvals.STEERING_TOOLS)


def test_the_revision_lies_on_the_one_line_to_head():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "alembic"))
    script = ScriptDirectory.from_config(config)
    (head,) = script.get_heads()  # one head; later revisions (G5's a5e1d0c7b2f9) chain on top
    line = [rev.revision for rev in script.walk_revisions("base", head)]
    assert "e045a1b2c3d4" in line and "e045b7c1d2f3" in line, line  # e045_provider_key_hosts follows it
    assert script.get_revision("e045b7c1d2f3").down_revision == "e045a1b2c3d4"
