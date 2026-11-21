"""ENH-010 044 FR-5, R2 review fix OR2-044a-H2 (2026-11-17): deleting a bundle, a probe or a run leaves every score
call and the whole per-row head budget in place.

The budget is the ONLY bound on reading a held-out row (FR-5, P68), and every call is recorded. Both tables used to
hang off the bundle by `ON DELETE CASCADE` (and the bundle off the probe and the run), and the append-only ORM guard
never sees a database cascade: deleting a run reset a held-out pool to a fresh 64 heads and erased its audit
(review experiment E1: 60 budget rows and 2 calls → 0 and 0).

Each deletion goes through the path production uses: the ORM delete of a bundle row and of a probe row (the database
cascades below them), and the run through `DELETE /api/v1/probe-monitors/runs/{id}`. After each, the deleted rows are
really gone (the cascade ran) and the call and budget counts are unchanged, each call still naming its bundle.

Negative controls (`0xcc/reviews/044a_r2_review_fixes_2026-11-17.md`): restore `ondelete="CASCADE"` on `bundle_id` →
the first test red; on `first_call_id` → the second test red.
"""
from __future__ import annotations

import pytest
from sqlalchemy import func, select

from src.models.probe_monitor import ProbeMonitor, ProbeMonitorRun
from src.models.wsp import WspBundle, WspHeldoutRowHead, WspScoreCall
from tests.unit.test_gpu_worker_watch import _run_row


async def _count(session, model) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def _probe(session, run_id):
    probe = ProbeMonitor(run_id=run_id, layer=1, rule="mean", rule_params={}, variant="dense", val_metrics={})
    session.add(probe)
    await session.commit()
    return probe.id


async def _bundle_with_calls(session, probe_id, *, calls, rows):
    """A ready bundle, `calls` score calls on it, and `rows` budget rows per call (distinct heads)."""
    bundle = WspBundle(probe_id=probe_id, status="ready")
    session.add(bundle)
    await session.commit()
    call_ids = []
    for c in range(calls):
        call = WspScoreCall(bundle_id=bundle.id, heldout_ref_sha256="0" * 64, labels=["candidate"], status="scored")
        session.add(call)
        await session.flush()
        call_ids.append(call.id)
        for r in range(rows):
            session.add(WspHeldoutRowHead(model="m", layer=1, scope="all", row_content_key=f"row-{r:04d}",
                                          head_tensors_sha256=f"{bundle.id}-{c}".ljust(64, "0")[:64],
                                          label="candidate", first_call_id=call.id))
    await session.commit()
    return bundle.id, call_ids


@pytest.mark.asyncio
async def test_deleting_a_bundle_probe_and_run_keeps_every_call_and_the_whole_budget(client, async_session):
    run_id = await _run_row(async_session, "completed")
    first, second = await _probe(async_session, run_id), await _probe(async_session, run_id)
    one, one_calls = await _bundle_with_calls(async_session, first, calls=2, rows=30)
    two, two_calls = await _bundle_with_calls(async_session, first, calls=1, rows=20)
    three, three_calls = await _bundle_with_calls(async_session, second, calls=1, rows=10)
    calls, budget = await _count(async_session, WspScoreCall), await _count(async_session, WspHeldoutRowHead)
    assert (calls, budget) == (4, 90)

    # 1. The bundle row, through the ORM.
    await async_session.delete(await async_session.get(WspBundle, one))
    await async_session.commit()
    assert await async_session.get(WspBundle, one) is None
    assert (await _count(async_session, WspScoreCall), await _count(async_session, WspHeldoutRowHead)) == (calls, budget)

    # 2. The probe row, through the ORM: its other bundle goes by the database cascade.
    await async_session.delete(await async_session.get(ProbeMonitor, first))
    await async_session.commit()
    async_session.expire_all()
    assert (await async_session.execute(select(WspBundle).where(WspBundle.id == two))).scalar_one_or_none() is None
    assert (await _count(async_session, WspScoreCall), await _count(async_session, WspHeldoutRowHead)) == (calls, budget)

    # 3. The run, through the real route: its remaining probe and bundle go by the cascade.
    response = await client.delete(f"/api/v1/probe-monitors/runs/{run_id}")
    assert response.status_code == 204, response.text
    async_session.expire_all()
    assert (await async_session.execute(select(ProbeMonitorRun).where(ProbeMonitorRun.id == run_id))
            ).scalar_one_or_none() is None
    assert (await async_session.execute(select(WspBundle).where(WspBundle.id == three))).scalar_one_or_none() is None
    assert (await _count(async_session, WspScoreCall), await _count(async_session, WspHeldoutRowHead)) == (calls, budget)

    # The record still names what it scored, and each budget row its first call.
    named = dict((await async_session.execute(select(WspScoreCall.id, WspScoreCall.bundle_id))).all())
    assert named == {**{c: one for c in one_calls}, **{c: two for c in two_calls}, **{c: three for c in three_calls}}
    firsts = set((await async_session.execute(select(WspHeldoutRowHead.first_call_id))).scalars())
    assert firsts == set(named)


@pytest.mark.asyncio
async def test_a_score_call_deleted_at_the_database_leaves_the_budget(async_session):
    """The budget's other link (R2 control H2b survived the first test: with `bundle_id` free, no cascade reaches a call
    row, so a `first_call_id` cascade never fired there). The ORM refuses to delete a call (append-only); a cleanup in
    SQL does not ask it. Deleting the calls that first counted every budget row leaves every budget row, still naming
    the gone call. Control: restore `ondelete="CASCADE"` on `first_call_id` → red."""
    from sqlalchemy import delete

    run_id = await _run_row(async_session, "completed")
    _bundle_id, call_ids = await _bundle_with_calls(async_session, await _probe(async_session, run_id), calls=2,
                                                    rows=25)
    assert await _count(async_session, WspHeldoutRowHead) == 50
    await async_session.execute(delete(WspScoreCall).where(WspScoreCall.id.in_(call_ids)))
    await async_session.commit()
    assert await _count(async_session, WspScoreCall) == 0
    assert await _count(async_session, WspHeldoutRowHead) == 50
    firsts = set((await async_session.execute(select(WspHeldoutRowHead.first_call_id))).scalars())
    assert firsts == set(call_ids)


@pytest.mark.asyncio
async def test_deleting_a_dataset_keeps_its_sensitivity_marks(client, async_session):
    """R3 review fix (OR3-044a-L1), 2026-11-17: `wsp_sensitivity_marks` is append-only history (an operator's decision
    about their data), and it hung off `datasets.id` by `ON DELETE CASCADE`, which the ORM guard never sees. The marks
    are made and the dataset deleted through the production routes; every mark remains, still naming its dataset.
    Control: restore `ForeignKey("datasets.id", ondelete="CASCADE")` on `dataset_id` → red (0 marks of 2 left)."""
    import uuid

    from src.models.dataset import Dataset, DatasetStatus
    from src.models.wsp import WspSensitivityMark

    dataset_id = uuid.uuid4()
    async_session.add(Dataset(id=dataset_id, name="marked", source="local", status=DatasetStatus.READY,
                              extra_metadata={}))
    await async_session.commit()
    for value, reason in (("not_sensitive", "public corpus"), ("sensitive", "found PII after all")):
        made = await client.post("/api/v1/wsp/sensitivity-marks",
                                 json={"dataset_id": str(dataset_id), "value": value, "reason": reason})
        assert made.status_code == 201, made.text
    assert await _count(async_session, WspSensitivityMark) >= 2
    before = set((await async_session.execute(
        select(WspSensitivityMark.id).where(WspSensitivityMark.dataset_id == dataset_id))).scalars())
    assert len(before) == 2

    response = await client.delete(f"/api/v1/datasets/{dataset_id}")
    assert response.status_code == 204, response.text
    async_session.expire_all()
    assert await async_session.get(Dataset, dataset_id) is None, "the dataset was really deleted"
    after = set((await async_session.execute(
        select(WspSensitivityMark.id).where(WspSensitivityMark.dataset_id == dataset_id))).scalars())
    assert after == before
