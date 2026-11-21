"""ENH-010 (044 task 3.2): sensitivity marks and score calls are APPEND-ONLY.

An operator's sensitivity decision is history (P11, P69: a later mark may raise, and the record of what was decided
must survive), and a score call is the audit and the reuse count the head budget is read against. Both refuse update
and delete at the ORM (`models/wsp.py`), the `PurgeTombstoneIsAppendOnly` pattern. Driven against the real database.

Control (0xcc/reviews/044_implementation_controls_2026-11-16.md): remove the `before_update` listener → red.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.wsp import WspRowIsAppendOnly, WspSensitivityMark


@pytest.fixture
def sync_db(async_engine):
    url = async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", ""))
    engine = create_engine(url)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def _mark(db):
    from src.models.dataset import Dataset, DatasetStatus

    dataset = Dataset(id=uuid.uuid4(), name="d", source="local", status=DatasetStatus.READY, raw_path="/x",
                      extra_metadata={})
    db.add(dataset)
    db.flush()
    mark = WspSensitivityMark(dataset_id=dataset.id, value="sensitive", reason="contains personal data")
    db.add(mark)
    db.commit()
    return mark


async def test_marks_are_append_only(sync_db):
    mark = _mark(sync_db)
    assert mark.id.startswith("wsm_") and mark.decided_by == "operator"
    mark.value = "not_sensitive"
    with pytest.raises(WspRowIsAppendOnly, match="cannot be updated"):
        sync_db.commit()
    sync_db.rollback()
    sync_db.delete(sync_db.get(WspSensitivityMark, mark.id))
    with pytest.raises(WspRowIsAppendOnly, match="cannot be deleted"):
        sync_db.commit()
    sync_db.rollback()
    assert sync_db.get(WspSensitivityMark, mark.id).value == "sensitive"


def test_score_calls_carry_the_same_guard():
    """Both guards are registered on the live mappers of BOTH append-only tables (assert the wiring)."""
    from sqlalchemy import event

    from src.models import wsp

    assert set(wsp.APPEND_ONLY) == {wsp.WspSensitivityMark, wsp.WspScoreCall}
    for model in wsp.APPEND_ONLY:
        assert event.contains(model, "before_update", wsp.refuse_update), model.__name__
        assert event.contains(model, "before_delete", wsp.refuse_delete), model.__name__


# ── 5.4: the effective value, read live (FR-7; P11, P35, P41, P69, F44-D24) ───────────────────────────────────────
#
# Controls (044 controls record): read only the registered record → the live-raise test red; ignore marks on a
# miDataworks value → the raise test red; drop the route's agent refusal → `test_agent_mark_refused` red.

import itertools  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import httpx  # noqa: E402

from src.services import wsp_sensitivity as ws  # noqa: E402

ORDER = {"not_sensitive": 0, "unknown": 1, "sensitive": 2}
NO_RECORD = "no-record"


def _oracle(registered, live, mark):
    """FR-7 as a table, written out independently of the module."""
    if registered == NO_RECORD:
        return mark or "unknown"
    if live is ws.UNREACHABLE:
        return "sensitive" if mark == "sensitive" else "unknown"
    if registered is None and live is None:
        return mark or "unknown"
    # FR-7 as amended (operator, 2026-11-17, F44-D27): a null is "not stated". A live null beside a stated registered
    # value is a withdrawn claim → `unknown` (OR2-044a-M8, kept); a live statement → the most sensitive STATED value,
    # a registration-time null ignored. A mark only ever raises, to `sensitive` (P69).
    if live is None:
        value = max(("unknown", registered), key=ORDER.__getitem__)
    else:
        value = max((v for v in (registered, live) if v is not None), key=ORDER.__getitem__)
    return "sensitive" if (value != "sensitive" and mark == "sensitive") else value


def _dataset(db):
    from src.models.dataset import Dataset, DatasetStatus

    dataset = Dataset(id=uuid.uuid4(), name="d", source="local", status=DatasetStatus.READY, raw_path="/x",
                      extra_metadata={})
    db.add(dataset)
    db.commit()
    return dataset.id


def _source(db, registered, mark):
    dataset_id = _dataset(db)
    if mark is not None:
        db.add(WspSensitivityMark(dataset_id=dataset_id, value=mark, reason="r"))
        db.commit()
    record = None if registered == NO_RECORD else {"version_id": "dsv_1", "sensitivity": registered}
    return SimpleNamespace(dataset_id=dataset_id, dataset_version=record)


def test_every_flag_and_mark_combination(sync_db):
    registered_values = ("sensitive", "not_sensitive", None, NO_RECORD)
    live_values = ("sensitive", "not_sensitive", None, ws.UNREACHABLE)
    for registered, live, mark in itertools.product(registered_values, live_values, (None, "sensitive",
                                                                                     "not_sensitive")):
        view = _source(sync_db, registered, mark)
        got = ws.source_sensitivity(sync_db, view, live=lambda _v, value=live: value).value
        assert got == _oracle(registered, live, mark), (registered, live, mark)


def test_a_dataworks_sensitive_is_never_lowered_by_a_mark(sync_db):
    view = _source(sync_db, "sensitive", "not_sensitive")
    assert ws.source_sensitivity(sync_db, view, live=lambda _v: "sensitive").value == "sensitive"
    assert ws.source_sensitivity(sync_db, view, live=lambda _v: None).value == "sensitive"


def test_live_raise_wins_over_registered(sync_db):
    view = _source(sync_db, "not_sensitive", None)
    result = ws.source_sensitivity(sync_db, view, live=lambda _v: "sensitive")
    assert result.value == "sensitive" and {"source": "midataworks", "ref": "live:dsv_1",
                                            "value": "sensitive"} in result.basis


def test_a_withdrawn_claim_reads_unknown_and_no_mark_lifts_it(sync_db):
    """OR2-044a-M8 (FR-7): miDataworks stated `not_sensitive` at registration and now answers `null` (the claim was
    withdrawn): the source reads `unknown`, and an operator's `not_sensitive` mark does not lift it (the mark decides
    only where both values are null). It used to read `not_sensitive` — the fail-open direction. (The reverse order,
    registered `null` then live `not_sensitive`, is trusted since the operator's 2026-11-17 amendment: see
    `test_the_four_cases_of_the_2026_11_17_amendment`.) Control: treat a live null as `not_sensitive` → red."""
    for registered, live in (("not_sensitive", None),):
        for mark in (None, "not_sensitive"):
            view = _source(sync_db, registered, mark)
            result = ws.source_sensitivity(sync_db, view, live=lambda _v, value=live: value)
            assert result.value == "unknown", (registered, live, mark, result)
            assert "operator_mark" not in {b["source"] for b in result.basis}
    raised = _source(sync_db, "not_sensitive", "sensitive")
    assert ws.source_sensitivity(sync_db, raised, live=lambda _v: None).value == "sensitive"   # P69: a raise holds


def test_the_four_cases_of_the_2026_11_17_amendment(sync_db):
    """FR-7 as amended by the operator, 2026-11-17 (F44-D27): a `null` means "not stated"; the value is the most
    sensitive STATED value, and the live read decides whether anything is stated now. The four cases, each with no
    mark and with a `not_sensitive` mark (which never changes them):
    1. registered `null`, live `not_sensitive` → `not_sensitive` (the change: a later live statement is trusted);
    2. registered `not_sensitive`, live `null` → `unknown` (a withdrawn claim; round 2's M8, kept);
    3. registered `sensitive`, live `not_sensitive` → `sensitive` (most sensitive; never lowered);
    4. miDataworks unreachable → `unknown`, and no mark lifts it (F44-D24).
    Controls: round 2's rule (a null reads `unknown` beside a stated value) → case 1 red; a live null read as
    `not_sensitive` → case 2 red."""
    cases = ((None, "not_sensitive", "not_sensitive"), ("not_sensitive", None, "unknown"),
             ("sensitive", "not_sensitive", "sensitive"), ("not_sensitive", ws.UNREACHABLE, "unknown"),
             (None, ws.UNREACHABLE, "unknown"))
    for registered, live, expected in cases:
        for mark in (None, "not_sensitive"):
            view = _source(sync_db, registered, mark)
            result = ws.source_sensitivity(sync_db, view, live=lambda _v, value=live: value)
            assert result.value == expected, (registered, live, mark, result)
            assert "operator_mark" not in {b["source"] for b in result.basis}, (registered, live, mark)
    # P69 holds in case 1: a `sensitive` mark raises the trusted `not_sensitive`.
    raised = _source(sync_db, None, "sensitive")
    assert ws.source_sensitivity(sync_db, raised, live=lambda _v: "not_sensitive").value == "sensitive"


def test_where_dataworks_states_nothing_the_operators_mark_decides(sync_db):
    """P35 as confirmed by the operator, 2026-11-17 (F44-D28): when miDataworks has stated nothing, at registration
    or in its live read, the operator's mark is the only statement and sets the value, in either direction (the
    newest mark wins); with no mark the source reads `unknown`. Contrast: a mark never lowers a value miDataworks
    stated, never lifts a withdrawn claim, and never lifts an unreachable miDataworks.
    Control: both-null ignores the mark (reads `unknown`) → red."""
    for mark, expected in ((None, "unknown"), ("not_sensitive", "not_sensitive"), ("sensitive", "sensitive")):
        view = _source(sync_db, None, mark)
        result = ws.source_sensitivity(sync_db, view, live=lambda _v: None)
        assert result.value == expected, (mark, result)
        assert ("operator_mark" in {b["source"] for b in result.basis}) is (mark is not None), (mark, result.basis)
    newest = _source(sync_db, None, "sensitive")
    sync_db.add(WspSensitivityMark(dataset_id=newest.dataset_id, value="not_sensitive", reason="re-reviewed"))
    sync_db.commit()
    assert ws.source_sensitivity(sync_db, newest, live=lambda _v: None).value == "not_sensitive"
    # The contrasts: a stated value is never lowered, a withdrawal and an unreachable miDataworks are never lifted.
    for registered, live in (("sensitive", None), ("not_sensitive", None), (None, ws.UNREACHABLE)):
        view = _source(sync_db, registered, "not_sensitive")
        got = ws.source_sensitivity(sync_db, view, live=lambda _v, value=live: value).value
        assert got != "not_sensitive", (registered, live, got)


def test_unreachable_dataworks_reads_unknown_and_no_mark_lifts_it(sync_db):
    view = _source(sync_db, "not_sensitive", "not_sensitive")
    assert ws.source_sensitivity(sync_db, view, live=lambda _v: ws.UNREACHABLE).value == "unknown"


def test_a_mark_may_raise_not_sensitive(sync_db):
    view = _source(sync_db, "not_sensitive", "sensitive")
    result = ws.source_sensitivity(sync_db, view, live=lambda _v: "not_sensitive")
    assert result.value == "sensitive" and result.basis[-1]["source"] == "operator_mark"


def test_the_newest_mark_decides(sync_db):
    view = _source(sync_db, NO_RECORD, "sensitive")
    sync_db.add(WspSensitivityMark(dataset_id=view.dataset_id, value="not_sensitive", reason="re-reviewed"))
    sync_db.commit()
    assert ws.source_sensitivity(sync_db, view).value == "not_sensitive"


def test_the_bundle_value_is_the_most_sensitive_source(sync_db):
    calm = _source(sync_db, NO_RECORD, "not_sensitive")
    hot = _source(sync_db, NO_RECORD, "sensitive")
    assert ws.effective(sync_db, [calm, hot])["value"] == "sensitive"
    assert ws.effective(sync_db, [calm])["value"] == "not_sensitive"
    assert ws.effective(sync_db, [])["value"] == "unknown"


@pytest.mark.parametrize("answer", [
    httpx.Response(200, text="<html>proxy</html>"),
    httpx.Response(200, json={"version_id": "dsv_1"}),           # an older miDataworks: no field → cannot say
    httpx.Response(503, json={"sensitivity": "not_sensitive"}),
    httpx.Response(200, json={"sensitivity": "maybe"}),
])
def test_a_live_read_that_cannot_say_is_unreachable(monkeypatch, answer):
    from src.core.config import settings

    monkeypatch.setattr(settings, "midataworks_api_url", "http://dataworks.test", raising=False)
    transport = httpx.MockTransport(lambda request: answer)
    assert ws.read_live("dsv_1", transport=transport) is ws.UNREACHABLE


def test_a_live_read_reads_the_version_route(monkeypatch):
    from src.core.config import settings

    seen = []
    monkeypatch.setattr(settings, "midataworks_api_url", "http://dataworks.test/", raising=False)

    def answer(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"sensitivity": None})

    assert ws.read_live("dsv_9", transport=httpx.MockTransport(answer)) is None
    assert seen == ["http://dataworks.test/api/v1/versions/dsv_9"]
    monkeypatch.setattr(settings, "midataworks_api_url", "", raising=False)
    assert ws.read_live("dsv_9") is ws.UNREACHABLE


@pytest.mark.asyncio
async def test_agent_mark_refused(client, sync_db):
    from src.services.probe_dataset_lifecycle import AGENT_HEADER

    dataset_id = _dataset(sync_db)
    body = {"dataset_id": str(dataset_id), "value": "not_sensitive", "reason": "public corpus"}
    refused = await client.post("/api/v1/wsp/sensitivity-marks", json=body, headers={AGENT_HEADER: "mcp"})
    assert refused.status_code == 403 and refused.json()["detail"]["code"] == "operator_or_service_only"
    assert sync_db.query(WspSensitivityMark).filter(WspSensitivityMark.dataset_id == dataset_id).count() == 0

    made = await client.post("/api/v1/wsp/sensitivity-marks", json=body)
    assert made.status_code == 201, made.text
    assert made.json()["effective"]["value"] == "not_sensitive"
    read = await client.get("/api/v1/wsp/sensitivity", params={"dataset_id": str(dataset_id)})
    assert read.status_code == 200 and read.json()["value"] == "not_sensitive"
