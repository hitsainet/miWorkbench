"""ENH-010 FU-7a C1 and C2 at the route: a bundle names its held-out sets and supplements, and a held-out set miForge
has already scored is refused `heldout_reused` (2026-11-25; operator decisions C1, C2).

* `ready_captures(view_ids=…)` takes exactly the named held-out sets plus the calibration capture, never a supplement;
* `POST /wsp/bundles` hands the build exactly what was named, and miForge's uses as it read them (payload AND count);
* every malformed request (unknown view, a view named twice, a view in the wrong list, a view with no ready capture)
  is refused before anything is queued or written;
* miForge's held-out uses are read through miForge's API by the REAL reader (`wsp_heldout_uses.read_heldout_uses`,
  on a recording transport), and a set they count is refused, by set name or by row-id digest; the calibration
  capture is exempt; another owner's use, or a zero count, is not a use;
* miForge absent, unreachable, failing or answering another shape is a 503, never read as "no uses".

Negative controls (`0xcc/reviews/fu7a_c1c2_2026-11-25.md`): drop the view filter in `ready_captures`, drop the
route's `refuse_used` call, drop `heldout_view_ids` from the task payload, or treat an unreadable answer as empty →
red.
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from tests.support.wsp_world import FakeForgeUses, add_view, seed_probe, sync_db, use_row  # noqa: F401 (fixture)

HEX = "a" * 64


def _capture(db, world, view, role, *, row_ids=None, **fields):
    from src.models.wsp import WspActivationCapture
    from src.services.wsp_capture import calibration_arrays_sha256

    parity = {"passed": True}
    if role == "calibration":
        parity.update(threshold_served=None, calibration_arrays_sha256=calibration_arrays_sha256(None, None, None))
    row = WspActivationCapture(probe_id=world.probe_id, dataset_id=view, role=role, layer=1, scope="all",
                               status="ready", row_ids_sha256=row_ids or (view[-1] * 64)[:64], rows=40,
                               parity=parity, **fields)
    db.add(row)
    db.commit()
    return row


@pytest.fixture
def named(sync_db):
    """A probe with two ready eval sets, a ready calibration capture and a ready supplement."""
    world = seed_probe(sync_db)
    a, b = add_view(sync_db, world, role="eval"), add_view(sync_db, world, role="eval")
    cal, supp = add_view(sync_db, world, role="calibration"), add_view(sync_db, world, role="supplement")
    captures = {"a": _capture(sync_db, world, a, "eval", row_ids="1" * 64),
                "b": _capture(sync_db, world, b, "eval", row_ids="2" * 64),
                "cal": _capture(sync_db, world, cal, "calibration", row_ids="3" * 64),
                "supp": _capture(sync_db, world, supp, "supplement", row_ids="4" * 64)}
    return SimpleNamespace(world=world, views={"a": a, "b": b, "cal": cal, "supp": supp}, captures=captures)


def _probe(db, world):
    from src.models.probe_monitor import ProbeMonitor

    return db.query(ProbeMonitor).filter(ProbeMonitor.id == world.probe_id).one()


# ── the filter (DB level) ──────────────────────────────────────────────────────


def test_ready_captures_takes_only_the_named_sets_and_the_calibration_capture(sync_db, named):
    from src.services.wsp_bundle import ready_captures, ready_supplements

    probe = _probe(sync_db, named.world)
    every = {c.dataset_id for c in ready_captures(sync_db, probe)}
    assert every == {named.views["a"], named.views["b"], named.views["cal"]}, "a supplement is never held out"
    chosen = {c.dataset_id for c in ready_captures(sync_db, probe, [named.views["a"]])}
    assert chosen == {named.views["a"], named.views["cal"]}
    assert [c.id for c in ready_supplements(sync_db, probe, [named.views["supp"]])] == [named.captures["supp"].id]


# ── the route ─────────────────────────────────────────────────────────────────


@pytest.fixture
def queue(monkeypatch):
    from src.workers import wsp_tasks

    sent = []
    monkeypatch.setattr(wsp_tasks.build_wsp_bundle, "delay",
                        lambda **kw: sent.append(kw) or SimpleNamespace(id=f"t-{len(sent)}"))
    return sent


def _bundles(db, world):
    from src.models.wsp import WspBundle

    db.expire_all()
    return db.query(WspBundle).filter(WspBundle.probe_id == world.probe_id).count()


async def _post(client, named, **body):
    return await client.post("/api/v1/wsp/bundles", json={"probe_id": named.world.probe_id, **body})


@pytest.mark.asyncio
async def test_the_build_gets_exactly_the_named_sets_and_miforges_uses(client, sync_db, named, queue, monkeypatch):
    from src.services.wsp_bundle import heldout_set_name

    # A use of ANOTHER set (b) travels with the task, for the build's row-key check.
    used = use_row(heldout_set_name(named.captures["b"]), "2" * 64)
    forge = FakeForgeUses([used, use_row("x", "9" * 64, owner="someone-else")]).install(monkeypatch)
    response = await _post(client, named, heldout_view_ids=[named.views["a"]],
                           supplement_view_ids=[named.views["supp"]])
    assert response.status_code == 202, response.text
    assert queue == [{"bundle_id": response.json()["bundle_id"], "heldout_view_ids": [named.views["a"]],
                      "supplement_view_ids": [named.views["supp"]],
                      "used_heldout": [{"set_name": used["set_name"], "row_ids_sha256": "2" * 64,
                                        "heldout_uses": 18}]}]
    assert [str(r.url) for r in forge.requests] == [f"{FakeForgeUses.BASE}/api/v1/wsp/heldout-uses"]
    assert [r.method for r in forge.requests] == ["GET"]


REFUSED = {
    "unknown view": (lambda n: {"heldout_view_ids": ["pmd_nosuchview0"]}, 404, "view_not_found"),
    "named twice": (lambda n: {"heldout_view_ids": [n.views["a"], n.views["a"]]}, 422, "bundle_request_invalid"),
    "in both lists": (lambda n: {"heldout_view_ids": [n.views["a"]], "supplement_view_ids": [n.views["a"]]}, 422,
                      "bundle_request_invalid"),
    "a supplement as held-out": (lambda n: {"heldout_view_ids": [n.views["supp"]]}, 422, "bundle_request_invalid"),
    "a held-out set as supplement": (lambda n: {"supplement_view_ids": [n.views["b"]]}, 422,
                                     "bundle_request_invalid"),
    "an empty list": (lambda n: {"heldout_view_ids": []}, 422, None),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(REFUSED))
async def test_a_malformed_request_is_refused_before_anything_is_queued(client, sync_db, named, queue, monkeypatch,
                                                                       case):
    body, status, code = REFUSED[case]
    FakeForgeUses().install(monkeypatch)
    response = await _post(client, named, **body(named))
    assert response.status_code == status, response.text
    if code is not None:
        assert response.json()["detail"]["code"] == code
    assert queue == [] and _bundles(sync_db, named.world) == 0


@pytest.mark.asyncio
async def test_a_named_view_without_a_ready_capture_is_refused(client, sync_db, named, queue, monkeypatch):
    FakeForgeUses().install(monkeypatch)
    fresh = add_view(sync_db, named.world, role="eval")
    response = await _post(client, named, heldout_view_ids=[named.views["a"], fresh])
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "capture_not_ready" and detail["views"] == [fresh]
    assert queue == [] and _bundles(sync_db, named.world) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["set_name", "row_ids_sha256"])
async def test_a_set_miforge_has_scored_is_refused_heldout_reused(client, sync_db, named, queue, monkeypatch, key):
    from src.services.wsp_bundle import heldout_set_name

    a = named.captures["a"]
    row = use_row(heldout_set_name(a) if key == "set_name" else "renamed-set", "1" * 64 if key != "set_name"
                  else "f" * 64)
    forge = FakeForgeUses([row]).install(monkeypatch)
    response = await _post(client, named, heldout_view_ids=[named.views["a"]])
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "heldout_reused" and detail["heldout_view"] == named.views["a"]
    assert detail["heldout_uses"] == 18 and len(forge.requests) == 1
    assert queue == [] and _bundles(sync_db, named.world) == 0


@pytest.mark.asyncio
async def test_the_old_sets_return_without_a_filter_and_are_refused(client, sync_db, named, queue, monkeypatch):
    """The case FU-7a exists for: a request that names nothing takes every ready set, so one scored set refuses it,
    and naming the fresh set alone is accepted."""
    from src.services.wsp_bundle import heldout_set_name

    FakeForgeUses([use_row(heldout_set_name(named.captures["b"]), "2" * 64)]).install(monkeypatch)
    everything = await _post(client, named)
    assert everything.status_code == 409 and everything.json()["detail"]["code"] == "heldout_reused"
    fresh = await _post(client, named, heldout_view_ids=[named.views["a"]])
    assert fresh.status_code == 202, fresh.text
    assert len(queue) == 1 and queue[0]["heldout_view_ids"] == [named.views["a"]]


@pytest.mark.asyncio
async def test_the_calibration_capture_is_exempt(client, sync_db, named, queue, monkeypatch):
    from src.services.wsp_bundle import heldout_set_name

    FakeForgeUses([use_row(heldout_set_name(named.captures["cal"]), "3" * 64)]).install(monkeypatch)
    response = await _post(client, named, heldout_view_ids=[named.views["a"]])
    assert response.status_code == 202, response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("row", [use_row("eval-x", "1" * 64, owner="miserver"), use_row("eval-x", "1" * 64, uses=0)])
async def test_another_owners_use_or_a_zero_count_is_not_a_use(client, sync_db, named, queue, monkeypatch, row):
    FakeForgeUses([row]).install(monkeypatch)
    response = await _post(client, named, heldout_view_ids=[named.views["a"]])
    assert response.status_code == 202, response.text
    assert queue[0]["used_heldout"] == []


@pytest.mark.asyncio
async def test_a_supplement_holding_scored_rows_is_refused(client, sync_db, named, queue, monkeypatch):
    FakeForgeUses([use_row("eval-somewhere", "4" * 64)]).install(monkeypatch)
    response = await _post(client, named, heldout_view_ids=[named.views["a"]],
                           supplement_view_ids=[named.views["supp"]])
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "heldout_overlaps_practice"
    assert queue == [] and _bundles(sync_db, named.world) == 0


UNREADABLE = {
    "500": dict(status=500),
    "not json": dict(body=b"<html>gateway</html>"),
    "another shape": dict(body=b'{"uses": []}'),
    "a malformed row": dict(body=b'{"heldout_uses": [{"owner": "miworkbench", "set_name": "x", '
                                 b'"row_ids_sha256": "a", "heldout_uses": "18"}]}'),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(UNREADABLE))
async def test_an_unreadable_answer_is_503_never_no_uses(client, sync_db, named, queue, monkeypatch, case):
    FakeForgeUses(**UNREADABLE[case]).install(monkeypatch)
    response = await _post(client, named, heldout_view_ids=[named.views["a"]])
    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "heldout_uses_unavailable"
    assert queue == [] and _bundles(sync_db, named.world) == 0


@pytest.mark.asyncio
async def test_miforge_not_configured_is_503(client, sync_db, named, queue, monkeypatch):
    from src.core.config import settings

    monkeypatch.setattr(settings, "miforge_api_url", "", raising=False)
    response = await _post(client, named, heldout_view_ids=[named.views["a"]])
    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "miforge_not_configured"
    assert queue == [] and _bundles(sync_db, named.world) == 0


def test_an_unreachable_miforge_is_unavailable():
    from src.services.wsp_bundle import BundleRefused
    from src.services.wsp_heldout_uses import read_heldout_uses

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(BundleRefused) as caught:
        read_heldout_uses("http://miforge.test", transport=httpx.MockTransport(refuse))
    assert (caught.value.code, caught.value.status) == ("heldout_uses_unavailable", 503)


def test_the_reader_answers_every_row_of_a_well_formed_answer():
    from src.services.wsp_heldout_uses import read_heldout_uses

    rows = [use_row("eval-pmd_0df63ef937a4", HEX), use_row("calibration-pmd_ca6b3814c34a", "b" * 64, uses=1)]
    fake = FakeForgeUses(rows)
    assert read_heldout_uses("http://miforge.test/", transport=httpx.MockTransport(fake.handler)) == rows
    assert str(fake.requests[0].url) == "http://miforge.test/api/v1/wsp/heldout-uses"
