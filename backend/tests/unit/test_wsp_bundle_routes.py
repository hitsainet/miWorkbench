"""ENH-010 044 (tasks 4.3, 5.3, 5.6, 7.2, 8.6): the WSP routes exist in the LIVE app (`app.openapi()["paths"]`, never
`app.routes`: memory `app-routes-is-not-a-route-list`), and the agent refusals are wired on the routes themselves.

Control (0xcc/reviews/044_implementation_controls_2026-11-16.md): drop the `include_router(wsp.router)` line → red.
"""
from __future__ import annotations

import pytest

#: (method, path) of every WSP route built so far, by task. Phase 3 and 4 routes join with their tasks.
ROUTES = {
    ("post", "/api/v1/wsp/captures"),   # 4.3
    ("get", "/api/v1/wsp/captures"),    # 4.3
    ("post", "/api/v1/wsp/sensitivity-marks"),  # 5.4
    ("get", "/api/v1/wsp/sensitivity"),         # 5.4
    ("post", "/api/v1/wsp/bundles"),                            # 5.3 / 6.6
    ("get", "/api/v1/wsp/bundles/{bundle_id}"),                 # 5.3
    ("get", "/api/v1/wsp/bundles/{bundle_id}/summary"),         # 5.3
    ("get", "/api/v1/wsp/bundles/{bundle_id}/files/{name}"),    # 5.3
    ("get", "/api/v1/wsp/bundles/{bundle_id}/sensitivity"),     # 5.6
    ("post", "/api/v1/wsp/score"),                              # 7.2
    ("post", "/api/v1/wsp/candidates/import"),                  # 8.6
}


@pytest.fixture(scope="module")
def paths():
    from src.main import app

    return app.openapi()["paths"]


def test_routes_in_openapi(paths):
    served = {(method, path) for path, item in paths.items() for method in item if path.startswith("/api/v1/wsp")}
    assert ROUTES <= served, f"missing: {sorted(ROUTES - served)}"


# ── 5.3 / 5.6: the bundle routes through the real app ─────────────────────────
#
# Controls: list a capture file in the manifest → `test_unlisted_name_is_404`'s capture assertion red (the route
# serves only what `files[]` lists, so the guard is the list); drop `refuse_agent_callers` → `test_agent_header_refused`
# red; serve the manifest's sensitivity snapshot → `test_sensitivity_route_reads_live` red.

import hashlib  # noqa: E402

from src.services.probe_dataset_lifecycle import AGENT_HEADER  # noqa: E402
from tests.support.wsp_world import FakeForgeUses, add_view, seed_probe, sync_db  # noqa: E402,F401 (fixture)


@pytest.fixture
def bundle(sync_db, tmp_path, monkeypatch):
    """A ready bundle row with two listed files on disk (the run's token file served in place, and heads), plus a
    capture file beside it that the manifest does NOT list."""
    from src.core.config import settings
    from src.models.wsp import WspActivationCapture, WspBundle
    from src.services import wsp_layout

    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    world = seed_probe(sync_db)
    view = add_view(sync_db, world, role="eval")
    capture = WspActivationCapture(probe_id=world.probe_id, dataset_id=view, role="eval", layer=1, scope="all",
                                   status="ready")
    sync_db.add(capture)
    row = WspBundle(probe_id=world.probe_id, status="ready")
    sync_db.add(row)
    sync_db.commit()
    directory = wsp_layout.bundle_dir(world.run_id, row.id)
    directory.mkdir(parents=True)
    run_dir = directory.parents[2]
    (run_dir / "tokens_layer1.f16").write_bytes(b"\x01\x02" * 8)
    (directory / "heads.safetensors").write_bytes(b"heads")
    capture_dir = wsp_layout.capture_dir(world.run_id, capture.id)
    capture_dir.mkdir(parents=True)
    (capture_dir / "tokens.f16").write_bytes(b"held-out")
    files = [{"name": p.name, "role": role, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
              "bytes": p.stat().st_size}
             for role, p in (("practice_activations", run_dir / "tokens_layer1.f16"),
                             ("heads", directory / "heads.safetensors"))]
    row.manifest = {"kind": "miforge.wsp-bundle/v1", "files": files, "practice": {"view_id": world.view_id},
                    "tensors": []}
    row.dir_path, row.capture_ids = str(directory), [capture.id]
    row.sensitivity, row.sensitivity_basis = "not_sensitive", [{"source": "operator_mark", "ref": "x",
                                                                 "value": "not_sensitive"}]
    sync_db.commit()
    return {"row": row, "world": world, "files": files, "eval_view": view}


@pytest.mark.asyncio
async def test_listed_files_stream_with_their_digest(client, bundle):
    for entry in bundle["files"]:
        response = await client.get(f"/api/v1/wsp/bundles/{bundle['row'].id}/files/{entry['name']}")
        assert response.status_code == 200, (entry, response.text)
        assert response.headers["x-content-sha256"] == entry["sha256"]
        assert hashlib.sha256(response.content).hexdigest() == entry["sha256"]


@pytest.mark.asyncio
async def test_unlisted_name_is_404(client, bundle):
    base = f"/api/v1/wsp/bundles/{bundle['row'].id}/files"
    for name in ("tokens.f16", "manifest.json", "..%2Fx", "../x", "offsets.npy", "positions.npy"):
        response = await client.get(f"{base}/{name}")
        assert response.status_code == 404, (name, response.status_code)


@pytest.mark.asyncio
async def test_agent_header_refused(client, bundle):
    bundle_id = bundle["row"].id
    for path in (f"/api/v1/wsp/bundles/{bundle_id}", f"/api/v1/wsp/bundles/{bundle_id}/sensitivity",
                 f"/api/v1/wsp/bundles/{bundle_id}/files/heads.safetensors"):
        response = await client.get(path, headers={AGENT_HEADER: "mcp"})
        assert response.status_code == 403, path
        assert response.json()["detail"]["code"] == "operator_or_service_only"
    assert (await client.get(f"/api/v1/wsp/bundles/{bundle_id}/summary", headers={AGENT_HEADER: "mcp"})
            ).status_code == 200


@pytest.mark.asyncio
async def test_summary_has_no_file_names(client, bundle):
    body = (await client.get(f"/api/v1/wsp/bundles/{bundle['row'].id}/summary")).json()
    text = repr(body)
    assert "files" not in body["manifest"]
    for entry in bundle["files"]:
        assert entry["name"] not in text and entry["sha256"] not in text
    full = (await client.get(f"/api/v1/wsp/bundles/{bundle['row'].id}")).json()
    assert [f["name"] for f in full["files"]] == [f["name"] for f in bundle["files"]]


@pytest.mark.asyncio
async def test_the_bundle_route_answers_exactly_the_manifest(client, bundle, sync_db):
    """R3 review fix (OR3-044a-H1): `GET /wsp/bundles/{id}` answers the manifest itself (FTDD §5.2), the document
    miForge validates and digests, never the operator's wrapper; until the bundle is `ready` it answers 409
    `bundle_not_ready`. Control: answer `bundle_view(...)` again → red (and red in the cross-repo test)."""
    row = bundle["row"]
    response = await client.get(f"/api/v1/wsp/bundles/{row.id}")
    assert response.status_code == 200, response.text
    assert response.json() == row.manifest
    for status in ("building", "failed"):
        row.status = status
        sync_db.commit()
        refused = await client.get(f"/api/v1/wsp/bundles/{row.id}")
        assert refused.status_code == 409 and refused.json()["detail"]["code"] == "bundle_not_ready", refused.text
    assert (await client.get("/api/v1/wsp/bundles/wsb_000000000000")).status_code == 404


@pytest.mark.asyncio
async def test_sensitivity_route_reads_live(client, bundle, sync_db):
    """The snapshot says `not_sensitive`; a mark made AFTER the build raises the source, and the route answers the
    new value with its basis — the snapshot is never served as the answer (P41)."""
    from src.models.wsp import WspSensitivityMark

    first = (await client.get(f"/api/v1/wsp/bundles/{bundle['row'].id}/sensitivity")).json()
    assert first["value"] == "unknown"   # no mark, no miDataworks record: the snapshot's value is not the answer
    sync_db.add(WspSensitivityMark(dataset_id=bundle["world"].dataset_id, value="sensitive", reason="found PII"))
    sync_db.commit()
    now = (await client.get(f"/api/v1/wsp/bundles/{bundle['row'].id}/sensitivity")).json()
    assert now["value"] == "sensitive" and any(b["source"] == "operator_mark" for b in now["basis"])
    assert now["read_at"]


@pytest.mark.asyncio
async def test_live_sensitivity_reads_the_heldout_sources(client, bundle, sync_db):
    """R1 review fix (OR1-044a-3), 2026-11-16: FR-7/P41 — the live value covers the practice view AND every
    referenced capture's view. Practice marked `not_sensitive`, one held-out source marked `sensitive`: the bundle is
    `sensitive`. Control (A29): read only the practice view → red."""
    import uuid

    from src.models.dataset import Dataset, DatasetStatus
    from src.models.wsp import WspActivationCapture, WspSensitivityMark

    other = Dataset(id=uuid.uuid4(), name="heldout-src", source="local", status=DatasetStatus.READY, raw_path="/y",
                    extra_metadata={})
    sync_db.add(other)
    sync_db.commit()
    view = add_view(sync_db, bundle["world"], role="everyday_text", dataset_id=other.id)
    capture = WspActivationCapture(probe_id=bundle["world"].probe_id, dataset_id=view, role="everyday_text", layer=1,
                                   scope="all", status="ready")
    sync_db.add(capture)
    sync_db.commit()
    row = bundle["row"]
    row.capture_ids = list(row.capture_ids) + [capture.id]
    sync_db.add_all([WspSensitivityMark(dataset_id=bundle["world"].dataset_id, value="not_sensitive", reason="ok"),
                     WspSensitivityMark(dataset_id=other.id, value="sensitive", reason="held-out PII")])
    sync_db.commit()
    body = (await client.get(f"/api/v1/wsp/bundles/{row.id}/sensitivity")).json()
    assert body["value"] == "sensitive", body
    assert "unresolved" not in body


@pytest.mark.asyncio
async def test_a_source_that_no_longer_resolves_reads_unknown(client, bundle, sync_db):
    """OR1-044a-3's second edge: a listed capture row that is gone is not "fewer sources" — the answer is `unknown`,
    even with every remaining source marked `not_sensitive`. Control: drop the unresolved check → red."""
    from src.models.wsp import WspSensitivityMark

    row = bundle["row"]
    row.capture_ids = list(row.capture_ids) + ["wac_gone00000000"]
    sync_db.add(WspSensitivityMark(dataset_id=bundle["world"].dataset_id, value="not_sensitive", reason="ok"))
    sync_db.commit()
    body = (await client.get(f"/api/v1/wsp/bundles/{row.id}/sensitivity")).json()
    assert body["value"] == "unknown" and body["unresolved"] == ["capture wac_gone00000000"], body


@pytest.mark.asyncio
async def test_a_bundle_needs_a_ready_capture(client, sync_db, monkeypatch):
    sent = []
    from src.workers import wsp_tasks

    monkeypatch.setattr(wsp_tasks.build_wsp_bundle, "delay", lambda **kw: sent.append(kw))
    world = seed_probe(sync_db)
    response = await client.post("/api/v1/wsp/bundles", json={"probe_id": world.probe_id})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "capture_not_ready" and sent == []


@pytest.mark.asyncio
async def test_a_bundle_request_queues_one_build(client, bundle, sync_db, monkeypatch):
    from types import SimpleNamespace

    from src.models.wsp import WspBundle
    from src.workers import wsp_tasks

    sent = []
    monkeypatch.setattr(wsp_tasks.build_wsp_bundle, "delay",
                        lambda **kw: sent.append(kw) or SimpleNamespace(id="t-1"))
    forge = FakeForgeUses().install(monkeypatch)
    response = await client.post("/api/v1/wsp/bundles", json={"probe_id": bundle["world"].probe_id})
    assert response.status_code == 202, response.text
    bundle_id = response.json()["bundle_id"]
    # FU-7a: the request's named sets (none here) and miForge's uses (none) travel with the task.
    assert sent == [{"bundle_id": bundle_id, "heldout_view_ids": None, "supplement_view_ids": None,
                     "used_heldout": []}]
    assert len(forge.requests) == 1
    assert sync_db.get(WspBundle, bundle_id).status == "building"


@pytest.mark.asyncio
async def test_a_calibration_set_probe_without_its_calibration_capture_is_refused_at_the_route(client, sync_db,
                                                                                               monkeypatch):
    """OR1-044a-1's route half (R2 review fix OR2-044a-M6: review mutation M12 survived): a probe whose bar was cut on
    its calibration set, with a ready eval capture and NO calibration capture, is refused 409 `capture_not_ready`
    before anything is queued or written. Twin: with the calibration capture the same request queues exactly one
    build. Control: switch the route's `calibration_capture_missing` check off → red."""
    from types import SimpleNamespace

    from src.models.wsp import WspActivationCapture, WspBundle
    from src.workers import wsp_tasks

    sent = []
    monkeypatch.setattr(wsp_tasks.build_wsp_bundle, "delay",
                        lambda **kw: sent.append(kw) or SimpleNamespace(id=f"t-{len(sent)}"))
    FakeForgeUses().install(monkeypatch)
    world = seed_probe(sync_db, threshold=0.5, target_fpr=0.01, threshold_source="calibration_set")
    view = add_view(sync_db, world, role="eval")
    sync_db.add(WspActivationCapture(probe_id=world.probe_id, dataset_id=view, role="eval", layer=1, scope="all",
                                     status="ready"))
    sync_db.commit()

    def bundles():
        sync_db.expire_all()
        return sync_db.query(WspBundle).filter(WspBundle.probe_id == world.probe_id).count()

    response = await client.post("/api/v1/wsp/bundles", json={"probe_id": world.probe_id})
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "capture_not_ready" and "calibration set" in detail["message"]
    assert sent == [] and bundles() == 0

    from src.services.wsp_capture import calibration_arrays_sha256

    calibration = add_view(sync_db, world, role="calibration")
    # Checked against the bar the probe serves now (OR3-044a-L3): its threshold and its (here unrecorded) arrays.
    sync_db.add(WspActivationCapture(probe_id=world.probe_id, dataset_id=calibration, role="calibration", layer=1,
                                     scope="all", status="ready",
                                     parity={"passed": True, "threshold_served": 0.5,
                                             "calibration_arrays_sha256": calibration_arrays_sha256(None, None,
                                                                                                    None)}))
    sync_db.commit()
    queued = await client.post("/api/v1/wsp/bundles", json={"probe_id": world.probe_id})
    assert queued.status_code == 202, queued.text
    assert sent == [{"bundle_id": queued.json()["bundle_id"], "heldout_view_ids": None, "supplement_view_ids": None,
                     "used_heldout": []}] and bundles() == 1
