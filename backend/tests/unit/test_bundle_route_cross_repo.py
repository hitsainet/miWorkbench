"""ENH-010 044 FR-2, FR-3 (R3 review fix OR3-044a-H1, 2026-11-17): miForge IMPORTS a real miWorkbench bundle through
miWorkbench's real routes, with miForge's own code.

The round-3 review found that `GET /api/v1/wsp/bundles/{id}` answered the operator's wrapper (`{bundle_id, status, …,
manifest: {…}}`) while miForge reads the body AS the manifest (`clients/miworkbench.py::fetch_bundle_manifest` →
`wsp/bundles.py::_check_manifest` → `schemas.validate_bundle`): every real import ended `failed: wsp_bundle_invalid`.
No test saw it because each side tested against its own picture of the other (miWorkbench read `full["manifest"]`,
miForge's fake served the bare manifest).

So this test runs miForge's import task body, `wsp.bundles.request_import` then `import_bundle`, loaded from the miForge
checkout `tests/support/sibling_repos.py` finds (never a copy), against a bundle this app BUILT for real. miForge's
`bundle_client` gets a transport into this app, so every request it makes (the manifest, then each `files[]` entry)
goes through the live routes. The import must end `imported`: `validate_bundle` against miForge's published schema,
`bundle_id` and the RFC 8785 digest of the body, each file's `X-Content-SHA256` and size, the tensor and row-id hashes,
and parity with miForge's own evaluator.

miForge is loaded only for this test: its top-level packages (`core`, `db`, `models`, `wsp`, `clients`) are put on
`sys.path` and taken out of `sys.modules` again afterwards, so nothing of it leaks into another test.

Loud skip without miForge; fails under `MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1`.

Negative control (`0xcc/reviews/044a_r3_review_fixes_2026-11-17.md`, H1-NC): the route answers the wrapper again →
red (`failed:wsp_bundle_invalid`).
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("datasets")

from tests.support.cross_repo import REQUIRE_NEW, require_cross_repo  # noqa: E402
from tests.support.sibling_repos import miforge_repo  # noqa: E402
from tests.support.wsp_pipeline import (  # noqa: E402,F401 (fixtures)
    arrow_dataset, heldout_arrow, monkeypatch_module, new_bundle, pipeline_db, tiny_lm, tiny_tokenizer,
    wsp_captures, wsp_data_dir, wsp_probe, wsp_world,
)

BASE = "http://miworkbench.test"


@contextmanager
def miforge_backend():
    """miForge's backend modules, imported from its checkout for the length of the block."""
    src = miforge_repo() / "backend" / "src"
    if not (src / "wsp" / "bundles.py").is_file() or not (src / "clients" / "miworkbench.py").is_file():
        message = (f"miForge's bundle import is not at {src} (set MIFORGE_REPO to a miForge checkout with 016 P1b). "
                   f"Skipping is expected on the public mirror; set {REQUIRE_NEW}=1 to make this a failure.")
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    tops = {p.name for p in src.iterdir() if (p / "__init__.py").is_file()} | {p.stem for p in src.glob("*.py")}
    saved = {name: module for name, module in sys.modules.items() if name.split(".")[0] in tops}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(src))
    try:
        import clients.miworkbench as client
        import wsp.bundles as bundles
        import wsp.models as models
        import wsp.settings as settings

        yield SimpleNamespace(bundles=bundles, client=client, models=models, settings=settings)
    finally:
        sys.path.remove(str(src))
        for name in [n for n in sys.modules if n.split(".")[0] in tops]:
            del sys.modules[name]
        sys.modules.update(saved)


class IntoTheApp(httpx.BaseTransport):
    """A synchronous transport that hands each request miForge's client makes to this app, and records it."""

    def __init__(self, app):
        from fastapi.testclient import TestClient

        self.app_client = TestClient(app, base_url=BASE)
        self.requests: list[tuple[str, str, dict]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode()
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        self.requests.append((request.method, path, headers))
        answer = self.app_client.request(request.method, path, headers=headers, content=request.read())
        return httpx.Response(answer.status_code, headers=answer.headers, content=answer.content)


@pytest.fixture
def route_session(pipeline_db, monkeypatch):
    """The routes read the pipeline database (where the real bundle is), as `test_wsp_import.py`'s route tests do."""
    from sqlalchemy.orm import sessionmaker

    from src.api.v1.endpoints import wsp as wsp_routes

    monkeypatch.setattr(wsp_routes, "_session", sessionmaker(bind=pipeline_db.get_bind()))


def test_miforge_imports_a_real_bundle_through_the_routes(pipeline_db, wsp_probe, wsp_captures, tiny_tokenizer,
                                                         route_session, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.main import app
    from src.services.probe_dataset_lifecycle import AGENT_HEADER
    from src.services.wsp_bundle import build_bundle

    bundle = new_bundle(pipeline_db, wsp_probe)
    build_bundle(pipeline_db, bundle, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(bundle)
    assert bundle.status == "ready", bundle.error
    transport = IntoTheApp(app)

    with miforge_backend() as mf:
        engine = create_engine(f"sqlite:///{tmp_path / 'miforge.sqlite'}")
        mf.models.WspBundle.__table__.create(engine)
        settings = mf.settings.WSPSettings(miworkbench_api_url=BASE, wsp_bundle_root=tmp_path / "wsp",
                                           wsp_providers_available="")
        with sessionmaker(bind=engine, expire_on_commit=False)() as session:
            row, queued = mf.bundles.request_import(
                session, {"owner": "miworkbench", "bundle_id": bundle.id, "bundle_sha256": bundle.bundle_sha256},
                received_by="operator", settings=settings)
            session.commit()
            assert queued
            outcome = mf.bundles.import_bundle(session, row.bundle_sha256, settings=settings, transport=transport)
            session.commit()
            assert outcome == "imported", (outcome, row.import_failure)
            assert row.manifest == bundle.manifest and row.bundle_id == bundle.id
            stored = mf.bundles.store_dir(settings, bundle.bundle_sha256)
            assert sorted(p.name for p in stored.iterdir()) == sorted(f["name"] for f in bundle.manifest["files"])
        engine.dispose()

    # What miForge asked for: the manifest, then each listed file once, and never as an agent (P31).
    asked = [(method, path) for method, path, _ in transport.requests]
    assert asked == [("GET", f"/api/v1/wsp/bundles/{bundle.id}")] + [
        ("GET", f"/api/v1/wsp/bundles/{bundle.id}/files/{f['name']}") for f in bundle.manifest["files"]]
    assert not any(AGENT_HEADER.lower() in {k.lower() for k in headers} for _, _, headers in transport.requests)


def test_miforge_refuses_a_bundle_that_is_not_ready(pipeline_db, wsp_probe, route_session, tmp_path):
    """The 409 `bundle_not_ready` reaches miForge as a failed fetch, not as a manifest it tries to read."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.main import app

    bundle = new_bundle(pipeline_db, wsp_probe)   # `building`: no manifest yet
    with miforge_backend() as mf:
        engine = create_engine(f"sqlite:///{tmp_path / 'miforge.sqlite'}")
        mf.models.WspBundle.__table__.create(engine)
        settings = mf.settings.WSPSettings(miworkbench_api_url=BASE, wsp_bundle_root=tmp_path / "wsp",
                                           wsp_providers_available="")
        with sessionmaker(bind=engine, expire_on_commit=False)() as session:
            row, _ = mf.bundles.request_import(
                session, {"owner": "miworkbench", "bundle_id": bundle.id, "bundle_sha256": "0" * 64},
                received_by="operator", settings=settings)
            session.commit()
            outcome = mf.bundles.import_bundle(session, row.bundle_sha256, settings=settings,
                                               transport=IntoTheApp(app))
            assert outcome == "failed:wsp_bundle_fetch_failed", (outcome, row.import_failure)
            assert "409" in row.import_failure["message"]
        engine.dispose()
