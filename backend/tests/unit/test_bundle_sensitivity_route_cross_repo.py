"""ENH-010 044 task 5.6 (P41, R1B-X1): miForge's OWN client for the bundle's live sensitivity
(`miForge/backend/src/clients/miworkbench.py::read_bundle_sensitivity`) parses miWorkbench's real route, through
the real app. Loud skip without miForge; fails under `MIWORKBENCH_REQUIRE_CROSS_REPO_CHECKS=1`.

The client is loaded from miForge's checkout by path and given a transport into this app; nothing of miForge is
imported otherwise: its function imports miForge's settings module first, which a stub stands in for."""
from __future__ import annotations

import importlib.util

import pytest

from tests.support.cross_repo import REQUIRE_NEW, require_cross_repo
from tests.support.sibling_repos import miforge_repo
from tests.unit.test_wsp_bundle_routes import bundle  # noqa: F401 (fixture)
from tests.support.wsp_world import sync_db  # noqa: F401 (fixture)


def _miforge_client():
    path = miforge_repo() / "backend" / "src" / "clients" / "miworkbench.py"
    if not path.is_file():
        message = (f"miForge's miWorkbench client is not at {path} (set MIFORGE_REPO to a miForge checkout with 016). "
                   f"Skipping is expected on the public mirror; set {REQUIRE_NEW}=1 to make this a failure.")
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    spec = importlib.util.spec_from_file_location("miforge_miworkbench_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_miforges_client_reads_the_live_value(bundle, sync_db, monkeypatch):
    from fastapi.testclient import TestClient

    from src.main import app
    from src.models.wsp import WspSensitivityMark

    import sys
    import types

    client_module = _miforge_client()
    # The function imports miForge's settings module before it reads `base_url`; a stub stands in for it (only
    # `base_url` is used when it is passed, so the stub's value is never read).
    settings_stub = types.ModuleType("wsp.settings")
    settings_stub.get_wsp_settings = lambda: types.SimpleNamespace(miworkbench_api_url="http://unused.invalid")
    monkeypatch.setitem(sys.modules, "wsp", types.ModuleType("wsp"))
    monkeypatch.setitem(sys.modules, "wsp.settings", settings_stub)
    calls = []

    class Routed(TestClient):
        def __init__(self, *, base_url, timeout=None, trust_env=None, **_):
            calls.append(base_url)
            super().__init__(app, base_url=base_url)

    monkeypatch.setattr(client_module.httpx, "Client", Routed)
    bundle_id = bundle["row"].id
    assert client_module.read_bundle_sensitivity(bundle_id, base_url="http://miworkbench.test") == "unknown"
    sync_db.add(WspSensitivityMark(dataset_id=bundle["world"].dataset_id, value="sensitive", reason="PII"))
    sync_db.commit()
    assert client_module.read_bundle_sensitivity(bundle_id, base_url="http://miworkbench.test") == "sensitive"
    assert calls == ["http://miworkbench.test", "http://miworkbench.test"]
