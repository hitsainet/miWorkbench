"""`GET /features/{id}/ablation` on a feature with fewer than 10 stored activations answers 409 with the reason, not
an unhandled 500 (found by the 046 P8 route cases, 2026-11-26). Real rows: the backstop world's features have none.

Control: drop the endpoint's `except ValueError` → this test red (500).
"""
from __future__ import annotations


async def test_too_few_activations_is_a_409_with_the_reason(async_session, monkeypatch, tmp_path):
    from src.core.config import settings
    from src.main import app
    from tests.support import backstop, backstop_world

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    world = await backstop_world.build_world(async_session, tmp_path, app)
    response = await backstop.call(app, "GET", f"/api/v1/features/{world['feature']}/ablation")
    assert response.status_code == 409, (response.status_code, response.text[:300])
    assert "Insufficient activation data" in response.json()["detail"] and "found 0" in response.json()["detail"]
