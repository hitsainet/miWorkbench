"""`tests/support/mcp_drive.RouteBackend` (046 FTASKS 3.1, FTID §11): per-route fake answers, every request recorded,
and an UNMAPPED route never answered with `{}`: it answers 599 and `patched_routes` fails naming it."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from tests.support import mcp_drive


def _get(backend: mcp_drive.RouteBackend, method: str, path: str, json=None) -> httpx.Response:
    async def go():
        async with httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(backend.handler)) as client:
            return await client.request(method, path, json=json)

    return asyncio.run(go())


def test_it_answers_per_route_and_matches_parameters_by_segment():
    backend = (mcp_drive.RouteBackend()
               .answer("GET", "/api/models/{model_id}", {"id": 1})
               .answer("GET", "/api/models/active", {"active": True})
               .answer("POST", "/api/models/{}/load", {"queued": True}, status=202))
    assert _get(backend, "GET", "/api/models/7").json() == {"id": 1}
    assert _get(backend, "GET", "/api/models/active").json() == {"active": True}  # the literal template wins
    response = _get(backend, "POST", "/api/models/7/load", json={"a": 1})
    assert response.status_code == 202 and response.json() == {"queued": True}
    assert backend.calls_to("POST", "/api/models/{}/load") == [{"a": 1}]
    assert backend.calls_to("GET", "/api/models/{}") == [None]


def test_an_unmapped_route_answers_599_and_is_named():
    backend = mcp_drive.RouteBackend().answer("GET", "/a", {})
    response = _get(backend, "GET", "/b/1")
    assert response.status_code == mcp_drive.RouteBackend.UNMAPPED_STATUS
    assert backend.unmapped == ["GET /b/1"]
    assert _get(backend, "POST", "/a").status_code == 599  # the method is part of the route


def test_patched_routes_fails_naming_an_unmapped_route(monkeypatch):
    with pytest.raises(AssertionError, match=r"GET /api/v1/nowhere"):
        with mcp_drive.patched_routes(monkeypatch) as backend:
            _get(backend, "GET", "/api/v1/nowhere")


def test_health_probes_answer_healthy_unless_down():
    backend = mcp_drive.RouteBackend()
    assert _get(backend, "GET", "/api/health").json() == {"status": "healthy"}
    backend.down = True
    assert _get(backend, "GET", "/api/health").status_code == 503


def test_a_text_body_is_sent_as_it_is():
    backend = mcp_drive.RouteBackend().answer("GET", "/f", "a\nb\n", headers={"content-type": "application/jsonl"})
    response = _get(backend, "GET", "/f")
    assert response.text == "a\nb\n" and response.headers["content-type"] == "application/jsonl"
