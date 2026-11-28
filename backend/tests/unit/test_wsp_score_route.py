"""ENH-010 044 tasks 7.2, 7.5 through the real app: `POST /wsp/score` refuses an agent caller, and answers 429
`scorer_busy` with `Retry-After` while a call holds the scorer.

Controls (044 controls record): drop the route's agent refusal → `test_agent_header_refused` red; skip the
semaphore → `test_busy_answers_429` red.
"""
from __future__ import annotations

import pytest

from src.services.probe_dataset_lifecycle import AGENT_HEADER

BODY = {"kind": "miworkbench.wsp-score-request/v1", "bundle_id": "wsb_0123456789ab", "bundle_sha256": "0" * 64,
        "run_ref": {"miforge_run_id": "r", "patch_sha256": None, "max_changed_entries": None},
        "heldout_ref": {"owner": "miworkbench", "sets": [{"name": "eval-x", "view_id": "pmd_x", "role": "eval",
                                                           "rows": 1, "row_ids_sha256": "0" * 64}]},
        "heads": [{"label": "parent", "tensors": None}], "primary_sets": []}


@pytest.mark.asyncio
async def test_agent_header_refused(client):
    response = await client.post("/api/v1/wsp/score", json=BODY, headers={AGENT_HEADER: "mcp"})
    assert response.status_code == 403 and response.json()["detail"]["code"] == "operator_or_service_only"


@pytest.mark.asyncio
async def test_busy_answers_429(client):
    from src.services import wsp_score

    assert wsp_score.SEMAPHORE.acquire(blocking=False)
    try:
        response = await client.post("/api/v1/wsp/score", json=BODY)
    finally:
        wsp_score.SEMAPHORE.release()
    assert response.status_code == 429 and response.json()["detail"]["code"] == "scorer_busy"
    assert response.headers["retry-after"] == "5"
    # Free again: the same body reaches the scorer, which answers the bundle's own refusal.
    again = await client.post("/api/v1/wsp/score", json=BODY)
    assert again.status_code == 404 and again.json()["detail"]["code"] == "bundle_not_found"


@pytest.mark.asyncio
async def test_the_call_record_names_the_caller_behind_the_proxy(client, monkeypatch):
    """FR-5 records the caller's address (R2 review fix OR2-044a-L2): behind the ingress the socket peer is the
    proxy, so the route hands the scorer the peer, the `X-Forwarded-For` chain as sent, and the `User-Agent` — one
    call, that payload. Control: drop `forwarded_for` from the route → red."""
    from src.services import wsp_score

    seen = []

    def fake_score(db, body, *, caller=None, stop=None):
        seen.append(caller)
        return {"recorded": True}

    monkeypatch.setattr(wsp_score, "score", fake_score)
    response = await client.post("/api/v1/wsp/score", json=BODY,
                                 headers={"X-Forwarded-For": "203.0.113.7, 10.42.0.1", "User-Agent": "miforge/1"})
    assert response.status_code == 200 and response.json() == {"recorded": True}
    assert len(seen) == 1 and seen[0]["forwarded_for"] == "203.0.113.7, 10.42.0.1"
    assert seen[0]["user_agent"] == "miforge/1" and set(seen[0]) == {"address", "forwarded_for", "user_agent"}
