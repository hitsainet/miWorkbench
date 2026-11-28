"""A J-lens poll of a task that returned without building (046-2 review round 1, finding 5).

`refuses_terminal_row` returns `{"status": "skipped", "reason": "already_<status>", ...}` and
`jlens_progress.owns_its_failure` returns `{"status": "cancelled" | "failed", "reason", "detail", ...}`; Celery
records either as SUCCESS. The readout poll built `ReadoutResponse(**info)` from it, a 500; the probe and decompose
polls answered SUCCESS with no scores, which reads as "the lens found nothing". Each now answers FAILURE (the state
the panel's `pollReadout` and the MCP polls end on; review round 2, finding 2) with the task's own status and why, on
its contract (the answer middleware checks every call here).
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

POLLS = {"readout": "/api/v1/jlens/readout/t1", "probe": "/api/v1/jlens/probe/t1",
         "decompose": "/api/v1/jlens/decompose/t1"}
#: Each return as its decorator really builds it (`core/cancellation.refuses_terminal_row`;
#: `workers/jlens_progress.owns_its_failure`, whose cancelled return carries `detail` and no `reason`).
STOPPED = [
    ({"status": "skipped", "reason": "already_completed", "scope": "jlens_task", "target_id": "t1"},
     "the task returned without building: skipped (already_completed)"),
    ({"status": "cancelled", "scope": "jlens_task", "target_id": "t1", "detail": ""},
     "the task returned without building: cancelled"),
    ({"status": "failed", "scope": "jlens_task", "target_id": "t1", "reason": "gpu_lease_lost",
      "detail": "its GPU lease was lost"}, "the task returned without building: failed (its GPU lease was lost)"),
]


@pytest.fixture
def celery_answers(monkeypatch):
    """Every `AsyncResult` reads as SUCCESS with `payload` (its class's properties patched: the polls build their own
    instances, through `celery.result.AsyncResult` or `celery_app.AsyncResult`)."""
    from celery.result import AsyncResult

    def install(payload):
        for name, value in (("state", "SUCCESS"), ("info", payload), ("result", payload)):
            monkeypatch.setattr(AsyncResult, name, property(lambda self, v=value: v))

    return install


def _get(path):
    from src.main import app

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(go())


@pytest.mark.parametrize("poll", sorted(POLLS))
@pytest.mark.parametrize(("payload", "error"), STOPPED, ids=[p["status"] for p, _e in STOPPED])
def test_a_task_that_returned_without_building_is_a_failure_with_why(celery_answers, poll, payload, error):
    """FAILURE, the terminal state the panel and the MCP polls end on, with the task's own status and why."""
    celery_answers(payload)
    response = _get(POLLS[poll])
    assert response.status_code == 200, response.text[:300]
    body = response.json()
    assert body["status"] == "FAILURE" and body["error"] == error, body
    assert not body.get("scores") and not body.get("decompositions") and not body.get("readout"), body


def test_a_real_success_still_answers_success(celery_answers):
    celery_answers({"scores": [], "mode": "probe", "lens_type": "LOGIT_LENS"})
    body = _get(POLLS["probe"]).json()
    assert body["status"] == "SUCCESS" and body["mode"] == "probe" and body["error"] is None, body


# ── the replication report's served model (046-2 review round 1, finding 2) ───────────────────────────────────────


def test_a_stored_non_finite_figure_is_answered_as_null_not_a_500(monkeypatch, tmp_path):
    """The report is read back with `json.loads` from a file `json.dumps` wrote, which allows NaN and infinity. Through
    the served `Optional[Dict[str, Any]]` (main's) they answer as null; with no response model they were a 500."""
    import json
    import math
    from types import SimpleNamespace

    from src.api.v1.endpoints import jlens
    from src.services.jlens_replication import REPLICATION_FILENAME

    report = {"model_id": "m", "reference_commit": "c", "complete": False, "missing": {}, "evaluation_sets": ["a"],
              "notes": "", "results": [{"lens": "l", "pass_at_k_auc": {"a": math.nan}, "ablation_kl": math.inf,
                                        "swap_success_rate": None, "coverage": []}]}
    (tmp_path / REPLICATION_FILENAME).write_text(json.dumps(report))
    service = SimpleNamespace(list_artifacts=lambda: [SimpleNamespace(slug="s1", directory=tmp_path)])
    monkeypatch.setattr(jlens, "_service", lambda: service)
    response = _get("/api/v1/jlens/reports/replication?slug=s1")
    assert response.status_code == 200, response.text[:300]
    row = response.json()["results"][0]
    assert row["pass_at_k_auc"] == {"a": None} and row["ablation_kl"] is None, row
