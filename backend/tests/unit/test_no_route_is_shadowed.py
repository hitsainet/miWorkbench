"""No route of the live app is captured by an earlier template (hotfix fix/labeling-route-shadowing, 2026-11-26).

Routes match in registration order, and a path parameter matches any segment. `GET /labeling/{labeling_job_id}` stood
above `GET /labeling/trials`, so every list of trials was answered as the status of a labeling job named "trials" (404
"Labeling job trials not found"), and the MCP tool `list_labeling_trials` failed on every call in production.

1. The guard, from the LIVE route table (`approval_gate.iter_routes(src.main.app)`, the order requests are matched in,
   through included routers): for each route, its own path with every parameter filled must not be matched by an
   EARLIER route sharing a method. Its own control: a planted router with the defect is caught.
2. The wiring: `GET /api/v1/labeling/trials` reaches `list_labeling_trials` (an empty list on an empty database, not
   the job-status 404), and `GET /api/v1/labeling/{id}` still reaches the job status.

Control (record in the hand-back): move `GET /labeling/{labeling_job_id}` back above the trials block → both red.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, FastAPI
from starlette.routing import compile_path


def _shadowed(app) -> list[str]:
    from src.core.approval_gate import iter_routes

    routes = [(path, route, compile_path(path)[0]) for path, route in iter_routes(app)]
    found = []
    for j, (path, route, _regex) in enumerate(routes):
        concrete = re.sub(r"\{[^}]+\}", "zz9", path)
        for earlier_path, earlier, regex in routes[:j]:
            shared = route.methods & earlier.methods
            if earlier_path != path and shared and regex.match(concrete):
                found.append(f"{sorted(shared)} {path} ({route.endpoint.__name__}) is captured by the earlier "
                             f"{earlier_path} ({earlier.endpoint.__name__})")
    return found


def test_no_route_of_the_live_app_is_captured_by_an_earlier_one():
    from src.core.approval_gate import iter_routes
    from src.main import app

    assert len(list(iter_routes(app))) > 300  # the walk reaches the included routers, not only the app's own
    assert _shadowed(app) == []


def test_the_guard_catches_a_planted_shadow():
    router = APIRouter()

    @router.get("/jobs/{job_id}")
    async def job(job_id: str):  # pragma: no cover - never called
        return {}

    @router.get("/jobs/trials")
    async def trials():  # pragma: no cover - never called
        return []

    planted = FastAPI()
    planted.include_router(router, prefix="/api/v1")
    assert _shadowed(planted) == [
        "['GET'] /api/v1/jobs/trials (trials) is captured by the earlier /api/v1/jobs/{job_id} (job)"]


async def test_the_trials_list_and_the_job_status_both_answer(client):
    listed = await client.get("/api/v1/labeling/trials")
    assert listed.status_code == 200, listed.text[:300]
    assert isinstance(listed.json(), (list, dict)) and "Labeling job" not in listed.text
    status = await client.get("/api/v1/labeling/lj_missing")
    assert status.status_code == 404 and "Labeling job lj_missing not found" in status.text
