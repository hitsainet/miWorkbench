"""Every covered route of `src/api/v1/endpoints/behaviours.py` answers over HTTP, on its contract (046 FTASKS 9.2; FR-046.17 as amended by WT8).

The cases are the committed contract's keys whose endpoint this router defines (`covered_route_cases`), never
written by hand: one per route as the operator, and a gated route's agent call (202, nothing queued) beside it. Each
runs on its own world through the live app, so the answer middleware checks the served bytes.
"""
from __future__ import annotations

import pytest

from tests.support import covered_route_cases as crc

ROUTER = "behaviours.py"
CASES = crc.cases(crc.covered_for_router(ROUTER))


def test_the_cases_are_this_routers_covered_subset():
    crc.assert_router_set(ROUTER, CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
async def test_the_route_answers_on_its_contract(case, async_session, monkeypatch, tmp_path):
    await crc.assert_case(case, async_session, monkeypatch, tmp_path)
