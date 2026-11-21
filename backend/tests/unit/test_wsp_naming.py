"""ENH-010 044 task 10.3 (FR-22): across the suite "probe" means a miWorkbench activation classifier; weightspace's
evaluation callbacks are "observation tools". No WSP surface may call a weight-space candidate, a WSP head or an
observation tool a "probe" in the wrong sense. Read from the LIVE surfaces: the app's OpenAPI text for `/wsp` routes
and the published scoring contract (MCP tool descriptions join with 044-C).

Control (0xcc/reviews/044_implementation_controls_2026-11-16.md): plant "WSP probes" in a `/wsp` route's description
→ red.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FORBIDDEN = re.compile(
    r"\b(wsp|weight[- ]space|observation)[ -]probes?\b|\bprobe[ -](callbacks?|tools?)\b|\bcandidate probes?\b",
    re.IGNORECASE)
REPO = Path(__file__).resolve().parents[3]


def _wsp_text() -> str:
    from src.main import app

    paths = {p: item for p, item in app.openapi()["paths"].items() if p.startswith("/api/v1/wsp")}
    assert paths, "no /wsp route in the live app; the scan would pass vacuously"
    contract = (REPO / "docs" / "schemas" / "miworkbench-wsp-scoring-v1.json").read_text()
    return json.dumps(paths) + contract


def test_no_wsp_surface_misuses_probe():
    hits = sorted(set(m.group(0) for m in FORBIDDEN.finditer(_wsp_text())))
    assert not hits, f"use 'observation tools' for weightspace's callbacks and never call a WSP head a probe: {hits}"


def test_the_pattern_catches_the_planted_phrases():
    for phrase in ("WSP probes", "weight-space probe", "observation probe", "probe callbacks", "candidate probe"):
        assert FORBIDDEN.search(phrase), phrase
    for fine in ("the parent probe's head", "a derived probe", "observation tools", "probe_id"):
        assert not FORBIDDEN.search(fine), fine
