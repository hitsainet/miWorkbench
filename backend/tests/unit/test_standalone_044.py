"""ENH-010 044 task 10.2 (FR-10; memory `midataworks-standalone-first`): phase 2 works with NO sibling — capture
planning, a bundle build with its baselines, a score call on the parent AND on a tensor-bearing candidate, and the
file import of that candidate all run with `miforge_api_url` and `midataworks_api_url` empty and every outbound HTTP
request failing. The phase-3 cases
(`::test_phase3_standalone_refuses_tensor_heads`, `::test_phase3_file_import_uses_the_local_score_call`) join with
044-C's miForge client.

Control (0xcc/reviews/044_implementation_controls_2026-11-16.md): make the bundle read miForge (any outbound call)
→ red.
"""
from __future__ import annotations

import pytest

pytest.importorskip("datasets")

from tests.support.wsp_pipeline import (  # noqa: E402,F401 (fixtures)
    arrow_dataset,
    heldout_arrow,
    monkeypatch_module,
    new_bundle,
    pipeline_db,
    tiny_lm,
    tiny_tokenizer,
    wsp_captures,
    wsp_data_dir,
    wsp_probe,
    wsp_world,
)


def test_phase2_standalone(pipeline_db, wsp_probe, wsp_captures, tiny_tokenizer, monkeypatch):
    import httpx

    from src.core.config import settings
    from src.schemas.wsp_scoring import WspScoreRequest
    from src.services import wsp_score
    from src.services.wsp_bundle import build_bundle
    from src.services.wsp_capture import plan_capture

    calls = []

    def no_network(self, request, *args, **kwargs):
        calls.append(str(request.url))
        raise httpx.ConnectError("standalone: no sibling is reachable", request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(settings, "miforge_api_url", "", raising=False)
    monkeypatch.setattr(settings, "midataworks_api_url", "", raising=False)

    assert plan_capture(pipeline_db, wsp_probe)
    bundle = new_bundle(pipeline_db, wsp_probe)
    build_bundle(pipeline_db, bundle, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(bundle)
    assert bundle.status == "ready" and bundle.manifest["sensitivity"]["value"] == "unknown"
    request = WspScoreRequest.model_validate({
        "kind": "miworkbench.wsp-score-request/v1", "bundle_id": bundle.id, "bundle_sha256": bundle.bundle_sha256,
        "run_ref": {"miforge_run_id": "local", "patch_sha256": None, "max_changed_entries": None},
        "heldout_ref": bundle.manifest["heldout_ref"], "heads": [{"label": "parent", "tensors": None}],
        "primary_sets": []})
    response = wsp_score.score(pipeline_db, request)
    assert response["sets"] and calls == []

    # R2 review fix (OR2-044a-L7), 2026-11-17: FR-10's phase-2 list also has the scorer on ANY head (a tensor-bearing
    # candidate, under the budget) and the file import — both with no sibling.
    from src.models.probe_monitor import ProbeMonitor
    from src.services import wsp_import
    from tests.support.wsp_candidate import make_candidate, score_candidate

    candidate = make_candidate(bundle, entries=[2, 12], delta=0.3)
    scored = score_candidate(pipeline_db, bundle, candidate)
    assert all("candidate" in s["bars"] for s in scored["sets"]) and calls == []
    done = wsp_import.import_candidate(pipeline_db, run_document=candidate.run, patch_manifest=candidate.manifest,
                                       edit_npz=candidate.npz)
    derived = pipeline_db.get(ProbeMonitor, done["probe_id"])
    assert derived is not None and derived.derived_from["score_call_id"] == scored["call_id"] and calls == []
