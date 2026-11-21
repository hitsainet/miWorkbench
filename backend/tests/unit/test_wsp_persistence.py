"""ENH-010 044 task 8.5 (P23): an imported candidate, RE-LOADED from disk by `load_probe`, scores the held-out rows bit
for bit as float32 exactly as the scorer scored the candidate in its run's call; the call is found by the run's id AND
the hash of the tensors miWorkbench itself applied (R1A-W8).

Negative controls (0xcc/reviews/044_implementation_controls_2026-11-16.md): the importer writes `weight` through a
float16 round trip → `test_reloaded_scores_are_bit_equal` red; the head is saved without `mean` → red; the call is
matched by run id alone → `test_wrong_call_does_not_satisfy_p23` red.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("datasets")

from src.models.probe_monitor import ProbeMonitor  # noqa: E402
from src.services import wsp_import  # noqa: E402
from tests.support.wsp_candidate import make_candidate, score_candidate  # noqa: E402
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


@pytest.fixture(scope="module")
def bundle(pipeline_db, wsp_probe, wsp_captures, tiny_tokenizer):
    from src.services.wsp_bundle import build_bundle

    row = new_bundle(pipeline_db, wsp_probe)
    build_bundle(pipeline_db, row, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(row)
    assert row.status == "ready"
    return row


def _import(db, candidate):
    return wsp_import.import_candidate(db, run_document=candidate.run, patch_manifest=candidate.manifest,
                                       edit_npz=candidate.npz)


def test_reloaded_scores_are_bit_equal(pipeline_db, bundle):
    from src.services.probe_monitor_run import load_probe

    candidate = make_candidate(bundle, entries=[3, 30], delta=0.3)
    response = score_candidate(pipeline_db, bundle, candidate)
    done = _import(pipeline_db, candidate)
    head, trained = load_probe(pipeline_db, done["probe_id"])
    sets = wsp_import.bundle_captures(pipeline_db, bundle)
    for one, entry in zip(sets, response["sets"]):
        if "scores" not in entry:
            continue
        reloaded = wsp_import._scores(head, trained, one.rows)[one.truncated == 0]
        assert reloaded.tobytes() == np.asarray(entry["scores"]["candidate"], dtype=np.float32).tobytes()
    assert pipeline_db.get(ProbeMonitor, done["probe_id"]).derived_from["score_call_id"] == response["call_id"]


def test_wrong_call_does_not_satisfy_p23(pipeline_db, bundle):
    from src.services.wsp_patch import WspTensor

    candidate = make_candidate(bundle, entries=[5], delta=0.4)
    # Two calls for ONE run: the right tensors first, then other tensors (newer).
    right = score_candidate(pipeline_db, bundle, candidate)
    other_bits = candidate.edited["weight"].bits.copy().reshape(-1)
    other_bits[6] ^= 1 << 20
    other = {**candidate.edited, "weight": WspTensor("float32", other_bits.reshape(candidate.edited["weight"].shape))}
    score_candidate(pipeline_db, bundle, candidate, tensors=other)
    done = _import(pipeline_db, candidate)
    assert pipeline_db.get(ProbeMonitor, done["probe_id"]).derived_from["score_call_id"] == right["call_id"]
    # Only the WRONG call exists for this run: nothing to reproduce.
    lonely = make_candidate(bundle, entries=[7], delta=0.4)
    score_candidate(pipeline_db, bundle, lonely, tensors={**lonely.edited, "weight": other["weight"]})
    before = pipeline_db.query(ProbeMonitor).count()
    with pytest.raises(wsp_import.ImportRefused) as refused:
        _import(pipeline_db, lonely)
    pipeline_db.rollback()
    assert refused.value.code == "persistence_reference_missing"
    assert pipeline_db.query(ProbeMonitor).count() == before
