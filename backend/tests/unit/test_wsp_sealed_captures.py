"""R4 review fix (C044a-M1), 2026-11-17: a bundle is scored only on the captures it was sealed on.

Round 3 (OR3-044a-L3) re-armed a stale READY calibration capture IN PLACE. Every earlier bundle names its captures by
id, so a recalibration followed by a capture request wiped the parity record under every earlier calibration-basis
bundle (unscorable while the new capture ran, for good when it failed parity) and, after a G5 re-cut, let the bundle be
scored on rows its manifest does not name. Now:

* a stale ready calibration capture is SUPERSEDED by a new row (a new directory); the old row keeps its status, parity
  and files, and every bundle naming it keeps scoring on it (`plan_capture_rows`);
* a bundle records the parity digest of each capture it was built on (`wsp_bundles.capture_seals`), and the scorer and
  the import refuse `capture_changed` (409) when a capture's rows, row ids or parity record differ from that seal and
  the manifest's `heldout_ref` (`wsp_bundle.capture_seal_broken`);
* a capture superseded or gone stale while a bundle is built refuses the bundle `capture_not_ready`.

One real tiny run, module scope (its own database). Each test leaves a READY current calibration capture at the
probe's original bar, so the tests do not depend on one another's order.

Controls (`0xcc/reviews/044a_r4_review_fixes_2026-11-17.md`): reset the stale row in place again →
`test_an_earlier_bundle_keeps_scoring_on_its_own_capture_after_a_recapture` red; drop the scorer's seal check →
`test_a_capture_that_differs_from_the_seal_is_refused_capture_changed` red (and the P1 test red); drop the import's →
the same test red at the import; drop the end-of-build check → `test_a_capture_superseded_during_the_build_refuses_it`
red; offer superseded captures to a new bundle → the P1 test red.
"""
from __future__ import annotations

import json

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

ARCH = "LlamaForCausalLM"


def _request(bundle):
    from src.schemas.wsp_scoring import WspScoreRequest

    body = {"kind": "miworkbench.wsp-score-request/v1", "bundle_id": bundle.id, "bundle_sha256": bundle.bundle_sha256,
            "run_ref": {"miforge_run_id": "wsr_test", "patch_sha256": "0" * 64, "max_changed_entries": None},
            "heldout_ref": json.loads(json.dumps(bundle.manifest["heldout_ref"])),
            "heads": [{"label": "parent", "tensors": None}], "primary_sets": []}
    return WspScoreRequest.model_validate(body)


@pytest.fixture
def loaded(monkeypatch):
    """The capture ids the scorer actually reads rows from, per call."""
    from src.services import wsp_score as ws

    calls = []
    real = ws.load_set

    def spy(db, capture, name):
        if calls and calls[-1] is not None:   # a score call open (a build's `scorer_parent_bar` also reads sets)
            calls[-1].append(capture.id)
        return real(db, capture, name)

    monkeypatch.setattr(ws, "load_set", spy)
    return calls


def _score(db, bundle, loaded):
    from src.services import wsp_score as ws

    loaded.append([])
    try:
        answer = ws.score(db, _request(bundle))
        db.commit()
        return answer["sets"], loaded[-1]
    finally:
        loaded.append(None)


def _refused(db, bundle, code):
    from src.services import wsp_score as ws

    with pytest.raises(ws.ScoreRefused) as refused:
        ws.score(db, _request(bundle))
    db.rollback()
    assert refused.value.code == code, (refused.value.code, refused.value.message)
    return refused.value


def _build(db, probe, tokenizer):
    from src.services.wsp_bundle import build_bundle

    bundle = new_bundle(db, probe)
    build_bundle(db, bundle, tokenizer=tokenizer, max_seconds=120)
    db.refresh(bundle)
    assert bundle.status == "ready", bundle.error
    return bundle


def _current_calibration(db, probe):
    from src.models.wsp import WspActivationCapture

    return (db.query(WspActivationCapture)
            .filter(WspActivationCapture.probe_id == probe.id, WspActivationCapture.role == "calibration",
                    WspActivationCapture.superseded_at.is_(None)).one())


def _capture(db, probe, row, lm, tokenizer):
    from src.services.wsp_capture import capture_one, plan_capture

    planned = {(p.view_id, p.role): p for p in plan_capture(db, probe)}
    capture_one(db, row, planned[(row.dataset_id, row.role)], model=lm, tokenizer=tokenizer, architecture=ARCH)
    db.refresh(row)
    return row


def test_an_earlier_bundle_keeps_scoring_on_its_own_capture_after_a_recapture(pipeline_db, wsp_probe, wsp_captures,
                                                                              tiny_lm, tiny_tokenizer, loaded):
    """The review's probe P1, made a regression. Bundle B1 is built on calibration capture C and scored. The probe is
    recalibrated (its bar moves) and a capture is requested: C is superseded by a NEW row N, never reset, and B1 still
    scores on C, the same answer. N is captured against the moved bar and fails parity: B1 still scores on C. The bar
    returns, N is captured again and passes, and a new bundle B2 is built on N while B1 still reads C.
    Control: reset C in place (round 3) → B1 refused `capture_changed` (seal) or `capture_not_ready` → red."""
    from src.api.v1.endpoints.wsp import plan_capture_rows
    from src.models.wsp import WspActivationCapture
    from src.services.wsp_bundle import ready_captures
    from src.services.wsp_capture import calibration_capture_stale

    b1 = _build(pipeline_db, wsp_probe, tiny_tokenizer)
    assert b1.manifest["operating_point"]["threshold_source"] == "calibration"
    c = _current_calibration(pipeline_db, wsp_probe)
    assert c.id in b1.capture_ids and b1.capture_seals and c.id in b1.capture_seals
    sealed = (c.status, json.dumps(c.parity, sort_keys=True), c.dir_path, c.rows, c.row_ids_sha256)
    first, read = _score(pipeline_db, b1, loaded)
    assert c.id in read
    original = float(wsp_probe.threshold)
    try:
        wsp_probe.threshold = original + 0.05
        pipeline_db.commit()
        planned = plan_capture_rows(pipeline_db, wsp_probe.id, None)
        pipeline_db.commit()
        n = _current_calibration(pipeline_db, wsp_probe)
        assert n.id != c.id and planned["to_run"] == [n.id] and n.status == "pending"
        pipeline_db.refresh(c)
        assert (c.status, json.dumps(c.parity, sort_keys=True), c.dir_path, c.rows, c.row_ids_sha256) == sealed
        assert c.superseded_at is not None and c.superseded_by == n.id
        assert (pipeline_db.query(WspActivationCapture)
                .filter(WspActivationCapture.probe_id == wsp_probe.id, WspActivationCapture.role == "calibration")
                .count()) == 2

        again, read = _score(pipeline_db, b1, loaded)
        assert again == first and c.id in read and n.id not in read

        n = _capture(pipeline_db, wsp_probe, n, tiny_lm, tiny_tokenizer)
        assert n.status == "parity_failed" and n.dir_path != c.dir_path        # its own directory
        again, read = _score(pipeline_db, b1, loaded)
        assert again == first and c.id in read and n.id not in read

        # The bar returns; C describes it again but stays superseded: never offered to a new bundle (N is current).
        wsp_probe.threshold = original
        pipeline_db.commit()
        assert calibration_capture_stale(wsp_probe, c) is None
        assert c.id not in [x.id for x in ready_captures(pipeline_db, wsp_probe)]
        # N (failed, never bundled) is re-armed and passes; new bundles are built on it.
        planned = plan_capture_rows(pipeline_db, wsp_probe.id, None)
        pipeline_db.commit()
        assert planned["to_run"] == [n.id]
        n = _capture(pipeline_db, wsp_probe, n, tiny_lm, tiny_tokenizer)
        assert n.status == "ready", n.parity
        b2 = _build(pipeline_db, wsp_probe, tiny_tokenizer)
        assert n.id in b2.capture_ids and c.id not in b2.capture_ids
        _sets, read = _score(pipeline_db, b2, loaded)
        assert n.id in read and c.id not in read
        again, read = _score(pipeline_db, b1, loaded)
        assert again == first and c.id in read and n.id not in read
    finally:
        pipeline_db.rollback()
        wsp_probe.threshold = original
        pipeline_db.commit()


def test_a_capture_that_differs_from_the_seal_is_refused_capture_changed(pipeline_db, wsp_probe, wsp_captures,
                                                                          tiny_tokenizer, loaded):
    """The scorer and the import read only the captures a bundle was sealed on: a capture whose parity record, rows or
    row ids differ from the bundle's seal and its manifest's `heldout_ref`, or a bundle with no seal, is refused
    `capture_changed` (409), never scored. Restored, the bundle scores again.
    Controls: drop the scorer's check → red; drop the import's check → red."""
    from src.schemas.wsp_scoring import refusal_status
    from src.services import wsp_import
    from src.services.wsp_bundle import capture_seal_broken

    bundle = _build(pipeline_db, wsp_probe, tiny_tokenizer)
    _score(pipeline_db, bundle, loaded)
    from src.models.wsp import WspActivationCapture

    # confirm-044a4 L1 (X06): EVERY sealed capture is checked, the calibration capture included (the row M1
    # supersedes), not only the eval set.
    roles = sorted({c.role for c in pipeline_db.query(WspActivationCapture)
                    .filter(WspActivationCapture.id.in_(bundle.capture_ids))})
    assert roles == ["calibration", "eval"], f"fixture: a calibration-basis bundle seals both roles, got {roles}"
    for role in roles:
        capture = (pipeline_db.query(WspActivationCapture)
                   .filter(WspActivationCapture.id.in_(bundle.capture_ids), WspActivationCapture.role == role).one())
        entry = next(e for e in bundle.manifest["heldout_ref"]["sets"] if e["view_id"] == capture.dataset_id
                     and e["role"] == role)
        assert capture_seal_broken(bundle, entry, capture) is None
        keep = (dict(capture.parity), capture.rows, capture.row_ids_sha256, dict(bundle.capture_seals))
        tampers = {
            "parity": lambda: setattr(capture, "parity", {**keep[0], "auroc_capture": 0.123}),
            # confirm-044a4 X14: every field of the parity record is sealed, the served bar included.
            "threshold_served": lambda: setattr(capture, "parity", {**keep[0], "threshold_served": 1234.5}),
            "rows": lambda: setattr(capture, "rows", keep[1] + 1),
            "row_ids": lambda: setattr(capture, "row_ids_sha256", "f" * 64),
            "no_seal": lambda: setattr(bundle, "capture_seals", None),
            "other_seal": lambda: setattr(bundle, "capture_seals", {**keep[3], capture.id: "0" * 64}),
        }
        assert refusal_status("capture_changed") == 409
        for name, tamper in tampers.items():
            tamper()
            pipeline_db.commit()
            try:
                refused = _refused(pipeline_db, bundle, "capture_changed")
                if name != "no_seal":   # with no seal at all, the first set is the one named
                    assert entry["name"] in refused.message or capture.id in refused.message, (name, refused.message)
                with pytest.raises(wsp_import.ImportRefused) as at_import:
                    wsp_import.bundle_captures(pipeline_db, bundle)
                assert at_import.value.code == "capture_changed", (name, at_import.value.code)
            finally:
                pipeline_db.rollback()
                capture.parity, capture.rows, capture.row_ids_sha256 = keep[0], keep[1], keep[2]
                bundle.capture_seals = keep[3]
                pipeline_db.commit()
    _sets, read = _score(pipeline_db, bundle, loaded)
    assert capture.id in read


def test_a_capture_superseded_during_the_build_refuses_it(pipeline_db, wsp_probe, wsp_captures, tiny_lm,
                                                          tiny_tokenizer, monkeypatch):
    """A recalibration and a capture request while a bundle builds: the captures read at the start no longer describe
    the bar the manifest states, so the bundle is refused `capture_not_ready`, never `ready` with an operating point
    no capture was checked against. Control: drop the end-of-build check → the bundle turns `ready` → red."""
    from sqlalchemy.orm import Session

    from src.api.v1.endpoints.wsp import plan_capture_rows
    from src.models.probe_monitor import ProbeMonitor
    from src.services import wsp_bundle

    original = float(wsp_probe.threshold)
    real = wsp_bundle.build_manifest

    def recalibrated_meanwhile(**kwargs):
        with Session(bind=pipeline_db.get_bind()) as other:   # another request, another session
            other.get(ProbeMonitor, wsp_probe.id).threshold = original + 0.05
            other.commit()
            plan_capture_rows(other, wsp_probe.id, None)
        return real(**kwargs)

    monkeypatch.setattr(wsp_bundle, "build_manifest", recalibrated_meanwhile)
    bundle = new_bundle(pipeline_db, wsp_probe)
    try:
        result = wsp_bundle.build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer, max_seconds=120)
        pipeline_db.refresh(bundle)
        assert result.get("refused", {}).get("code") == "capture_not_ready", result
        assert bundle.status == "failed" and "superseded or went stale" in bundle.error
    finally:
        monkeypatch.setattr(wsp_bundle, "build_manifest", real)
        pipeline_db.rollback()
        pipeline_db.refresh(wsp_probe)
        wsp_probe.threshold = original
        pipeline_db.commit()
        current = _current_calibration(pipeline_db, wsp_probe)
        if current.status != "ready":
            assert _capture(pipeline_db, wsp_probe, current, tiny_lm, tiny_tokenizer).status == "ready"
