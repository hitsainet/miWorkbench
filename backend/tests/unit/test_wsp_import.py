"""ENH-010 044 tasks 8.2 – 8.6: candidate import from files, on a REAL tiny run, real captures and a real bundle
(FTID I44-D1), with miForge-shaped documents validated against the vendored schemas (`tests/support/wsp_candidate.py`).

Controls (0xcc/reviews/044_implementation_controls_2026-11-16.md): remove the `derived_probe` exclusion in
`selection_exclusions` → `test_derived_probe_is_never_selected` red; drop the phase-2 refusal on the route →
`test_phase2_agent_caller_refused` red; hash the document WITH `run_sha256` → `test_run_sha256_mismatch_refused` red;
skip the acknowledgement check → `test_not_improved_needs_acknowledgement` red; skip `check_limits` →
`test_outside_limits_refused` red; copy the parent's window bars → `test_window_and_band_bars_are_null` red; drop the
derived rung detail → `test_heldout_reuse_recorded` red; drop the route decorator → `test_routes_in_openapi` red.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.probe_monitor import ProbeMonitor
from tests.support.wsp_world import seed_probe, sync_db  # noqa: F401 (fixture)


def test_derived_probe_is_never_selected(sync_db):
    from src.services.probe_monitor_run import select_best_probe, selection_exclusions

    world = seed_probe(sync_db, val_metrics={"val_auroc": 0.80})
    parent = sync_db.get(ProbeMonitor, world.probe_id)
    derived = ProbeMonitor(run_id=world.run_id, layer=1, rule="mean", val_metrics={"val_auroc": 0.99},
                           derived_from={"parent_probe_id": parent.id})
    sync_db.add(derived)
    sync_db.commit()
    assert selection_exclusions([parent, derived], None)[derived.id] == [{"code": "derived_probe"}]
    assert select_best_probe(sync_db, [parent, derived]).id == parent.id
    assert select_best_probe(sync_db, [derived]) is None


# ── 8.2 – 8.6 on a real run ───────────────────────────────────────────────────

pytest.importorskip("datasets")

from src.services import wsp_import  # noqa: E402
from tests.support.wsp_candidate import (  # noqa: E402
    bundle_heads,
    make_candidate,
    score_candidate,
)
from tests.support.wsp_pipeline import (  # noqa: E402,F401 (fixtures)
    arrow_dataset,
    heldout_arrow,
    monkeypatch_module,
    new_bundle,
    pb_captures,
    pb_heldout_arrow,
    pb_probe,
    pb_world,
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


@pytest.fixture(autouse=True)
def fresh_settings():
    from src.core import runtime_settings

    runtime_settings.reset()
    yield
    runtime_settings.reset()


def _import(db, candidate, **kwargs):
    return wsp_import.import_candidate(db, run_document=candidate.run, patch_manifest=candidate.manifest,
                                       edit_npz=candidate.npz, **kwargs)


def _refused(db, candidate, code, **kwargs):
    before = db.query(ProbeMonitor).count()
    with pytest.raises(wsp_import.ImportRefused) as refused:
        _import(db, candidate, **kwargs)
    db.rollback()
    assert refused.value.code == code, refused.value.message
    assert db.query(ProbeMonitor).count() == before, "a refused import registered a probe"
    return refused.value


def _scored(db, bundle, **kwargs):
    candidate = make_candidate(bundle, **kwargs)
    score_candidate(db, bundle, candidate)
    return candidate


def _derived(db, probe_id):
    db.expire_all()
    return db.get(ProbeMonitor, probe_id)


# 8.2: the ordered checks


def test_patch_base_mismatch_refused(pipeline_db, bundle):
    # The patch was made against a different head: one base entry moved before the patch was cut.
    from src.services.wsp_patch import WspTensor

    base = bundle_heads(bundle)
    moved = base["weight"].bits.copy().reshape(-1)
    moved[3] ^= 1
    other = {**base, "weight": WspTensor("float32", moved.reshape(base["weight"].bits.shape))}
    _refused(pipeline_db, make_candidate(bundle, entries=[0], base=other), "patch_base_mismatch")
    # The run names a bundle this miWorkbench does not hold.
    candidate = make_candidate(bundle, entries=[0])
    candidate.run["request"]["bundle"]["bundle_id"] = "wsb_000000000000"
    candidate.run["run_sha256"] = wsp_import.run_sha256_of(candidate.run)
    _refused(pipeline_db, candidate, "patch_base_mismatch")


def test_edited_hash_mismatch_refused(pipeline_db, bundle):
    _refused(pipeline_db, make_candidate(bundle, entries=[1], tamper_edited_sha256=True), "patch_hash_mismatch")
    # And a patch that is not the run's own.
    candidate, other = make_candidate(bundle, entries=[1]), make_candidate(bundle, entries=[2])
    candidate.manifest = other.manifest
    _refused(pipeline_db, candidate, "patch_hash_mismatch")


def test_outside_limits_refused(pipeline_db, bundle):
    refused = _refused(pipeline_db, _scored(pipeline_db, bundle, entries=[4, 5], max_changed_entries=1),
                       "patch_outside_limits")
    assert refused.extra["limit"] == 1
    _refused(pipeline_db, _scored(pipeline_db, bundle, entries=[6], frozen_tensors=["weight"]),
             "patch_outside_limits")


def test_no_change_registers_nothing(pipeline_db, bundle):
    _refused(pipeline_db, _scored(pipeline_db, bundle, entries=[7], verdict="no_change"), "nothing_to_import")
    _refused(pipeline_db, _scored(pipeline_db, bundle, entries=[8], outcome="no_change"), "nothing_to_import")


def test_not_improved_needs_acknowledgement(pipeline_db, bundle):
    candidate = _scored(pipeline_db, bundle, entries=[9], verdict="not_significant")
    refused = _refused(pipeline_db, candidate, "acknowledgement_required")
    assert refused.extra["verdict"] == "not_significant"
    # An acknowledgement of ANOTHER verdict acknowledges nothing.
    _refused(pipeline_db, candidate, "acknowledgement_required",
             acknowledgement={"verdict": "regressed", "reason": "x"})
    done = _import(pipeline_db, candidate, acknowledgement={"verdict": "not_significant", "reason": "kept for study"})
    assert _derived(pipeline_db, done["probe_id"]).derived_from["verdict"]["value"] == "not_significant"


def test_unresolved_chain_refused(pipeline_db, bundle):
    first = _scored(pipeline_db, bundle, entries=[10])
    chained = make_candidate(bundle, entries=[11], base=first.edited,
                             parent={"kind": "candidate", "patch_sha256": first.manifest["patch_sha256"],
                                     "ref": first.run["candidate_id"]},
                             parent_patch_sha256=first.manifest["patch_sha256"])
    score_candidate(pipeline_db, bundle, chained)
    # The earlier patch was never imported: refused.
    _refused(pipeline_db, chained, "patch_chain_unresolved")
    # Imported first, the chain resolves and is applied on top of it.
    _import(pipeline_db, first)
    done = _import(pipeline_db, chained)
    assert _derived(pipeline_db, done["probe_id"]).derived_from["chained_on"] is not None


def test_run_sha256_mismatch_refused(pipeline_db, bundle):
    candidate = _scored(pipeline_db, bundle, entries=[12])
    candidate.run["usage"]["requests"] = 9   # changed after the verdict was stored
    _refused(pipeline_db, candidate, "run_sha256_mismatch")


def test_run_sha256_is_miforges_definition():
    """P63/P65: RFC 8785 over the document without `run_sha256` and `program_reruns` (a re-run moves neither)."""
    from src.services.wsp_patch import jcs_sha256

    document = {"a": 1e-5, "run_sha256": "x", "program_reruns": ["wsp_1"], "b": [1, 2]}
    assert wsp_import.run_sha256_of(document) == jcs_sha256({"a": 1e-5, "b": [1, 2]})
    assert wsp_import.run_sha256_of({**document, "program_reruns": []}) == wsp_import.run_sha256_of(document)


def _verdict_set(truncated_fired):
    return {"deltas": [], "excluded_truncated": 9, "heads": [], "heldout_uses": 1, "name": "eval", "negatives": 1,
            "positives": 1, "primary": True, "role": "eval", "rows": 11, "truncated_fired": truncated_fired,
            "value": "improved"}


def _set_verdict_errors(truncated_fired):
    """The vendored PACKAGE copy (what `validate_run_document` reads in the image), at miForge's `SetVerdict`."""
    from jsonschema import Draft202012Validator

    schema = wsp_import._run_schema()
    validator = Draft202012Validator({"$defs": schema["$defs"], "$ref": "#/$defs/SetVerdict"})
    return [e.message for e in validator.iter_errors(_verdict_set(truncated_fired))]


def test_truncated_fired_is_a_map_of_every_heads_count():
    """miForge F-2 (2026-11-19): `truncated_fired` maps each head label the scorer answered to its count. The
    vendored run schema read `{label: integer}` as ONE literal key, so a miForge verdict carrying every head's count
    would have been refused at import as `run_document_invalid`."""
    every = {"parent": 1, "candidate": 2, "control": 3, "baseline:lda": 0}
    assert _set_verdict_errors(every) == []
    assert _set_verdict_errors({"label": 4}) == []            # the old shape still validates (a loosening)
    for bad in (-1, 1.5, "2", None):
        assert _set_verdict_errors({**every, "control": bad}), bad


def test_a_negative_truncated_fired_count_is_refused_at_import(pipeline_db, bundle):
    candidate = _scored(pipeline_db, bundle, entries=[12])
    candidate.run["verdict"]["per_set"][0]["truncated_fired"] = {"candidate": -1, "control": 0, "parent": 0}
    with pytest.raises(wsp_import.ImportRefused) as refused:
        wsp_import.validate_run_document(candidate.run)
    assert refused.value.code == "run_document_invalid"
    assert refused.value.extra["path"] == "$.verdict.per_set[0].truncated_fired.candidate"


# 8.3 – 8.5: the new probe, the rung, P23


def _restore_rung(db, probe_id, rung):
    db.rollback()
    db.get(ProbeMonitor, probe_id).rung = rung
    db.commit()


def test_new_probe_at_rung_zero_parent_unchanged(pipeline_db, bundle, wsp_probe, monkeypatch, request):
    from src.services import probe_monitor_run

    parent_file = Path(wsp_probe.weights_path).read_bytes()
    # The parent stands at rung 2 for this import, so a derived probe that inherited its rung could not pass for one
    # that started at 0 (the tiny run's own parent is at rung 0).
    held = pipeline_db.get(ProbeMonitor, wsp_probe.id)
    saved_rung, held.rung = held.rung, 2
    pipeline_db.commit()
    request.addfinalizer(lambda: _restore_rung(pipeline_db, wsp_probe.id, saved_rung))
    columns = [c.name for c in ProbeMonitor.__table__.columns]
    pipeline_db.expire_all()
    parent_before = {c: getattr(pipeline_db.get(ProbeMonitor, wsp_probe.id), c) for c in columns}
    created_rungs = []
    real = probe_monitor_run.recompute_rung

    def spy(db, probe_id):
        created_rungs.append(db.get(ProbeMonitor, probe_id).rung)
        return real(db, probe_id)

    monkeypatch.setattr(probe_monitor_run, "recompute_rung", spy)
    done = _import(pipeline_db, _scored(pipeline_db, bundle, entries=[13]))
    assert created_rungs == [0], "the derived probe must start at rung 0, never the parent's rung"
    derived = _derived(pipeline_db, done["probe_id"])
    assert derived.id != wsp_probe.id and derived.selected is False and derived.run_id == wsp_probe.run_id
    assert derived.derived_from["parent_probe_id"] == wsp_probe.id
    assert (derived.layer, derived.rule, derived.rule_params) == (wsp_probe.layer, wsp_probe.rule,
                                                                  wsp_probe.rule_params)
    assert Path(wsp_probe.weights_path).read_bytes() == parent_file
    pipeline_db.expire_all()
    assert {c: getattr(pipeline_db.get(ProbeMonitor, wsp_probe.id), c) for c in columns} == parent_before


def test_rung_rises_only_from_evaluations(pipeline_db, bundle):
    from src.models.probe_monitor import ProbeMonitorEvaluation
    from src.services.probe_monitor_run import recompute_rung

    done = _import(pipeline_db, _scored(pipeline_db, bundle, entries=[14]))
    evaluations = pipeline_db.query(ProbeMonitorEvaluation).filter(
        ProbeMonitorEvaluation.probe_id == done["probe_id"]).all()
    assert evaluations and {e.metrics["source"] for e in evaluations} == {"wsp_capture"}
    for evaluation in evaluations:   # the five G5 arrays, each on disk
        for column in ("scores_path", "positions_path", "n_scored_path", "labels_path", "truncated_path"):
            assert Path(getattr(evaluation, column)).is_file()
    assert done["rung"] == recompute_rung(pipeline_db, done["probe_id"])["rung"]
    for evaluation in evaluations:
        pipeline_db.delete(evaluation)
    pipeline_db.commit()
    assert recompute_rung(pipeline_db, done["probe_id"])["rung"] == 0


def test_heldout_reuse_recorded(pipeline_db, bundle):
    from src.models.wsp import WspScoreCall

    candidate = _scored(pipeline_db, bundle, entries=[15], uses=5)
    done = _import(pipeline_db, candidate)
    detail = _derived(pipeline_db, done["probe_id"]).rung_detail
    calls = pipeline_db.query(WspScoreCall).filter(WspScoreCall.bundle_id == bundle.id).count()
    sets = bundle.manifest["heldout_ref"]["sets"]
    assert detail["heldout_reuse"] == {s["name"]: {"verdict_uses": 5 + i, "score_calls": calls}
                                       for i, s in enumerate(sets)}
    assert detail["bars_note"] == wsp_import.BARS_NOTE


def test_window_and_band_bars_are_null(pipeline_db, bundle, wsp_probe):
    parent = pipeline_db.get(ProbeMonitor, wsp_probe.id)
    saved = (parent.window_decisions, parent.length_bands)
    parent.window_decisions = {"last_16": {"threshold": 1.0}}
    parent.length_bands = [{"upto": 10, "threshold": 1.0}]
    pipeline_db.commit()
    try:
        done = _import(pipeline_db, _scored(pipeline_db, bundle, entries=[16]))
    finally:
        parent = pipeline_db.get(ProbeMonitor, wsp_probe.id)
        parent.window_decisions, parent.length_bands = saved
        pipeline_db.commit()
    derived = _derived(pipeline_db, done["probe_id"])
    assert derived.window_decisions is None and derived.length_bands is None
    assert derived.threshold is not None and derived.threshold_source == wsp_probe.threshold_source


def test_capture_bar_matches_the_scorers_candidate_bar(pipeline_db, bundle):
    """The derived probe's global bar is the bar the scorer cut for the candidate on the same calibration rows."""
    candidate = make_candidate(bundle, entries=[17])
    response = score_candidate(pipeline_db, bundle, candidate)
    done = _import(pipeline_db, candidate)
    bars = {s["bars"]["candidate"]["threshold"] for s in response["sets"]}
    assert len(bars) == 1 and _derived(pipeline_db, done["probe_id"]).threshold == bars.pop()


# R1 review fixes (OR1-044a-1, -2, -6), 2026-11-16


def test_persistence_mismatch_refuses_and_leaves_nothing(pipeline_db, bundle, monkeypatch):
    """OR1-044a-2: P23 is wired, not only correct. The head as WRITTEN differs from the scored candidate by one bit;
    the import refuses `persistence_mismatch`, registers no probe and leaves no file. Control (A13): drop the
    `persistence_check` call → red."""
    from src.services.probe_monitor_run import artifact_dir_for

    candidate = _scored(pipeline_db, bundle, entries=[21])
    real = wsp_import.write_head

    def flipped(path, arrays):
        # ONE bit: the sign of the first non-zero entry of the scoring weight, which every row's score reads.
        bad = {**arrays, "weight": arrays["weight"].copy()}
        flat = bad["weight"].reshape(-1)
        index = int(np.flatnonzero(flat)[0])
        flat[index] = -flat[index]
        return real(path, bad)

    monkeypatch.setattr(wsp_import, "write_head", flipped)
    directory = artifact_dir_for(pipeline_db.get(ProbeMonitor, bundle.probe_id).run_id)
    before = set(directory.glob("*.safetensors"))
    _refused(pipeline_db, candidate, "persistence_mismatch")
    assert set(directory.glob("*.safetensors")) == before, "the refused import left its head on disk"


def test_run_not_evaluated_is_its_own_refusal(pipeline_db, bundle):
    """OR1-044a-6 / OVN-34: a run with no verdict is refused `run_not_evaluated` (nothing to check P65 against), not
    later as `run_sha256_mismatch` with the wrong next step. Control (A14): drop the check → red."""
    candidate = _scored(pipeline_db, bundle, entries=[22])
    import dataclasses

    candidate = dataclasses.replace(candidate, run={**candidate.run, "verdict": None})
    _refused(pipeline_db, candidate, "run_not_evaluated")


@pytest.mark.parametrize("kind", ["bundle_zeroed", "checkpoint"])
def test_undefined_patch_parents_are_refused(pipeline_db, bundle, kind):
    """OR1-044a-6 / OVN-37: a `bundle_zeroed` or `checkpoint` parent has no defined base here (OPERATOR 13); refused
    `patch_base_mismatch`, even with this bundle's own ref. Control (A10): accept them as `bundle` → red."""
    candidate = _scored(pipeline_db, bundle, entries=[23],
                        parent={"kind": kind, "patch_sha256": None, "ref": bundle.bundle_sha256})
    _refused(pipeline_db, candidate, "patch_base_mismatch")


def test_a_calibration_basis_import_without_its_calibration_set_is_refused(pipeline_db, bundle):
    """OR1-044a-1 (import half): the derived probe's bar is never re-based on practice rows. With the calibration set
    gone from the bundle the import refuses `capture_not_ready`. Control: restore the practice fallback → red."""
    import copy

    candidate = _scored(pipeline_db, bundle, entries=[24])
    original = copy.deepcopy(bundle.manifest)
    trimmed = copy.deepcopy(original)
    trimmed["heldout_ref"]["sets"] = [s for s in trimmed["heldout_ref"]["sets"] if not s["name"].startswith("calibration")]
    bundle.manifest = trimmed
    pipeline_db.commit()
    try:
        _refused(pipeline_db, candidate, "capture_not_ready")
    finally:
        bundle.manifest = original
        pipeline_db.commit()


# 8.6: the route


@pytest.fixture
def route_session(pipeline_db, monkeypatch):
    from sqlalchemy.orm import sessionmaker

    from src.api.v1.endpoints import wsp as wsp_routes

    make = sessionmaker(bind=pipeline_db.get_bind())
    monkeypatch.setattr(wsp_routes, "_session", make)
    return make


@pytest.mark.asyncio
async def test_an_agent_caller_needs_an_approval(client, route_session, pipeline_db, bundle):
    """044 FTASKS 14.2 replaced phase 2's refusal of every agent (`agent_route_not_yet_gated`) with route-side
    consumption: a real candidate's body from the MCP pod without an approved id is 403 `approval_required`. The
    gate's every shape is in `test_wsp_import_gate.py`."""
    from src.services.probe_dataset_lifecycle import AGENT_HEADER

    body = _scored(pipeline_db, bundle, entries=[18]).body()
    response = await client.post("/api/v1/wsp/candidates/import", json=body, headers={AGENT_HEADER: "mcp"})
    assert response.status_code == 403 and response.json()["detail"]["code"] == "approval_required"


@pytest.mark.asyncio
async def test_route_checks_then_queues_one_import(client, route_session, pipeline_db, bundle, monkeypatch):
    """The operator's call is checked synchronously (a refusal answers its code) and queues exactly one import task
    carrying the body it checked; the task, run, imports it."""
    from src.workers import wsp_tasks

    sent = []
    monkeypatch.setattr(wsp_tasks.import_wsp_candidate, "delay",
                        lambda **kwargs: sent.append(kwargs) or type("T", (), {"id": "task-1"})())
    refused = make_candidate(bundle, entries=[19], verdict="regressed")
    score_candidate(pipeline_db, bundle, refused)
    response = await client.post("/api/v1/wsp/candidates/import", json=refused.body())
    assert response.status_code == 409 and response.json()["detail"]["code"] == "acknowledgement_required"
    assert sent == []
    candidate = _scored(pipeline_db, bundle, entries=[20])
    response = await client.post("/api/v1/wsp/candidates/import", json=candidate.body())
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "importing" and response.json()["task_id"] == "task-1"
    assert len(sent) == 1
    assert sent[0] == {"run_document": candidate.run, "patch_manifest": candidate.manifest,
                       "edit_npz_b64": candidate.npz_b64, "acknowledgement": None}
    monkeypatch.setattr(wsp_tasks.import_wsp_candidate, "get_db", lambda: _Ctx(route_session()), raising=False)
    result = wsp_tasks.import_wsp_candidate.run(**sent[0])
    assert result.get("probe_id"), result
    assert _derived(pipeline_db, result["probe_id"]).derived_from["miforge_run_id"] == candidate.run["run_id"]


class _Ctx:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, *exc):
        self.session.close()


@pytest.mark.asyncio
async def test_import_body_is_json_under_the_cap(client):
    from src.main import app

    content = app.openapi()["paths"]["/api/v1/wsp/candidates/import"]["post"]["requestBody"]["content"]
    assert set(content) == {"application/json"}
    big = {"run_document": {}, "patch_manifest": {}, "edit_npz_b64": "A" * (1_100_000)}
    response = await client.post("/api/v1/wsp/candidates/import", json=big)
    assert response.status_code == 413



# ── R2 review fixes (OR2-044a-M3, -L5), 2026-11-17: the practice basis through the import ──────────────────────────


@pytest.fixture(scope="module")
def pb_bundle(pipeline_db, pb_probe, pb_captures, tiny_tokenizer):
    from src.services.wsp_bundle import build_bundle

    row = new_bundle(pipeline_db, pb_probe)
    build_bundle(pipeline_db, row, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(row)
    assert row.status == "ready" and row.manifest["operating_point"]["threshold_source"] == "practice", row.error
    return row


def _practice_rows(bundle, *, validation_only):
    directory = Path(bundle.dir_path)
    practice = bundle.manifest["practice"]
    offsets = np.load(directory / "practice_offsets.npy")
    labels = np.load(directory / "practice_labels.npy")
    validation = np.load(directory / "practice_validation_mask.npy").astype(bool)
    truncated = np.load(directory / "practice_truncated_mask.npy").astype(bool)
    name = next(f["name"] for f in bundle.manifest["files"] if f["role"] == "practice_activations")
    memmap = np.memmap(directory.parents[2] / name, dtype=np.dtype(practice["dtype"]), mode="r",
                       shape=(int(practice["tokens"]), int(practice["d_model"])))
    mask = (labels == 0) & ((validation & ~truncated) if validation_only else True)
    return [memmap[int(offsets[i]):int(offsets[i + 1])] for i in np.flatnonzero(mask)]


def test_a_practice_basis_import_cuts_on_the_practice_validation_negatives(pipeline_db, pb_bundle, pb_probe):
    """The import's half of the practice basis (`wsp_import.calibrate_from_capture`, review mutations M19 and M13):
    the derived probe's bar is cut — as its parent's was — on the practice VALIDATION negatives that were not
    truncated, computed here from the bundle's arrays and the derived probe as re-loaded; and it is the bar the scorer
    cut for the candidate. Control: cut on every practice negative → red."""
    from src.services.probe_monitor_run import _aggregate_rows, load_probe
    from src.services.probe_monitor_trainer import calibrate

    candidate = make_candidate(pb_bundle, entries=[9, 19], delta=0.4)
    response = score_candidate(pipeline_db, pb_bundle, candidate)
    done = _import(pipeline_db, candidate)
    derived = _derived(pipeline_db, done["probe_id"])
    head, trained = load_probe(pipeline_db, derived.id)
    params = dict(trained.rule_params or {})

    def bar(rows):
        scores = np.asarray(_aggregate_rows(head, trained.rule, params, rows), dtype=np.float32)
        return calibrate([float(v) for v in scores], target_fpr=float(pb_probe.target_fpr), source="x").threshold

    expected = bar(_practice_rows(pb_bundle, validation_only=True))
    assert expected is not None and expected != bar(_practice_rows(pb_bundle, validation_only=False)), \
        "fixture: the two rules must place different bars"
    assert (derived.threshold_source, derived.threshold) == ("validation_negatives", expected)
    assert {s["bars"]["candidate"]["threshold"] for s in response["sets"]} == {expected}


def test_an_import_without_targets_cuts_at_the_parents(pipeline_db, pb_bundle, pb_probe):
    """OR2-044a-L5's import half: with no target on the derived probe and no operating point in the bundle, the bar is
    cut at the PARENT's target FPR (0.25 here), never a constant 0.01. Control: restore the 0.01 fallback → red."""
    import copy
    from types import SimpleNamespace

    from src.services.probe_monitor_run import load_probe

    assert float(pb_probe.target_fpr) == 0.25, "fixture: a target other than the old fallback"
    head, trained = load_probe(pipeline_db, pb_probe.id)
    sets = wsp_import.bundle_captures(pipeline_db, pb_bundle)
    original = copy.deepcopy(pb_bundle.manifest)
    pb_bundle.manifest = {**copy.deepcopy(original), "operating_point": None}
    pipeline_db.commit()
    stand_in = SimpleNamespace(id="pm_l5standin0000", run_id=pb_probe.run_id, target_fpr=None)
    try:
        wsp_import.calibrate_from_capture(pipeline_db, stand_in, pb_bundle, sets, head, trained)
    finally:
        pb_bundle.manifest = original
        pipeline_db.commit()
    assert (stand_in.target_fpr, stand_in.threshold_source) == (0.25, "validation_negatives")
