"""ENH-010 044 tasks 5.1, 5.2, 6.1 (storage), 6.5, 6.6: a bundle built for real from a real tiny run with real
captures (FTID I44-D1), its manifest in exactly miForge's shape, and its baselines fitted on practice only.

The FTASKS name several of these tests in `test_wsp_bundle.py`; they live here because they share one real run
(module scope) and `test_wsp_bundle.py` patches `settings.data_dir` per test (OVN-wsp-workbench-29). Same names.

Controls (0xcc/reviews/044_implementation_controls_2026-11-16.md): drop the size check → misaligned build passes →
`test_alignment_mismatch_refused` red; write `temperature` for `tau` → `test_params_keys_equal_the_schema` red; a file
name with `/` → `test_manifest_validates_against_the_vendored_schema` red; hash the manifest WITH its digest →
`test_bundle_sha256_excludes_itself` red; remove a registry entry → `test_every_kind_is_fitted_and_stored` red; fit LDA
on a capture → `test_no_heldout_row_reaches_a_fit` red; drop the fit timeout → `test_slow_fit_reads_pending_not_missing`
red.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("datasets")

from tests.support.wsp_pipeline import (  # noqa: E402,F401 (fixtures)
    arrow_dataset,
    cal_overlap_arrow,
    co_captures,
    co_probe,
    co_world,
    heldout_arrow,
    monkeypatch_module,
    new_bundle,
    ov_captures,
    ov_probe,
    ov_world,
    overlap_arrow,
    pipeline_db,
    tiny_lm,
    tiny_tokenizer,
    wsp_captures,
    wsp_data_dir,
    wsp_probe,
    wsp_world,
)


@pytest.fixture(scope="module")
def built(pipeline_db, wsp_probe, wsp_captures, tiny_tokenizer):
    from src.services.wsp_bundle import build_bundle

    bundle = new_bundle(pipeline_db, wsp_probe)
    result = build_bundle(pipeline_db, bundle, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(bundle)
    return bundle, result


def test_manifest_validates_against_the_vendored_schema(built):
    from src.services.wsp_bundle import schema_errors

    bundle, result = built
    assert bundle.status == "ready" and result["status"] == "ready"
    assert schema_errors(bundle.manifest) == []
    assert all("/" not in f["name"] for f in bundle.manifest["files"])
    on_disk = json.loads((Path(bundle.dir_path) / "manifest.json").read_text())
    assert on_disk == bundle.manifest


def test_manifest_validates_against_miforges_published_schema(built):
    """The producer's own file (cross-repo; loud skip without miForge; fails under the switch)."""
    from jsonschema.validators import validator_for

    from tests.support.cross_repo import REQUIRE_NEW, require_cross_repo
    from tests.support.sibling_repos import miforge_repo

    published = miforge_repo() / "docs" / "schemas" / "miforge-wsp-bundle-v1.json"
    if not published.is_file():
        message = (f"miForge's bundle schema is not at {published} (set MIFORGE_REPO). Skipping is expected on the "
                   f"public mirror; set {REQUIRE_NEW}=1 to make this a failure.")
        if require_cross_repo():
            pytest.fail(message)
        pytest.skip(message)
    schema = json.loads(published.read_text())
    errors = list(validator_for(schema)(schema).iter_errors(built[0].manifest))
    assert errors == []


def test_bundle_sha256_excludes_itself(built):
    import hashlib

    import rfc8785

    manifest = built[0].manifest
    without = {k: v for k, v in manifest.items() if k != "bundle_sha256"}
    assert manifest["bundle_sha256"] == hashlib.sha256(rfc8785.dumps(without)).hexdigest()
    assert built[0].bundle_sha256 == manifest["bundle_sha256"] and "bundle_id" in without


def test_jcs_writes_1e_5_as_0_00001():
    """R1B-X9: `json.dumps` writes 1e-05; RFC 8785 writes 0.00001. The digest uses the latter only."""
    import rfc8785

    assert json.dumps({"x": 1e-5}, separators=(",", ":")) == '{"x":1e-05}'
    assert bytes(rfc8785.dumps({"x": 1e-5})) == b'{"x":0.00001}'


def test_params_keys_equal_the_schema(built):
    from src.services.wsp_bundle import load_schema

    wanted = set(load_schema()["$defs"]["LinearProbeFn"]["properties"]["params"]["required"])
    assert set(built[0].manifest["function"]["linear_probe"]["params"]) == wanted == {"tau", "window"}


def test_mean_and_std_not_editable(built):
    tensors = {t["name"]: t for t in built[0].manifest["tensors"]}
    assert tensors["mean"]["editable"] is False and tensors["std"]["editable"] is False
    assert tensors["weight"]["editable"] is True and tensors["bias"]["editable"] is True
    assert (tensors["bias"]["dtype"], tensors["bias"]["shape"]) == ("float64", [1])


def test_the_tensor_digests_are_the_heads_file(built):
    from safetensors.numpy import load_file

    from src.services.wsp_patch import tensor_sha256

    heads = load_file(str(Path(built[0].dir_path) / "heads.safetensors"))
    assert {t["name"]: t["sha256"] for t in built[0].manifest["tensors"]} == {
        name: tensor_sha256(array) for name, array in heads.items()}


def test_practice_rows_reproduce_the_run_split(pipeline_db, built, wsp_probe):
    """The practice validation rows, read from the run's token file through the bundle's arrays, reproduce the
    probe's own recorded validation AUROC (11.7's D128 check, on the tiny run)."""
    from src.services.probe_monitor_metrics import auroc
    from src.services.probe_monitor_run import _aggregate_rows, load_probe

    bundle = built[0]
    directory = Path(bundle.dir_path)
    offsets = np.load(directory / "practice_offsets.npy")
    labels = np.load(directory / "practice_labels.npy")
    validation = np.load(directory / "practice_validation_mask.npy").astype(bool)
    practice = bundle.manifest["practice"]
    tokens = directory.parents[2] / next(f["name"] for f in bundle.manifest["files"]
                                         if f["role"] == "practice_activations")
    memmap = np.memmap(tokens, dtype=np.dtype(practice["dtype"]), mode="r",
                       shape=(practice["tokens"], practice["d_model"]))
    rows = [memmap[int(offsets[i]):int(offsets[i + 1])] for i in np.flatnonzero(validation)]
    head, trained = load_probe(pipeline_db, wsp_probe.id)
    scores = _aggregate_rows(head, trained.rule, dict(trained.rule_params or {}), rows)
    assert practice["validation_rows"] == int(validation.sum()) > 0
    assert abs(auroc(scores, labels[validation].tolist()) - wsp_probe.val_metrics["val_auroc"]) <= 1e-4


def test_function_parity_rows_reproduce(pipeline_db, built, wsp_probe):
    from src.services.probe_monitor_run import _aggregate_rows, load_probe

    bundle = built[0]
    directory = Path(bundle.dir_path)
    parity = bundle.manifest["function"]["parity"]
    offsets = np.load(directory / "practice_offsets.npy")
    practice = bundle.manifest["practice"]
    tokens = directory.parents[2] / next(f["name"] for f in bundle.manifest["files"]
                                         if f["role"] == "practice_activations")
    memmap = np.memmap(tokens, dtype=np.dtype(practice["dtype"]), mode="r",
                       shape=(practice["tokens"], practice["d_model"]))
    head, trained = load_probe(pipeline_db, wsp_probe.id)
    scores = _aggregate_rows(head, trained.rule, dict(trained.rule_params or {}),
                             [memmap[int(offsets[i]):int(offsets[i + 1])] for i in parity["rows"]])
    assert len(parity["rows"]) == 16 and np.allclose(scores, parity["scores"], atol=parity["tolerance"], rtol=0)


def test_no_capture_file_is_in_the_file_list(built, wsp_captures):
    names = {f["name"] for f in built[0].manifest["files"]}
    for capture in wsp_captures:
        assert not names & {p.name for p in Path(capture.dir_path).iterdir()}, capture.id
    assert {f["role"] for f in built[0].manifest["files"]} == {
        "heads", "practice_activations", "practice_offsets", "practice_positions", "practice_labels",
        "practice_validation_mask", "practice_truncated_mask"}


def test_heldout_ref_names_every_ready_capture(built, wsp_captures):
    sets = built[0].manifest["heldout_ref"]["sets"]
    assert sorted((s["view_id"], s["role"], s["rows"], s["row_ids_sha256"]) for s in sets) == sorted(
        (c.dataset_id, c.role, c.rows, c.row_ids_sha256) for c in wsp_captures)


def test_alignment_mismatch_refused(pipeline_db, wsp_probe, wsp_captures, tiny_tokenizer):
    """A token file one row's worth too long (the render drifted) refuses before anything is produced."""
    from src.services.wsp_bundle import build_bundle_row, practice_tokens_file
    from src.services.probe_monitor_run import artifact_dir_for

    path, dtype = practice_tokens_file(artifact_dir_for(wsp_probe.run_id), wsp_probe.layer)
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\0" * (np.dtype(dtype).itemsize * 32))
        bundle = new_bundle(pipeline_db, wsp_probe)
        result = build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer, max_seconds=60)
    finally:
        path.write_bytes(original)
    pipeline_db.refresh(bundle)
    assert result["refused"]["code"] == "practice_capture_mismatch"
    assert bundle.status == "failed" and bundle.error.startswith("practice_capture_mismatch") and not bundle.manifest


def test_sae_probe_refused(pipeline_db, wsp_probe, tiny_tokenizer):
    from src.services.wsp_bundle import build_bundle_row

    saved, wsp_probe.variant = wsp_probe.variant, "sae"
    pipeline_db.commit()
    try:
        bundle = new_bundle(pipeline_db, wsp_probe)
        result = build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer)
    finally:
        wsp_probe.variant = saved
        pipeline_db.commit()
    assert result["refused"]["code"] == "bundle_kind_unsupported"


def test_a_calibration_set_probe_without_its_calibration_capture_is_refused(pipeline_db, wsp_probe, wsp_captures,
                                                                              tiny_tokenizer):
    """R1 review fix (OR1-044a-1), 2026-11-16 — the reviewer's scenario: the pipeline probe is calibrated on its
    calibration set; with that capture `parity_failed`, the build used to return `ready` (heldout `['eval']`, bars
    on practice rows) under a manifest saying `threshold_source: calibration`. It is refused `capture_not_ready`,
    naming the calibration set. Twin: with the capture ready, the same build succeeds (`built`). Control: drop the
    check → red."""
    from src.services.wsp_bundle import build_bundle_row

    assert wsp_probe.threshold_source == "calibration_set", "fixture: the twin needs a calibration-set probe"
    calibration = next(c for c in wsp_captures if c.role == "calibration")
    saved, calibration.status = calibration.status, "parity_failed"
    pipeline_db.commit()
    try:
        bundle = new_bundle(pipeline_db, wsp_probe)
        result = build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer)
    finally:
        calibration.status = saved
        pipeline_db.commit()
    assert result["refused"]["code"] == "capture_not_ready", result
    assert "calibration set" in result["refused"]["message"]


# ── baselines (6.1 storage, 6.5, 6.6) ─────────────────────────────────────────


def test_every_kind_is_fitted_and_stored(pipeline_db, built):
    from src.models.wsp import WspBaseline
    from src.services.wsp_baselines import BASELINE_KINDS

    bundle, result = built
    rows = {r.kind: r for r in pipeline_db.query(WspBaseline).filter(WspBaseline.bundle_id == bundle.id).all()}
    assert set(rows) == set(BASELINE_KINDS) == set(result["baselines"])
    fitted = [k for k, state in result["baselines"].items() if state == "fitted"]
    assert bundle.manifest["baselines_available"] == fitted
    for kind, spec in BASELINE_KINDS.items():
        row = rows[kind]
        if result["baselines"][kind] != "fitted":
            continue
        if spec.stores_weights:
            path = Path(row.weights_path)
            assert path.is_file() and path.parent == Path(bundle.dir_path) / "baselines", kind
        else:
            assert row.weights_path is None and row.bar is not None, kind
    assert {"logistic_regression", "lda", "diff_of_means", "threshold_recut"} <= set(fitted)


def test_no_heldout_row_reaches_a_fit(pipeline_db, wsp_probe, wsp_captures, tiny_tokenizer, monkeypatch):
    """A spy on every registered fit: each sees exactly the practice rows (by content), never a capture row."""
    from dataclasses import replace

    from src.services import wsp_baselines
    from src.services.wsp_bundle import build_bundle

    seen = {}

    def spy(kind, fit):
        def wrapped(parent, practice, **kwargs):
            seen[kind] = {np.asarray(r, dtype=np.float32).tobytes() for r in practice.rows}
            return fit(parent, practice, **kwargs)
        return wrapped

    monkeypatch.setattr(wsp_baselines, "BASELINE_KINDS", {
        kind: replace(spec, fit=spy(kind, spec.fit)) for kind, spec in wsp_baselines.BASELINE_KINDS.items()})
    bundle = new_bundle(pipeline_db, wsp_probe)
    build_bundle(pipeline_db, bundle, tokenizer=tiny_tokenizer, max_seconds=120)
    captured_rows = set()
    for capture in wsp_captures:
        directory = Path(capture.dir_path)
        offsets = np.load(directory / "offsets.npy")
        tokens = next(directory.glob("tokens.f*"))
        memmap = np.memmap(tokens, dtype=np.dtype(capture.dtype), mode="r", shape=(capture.tokens, capture.d_model))
        captured_rows |= {np.asarray(memmap[int(offsets[i]):int(offsets[i + 1])], dtype=np.float32).tobytes()
                          for i in range(len(offsets) - 1)}
    assert set(seen) == set(wsp_baselines.BASELINE_KINDS)
    for kind, rows in seen.items():
        assert len(rows) > 0 and not (rows & captured_rows), kind


def test_slow_fit_reads_pending_not_missing(pipeline_db, wsp_probe, wsp_captures, tiny_tokenizer, monkeypatch):
    """The runtime setting `wsp_baseline_max_seconds` bounds each fit: a slower one is `pending` (a row says so and
    the kind is absent from `baselines_available`), never silently missing; the others still fit."""
    from dataclasses import replace

    from src.core import runtime_settings
    from src.models.wsp import WspBaseline
    from src.services import wsp_baselines
    from src.services.wsp_bundle import build_bundle

    def slow(parent, practice, **kwargs):
        time.sleep(3)
        raise AssertionError("a pending fit's result must never be used")

    kinds = dict(wsp_baselines.BASELINE_KINDS)
    kinds["lda"] = replace(kinds["lda"], fit=slow)
    monkeypatch.setattr(wsp_baselines, "BASELINE_KINDS", kinds)
    runtime_settings.apply({"wsp_baseline_max_seconds": 1})
    try:
        bundle = new_bundle(pipeline_db, wsp_probe)
        result = build_bundle(pipeline_db, bundle, tokenizer=tiny_tokenizer)
    finally:
        runtime_settings.reset()
    row = pipeline_db.query(WspBaseline).filter(WspBaseline.bundle_id == bundle.id, WspBaseline.kind == "lda").one()
    assert result["baselines"]["lda"] == "pending" and row.fit["pending"] is True and row.weights_path is None
    assert "lda" not in bundle.manifest["baselines_available"]
    assert result["baselines"]["diff_of_means"] == "fitted"


def test_a_heldout_split_as_practice_is_refused_by_the_build(pipeline_db, wsp_probe, wsp_world, wsp_captures,
                                                              tiny_tokenizer):
    """5.5b's wiring: the build itself reads the recorded split roles (control: drop the call → red)."""
    from src.services.wsp_bundle import build_bundle_row

    train = wsp_world["train"]
    saved, train.dataset_version = train.dataset_version, {"version_id": "dsv_x", "wsp_role": "wsp_heldout"}
    pipeline_db.commit()
    try:
        bundle = new_bundle(pipeline_db, wsp_probe)
        result = build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer, max_seconds=60)
    finally:
        train.dataset_version = saved
        pipeline_db.commit()
    assert result["refused"]["code"] == "heldout_overlaps_practice" and result["refused"]["key"] == "partition_role"


# ── R2 review fixes (OR2-044a-M2, -M5), 2026-11-17 ────────────────────────────────────────────────────────────────


def test_a_sampled_heldout_overlap_at_a_sampled_position_is_refused(pipeline_db, ov_probe, ov_captures, ov_world,
                                                                    tiny_tokenizer):
    """P45 over a SAMPLED held-out evaluation (review mutation M17 survived): the held-out corpus repeats practice
    texts only from row 80 on, and its evaluation is a seeded sample of 80 of its 160 rows, so the overlap sits at
    stored positions a whole-view-in-order reading (`arange(rows)`) never reaches. The build reads each capture's
    STORED positions and refuses `heldout_overlaps_practice` by the text key. Control: read `arange(rows)` instead of
    `positions.npy` → the bundle builds → red."""
    from src.services.wsp_bundle import build_bundle_row
    from tests.support.wsp_pipeline import OV_REPEAT_FROM, OV_SAMPLE

    (capture,) = ov_captures
    positions = np.load(Path(capture.dir_path) / "positions.npy")
    first_repeat = OV_REPEAT_FROM
    assert len(positions) == capture.rows == OV_SAMPLE and (positions >= first_repeat).any(), \
        "fixture: the sample holds rows that repeat practice texts"
    assert not (np.arange(capture.rows) >= first_repeat).any(), "fixture: arange(rows) reaches no repeated row"
    bundle = new_bundle(pipeline_db, ov_probe)
    result = build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer, max_seconds=60)
    refused = result.get("refused") or {}
    assert (refused.get("code"), refused.get("key")) == ("heldout_overlaps_practice", "text"), result
    assert 1 <= refused["rows"] <= int((positions >= first_repeat).sum())


def test_a_calibration_set_repeating_practice_texts_is_refused(pipeline_db, co_probe, co_captures, tiny_tokenizer):
    """R3 review fix (OR3-044a-M2): P45/P67 covers the CALIBRATION capture too (review survivor X39). The eval view is
    clean; the calibration corpus repeats practice texts from row 80 on, and the run calibrated on a seeded sample of
    its negatives, so the overlap sits only in the calibration capture's stored positions. A bar cut on rows miForge
    has read as practice would be gameable (OR1-044a-1), so the build refuses `heldout_overlaps_practice` by the text
    key. Control: leave calibration captures out of `heldout_keys` (X39) → the bundle builds → red."""
    from src.services.wsp_bundle import build_bundle_row
    from tests.support.wsp_pipeline import CO_REPEAT_FROM

    by_role = {c.role: c for c in co_captures}
    assert set(by_role) == {"eval", "calibration"}, "fixture: one eval and one calibration capture"
    cal_positions = np.load(Path(by_role["calibration"].dir_path) / "positions.npy")
    eval_positions = np.load(Path(by_role["eval"].dir_path) / "positions.npy")
    repeated = int((cal_positions >= CO_REPEAT_FROM).sum())
    assert repeated >= 1 and repeated < len(cal_positions), "fixture: the sample holds repeated AND clean rows"
    assert not np.array_equal(np.sort(cal_positions), np.arange(len(cal_positions))), "fixture: a sample"
    assert len(eval_positions) and by_role["eval"].dataset_id != by_role["calibration"].dataset_id
    bundle = new_bundle(pipeline_db, co_probe)
    result = build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer, max_seconds=60)
    refused = result.get("refused") or {}
    assert (refused.get("code"), refused.get("key")) == ("heldout_overlaps_practice", "text"), result
    assert refused["heldout_view"] == by_role["calibration"].dataset_id and 1 <= refused["rows"] <= repeated


# ── R3 review fix (OR3-044a-L2): a queued bundle is not reaped, and a reaped bundle is never built ─────────────────


def test_a_fit_longer_than_the_janitor_window_keeps_the_build(pipeline_db, wsp_probe, wsp_captures, tiny_tokenizer,
                                                               monkeypatch):
    """R4 review fix (C044a-L2): `wsp_baseline_max_seconds` may be set up to a day, and the janitor reaps a started
    bundle silent for `BUNDLE_STALE`. The build waits on each fit in `BASELINE_BEAT_SECONDS` slices with a heartbeat
    between them, so a fit longer than that window is still the build's. Here the window is shrunk to 0.6 s and the
    beat to 0.1 s; one fit runs 1.5 s and the REAL janitor sweeps (its own session) 1.0 s into it: the bundle is not
    reaped, every kind fits and it turns `ready`. Control: wait on the fit without beating → reaped `worker_lost`, the
    build stops at its next heartbeat → red."""
    from dataclasses import replace
    from datetime import timedelta

    from sqlalchemy.orm import sessionmaker

    from src.services import wsp_baselines, wsp_bundle
    from src.workers import cleanup_stuck_wsp_rows as janitor

    monkeypatch.setattr(janitor, "BUNDLE_STALE", timedelta(seconds=0.6))
    monkeypatch.setattr(wsp_bundle, "BASELINE_BEAT_SECONDS", 0.1)
    swept = []
    kinds = dict(wsp_baselines.BASELINE_KINDS)
    real_fit = kinds["lda"].fit

    def long_fit(parent, practice, **kwargs):
        time.sleep(1.0)
        with sessionmaker(bind=pipeline_db.get_bind())() as session:
            swept.append(janitor.sweep(session))
        time.sleep(0.5)
        return real_fit(parent, practice, **kwargs)

    kinds["lda"] = replace(kinds["lda"], fit=long_fit)
    monkeypatch.setattr(wsp_baselines, "BASELINE_KINDS", kinds)
    bundle = new_bundle(pipeline_db, wsp_probe)
    result = wsp_bundle.build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(bundle)
    assert len(swept) == 1, swept   # it ran (it may reap other tests' abandoned rows; never this one)
    assert result.get("status") == "ready" and bundle.status == "ready", (result, bundle.error)
    assert result["baselines"]["lda"] == "fitted"


def test_the_fit_heartbeat_is_well_inside_the_janitor_window():
    """The relation the long-fit test shrinks: a fit's heartbeat slice is far inside the janitor's window."""
    from src.services.wsp_bundle import BASELINE_BEAT_SECONDS
    from src.workers.cleanup_stuck_wsp_rows import BUNDLE_STALE

    assert 0 < BASELINE_BEAT_SECONDS * 10 <= BUNDLE_STALE.total_seconds()


def _reap(db, bundle_id, code="worker_lost"):
    """What the janitor writes, from its OWN session (another process in production), so the build's session never
    commits on its behalf."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from src.workers.cleanup_stuck_wsp_rows import WORDS

    with Session(bind=db.get_bind()) as janitor_session:
        janitor_session.execute(text("SET LOCAL lock_timeout = '10s'"))
        janitor_session.execute(text("UPDATE wsp_bundles SET status = 'failed', error = :error WHERE id = :id"),
                                {"error": WORDS[code], "id": bundle_id})
        janitor_session.commit()


def test_a_bundle_queued_behind_a_long_build_is_built_not_reaped(pipeline_db, wsp_probe, wsp_captures,
                                                                 tiny_tokenizer):
    """The solo `wsp` worker reaches a bundle two hours after it was asked for (another build ran first). The janitor
    spares it (no heartbeat: not started), and the task claims and builds it. Controls: age a queued bundle from
    `created_at` → reaped `worker_lost` → red."""
    from datetime import timedelta

    from src.core.clock import utc_now
    from src.services.wsp_bundle import build_bundle_row
    from src.workers import cleanup_stuck_wsp_rows as janitor

    bundle = new_bundle(pipeline_db, wsp_probe)
    bundle.created_at = utc_now() - janitor.BUNDLE_STALE - timedelta(hours=1)
    pipeline_db.commit()
    janitor.sweep(pipeline_db)
    pipeline_db.refresh(bundle)
    assert (bundle.status, bundle.heartbeat_at) == ("building", None), bundle.error
    result = build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(bundle)
    assert result["status"] == bundle.status == "ready" and bundle.heartbeat_at is not None, result


def test_a_reaped_bundle_is_never_built(pipeline_db, wsp_probe, wsp_captures, tiny_tokenizer, monkeypatch):
    """A row the janitor failed before its task arrived stays failed with the janitor's reason: the task claims the row
    FIRST (its first heartbeat, before any work), so it loads nothing, writes nothing, builds no file and fits no
    baseline. Control: skip the claim at the task's start → the probe is loaded (the first in-build heartbeat still
    stops the build) → red."""
    from src.models.wsp import WspBaseline
    from src.services import probe_monitor_run
    from src.services.wsp_bundle import build_bundle_row

    bundle = new_bundle(pipeline_db, wsp_probe)
    _reap(pipeline_db, bundle.id, "never_started")
    loaded = []
    real_load = probe_monitor_run.load_probe
    monkeypatch.setattr(probe_monitor_run, "load_probe", lambda *a, **k: loaded.append(a) or real_load(*a, **k))
    result = build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer, max_seconds=120)
    assert loaded == [], "the task did work on a row that is no longer building"
    pipeline_db.refresh(bundle)
    assert result == {"bundle_id": bundle.id, "status": "failed", "skipped": "bundle_not_building"}
    assert (bundle.status, bundle.manifest, bundle.dir_path) == ("failed", None, None)
    assert bundle.error.startswith("never_started")
    assert pipeline_db.query(WspBaseline).filter(WspBaseline.bundle_id == bundle.id).count() == 0


def test_a_bundle_reaped_mid_build_stops_at_its_next_heartbeat(pipeline_db, wsp_probe, wsp_captures, tiny_tokenizer,
                                                               monkeypatch):
    """Reaped after its files are written: the next heartbeat (the first baseline's) finds the row failed and the build
    stops there, never `ready`, and NO baseline is committed: the first one is pending in the session when its
    heartbeat is refused, and a refused claim rolls it back (R4 review fix C044a-I1). Controls: a heartbeat that does
    not check the row → all five baselines fit (the final check still stops `ready`) → red; the claim commits before
    reading its row count, or the baseline is committed before its heartbeat → one baseline → red."""
    from dataclasses import replace

    from src.models.wsp import WspBaseline
    from src.services import wsp_baselines, wsp_bundle

    bundle = new_bundle(pipeline_db, wsp_probe)
    real = wsp_bundle.write_bundle_files
    fits = []

    def then_reaped(*args, **kwargs):
        files = real(*args, **kwargs)
        _reap(pipeline_db, bundle.id)
        return files

    def counted(kind, fit):
        def run(*args, **kwargs):
            fits.append(kind)
            return fit(*args, **kwargs)
        return run

    monkeypatch.setattr(wsp_baselines, "BASELINE_KINDS", {kind: replace(spec, fit=counted(kind, spec.fit))
                                                         for kind, spec in wsp_baselines.BASELINE_KINDS.items()})
    monkeypatch.setattr(wsp_bundle, "write_bundle_files", then_reaped)
    result = wsp_bundle.build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(bundle)
    assert result["skipped"] == "bundle_not_building" and bundle.status == "failed", result
    assert bundle.error.startswith("worker_lost") and bundle.manifest is None
    # It STOPPED at that heartbeat: one fit ran, and nothing it produced was committed (R4: the C28 control had
    # survived once the refused claim rolled back, so the fits themselves are counted).
    assert len(fits) == 1, fits
    assert pipeline_db.query(WspBaseline).filter(WspBaseline.bundle_id == bundle.id).count() == 0


def test_a_bundle_reaped_after_its_last_heartbeat_never_turns_ready(pipeline_db, wsp_probe, wsp_captures,
                                                                    tiny_tokenizer, monkeypatch):
    """Reaped after the last heartbeat (every baseline fitted), while the manifest is written: the row-locked status
    read before `ready` refuses it. Control: set `ready` without that read → the reaped row turns `ready` → red."""
    from src.services import wsp_bundle

    bundle = new_bundle(pipeline_db, wsp_probe)
    real = wsp_bundle.build_manifest

    def then_reaped(**kwargs):
        _reap(pipeline_db, bundle.id)
        return real(**kwargs)

    monkeypatch.setattr(wsp_bundle, "build_manifest", then_reaped)
    result = wsp_bundle.build_bundle_row(pipeline_db, bundle.id, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(bundle)
    assert result["skipped"] == "bundle_not_building" and bundle.status == "failed", result
    assert bundle.error.startswith("worker_lost") and bundle.manifest is None


def test_the_sealed_sensitivity_reads_the_heldout_sources(pipeline_db, wsp_probe, wsp_captures, wsp_world,
                                                          tiny_tokenizer):
    """FR-7 at bundle build (review mutation M05 survived; OR1-044a-3 fixed only the live route): the manifest's sealed
    `sensitivity` and the row's snapshot cover the practice view AND every held-out source. Practice marked
    `not_sensitive`, the held-out source `sensitive`: the bundle is sealed `sensitive`. Control: compute the build's
    value from the practice view alone → red."""
    from sqlalchemy import text

    from src.models.wsp import WspSensitivityMark
    from src.services.wsp_bundle import build_bundle

    marks = [WspSensitivityMark(dataset_id=wsp_world["practice_ds"].id, value="not_sensitive", reason="public"),
             WspSensitivityMark(dataset_id=wsp_world["heldout_ds"].id, value="sensitive", reason="held-out PII")]
    pipeline_db.add_all(marks)
    pipeline_db.commit()
    ids = [m.id for m in marks]
    try:
        bundle = new_bundle(pipeline_db, wsp_probe)
        build_bundle(pipeline_db, bundle, tokenizer=tiny_tokenizer, max_seconds=120)
        pipeline_db.refresh(bundle)
    finally:
        # Marks are append-only at the ORM; the test's own rows leave by SQL so later tests start unmarked.
        pipeline_db.execute(text("DELETE FROM wsp_sensitivity_marks WHERE id = ANY(:ids)"), {"ids": ids})
        pipeline_db.commit()
    assert bundle.status == "ready", bundle.error
    assert bundle.manifest["sensitivity"]["value"] == "sensitive" and bundle.sensitivity == "sensitive"
    marked = {b["ref"]: b["value"] for b in bundle.manifest["sensitivity"]["basis"] if b["source"] == "operator_mark"}
    assert marked == {ids[0]: "not_sensitive", ids[1]: "sensitive"}
