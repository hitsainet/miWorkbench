"""ENH-010 044 tasks 7.2 – 7.5: the WSP scorer on a REAL tiny run, real captures and a real bundle (FTID I44-D1).

Controls (0xcc/reviews/044_implementation_controls_2026-11-16.md): key the budget on the request digest → the
two-bundle test red; count only the candidate → the controls test red; key on `row_ids_sha256` → the re-registration
test red; a per-set key → the overlapping re-sample test red; drop the advisory lock → the concurrency test red;
default the clamp to 256 → the null-clamp test red; fused matmul → `test_scores_match_native_aggregate_rows` red;
include truncated rows → the truncation test red; metrics from float64 → the exact-recompute test red; add a
`debug_rows` field with a row's text → the leak test red; skip the record → the recorded-once test red.
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    pb_captures,
    pb_heldout_arrow,
    pb_probe,
    pb_world,
    wsp_captures,
    wsp_data_dir,
    wsp_probe,
    wsp_world,
)

from src.services import wsp_score as ws  # noqa: E402
from src.services.wsp_patch import tensor_sha256  # noqa: E402


@pytest.fixture(scope="module")
def bundle(pipeline_db, wsp_probe, wsp_captures, tiny_tokenizer):
    from src.services.wsp_bundle import build_bundle

    row = new_bundle(pipeline_db, wsp_probe)
    build_bundle(pipeline_db, row, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(row)
    assert row.status == "ready"
    return row


@pytest.fixture(scope="module")
def pb_bundle(pipeline_db, pb_probe, pb_captures, tiny_tokenizer):
    """A PRACTICE-basis bundle over a SAMPLED held-out evaluation (R2 review, OR2-044a-M3, -M4)."""
    from src.services.wsp_bundle import build_bundle

    row = new_bundle(pipeline_db, pb_probe)
    build_bundle(pipeline_db, row, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(row)
    assert row.status == "ready" and row.manifest["operating_point"]["threshold_source"] == "practice", row.error
    return row


#: The committed scoring contract (7.6), which miForge vendors byte-identical and validates every answer against.
CONTRACT = json.loads((Path(__file__).resolve().parents[3] / "docs" / "schemas"
                       / "miworkbench-wsp-scoring-v1.json").read_text())


def contract_checked(response):
    """R2 review fix (OR2-044a-M4), 2026-11-17: a scorer answer is the published contract's — the response model
    (`extra="forbid"` at every level) AND, independently, the committed JSON schema."""
    from jsonschema import Draft202012Validator

    from src.schemas.wsp_scoring import WspScoreResponse

    WspScoreResponse.model_validate(response)
    errors = sorted(Draft202012Validator(CONTRACT["response"]).iter_errors(response), key=str)
    assert errors == [], [e.message for e in errors[:5]]
    return response


@pytest.fixture(autouse=True)
def every_answer_is_the_contracts(monkeypatch):
    """EVERY `ws.score` answer in this module is checked against the contract (both bases, every head kind), not one
    named fixture. Control: add a `rows` field to each `metrics` entry (review mutation M09) → red."""
    real = ws.score
    answers = []

    def checked(*args, **kwargs):
        answers.append(contract_checked(real(*args, **kwargs)))
        return answers[-1]

    monkeypatch.setattr(ws, "score", checked)
    return answers


@pytest.fixture(autouse=True)
def fresh_settings():
    from src.core import runtime_settings

    runtime_settings.reset()
    yield
    runtime_settings.reset()


@pytest.fixture
def empty_budget(pipeline_db):
    from src.models.wsp import WspHeldoutRowHead

    pipeline_db.query(WspHeldoutRowHead).delete()
    pipeline_db.commit()
    yield


def _tensor(name, array):
    array = np.ascontiguousarray(array)
    return {"name": name, "dtype": str(array.dtype), "shape": list(array.shape),
            "data_b64": base64.b64encode(array.astype(array.dtype.newbyteorder("<")).tobytes()).decode(),
            "sha256": tensor_sha256(array)}


def _parent(bundle):
    return ws._parent_arrays(bundle)


def _edited(bundle, *, entries=1, delta=0.25, tensor="weight", seed=0):
    arrays = {k: v.copy() for k, v in _parent(bundle).items()}
    flat = arrays[tensor].reshape(-1)
    for i in range(entries):
        flat[(seed + i) % flat.size] = flat[(seed + i) % flat.size] + np.asarray(delta, dtype=flat.dtype)
    return arrays


def _head(label, arrays=None):
    return {"label": label, "tensors": None if arrays is None else [_tensor(n, a) for n, a in arrays.items()]}


def _request(bundle, heads, **extra):
    from src.schemas.wsp_scoring import WspScoreRequest

    body = {"kind": "miworkbench.wsp-score-request/v1", "bundle_id": bundle.id, "bundle_sha256": bundle.bundle_sha256,
            "run_ref": {"miforge_run_id": "wsr_test", "patch_sha256": "0" * 64, "max_changed_entries": None},
            "heldout_ref": json.loads(json.dumps(bundle.manifest["heldout_ref"])), "heads": heads,
            "primary_sets": []}
    body.update(extra)
    return WspScoreRequest.model_validate(body)


def _refused(db, request, code):
    with pytest.raises(ws.ScoreRefused) as refused:
        ws.score(db, request)
    db.rollback()
    assert refused.value.code == code, refused.value.message
    return refused.value


# ── 7.2 refusals ──────────────────────────────────────────────────────────────


def test_parent_tensors_refused(pipeline_db, bundle):
    _refused(pipeline_db, _request(bundle, [_head("parent", _parent(bundle))]), "parent_tensors_not_accepted")
    _refused(pipeline_db, _request(bundle, [_head("baseline:lda", _parent(bundle))]), "parent_tensors_not_accepted")


def test_head_tensors_required(pipeline_db, bundle):
    _refused(pipeline_db, _request(bundle, [_head("candidate")]), "head_tensors_required")


def test_head_shape_mismatch_refused(pipeline_db, bundle):
    arrays = _edited(bundle)
    arrays["weight"] = arrays["weight"][:-1]
    _refused(pipeline_db, _request(bundle, [_head("candidate", arrays)]), "head_shape_mismatch")
    good = [_tensor(n, a) for n, a in _edited(bundle).items()]
    good[0]["sha256"] = "f" * 64
    _refused(pipeline_db, _request(bundle, [{"label": "candidate", "tensors": good}]), "head_shape_mismatch")


def test_heldout_ref_mismatch_refused(pipeline_db, bundle):
    ref = json.loads(json.dumps(bundle.manifest["heldout_ref"]))
    ref["sets"][0]["rows"] += 1
    _refused(pipeline_db, _request(bundle, [_head("parent")], heldout_ref=ref), "heldout_ref_mismatch")


def test_subset_reference_refused(pipeline_db, bundle):
    ref = json.loads(json.dumps(bundle.manifest["heldout_ref"]))
    assert len(ref["sets"]) >= 2
    ref["sets"] = ref["sets"][:1]
    _refused(pipeline_db, _request(bundle, [_head("parent")], heldout_ref=ref), "heldout_ref_mismatch")


def test_bundle_digest_and_kind_refusals(pipeline_db, bundle):
    _refused(pipeline_db, _request(bundle, [_head("parent")], bundle_sha256="a" * 64), "bundle_sha256_mismatch")
    _refused(pipeline_db, _request(bundle, [_head("parent")], bundle_id="wsb_000000000000"), "bundle_not_found")


def test_too_many_heads_refused(pipeline_db, bundle):
    from src.core import runtime_settings

    runtime_settings.apply({"wsp_score_max_heads": 2})
    _refused(pipeline_db, _request(bundle, [_head("parent"), _head("baseline:lda"), _head("baseline:diff_of_means")]),
             "too_many_heads")


def test_timeout_refuses_and_records_nothing(pipeline_db, bundle, empty_budget, monkeypatch):
    """3.5: `wsp_score_timeout_seconds` is read per call; a call still scoring past it answers 503 `score_timeout` and
    leaves no call row and no budget row (a call that did not answer used nothing). Control: drop the checkpoint
    (`wsp_score.checkpoint`'s deadline line) → red. That the deadline bites INSIDE a set is
    `test_the_deadline_bites_inside_a_set`."""
    from src.core import runtime_settings
    from src.models.wsp import WspHeldoutRowHead, WspScoreCall

    clock = iter([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    monkeypatch.setattr(ws, "_NOW", lambda: next(clock))
    runtime_settings.apply({"wsp_score_timeout_seconds": 5})
    calls = pipeline_db.query(WspScoreCall).count()
    refused = _refused(pipeline_db, _request(bundle, [_head("parent"), _head("candidate", _edited(bundle, seed=40))]),
                       "score_timeout")
    assert refused.status == 503 and "wsp_score_timeout_seconds = 5" in refused.message
    assert pipeline_db.query(WspScoreCall).count() == calls
    assert pipeline_db.query(WspHeldoutRowHead).count() == 0
    # The same call under a limit it fits answers (the setting, not the clock alone, decides).
    clock2 = iter([0.0] + [1.0] * 50)
    monkeypatch.setattr(ws, "_NOW", lambda: next(clock2))
    runtime_settings.apply({"wsp_score_timeout_seconds": 600})
    assert ws.score(pipeline_db, _request(bundle, [_head("parent"), _head("candidate", _edited(bundle, seed=40))]))


@pytest.fixture
def one_row_buckets(monkeypatch):
    """Every row its own bucket (a one-slot budget), so a set is many buckets and a check between them is visible."""
    from src.services import probe_monitor_trainer

    monkeypatch.setattr(probe_monitor_trainer, "slots_for", lambda d_model, budget_bytes=None: 1)


@pytest.fixture
def scored_buckets(monkeypatch):
    """Every `combine` call `_aggregate_rows_many` makes (one per weighted head per bucket), with the torch thread count
    in force when it ran."""
    import torch

    from src.ml import probe_monitor_model

    real = probe_monitor_model.combine
    seen = []

    def combine(*args, **kwargs):
        seen.append(torch.get_num_threads())
        return real(*args, **kwargs)

    monkeypatch.setattr(probe_monitor_model, "combine", combine)
    return seen


def test_the_deadline_bites_inside_a_set(pipeline_db, bundle, empty_budget, monkeypatch, one_row_buckets,
                                         scored_buckets):
    """2026-11-22 (the D128 defect): the deadline was checked only after a whole SET, so a call inside D128's 39 GB
    everyday_text set ran on past `wsp_score_timeout_seconds`. Now a call ends at the deadline plus at most one bucket.
    Here the clock passes the deadline during the first bucket of the first set; the second bucket is never scored.
    Control: score `score`'s sets without `between_buckets` → red (every bucket of the first set is scored)."""
    from src.core import runtime_settings
    from src.models.wsp import WspHeldoutRowHead, WspScoreCall

    now = [0.0]
    monkeypatch.setattr(ws, "_NOW", lambda: now[0])
    runtime_settings.apply({"wsp_score_timeout_seconds": 5})
    from src.ml import probe_monitor_model

    counting = probe_monitor_model.combine   # the `scored_buckets` spy

    def slow_first_bucket(*args, **kwargs):
        now[0] = 100.0   # the first bucket took the call past its deadline
        return counting(*args, **kwargs)

    monkeypatch.setattr(probe_monitor_model, "combine", slow_first_bucket)
    calls = pipeline_db.query(WspScoreCall).count()
    assert bundle.manifest["heldout_ref"]["sets"][0]["rows"] > 2, "fixture too weak: one bucket"
    refused = _refused(pipeline_db, _request(bundle, [_head("parent"), _head("candidate", _edited(bundle, seed=43))]),
                       "score_timeout")
    weighted = 2   # parent, candidate: one combine each per bucket
    assert len(scored_buckets) == weighted, \
        f"{len(scored_buckets) // weighted} buckets were scored; the deadline passed in the first"
    assert "set " in refused.message
    assert pipeline_db.query(WspScoreCall).count() == calls and pipeline_db.query(WspHeldoutRowHead).count() == 0


async def test_an_abandoned_call_stops_records_nothing_and_frees_the_scorer(pipeline_db, bundle, empty_budget,
                                                                           monkeypatch, one_row_buckets,
                                                                           scored_buckets):
    """2026-11-22 (the D128 defect): miForge gave up at 600 s and the scorer went on with nobody waiting, holding
    `SEMAPHORE`, so 46 calls got 429 `scorer_busy`. Through the REAL route: the client disconnects (ASGI
    `http.disconnect`) while the call is scoring; the call stops before its next bucket, records no call row and no
    budget row, answers 499 to nobody, and the scorer takes the next call.

    Controls: drop the route's `watch_client` start → red (the call scores to the end and answers 200); drop the
    abandoned check in `wsp_score.checkpoint` → red; drop the route's `SEMAPHORE.release()` → red (the last
    assertion)."""
    import asyncio
    import threading
    from types import SimpleNamespace

    from sqlalchemy.orm import sessionmaker

    from src.api.v1.endpoints import wsp as route
    from src.models.wsp import WspHeldoutRowHead, WspScoreCall
    from src.services import probe_monitor_run

    monkeypatch.setattr(route, "_session", sessionmaker(bind=pipeline_db.get_bind()))
    loop = asyncio.get_running_loop()
    gone = asyncio.Event()

    async def receive():
        await gone.wait()
        return {"type": "http.disconnect"}

    request = SimpleNamespace(client=SimpleNamespace(host="203.0.113.9"), headers={}, receive=receive)
    hung_up = threading.Event()
    scored_at_hang_up = []
    real_many = probe_monitor_run._aggregate_rows_many

    def many(heads, rule, params, rows, *, between_buckets=None):
        def hook():
            if not hung_up.is_set():
                # Mid-call: one set's first bucket is next. Hang up, then let the watch see it (bounded).
                scored_at_hang_up.append(len(scored_buckets))
                hung_up.set()
                loop.call_soon_threadsafe(gone.set)
                end = time.monotonic() + 5.0
                while time.monotonic() < end:
                    between_buckets()   # raises ScoreAbandoned once the watch has set the call's stop
                    time.sleep(0.01)
            between_buckets()

        return real_many(heads, rule, params, rows, between_buckets=hook if between_buckets else None)

    monkeypatch.setattr(probe_monitor_run, "_aggregate_rows_many", many)
    calls = pipeline_db.query(WspScoreCall).count()
    body = _request(bundle, [_head("parent"), _head("candidate", _edited(bundle, seed=44))])
    answer = await route.score_heads(body, request)
    assert getattr(answer, "status_code", None) == route.CLIENT_CLOSED_STATUS, answer
    assert hung_up.is_set() and len(scored_buckets) == scored_at_hang_up[0], "a bucket was scored after the hang-up"
    pipeline_db.expire_all()
    assert pipeline_db.query(WspScoreCall).count() == calls and pipeline_db.query(WspHeldoutRowHead).count() == 0
    assert ws.SEMAPHORE.acquire(blocking=False), "the abandoned call kept the scorer: every next call would be 429"
    ws.SEMAPHORE.release()


def test_the_thread_bound_is_the_runtime_setting(pipeline_db, bundle, empty_budget, scored_buckets):
    """2026-11-22: the scorer runs in the API process; unbounded, D128 held about 9.4 cores of it for an hour. Every
    bucket is scored at `wsp_score_torch_threads` torch threads, on a worker thread as the route runs it, and the
    previous count is back afterwards. Control: drop `with scorer_threads():` → red."""
    import threading

    import torch

    from src.core import runtime_settings

    results = {}

    def run():
        before = torch.get_num_threads()
        wanted = 1 if before > 1 else 2
        results["before"], results["wanted"] = before, wanted
        if wanted > ws._usable_cpus():
            return
        runtime_settings.apply({"wsp_score_torch_threads": wanted})
        ws.score(pipeline_db, _request(bundle, [_head("parent"), _head("candidate", _edited(bundle, seed=45))]))
        results["after"] = torch.get_num_threads()

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(120)
    if "after" not in results:
        pytest.skip(f"one CPU and one torch thread: a bound of {results.get('wanted')} cannot be told apart")
    assert scored_buckets and set(scored_buckets) == {results["wanted"]}, (scored_buckets, results)
    assert results["after"] == results["before"]


def test_non_editable_tensor_change_refused(pipeline_db, bundle, empty_budget):
    _refused(pipeline_db, _request(bundle, [_head("candidate", _edited(bundle, tensor="mean"))]),
             "candidate_outside_patch_limits")


def test_null_max_changed_entries_takes_the_ceiling(pipeline_db, bundle, empty_budget):
    from src.core import runtime_settings

    runtime_settings.apply({"wsp_score_max_changed_entries_ceiling": 3})
    _refused(pipeline_db, _request(bundle, [_head("candidate", _edited(bundle, entries=4))]),
             "candidate_outside_patch_limits")
    assert ws.score(pipeline_db, _request(bundle, [_head("candidate", _edited(bundle, entries=3, seed=7))]))
    # Empty ceiling = every editable entry of the parent: a head changing ALL of `weight` is inside it.
    runtime_settings.apply({"wsp_score_max_changed_entries_ceiling": None})
    whole = _edited(bundle, entries=int(_parent(bundle)["weight"].size), seed=0, delta=0.5)
    assert ws.score(pipeline_db, _request(bundle, [_head("candidate", whole)]))


def test_the_empty_ceiling_is_every_editable_entry_not_a_constant():
    """At D128's width (4,096) the empty ceiling is the parent's editable entry count — a fixed default (256, say)
    would refuse an honest dense edit. `mean` and `std` are not editable and do not count."""
    from src.core import runtime_settings

    runtime_settings.apply({"wsp_score_max_changed_entries_ceiling": None})
    d = 4096
    parent = {"weight": np.zeros(d, np.float32), "bias": np.zeros(1, np.float64), "mean": np.zeros(d, np.float32),
              "std": np.ones(d, np.float32)}
    editable = {"weight": True, "bias": True, "mean": False, "std": False}
    run_ref = SimpleNamespace(max_changed_entries=None)
    assert ws.entries_limit(run_ref, parent, editable) == d + 1
    assert ws.entries_limit(SimpleNamespace(max_changed_entries=7), parent, editable) == 7
    runtime_settings.apply({"wsp_score_max_changed_entries_ceiling": 5})
    assert ws.entries_limit(SimpleNamespace(max_changed_entries=7), parent, editable) == 5


def test_run_ref_limit_overrides_fallback(pipeline_db, bundle, empty_budget):
    request = _request(bundle, [_head("candidate", _edited(bundle, entries=2, seed=11))],
                       run_ref={"miforge_run_id": "wsr_x", "patch_sha256": None, "max_changed_entries": 1})
    _refused(pipeline_db, request, "candidate_outside_patch_limits")


def test_capture_not_ready_refused(pipeline_db, bundle, wsp_captures):
    capture = wsp_captures[0]
    for status, code in (("running", "capture_not_ready"), ("parity_failed", "capture_parity_failed")):
        capture.status = status
        pipeline_db.commit()
        try:
            _refused(pipeline_db, _request(bundle, [_head("parent")]), code)
        finally:
            capture.status = "ready"
            pipeline_db.commit()


def test_heldout_ref_sha256_is_jcs_of_the_request(pipeline_db, bundle):
    import rfc8785

    request = _request(bundle, [_head("parent")])
    response = ws.score(pipeline_db, request)
    independent = hashlib.sha256(rfc8785.dumps(json.loads(json.dumps(bundle.manifest["heldout_ref"])))).hexdigest()
    assert response["heldout_ref_sha256"] == independent


def test_honest_probe_call_fits_the_body_cap():
    """Two tensor-bearing heads at d_model 4096 (weight, mean, std float32; bias float64) stay under the app's 1 MiB."""
    from src.core.body_limit import MAX_BODY_BYTES

    d = 4096
    arrays = {"weight": np.ones(d, np.float32), "bias": np.zeros(1, np.float64), "mean": np.ones(d, np.float32),
              "std": np.ones(d, np.float32)}
    body = {"kind": "miworkbench.wsp-score-request/v1", "bundle_id": "wsb_0123456789ab", "bundle_sha256": "0" * 64,
            "run_ref": {"miforge_run_id": "r", "patch_sha256": "0" * 64, "max_changed_entries": 1},
            "heldout_ref": {"owner": "miworkbench", "sets": [
                {"name": f"eval-pmd_{i:012d}", "view_id": f"pmd_{i:012d}", "role": "eval", "rows": 7000,
                 "row_ids_sha256": "0" * 64} for i in range(6)]},
            "heads": [_head("parent"), _head("candidate", arrays), _head("control", arrays)]
            + [_head(f"baseline:{k}") for k in ("logistic_regression", "lda", "diff_of_means", "threshold_recut",
                                               "affine_recalibration")],
            "primary_sets": []}
    size = len(json.dumps(body).encode())
    assert size < MAX_BODY_BYTES, size


# ── the budget (P56, P64, P70) ────────────────────────────────────────────────


def test_controls_count_against_the_budget(pipeline_db, bundle, empty_budget):
    from src.core import runtime_settings

    runtime_settings.apply({"wsp_score_max_heads_per_set": 2})
    heads = [_head("candidate", _edited(bundle, seed=1)), _head("control", _edited(bundle, seed=2)),
             _head("control:1", _edited(bundle, seed=3))]
    _refused(pipeline_db, _request(bundle, heads), "heldout_set_head_budget_exhausted")
    assert ws.score(pipeline_db, _request(bundle, heads[:2]))
    # The same two heads again cost nothing; a third distinct one is refused.
    assert ws.score(pipeline_db, _request(bundle, heads[:2]))
    _refused(pipeline_db, _request(bundle, [heads[2]]), "heldout_set_head_budget_exhausted")


def test_head_budget_is_per_set(pipeline_db, bundle, wsp_probe, tiny_tokenizer, empty_budget):
    """Two bundles of one probe reference the same captured rows: their calls share one budget."""
    from src.core import runtime_settings
    from src.services.wsp_bundle import build_bundle

    other = new_bundle(pipeline_db, wsp_probe)
    build_bundle(pipeline_db, other, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(other)
    runtime_settings.apply({"wsp_score_max_heads_per_set": 2})
    assert ws.score(pipeline_db, _request(bundle, [_head("candidate", _edited(bundle, seed=4)),
                                                   _head("control", _edited(bundle, seed=5))]))
    refusal = _refused(pipeline_db, _request(other, [_head("candidate", _edited(other, seed=6))]),
                       "heldout_set_head_budget_exhausted")
    assert refusal.status == 409


def test_exhaustion_is_409_with_the_set(pipeline_db, bundle, empty_budget):
    from src.core import runtime_settings

    runtime_settings.apply({"wsp_score_max_heads_per_set": 2})
    assert ws.score(pipeline_db, _request(bundle, [_head("candidate", _edited(bundle, seed=8)),
                                                   _head("control", _edited(bundle, seed=10))]))
    refusal = _refused(pipeline_db, _request(bundle, [_head("candidate", _edited(bundle, seed=9))]),
                       "heldout_set_head_budget_exhausted")
    first = bundle.manifest["heldout_ref"]["sets"][0]
    assert refusal.status == 409 and refusal.extra["set"] == first["name"]
    assert refusal.extra["row_ids_sha256"] == first["row_ids_sha256"]
    assert refusal.extra["rows_exhausted"] == first["rows"]   # every row is at the ceiling (a COUNT)


def _reregistered(db, view, name):
    """A second view over the same downloaded split: a re-registration (new id, the same rows)."""
    from src.models.probe_monitor import ProbeMonitorDataset

    clone = ProbeMonitorDataset(name=name, dataset_id=view.dataset_id, input_column=view.input_column,
                                label_column=view.label_column, label_mapping=dict(view.label_mapping),
                                role=view.role, distribution=view.distribution, counts={})
    db.add(clone)
    db.commit()
    return clone


def _clone_capture(db, capture, view, rows):
    """The captured rows `rows` (indices into `capture`), as a ready capture taken through `view`: the same rows'
    activations, re-sliced into their own files (what a re-registered view or a re-sample of the pool captures)."""
    from src.models.probe_monitor import ProbeMonitor
    from src.models.wsp import WspActivationCapture
    from src.services.wsp_digests import row_ids_sha256
    from src.services.wsp_layout import capture_dir

    source = Path(capture.dir_path)
    offsets = np.load(source / "offsets.npy")
    tokens = next(source.glob("tokens.f*"))
    memmap = np.memmap(tokens, dtype=np.dtype(capture.dtype), mode="r", shape=(int(capture.tokens),
                                                                               int(capture.d_model)))
    rows = [int(r) for r in rows]
    row = WspActivationCapture(probe_id=capture.probe_id, dataset_id=view.id, role=capture.role, layer=capture.layer,
                               scope=capture.scope, status="ready", parity=capture.parity, d_model=capture.d_model,
                               dtype=capture.dtype)
    db.add(row)
    db.flush()
    run_id = db.get(ProbeMonitor, capture.probe_id).run_id
    directory = capture_dir(run_id, row.id)
    directory.mkdir(parents=True)
    blocks = [np.asarray(memmap[int(offsets[i]):int(offsets[i + 1])]) for i in rows]
    np.concatenate(blocks).tofile(directory / tokens.name)
    np.save(directory / "offsets.npy", np.concatenate([[0], np.cumsum([len(b) for b in blocks])]).astype(np.int64))
    for name in ("positions", "labels", "n_scored", "truncated"):
        np.save(directory / f"{name}.npy", np.load(source / f"{name}.npy")[rows])
    positions = np.load(directory / "positions.npy")
    row.rows, row.tokens, row.dir_path = len(rows), int(sum(len(b) for b in blocks)), str(directory)
    row.row_ids_sha256 = row_ids_sha256(view.id, positions)
    db.commit()
    return row


def _bundle_over(db, probe, captures, tokenizer, monkeypatch):
    """A real bundle over exactly `captures` (the build takes the probe's ready captures; here, these)."""
    from src.services import wsp_bundle

    row = new_bundle(db, probe)
    with monkeypatch.context() as patch:   # never `undo()`: that would also lift the autouse contract check
        patch.setattr(wsp_bundle, "ready_captures", lambda _db, _probe, _view_ids=None: list(captures))
        wsp_bundle.build_bundle(db, row, tokenizer=tokenizer, max_seconds=120)
    db.refresh(row)
    assert row.status == "ready", row.error
    return row


def _one_set(bundle):
    (only,) = bundle.manifest["heldout_ref"]["sets"]
    return only


def test_reregistered_view_shares_the_budget(pipeline_db, pb_bundle, pb_probe, pb_captures, pb_world,
                                            tiny_tokenizer, empty_budget, monkeypatch):
    """P67, R3B-6, through `score()` (R2 review fix OR2-044a-L4: this used to call a helper production never calls).
    The probe's held-out pool is read to the ceiling through its view; the same rows registered again as ANOTHER view
    (a new view id, so a new `row_ids_sha256`) get no fresh heads. Control: key the budget on the set's
    `row_ids_sha256` instead of each row's content key → red."""
    from src.core import runtime_settings

    runtime_settings.apply({"wsp_score_max_heads_per_set": 2})
    assert ws.score(pipeline_db, _request(pb_bundle, [_head("candidate", _edited(pb_bundle, seed=51)),
                                                      _head("control", _edited(pb_bundle, seed=52))]))
    view = _reregistered(pipeline_db, pb_world["eval"], "unseen-again")
    again = _bundle_over(pipeline_db, pb_probe, [_clone_capture(pipeline_db, pb_captures[0], view,
                                                                range(pb_captures[0].rows))],
                         tiny_tokenizer, monkeypatch)
    assert _one_set(again)["row_ids_sha256"] != _one_set(pb_bundle)["row_ids_sha256"], "fixture: another view's digest"
    refusal = _refused(pipeline_db, _request(again, [_head("candidate", _edited(again, seed=53))]),
                       "heldout_set_head_budget_exhausted")
    assert refusal.extra["set"] == _one_set(again)["name"] and refusal.extra["rows_exhausted"] == pb_captures[0].rows


def test_overlapping_resample_shares_the_row_budget(pipeline_db, pb_probe, pb_captures, pb_world, tiny_tokenizer,
                                                    empty_budget, monkeypatch):
    """P70, R4B-4, through `score()`: rows 0–19 of the pool read to the ceiling through one sample; a re-sample over
    rows 10–29 is refused on the 10 shared rows only (`rows_exhausted` = 10), and a sample over rows 20–29 — disjoint
    from the first — is scored. Control: a per-set key → the re-sample passes → red."""
    from src.core import runtime_settings

    runtime_settings.apply({"wsp_score_max_heads_per_set": 2})
    capture = pb_captures[0]
    assert capture.rows >= 30, "fixture: the sampled pool has at least 30 rows"

    def sample(name, rows):
        view = _reregistered(pipeline_db, pb_world["eval"], name)
        return _bundle_over(pipeline_db, pb_probe, [_clone_capture(pipeline_db, capture, view, rows)],
                            tiny_tokenizer, monkeypatch)

    first, resample, disjoint = sample("s-0-19", range(0, 20)), sample("s-10-29", range(10, 30)), \
        sample("s-20-29", range(20, 30))
    assert ws.score(pipeline_db, _request(first, [_head("candidate", _edited(first, seed=61)),
                                                  _head("control", _edited(first, seed=62))]))
    refusal = _refused(pipeline_db, _request(resample, [_head("candidate", _edited(resample, seed=63))]),
                       "heldout_set_head_budget_exhausted")
    assert refusal.extra["rows_exhausted"] == 10
    assert ws.score(pipeline_db, _request(disjoint, [_head("candidate", _edited(disjoint, seed=63))]))


def test_concurrent_calls_cannot_both_pass_the_ceiling(pipeline_db, pb_bundle, empty_budget, monkeypatch):
    """Two sessions call `score()` at once, every row one head below the ceiling, each with a new head: the advisory
    lock lets exactly one through (the pause sits between the count and the insert, inside the lock). Control: drop
    the `pg_advisory_xact_lock` → both pass → red."""
    from sqlalchemy.orm import sessionmaker

    from src.core import runtime_settings

    runtime_settings.apply({"wsp_score_max_heads_per_set": 2})
    assert ws.score(pipeline_db, _request(pb_bundle, [_head("candidate", _edited(pb_bundle, seed=71))]))
    monkeypatch.setattr(ws, "_BETWEEN_COUNT_AND_INSERT", lambda: time.sleep(1.0))
    Session = sessionmaker(bind=pipeline_db.get_bind())
    outcomes = []

    def caller(seed):
        session = Session()
        try:
            ws.score(session, _request(pb_bundle, [_head("candidate", _edited(pb_bundle, seed=seed))]))
            outcomes.append("passed")
        except ws.ScoreRefused as refused:
            session.rollback()
            outcomes.append(refused.code)
        finally:
            session.close()

    threads = [threading.Thread(target=caller, args=(seed,)) for seed in (72, 73)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["heldout_set_head_budget_exhausted", "passed"]


# ── 7.3 scoring, bars, metrics ────────────────────────────────────────────────


@pytest.fixture
def all_heads(bundle, pipeline_db):
    from src.models.wsp import WspBaseline

    kinds = [b.kind for b in pipeline_db.query(WspBaseline).filter(WspBaseline.bundle_id == bundle.id).all()
             if not (b.fit or {}).get("pending") and not (b.fit or {}).get("refused")]
    return [_head("parent"), _head("candidate", _edited(bundle, seed=21))] + [_head(f"baseline:{k}") for k in kinds]


def test_scores_match_native_aggregate_rows(pipeline_db, bundle, all_heads, empty_budget):
    from src.services.probe_monitor_run import _aggregate_rows

    response = ws.score(pipeline_db, _request(bundle, all_heads))
    parent = ws.to_probe_head(_parent(bundle), bundle.manifest["identity"]["layer"])
    function = bundle.manifest["function"]["linear_probe"]
    params = {k: v for k, v in function["params"].items() if v is not None}
    for entry in response["sets"]:
        if "scores" not in entry:
            continue
        one = next(s for s in _sets(pipeline_db, bundle) if s.name == entry["name"])
        rows = [r for r, t in zip(one.rows, one.truncated) if t == 0]
        native = np.asarray(_aggregate_rows(parent, function["rule"], params, rows), dtype=np.float32)
        assert np.array_equal(np.asarray(entry["scores"]["parent"], dtype=np.float32), native)


def _sets(db, bundle):
    from src.models.wsp import WspActivationCapture
    from src.services.wsp_bundle import heldout_set_name

    captures = db.query(WspActivationCapture).filter(WspActivationCapture.id.in_(bundle.capture_ids)).all()
    return [ws.load_set(db, c, heldout_set_name(c)) for c in captures]


def test_truncated_rows_excluded_and_counted(pipeline_db, bundle, wsp_captures, empty_budget):
    capture = next(c for c in wsp_captures if c.role == "eval")
    path = Path(capture.dir_path) / "truncated.npy"
    original = path.read_bytes()
    mask = np.zeros(capture.rows, dtype=np.int8)
    mask[:3] = 1
    np.save(path, mask)
    try:
        response = ws.score(pipeline_db, _request(bundle, [_head("parent")]))
    finally:
        path.write_bytes(original)
    entry = next(s for s in response["sets"] if s["role"] == "eval")
    assert entry["excluded"]["truncated"] == 3 and entry["rows"] == capture.rows
    assert len(entry["scores"]["parent"]) == len(entry["labels"]) == capture.rows - 3
    assert entry["scored_row_ids_sha256"] != entry["row_ids_sha256"]


def test_metrics_equal_probe_monitor_metrics(pipeline_db, bundle, all_heads, empty_budget):
    from src.services.probe_monitor_metrics import auroc

    response = ws.score(pipeline_db, _request(bundle, all_heads))
    for entry in (s for s in response["sets"] if "metrics" in s):
        for label, metrics in entry["metrics"].items():
            assert metrics["auroc"] == auroc(entry["scores"][label], entry["labels"]), label


def test_metrics_recompute_exactly_from_the_response(pipeline_db, bundle, all_heads, empty_budget):
    """A consumer recomputing from the response's float32 scores gets the same numbers: the counts exactly, AUROC
    within 1e-12 (an independent implementation)."""
    from sklearn.metrics import roc_auc_score

    response = ws.score(pipeline_db, _request(bundle, all_heads))
    for entry in (s for s in response["sets"] if "metrics" in s):
        labels = np.asarray(entry["labels"])
        for label, metrics in entry["metrics"].items():
            # The response's scores ARE float32 values (cast before any metric), so a consumer's float32 view of
            # them is the exact numbers the metrics were computed from.
            assert all(float(np.float32(v)) == v for v in entry["scores"][label]), label
            scores = np.asarray(entry["scores"][label], dtype=np.float32)
            bar = entry["bars"][label]["threshold"]
            fires = np.zeros(len(scores), bool) if bar is None else scores >= np.float32(bar)
            assert metrics["false_alarms_at_bar"] == int(fires[labels == 0].sum())
            assert metrics["negatives"] == int((labels == 0).sum()) and metrics["positives"] == int(labels.sum())
            if metrics["positives"]:
                assert metrics["recall_at_bar"] == fires[labels == 1].sum() / labels.sum()
            if metrics["auroc"] is not None:
                assert abs(metrics["auroc"] - roc_auc_score(labels, scores.astype(np.float64))) <= 1e-12


def test_threshold_recut_bar_is_cut_on_practice(pipeline_db, bundle, empty_budget):
    from src.models.wsp import WspBaseline

    stored = pipeline_db.query(WspBaseline).filter(WspBaseline.bundle_id == bundle.id,
                                                    WspBaseline.kind == "threshold_recut").one().bar
    response = ws.score(pipeline_db, _request(bundle, [_head("parent"), _head("baseline:threshold_recut")]))
    for entry in response["sets"]:
        assert entry["bars"]["baseline:threshold_recut"]["source"] == "practice"
        assert entry["bars"]["baseline:threshold_recut"]["threshold"] == stored["threshold"]
        if "scores" in entry:
            assert entry["scores"]["baseline:threshold_recut"] == entry["scores"]["parent"]


def test_affine_bar_is_logit_zero(pipeline_db, bundle, empty_budget):
    from src.models.wsp import WspBaseline

    row = pipeline_db.query(WspBaseline).filter(WspBaseline.bundle_id == bundle.id,
                                                 WspBaseline.kind == "affine_recalibration").one()
    if (row.fit or {}).get("refused"):
        pytest.skip(f"the tiny run's affine fit was refused ({row.fit}); covered by test_wsp_baselines")
    response = ws.score(pipeline_db, _request(bundle, [_head("parent"), _head("baseline:affine_recalibration")]))
    a, c = row.transform["a"], row.transform["c"]
    for entry in response["sets"]:
        bar = entry["bars"]["baseline:affine_recalibration"]
        assert (bar["threshold"], bar["source"]) == (0.0, "practice_logit_zero")
        if "scores" in entry:
            expected = np.float32(a * np.asarray(entry["scores"]["parent"], np.float32).astype(np.float64) + c)
            assert np.array_equal(np.asarray(entry["scores"]["baseline:affine_recalibration"], np.float32), expected)


def test_bars_are_equal_across_sets_for_a_head(pipeline_db, bundle, all_heads, empty_budget):
    response = ws.score(pipeline_db, _request(bundle, all_heads))
    for head in all_heads:
        bars = [json.dumps(s["bars"][head["label"]], sort_keys=True) for s in response["sets"]]
        assert len(set(bars)) == 1, head["label"]
    calibration = next(s for s in response["sets"] if s["role"] == "calibration")
    assert "scores" not in calibration and "labels" not in calibration and "metrics" not in calibration


# ── 7.4 nothing leaks ─────────────────────────────────────────────────────────


def test_no_text_or_activation_in_the_response(pipeline_db, bundle, all_heads, wsp_world, heldout_arrow,
                                               empty_budget):
    from datasets import load_from_disk

    response = ws.score(pipeline_db, _request(bundle, all_heads))
    assert set(response) == {"kind", "bundle_sha256", "heldout_ref_sha256", "function", "sets", "scored_at",
                             "call_id"}
    assert set(response["function"]) == {"rule", "params", "layer", "scope", "skip_template_prefix"}
    for entry in response["sets"]:
        base = {"name", "view_id", "role", "row_ids_sha256", "scored_row_ids_sha256", "rows", "excluded", "bars"}
        assert set(entry) == (base if entry["role"] == "calibration" else base | {"labels", "scores", "metrics"})
        assert set(entry["excluded"]) == {"truncated", "truncated_fired"}
    text = json.dumps(response)
    for row_text in load_from_disk(str(heldout_arrow))["inputs"]:
        assert row_text not in text
    import re

    assert not [m for m in re.findall(r"[A-Za-z0-9+/=]{88,}", text) if not re.fullmatch(r"[0-9a-f]+", m)]


# ── 7.5 one at a time, recorded once ──────────────────────────────────────────


def test_each_call_is_recorded_once(pipeline_db, bundle, empty_budget):
    from src.models.wsp import WspScoreCall

    candidate = _edited(bundle, seed=31)
    before = pipeline_db.query(WspScoreCall).count()
    response = ws.score(pipeline_db, _request(bundle, [_head("parent"), _head("candidate", candidate)]))
    calls = pipeline_db.query(WspScoreCall).all()
    assert len(calls) == before + 1
    call = pipeline_db.get(WspScoreCall, response["call_id"])
    from src.services.wsp_digests import head_tensors_sha256

    assert call.miforge_run_id == "wsr_test" and call.patch_sha256 == "0" * 64
    assert call.candidate_tensors_sha256 == head_tensors_sha256(candidate)
    assert call.heads == [{"label": "candidate", "head_tensors_sha256": head_tensors_sha256(candidate)}]
    digest = hashlib.sha256()
    for entry in response["sets"]:
        if "scores" in entry:
            digest.update(np.asarray(entry["scores"]["candidate"], dtype=np.float32).tobytes())
    assert call.candidate_scores_sha256 == digest.hexdigest() and call.labels == ["parent", "candidate"]


# ── R1 review fixes (OR1-044a-1, OR1-044a-5), 2026-11-16 ───────────────────────────────────────────────────────────


def test_a_calibration_basis_bundle_without_its_calibration_set_is_refused(pipeline_db, bundle, empty_budget):
    """OR1-044a-1: the bar basis is the MANIFEST's (`operating_point.threshold_source = calibration`), never "whichever
    set is present". With the calibration set gone from the bundle, a parent-only call used to answer
    `bars.parent = {threshold: None, source: practice, n_negatives: 9}` (the parent's real bar is about -0.074); it is
    refused `capture_not_ready`. Twin: the full bundle cuts on `calibration`. Control: restore the practice fallback →
    red."""
    import copy

    assert bundle.manifest["operating_point"]["threshold_source"] == "calibration", "fixture: a calibration basis"
    full = ws.score(pipeline_db, _request(bundle, [_head("parent")]))
    assert all(s["bars"]["parent"]["source"] == "calibration" for s in full["sets"])
    original = copy.deepcopy(bundle.manifest)
    trimmed = copy.deepcopy(original)
    trimmed["heldout_ref"]["sets"] = [s for s in trimmed["heldout_ref"]["sets"] if not s["name"].startswith("calibration")]
    assert len(trimmed["heldout_ref"]["sets"]) < len(original["heldout_ref"]["sets"]), "fixture: no calibration set"
    bundle.manifest = trimmed
    pipeline_db.commit()
    try:
        _refused(pipeline_db, _request(bundle, [_head("parent")]), "capture_not_ready")
    finally:
        bundle.manifest = original
        pipeline_db.commit()


@pytest.fixture
def late_clock(monkeypatch):
    """A clock that reads 0 until `late["now"]` is set, then far past any deadline."""
    late = {"now": False}
    monkeypatch.setattr(ws, "_NOW", lambda: 1e12 if late["now"] else 0.0)
    return late


def test_the_deadline_is_checked_after_the_bars(pipeline_db, bundle, empty_budget, late_clock, monkeypatch):
    """OR1-044a-5: the clock crosses ONLY while the bars are cut; the bars' own check refuses. Control: drop the check
    after the bars → red (the per-set checks all passed before)."""
    real = ws.cut_bar

    def cut_then_late(*args, **kwargs):
        late_clock["now"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(ws, "cut_bar", cut_then_late)
    refused = _refused(pipeline_db, _request(bundle, [_head("parent")]), "score_timeout")
    assert "(bars)" in refused.message, refused.message


def test_the_deadline_is_checked_after_each_set(pipeline_db, bundle, empty_budget, late_clock, monkeypatch):
    """OR1-044a-5: the clock crosses while the FIRST set is scored; that set's own check refuses, naming it. Control:
    drop the per-set check → red (only the bars' check fires, naming the bars)."""
    from src.services import probe_monitor_run

    real = probe_monitor_run._aggregate_rows_many

    def score_then_late(*args, **kwargs):
        late_clock["now"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(probe_monitor_run, "_aggregate_rows_many", score_then_late)
    refused = _refused(pipeline_db, _request(bundle, [_head("parent")]), "score_timeout")
    assert "(set " in refused.message, refused.message


# ── R2 review fixes (OR2-044a-M3, -M4, -L5), 2026-11-17: the practice basis, the contract, the bar target ─────────


def _practice_bar_rows(bundle):
    """Independently of the scorer: the bundle's practice rows a practice-basis bar is cut on (validation negatives
    not truncated), and every practice negative (to show the fixture tells the two rules apart)."""
    directory = Path(bundle.dir_path)
    practice = bundle.manifest["practice"]
    offsets = np.load(directory / "practice_offsets.npy")
    labels = np.load(directory / "practice_labels.npy")
    validation = np.load(directory / "practice_validation_mask.npy").astype(bool)
    truncated = np.load(directory / "practice_truncated_mask.npy").astype(bool)
    name = next(f["name"] for f in bundle.manifest["files"] if f["role"] == "practice_activations")
    memmap = np.memmap(directory.parents[2] / name, dtype=np.dtype(practice["dtype"]), mode="r",
                       shape=(int(practice["tokens"]), int(practice["d_model"])))

    def rows(mask):
        return [memmap[int(offsets[i]):int(offsets[i + 1])] for i in np.flatnonzero(mask)]

    return rows(validation & ~truncated & (labels == 0)), rows(labels == 0)


def _independent_bar(bundle, arrays, rows, target):
    from src.services.probe_monitor_run import _aggregate_rows
    from src.services.probe_monitor_trainer import calibrate

    function = bundle.manifest["function"]["linear_probe"]
    head = ws.to_probe_head(arrays, int(function["layer"]))
    params = {k: v for k, v in function["params"].items() if v is not None}
    scores = np.asarray(_aggregate_rows(head, function["rule"], params, rows), dtype=np.float32)
    return calibrate([float(s) for s in scores], target_fpr=target, source="x").threshold


def test_practice_basis_bars_are_cut_on_the_validation_negatives(pipeline_db, pb_bundle, pb_probe, empty_budget):
    """OR2-044a-M3: on a practice-basis bundle every head's bar is cut on the practice VALIDATION negatives that were
    not truncated — computed here from the bundle's own arrays — and the parent's equals the bar as served (nothing is
    truncated on this fixture), which the bundle records (`scorer_parent_bar`). Control: cut on every practice
    negative (review mutation M19) → red."""
    candidate = _edited(pb_bundle, seed=81, entries=3, delta=0.5)
    response = ws.score(pipeline_db, _request(pb_bundle, [_head("parent"), _head("candidate", candidate)]))
    validation, every = _practice_bar_rows(pb_bundle)
    target = float(pb_probe.target_fpr)
    for label, arrays in (("parent", _parent(pb_bundle)), ("candidate", candidate)):
        expected = _independent_bar(pb_bundle, arrays, validation, target)
        assert expected != _independent_bar(pb_bundle, arrays, every, target), "fixture: the rules must differ"
        for entry in response["sets"]:
            bar = entry["bars"][label]
            assert (bar["threshold"], bar["source"], bar["n_negatives"]) == (expected, "practice", len(validation))
    assert np.float32(response["sets"][0]["bars"]["parent"]["threshold"]) == np.float32(pb_probe.threshold)
    recorded = pb_bundle.scorer_parent_bar
    assert recorded["same"] is True and recorded["difference"] == pytest.approx(0.0, abs=1e-6)
    assert recorded["scorer"] == response["sets"][0]["bars"]["parent"]
    assert recorded["served"] == {"threshold": pb_probe.threshold, "threshold_source": "practice"}


def test_a_truncated_validation_negative_does_not_place_the_bar(pipeline_db, pb_probe, pb_captures, tiny_tokenizer,
                                                                empty_budget, monkeypatch):
    """G5 (GR-6) on the practice basis: the highest-scoring validation negative is cut at `max_length` (marked in the
    rebuilt practice, the one input the build takes it from). Its bar no longer counts it, so the scorer's parent bar
    differs from the bar as served (the served validation-negatives bar keeps every negative: that path records no
    mask), and the bundle reports both and their difference, as FR-1 asks; the summary view carries it. Controls: keep
    truncated rows in `practice_bar_indices` → red; drop `scorer_parent_bar` from the build → red."""
    from src.api.v1.endpoints.wsp import bundle_view
    from src.services import wsp_bundle
    from src.services.probe_monitor_run import _aggregate_rows, load_probe

    head, trained = load_probe(pipeline_db, pb_probe.id)
    real = wsp_bundle.rebuild_practice
    marked = {}

    def with_a_truncated_negative(*args, **kwargs):
        practice = real(*args, **kwargs)
        rows = np.flatnonzero(practice.validation_mask.astype(bool) & (practice.labels == 0))
        scores = _aggregate_rows(head, trained.rule, dict(trained.rule_params or {}), practice.rows(rows))
        marked["row"] = int(rows[int(np.argmax(scores))])
        practice.truncated_mask[marked["row"]] = 1
        return practice

    row = new_bundle(pipeline_db, pb_probe)
    with monkeypatch.context() as patch:
        patch.setattr(wsp_bundle, "rebuild_practice", with_a_truncated_negative)
        wsp_bundle.build_bundle(pipeline_db, row, tokenizer=tiny_tokenizer, max_seconds=120)
    pipeline_db.refresh(row)
    assert row.status == "ready", row.error
    assert int(np.load(Path(row.dir_path) / "practice_truncated_mask.npy")[marked["row"]]) == 1
    response = ws.score(pipeline_db, _request(row, [_head("parent")]))
    validation, _ = _practice_bar_rows(row)
    expected = _independent_bar(row, _parent(row), validation, float(pb_probe.target_fpr))
    bar = response["sets"][0]["bars"]["parent"]
    assert (bar["threshold"], bar["n_negatives"]) == (expected, len(validation))
    assert expected is None or np.float32(expected) != np.float32(pb_probe.threshold), \
        "fixture: the truncated negative placed the served bar"
    shown = bundle_view(pipeline_db, row)["parent_bar"]
    assert shown["scorer"] == bar and shown["served"]["threshold"] == pb_probe.threshold and shown["same"] is False
    assert shown["difference"] == (None if expected is None else pytest.approx(expected - pb_probe.threshold))


def test_a_practice_basis_bundle_ignores_a_calibration_capture(pipeline_db, bundle, empty_budget):
    """OR1-044a-1's scorer half, pinned (review mutation M13 survived): a bundle whose manifest states the PRACTICE
    basis cuts every bar on practice rows even when it holds a calibration capture. Control: keep the calibration
    capture on a practice basis → red (the bars read `calibration`)."""
    import copy

    original = copy.deepcopy(bundle.manifest)
    assert any(s["role"] == "calibration" for s in original["heldout_ref"]["sets"]), "fixture: a calibration capture"
    practice = copy.deepcopy(original)
    practice["operating_point"]["threshold_source"] = "practice"
    bundle.manifest = practice
    pipeline_db.commit()
    try:
        response = ws.score(pipeline_db, _request(bundle, [_head("parent")]))
    finally:
        bundle.manifest = original
        pipeline_db.commit()
    validation, _ = _practice_bar_rows(bundle)
    expected = _independent_bar(bundle, _parent(bundle), validation, float(original["operating_point"]["target_fpr"]))
    for entry in response["sets"]:
        assert (entry["bars"]["parent"]["source"], entry["bars"]["parent"]["threshold"]) == ("practice", expected)


def test_a_bundle_without_an_operating_point_cuts_at_the_probes_target(pipeline_db, pb_bundle, pb_probe,
                                                                         empty_budget):
    """OR2-044a-L5: with `operating_point` null (a probe with no bar), the bars are cut at the probe's own target FPR
    (0.25 here), never a constant 0.01. Control: restore the 0.01 fallback → red."""
    import copy

    assert float(pb_probe.target_fpr) == 0.25, "fixture: a target other than the old fallback"
    original = copy.deepcopy(pb_bundle.manifest)
    pb_bundle.manifest = {**copy.deepcopy(original), "operating_point": None}
    pipeline_db.commit()
    try:
        response = ws.score(pipeline_db, _request(pb_bundle, [_head("parent")]))
    finally:
        pb_bundle.manifest = original
        pipeline_db.commit()
    bar = response["sets"][0]["bars"]["parent"]
    assert (bar["target_fpr"], bar["source"]) == (0.25, "practice")


@pytest.mark.parametrize("basis", ["calibration", "practice"])
def test_every_answer_validates_against_the_published_contract(pipeline_db, basis, request, empty_budget,
                                                               every_answer_is_the_contracts):
    """OR2-044a-M4 (review mutation M09 survived): an answer with every head kind, on each basis, is the published
    contract's — the response model and the committed JSON schema. (The autouse wrapper checks every other answer in
    this module the same way.)"""
    from src.models.wsp import WspBaseline

    target = request.getfixturevalue("bundle" if basis == "calibration" else "pb_bundle")
    kinds = [b.kind for b in pipeline_db.query(WspBaseline).filter(WspBaseline.bundle_id == target.id).all()
             if not (b.fit or {}).get("pending") and not (b.fit or {}).get("refused")]
    heads = [_head("parent"), _head("candidate", _edited(target, seed=91)),
             _head("control", _edited(target, seed=92))] + [_head(f"baseline:{k}") for k in kinds]
    response = ws.score(pipeline_db, _request(target, heads))
    assert every_answer_is_the_contracts and every_answer_is_the_contracts[-1] is response
    assert {s["role"] for s in response["sets"]} == ({"eval", "calibration"} if basis == "calibration" else {"eval"})
    assert all(entry["bars"]["parent"]["source"] == basis for entry in response["sets"])


def test_a_calibration_set_carrying_rows_fails_the_contract(pipeline_db, bundle, empty_budget):
    """R3 review fix (OR3-044a-L5): "the calibration set carries no scores, labels or metrics" is in the published
    schema, not only its prose, so miForge's validator refuses a calibration row that carries any of them. The real
    calibration-basis answer validates; the same answer with the eval set's `scores`, `labels` or `metrics` copied onto
    the calibration set fails the committed schema AND the response model, while the eval set carrying them is fine.
    Control: drop the schema's `if role == calibration` clause (regenerated) → red."""
    import copy

    from jsonschema import Draft202012Validator
    from pydantic import ValidationError

    from src.schemas.wsp_scoring import WspScoreResponse

    response = ws.score(pipeline_db, _request(bundle, [_head("parent")]))
    by_role = {entry["role"]: i for i, entry in enumerate(response["sets"])}
    assert set(by_role) == {"eval", "calibration"}
    calibration, evaluated = response["sets"][by_role["calibration"]], response["sets"][by_role["eval"]]
    assert not {"scores", "labels", "metrics"} & set(calibration) and {"scores", "labels", "metrics"} <= set(evaluated)
    validator = Draft202012Validator(CONTRACT["response"])
    for field in ("scores", "labels", "metrics"):
        carrying = copy.deepcopy(response)
        carrying["sets"][by_role["calibration"]][field] = copy.deepcopy(evaluated[field])
        errors = list(validator.iter_errors(carrying))
        assert errors and any(list(e.absolute_path)[:2] == ["sets", by_role["calibration"]] for e in errors), field
        with pytest.raises(ValidationError, match="calibration set carries no"):
            WspScoreResponse.model_validate(carrying)
    # Null is "carries none": absent or null both validate.
    nulled = copy.deepcopy(response)
    nulled["sets"][by_role["calibration"]].update(scores=None, labels=None, metrics=None)
    assert list(validator.iter_errors(nulled)) == []
    WspScoreResponse.model_validate(nulled)
