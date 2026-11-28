"""ENH-010 (044 task 1.1, T44-D3, FTID I44-D2): `_aggregate_rows_many` is `_aggregate_rows`, bit for bit, per head.

`_FROZEN` below is a COPY of `_aggregate_rows` as it stood on 2026-11-15 (miWorkbench `e393381ec`), before it began to
delegate. The WSP scorer scores a parent, a candidate, controls and five baselines on the same captured rows through
the new function; a candidate that "improved" because the arithmetic changed would be a fabricated verdict, so the
pins compare with `np.array_equal` on float32, never with a tolerance.

Control (0xcc/reviews/044_implementation_controls_2026-11-16.md): change the packing dtype to float16 in the new
function → both tests red.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pytest
import torch

from src.ml.probe_monitor_model import RULES, ProbeHead
from src.services.probe_monitor_run import _aggregate_rows, _aggregate_rows_many

D_MODEL = 24


def _FROZEN(head: Any, rule: str, rule_params: Dict[str, Any], rows: Sequence[Any]) -> List[float]:
    """`services/probe_monitor_run.py::_aggregate_rows` at e393381ec, verbatim but for the imports and for the two
    rules B5a (2026-11-24) made part of how ANY probe score is computed: the head's matmul vectors in fresh aligned
    allocations (`steady_head`) and each matmul on one MKL thread (`one_blas_thread`). Without them this reference
    ran gemv at the process's thread count, and on MKL's Intel AVX-512 kernels (an Intel CI runner) a row's bits
    change with the thread count, so 27 pins went red on CI run 45650762124 while nothing in the bucket, mask or
    padding arithmetic had moved (B5c, record `0xcc/reviews/b5c_test_reference_2026-11-25.md`). Everything else
    (fresh `torch.zeros` buckets, a float32 cast per row, `token_scores` per head) is the old function's own."""
    from src.ml.probe_monitor_model import combine
    from src.services.packed_scoring import one_blas_thread, steady_head
    from src.services.probe_monitor_trainer import plan_length_buckets, slots_for

    if not rows:
        return []
    head = steady_head(head)
    widths = [int(np.asarray(r).shape[0]) for r in rows]
    d_model = int(np.asarray(rows[0]).shape[1])
    out = [0.0] * len(rows)
    for indices in plan_length_buckets(widths, budget_slots=slots_for(d_model)):
        width = max(1, max(widths[i] for i in indices))
        packed = torch.zeros((len(indices), width, d_model), dtype=torch.float32)
        mask = torch.zeros((len(indices), width), dtype=torch.bool)
        for position, row_index in enumerate(indices):
            array = np.asarray(rows[row_index], dtype=np.float32)
            packed[position, : array.shape[0]] = torch.from_numpy(array)
            mask[position, : array.shape[0]] = True
        with one_blas_thread():
            scores = head.token_scores(packed)
            logits = head.attention_logits(packed) if rule == "attention" else None
        aggregates = combine(
            rule, scores, mask=mask, attention_logits=logits,
            **{k: v for k, v in rule_params.items() if k in ("tau", "window")},
        )
        for row_index, value in zip(indices, aggregates.detach().cpu().tolist()):
            out[row_index] = float(value)
    return out


def _head(seed: int) -> ProbeHead:
    generator = torch.Generator().manual_seed(seed)
    return ProbeHead(
        weight=torch.randn(D_MODEL, generator=generator), bias=0.1 * seed,
        mean=torch.randn(D_MODEL, generator=generator) * 0.1, std=torch.rand(D_MODEL, generator=generator) + 0.5,
        attention_query=torch.randn(D_MODEL, generator=generator), layer=3,
    )


def _rows(n: int = 37) -> List[np.ndarray]:
    rng = np.random.default_rng(5)
    # Ragged widths (1..90 tokens), stored as float16 like a capture, so padding and buckets are exercised.
    return [rng.normal(size=(int(w), D_MODEL)).astype(np.float16) for w in rng.integers(1, 90, size=n)]


def _params(rule: str) -> Dict[str, Any]:
    return {"window": 4} if rule == "rolling_mean_max" else ({"tau": 0.7} if rule == "softmax" else {})


@pytest.mark.parametrize("rule", RULES)
def test_one_head_equals_the_old_function(rule):
    rows, head = _rows(), _head(1)
    want = np.asarray(_FROZEN(head, rule, _params(rule), rows), dtype=np.float32)
    got = np.asarray(_aggregate_rows(head, rule, _params(rule), rows), dtype=np.float32)
    assert np.array_equal(got, want), f"{rule}: the delegating _aggregate_rows moved"


@pytest.mark.parametrize("rule", RULES)
def test_many_equals_each_alone(rule):
    rows, heads = _rows(), [_head(s) for s in (1, 2, 3, 4)]
    together = _aggregate_rows_many(heads, rule, _params(rule), rows)
    assert len(together) == len(heads) and all(len(scores) == len(rows) for scores in together)
    for head, scores in zip(heads, together):
        alone = _FROZEN(head, rule, _params(rule), rows)
        assert np.array_equal(np.asarray(scores, dtype=np.float32), np.asarray(alone, dtype=np.float32)), rule
    assert len({tuple(s) for s in together}) == len(heads), "fixture too weak: the heads score alike"


def test_no_rows_is_one_empty_list_per_head():
    assert _aggregate_rows_many([_head(1), _head(2)], "mean", {}, []) == [[], []]


# ── 2026-11-22: the D128 scorer rewrite (`packed_scoring`) changes memory traffic, never a float ───────────────────
#
# Controls (0xcc/reviews/wsp_score_d128_2026-11-22.md): standardise with `sub` and `div` fused through a reciprocal
# (`mul_(1 / divisor)`) → the kinds test red; matmul over half-bucket slices → red; drop the zero padding → the mean
# test red; score every head on its own `token_scores` again → the allocation guard red.


class _DuckHead:
    """Not a `ProbeHead`: scored by its own methods on the float32 bucket, as before."""

    def __init__(self, inner):
        self.inner = inner

    def token_scores(self, activations):
        return self.inner.token_scores(activations) * 2.0

    def attention_logits(self, activations):
        return self.inner.attention_logits(activations)


class _OwnStandardise(ProbeHead):
    """A subclass that overrides the arithmetic `packed_scoring` reproduces: never grouped."""

    def standardise(self, activations):
        return super().standardise(activations) + 0.5


def _kinds(d: int = D_MODEL):
    """Every head shape the grouping must handle: two sharing statistics bit for bit, one with its own, one with
    degenerate channels (std at or below eps: zeroed, not divided), mean only, std only, neither, a duck and a
    subclass. Each has an attention query (the `attention` rule's own path)."""
    g = torch.Generator().manual_seed(9)
    mean, std = torch.randn(d, generator=g) * 0.1, torch.rand(d, generator=g) + 0.5
    degenerate = std.clone()
    degenerate[:3] = 0.0
    degenerate[3] = 1e-7

    def head(**kw):
        base = dict(weight=torch.randn(d, generator=g), bias=0.3, mean=mean, std=std,
                    attention_query=torch.randn(d, generator=g), layer=3)
        base.update(kw)
        return ProbeHead(**base)

    shared = [head(), head(), head(mean=mean.clone(), std=std.clone())]   # equal bits, separate tensors: one group
    return shared + [
        head(mean=torch.randn(d, generator=g), std=torch.rand(d, generator=g) + 0.2),
        head(std=degenerate),
        head(std=None), head(mean=None), head(mean=None, std=None),
        _DuckHead(head()), _OwnStandardise(**vars(head())),
    ]


def _ragged(dtype, n: int = 41, d: int = D_MODEL, seed: int = 5, widest: int = 300) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    widths = rng.integers(1, widest, size=n)
    if dtype == "mixed":
        return [rng.normal(size=(int(w), d)).astype(np.float16 if i % 2 else np.float32)
                for i, w in enumerate(widths)]
    return [rng.normal(size=(int(w), d)).astype(dtype) for w in widths]


@pytest.fixture
def small_buckets(monkeypatch):
    """Many buckets (a 600-slot budget), standardisation slices that cut through rows, and the pack pool on, with row
    pieces that cut through rows too: every boundary the rewrite introduced is crossed."""
    from src.services import packed_scoring, probe_monitor_trainer

    monkeypatch.setattr(probe_monitor_trainer, "slots_for", lambda d_model, budget_bytes=None: 600)
    monkeypatch.setattr(packed_scoring, "STANDARDISE_SLICE_TOKENS", 37)
    monkeypatch.setattr(packed_scoring, "PACK_PIECE_TOKENS", 29)
    monkeypatch.setattr(packed_scoring, "PACK_POOL_MIN_BYTES", 0)
    before = torch.get_num_threads()
    torch.set_num_threads(max(2, before))
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, "mixed"])
@pytest.mark.parametrize("rule", RULES)
def test_every_head_kind_and_row_dtype_equals_the_old_function(rule, dtype, small_buckets):
    from src.services.probe_monitor_trainer import plan_length_buckets, slots_for

    rows, heads = _ragged(dtype), _kinds()
    assert len(plan_length_buckets([r.shape[0] for r in rows], budget_slots=slots_for(D_MODEL))) > 5
    together = _aggregate_rows_many(heads, rule, _params(rule), rows)
    for index, (head, scores) in enumerate(zip(heads, together)):
        alone = _FROZEN(head, rule, _params(rule), rows)
        assert np.array_equal(np.asarray(scores, dtype=np.float32), np.asarray(alone, dtype=np.float32)), \
            f"{rule} {dtype}: head {index} ({type(head).__name__}) moved"
    assert len({tuple(s) for s in together}) == len(heads), "fixture too weak: two heads score alike"


@pytest.mark.parametrize("rule", ["max", "mean"])
def test_production_width_buckets_equal_the_old_function(rule, small_buckets, monkeypatch):
    """d_model 4096, float16 rows, the production slice and piece sizes, several buckets: the whole-bucket matmul at
    the width MKL's kernels actually run (a mismatch from slicing or misalignment shows here first)."""
    from src.services import packed_scoring, probe_monitor_trainer

    monkeypatch.setattr(packed_scoring, "STANDARDISE_SLICE_TOKENS", 256)
    monkeypatch.setattr(packed_scoring, "PACK_PIECE_TOKENS", 1024)
    monkeypatch.setattr(probe_monitor_trainer, "slots_for", lambda d_model, budget_bytes=None: 2048)
    rows, heads = _ragged(np.float16, n=24, d=4096, seed=7, widest=700), _kinds(4096)[:5]
    together = _aggregate_rows_many(heads, rule, {}, rows)
    for head, scores in zip(heads, together):
        assert np.array_equal(np.asarray(scores, dtype=np.float32),
                              np.asarray(_FROZEN(head, rule, {}, rows), dtype=np.float32)), rule


def test_a_bucket_allocates_no_bucket_sized_temporary_per_head(monkeypatch):
    """The performance guard, with no clock in it (memory `check-ci-not-just-local-green`): the bytes the call
    allocates, as torch's profiler counts them, are the reused bucket buffers plus per-TOKEN scores, independent of
    how many heads and buckets there are. The old loop allocated a bucket and four bucket-sized temporaries per head
    per bucket (D128: 281 buckets x 6 heads x 4 x 512 MiB). The second assertion proves the guard can fail."""
    from torch.profiler import ProfilerActivity, profile

    from src.services import probe_monitor_trainer
    from src.services.probe_monitor_trainer import plan_length_buckets

    monkeypatch.setattr(probe_monitor_trainer, "slots_for", lambda d_model, budget_bytes=None: 1000)
    d = 256
    rows, heads = _ragged(np.float16, n=200, d=d, seed=3, widest=120), _kinds(d)[:6]
    widths = [r.shape[0] for r in rows]
    plan = plan_length_buckets(widths, budget_slots=1000)
    capacity = max(len(ix) * max(widths[i] for i in ix) for ix in plan)
    assert len(plan) > 8

    def allocated(call):
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            call()
        return sum(max(0, event.self_cpu_memory_usage) for event in prof.events())

    bound = 3 * capacity * d * 4 + 32 * sum(widths) * len(heads)
    new = allocated(lambda: _aggregate_rows_many(heads, "max", {}, rows))
    old = allocated(lambda: [_FROZEN(head, "max", {}, rows) for head in heads])
    assert new <= bound, f"{new} bytes allocated; the bucket buffers and per-token scores are {bound}"
    assert old > 4 * bound, f"fixture too weak: the old loop allocated only {old} bytes"


@pytest.mark.parametrize("rule", ["mean", "softmax"])
def test_padding_is_zero_whatever_the_reused_buffer_held(rule, small_buckets, monkeypatch):
    """The bucket buffers are reused (and `torch.empty` hands back whatever memory held): a pad position must be zero
    as the old `torch.zeros` bucket was, or a rule that multiplies pads by the mask (`mean`) turns a NaN pad into a NaN
    score. The buffers start as NaN here. Control: drop `rows[position, length:] = 0` in `BucketBuffers.pack` → red."""
    from src.services import packed_scoring

    real = packed_scoring.BucketBuffers._alloc

    def poisoned(self, name, dtype):
        buffer = real(self, name, dtype)
        buffer.fill_(float("nan"))
        return buffer

    monkeypatch.setattr(packed_scoring.BucketBuffers, "_alloc", poisoned)
    rows, heads = _ragged(np.float16), _kinds()[:4]
    together = _aggregate_rows_many(heads, rule, _params(rule), rows)
    for head, scores in zip(heads, together):
        assert np.array_equal(np.asarray(scores, dtype=np.float32),
                              np.asarray(_FROZEN(head, rule, _params(rule), rows), dtype=np.float32)), rule
