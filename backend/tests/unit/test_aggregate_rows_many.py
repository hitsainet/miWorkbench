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
    """`services/probe_monitor_run.py::_aggregate_rows` at e393381ec, verbatim but for the imports."""
    from src.ml.probe_monitor_model import combine
    from src.services.probe_monitor_trainer import plan_length_buckets, slots_for

    if not rows:
        return []
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
