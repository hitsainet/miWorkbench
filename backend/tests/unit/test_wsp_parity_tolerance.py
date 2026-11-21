"""R1 review fix (OR1-044a-4), 2026-11-16: the capture parity statistics, exercised directly (no model, no database).

The per-row band (`ROW_ATOL + ROW_RTOL·|stored|`) is tested in both directions, the AUROC criterion and the
caught-at-the-bar count are each pinned on their own (with the row band opened, so only the criterion under test can
fail), and an everyday-text set compares its false alarms.

Controls (0xcc/reviews/044_implementation_controls_2026-11-16.md, OR1-044a-4-*): widen ROW_ATOL ×10 →
`test_just_outside_the_band_fails` red; narrow it ÷10 → `test_inside_the_band_passes` red; drop the AUROC check →
`test_the_auroc_criterion_alone_fails_a_reordered_capture` red; drop the count check → `test_the_count_alone_...` red.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.services import wsp_capture as wc

STORED = np.array([-2.0, -1.0, -0.5, 0.25, 0.75, 1.5, 2.5, 3.0], dtype=np.float32)
LABELS = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int8)
MASK = np.ones(len(STORED), dtype=bool)
PROBE = SimpleNamespace(threshold=0.5, target_fpr=0.25)


def _planned(role="eval", scores=STORED, labels=LABELS):
    return SimpleNamespace(role=role, scores=scores, labels=labels)


def _band(stored):
    # The SPEC's band, written out (1e-3 + 1e-3·|stored|), not read from the module: a test that reads the constant
    # it checks moves with it and pins nothing.
    return 1e-3 + 1e-3 * np.abs(stored.astype(np.float64))


def test_inside_the_band_passes():
    capture = (STORED.astype(np.float64) + 0.9 * _band(STORED)).astype(np.float32)
    out = wc._statistics(capture, _planned(), MASK, PROBE)
    assert out["rows_outside_tolerance"] == 0 and out["passed"], out


def test_just_outside_the_band_fails():
    capture = STORED.astype(np.float64).copy()
    capture[3] += 1.1 * _band(STORED)[3]
    out = wc._statistics(capture.astype(np.float32), _planned(), MASK, PROBE)
    assert out["rows_outside_tolerance"] == 1 and not out["passed"], out


@pytest.fixture
def open_band(monkeypatch):
    monkeypatch.setattr(wc, "ROW_ATOL", 1e9)


def test_the_auroc_criterion_alone_fails_a_reordered_capture(open_band):
    """Swap a positive below a negative while the caught count at the bar stays equal: only AUROC can fail it."""
    capture = STORED.copy()
    capture[3], capture[2] = -0.6, 0.3   # positive 0.25 → -0.6 (still below the bar), negative -0.5 → 0.3 (below)
    out = wc._statistics(capture, _planned(), MASK, PROBE)
    assert out["count_stored"] == out["count_capture"], "fixture: the count must agree"
    assert out["auroc_capture"] != out["auroc_stored"] and not out["passed"], out


def test_the_count_alone_fails_a_capture_that_moves_one_positive_across_the_bar(open_band):
    """Every row's ORDER kept (AUROC equal), one positive moved across the bar: only the count can fail it."""
    capture = STORED.copy()
    capture[3] = 0.6   # still between -0.5 and 0.75, so the ranking is unchanged; now above the 0.5 bar
    out = wc._statistics(capture, _planned(), MASK, PROBE)
    assert out["auroc_capture"] == out["auroc_stored"], "fixture: the ranking must agree"
    assert out["count_stored"] != out["count_capture"] and not out["passed"], out


def test_an_everyday_text_set_compares_its_false_alarms(open_band):
    negatives = np.zeros(len(STORED), dtype=np.int8)
    same = wc._statistics(STORED.copy(), _planned("everyday_text", labels=negatives), MASK, PROBE)
    assert same["passed"] and same["count_stored"] == same["count_capture"]
    moved = STORED.copy()
    moved[2] = 0.6   # one more row over the bar: one more false alarm
    out = wc._statistics(moved, _planned("everyday_text", labels=negatives), MASK, PROBE)
    assert out["count_capture"] == out["count_stored"] + 1 and not out["passed"], out
