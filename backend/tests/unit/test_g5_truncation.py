"""ENH-010 G5: a truncated evaluation row is reported, not scored as a negative or a positive (GR-1, GR-2).

`_evaluate_probes_on_sets` and `_prepare_examples` run for real over an Arrow dataset on disk; only the renderer and
the model forward are faked. The fake renderer marks a row truncated when its text carries `LONG`, the way
`RenderedExample.keep_tail` sets the flag on a row longer than `max_length`. The fixture's twin rows differ ONLY in
that flag: the truncated negatives are scored highest, so a metric that counts them moves.

Controls (0xcc/reviews/G5_implementation_controls_2026-11-16.md):
  GR-1  drop the `truncated_path` entry from the save loop          → test_evaluation_stores_a_truncated_mask red
  GR-2  pass all rows to the untruncated `evaluate_scores` call      → test_metrics_untruncated_excludes_truncated_rows red
  GR-2  filter the all-rows call                                     → test_all_rows_auroc_is_unchanged red
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.core.config import settings
from src.models.dataset import Dataset
from src.models.probe_monitor import ProbeMonitorDataset
from src.services import probe_monitor_capture, probe_monitor_render, probe_monitor_run
from src.services.probe_monitor_metrics import auroc

#: Negative rows cut at `max_length` in the fixture: the 2nd, 4th and 6th negatives.
TRUNCATED_NEGATIVES = (3, 7, 11)


def _text(i: int) -> str:
    return ("pos" if i % 2 == 0 else "neg") + (" LONG" if i in TRUNCATED_NEGATIVES else "")


def _make_world(tmp_path, monkeypatch, n_rows):
    from datasets import Dataset as HFDataset

    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    raw = tmp_path / "datasets" / f"org_g5_{n_rows}"
    HFDataset.from_dict({
        "text": [_text(i) for i in range(n_rows)],
        "stakes": ["high" if i % 2 == 0 else "low" for i in range(n_rows)],
    }).save_to_disk(str(raw))

    def fake_render_all(tokenizer, conversations, **kwargs):
        rendered = []
        for conversation in conversations:
            text = " ".join(m["content"] for m in conversation)
            rendered.append(SimpleNamespace(text=text, truncated="LONG" in text))
        return rendered, SimpleNamespace(as_dict=lambda: {})

    monkeypatch.setattr(probe_monitor_render, "render_all", fake_render_all)

    def fake_forward_scores_many(model, rendered, specs, **kwargs):
        # Positives 1.0..; untruncated negatives 0.0..; the truncated negatives fire hardest of all (the pilot:
        # every truncated row fired), so they are what an all-rows metric gets wrong.
        out = []
        for _spec in specs:
            rows = []
            for k, r in enumerate(rendered):
                if r.truncated:
                    value = 10.0 + k * 0.001
                elif r.text.startswith("pos"):
                    value = 1.0 + k * 0.001
                else:
                    value = 0.0 + k * 0.001
                rows.append(SimpleNamespace(aggregate=value, n_scored=4))
            out.append(rows)
        return out

    monkeypatch.setattr(probe_monitor_capture, "forward_scores_many", fake_forward_scores_many)
    # The fake database answers only the dataset and view lookups. Since 2026-11-17 the per-set commit is
    # followed by `recompute_rung`, which needs the real tables; it is pinned on its own by
    # test_rung_follows_each_committed_set.py, so here it is a no-op.
    monkeypatch.setattr(probe_monitor_run, "recompute_rung", lambda *_a, **_k: None)
    dataset = SimpleNamespace(id="ds-g5", raw_path=str(raw))
    view = SimpleNamespace(
        id="pmd_g5", dataset_id="ds-g5", input_column="text", label_column="stakes", pair_column=None,
        label_mapping={"high": "positive", "low": "negative"}, keyword_filter=None, split=None, name="g5",
        distribution="out_of_distribution", role="eval",
    )
    artifact_dir = tmp_path / "probe_monitors" / "pmr_g5"
    artifact_dir.mkdir(parents=True)
    return {"dataset": dataset, "view": view, "dir": artifact_dir}


class FakeDB:
    def __init__(self, world):
        self.world, self.added, self.commits = world, [], 0

    def query(self, model):
        target = {Dataset: self.world["dataset"], ProbeMonitorDataset: self.world["view"]}[model]
        return SimpleNamespace(filter=lambda *a, **k: SimpleNamespace(first=lambda: target))

    def add(self, row):
        self.added.append(row)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def _evaluate(world, threshold=0.5):
    db = FakeDB(world)
    probe = SimpleNamespace(id="pm_g5", target_fpr=None, variant="dense", sae_id=None, threshold=threshold)
    context = SimpleNamespace(artifact_dir=world["dir"], scope="all", max_length=64, skip_template_prefix=False,
                              target_fpr=0.1, seed=0)
    probe_monitor_run._evaluate_probes_on_sets(
        db, [(probe, object(), SimpleNamespace(rule="mean", rule_params={}), None, None)], context, None,
        SimpleNamespace(pad_token_id=0), "LlamaForCausalLM", ["pmd_g5"],
    )
    (row,) = db.added
    return row


def test_evaluation_stores_a_truncated_mask(tmp_path, monkeypatch):
    """GR-1: a fifth aligned array, int8, 1 exactly at the truncated rows."""
    row = _evaluate(_make_world(tmp_path, monkeypatch, 40))
    assert row.truncated_path is not None and row.truncated_path.endswith("__truncated.npy")
    mask = np.load(row.truncated_path)
    assert mask.dtype == np.int8 and len(mask) == 40
    assert np.flatnonzero(mask).tolist() == list(TRUNCATED_NEGATIVES)
    positions = np.load(row.positions_path)
    assert len(positions) == len(mask) == len(np.load(row.scores_path))


def test_metrics_untruncated_excludes_truncated_rows(tmp_path, monkeypatch):
    """GR-2: `metrics.untruncated` is `evaluate()` over the untruncated rows only, and differs from the all-rows
    figure because the truncated negatives score highest."""
    row = _evaluate(_make_world(tmp_path, monkeypatch, 60))
    scores = np.load(row.scores_path).astype(float)
    labels = np.load(row.labels_path)
    mask = np.load(row.truncated_path)
    keep = mask == 0
    want = auroc(scores[keep].tolist(), labels[keep].tolist())
    block = row.metrics["untruncated"]
    assert block["scored"] is True
    assert block["auroc"] == pytest.approx(want, abs=1e-12)
    assert (block["n_positive"], block["n_negative"]) == (30, 27)
    assert abs(block["auroc"] - row.metrics["auroc"]) > 1e-3, "fixture too weak: the exclusion moved nothing"
    # The rest of the truncated rows are recorded, not lost.
    assert row.metrics["truncation"] == {
        "rows": 3, "positives": 0, "negatives": 3, "fired_at_bar": 3, "threshold": 0.5, "max_length": 64,
        "policy": "excluded",
    }
    assert row.metrics["not_scored"] == {"truncated": 3}
    # The role's own count over the same rows: positives only, none truncated here.
    assert block["catch_rate"]["positives"] == 30


def test_all_rows_auroc_is_unchanged(tmp_path, monkeypatch):
    """GR-2 / P44: `metrics.auroc` keeps the 035 meaning, every row — the value the scores route serves as
    `reported_auroc` and miDataworks' `accept_source` recomputes over every served row."""
    row = _evaluate(_make_world(tmp_path, monkeypatch, 60))
    scores = np.load(row.scores_path).astype(float)
    labels = np.load(row.labels_path)
    assert row.metrics["auroc"] == pytest.approx(auroc(scores.tolist(), labels.tolist()), abs=1e-12)
    assert (row.metrics["n_positive"], row.metrics["n_negative"]) == (30, 30)
    assert row.metrics["catch_rate"]["positives"] == 30


def test_untruncated_block_refused_when_too_few(tmp_path, monkeypatch):
    """GR-2: 20 negatives, 3 truncated → 17 untruncated, below MIN_PER_CLASS: the block is refused
    `too_few_untruncated` and carries no AUROC, while the all-rows block still scores."""
    row = _evaluate(_make_world(tmp_path, monkeypatch, 40))
    assert row.metrics["scored"] is True and row.metrics["auroc"] is not None
    block = row.metrics["untruncated"]
    assert block["scored"] is False
    assert block["refused"] == "too_few_untruncated"
    assert "auroc" not in block or block["auroc"] is None
