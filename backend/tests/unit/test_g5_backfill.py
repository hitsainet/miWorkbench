"""ENH-010 G5 (GR-5): the CPU backfill proves a pre-G5 evaluation's truncation mask, or refuses and writes nothing.

Real rows in the test database (model, dataset, view, run, probe, evaluation) and a real Arrow dataset on disk;
`backfill_truncation` runs for real, including `_prepare_examples`. Only the renderer is faked, and it builds REAL
`RenderedExample`s, so `_row_mask` — the capture's own per-row mask — counts the scored tokens exactly as it does on
the GPU path. Rows whose text carries `LONG` render truncated. The evaluation's stored arrays are what the
evaluator would have written before G5: scores, positions, n_scored, labels, and no mask.

Controls (0xcc/reviews/G5_implementation_controls_2026-11-16.md):
  accept a mismatch (drop the refusal)            → test_backfill_refuses_on_one_n_scored_mismatch red
  delete the route decorator                      → test_mcp_tools_call_real_routes.py red
  delete the `@mcp.tool` of backfill_probe_truncation → test_reachability.py red
"""
from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.services import probe_monitor_render
from src.services.probe_monitor_metrics import auroc
from src.services.probe_monitor_render import RenderedExample

N_ROWS = 60
TRUNCATED_ROWS = (1, 5, 9)          # three negatives


def _text(i: int) -> str:
    return ("pos" if i % 2 == 0 else "neg") + (" LONG" if i in TRUNCATED_ROWS else "") + f" row{i}"


@pytest.fixture
def sync_db(async_engine):
    url = async_engine.url.set(drivername=async_engine.url.drivername.replace("+asyncpg", ""))
    engine = create_engine(url)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


@pytest.fixture
def world(sync_db, tmp_path, monkeypatch):
    from datasets import Dataset as HFDataset

    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorEvaluation, ProbeMonitorRun
    from src.services.probe_monitor_capture import _row_mask

    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    raw = tmp_path / "datasets" / "g5_backfill"
    HFDataset.from_dict({
        "text": [_text(i) for i in range(N_ROWS)],
        "stakes": ["high" if i % 2 == 0 else "low" for i in range(N_ROWS)],
    }).save_to_disk(str(raw))

    def fake_render_all(tokenizer, conversations, **kwargs):
        out = []
        for conversation in conversations:
            text = " ".join(m["content"] for m in conversation)
            n = 6 + len(text) % 5
            out.append(RenderedExample(input_ids=list(range(n)), token_roles=["user"] * n, token_message=[0] * n,
                                       text=text, truncated="LONG" in text))
        return out, SimpleNamespace(as_dict=lambda: {})

    monkeypatch.setattr(probe_monitor_render, "render_all", fake_render_all)

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny", status=ModelStatus.READY, file_path="/x",
                  architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="g5", source="local", status=DatasetStatus.READY, raw_path=str(raw),
                      extra_metadata={})
    sync_db.add_all([model, dataset])
    sync_db.flush()
    train = ProbeMonitorDataset(name="train", dataset_id=dataset.id, input_column="text", label_column="stakes",
                                label_mapping={"high": "positive", "low": "negative"}, role="train", counts={})
    view = ProbeMonitorDataset(name="carriers", dataset_id=dataset.id, input_column="text", label_column="stakes",
                               label_mapping={"high": "positive", "low": "negative"}, role="eval",
                               distribution="out_of_distribution", counts={})
    sync_db.add_all([train, view])
    sync_db.flush()
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=train.id, eval_dataset_ids=[view.id],
                          config={"scope": "all", "max_length": 64}, status="completed", environment={})
    sync_db.add(run)
    sync_db.flush()
    probe = ProbeMonitor(run_id=run.id, layer=0, rule="mean", rule_params={}, variant="dense", val_metrics={},
                         rung=0, rung_reasons=[], threshold=0.5, target_fpr=0.01, weights_path="/nonexistent/w.pt")
    sync_db.add(probe)
    sync_db.flush()

    # What the pre-G5 evaluator stored: every row, positions = source rows, n_scored by the capture's own mask.
    rendered, _ = fake_render_all(None, [[{"role": "user", "content": _text(i)}] for i in range(N_ROWS)])
    positions = list(range(N_ROWS))
    n_scored = [sum(_row_mask(r, "all")) for r in rendered]
    labels = [1 if i % 2 == 0 else 0 for i in range(N_ROWS)]
    # Truncated negatives score highest, so the untruncated AUROC differs from the all-rows one.
    scores = [9.0 + i * 0.01 if i in TRUNCATED_ROWS else (1.0 if labels[i] else 0.0) + i * 0.001
              for i in range(N_ROWS)]
    directory = tmp_path / "probe_monitors" / run.id
    directory.mkdir(parents=True)
    evaluation_id = f"pme_{uuid.uuid4().hex[:12]}"
    stem = directory / f"{probe.id}__{view.id}__{evaluation_id}"
    np.save(f"{stem}.npy", np.asarray(scores, dtype=np.float32))
    np.save(f"{stem}__positions.npy", np.asarray(positions, dtype=np.int64))
    np.save(f"{stem}__n_scored.npy", np.asarray(n_scored, dtype=np.int32))
    np.save(f"{stem}__labels.npy", np.asarray(labels, dtype=np.int8))
    from src.services.probe_monitor_metrics import evaluate

    all_rows = evaluate([float(np.float32(s)) for s in scores], labels, name="carriers", out_of_distribution=True,
                        lengths=n_scored, target_fpr=0.01)
    evaluation = ProbeMonitorEvaluation(
        id=evaluation_id, probe_id=probe.id, dataset_id=view.id, status="completed",
        n_positive=all_rows["n_positive"], n_negative=all_rows["n_negative"], metrics=all_rows,
        scores_path=f"{stem}.npy", positions_path=f"{stem}__positions.npy", n_scored_path=f"{stem}__n_scored.npy",
        labels_path=f"{stem}__labels.npy",
    )
    sync_db.add(evaluation)
    sync_db.commit()
    return SimpleNamespace(probe=probe, evaluation=evaluation, stem=stem, scores=scores, labels=labels)


def _backfill(sync_db, world):
    from src.services.probe_monitor_run import backfill_truncation

    return backfill_truncation(sync_db, world.evaluation.id, tokenizer=object())


def test_backfill_writes_the_mask_when_every_row_agrees(sync_db, world):
    result = _backfill(sync_db, world)
    assert result["status"] == "backfilled" and (result["rows"], result["truncated"]) == (N_ROWS, 3)
    sync_db.refresh(world.evaluation)
    assert world.evaluation.truncated_path == f"{world.stem}__truncated.npy"
    mask = np.load(world.evaluation.truncated_path)
    assert mask.dtype == np.int8 and np.flatnonzero(mask).tolist() == list(TRUNCATED_ROWS)
    # The rung was recomputed from the backfilled evaluation.
    assert "rung" in result and result["rung"]["catch_rate"]["sets"][0]["truncated_excluded"] == 0
    # A second call is idempotent and writes nothing new.
    assert _backfill(sync_db, world)["status"] == "already_recorded"


def test_backfill_adds_the_untruncated_block_and_keeps_all_rows(sync_db, world):
    before = dict(world.evaluation.metrics)
    _backfill(sync_db, world)
    sync_db.refresh(world.evaluation)
    metrics = world.evaluation.metrics
    for key, value in before.items():
        assert metrics[key] == value, f"the all-rows field {key} moved"
    keep = [i for i in range(N_ROWS) if i not in TRUNCATED_ROWS]
    scores32 = [float(np.float32(s)) for s in world.scores]
    want = auroc([scores32[i] for i in keep], [world.labels[i] for i in keep])
    assert metrics["untruncated"]["auroc"] == pytest.approx(want, abs=1e-12)
    assert abs(metrics["untruncated"]["auroc"] - metrics["auroc"]) > 1e-3, "fixture too weak"
    assert metrics["truncation"]["policy"] == "excluded" and metrics["truncation"]["negatives"] == 3
    assert metrics["truncation"]["backfilled"]["rows_matched"] == N_ROWS


def test_backfill_refuses_on_one_n_scored_mismatch(sync_db, world):
    from src.services.probe_monitor_run import TruncationBackfillRefused

    n_scored = np.load(f"{world.stem}__n_scored.npy")
    n_scored[17] += 1
    np.save(f"{world.stem}__n_scored.npy", n_scored)
    with pytest.raises(TruncationBackfillRefused) as caught:
        _backfill(sync_db, world)
    assert caught.value.code == "truncation_backfill_mismatch"
    assert caught.value.extra["n_scored_disagree"] == 1 and caught.value.extra["not_rerendered"] == 0
    sync_db.refresh(world.evaluation)
    assert world.evaluation.truncated_path is None
    assert not Path(f"{world.stem}__truncated.npy").exists()
    assert "untruncated" not in world.evaluation.metrics
