"""A probe-training run records the threads it had and the optimiser it used (ENH-011 WB-2, #10 G25 and #11 G6).

#10: 16-bit probe training ran about 5× slower and nothing recorded the thread count a run actually had
(`celery-gpu-workers` sets no `OMP_NUM_THREADS`; the supervisor sets none per worker). At the start of training the
run's environment (the FR-15 reproducibility record, committed at every stage boundary) now holds
`environment["threads"]`: torch's intra- and inter-op threads, `os.cpu_count()`, the CPUs in the process's affinity
mask (None off Linux), `OMP_NUM_THREADS` and `MKL_NUM_THREADS` (None when unset) and the worker's card.

#11: lr 0.05 is unstable, and the run stored no lr, patience, weight decay or schedule, so a later default change would
silently change what a re-run means. `environment["optimisation"]` holds them, defaults included, and they are the
values the run hands `train_rule` (one source). `TrainedRule.optimisation` holds what `train_rule` itself used, so a
caller passing an explicit lr records that lr. Not in the exported probe-definition document (a contract; out of
scope). RECORDS ONLY (E11-4): no default, thread setting or training behaviour changes.

Driven through the REAL `execute_probe_run` on CPU with the pipeline test's tiny model, read back from the database.
Controls (0xcc/reviews/ENH-011_WB_controls_2026-11-26.md): C5 the thread fields dropped; C6 record only explicit
values (the defaults case); C7 `lr` dropped.
"""
import os
import uuid

import pytest
import torch

pytest.importorskip("datasets")

from tests.unit.test_probe_monitor_pipeline import (  # noqa: E402,F401 - fixtures by import
    _loader_that_records_its_placement,
    arrow_dataset,
    monkeypatch_module,
    pipeline_db,
    tiny_lm,
    tiny_tokenizer,
)

CARD = "GPU-00000000-1111-2222-3333-444444444444"
THREADS = 2


@pytest.fixture(scope="module")
def recorded_run(pipeline_db, arrow_dataset, tiny_lm, tiny_tokenizer, tmp_path_factory, monkeypatch_module):  # noqa: F811
    """One real run with known thread settings; returns (row read back, the train_rule calls, the settings)."""
    from src.core import config as config_module
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun
    from src.services import probe_monitor_run as run_module
    from src.workers.gpu_supervisor import WORKER_GPU_ENV

    model_row = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny-random", status=ModelStatus.READY,
                      file_path=str(arrow_dataset), architecture="LlamaForCausalLM", params_count=1_000)
    dataset_row = Dataset(id=uuid.uuid4(), name="probe-corpus", source="local", status=DatasetStatus.READY,
                          raw_path=str(arrow_dataset), extra_metadata={})
    pipeline_db.add_all([model_row, dataset_row])
    pipeline_db.commit()
    mapping = {"high": "positive", "low": "negative"}
    train = ProbeMonitorDataset(name="train", dataset_id=dataset_row.id, input_column="inputs", label_column="stakes",
                                label_mapping=mapping, pair_column="pair", role="train", counts={})
    held = ProbeMonitorDataset(name="eval", dataset_id=dataset_row.id, input_column="inputs", label_column="stakes",
                               label_mapping=mapping, role="eval", distribution="out_of_distribution", counts={})
    pipeline_db.add_all([train, held])
    pipeline_db.commit()
    # The DEFAULTS: no optimiser setting is (or can be) requested.
    run = ProbeMonitorRun(model_id=model_row.id, train_dataset_id=train.id, eval_dataset_ids=[held.id],
                          config={"layers": [1], "rules": ["mean"], "scope": "all", "max_length": 64,
                                  "val_fraction": 0.25, "seed": 1337, "top_n_layers": 1, "target_fpr": 0.1},
                          status="running", environment={})
    pipeline_db.add(run)
    pipeline_db.commit()

    monkeypatch_module.setattr(config_module.settings, "data_dir", tmp_path_factory.mktemp("threads_artifacts"),
                               raising=False)
    monkeypatch_module.setenv("OMP_NUM_THREADS", "3")
    monkeypatch_module.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch_module.setenv(WORKER_GPU_ENV, CARD)
    # Where the run looks it up: `execute_probe_run` imports `train_rule` from the trainer inside its body.
    from src.services import probe_monitor_trainer as trainer_module

    calls = []
    real = trainer_module.train_rule

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch_module.setattr(trainer_module, "train_rule", spy)
    before = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    try:
        run_module.execute_probe_run(pipeline_db, run.id,
                                     model_loader=_loader_that_records_its_placement(tiny_lm, tiny_tokenizer))
        expected = {"interop": torch.get_num_interop_threads(), "cpu_count": os.cpu_count(),
                    "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    finally:
        torch.set_num_threads(before)
    pipeline_db.expire_all()
    row = pipeline_db.get(ProbeMonitorRun, run.id)
    return row, calls, expected


def test_the_run_records_the_threads_it_trained_with(recorded_run):
    """2.1: read back from the stored row, not from the process."""
    row, _calls, expected = recorded_run
    assert row.status == "completed", (row.status, row.error_message)
    assert row.environment["threads"] == {
        "torch_num_threads": THREADS,
        "torch_num_interop_threads": expected["interop"],
        "cpu_count": expected["cpu_count"],
        "affinity_cpus": expected["affinity"],
        "OMP_NUM_THREADS": "3",
        "MKL_NUM_THREADS": None,
        "worker_card": CARD,
    }


def test_the_affinity_falls_back_off_linux(monkeypatch):
    from src.services.probe_monitor_run import threads_record

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    record = threads_record()
    assert record["affinity_cpus"] is None and record["cpu_count"] == os.cpu_count()


def test_a_run_with_the_defaults_records_every_optimiser_setting(recorded_run):
    """3.1: the defaults case. Every key, the measured defaults, and none left for a reader to assume."""
    from src.services import probe_monitor_trainer as trainer

    row, _calls, _ = recorded_run
    assert row.environment["optimisation"] == {
        "optimiser": "AdamW",
        "lr": trainer.DEFAULT_LR,
        "weight_decay": trainer.DEFAULT_WEIGHT_DECAY,
        "patience": trainer.DEFAULT_PATIENCE,
        "max_epochs": trainer.DEFAULT_EPOCHS,
        "schedule": "none",
        "selection": "max validation AUROC",
    }
    assert row.environment["optimisation"]["lr"] == 0.05, "the record holds a number, not a pointer to a default"


def test_the_run_trains_with_exactly_the_recorded_settings(recorded_run):
    """One source: every `train_rule` call of the run got the recorded lr, weight decay, patience and epoch cap."""
    row, calls, _ = recorded_run
    recorded = row.environment["optimisation"]
    assert calls, "the run trained nothing: the test proves nothing"
    for kwargs in calls:
        assert (kwargs["lr"], kwargs["weight_decay"], kwargs["patience"], kwargs["epochs"]) == (
            recorded["lr"], recorded["weight_decay"], recorded["patience"], recorded["max_epochs"]), kwargs


def test_the_config_is_still_the_request_echoed(recorded_run):
    """The record lives in the environment: `config` stays what the run was asked to do (FR-15)."""
    row, _calls, _ = recorded_run
    assert set(row.config) == {"layers", "rules", "scope", "max_length", "val_fraction", "seed", "top_n_layers",
                               "target_fpr"}


def test_an_explicit_lr_is_recorded_as_given():
    """3.1: `train_rule` records what it used, so an explicit lr is that lr (and AdamW really got it)."""
    from unittest.mock import patch

    import numpy as np

    from src.services import probe_monitor_trainer as trainer

    rng = np.random.default_rng(0)
    rows = [rng.normal(size=(4, 8)).astype(np.float32) + (i % 2) for i in range(16)]
    labels = [i % 2 for i in range(16)]
    seen = []
    real = torch.optim.AdamW

    def adamw(params, **kwargs):
        seen.append(kwargs)
        return real(params, **kwargs)

    with patch.object(torch.optim, "AdamW", adamw):
        trained = trainer.train_rule("mean", rows[:12], labels[:12], rows[12:], labels[12:], lr=0.01, epochs=3,
                                     patience=2, device=torch.device("cpu"))
    assert seen and seen[0]["lr"] == 0.01
    assert trained.optimisation == {"optimiser": "AdamW", "lr": 0.01, "weight_decay": trainer.DEFAULT_WEIGHT_DECAY,
                                    "patience": 2, "max_epochs": 3, "schedule": "none",
                                    "selection": "max validation AUROC"}
