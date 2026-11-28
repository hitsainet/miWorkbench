"""A later job on a probe never rewrites the card its training run recorded; each job records its own (2026-11-27).

`_load_model_for_run(run)` set `gpu_uuid` / `gpu_uuids` from its placement on the row it was given, and every caller
gives it the TRAINING run's row: an evaluation, a WSP held-out capture, a window re-cut, a definition build, a score.
The evaluation's per-view commit then saved it, so a finished run's recorded card changed after a later evaluation.
The loader now returns the placement (`LoadedModel.placement`) and writes nothing; training records it on its run
row, and an evaluation and a capture record it on their own task-queue rows through `RowNarration.placed`.

These run the REAL `_load_model_for_run` and the real evaluation / capture on the tiny model of
`tests/support/wsp_pipeline.py`. Only the loader's edges are stubbed: the model-row session (pointed at this module's
database, as is every row write the narration makes), the weights directory, NVML placement (`place_job`, returning a
real `Placement` over one or two named cards: 1+n GPUs) and the weights load. What is asserted is read back from the
database through a fresh connection, so an uncommitted attribute cannot pass for a recorded one.

Controls (0xcc/reviews/eval_card_and_retry_2026-11-27.md):
  C1  the loader writes `placement.gpu_columns()` onto its row again        -> the evaluation and capture tests
  C2  `execute_probe_run` drops its own write of the placement              -> the training test
  C3  `evaluate_probe` reports the run row's columns instead of its load's  -> the evaluation tests
  C4  `run_captures` reports the run row's columns instead of its load's    -> the capture test
  C5  `placed_cards` drops a split's card list                              -> the split tests
"""
from __future__ import annotations

import types
import uuid

import pytest

pytest.importorskip("datasets")

import torch  # noqa: E402
from sqlalchemy import text  # noqa: E402

from tests.support.wsp_pipeline import (  # noqa: E402,F401 (fixtures)
    arrow_dataset,
    heldout_arrow,
    monkeypatch_module,
    pipeline_db,
    tiny_lm,
    tiny_tokenizer,
    wsp_data_dir,
    wsp_probe,
    wsp_world,
)

TRAIN_CARD = "GPU-77777777-0000-0000-0000-000000000007"
CARD_A = "GPU-aaaaaaaa-0000-0000-0000-00000000000a"
CARD_B = "GPU-bbbbbbbb-0000-0000-0000-00000000000b"


def placement(*uuids):
    """A real `Placement` over named cards (no CUDA needed: the devices are the CPU)."""
    from src.services.gpu_placement import GpuCard, Placement

    cards = tuple(GpuCard(index=i, uuid=u, name="test card", total_mb=24576, free_mb=20000)
                  for i, u in enumerate(uuids))
    cpu = torch.device("cpu")
    if len(cards) == 1:
        return Placement(card=cards[0], device=cpu)
    return Placement(card=cards[0], device=cpu, cards=cards, devices=(cpu,) * len(cards),
                     max_memory_mb={i: 20000 for i in range(len(cards))})


@pytest.fixture
def real_loader(monkeypatch, pipeline_db, tiny_lm, tiny_tokenizer):
    """The production `_load_model_for_run` with only its edges stubbed; `state["placement"]` is what NVML chose."""
    from sqlalchemy.orm import sessionmaker

    from src.services import probe_monitor_run as run_module
    from src.services.activation_service import ActivationService

    state = {"placement": placement(CARD_A), "asked": []}
    monkeypatch.setattr("src.core.database.SyncSessionLocal", sessionmaker(bind=pipeline_db.get_bind()))
    monkeypatch.setattr(run_module, "resolve_weights_dir", lambda *a, **k: "/nonexistent/weights")

    def place_job(request):
        state["asked"].append(request)
        return state["placement"]

    monkeypatch.setattr("src.services.gpu_placement.place_job", place_job)
    monkeypatch.setattr(ActivationService, "_load_model",
                        lambda self, path, quantization, where: (tiny_lm, tiny_tokenizer))
    return state


@pytest.fixture
def gates_pass(monkeypatch):
    """The fixture run predates precision recording; these refusals are not under test."""
    monkeypatch.setattr("src.services.probe_monitor_run.require_matching_precision", lambda *a, **k: None)
    monkeypatch.setattr("src.services.probe_monitor_run.require_served_render", lambda *a, **k: None)


@pytest.fixture
def trained_on(pipeline_db, wsp_probe):
    """The training run recorded TRAIN_CARD (a split's list: None, one card)."""
    from src.models.probe_monitor import ProbeMonitorRun

    run = pipeline_db.query(ProbeMonitorRun).filter(ProbeMonitorRun.id == wsp_probe.run_id).one()
    run.gpu_uuid, run.gpu_uuids = TRAIN_CARD, None
    pipeline_db.commit()
    return run.id


def recorded_card(pipeline_db, table, key, value):
    """`(gpu_uuid, gpu_uuids)` as COMMITTED, through a connection of its own."""
    with pipeline_db.get_bind().connect() as conn:
        row = conn.execute(text(f"SELECT gpu_uuid, gpu_uuids FROM {table} WHERE {key} = :v"), {"v": value}).one()
    return row[0], row[1]


def job_row(pipeline_db, task_type, probe_id):
    """The task-queue row the route opens before it sends (`_task_rows.open_task_row`), and a task bound to it."""
    from src.models.task_queue import TaskQueue

    celery_id = str(uuid.uuid4())
    pipeline_db.add(TaskQueue(id=f"tq_{uuid.uuid4().hex[:12]}", task_id=celery_id, task_type=task_type,
                              entity_id=probe_id, entity_type="probe_monitor", status="queued", progress=0.0,
                              retry_params={}, retry_count=0))
    pipeline_db.commit()
    task = types.SimpleNamespace(request=types.SimpleNamespace(id=celery_id), update_state=lambda **kw: None)
    return celery_id, task


@pytest.mark.parametrize("cards", [(CARD_A,), (CARD_A, CARD_B)], ids=["one_card", "two_card_split"])
def test_an_evaluation_leaves_the_training_card_and_records_its_own(pipeline_db, wsp_world, wsp_probe, trained_on,
                                                                     real_loader, gates_pass, cards):
    from src.services.probe_monitor_run import evaluate_probe
    from src.workers.jlens_progress import PROBE_EVALUATION, RowNarration

    real_loader["placement"] = placement(*cards)
    celery_id, task = job_row(pipeline_db, PROBE_EVALUATION, wsp_probe.id)
    narration = RowNarration(task)
    narration.start()

    result = evaluate_probe(pipeline_db, wsp_probe.id, [wsp_world["eval"].id], sample_rows=None,
                            progress=narration.meter(PROBE_EVALUATION))

    assert result["evaluations"] == 1
    assert real_loader["asked"], "the real loader did not run"
    assert recorded_card(pipeline_db, "probe_monitor_runs", "id", trained_on) == (TRAIN_CARD, None), (
        "the evaluation rewrote the card the training run recorded")
    expected = (CARD_A, None) if len(cards) == 1 else (CARD_A, [CARD_A, CARD_B])
    assert recorded_card(pipeline_db, "task_queue", "task_id", celery_id) == expected, (
        "the evaluation's own row does not name every card it used")


@pytest.mark.parametrize("cards", [(CARD_A,), (CARD_A, CARD_B)], ids=["one_card", "two_card_split"])
def test_a_capture_leaves_the_training_card_and_records_its_own(pipeline_db, wsp_probe, trained_on, real_loader,
                                                                gates_pass, cards):
    from src.api.v1.endpoints.wsp import plan_capture_rows
    from src.models.wsp import WspActivationCapture
    from src.services import wsp_capture
    from src.workers.jlens_progress import WSP_HELDOUT_CAPTURE, RowNarration

    # A fresh plan each time: the previous parametrisation's rows are `ready`, so clear them first.
    pipeline_db.query(WspActivationCapture).filter_by(probe_id=wsp_probe.id).delete()
    pipeline_db.commit()
    ids = plan_capture_rows(pipeline_db, wsp_probe.id, None)["to_run"]
    pipeline_db.commit()
    assert ids, "nothing to capture"
    real_loader["placement"] = placement(*cards)
    celery_id, task = job_row(pipeline_db, WSP_HELDOUT_CAPTURE, wsp_probe.id)
    narration = RowNarration(task)
    narration.start()

    out = wsp_capture.run_captures(pipeline_db, wsp_probe.id, ids, progress=narration.meter(WSP_HELDOUT_CAPTURE))

    assert sorted(set(out["captures"].values())) == ["ready"], out
    assert recorded_card(pipeline_db, "probe_monitor_runs", "id", trained_on) == (TRAIN_CARD, None), (
        "the capture rewrote the card the training run recorded")
    expected = (CARD_A, None) if len(cards) == 1 else (CARD_A, [CARD_A, CARD_B])
    assert recorded_card(pipeline_db, "task_queue", "task_id", celery_id) == expected


def test_the_loader_writes_nothing_to_the_row_it_is_given(real_loader):
    """Every caller passes the training run's row (re-cut, definition build, score, family score too)."""
    from src.services.probe_monitor_run import _load_model_for_run, placed_cards

    real_loader["placement"] = placement(CARD_A, CARD_B)
    row = types.SimpleNamespace(model_id="m_test", gpu_request="auto", gpu_uuid=TRAIN_CARD, gpu_uuids=None)
    # The model row is looked up; give the loader one through its own session.
    from unittest.mock import patch

    model_row = types.SimpleNamespace(file_path="x", quantized_path=None, quantization="FP16", architecture="llama")
    session = types.SimpleNamespace(
        query=lambda *_a: types.SimpleNamespace(filter=lambda *_b: types.SimpleNamespace(first=lambda: model_row)),
        close=lambda: None)
    with patch("src.core.database.SyncSessionLocal", lambda: session):
        loaded = _load_model_for_run(row)
    model, tokenizer, architecture = loaded
    assert architecture == "llama"
    assert (row.gpu_uuid, row.gpu_uuids) == (TRAIN_CARD, None)
    assert placed_cards(loaded) == (CARD_A, [CARD_A, CARD_B])


def test_training_records_its_own_placement_on_its_run_row(pipeline_db, wsp_world, tiny_lm, tiny_tokenizer,
                                                            real_loader, gates_pass):
    """The other half: training is the one job that writes the run row, from the placement its load returned."""
    from src.models.probe_monitor import ProbeMonitorRun
    from src.services import probe_monitor_run as run_module

    src = wsp_world["run"]
    clone = ProbeMonitorRun(**{c.name: getattr(src, c.name) for c in ProbeMonitorRun.__table__.columns
                               if c.name not in ("id", "status", "stage", "progress", "progress_detail",
                                                 "error_message", "celery_task_id", "gpu_uuid", "gpu_uuids",
                                                 "artifact_dir", "environment", "layer_selection", "created_at",
                                                 "updated_at", "completed_at")})
    clone.id = f"pmr_{uuid.uuid4().hex[:12]}"
    clone.status = "pending"
    pipeline_db.add(clone)
    pipeline_db.commit()
    real_loader["placement"] = placement(CARD_A, CARD_B)

    run_module.execute_probe_run(pipeline_db, clone.id)

    assert recorded_card(pipeline_db, "probe_monitor_runs", "id", clone.id) == (CARD_A, [CARD_A, CARD_B])


def test_a_definition_build_records_its_cards_on_its_own_task_row(monkeypatch):
    """The build task hands `build` an `on_placed` that writes the card(s) to ITS task-queue row (by its Celery id).

    Control C7: `_build_definition` stops passing `on_placed` -> red."""
    import contextlib

    import src.services.probe_definition_builder as builder
    import src.workers.jlens_progress as jlens_progress
    import src.workers.probe_monitor_tasks as tasks
    from src.workers.base_task import DatabaseTask

    recorded, seen = [], {}

    def fake_build(db, probe_id, **kwargs):
        seen.update(kwargs)
        kwargs["on_placed"](CARD_A, [CARD_A, CARD_B])
        raise builder.ProbeExportRefused("stop here: only the placement is under test", status=409)

    monkeypatch.setattr(builder, "build", fake_build)
    monkeypatch.setattr(jlens_progress, "record_gpu", lambda *a, **k: recorded.append((a, k)) or True)
    monkeypatch.setattr(jlens_progress, "mark_running", lambda *a, **k: True)
    monkeypatch.setattr(jlens_progress, "update_row", lambda *a, **k: True)
    monkeypatch.setattr(tasks, "_require_row", lambda *a, **k: types.SimpleNamespace(run_id="pmr_x"))
    monkeypatch.setattr(tasks, "emit_probe_definition_failed", lambda *a, **k: None)
    monkeypatch.setattr(DatabaseTask, "get_db", lambda _self: contextlib.nullcontext(object()))

    task = tasks.build_probe_definition
    task.push_request(id="celery-build-1")
    try:
        result = task.run(probe_id="pm_x")
    finally:
        task.pop_request()

    assert result["status"] == "refused"
    assert recorded == [(("celery-build-1", CARD_A, [CARD_A, CARD_B]), {"attempts": 1})]
