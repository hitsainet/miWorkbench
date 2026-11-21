"""Real-database rows for the WSP (ENH-010 044) tests: a model, a dataset, a training view, a run and one probe.

Synchronous (the WSP services and Celery tasks use a sync session). Each call mints fresh ids, so tests on one
database never collide. Fixtures differ only in the field under test: callers pass that field.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


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


@dataclass
class World:
    model_id: str
    dataset_id: uuid.UUID
    view_id: str
    run_id: str
    probe_id: str


def seed_probe(db, *, view_role: str = "train", **probe_fields) -> World:
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="t", status=ModelStatus.READY, file_path="/x",
                  architecture="L", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="c", source="local", status=DatasetStatus.READY, raw_path="/x",
                      extra_metadata={})
    db.add_all([model, dataset])
    db.flush()
    view = ProbeMonitorDataset(name="t", dataset_id=dataset.id, input_column="t", label_column="l",
                               label_mapping={"1": "positive", "0": "negative"}, role=view_role, counts={})
    db.add(view)
    db.flush()
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=view.id, eval_dataset_ids=[], config={},
                          status="completed", environment={})
    db.add(run)
    db.flush()
    fields = {"layer": 1, "rule": "mean", **probe_fields}
    probe = ProbeMonitor(run_id=run.id, **fields)
    db.add(probe)
    db.commit()
    return World(model.id, dataset.id, view.id, run.id, probe.id)


def add_view(db, world: World, *, role: str = "eval", dataset_id=None) -> str:
    from src.models.probe_monitor import ProbeMonitorDataset

    view = ProbeMonitorDataset(name=f"v-{role}", dataset_id=dataset_id or world.dataset_id, input_column="t",
                               label_column="l", label_mapping={"1": "positive", "0": "negative"}, role=role,
                               counts={})
    db.add(view)
    db.commit()
    return view.id
