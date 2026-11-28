"""045 FR-18 (CHAIN_PLAN A17, A28, A42; FTDD §18; FTASKS 21.7 part, 21.9 part): agents read the Hub anonymously.

Built so far:
* `core.hub_auth`: `hub_auth_for` (anonymous for an agent other than the MCP pod) and `hub_token` (`False` under
  anonymous, never `None`, which huggingface_hub reads as "use HF_TOKEN"; fails closed on an unknown policy).
* the logit-lens GET (A42): for an agent, a model whose files are not on disk is `409 model_files_not_on_disk` and the
  full model load (which would reach the Hub) is never called; the operator and the MCP pod keep today's fallback.

Controls (0xcc/reviews/045_A0_implementation_controls_2026-11-16.md):
  `hub_token` returns None under anonymous                 → test_hub_token_is_false_for_an_agent red
  drop the refusal in `calculate_logit_lens`               → test_the_logit_lens_refuses_an_agent_without_files red
  pass no policy from the route (the service default)       → test_the_operator_keeps_the_fallback red
"""
from __future__ import annotations

import uuid

from unittest.mock import MagicMock

import httpx
import pytest
import torch

from src.core.hub_auth import hub_auth_for, hub_token


class _Req:
    def __init__(self, headers):
        self.headers = headers


@pytest.mark.parametrize("headers,policy", [
    ({}, "stored"),
    ({"X-MiStudio-Agent": "mcp"}, "stored"),
    ({"X-MiStudio-Agent": "agent:miassist"}, "anonymous"),
    ({"X-MiStudio-Agent": "miassist"}, "anonymous"),
    ({"X-MiStudio-Agent": ""}, "anonymous"),
])
def test_hub_auth_for(headers, policy):
    assert hub_auth_for(_Req(headers)) == policy


def test_hub_token_is_false_for_an_agent(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_deployment_token")
    assert hub_token("anonymous", "hf_explicit") is False
    assert hub_token("something-else") is False, "an unknown policy fails closed"
    assert hub_token("stored", "hf_explicit") == "hf_explicit"


@pytest.fixture
async def feature_without_model_files(async_session):
    """A feature of an external SAE whose model row has no files on disk (rows in the test database)."""
    from src.models.external_sae import ExternalSAE, SAEStatus
    from src.models.extraction_job import ExtractionJob, ExtractionStatus
    from src.models.feature import Feature
    from src.models.model import Model

    tag = uuid.uuid4().hex[:8]
    model_id, sae_id, job_id, feature_id = f"m_ll{tag}", f"sae_ll{tag}", f"ext_ll{tag}", f"feat_ll{tag}"
    async_session.add(Model(id=model_id, name=f"ll {tag}", architecture="llama", params_count=1,
                            repo_id="fixture-org/not-on-disk", file_path=None))
    async_session.add(ExternalSAE(id=sae_id, name="ll", source="local", status=SAEStatus.READY.value,
                                  local_path="fixture/sae", model_id=model_id))
    async_session.add(ExtractionJob(id=job_id, external_sae_id=sae_id, config={}, status=ExtractionStatus.COMPLETED))
    async_session.add(Feature(id=feature_id, external_sae_id=sae_id, extraction_job_id=job_id, neuron_index=0,
                              name="f", activation_frequency=0.1, interpretability_score=0.5, max_activation=1.0))
    await async_session.commit()
    return feature_id


@pytest.fixture
def loads(monkeypatch):
    """The SAE read is stubbed (a decoder of 4 latents × 8 dims); the full model load records and stops."""
    from src.services import analysis_service

    seen: list[dict] = []
    monkeypatch.setattr(analysis_service, "load_sae_auto_detect",
                        lambda *a, **k: ({"W_dec": torch.zeros(4, 8)}, {}, "fixture"))

    def _model(**kwargs):
        seen.append(kwargs)
        raise RuntimeError("stopped by the test at the full model load")

    monkeypatch.setattr(analysis_service, "load_model_from_hf", _model)
    return seen


async def _get(feature_id, headers):
    from src.main import app

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.get(f"/api/v1/features/{feature_id}/logit-lens", headers=headers)


async def test_the_logit_lens_refuses_an_agent_without_files(feature_without_model_files, loads):
    response = await _get(feature_without_model_files, {"X-MiStudio-Agent": "agent:miassist"})
    assert response.status_code == 409, response.text[:300]
    assert response.json()["detail"]["code"] == "model_files_not_on_disk"
    assert loads == [], "the full model load (a Hub read) is never reached for an agent"


@pytest.mark.parametrize("headers", [{}, {"X-MiStudio-Agent": "mcp"}], ids=["operator", "mcp"])
async def test_the_operator_keeps_the_fallback(feature_without_model_files, loads, headers):
    response = await _get(feature_without_model_files, headers)
    assert response.status_code != 409, response.text[:300]
    assert len(loads) == 1 and loads[0]["local_files_only"] in (False, None)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 21.7: the wire cases. HF_TOKEN is SET, sockets beyond localhost are blocked, and the fake Hub sits beneath
# huggingface_hub's own client (so beneath transformers and datasets too). An agent's request never carries
# `Authorization`; a tokenizer load never fetches a weight file (A36's allow_patterns).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
from tests.support.hub_spy import blocked_sockets, fake_hub  # noqa: E402,F401 - fixtures

AGENT = {"X-MiStudio-Agent": "agent:miassist"}
WEIGHTS = {"model.safetensors", "pytorch_model.bin"}


def _no_token(hub):
    assert hub.requests, "the test never reached the fake Hub: it proves nothing"
    assert hub.authorized == [], f"an agent's Hub request carried Authorization: {hub.authorized[:3]}"


async def _call(method, url, headers=None, **kwargs):
    from src.main import app

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.request(method, url, headers=headers or {}, **kwargs)


def test_hub_load_tokenizer_is_anonymous_and_fetches_no_weights(fake_hub):
    from src.core.hub_auth import hub_load

    tok = hub_load("tokenizer", "fixture-org/tok", "anonymous")
    assert tok("hello world")["input_ids"] == [3, 4]
    _no_token(fake_hub)
    assert not (fake_hub.fetched() & WEIGHTS), f"a tokenizer load fetched weights: {fake_hub.fetched()}"


def test_the_stored_policy_keeps_todays_token(fake_hub):
    """The control for every wire case: the same load as the operator DOES send the deployment's token, so the spy
    would see one if an agent path leaked it."""
    from src.core.hub_auth import hub_load

    hub_load("tokenizer", "fixture-org/tok", "stored")
    assert fake_hub.authorized, "the operator's load sent no token: the spy cannot see a leak"


@pytest.fixture
def clear_tokenizer_cache():
    from src.api.v1.endpoints.datasets import load_tokenizer_cached

    load_tokenizer_cached.cache_clear()
    yield
    load_tokenizer_cached.cache_clear()


async def test_agent_tokenize_preview_is_anonymous(fake_hub, clear_tokenizer_cache):
    r = await _call("POST", "/api/v1/datasets/tokenize-preview", AGENT,
                    json={"tokenizer_name": "fixture-org/tok", "text": "hello world"})
    assert r.status_code == 200, r.text[:300]
    _no_token(fake_hub)
    assert not (fake_hub.fetched() & WEIGHTS)


async def test_the_tokenizer_cache_is_keyed_by_policy(fake_hub, clear_tokenizer_cache):
    body = {"tokenizer_name": "fixture-org/tok", "text": "hello world"}
    assert (await _call("POST", "/api/v1/datasets/tokenize-preview", json=body)).status_code == 200
    assert fake_hub.authorized, "the operator's load is the stored one"
    before = len(fake_hub.requests)
    assert (await _call("POST", "/api/v1/datasets/tokenize-preview", AGENT, json=body)).status_code == 200
    agent_requests = fake_hub.requests[before:]
    assert agent_requests, "the agent was served the operator's cached tokenizer"
    assert [r for r in agent_requests if r.authorization] == []


def test_agent_tokenize_task_is_anonymous(fake_hub):
    """The tokenize task's two loads (the task's own, and the mapper's in each map worker) under `anonymous`."""
    from src.services.tokenization_service import TokenizationService, _TokenizationMapper

    tok = TokenizationService.load_tokenizer("fixture-org/tok", cache_dir=None, hub_auth="anonymous")
    mapper = _TokenizationMapper("fixture-org/tok", "text", 8, True, "max_length", True, True, hub_auth="anonymous")
    assert mapper._get_tokenizer()("the cat")["input_ids"] == tok("the cat")["input_ids"]
    _no_token(fake_hub)
    assert not (fake_hub.fetched() & WEIGHTS)


def test_the_tokenize_task_defaults_to_anonymous(fake_hub):
    """A message naming no policy (a retry, a message from before the deploy) fails closed (A28, R1A-5)."""
    from src.services.tokenization_service import TokenizationService

    TokenizationService.load_tokenizer("fixture-org/tok")
    _no_token(fake_hub)


def test_agent_model_download_is_anonymous(fake_hub, tmp_path):
    """The model download's source under `anonymous`: config, tokenizer and ONE weight format, no token; the
    config and tokenizer then load from the local directory."""
    from transformers import AutoConfig, AutoTokenizer

    from src.core.hub_auth import hub_model_source

    src = hub_model_source("fixture-org/tok", "anonymous", cache_dir=str(tmp_path))
    assert src.token is False and src.local_files_only is True
    AutoConfig.from_pretrained(src.source, local_files_only=True)
    AutoTokenizer.from_pretrained(src.source, local_files_only=True)
    _no_token(fake_hub)
    fetched = fake_hub.fetched()
    assert "config.json" in fetched and "model.safetensors" in fetched
    assert "pytorch_model.bin" not in fetched, "a duplicate weight format was downloaded"


def test_the_model_download_task_uses_the_anonymous_source(monkeypatch, tmp_path):
    """Wiring: the task asks `hub_model_source` under its own `hub_auth` and loads what it returns, with no token."""
    from src.core import hub_auth
    from tests.unit.test_gpu_model_download_placement import Placement, CARDS, METADATA, _drive

    seen = {}

    def source(repo_id, policy, **kwargs):
        seen["policy"] = policy
        return hub_auth.ModelSource("/local/snapshot", False, True)

    def load(**kwargs):
        seen["load"] = kwargs
        return MagicMock(), MagicMock(), MagicMock(), METADATA

    monkeypatch.setattr(hub_auth, "hub_model_source", source)
    place = MagicMock(return_value=Placement(card=CARDS[1], device=torch.device("cuda", 1)))
    outcome = _drive(place, load, tmp_path, hub_auth=None)
    assert outcome.error is None, outcome.error
    assert seen["policy"] == "anonymous", "a message naming no policy must fail closed"
    assert seen["load"]["repo_id"] == "/local/snapshot"
    assert seen["load"]["hf_token"] is False and seen["load"]["local_files_only"] is True


async def _dataset_row(async_session, **fields):
    from src.models.dataset import Dataset, DatasetStatus

    ds = Dataset(name=f"hub fixture {uuid.uuid4().hex[:6]}", source="HuggingFace", status=DatasetStatus.READY,
                 **fields)
    async_session.add(ds)
    await async_session.commit()
    return ds


async def test_agent_samples_dataset_fallback_is_anonymous(fake_hub, async_session):
    ds = await _dataset_row(async_session, hf_repo_id="fixture-org/ds", raw_path=None,
                            extra_metadata={"split": "train"})
    r = await _call("GET", f"/api/v1/datasets/{ds.id}/samples", AGENT)
    assert r.status_code == 200, r.text[:300]
    assert r.json()["pagination"]["total"] == 2 if "pagination" in r.json() else True
    _no_token(fake_hub)


async def test_agent_samples_tokenizer_is_anonymous(fake_hub, async_session, tmp_path):
    from datasets import Dataset as HFDataset

    from src.core.config import settings
    from src.models.dataset_tokenization import DatasetTokenization, TokenizationStatus
    from src.models.model import Model, ModelStatus

    tag = uuid.uuid4().hex[:8]
    path = settings.data_dir / "hub_fixture" / tag
    HFDataset.from_dict({"input_ids": [[3, 4], [5, 6, 7]]}).save_to_disk(str(path))
    async_session.add(Model(id=f"m_hub{tag}", name="hub", architecture="gpt2", params_count=1,
                            repo_id="fixture-org/tok", status=ModelStatus.READY))
    ds = await _dataset_row(async_session, hf_repo_id=None, raw_path=None,
                            extra_metadata={"tokenization": {"tokenizer_name": "fixture-org/tok"}})
    async_session.add(DatasetTokenization(id=f"tok_hub{tag}", dataset_id=ds.id, model_id=f"m_hub{tag}",
                                          tokenized_path=str(path), tokenizer_repo_id="fixture-org/tok",
                                          status=TokenizationStatus.READY))
    await async_session.commit()
    try:
        r = await _call("GET", f"/api/v1/datasets/{ds.id}/samples", AGENT)
        assert r.status_code == 200, r.text[:300]
        assert "hello world" in r.text, "the tokenizer did not decode: the test proves nothing"
        _no_token(fake_hub)
        assert not (fake_hub.fetched() & WEIGHTS)
    finally:
        import shutil

        shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize("route,body", [
    ("/api/v1/saes/hf/preview", {"repo_id": "fixture-org/tok", "access_token": "hf_agent_own"}),
    ("/api/v1/jlens/acquire/preview", {"repo_id": "fixture-org/tok", "access_token": "hf_agent_own"}),
    ("/api/v1/datasets/download", {"repo_id": "fixture-org/ds", "access_token": "hf_agent_own"}),
    ("/api/v1/saes/download", {"repo_id": "fixture-org/tok", "filepath": "x", "access_token": "hf_agent_own"}),
])
async def test_agent_token_is_refused(fake_hub, route, body, monkeypatch):
    """An agent's own Hub token on an ungated Hub route: 422 before anything runs (the gated ones: V-9 in the gate)."""
    import src.workers.dataset_tasks as dt
    import src.workers.sae_tasks as st

    for task in (dt.download_dataset_task, st.download_sae_task):
        monkeypatch.setattr(task, "delay", MagicMock(side_effect=AssertionError("queued")))
    r = await _call("POST", route, AGENT, json=body)
    assert r.status_code == 422, r.text[:300]
    assert r.json()["detail"]["code"] == "agent_token_not_accepted"
    assert fake_hub.requests == [], "the Hub was reached before the refusal"


async def test_agent_preview_is_anonymous(fake_hub):
    r = await _call("POST", "/api/v1/saes/hf/preview", AGENT, json={"repo_id": "fixture-org/tok"})
    assert r.status_code == 200, r.text[:300]
    _no_token(fake_hub)


async def test_non_agent_preview_resolves_the_stored_token(fake_hub):
    for headers in ({}, {"X-MiStudio-Agent": "mcp"}):
        fake_hub.requests.clear()
        r = await _call("POST", "/api/v1/saes/hf/preview", headers, json={"repo_id": "fixture-org/tok"})
        assert r.status_code == 200, r.text[:300]
        assert fake_hub.authorized, f"{headers or 'the operator'} lost today's token"


async def test_agent_download_carries_no_token(fake_hub, async_session, monkeypatch):
    """`POST /datasets/download` for an agent: the revision check is anonymous, `resolve_hf_token` is never
    called, and the queued message holds `hub_auth="anonymous"` and no token (by value and base64)."""
    import base64
    import json as _json

    import src.services.huggingface_sae_service as hs
    import src.workers.dataset_tasks as dt

    monkeypatch.setattr(hs, "resolve_hf_token", MagicMock(side_effect=AssertionError("resolved for an agent")))
    queued = MagicMock(return_value=MagicMock(id="t1"))
    monkeypatch.setattr(dt.download_dataset_task, "delay", queued)
    r = await _call("POST", "/api/v1/datasets/download", AGENT, json={"repo_id": "fixture-org/ds"})
    assert r.status_code == 202, r.text[:300]
    _no_token(fake_hub)
    assert queued.call_count == 1 and queued.call_args.kwargs["hub_auth"] == "anonymous"
    message = _json.dumps(queued.call_args.kwargs, default=str)
    for secret in ("hf_DEPLOYMENT_TOKEN_must_not_leave",):
        assert secret not in message and base64.b64encode(secret.encode()).decode() not in message


async def _operator_style_download(fake_hub, monkeypatch, headers):
    """`POST /datasets/download` for a caller that keeps today's resolution: the revision check reaches the fake Hub
    WITH the stored token, and the queued message says `hub_auth="stored"` (the worker resolves the token itself)."""
    import src.workers.dataset_tasks as dt

    queued = MagicMock(return_value=MagicMock(id="t1"))
    monkeypatch.setattr(dt.download_dataset_task, "delay", queued)
    fake_hub.requests.clear()
    r = await _call("POST", "/api/v1/datasets/download", headers, json={"repo_id": "fixture-org/ds"})
    assert r.status_code == 202, r.text[:300]
    assert fake_hub.requests, "the revision check never reached the fake Hub: the test proves nothing"
    assert fake_hub.authorized, f"{headers or 'the operator'} lost today's token on the dataset download"
    assert queued.call_count == 1 and queued.call_args.kwargs["hub_auth"] == "stored"
    return queued


async def test_non_agent_download_resolves_the_stored_token(fake_hub, async_session, monkeypatch):
    """FTASKS 21.7 (R1 review fix OR1-045a0-4, 2026-11-16): the OPERATOR's own dataset download keeps the stored
    token, on the Hub and in the message's policy. Control (H5): force `hub_auth = "anonymous"` in
    `datasets.download_dataset` → red (it survived every test before this one)."""
    await _operator_style_download(fake_hub, monkeypatch, {})


async def test_mcp_keeps_todays_behaviour(fake_hub, async_session, monkeypatch):
    """FTASKS 21.7 (A15; R1 review fix OR1-045a0-4, 2026-11-16): the MCP pod (`mcp`) keeps today's Hub resolution on
    the dataset download and the model download: the token reaches the Hub and both queue `hub_auth="stored"`.
    Controls: `hub_auth_for` anonymous for `mcp` → red; force `hub_auth="anonymous"` at the model enqueue → red."""
    import uuid

    from src.api.v1.endpoints import models as models_ep

    mcp = {"X-MiStudio-Agent": "mcp"}
    await _operator_style_download(fake_hub, monkeypatch, mcp)

    task = MagicMock()
    task.delay.return_value = MagicMock(id="celery-1")
    monkeypatch.setattr(models_ep, "download_and_load_model", task)
    for headers in (mcp, {}):
        task.delay.reset_mock()
        repo = f"fixture-org/model-{uuid.uuid4().hex[:8]}"
        r = await _call("POST", "/api/v1/models/download", headers, json={"repo_id": repo})
        assert r.status_code == 202, (headers, r.text[:300])
        assert task.delay.call_count == 1, headers
        assert task.delay.call_args.kwargs["hub_auth"] == "stored", (headers, task.delay.call_args.kwargs)


def test_the_dataset_worker_is_anonymous_by_default(fake_hub, monkeypatch):
    """`download_dataset_task` without `hub_auth` resolves nothing (fails closed)."""
    import inspect

    import src.workers.dataset_tasks as dt

    assert inspect.signature(dt.download_dataset_task.run).parameters["hub_auth"].default == "anonymous"
    from src.core.hub_auth import hub_token

    assert hub_token("anonymous", "hf_x") is False


def _run_dataset_worker(tmp_path, monkeypatch, hub_auth):
    """The REAL `download_dataset_task` body against a real row in the test database, with the REAL
    `datasets.load_dataset` reading the fake Hub (as `test_dataset_task_revision.py` drives it, without its fake load)."""
    import contextlib
    import uuid as _uuid

    from src.core import cancellation
    from src.core.config import settings
    from src.core.database import SyncSessionLocal
    from src.models.dataset import Dataset, DatasetStatus
    from src.workers import dataset_tasks
    from tests.support.hub_spy import SHA

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data", raising=False)
    monkeypatch.setattr(dataset_tasks, "emit_dataset_progress", lambda *a, **k: None)
    monkeypatch.setattr(cancellation, "CancelWatchdog", lambda *a, **k: contextlib.nullcontext())
    with SyncSessionLocal() as db:
        row = Dataset(id=_uuid.uuid4(), name="ds", source="HuggingFace", hf_repo_id="fixture-org/ds",
                      status=DatasetStatus.DOWNLOADING, revision=SHA, extra_metadata={"split": "train"})
        db.add(row)
        db.commit()
        dataset_id = str(row.id)
    dataset_tasks.download_dataset_task(dataset_id=dataset_id, repo_id="fixture-org/ds", split="train",
                                        revision=SHA, hub_auth=hub_auth)
    with SyncSessionLocal() as db:
        done = db.get(Dataset, _uuid.UUID(dataset_id))
        assert done.status == DatasetStatus.READY, (done.status, done.error_message)


def test_the_dataset_worker_is_anonymous_on_the_wire(fake_hub, async_engine, tmp_path, monkeypatch):
    """R1 review fix (OR1-045a0-7), 2026-11-16: the worker's own body under `hub_auth="anonymous"`, with `HF_TOKEN`
    and a stored token set: the load COMPLETES (the data file itself is fetched) and no request carries
    `Authorization`. Control: `hub_token` returns None under anonymous (the library then reads `HF_TOKEN`) → red."""
    _run_dataset_worker(tmp_path, monkeypatch, "anonymous")
    assert "data/train.jsonl" in fake_hub.fetched(), "the data file was never fetched: the load proves nothing"
    _no_token(fake_hub)


def test_the_operators_dataset_worker_sends_the_token(fake_hub, async_engine, tmp_path, monkeypatch):
    """The same body under `"stored"` (the operator): the token IS sent, so the spy above can see one."""
    _run_dataset_worker(tmp_path, monkeypatch, "stored")
    assert "data/train.jsonl" in fake_hub.fetched()
    assert fake_hub.authorized, "the operator's worker lost its token"


# ── the enqueue inventory (AST) ──────────────────────────────────────────────────────────────────────────────────
import ast  # noqa: E402
from pathlib import Path  # noqa: E402

HUB_TASKS = {"download_dataset_task", "download_and_load_model", "download_sae_task", "acquire_jlens_artifact",
             "tokenize_dataset_task"}
SRC = Path(__file__).resolve().parents[2] / "src"
# Every enqueue of a Hub task on this tree (datasets download + tokenize, models download + redownload, the
# task-queue retry's model, dataset and tokenization branches, SAE download, J-lens acquire). A site added or gone
# moves the pin. (The guard found the tokenization retry, which the FTDD's list missed.)
EXPECTED_HUB_ENQUEUES = 9


def _name(node):
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None


def hub_enqueues(root: Path = SRC):
    """(file, line, passes_hub_auth) for every `.delay` / `.apply_async` / `dispatch_gpu_task` / `gpu_delay(...)(…)` /
    `_queue_without_leaking_token` enqueue of one of the five Hub tasks."""
    out = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if not isinstance(node, ast.Call):
                continue
            func, task, keywords = node.func, None, list(node.keywords)
            if isinstance(func, ast.Attribute) and func.attr in ("delay", "apply_async", "s", "si"):
                task = _name(func.value)
            elif _name(func) in ("dispatch_gpu_task", "_queue_without_leaking_token") and node.args:
                task = _name(node.args[0])
            elif isinstance(func, ast.Call) and _name(func.func) == "gpu_delay" and func.args:
                task = _name(func.args[0])
            if task not in HUB_TASKS:
                continue
            names = {k.arg for k in keywords}
            for k in keywords:  # apply_async(kwargs={...}) / dispatch_gpu_task(kwargs={...})
                if k.arg == "kwargs" and isinstance(k.value, ast.Dict):
                    names |= {getattr(key, "value", None) for key in k.value.keys}
            out.append((str(path.relative_to(root.parent)), node.lineno, "hub_auth" in names))
    return out


def test_every_enqueue_of_a_hub_task_passes_hub_auth():
    sites = hub_enqueues()
    assert len(sites) == EXPECTED_HUB_ENQUEUES, sites
    missing = [s for s in sites if not s[2]]
    assert missing == [], f"a Hub task enqueued without its caller's policy (fails closed, but the operator loses " \
                          f"the stored token): {missing}"


def test_the_enqueue_inventory_sees_an_omission(tmp_path):
    """The guard's own fixture: an enqueue without `hub_auth` is flagged."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("download_sae_task.delay(sae_id='x')\ngpu_delay(tokenize_dataset_task)(d=1)\n")
    assert [s[2] for s in hub_enqueues(tmp_path / "src")] == [False, False]


@pytest.mark.parametrize("task_path", [
    "src.workers.dataset_tasks:download_dataset_task", "src.workers.dataset_tasks:tokenize_dataset_task",
    "src.workers.model_tasks:download_and_load_model", "src.workers.sae_tasks:download_sae_task",
    "src.workers.jlens_acquire_tasks:acquire_jlens_artifact",
])
def test_the_five_hub_tasks_default_to_anonymous(task_path):
    """A28 / R1A-5: the default fails closed; a message naming no policy gets no token."""
    import importlib
    import inspect

    module, name = task_path.split(":")
    task = getattr(importlib.import_module(module), name)
    assert inspect.signature(task.run).parameters["hub_auth"].default == "anonymous"


# ── 21.9: the Hub-call inventory (AST), with the stricter token rule (R3A-8) ─────────────────────────────────────
HUB_CALLEES = {"HfApi", "snapshot_download", "hf_hub_download", "list_repo_files", "list_repo_tree", "model_info",
               "dataset_info", "repo_info", "get_safetensors_metadata", "load_dataset", "load_dataset_builder",
               "from_pretrained"}
# (file, enclosing function, callee) → why the call needs no explicit agent-safe token. Uploads/publishes (gated for
# agents, FR-13 §16.2b, and an agent's own token refused, V-9), and loads in operator-approved GPU jobs of a model the
# operator downloaded. The stricter rule (A36) still applies to every call NOT listed here.
_GPU_JOB = ("an operator-approved GPU job (gated for agents, FR-13) loading a model the operator downloaded; "
            "OPERATOR item: pass local_files_only=True once the path is proven always local")
_PUBLISH = "an upload or publish with the operator's write token: gated for agents (FTDD §16.2b), agent token refused"
HUB_CALLS_WITHOUT_TOKEN = {
    ("src/api/v1/endpoints/probe_monitors.py", "_huggingface_rejects", "HfApi"):
        "publish preflight `whoami` (" + _PUBLISH + ")",
    ("src/services/huggingface_sae_service.py", "_upload", "HfApi"): _PUBLISH,
    ("src/services/huggingface_sae_service.py", "_check", "HfApi"): "check_token_permissions: the caller's OWN token",
    ("src/services/jlens_acquire_service.py", "publish_artifact", "HfApi"): _PUBLISH,
    ("src/services/probe_definition_publisher.py", "*", "HfApi"): _PUBLISH,
    ("src/services/probe_definition_publisher.py", "*", "model_info"): _PUBLISH,
    ("src/ml/model_loader.py", "load_model_from_hf", "from_pretrained"):
        "token=hf_token: checked at every CALLER by test_every_load_model_from_hf_call_passes_a_policy_token",
    ("src/workers/model_tasks.py", "required_mb_for_load", "from_pretrained"):
        "token= arrives in **config_kwargs from download_and_load_model's hub_model_source (wire-tested)",
    ("src/services/activation_service.py", "*", "from_pretrained"): _GPU_JOB,
    ("src/services/local_labeling_service.py", "*", "from_pretrained"): _GPU_JOB,
    ("src/services/logit_lens_service.py", "*", "from_pretrained"): _GPU_JOB,
    ("src/services/steering_service.py", "*", "from_pretrained"): _GPU_JOB,
    ("src/services/probe_monitor_run.py", "*", "from_pretrained"): _GPU_JOB + " (a resolved local snapshot)",
    ("src/utils/hf_utils.py", "*", "load_dataset_builder"): "uncalled today (FPRD FR-18); a caller must pass hub_token",
}


def _aliases(tree):
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                out[a.asname or a.name] = a.name
        elif isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = a.name
    return out


def _callee(node, aliases):
    f = node.func
    if isinstance(f, ast.Name):
        return aliases.get(f.id, f.id).split(".")[-1]
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


class _Index:
    """Functions by name across `src/` (for the transitive parameter rule) and their call sites."""

    def __init__(self, root):
        self.funcs, self.calls, self.trees, self.enclosing = {}, {}, {}, {}
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), str(path))
            rel = str(path.relative_to(root.parent))
            self.trees[rel] = tree
            for outer in ast.walk(tree):
                if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for inner in ast.walk(outer):
                        if inner is not outer and isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.enclosing.setdefault(id(inner), outer)
            for fn in ast.walk(tree):
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.funcs.setdefault(fn.name, []).append((rel, fn))
                    for call in ast.walk(fn):
                        if isinstance(call, ast.Call):
                            name = getattr(call.func, "id", getattr(call.func, "attr", None))
                            self.calls.setdefault(name, []).append((rel, fn, call))

    def approved(self, expr, fn, seen=frozenset()):
        if isinstance(expr, ast.Constant) and expr.value is False:
            return True
        if isinstance(expr, ast.Call) and getattr(expr.func, "id", getattr(expr.func, "attr", None)) == "hub_token":
            return True
        if isinstance(expr, ast.Attribute) and expr.attr == "token" and isinstance(expr.value, ast.Name):
            # ENH-011 N4: `hub_token_detail` is `hub_token` with the reason a saved token is missing.
            return self._assigned_from(expr.value.id, fn, ("hub_model_source", "hub_token_detail"))
        if isinstance(expr, ast.Name) and fn is not None:
            if self._assigned_from(expr.id, fn, ("hub_token",)):
                return True
            params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
            if expr.id in params and (fn.name, expr.id) not in seen:
                return self._every_caller_passes(fn, expr.id, params.index(expr.id), seen | {(fn.name, expr.id)})
            assigned_here = any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == expr.id
                                                                  for t in n.targets) for n in ast.walk(fn))
            outer = self.enclosing.get(id(fn))
            if not assigned_here and expr.id not in params and outer is not None:
                return self.approved(expr, outer, seen)  # a closure reading its enclosing function's variable
        return False

    def _assigned_from(self, name, fn, producers):
        values = [n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        return bool(values) and all(
            (isinstance(v, ast.Constant) and v.value is False)
            or (isinstance(v, ast.Call) and getattr(v.func, "id", getattr(v.func, "attr", None)) in producers)
            # `name = resolution.token`, where `resolution = hub_token_detail(...)` (ENH-011 N4)
            or (isinstance(v, ast.Attribute) and v.attr == "token" and isinstance(v.value, ast.Name)
                and v.value.id != name and self._assigned_from(v.value.id, fn, ("hub_token_detail",)))
            for v in values)

    def _every_caller_passes(self, fn, param, index, seen):
        sites = [(rel, caller, call) for rel, caller, call in self.calls.get(fn.name, [])]
        if not sites:
            return False
        offset = 1 if fn.args.args and fn.args.args[0].arg in ("self", "cls") else 0
        for _rel, caller, call in sites:
            given = {k.arg: k.value for k in call.keywords}.get(param)
            if given is None and index - offset < len(call.args) and index - offset >= 0:
                given = call.args[index - offset]
            if given is None or not self.approved(given, caller, seen):
                return False
        return True


def hub_call_violations(root: Path = SRC):
    index, out = _Index(root), []
    for rel, tree in index.trees.items():
        aliases = _aliases(tree)
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _callee(node, aliases) not in HUB_CALLEES:
                continue
            callee = _callee(node, aliases)
            fn = node
            while fn is not None and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = parents.get(fn)
            fname = fn.name if fn is not None else "<module>"
            if (rel, fname, callee) in HUB_CALLS_WITHOUT_TOKEN or (rel, "*", callee) in HUB_CALLS_WITHOUT_TOKEN:
                continue
            kw = {k.arg: k.value for k in node.keywords}
            local = isinstance(kw.get("local_files_only"), ast.Constant) and kw["local_files_only"].value is True
            if local:
                continue
            is_tokenizer = callee == "from_pretrained" and "Tokenizer" in ast.unparse(node.func)
            if is_tokenizer and not (rel == "src/core/hub_auth.py" and fname == "hub_load"):
                out.append((rel, node.lineno, "AutoTokenizer.from_pretrained outside hub_load (A36)"))
                continue
            token = kw.get("token")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call) and token is None:
                inner = node.func.value  # HfApi(token=…).dataset_info(…): the client's own token
                if _callee(inner, aliases) == "HfApi":
                    token = {k.arg: k.value for k in inner.keywords}.get("token")
            if token is None and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                # api = HfApi(token=…); api.repo_info(…)
                built = [n.value for n in ast.walk(fn or tree) if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == node.func.value.id for t in n.targets)
                         and isinstance(n.value, ast.Call) and _callee(n.value, aliases) == "HfApi"]
                if built:
                    token = {k.arg: k.value for k in built[0].keywords}.get("token")
            if token is None or not index.approved(token, fn):
                out.append((rel, node.lineno, f"{callee}: token={ast.unparse(token) if token else None}"))
    return sorted(set(out))


def test_every_hub_call_passes_token():
    assert hub_call_violations() == []


def test_the_guard_resolves_aliases(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text(
        "from datasets import load_dataset as hf_load_dataset\n"
        "def f(repo):\n    return hf_load_dataset(repo)\n")
    assert [v[1] for v in hub_call_violations(tmp_path / "src")] == [3]


def test_the_guard_applies_the_stricter_token_rule(tmp_path):
    """R3A-8: `token=hf_token` with a caller passing nothing is red; `token=False`, `hub_token(...)` and a parameter
    every caller fills from `hub_token` are green; a tokenizer outside `hub_load` is red even with token=False."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text(
        "from huggingface_hub import snapshot_download\n"
        "from transformers import AutoTokenizer\n"
        "def load(repo, hf_token=None):\n    return snapshot_download(repo, token=hf_token)\n"   # 4: red (caller)
        "def caller(repo):\n    return load(repo)\n"
        "def good(repo, tok):\n    return snapshot_download(repo, token=tok)\n"           # 8: green
        "def good_caller(repo):\n    return good(repo, hub_token('anonymous'))\n"
        "def anon(repo):\n    return snapshot_download(repo, token=False)\n"            # 12: green
        "def tok(repo):\n    return AutoTokenizer.from_pretrained(repo, token=False)\n")  # 14: red (A36)
    assert [v[1] for v in hub_call_violations(tmp_path / "src")] == [4, 14]


def test_every_load_model_from_hf_call_passes_a_policy_token():
    """`load_model_from_hf`'s `hf_token` reaches three Hub loads; the one caller a REST agent can reach (the model
    download) passes `hub_source.token` (A36); the in-request logit lens loads local-only for an agent (A42). The
    rest are operator-approved GPU jobs (listed above)."""
    index = _Index(SRC)
    callers = [(rel, call) for rel, _fn, call in index.calls.get("load_model_from_hf", [])]
    download = [c for rel, c in callers if rel == "src/workers/model_tasks.py"]
    assert len(download) == 1
    kw = {k.arg: ast.unparse(k.value) for k in download[0].keywords}
    assert kw["hf_token"] == "hub_source.token" and kw["local_files_only"] == "hub_source.local_files_only"
    assert kw["repo_id"] == "hub_source.source"


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 21.9 — the DYNAMIC Hub backstop (A42; A43 V-2). EVERY live route, every method, is sent its fixture request
# (`tests/support/backstop_world.py`: rows in the test database, files under the test's data root) with the agent
# header, `HF_TOKEN` set and a stored token in the settings. The fake Hub sits beneath huggingface_hub's own client
# (so beneath transformers and datasets) and records each request with its `Authorization`; a request to a Hub host
# through any other client is recorded at the transport (`backstop.install_primitives`) and refused. No name
# resolves beyond localhost, every primitive records and raises, so nothing is queued and nothing leaves.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
from tests.support import backstop as _bs  # noqa: E402
from tests.support import backstop_world as _bw  # noqa: E402
from tests.support.hub_spy import installed_fake_hub  # noqa: E402

HUB_AGENT = {"X-MiStudio-Agent": "miassist"}

#: The routes whose fixture request REACHES the Hub for an agent, request by request ((method, route, variant):
#: what it reads). The set reached must equal this exactly (A43 V-2): a route that stops reaching the Hub turns
#: the test red (it would otherwise pass vacuously), and a new Hub path is a visible change.
HUB_ROUTES: dict[tuple[str, str, int], str] = {
    ("GET", "/api/v1/datasets/{dataset_id}/samples", 0): "the aliased `load_dataset` fallback (R2A-3)",
    ("GET", "/api/v1/datasets/{dataset_id}/samples", 1): "the tokenizer that decodes the samples (`hub_load`)",
    ("POST", "/api/v1/datasets/download", 0): "the revision check before the download is queued",
    ("POST", "/api/v1/datasets/tokenize-preview", 0): "the tokenizer (`hub_load`, cached per policy)",
    ("POST", "/api/v1/saes/hf/preview", 0): "the repository's file list",
    ("POST", "/api/v1/jlens/acquire/preview", 0): "the repository's file list",
}
HUB_ROUTES_SIZE = 6

#: FR-18's routes whose request does NOT reach the Hub for an agent, each with its reason (A43 V-2). Each is swept
#: and must reach nothing; a row that starts reaching the Hub belongs in HUB_ROUTES instead.
_GATED = "gated (FR-13): an agent's call files an approval; the approved re-call runs under `hub_auth=\"anonymous\"`"
ROUTES_WITHOUT_A_HUB_PATH: dict[tuple[str, str], str] = {
    ("GET", "/api/v1/features/{feature_id}/logit-lens"): (
        "A42: for an agent the logit lens never falls back to the Hub; a model whose files are not on disk is "
        "`409 model_files_not_on_disk` (the fixture names such a model, so a restored fallback reaches the Hub)"),
    ("POST", "/api/v1/datasets/{dataset_id}/tokenize"): (
        "the tokenizer is read in the worker (`tokenize_dataset_task`, `hub_auth` in the message; "
        "`test_agent_tokenize_task_is_anonymous`)"),
    ("POST", "/api/v1/saes/download"): (
        "the files are read in the worker (`download_sae_task`, `hub_auth` in the message)"),
    ("POST", "/api/v1/models/download"): _GATED + " (row 19)",
    ("POST", "/api/v1/models/{model_id}/redownload"): _GATED + " (row 20)",
    ("POST", "/api/v1/task-queue/{task_queue_id}/retry"): _GATED + " (row 43)",
    ("POST", "/api/v1/jlens/acquire"): _GATED + " (row 18)",
    ("POST", "/api/v1/probe-monitors/probes/{probe_id}/publish"): _GATED + " (P1)",
    ("POST", "/api/v1/probe-monitors/behaviours/{key}/family/publish"): _GATED + " (P2)",
    ("POST", "/api/v1/saes/upload"): _GATED + " (P3)",
    ("POST", "/api/v1/jlens/publish"): _GATED + " (P4)",
}

#: GET routes whose handler the agent's fixture cannot reach, with the reason (the write routes are the GPU
#: backstop's `BACKSTOP_UNREACHED`; gated routes answer 202 before their handler).
HUB_SWEEP_UNREACHED: dict[tuple[str, str], str] = {
    # ENH-010 044-A (R2 review fix OR2-044a-H1, 2026-11-17): the bundle's full manifest, its files and its live
    # sensitivity refuse every agent (`operator_or_service_only`) before the handler; agents read `/summary`.
    ("GET", "/api/v1/wsp/bundles/{bundle_id}"): "operator_or_service_only refuses every agent before the handler",
    ("GET", "/api/v1/wsp/bundles/{bundle_id}/files/{name}"): (
        "operator_or_service_only refuses every agent before the handler"),
    ("GET", "/api/v1/wsp/bundles/{bundle_id}/sensitivity"): (
        "operator_or_service_only refuses every agent before the handler"),
    ("GET", "/api/v1/probe-monitors/dataset-purges"): "an operator-only dependency refuses every agent 403",
}

_HUB_SWEEP: dict[str, list] = {}


@pytest.fixture
async def hub_world(async_session, monkeypatch, tmp_path, blocked_sockets):
    from src.core.config import settings
    from src.main import app
    from src.services import analysis_service

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(settings, "data_dir", data)
    world = await _bw.build_world(async_session, data, app)
    _bw.stub_siblings(monkeypatch, world)
    # The SAE read of the logit lens (a local file) is stubbed, so the route reaches its model check.
    monkeypatch.setattr(analysis_service, "load_sae_auto_detect",
                        lambda *a, **k: ({"W_dec": torch.zeros(4, 8)}, {}, "fixture"))
    rec = _bs.install_primitives(monkeypatch)
    with installed_fake_hub(monkeypatch, tmp_path, _bw.hub_repos()) as hub:
        yield app, world, rec, hub


async def _hub_sweep(app, world, rec, hub) -> list:
    from src.core.approval_gate import iter_routes

    routes = [(m, p, r) for p, r in iter_routes(app) for m in sorted(r.methods)
              if m in ("GET", "POST", "PUT", "PATCH", "DELETE")]
    return await _bs.sweep(app, routes, lambda m, p: _bw.world_requests(app, m, p, world), rec, headers=HUB_AGENT,
                           hub=hub)


def _tokens_sent(outcomes) -> list:
    return [(o.method, o.path, o.variant, h) for o in outcomes for h in o.hub if h[2]]


async def test_no_route_sends_a_hub_token_for_an_agent(hub_world):
    """Every route, every method, under the agent header: no request reaching the Hub carries `Authorization`."""
    outcomes = await _hub_sweep(*hub_world)
    _HUB_SWEEP["outcomes"] = outcomes
    assert len({(o.method, o.path) for o in outcomes}) >= 420, "the sweep did not see every route"
    assert sum(len(o.hub) for o in outcomes) >= HUB_ROUTES_SIZE, "the sweep never reached the Hub: it proves nothing"
    assert _tokens_sent(outcomes) == []


async def test_the_hub_backstop_reaches_the_hub(hub_world):
    """A43 (V-2): each listed Hub route's request reaches the fake Hub, the set reached is exactly the list (size
    pinned), FR-18's other routes reach nothing for an agent, and every handler runs unless gated or listed."""
    from src.core.approval_gate import gated_routes
    from tests.unit.test_rest_gpu_gate import BACKSTOP_UNREACHED

    app = hub_world[0]
    outcomes = _HUB_SWEEP.get("outcomes") or await _hub_sweep(*hub_world)
    assert len(HUB_ROUTES) == HUB_ROUTES_SIZE
    reached = {(o.method, o.path, o.variant) for o in outcomes if o.hub}
    assert reached == set(HUB_ROUTES), {"listed, never reached the Hub": sorted(set(HUB_ROUTES) - reached),
                                        "reached the Hub, not listed": sorted(reached - set(HUB_ROUTES))}
    swept = {(o.method, o.path): o for o in outcomes}
    for key, reason in ROUTES_WITHOUT_A_HUB_PATH.items():
        assert reason.strip() and key in swept, key
        assert not swept[key].hub, (key, swept[key].hub)
    assert not {(m, p) for m, p, _ in HUB_ROUTES} & set(ROUTES_WITHOUT_A_HUB_PATH)
    logit = swept[("GET", "/api/v1/features/{feature_id}/logit-lens")]
    assert logit.entered and logit.status == 409 and "model_files_not_on_disk" in logit.text, logit
    unentered = {(o.method, o.path) for o in outcomes if not o.entered}
    expected = set(gated_routes(app)) | set(BACKSTOP_UNREACHED) | set(HUB_SWEEP_UNREACHED)
    assert unentered == expected, {"unentered, not listed": sorted(unentered - expected),
                                   "listed, entered": sorted(expected - unentered)}


async def test_the_hub_backstop_catches_a_token(hub_world):
    """The backstop's own positive control (21.9): a fixture route calling `HfApi(token=None).model_info(...)` under
    the agent header sends the deployment's token, and the sweep reports it — as it reports a token sent to a Hub
    host through another client; a fixture route that reads nothing is reported as not reaching the Hub."""
    from fastapi import FastAPI
    from huggingface_hub import HfApi

    from src.core.approval_gate import iter_routes

    _, _, rec, hub = hub_world
    fixture = FastAPI()

    @fixture.post("/fixture/leaks")
    def leaks():
        HfApi(token=None).model_info("fixture-org/fixture-model")

    @fixture.post("/fixture/anonymous")
    def anonymous():
        HfApi(token=False).model_info("fixture-org/fixture-model")

    @fixture.post("/fixture/direct")
    def direct():  # a Hub request beside huggingface_hub's client, with a token: seen at the transport
        try:
            httpx.get("https://huggingface.co/api/models/fixture-org/fixture-model",
                      headers={"Authorization": "Bearer hf_DEPLOYMENT_TOKEN_must_not_leave"})
        except httpx.HTTPError:
            pass

    @fixture.post("/fixture/silent")
    def silent():
        return {}

    routes = [(m, p, r) for p, r in iter_routes(fixture) for m in r.methods]
    blank = {"query": {}, "body": None}
    outcomes = await _bs.sweep(fixture, routes, lambda m, p: [{**blank, "path": p}], rec, headers=HUB_AGENT, hub=hub)
    assert {o.path for o in outcomes if any(h[2] for h in o.hub)} == {"/fixture/leaks", "/fixture/direct"}
    assert {o.path for o in outcomes if not o.hub} == {"/fixture/silent"}
