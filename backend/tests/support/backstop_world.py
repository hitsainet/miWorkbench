"""Per-route fixtures for the dynamic backstops (045 FTASKS 21.9, 22.1; CHAIN_PLAN A40, A43 V-2).

`build_world(session)` writes ONE set of rows into the test database under the fixture prefix `bkst`: a ready model with
a dataset, a tokenization and a completed activation extraction; trainings (completed and paused); an external SAE
with a completed feature extraction and features; a failed task-queue entry; circuit capture / discovery runs, a
circuit and its manifests; probe-monitor datasets, a run, a probe, a behaviour and a judge prompt. Every row a gated
route's body or path names exists, so the route's own checks pass and its handler reaches its dispatch: a refusal is
the gate's, or a failure, never a vacuous pass (A40).

`world_request(app, method, template, world)` builds the request for one route: path parameters from the world by
name (per-route overrides where one name means different rows), the body the route's own schema admits
(`gate_fixtures._model_sample`) merged with the per-route values below, which name world rows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

P = "bkst"  # the fixture prefix


@dataclass
class World:
    ids: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, key: str) -> str:
        return self.ids[key]


async def build_world(session, data_dir, app=None) -> World:  # noqa: C901 - one fixture, many rows
    """`data_dir` is the test's data root (the caller points `settings.data_dir` at it); `app` builds the rows a
    route creates best (a catalogued behaviour, a judge prompt, the labelling endpoint settings) through the live
    API, as the operator would."""
    import uuid
    from pathlib import Path

    from src.models.activation_extraction import ActivationExtraction, ExtractionStatus as AStatus
    from src.models.circuit import Circuit
    from src.models.circuit_runs import CircuitCaptureRun, CircuitDiscoveryRun
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.dataset_tokenization import DatasetTokenization, TokenizationStatus
    from src.models.external_sae import ExternalSAE, SAEStatus
    from src.models.extraction_job import ExtractionJob, ExtractionStatus
    from src.models.feature import Feature
    from src.models.labeling_prompt_template import LabelingPromptTemplate
    from src.models.model import Model, ModelStatus, QuantizationFormat
    from src.models.task_queue import TaskQueue
    from src.models.training import Training, TrainingStatus
    from src.models.validation_manifest import ValidationManifest
    from tests.support import detector_fixtures as dfx
    from tests.support import family_fixtures as ff
    from tests.support.backstop import call

    data_dir = Path(data_dir)
    w = World()
    raw = dfx.write_split(data_dir, {"text": ["hello world", "the cat sat", "a b c", "d e f"] * 12,
                                     "label": [1, 0, 1, 0] * 12}, name=f"{P}_split")
    dataset_uuid = uuid.uuid4()
    w.ids.update(code_sha="0" * 63 + "b", model=f"m_{P}", dataset=str(dataset_uuid), tokenization=f"tok_{P}", activation=f"ext_m_{P}_1",
                 training=f"train_{P}_done", training_paused=f"train_{P}_paused", sae=f"sae_{P}",
                 extraction=f"ext_sae_{P}", feature=f"feat_{P}_0", task_queue=f"tq_{P}",
                 capture=f"cap_{P}", discovery=f"dsc_{P}", circuit=f"crc_{P}", edge_manifest=f"vman_{P}_e",
                 calibration_manifest=f"vman_{P}_c")
    model_dir = data_dir / "models" / "raw" / w["model"]
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text('{"model_type": "gpt2"}')

    session.add(Model(id=w["model"], name=f"{P} model", repo_id="fixture-org/fixture-model", architecture="gpt2",
                      params_count=124_000_000, quantization=QuantizationFormat.FP16, status=ModelStatus.READY,
                      file_path=str(model_dir),
                      architecture_config={"num_hidden_layers": 12, "hidden_size": 768, "vocab_size": 50257,
                                           "num_attention_heads": 12, "model_type": "gpt2"}))
    session.add(Dataset(id=dataset_uuid, name=f"{P} dataset", source="local", hf_repo_id="fixture-org/ds",
                        status=DatasetStatus.READY, num_samples=48, raw_path=raw, extra_metadata={}))
    await session.flush()
    session.add(DatasetTokenization(id=w["tokenization"], dataset_id=dataset_uuid, model_id=w["model"],
                                    tokenizer_repo_id="fixture-org/fixture-model", status=TokenizationStatus.READY,
                                    max_length=128, tokenized_path=str(data_dir / "tokenized" / f"tok_{P}")))
    session.add(LabelingPromptTemplate(id=f"lpt_{P}", name=f"{P} template", system_message="Label it.",
                                       user_prompt_template="{examples_block}", is_default=True))
    session.add(ActivationExtraction(id=w["activation"], model_id=w["model"], dataset_id=w["dataset"],
                                     layer_indices=[0], hook_types=["residual"], max_samples=10, batch_size=8,
                                     status=AStatus.FAILED))
    hyper = {"d_model": 768, "latent_dim": 1024, "hook_type": "residual", "training_layers": [0]}
    session.add(Training(id=w["training"], model_id=w["model"], dataset_id=w["dataset"], dataset_ids=[w["dataset"]],
                         total_steps=100, current_step=100, status=TrainingStatus.COMPLETED.value,
                         hyperparameters=hyper))
    (data_dir / "trainings" / w["training"] / "community_format").mkdir(parents=True, exist_ok=True)
    session.add(Training(id=w["training_paused"], model_id=w["model"], dataset_id=w["dataset"],
                         dataset_ids=[w["dataset"]], total_steps=100, current_step=50,
                         status=TrainingStatus.PAUSED.value, hyperparameters=hyper))
    session.add(ExternalSAE(id=w["sae"], name=f"{P} sae", source="local", status=SAEStatus.READY.value,
                            local_path=f"{P}/sae", model_id=w["model"], model_name="gpt2", layer=0,
                            hook_type="hook_resid_post", n_features=16, d_model=768))
    await session.flush()
    session.add(ExtractionJob(id=w["extraction"], external_sae_id=w["sae"], config={"dataset_id": w["dataset"]},
                              status=ExtractionStatus.COMPLETED, layer_index=0, hook_type="residual"))
    await session.flush()
    for i in range(3):
        session.add(Feature(id=f"feat_{P}_{i}", external_sae_id=w["sae"], extraction_job_id=w["extraction"],
                            neuron_index=i, name=f"feature {i}", activation_frequency=0.1,
                            interpretability_score=0.5, max_activation=1.0))
    session.add(TaskQueue(id=w["task_queue"], task_type="download", entity_id=w["model"], entity_type="model",
                          status="failed", retry_params={"repo_id": "fixture-org/fixture-model",
                                                         "quantization": "FP16"}))
    # Circuits: an estimated capture, a completed discovery with candidates, a circuit with members, manifests.
    session.add(CircuitCaptureRun(id=w["capture"], status="estimated",
                                  manifest={"model_id": w["model"], "dataset_id": w["dataset"],
                                            "layers": [{"layer": 0, "sae_id": w["sae"]}]}))
    candidate = {"src": {"layer": 0, "feature_idx": 0}, "dst": {"layer": 0, "feature_idx": 1}, "score": 0.5}
    session.add(CircuitDiscoveryRun(id=w["discovery"], capture_run_id=w["capture"], status="completed",
                                    params={"granularity": "feature", "mode": "open"}, report={},
                                    candidates=[candidate], attribution_status="completed"))
    member = {"layer": 0, "feature_idx": 0, "sae_id": w["sae"], "strength": 1.0}
    session.add(Circuit(id=w["circuit"], name=f"{P} circuit", members=[member], edges=[],
                        saes=[{"layer": 0, "sae_id": w["sae"]}], discovery_run_id=w["discovery"],
                        model_id=w["model"], model_hf_id="fixture-org/fixture-model"))
    session.add(ValidationManifest(id=w["edge_manifest"], kind="edge_batch", discovery_run_id=w["discovery"],
                                   payload={"config": {}}))
    session.add(ValidationManifest(id=w["calibration_manifest"], kind="calibration", circuit_id=w["circuit"],
                                   payload={"config": {}}))
    # Probe monitors: train / eval / calibration views over the split, a completed run with a probe.
    source = await dfx.dataset_row(session, raw)
    w.ids["probe_source"] = str(source.id)
    await session.commit()

    key = None
    if app is not None:
        from tests.support.random_behaviour import behaviour_body, rng_for

        body = behaviour_body(rng_for(P), key=f"{P}-behaviour")
        made = await call(app, "POST", "/api/v1/probe-monitors/behaviours", body=body)
        assert made.status_code == 201, made.text
        key = body["key"]
        from tests.unit.test_judge_prompt_library import NEW_PROMPT

        made = await call(app, "POST", "/api/v1/probe-monitors/judge-prompts",
                          body={**NEW_PROMPT, "name": f"{P}-prompt", "behaviour": key})
        assert made.status_code == 201, made.text
        w.ids["prompt"] = made.json()["id"]
        for k, v in (("openai_compatible_endpoint", "http://127.0.0.1:9/v1"),
                     ("openai_compatible_model", "fixture-judge")):
            done = await call(app, "PUT", "/api/v1/settings", body={"key": k, "value": v})
            assert done.status_code in (200, 201), done.text
    w.ids["behaviour"] = key or f"{P}-behaviour"
    train = await ff.view(session, source, key=key, role="train")
    evaluation = await ff.view(session, source, key=key, role="eval", distribution="out_of_distribution")
    calibration = await ff.view(session, source, key=key, role="calibration")
    model_row = (await session.get(Model, w["model"]))
    probe = await ff.member(session, model_row, source, key=key, role="primary" if key else None,
                            train_view=train, render=SERVED_RENDER)
    w.ids.update(train_view=train.id, eval_view=evaluation.id, calibration_view=calibration.id, probe=probe.id,
                 probe_run=probe.run_id)
    await session.commit()
    # The Hub sweep's GET fixtures (21.9; V-2): a feature whose model has NO files on disk (the logit lens), a
    # dataset whose samples fall back to the Hub, and one whose samples decode with a Hub tokenizer.
    from datasets import Dataset as HFDataset

    w.ids.update(model_nofiles=f"m_{P}_nofiles", sae_nofiles=f"sae_{P}_nofiles", feature_nofiles=f"feat_{P}_nofiles",
                 ds_hub_fallback=str(uuid.uuid4()), ds_hub_tokenized=str(uuid.uuid4()))
    session.add(Model(id=w["model_nofiles"], name=f"{P} no files", repo_id="fixture-org/fixture-model",
                      architecture="gpt2", params_count=1, status=ModelStatus.READY, file_path=None))
    await session.flush()
    session.add(ExternalSAE(id=w["sae_nofiles"], name=f"{P} sae 2", source="local", status=SAEStatus.READY.value,
                            local_path=f"{P}/sae2", model_id=w["model_nofiles"]))
    await session.flush()
    session.add(ExtractionJob(id=f"ext_sae_{P}_nofiles", external_sae_id=w["sae_nofiles"], config={},
                              status=ExtractionStatus.COMPLETED))
    await session.flush()
    session.add(Feature(id=w["feature_nofiles"], external_sae_id=w["sae_nofiles"],
                        extraction_job_id=f"ext_sae_{P}_nofiles", neuron_index=0, name="f", activation_frequency=0.1,
                        interpretability_score=0.5, max_activation=1.0))
    session.add(Dataset(id=uuid.UUID(w["ds_hub_fallback"]), name=f"{P} hub fallback", source="HuggingFace",
                        status=DatasetStatus.READY, hf_repo_id="fixture-org/ds", raw_path=None,
                        extra_metadata={"split": "train"}))
    tokenized = data_dir / "tokenized" / f"{P}_hub"
    HFDataset.from_dict({"input_ids": [[3, 4], [5, 6, 7]]}).save_to_disk(str(tokenized))
    session.add(Dataset(id=uuid.UUID(w["ds_hub_tokenized"]), name=f"{P} hub tokenized", source="HuggingFace",
                        status=DatasetStatus.READY, raw_path=None,
                        extra_metadata={"tokenization": {"tokenizer_name": "fixture-org/fixture-model"}}))
    await session.flush()
    session.add(DatasetTokenization(id=f"tok_{P}_hub", dataset_id=uuid.UUID(w["ds_hub_tokenized"]),
                                    model_id=w["model_nofiles"], tokenized_path=str(tokenized),
                                    tokenizer_repo_id="fixture-org/fixture-model", status=TokenizationStatus.READY))
    await session.commit()
    # A staged J-lens artifact (re-validation reads one; `_ref_for` needs exactly one lens file).
    from src.core.config import settings
    from src.services.jlens_artifact_service import JLensArtifactService

    staging = JLensArtifactService(settings.jlens_artifacts_dir).staging_dir("fixture-org/fixture-model")
    staging.mkdir(parents=True, exist_ok=True)
    (staging / f"{P}_jacobian_lens.pt").write_bytes(b"\0")
    return w


#: A probe's render form recorded in full (served, with its start-of-text record), so no check loads a tokenizer.
SERVED_RENDER = {"generation_prompt": True, "add_special_tokens": False,
                 "bos_handling": {"template_wrote_bos": True, "tokenizer_added_bos": False, "bos_count": 1}}


def hub_repos() -> dict:
    """The repositories the fake Hub serves for the world (`hub_spy.FakeHub`)."""
    import json

    from tests.support.hub_spy import tiny_dataset_files, tiny_tokenizer_files

    sae = {"layer_0/cfg.json": json.dumps({"d_in": 8, "d_sae": 16}).encode(),
           "layer_0/sae_weights.safetensors": b"\0" * 64}
    return {("model", "fixture-org/fixture-model"): tiny_tokenizer_files(),
            ("model", "fixture-org/fixture-sae"): sae,
            ("dataset", "fixture-org/ds"): tiny_dataset_files()}


#: Routes the Hub sweep sends more than one request (each a set of path-parameter → world-key overrides): the
#: samples route has two Hub paths (the dataset fallback and the tokenizer).
REQUEST_VARIANTS: dict[str, list[dict[str, str]]] = {
    "/api/v1/datasets/{dataset_id}/samples": [{"dataset_id": "ds_hub_fallback"}, {"dataset_id": "ds_hub_tokenized"}],
    "/api/v1/features/{feature_id}/logit-lens": [{"feature_id": "feature_nofiles"}],
}


def world_requests(app, method: str, template: str, world: World) -> list[dict]:
    variants = REQUEST_VARIANTS.get(template)
    if not variants:
        return [world_request(app, method, template, world)]
    out = []
    for overrides in variants:
        saved = PATH_OVERRIDES.get(template)
        PATH_OVERRIDES[template] = {**(saved or {}), **overrides}
        try:
            out.append(world_request(app, method, template, world))
        finally:
            if saved is None:
                PATH_OVERRIDES.pop(template)
            else:
                PATH_OVERRIDES[template] = saved
    return out


def stub_siblings(monkeypatch, world: World) -> None:
    """miServer's side of a spot-check, as a running miServer would answer it: available, and the world's primary
    probe armed. (Reads only; the sockets stay blocked.)"""
    from src.core.config import settings
    from src.services import miserver_read

    monkeypatch.setattr(settings, "miserver_api_url", "http://127.0.0.1:9", raising=False)
    # A configured local Neuronpedia, so `POST /neuronpedia/push-local` gets past its 503 "not configured" and the
    # backstop's positive control reaches the task (row 42). Hermetic: it passed only where a developer's `.env`
    # set NEURONPEDIA_LOCAL_DB_URL, and CI has no `.env`. Loopback, nothing listens, and the sockets stay blocked.
    monkeypatch.setattr(settings, "neuronpedia_local_db_url",
                        "postgresql://fixture@127.0.0.1:9/neuronpedia", raising=False)
    monkeypatch.setattr(miserver_read, "spot_check_availability",
                        lambda: {"available": True, "url": "http://127.0.0.1:9", "reason": None})
    monkeypatch.setattr(miserver_read, "armed_probes",
                        lambda: [{"id": "mp_bkst", "source_probe_id": world["probe"]}])


#: path parameter → world key, by name (per-route overrides in PATH_OVERRIDES).
PATH_BY_NAME = {
    "model_id": "model", "training_id": "training", "sae_id": "sae", "extraction_id": "extraction",
    "extraction_job_id": "extraction", "feature_id": "feature", "task_queue_id": "task_queue",
    "dataset_id": "dataset", "tokenization_id": "tokenization", "circuit_id": "circuit", "probe_id": "probe",
    "key": "behaviour", "run_id": "discovery", "manifest_id": "edge_manifest", "prompt_id": "prompt",
    "code_sha256": "code_sha",
}
PATH_OVERRIDES: dict[str, dict[str, str]] = {
    "/api/v1/models/{model_id}/extractions/{extraction_id}/retry": {"extraction_id": "activation"},
    "/api/v1/trainings/{training_id}/control": {"training_id": "training_paused"},
    "/api/v1/circuit-capture/{run_id}/confirm": {"run_id": "capture"},
    "/api/v1/circuits/calibration-manifests/{manifest_id}/reproduce": {"manifest_id": "calibration_manifest"},
    "/api/v1/circuit-capture/{run_id}/cancel": {"run_id": "capture"},
    "/api/v1/probe-monitors/datasets/{dataset_id}/columns": {"dataset_id": "eval_view"},
    "/api/v1/probe-monitors/datasets/{dataset_id}/archive": {"dataset_id": "eval_view"},
    "/api/v1/probe-monitors/datasets/{dataset_id}/unarchive": {"dataset_id": "eval_view"},
    "/api/v1/probe-monitors/datasets/{dataset_id}/confirm-positive": {"dataset_id": "eval_view"},
    "/api/v1/probe-monitors/behaviours/{key}/views/{dataset_id}": {"dataset_id": "eval_view"},
    "/api/v1/probe-monitors/runs/{run_id}/cancel": {"run_id": "probe_run"},
}

_LOCAL_LLM = "http://127.0.0.1:9/v1"  # loopback: the endpoint checks accept it; nothing listens (no socket leaves)
_TEXT = {"text": "fixture text"}

#: route template → a function of the world giving the body values that name world rows.
BODIES: dict[str, Callable[[World], Any]] = {
    "/api/v1/circuit-capture": lambda w: {"dataset_id": w["dataset"], "model_id": w["model"],
                                          "layers": [{"layer": 0, "sae_id": w["sae"]}]},
    "/api/v1/circuits/{circuit_id}/calibration": lambda w: {"judge_endpoint": _LOCAL_LLM,
                                                            "judge_model": "fixture-judge"},
    "/api/v1/circuits/steering-samples": lambda w: {"artifact": {"kind": "circuit", "circuit_id": w["circuit"]},
                                                    "dials": [1.0], "prompts": ["fixture prompt"]},
    "/api/v1/jlens/fit": lambda w: {"model_id": w["model"], "prompts": ["The capital of France is"]},
    "/api/v1/jlens/artifacts/revalidate": lambda w: {"model_id": w["model"]},
    "/api/v1/jlens/band-report": lambda w: {"model_id": w["model"]},
    "/api/v1/jlens/readout": lambda w: {"model_id": w["model"]},
    "/api/v1/jlens/probe": lambda w: {"model_id": w["model"]},
    "/api/v1/jlens/decompose": lambda w: {"model_id": w["model"]},
    "/api/v1/jlens/template-lens": lambda w: {"model_id": w["model"], "vocabulary": {"capital": [" Paris"]}},
    "/api/v1/jlens/interventions": lambda w: {"model_id": w["model"]},
    "/api/v1/jlens/acquire": lambda w: {"model_id": w["model"], "replace_staged": True},
    "/api/v1/models/download": lambda w: {"repo_id": "fixture-org/fixture-model"},
    "/api/v1/models/{model_id}/extract-activations": lambda w: {
        "dataset_id": w["dataset"], "layer_indices": [0], "hook_types": ["residual"], "max_samples": 10},
    "/api/v1/probe-monitors/runs": lambda w: {
        "model_id": w["model"], "train_dataset_id": w["train_view"], "eval_dataset_ids": [w["eval_view"]],
        "config": {"layers": [0], "stride": None, "acknowledge_rule_choice": "backstop fixture"}},
    "/api/v1/probe-monitors/probes/{probe_id}/evaluate": lambda w: [w["eval_view"]],
    "/api/v1/probe-monitors/probes/{probe_id}/score": lambda w: _TEXT,
    "/api/v1/probe-monitors/probes/{probe_id}/score-batch": lambda w: {"inputs": [_TEXT]},
    "/api/v1/probe-monitors/behaviours/{key}/family/score-batch": lambda w: {"inputs": [_TEXT]},
    "/api/v1/probe-monitors/judge-runs": lambda w: {"endpoint": _LOCAL_LLM, "model": "fixture-judge",
                                                     "dataset_ids": [w["eval_view"]], "prompt_id": w["prompt"]},
    "/api/v1/probe-monitors/spot-checks": lambda w: {"behaviour": w["behaviour"], "prompt_id": w["prompt"],
                                                     "endpoint": _LOCAL_LLM, "model": "fixture-judge", "size": 3},
    "/api/v1/probe-monitors/judge-prompts/try": lambda w: {"prompt_id": w["prompt"], "endpoint": _LOCAL_LLM,
                                                           "model": "fixture-judge", "dataset_id": w["eval_view"],
                                                           "rows": 2},
    "/api/v1/trainings": lambda w: {"model_id": w["model"], "dataset_ids": [w["dataset"]]},
    "/api/v1/trainings/{training_id}/control": lambda w: {"action": "resume"},
    "/api/v1/saes/{sae_id}/extract-features": lambda w: {"dataset_ids": [w["dataset"]]},
    "/api/v1/saes/batch-extract-features": lambda w: {"sae_ids": [w["sae"]], "dataset_id": w["dataset"]},
    "/api/v1/labeling": lambda w: {"extraction_job_id": w["extraction"], "labeling_method": "local"},
    "/api/v1/labeling/panel": lambda w: {"extraction_job_id": w["extraction"], "labeling_method": "local",
                                         "feature_ids": [w["feature"]]},
    "/api/v1/extractions/{extraction_id}/label": lambda w: {"labeling_method": "local"},
    "/api/v1/labeling/{extraction_job_id}/resume-sweep": lambda w: {"config": {"labeling_method": "local"}},
    "/api/v1/labeling/trials": lambda w: {"extraction_job_id": w["extraction"], "feature_ids": [w["feature"]]},
    "/api/v1/neuronpedia/export": lambda w: {"sae_id": w["sae"]},
    "/api/v1/neuronpedia/compute-dashboard-data": lambda w: {"sae_id": w["sae"]},
    "/api/v1/extractions/{extraction_id}/label/enhanced/batch": lambda w: {"feature_ids": [w["feature"]]},
    # ---- routes that are NOT gated: a body their own parsing accepts, naming world rows (22.1's other half).
    "/api/v1/datasets/download": lambda w: {"repo_id": "fixture-org/ds"},
    "/api/v1/datasets/tokenize-preview": lambda w: {"tokenizer_name": "fixture-org/fixture-model", "text": "hello world"},
    "/api/v1/datasets/{dataset_id}/tokenize": lambda w: {"model_id": w["model"]},
    "/api/v1/models/{model_id}/estimate-extraction": lambda w: {"hook_types": ["residual"], "layer_indices": [0]},
    "/api/v1/extraction-templates": lambda w: {"hook_types": ["residual"], "layer_indices": [0]},
    "/api/v1/extraction-templates/{template_id}": lambda w: {"hook_types": ["residual"]},
    "/api/v1/labeling-prompt-templates": lambda w: {"user_prompt_template": "{examples_block}"},
    "/api/v1/saes/hf/preview": lambda w: {"repo_id": "fixture-org/fixture-sae"},
    "/api/v1/jlens/acquire/preview": lambda w: {"repo_id": "fixture-org/fixture-model"},
    "/api/v1/saes/download": lambda w: {"repo_id": "fixture-org/fixture-sae", "filepath": "layer_0"},
    "/api/v1/circuit-discovery/refine-cluster-edge": lambda w: {"capture_run_id": w["capture"], "up_layer": 0,
                                                                "down_layer": 1},
    "/api/v1/circuit-discovery": lambda w: {"capture_run_id": w["capture"]},
    "/api/v1/probe-monitors/datasets": lambda w: {"name": f"{P}-view", "dataset_id": w["probe_source"],
                                                  "input_column": "text", "label_column": "label",
                                                  "label_mapping": {"1": "positive", "0": "negative"}},
    "/api/v1/probe-monitors/judge-prompts": lambda w: {"name": f"{P}-other", "behaviour": w["behaviour"],
                                                       "answer_form": {"kind": "binary"}},
    "/api/v1/probe-monitors/judge-prompts/preview": lambda w: {"prompt_id": w["prompt"],
                                                               "dataset_id": w["eval_view"]},
    "/api/v1/probe-monitors/judge-prompts/import-rubric": lambda w: {"name": f"{P}-rubric",
                                                                     "behaviour": w["behaviour"], "document": {}},
    "/api/v1/probe-monitors/judge-prompts/{prompt_id}/clone": lambda w: {"name": f"{P}-clone"},
    "/api/v1/mcp/approvals": lambda w: {"tool_name": "start_probe_training_run",
                                        "payload": {"model_id": w["model"]}},
    "/api/v1/probe-monitors/datasets/{dataset_id}/columns": lambda w: {"text_column": "text"},
    "/api/v1/probe-monitors/rule-sets/validate": lambda w: {"behaviour_key": w["behaviour"]},
    "/api/v1/probe-monitors/rule-signals/evaluate": lambda w: {"behaviour_key": w["behaviour"], "text": "hello"},
    "/api/v1/probe-monitors/blend-reports": lambda w: {"behaviour_key": w["behaviour"]},
    "/api/v1/steering/experiments": lambda w: {"sae_id": w["sae"], "model_id": w["model"]},
    # ENH-010 044-A (R3 review fix OR3-044a-L4): a body the scoring contract accepts, so an operator's request enters
    # the handler (it answers `bundle_not_found`) and an agent's 403 is the dependency's, not a 422 on the body.
    "/api/v1/wsp/score": lambda w: {
        "kind": "miworkbench.wsp-score-request/v1", "bundle_id": "wsb_000000b0057e", "bundle_sha256": "0" * 64,
        "run_ref": {"miforge_run_id": f"{P}-run"},
        "heldout_ref": {"owner": "miworkbench", "sets": [{"name": "unseen", "view_id": w["eval_view"], "role": "eval",
                                                          "rows": 1, "row_ids_sha256": "0" * 64}]},
        "heads": [{"label": "parent"}], "primary_sets": ["unseen"]},
}
QUERIES: dict[str, Callable[[World], dict]] = {
    "/api/v1/neuronpedia/push-local": lambda w: {"sae_id": w["sae"]},
    "/api/v1/probe-monitors/probes/{probe_id}/recalibrate": lambda w: {"recut_windows": True},
}


def _merge(base: Any, patch: dict) -> Any:
    if not isinstance(base, dict):
        return patch
    out = dict(base)
    for key, value in patch.items():
        out[key] = _merge(out.get(key), value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def world_request(app, method: str, template: str, world: World) -> dict:
    """`{"path", "path_params", "query", "body"}` for one route, naming world rows."""
    from src.core.approval_gate import _flat, iter_routes
    from tests.support.gate_fixtures import _model_sample

    route = {(m, p): r for p, r in iter_routes(app) for m in r.methods}[(method, template)]
    import uuid as _uuid

    overrides = PATH_OVERRIDES.get(template, {})
    annotations = {p.name: p.field_info.annotation for p in route.dependant.path_params}
    path_params = {}
    for name in re.findall(r"\{([^}]+)\}", template):
        key = overrides.get(name) or PATH_BY_NAME.get(name)
        if key and key in world.ids:
            path_params[name] = world.ids[key]
        elif annotations.get(name) is _uuid.UUID:
            path_params[name] = str(_uuid.UUID(int=0xB057))  # a well-formed id that names no row
        elif annotations.get(name) is int:
            path_params[name] = "0"
        else:
            path_params[name] = f"{P}_{name}"
    flat = _flat(route)
    query = {p.alias: _model_sample(p.field_info.annotation) for p in flat.query_params if p.field_info.is_required()}
    query.update(QUERIES.get(template, lambda w: {})(world))
    if not flat.body_params:
        body = None
    elif len(flat.body_params) == 1 and not getattr(flat.body_params[0].field_info, "embed", False):
        body = _model_sample(flat.body_params[0].field_info.annotation)
    else:
        body = {p.alias: _model_sample(p.field_info.annotation) for p in flat.body_params}
    if template in BODIES:
        patch = BODIES[template](world)
        body = _merge(body if body is not None else {}, patch) if isinstance(patch, dict) else patch
    concrete = template
    for name, value in path_params.items():
        concrete = concrete.replace("{" + name + "}", value)
    return {"path": concrete, "path_params": path_params, "query": query, "body": body}
