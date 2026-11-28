"""An HTTP case for every covered route (046 P8, FTASKS 9.1; FTID §8.4; WT8).

A covered route is a key of the committed answer contract's `responses` or `excluded`
(`docs/schemas/miworkbench-answers-v1.json`): the routes an MCP tool or a sibling app calls. Each gets one case, or two
when the route is gated (an agent call outside `MCP_POD_AGENTS` answers 202 and files an approval, IC-4; the operator's
call answers the route's own status). A case drives the LIVE app, so the answer middleware
(`tests/support/answer_middleware.py`) checks the real served bytes against the contract.

- **Rows.** `backstop_world.build_world` writes one world of rows (a ready model, dataset, extraction, SAE, features,
  circuits, probe-monitor rows, a behaviour, a judge prompt); a case names them through `backstop_world.world_request`
  (path parameters by name, the body the route's own schema admits, merged with `backstop_world.BODIES`). `OVERRIDES`
  below holds what a covered route needs beyond the gated set the world was built for.
- **Dispatch.** GPU and worker dispatch is patched at the primitives `tests/support/route_walk.py` names
  (`GPU_PRIMITIVES`, `ENQUEUE_METHODS`): nothing is queued for real, and each reach is recorded.
- **The router map.** `covered_for_router("<file>.py")` gives the keys whose endpoint is defined in that file (the
  endpoint map is the live app's, through `answers_contract.operations`), for the per-router modules of 9.2.
"""
from __future__ import annotations

import functools
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

AGENT_HEADERS = {"X-MiStudio-Agent": "miassist"}  # an agent outside MCP_POD_AGENTS: a gated route files an approval


@dataclass(frozen=True)
class Case:
    key: str                 # "<METHOD> <path template>", the contract key
    caller: str              # "operator" or "agent"
    expect: int              # the status the case requires

    @property
    def id(self) -> str:
        return f"{self.key} [{self.caller}]"


@dataclass
class Override:
    """What one key needs beyond `world_request`: path-parameter world keys, a body patch or a whole body, query
    values, a status other than the contract's lone 2xx, and a setup run against the world before the call."""
    path: dict[str, str] = field(default_factory=dict)
    body: Optional[Callable[[Any], Any]] = None
    replace_body: bool = False
    query: Optional[Callable[[Any], dict]] = None
    expect: Optional[int] = None
    setup: Optional[Callable[..., Any]] = None  # async (Context) -> None, after the family's setup


async def extend_world(session: Any, world: Any) -> Any:
    """Rows the covered reads name beyond `backstop_world.build_world` (which was built for the gated set): a saved
    steering experiment, a cluster profile, and enough stored activations (10) on the world's first feature for the
    ablation estimate. Ids land in `world.ids`."""
    import uuid

    from src.models.cluster_profile import ClusterProfile
    from src.models.feature_activation import FeatureActivation
    from src.models.steering_experiment import SteeringExperiment

    experiment = SteeringExperiment(id=uuid.uuid4(), name="bkst experiment", sae_id=world["sae"],
                                    model_id=world["model"], comparison_id=f"cmp_{uuid.uuid4().hex[:8]}",
                                    prompt="The cat", selected_features=[], generation_params={}, results={}, tags=[])
    profile = ClusterProfile(name="bkst profile", sae_id=world["sae"], model_id=world["model"],
                             members=[{"feature_idx": 0, "strength": 1.0}], schema_version="1")
    session.add_all([experiment, profile])
    for i in range(10):
        session.add(FeatureActivation(feature_id=world["feature"], sample_index=i, max_activation=1.0 + i,
                                      tokens=["a", "b"], activations=[0.0, 1.0 + i]))
    await session.commit()
    # The SAE's files where `settings.resolve_data_path` looks (the steering routes check the path exists).
    from src.core.config import settings
    from tests.support.backstop_world import hub_repos

    sae_dir = settings.resolve_data_path(f"{backstop_world_prefix()}/sae")
    for name, content in hub_repos()[("model", "fixture-org/fixture-sae")].items():
        (sae_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (sae_dir / name).write_bytes(content)
    # The world model's checkpoint as a download leaves it (the Hub cache layout under the row's `file_path`): a
    # tiny real Llama and the fake Hub's tokenizer, so the weight-only reads (the J-lens annotation, the attention
    # readout, the token check) run on real tensors and nothing loads a model.
    model_row = await session.get(__import__("src.models.model", fromlist=["Model"]).Model, world["model"])
    write_snapshot(settings.resolve_data_path(model_row.file_path), "fixture-org/fixture-model")
    world.ids.update(experiment=str(experiment.id), profile=profile.id, jlens_slug=_jlens_artifact())
    return world


#: The tiny checkpoint's files (built once per process): config, safetensors weights and the tokenizer.
_SNAPSHOT: dict[str, bytes] = {}
SNAPSHOT_REVISION = "c0ffee" + "0" * 34
TINY_D_MODEL = 8


def _snapshot_files() -> dict[str, bytes]:
    if not _SNAPSHOT:
        import tempfile

        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        from tests.support.hub_spy import tiny_tokenizer_files

        config = LlamaConfig(hidden_size=TINY_D_MODEL, intermediate_size=16, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=2, vocab_size=8,
                             max_position_embeddings=64)
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            LlamaForCausalLM(config).save_pretrained(tmp, safe_serialization=True)
            for path in Path(tmp).iterdir():
                _SNAPSHOT[path.name] = path.read_bytes()
        for name, content in tiny_tokenizer_files().items():
            if name not in ("config.json", "model.safetensors", "pytorch_model.bin"):
                _SNAPSHOT[name] = content
    return dict(_SNAPSHOT)


def write_snapshot(model_dir: Path, repo_id: str) -> Path:
    """The tiny checkpoint under `model_dir` in the Hub cache layout `resolve_snapshot_dir` and
    `AutoTokenizer.from_pretrained(repo_id, cache_dir=model_dir, local_files_only=True)` read."""
    base = Path(model_dir) / ("models--" + repo_id.replace("/", "--"))
    snap = base / "snapshots" / SNAPSHOT_REVISION
    snap.mkdir(parents=True, exist_ok=True)
    for name, content in _snapshot_files().items():
        (snap / name).write_bytes(content)
    (base / "refs").mkdir(parents=True, exist_ok=True)
    (base / "refs" / "main").write_text(SNAPSHOT_REVISION)
    return snap


def backstop_world_prefix() -> str:
    from tests.support.backstop_world import P

    return P


def _jlens_artifact(repo_id: str = "fixture-org/fixture-model") -> str:
    """A committed J-lens artifact for the world's model under `settings.jlens_artifacts_dir` (the caller points
    `settings.data_dir` at the test's root): identity Jacobians at layers 0 to 2, validated with the three live checks
    passing, as `test_jlens_artifact_service.py` builds one. Returns its slug."""
    import torch

    from src.core.config import settings
    from src.services.jlens_artifact_service import JLensArtifactService
    from src.services.jlens_validation import CheckClass, CheckResult, CheckStatus

    service = JLensArtifactService(settings.jlens_artifacts_dir)
    # What a document states, its dimensions included (`jspace_documents.artifact_document` never defaults them).
    recipe = (f"model: {repo_id}\ncorpus: fixture-corpus\nn_prompts: 4\nseq_len: 16\n"
              f"d_model: {TINY_D_MODEL}\nn_layers: 3\nn_vocab: 8\n")
    passes = {"semantic_result": CheckResult(CheckClass.SEMANTIC, CheckStatus.PASS, "ok"),
              "cross_impl_result": CheckResult(CheckClass.CROSS_IMPLEMENTATION, CheckStatus.PASS, "ok"),
              "round_trip_result": CheckResult(CheckClass.ROUND_TRIP, CheckStatus.PASS, "ok")}
    jacobians = {i: torch.eye(TINY_D_MODEL, dtype=torch.float16) for i in (0, 1, 2)}
    # Committed twice, so the first is archived (`.superseded`, what a restore reads), then staged once more (what a
    # re-validation reads: a commit empties staging).
    slug = None
    for _ in range(2):
        ref = service.write_staged(repo_id, jacobians, recipe)
        slug = service.commit(repo_id, service.validate(ref, TINY_D_MODEL, [0, 1, 2], 512, **passes)).slug
    service.write_staged(repo_id, jacobians, recipe)
    return slug


async def _blend_world(ctx: "Context") -> None:
    """The probe-monitor family's rows (`blend_world.build`): a catalogued behaviour with a family of three members,
    blend-fit, two test and a calibration view, every member's stored evaluations, and one finished blend report
    (the REAL task run in-process). The case's behaviour, probe, view, evaluation and report are this world's."""
    import hashlib

    import numpy as np

    from src.models.dataset import Dataset
    from tests.support import blend_world
    from tests.support import detector_fixtures as dfx
    from tests.support import rule_set_fixtures as rf
    from tests.support.backstop import call
    from tests.support.backstop_world import SERVED_RENDER

    seed = f"crc-{ctx.case.id}"
    built = await blend_world.build(ctx.client, ctx.session, seed=seed, rule_set=lambda key: rf.masking_set(seed, key))
    # Each member BUILT, as a build leaves it: its definition file on disk and the identity the build resolved (039),
    # before the report is computed (a report records its members' digests). Rendered in full, with its
    # start-of-text record, so no check loads a tokenizer (`SERVED_RENDER`).
    definitions = ctx.data_dir / "probe_definitions"
    definitions.mkdir(parents=True, exist_ok=True)
    for j, member in enumerate(built.members):
        path = definitions / f"{member.id}.probe.json"
        path.write_bytes(json.dumps(_probe_definition(f"probe-{member.id}", member.rung), indent=1).encode() + b"\n")
        member.definition_path = str(path)
        member.definition_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        member.definition_build = {"resolved": {
            "detector_digest": hashlib.sha256(f"{member.id}:{j}".encode()).hexdigest(),
            "decision_digest": member.decision_digest, "decision_revision": 1}}
        member.render_form = dict(SERVED_RENDER)
    await ctx.session.commit()
    await blend_world.seed_rule_verdicts(ctx.session, built, seed=seed, veto_share=0.3)  # the rule set's, cached
    report = await blend_world.done_report(ctx.client, ctx.monkeypatch, built)
    probe = built.members[0]
    view = built.tests[0]
    evaluation = built.evaluations[probe.id][view.id]
    # The family's corpus on disk (`family_fixtures.source` records `/nonexistent`): rows a read can open, as many
    # as the stored positions reach (a calibration view holds 2n positions drawn from 6n), each carrying the label
    # the case's evaluation stored for it (a row whose label moved is refused `rows_moved`).
    labels = [0] * 3000
    for position, label in zip(np.load(evaluation.positions_path).tolist(), np.load(evaluation.labels_path).tolist(),
                               strict=True):
        labels[position] = int(label)
    corpus = await ctx.session.get(Dataset, view.dataset_id)
    corpus.raw_path = dfx.write_split(ctx.data_dir, {"text": [f"row {i}" for i in range(3000)], "label": labels,
                                                     "channel": ["chat" if i % 3 else "mail" for i in range(3000)]},
                                      name="crc_blend")
    await ctx.session.commit()
    # A judge prompt for this behaviour (a spot-check refuses a prompt that judges another).
    from src.main import app
    from tests.unit.test_judge_prompt_library import NEW_PROMPT

    made = await call(app, "POST", "/api/v1/probe-monitors/judge-prompts",
                      body={**NEW_PROMPT, "name": "crc-blend-prompt", "behaviour": built.key})
    assert made.status_code == 201, made.text
    # `eval_view` stays the backstop world's (its files exist); the blend world's view is `blend_view`.
    ctx.world.ids.update(behaviour=built.key, probe=probe.id, blend_view=view.id, probe_run=probe.run_id,
                         evaluation=evaluation.id, report=report, model_name=built.model.id,
                         prompt=made.json()["id"], blend_train_view=built.runs[probe.id].train_dataset_id)


async def _record_run(ctx: "Context") -> None:
    from src.models.steering_record_run import SteeringRecordRun

    run = SteeringRecordRun(status="running", artifact_kind="circuit", artifact_ref=ctx.world["circuit"])
    ctx.session.add(run)
    await ctx.session.commit()
    ctx.world.ids["record_run"] = run.id


@dataclass
class Context:
    """What a setup step gets: the case, its world, an ASGI client on the live app, the test's session and
    monkeypatch, and the test's data root."""
    case: Case
    world: Any
    client: Any
    session: Any
    monkeypatch: Any
    data_dir: Path


#: Rows a whole route family needs beyond the world (prefix → setup), built before the family's every case.
FAMILY_SETUPS: dict[str, Callable[[Context], Any]] = {
    "/api/v1/probe-monitors/": _blend_world,
}


def environment(monkeypatch: Any, data_dir: Path, world: Any) -> Any:
    """What every case runs inside, as a deployment with its siblings would answer: the fake Hub serving the world's
    repositories (`hub_spy.installed_fake_hub`, the deployment's token set), miServer's reads
    (`backstop_world.stub_siblings`), and two GPUs on the monitor. A context manager; the network stays blocked."""
    from src.api.v1.endpoints import system
    from tests.support import backstop_world, hub_spy

    backstop_world.stub_siblings(monkeypatch, world)

    class _Info:
        def __init__(self, i: int) -> None:
            self.i = i

        def to_dict(self) -> dict[str, Any]:
            return {"index": self.i, "name": f"card {self.i}"}

    class _Monitor:
        def is_available(self) -> bool:
            return True

        def get_device_count(self) -> int:
            return 2

        def get_all_gpu_info(self) -> list[Any]:
            return [_Info(0), _Info(1)]

    monkeypatch.setattr(system, "get_gpu_monitor_service", lambda: _Monitor())
    return hub_spy.installed_fake_hub(monkeypatch, data_dir, backstop_world.hub_repos())


async def prepare(ctx: Context, families_done: Optional[set] = None) -> None:
    """Run the family's setup, then the key's own (`Override.setup`). With `families_done` (a batch on one world), a
    family's setup runs once, before its first case."""
    template = ctx.case.key.split(" ", 1)[1]
    for prefix, setup in FAMILY_SETUPS.items():
        if template.startswith(prefix):
            if families_done is not None:
                if prefix in families_done:
                    continue
                families_done.add(prefix)
            await setup(ctx)
    override = OVERRIDES.get(ctx.case.key)
    if override is not None and override.setup is not None:
        await override.setup(ctx)


#: Path parameters whose name means a different world row under a route family (prefix → {name: world key}).
FAMILY_PATHS: dict[str, dict[str, str]] = {
    "/api/v1/probe-monitors/": {"dataset_id": "eval_view", "run_id": "probe_run", "report_id": "report",
                                "evaluation_id": "evaluation"},
    "/api/v1/steering/experiments/": {"experiment_id": "experiment"},
    "/api/v1/cluster-profiles/": {"profile_id": "profile"},
    "/api/v1/jlens/": {"slug": "jlens_slug"},
}

#: A sampled body or query names a world row by field name: the sampler's placeholder string ("x",
#: `gate_fixtures._model_sample`) under one of these names becomes the world's id. A value the world or an override
#: set is kept.
NAMED_ROWS = {"sae_id": "sae", "model_id": "model", "dataset_id": "dataset", "extraction_id": "extraction",
              "extraction_job_id": "extraction", "feature_id": "feature", "probe_id": "probe", "circuit_id": "circuit",
              "prompt_id": "prompt", "behaviour": "behaviour", "behaviour_key": "behaviour", "training_id": "training",
              "discovery_run_id": "discovery", "capture_run_id": "capture", "dataset_ids": "dataset",
              "sae_ids": "sae", "feature_ids": "feature", "view_ids": "eval_view", "view_id": "eval_view",
              "profile_id": "profile", "train_dataset_id": "train_view"}
PLACEHOLDER = "x"


def _name_rows(value: Any, world: Any) -> Any:
    if isinstance(value, list):
        return [_name_rows(v, world) for v in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, sub in value.items():
        row = NAMED_ROWS.get(key)
        if row and sub == PLACEHOLDER and row in world.ids:
            out[key] = world.ids[row]
        elif row and isinstance(sub, list) and sub == [PLACEHOLDER] and row in world.ids:
            out[key] = [world.ids[row]]
        else:
            out[key] = _name_rows(sub, world)
    return out


def _set(model: str, key: str, **values: Any) -> Callable[[Context], Any]:
    """A setup that puts the world row `world[key]` of `src.models.<model>` into a state (a cancel needs a running
    row; a delete needs a row nothing depends on)."""
    async def setup(ctx: Context) -> None:
        import importlib

        module, _, name = model.rpartition(".")
        cls = getattr(importlib.import_module(f"src.models.{module}"), name)
        ident = ctx.world[key]
        row = await ctx.session.get(cls, uuid_or(ident, cls))
        assert row is not None, (model, ident)
        for field_name, value in values.items():
            setattr(row, field_name, value() if callable(value) else value)
        await ctx.session.commit()
    return setup


def uuid_or(ident: str, cls: Any) -> Any:
    import uuid

    column = cls.__table__.c.id
    return uuid.UUID(ident) if getattr(column.type, "as_uuid", False) else ident


def _status(enum_path: str, name: str) -> Callable[[], Any]:
    def value() -> Any:
        import importlib

        module, _, cls = enum_path.rpartition(".")
        return getattr(getattr(importlib.import_module(f"src.models.{module}"), cls), name)
    return value


def _via(method: str, template: str, body: Optional[Callable[[Any], Any]] = None, keep: Optional[dict] = None,
         status: tuple = (200, 201, 202)) -> Callable[["Context"], Any]:
    """A setup that makes the row a case needs through the live API, as an operator would: `template`'s `{name}`
    parameters filled from `world` (by `FAMILY_PATHS`, then by name), `keep` = {world key: answer field} stored."""
    import re

    async def setup(ctx: "Context") -> None:
        family = next((p for prefix, p in FAMILY_PATHS.items() if template.startswith(prefix)), {})
        from tests.support.backstop_world import PATH_BY_NAME

        def fill(match: Any) -> str:
            name = match.group(1)
            key = family.get(name) or PATH_BY_NAME.get(name) or name
            return str(ctx.world[key])

        path = re.sub(r"\{([^}]+)\}", fill, template)
        kwargs = {"json": body(ctx.world)} if body is not None else {}
        response = await ctx.client.request(method, path, **kwargs)
        assert response.status_code in status, (method, path, response.status_code, response.text[:400])
        for world_key, answer_field in (keep or {}).items():
            ctx.world.ids[world_key] = response.json() if answer_field == "*" else response.json()[answer_field]
    return setup


def _chain(*steps: Callable[["Context"], Any]) -> Callable[["Context"], Any]:
    async def setup(ctx: "Context") -> None:
        for step in steps:
            await step(ctx)
    return setup


def _call(key: str, *, path: Optional[dict] = None, body: Optional[Callable[[Any], Any]] = None,
          keep: Optional[dict] = None, status: tuple = (200, 201, 202), once: Optional[str] = None
          ) -> Callable[["Context"], Any]:
    """A setup that calls the covered route `key` as its own operator case builds it (the world's body for it, its
    family's path names, its override), with `path` ({name: world key}) and `body` (replacing it) on top; `keep` =
    {world key: answer field, "*" for the whole answer} stored. `once`: skipped when that world key is already set
    (a batch's earlier case made the row)."""
    async def setup(ctx: "Context") -> None:
        if once is not None and once in ctx.world.ids:
            return
        from src.main import app

        probe = Case(key, "operator", 0)
        saved = OVERRIDES.get(key)
        if path:
            OVERRIDES[key] = Override(path={**(saved.path if saved else {}), **path},
                                      body=saved.body if saved else None,
                                      replace_body=saved.replace_body if saved else False,
                                      query=saved.query if saved else None)
        try:
            req = request_for(app, probe, ctx.world)
        finally:
            if path:
                if saved is None:
                    OVERRIDES.pop(key, None)
                else:
                    OVERRIDES[key] = saved
        payload = body(ctx.world) if body is not None else req["body"]
        kwargs: dict[str, Any] = {"params": req["query"]}
        if payload is not None:
            kwargs["json"] = payload
        response = await ctx.client.request(req["method"], req["path"], **kwargs)
        assert response.status_code in status, (key, req["path"], response.status_code, response.text[:400])
        for world_key, answer_field in (keep or {}).items():
            ctx.world.ids[world_key] = response.json() if answer_field == "*" else response.json()[answer_field]
    return setup


def _once(world_key: str, step: Callable[["Context"], Any]) -> Callable[["Context"], Any]:
    async def setup(ctx: "Context") -> None:
        if world_key not in ctx.world.ids:
            await step(ctx)
    return setup


async def _lone_model(ctx: "Context") -> None:
    """A model nothing references (a delete of it has nothing to refuse over)."""
    from src.models.model import Model, ModelStatus

    ctx.session.add(Model(id="m_crc_lone", name="crc lone", repo_id="fixture-org/lone", architecture="gpt2",
                          params_count=1, status=ModelStatus.READY, file_path=None))
    await ctx.session.commit()
    ctx.world.ids["model_lone"] = "m_crc_lone"


async def _grouping(ctx: "Context") -> None:
    """The world's extraction grouped by the REAL service (`FeatureGroupingService.compute`, as the worker runs it):
    two features whose top examples share a prime token, so one group forms. Then a second run left running (a
    cancel's subject)."""
    from starlette.concurrency import run_in_threadpool

    from src.core.database import SyncSessionLocal
    from src.models.feature_activation import FeatureActivation
    from src.models.feature_grouping import FeatureGroup, FeatureGroupingRun
    from src.services.feature_grouping_service import FeatureGroupingService

    for feature in ("feat_bkst_1", "feat_bkst_2"):
        for i in range(10):
            ctx.session.add(FeatureActivation(feature_id=feature, sample_index=100 + i, max_activation=2.0 + i,
                                              tokens=["the", "cat", "sat"], activations=[0.0, 2.0 + i, 0.1],
                                              prime_token="cat", prefix_tokens=["the", "big"],
                                              suffix_tokens=["sat", "down"]))
    await ctx.session.commit()

    def compute() -> tuple:
        with SyncSessionLocal() as db:
            run = FeatureGroupingService().compute(db, ctx.world["extraction"])
            group = db.query(FeatureGroup).filter(FeatureGroup.run_id == run.id).first()
            running = FeatureGroupingRun(extraction_id=ctx.world["extraction"], status="computing", params={},
                                         params_hash="0" * 64)
            db.add(running)
            db.commit()
            return run.id, group.id if group else None, running.id

    run_id, group_id, running_id = await run_in_threadpool(compute)
    assert group_id is not None, "the grouping service formed no group over the fixture's features"
    ctx.world.ids.update(grouping_run=run_id, group=group_id, grouping_running=running_id)


async def _trials(ctx: "Context") -> None:
    """Two completed trials over one panel (as `label_features_trial_task` stores them: the panel and each feature's
    result), so the compare reads two comparable runs."""
    from src.models.labeling_trial_run import LabelingTrialRun

    panel = "pnl_" + "0" * 64
    for name, label in (("a", "cats"), ("b", "felines")):
        ctx.session.add(LabelingTrialRun(
            id=f"ltr_crc_{name}", panel_id=panel, extraction_job_id=ctx.world["extraction"], name=f"crc {name}",
            status="completed", payload={"panel": {"panel_id": panel, "feature_ids": [ctx.world["feature"]]},
                                         "results": [{"feature_id": ctx.world["feature"], "status": "ok",
                                                      "specific": label, "category": "animals"}]}))
    await ctx.session.commit()
    ctx.world.ids.update(trial_a="ltr_crc_a", trial_b="ltr_crc_b")


async def _rubrics(ctx: "Context") -> None:
    """miDataworks connected, answering `GET /api/v1/rubrics` as its route does (a list of `RubricOut`)."""
    from src.core.config import settings
    from src.services import judge_prompt_library as library

    ctx.monkeypatch.setattr(settings, "midataworks_api_url", "http://midataworks.invalid", raising=False)
    rubric = {"id": "rub_crc", "name": "pii-check", "version": 3, "ref": "pii-check@3", "content_hash": "0" * 64,
              "style": "binary", "body": {"style": "binary", "input_fields": ["passage"],
                                          "allowed_verdicts": ["yes", "no"]},
              "behaviour_key": None, "used": False, "created_by": "operator", "created_by_origin": "ui",
              "created_at": "2026-11-27T00:00:00Z"}

    async def fake_get(path: str) -> Any:
        assert path == "/api/v1/rubrics", path
        return [rubric]

    ctx.monkeypatch.setattr(library, "_midataworks_get", fake_get)


async def _heldout_evaluation(ctx: "Context") -> None:
    """The world probe evaluated on its held-out view with every row recorded (positions, labels, scores), as an
    evaluation leaves it: what a WSP capture plans from."""
    import numpy as np

    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun
    from tests.support.detector_fixtures import Arrays, evaluation_row

    probe = await ctx.session.get(ProbeMonitor, ctx.world["probe"])
    run = await ctx.session.get(ProbeMonitorRun, probe.run_id)
    view = await ctx.session.get(ProbeMonitorDataset, ctx.world["eval_view"])
    labels = np.array([1, 0, 1, 0] * 6, dtype=np.int8)
    arrays = Arrays(scores=np.linspace(-1, 1, labels.size).astype(np.float32),
                    positions=np.arange(labels.size, dtype=np.int64),
                    n_scored=np.full(labels.size, 4, dtype=np.int32), labels=labels)
    await evaluation_row(ctx.session, probe, run, view, arrays)
    await ctx.session.commit()


async def _code_rule(ctx: "Context") -> None:
    """A code rule document and the stub runner its cases run on (`code_rule_world`, `stub_code_runner`)."""
    from src.rule_runner import contract
    from tests.support import code_rule_world as crw
    from tests.support import stub_code_runner as stub

    stub.install(ctx.monkeypatch)
    _rules, doc, _sid, _expr = crw.code_set("crc-code-rule", ctx.world["behaviour"])
    ctx.world.ids.update(code_doc=doc, code_doc_sha=contract.code_sha256(doc))


async def _mask_recorded(ctx: "Context") -> None:
    """The case's evaluation with its truncation mask already written (no row truncated), as a G5 evaluator or an
    earlier backfill leaves it: the backfill answers `already_recorded`. (Proving a mask needs the run's model on
    disk; `test_g5_backfill.py` covers that path and its refusals.)"""
    import numpy as np

    from src.models.probe_monitor import ProbeMonitorEvaluation
    from src.services.probe_monitor_run import _truncated_path_beside

    evaluation = await ctx.session.get(ProbeMonitorEvaluation, ctx.world["evaluation"])
    mask = _truncated_path_beside(evaluation.scores_path)
    np.save(mask, np.zeros(np.load(evaluation.positions_path).size, dtype=bool))
    evaluation.truncated_path = str(mask)
    await ctx.session.commit()


async def _wsp_candidate(ctx: "Context") -> None:
    """A miForge candidate the import accepts, made as the WSP tests make one (`wsp_pipeline`, `wsp_candidate`) but
    in the app's own database: a REAL tiny probe run (practice corpus, held-out and calibration views), its
    held-out captures, a built bundle, a one-entry sparse patch over its head, and the scorer's record of scoring
    that head for the run (P23). `world["wsp_candidate"]` is the request body."""
    from starlette.concurrency import run_in_threadpool

    def build() -> dict:
        import numpy as np
        from datasets import Dataset as HFDataset

        from src.core.database import SyncSessionLocal
        from src.services.wsp_bundle import build_bundle
        from tests.support import wsp_candidate, wsp_pipeline
        from tests.unit import test_probe_monitor_pipeline as pipeline

        lm, tokenizer = _tiny_lm(pipeline), _tiny_tokenizer()
        rng = np.random.default_rng(7)
        inputs, labels, pairs = [], [], []
        for i in range(pipeline.N_ROWS):  # `arrow_dataset`'s rows
            marker = "risk" if i % 2 == 0 else "calm"
            inputs.append(" ".join([f"{marker}{rng.integers(0, 8)}" for _ in range(3)]
                                   + [f"filler{rng.integers(0, 16)}" for _ in range(2)]))
            labels.append("high" if i % 2 == 0 else "low")
            pairs.append(f"pair{i // 2}")
        practice = ctx.data_dir / "wsp_practice"
        HFDataset.from_dict({"inputs": inputs, "stakes": labels, "pair": pairs}).save_to_disk(str(practice))
        heldout = wsp_pipeline.write_heldout(ctx.data_dir / "wsp_heldout", wsp_pipeline.HELDOUT_ROWS, 11)
        with SyncSessionLocal() as db:
            world = wsp_pipeline.make_world(db, practice, heldout)
            probe = wsp_pipeline.run_probe(db, world, lm, tokenizer)
            wsp_pipeline.capture_all(db, probe, lm, tokenizer)
            bundle = wsp_pipeline.new_bundle(db, probe)
            build_bundle(db, bundle, tokenizer=tokenizer, max_seconds=120)
            db.refresh(bundle)
            assert bundle.status == "ready", bundle.status
            candidate = wsp_candidate.make_candidate(bundle, entries=[0])
            wsp_candidate.score_candidate(db, bundle, candidate)
            return candidate.body()

    ctx.world.ids["wsp_candidate"] = await run_in_threadpool(build)


def _tiny_lm(pipeline: Any) -> Any:
    """`test_probe_monitor_pipeline.tiny_lm`'s model (a fixture there)."""
    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=pipeline.D_MODEL, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def _tiny_tokenizer() -> Any:
    """`test_probe_monitor_pipeline.tiny_tokenizer`'s tokenizer (a fixture there): the class markers, filler, and a
    chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<turn>", "</turn>", "user", "assistant"]
    words += [f"risk{i}" for i in range(8)] + [f"calm{i}" for i in range(8)] + [f"filler{i}" for i in range(16)]
    backend = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for m in messages %}<turn> {{ m['role'] }} {{ m['content'] }} </turn> "
        "{% endfor %}{% if add_generation_prompt %}<turn> assistant {% endif %}"
    )
    tokenizer.pad_token = "<unk>"
    return tokenizer


async def _mark(ctx: "Context", world_key: str) -> None:
    ctx.world.ids[world_key] = "done"


async def _jlens_task(ctx: "Context") -> None:
    from src.models.task_queue import TaskQueue

    ctx.session.add(TaskQueue(id="tq_crc_jlens", task_id="jl_crc_task", task_type="jlens_fit",
                              entity_id=ctx.world["model"], entity_type="model", status="queued"))
    await ctx.session.commit()
    ctx.world.ids["jlens_task"] = "jl_crc_task"


def _probe_definition(name: str, rung: int) -> dict[str, Any]:
    """A `mistudio.probe-definition/v1` document as a build writes one (046 P7: `GET /probes/{id}/definition` answers
    the file unchanged, documented as a `$ref` to `docs/schemas/probe-definition-v1.json`, which the answer middleware
    checks): every field the schema declares filled, with the family's `name` and the member's rung."""
    from tests.support.answer_samples import sample

    schema = json.loads((Path(__file__).resolve().parents[3] / "docs" / "schemas" / "probe-definition-v1.json")
                        .read_text(encoding="utf-8"))
    document = sample(schema, schema)  # every format and pattern the schema states, filled (`answer_samples`)
    document["name"] = name
    document["evidence"]["rung"] = rung
    return document


def _member() -> dict[str, Any]:
    return {"layer": 0, "member_kind": "feature_ref", "feature": {"feature_idx": 0, "strength": 1.0}}


#: key → its override (filled as the 9.2 modules need them; each entry says why).
OVERRIDES: dict[str, Override] = {
    # A circuit needs an SAE per layer and a member that validates (`CircuitDefinitionV1`).
    "POST /api/v1/circuits": Override(body=lambda w: {"name": "bkst circuit 2", "saes": [{"layer": 0, "sae_id": w["sae"]}],
                                                     "members": [_member()]}, replace_body=True),
    # The steering routes name a real SAE layer (the world's SAE is at layer 0).
    "POST /api/v1/steering/async/compare": Override(body=lambda w: {
        "sae_id": w["sae"], "prompt": "The cat", "selected_features": [{"feature_idx": 0, "layer": 0, "strength": 1.0}]}),
    "POST /api/v1/steering/cluster-allocation": Override(body=lambda w: {
        "sae_id": w["sae"], "members": [{"feature_idx": 0, "layer": 0}]}),
    # Cancels need a running row; a delete needs a row nothing depends on.
    "DELETE /api/v1/datasets/{dataset_id}/cancel": Override(setup=_set(
        "dataset.Dataset", "dataset", status=_status("dataset.DatasetStatus", "DOWNLOADING"))),
    "DELETE /api/v1/models/{model_id}": Override(path={"model_id": "model_lone"}, setup=_lone_model),
    "DELETE /api/v1/models/{model_id}/cancel": Override(path={"model_id": "model_nofiles"}, setup=_set(
        "model.Model", "model_nofiles", status=_status("model.ModelStatus", "DOWNLOADING"))),
    "POST /api/v1/circuit-capture/{run_id}/cancel": Override(setup=_set(
        "circuit_runs.CircuitCaptureRun", "capture", status="running")),
    "POST /api/v1/circuit-discovery": Override(setup=_set(
        "circuit_runs.CircuitCaptureRun", "capture", status="completed")),
    "POST /api/v1/circuit-discovery/{run_id}/cancel": Override(setup=_set(
        "circuit_runs.CircuitDiscoveryRun", "discovery", status="running")),
    "POST /api/v1/circuit-discovery/{run_id}/attribution/cancel": Override(setup=_set(
        "circuit_runs.CircuitDiscoveryRun", "discovery", attribution_status="running")),
    "POST /api/v1/circuit-discovery/{run_id}/validate/cancel": Override(setup=_set(
        "circuit_runs.CircuitDiscoveryRun", "discovery", validation_status="running")),
    "POST /api/v1/circuits/{circuit_id}/calibration/cancel": Override(setup=_set(
        "circuit.Circuit", "circuit", calibration_status="running")),
    "POST /api/v1/circuits/{circuit_id}/faithfulness/cancel": Override(setup=_set(
        "circuit.Circuit", "circuit", faithfulness_status="running")),
    "POST /api/v1/circuits/steering-samples/{run_id}/cancel": Override(
        path={"run_id": "record_run"}, setup=lambda ctx: _record_run(ctx)),
    # A saved experiment carries the comparison it saves (comparisons are ephemeral).
    "POST /api/v1/steering/experiments": Override(body=lambda w: {"result": {"comparison_id": "cmp_bkst", "outputs": []}}),
    "POST /api/v1/steering/async/sweep": Override(body=lambda w: {"sae_id": w["sae"], "feature_idx": 0, "layer": 0}),
    "POST /api/v1/steering/async/combined": Override(body=lambda w: {
        "sae_id": w["sae"], "selected_features": [{"feature_idx": 0, "layer": 0, "strength": 1.0}]}),
    # The behaviour family (blend world): a link before its unlink; a mark before the suggested blend is read.
    "DELETE /api/v1/probe-monitors/behaviours/{key}/views/{dataset_id}": Override(
        setup=_via("POST", "/api/v1/probe-monitors/behaviours/{key}/views/{dataset_id}")),
    "GET /api/v1/probe-monitors/behaviours/{key}/suggested-blend": Override(
        query=lambda w: {"model": w["model_name"]},
        setup=_via("PUT", "/api/v1/probe-monitors/behaviours/{key}/suggested-blend",
                   body=lambda w: {"model": w["model_name"], "report_id": w["report"], "candidate": "max"})),
    "PUT /api/v1/probe-monitors/behaviours/{key}/suggested-blend": Override(
        body=lambda w: {"model": w["model_name"], "report_id": w["report"], "candidate": "max"}, replace_body=True),
    "GET /api/v1/probe-monitors/blend-reports/{report_id}/scores": Override(
        query=lambda w: {"candidate": "max", "view_id": w["blend_view"]}),
    "POST /api/v1/probe-monitors/behaviours": Override(body=lambda w: {"key": "crc-new-behaviour"}),
    # The blend world made version 2 of its behaviour (`blend_world.build`); a new version names it as its base.
    "POST /api/v1/probe-monitors/behaviours/{key}/versions": Override(body=lambda w: {"base_version": 2,
                                                                                     "target_fpr": 0.02}),
    "POST /api/v1/probe-monitors/behaviours/{key}/behaviour-verdict/export": Override(
        body=lambda w: {"model": w["model_name"], "report_id": w["report"], "candidate": "max"}),
    "POST /api/v1/probe-monitors/blend-reports": Override(body=lambda w: {
        "behaviour_key": w["behaviour"], "model": w["model_name"], "dataset_version": "dv-blend",
        "calibration_sample_rows": 2000}, replace_body=True),
    # A judge prompt's own export, imported back under another name; a binary rubric (miDataworks' format).
    "POST /api/v1/probe-monitors/judge-prompts/import": Override(
        setup=_via("GET", "/api/v1/probe-monitors/judge-prompts/{prompt_id}/export", keep={"prompt_doc": "*"}),
        body=lambda w: {"document": w["prompt_doc"]}, replace_body=True),
    "POST /api/v1/probe-monitors/judge-prompts/import-rubric": Override(body=lambda w: {
        "title": "PII (rubric)", "field_map": {"passage": "text"}, "literals": {"question": "Is a person named?"},
        "positive_verdicts": ["yes"], "document": {
        "format": "midataworks.rubric/v1", "name": "pii-check", "version": 3, "body": {
            "style": "binary", "input_fields": ["passage"], "parser": "verdict_line_v1", "allowed_verdicts": ["yes", "no"],
            "messages": [{"role": "system", "content": "Does the text name a person? End with VERDICT: yes or VERDICT: no."},
                         {"role": "user", "content": "Question: {question}\nText: {passage}"}]}}}),
    # Reads of a rules evaluation, a success report and a spot-check name one the case makes first.
    "GET /api/v1/probe-monitors/rules-evaluations/{evaluation_id}": Override(
        path={"evaluation_id": "rules_evaluation"},
        setup=_via("POST", "/api/v1/probe-monitors/probes/{probe_id}/rules-evaluations",
                   body=lambda w: {"view_ids": [w["blend_view"]], "calibration": False},
                   keep={"rules_evaluation": "evaluation_id"})),
    "GET /api/v1/probe-monitors/success-reports/{report_id}": Override(
        path={"report_id": "success_report"},
        setup=_via("POST", "/api/v1/probe-monitors/behaviours/{key}/success-reports", body=lambda w: {},
                   keep={"success_report": "id"})),
    "POST /api/v1/probe-monitors/probes/{probe_id}/rules-evaluations": Override(
        body=lambda w: {"view_ids": [w["blend_view"]], "calibration": False}),
    # The grouping index (the real service over the world's features), a group, and a run left computing.
    "GET /api/v1/extractions/{extraction_id}/feature-groups/{group_id}": Override(
        path={"group_id": "group"}, setup=_once("group", _grouping)),
    "GET /api/v1/extractions/{extraction_id}/features/by-token": Override(
        query=lambda w: {"token": "cat"}, setup=_once("group", _grouping)),
    "POST /api/v1/feature-groups/runs/{run_id}/cancel": Override(path={"run_id": "grouping_running"}, setup=_grouping),
    # Labeling: a sweep the operator started; two completed trials over one panel.
    "GET /api/v1/labeling/resume-sweeps/{sweep_id}": Override(path={"sweep_id": "sweep"}, setup=_call(
        "POST /api/v1/labeling/{extraction_job_id}/resume-sweep", keep={"sweep": "id"}, once="sweep")),
    "POST /api/v1/labeling/resume-sweeps/{sweep_id}/cancel": Override(path={"sweep_id": "sweep"}, setup=_call(
        "POST /api/v1/labeling/{extraction_job_id}/resume-sweep", keep={"sweep": "id"})),
    "GET /api/v1/labeling/trials/{trial_run_id}": Override(path={"trial_run_id": "trial_a"},
                                                           setup=_once("trial_a", _trials)),
    "GET /api/v1/labeling/trials/compare/{run_a}/{run_b}": Override(path={"run_a": "trial_a", "run_b": "trial_b"},
                                                                    setup=_once("trial_a", _trials)),
    # Approvals: one filed by the operator's call; the consumed one approved first, then spent with its payload.
    "GET /api/v1/mcp/approvals/{request_id}": Override(path={"request_id": "approval"}, setup=_call(
        "POST /api/v1/mcp/approvals", keep={"approval": "id"}, once="approval")),
    "POST /api/v1/mcp/approvals/{request_id}/consume": Override(
        path={"request_id": "approval"},
        body=lambda w: {"tool_name": "start_probe_training_run", "payload": {"model_id": w["model"]}}, replace_body=True,
        setup=_chain(_call("POST /api/v1/mcp/approvals", keep={"approval": "id"}),
                     _call("POST /api/v1/mcp/approvals/{request_id}/approve", path={"request_id": "approval"}))),
    # Operator code: a document whose cases the stub runner runs.
    "POST /api/v1/code-rules/{code_sha256}/approval-requests": Override(
        path={"code_sha256": "code_doc_sha"}, setup=_code_rule, replace_body=True,
        body=lambda w: {"document": w["code_doc"], "cases": [__import__(
            "tests.support.code_rule_world", fromlist=["case_for"]).case_for(w["code_doc"])]}),
    # miDataworks connected (its rubric list), a library prompt's export imported back, a rubric mapped.
    "GET /api/v1/probe-monitors/judge-prompts/midataworks/rubrics": Override(setup=_rubrics),
    "POST /api/v1/probe-monitors/judge-prompts/{prompt_id}/versions": Override(
        body=lambda w: {"system_prompt": 'Did the assistant refuse here? Reply {"answer": "yes" or "no"}.'}),
    "POST /api/v1/probe-monitors/judge-runs/{judge_run_id}/cancel": Override(
        path={"judge_run_id": "judge_run"},
        setup=_call("POST /api/v1/probe-monitors/judge-runs", keep={"judge_run": "id"})),
    # Probe-monitor rows in the states a read or a cancel needs.
    "GET /api/v1/probe-monitors/probes/{probe_id}/evaluations/{evaluation_id}/breakdown": Override(
        query=lambda w: {"by_column": "channel"},
        setup=_call("PATCH /api/v1/probe-monitors/datasets/{dataset_id}/columns", path={"dataset_id": "blend_view"},
                    body=lambda w: {"group_columns": ["channel"]}, once="columns_declared")),
    "POST /api/v1/probe-monitors/rules-evaluations/{evaluation_id}/cancel": Override(
        path={"evaluation_id": "rules_evaluation"},
        setup=_call("POST /api/v1/probe-monitors/probes/{probe_id}/rules-evaluations",
                    keep={"rules_evaluation": "evaluation_id"})),
    "POST /api/v1/probe-monitors/probes/{probe_id}/evaluations/{evaluation_id}/backfill-truncation": Override(
        setup=_mask_recorded),
    "POST /api/v1/probe-monitors/runs/{run_id}/cancel": Override(setup=_set(
        "probe_monitor.ProbeMonitorRun", "probe_run", status="running")),
    "GET /api/v1/probe-monitors/spot-checks/{check_id}": Override(path={"check_id": "spot_check"}, setup=_call(
        "POST /api/v1/probe-monitors/spot-checks", keep={"spot_check": "id"}, once="spot_check")),
    "POST /api/v1/probe-monitors/spot-checks/{check_id}/cancel": Override(path={"check_id": "spot_check"}, setup=_call(
        "POST /api/v1/probe-monitors/spot-checks", keep={"spot_check": "id"})),
    "PUT /api/v1/probe-monitors/behaviours/{key}/probes/{probe_id}/role": Override(setup=_call(
        "POST /api/v1/probe-monitors/datasets/{dataset_id}/confirm-positive", path={"dataset_id": "blend_train_view"},
        body=lambda w: {"key": w["behaviour"]})),
    # WSP: the world probe's held-out evaluation, every row recorded.
    "GET /api/v1/wsp/captures/preflight": Override(setup=_once("heldout", _chain(
        _heldout_evaluation, lambda ctx: _mark(ctx, "heldout")))),
    "POST /api/v1/wsp/captures": Override(setup=_heldout_evaluation),
    "POST /api/v1/wsp/candidates/import": Override(setup=_once("wsp_candidate", _wsp_candidate),
                                                   body=lambda w: w["wsp_candidate"], replace_body=True),
    # A circuit's own export imported back.
    "POST /api/v1/circuits/import": Override(
        setup=_call("GET /api/v1/circuits/{circuit_id}/export", keep={"circuit_doc": "*"}),
        body=lambda w: w["circuit_doc"], replace_body=True),
    # J-lens: a known operation, the world's artifact by slug, a watchlist with one concept.
    "POST /api/v1/jlens/documents/validate": Override(
        setup=_call("GET /api/v1/jlens/artifacts/{slug}/document", keep={"jlens_doc": "*"}),
        body=lambda w: w["jlens_doc"], replace_body=True),
    "POST /api/v1/jlens/tasks/{task_id}/cancel": Override(path={"task_id": "jlens_task"}, setup=_jlens_task),
    "POST /api/v1/jlens/annotate": Override(body=lambda w: {"model_id": w["model"], "feature_id": w["feature"],
                                                            "layer": 0, "direction": [0.1] * TINY_D_MODEL}),
    "POST /api/v1/jlens/attention-readout": Override(body=lambda w: {"model_id": w["model"], "layer": 1}),
    "GET /api/v1/jlens/cost-estimate": Override(query=lambda w: {"operation": "readout"}),
    "GET /api/v1/jlens/reports/replication": Override(query=lambda w: {"slug": w["jlens_slug"]}),
    "POST /api/v1/jlens/watchlists": Override(body=lambda w: {
        "artifact_ref": w["jlens_slug"], "concepts": [{"token": " Paris", "threshold": 0.5}]}),
}


@functools.lru_cache(maxsize=1)
def _contract() -> dict[str, Any]:
    from src.api.answers_contract import DEFAULT_PATH

    return json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))


def covered_keys() -> frozenset[str]:
    found = _contract()
    return frozenset(found["responses"]) | frozenset(found["excluded"])


@functools.lru_cache(maxsize=1)
def endpoint_files() -> dict[str, str]:
    """contract key → the file defining its endpoint, relative to `src/` (`api/v1/endpoints/circuits.py`, `main.py`)."""
    from src.api.answers_contract import operations
    from src.main import app

    out = {}
    for method, path, route in operations(app):
        source = inspect.getsourcefile(inspect.unwrap(route.endpoint)) or ""
        out[f"{method} {path}"] = source.split("/src/", 1)[-1]
    return out


def covered_for_router(router: str) -> list[str]:
    """The covered keys whose endpoint is defined in `router` (a basename: `circuits.py`, or `main.py`)."""
    files = endpoint_files()
    return sorted(k for k in covered_keys() if Path(files[k]).name == router)


def routers() -> dict[str, list[str]]:
    """router basename → its covered keys (every covered key appears under exactly one)."""
    files = endpoint_files()
    out: dict[str, list[str]] = {}
    for key in sorted(covered_keys()):
        out.setdefault(Path(files[key]).name, []).append(key)
    return out


def gated_keys() -> frozenset[str]:
    """`POST <template>` of every gated route (the design table, less its deferred rows)."""
    from tests.support.gate_fixtures import DEFERRED, DESIGN

    return frozenset(f"POST {path}" for row, (path, _action) in DESIGN.items() if row not in DEFERRED)


@functools.lru_cache(maxsize=1)
def _routes() -> dict[str, Any]:
    from src.api.answers_contract import operations
    from src.main import app

    return {f"{method} {path}": route for method, path, route in operations(app)}


def own_status(key: str) -> int:
    """The route's own success status: its decorator's `status_code` (FastAPI answers 200 when it sets none)."""
    return _routes()[key].status_code or 200


def cases_for(key: str) -> list[Case]:
    override = OVERRIDES.get(key, Override())
    own = override.expect or own_status(key)
    if key in gated_keys():
        return [Case(key, "agent", 202), Case(key, "operator", own)]
    return [Case(key, "operator", own)]


def cases(keys: list[str]) -> list[Case]:
    return [case for key in keys for case in cases_for(key)]


def request_for(app: Any, case: Case, world: Any) -> dict[str, Any]:
    """`{"method", "path", "query", "body", "headers"}` for one case, naming world rows."""
    from tests.support import backstop_world

    method, template = case.key.split(" ", 1)
    override = OVERRIDES.get(case.key, Override())
    family = next((paths for prefix, paths in FAMILY_PATHS.items() if template.startswith(prefix)), {})
    saved = backstop_world.PATH_OVERRIDES.get(template)
    paths = {**family, **(saved or {}), **override.path}  # the case's own, then the world's per-route, then the family
    if paths:
        backstop_world.PATH_OVERRIDES[template] = paths
    try:
        built = backstop_world.world_request(app, method, template, world)
    finally:
        if paths:
            if saved is None:
                backstop_world.PATH_OVERRIDES.pop(template, None)
            else:
                backstop_world.PATH_OVERRIDES[template] = saved
    body = _name_rows(built["body"], world)
    if override.body is not None:
        patch = override.body(world)
        body = patch if override.replace_body or not isinstance(patch, dict) or not isinstance(body, dict) else (
            backstop_world._merge(body, patch))  # noqa: SLF001
    query = _name_rows(dict(built["query"]), world)
    if override.query is not None:
        query.update(override.query(world))
    headers = dict(AGENT_HEADERS) if case.caller == "agent" else {}
    return {"method": method, "path": built["path"], "query": query, "body": body, "headers": headers}


#: What the patched judge answers: a score a 0/1 judge prompt parses.
JUDGE_VERDICT = "1"


class _Handle:
    """What a queued task hands back (`AsyncResult`'s fields a route reads)."""

    def __init__(self, label: str, n: int) -> None:
        self.id = self.task_id = f"task-{label.split(':')[-1].rsplit('.', 1)[-1]}-{n}"
        self.status = self.state = "PENDING"

    def __str__(self) -> str:
        return self.id


def patch_dispatch(monkeypatch) -> Any:
    """Every primitive of `tests/support/backstop.py` (the GPU starts at `route_walk`'s `GPU_PRIMITIVES` and
    `ENQUEUE_METHODS`, the outside paid calls, the transports, `subprocess`, `os.kill`) installed, then the queueing
    ones made to ANSWER as a running worker would: the task is recorded in `reached` and a task handle returned, so a
    route that queues work answers its own status and nothing runs. The network stays blocked."""
    from celery import Celery
    from celery.app.task import Task
    from fastapi import BackgroundTasks

    from tests.support import backstop
    from tests.support.route_walk import ENQUEUE_METHODS, GPU_PRIMITIVES

    backstop.blocked_network(monkeypatch)
    rec = backstop.install_primitives(monkeypatch)

    def handle(label: str) -> _Handle:
        rec.reached.append(label)
        return _Handle(label, len(rec.reached))

    def task_answer(kind: str) -> Callable[..., Any]:
        def _answer(self: Any, *args: Any, **kwargs: Any) -> Any:
            return handle(f"{kind}:{getattr(self, 'name', '?')}")
        return _answer

    for method in ENQUEUE_METHODS:
        monkeypatch.setattr(Task, method, task_answer(f"Task.{method}"))
    monkeypatch.setattr(Celery, "send_task", lambda self, name, *a, **k: handle(f"Celery.send_task:{name}"))
    monkeypatch.setattr(BackgroundTasks, "add_task", lambda self, func, *a, **k: rec.reached.append(
        f"BackgroundTasks.add_task:{getattr(func, '__name__', func)}"))

    def dispatch_gpu_task(task: Any, *args: Any, **kwargs: Any) -> Any:
        return handle(f"dispatch_gpu_task:{getattr(task, 'name', '?')}")

    def gpu_delay(task: Any, *args: Any, **kwargs: Any) -> Any:
        return lambda *a, **k: handle(f"gpu_delay:{getattr(task, 'name', '?')}")

    async def ensure_steering_worker(card: Any, single_worker_ensure: Any) -> tuple:
        rec.reached.append(f"ensure_steering_worker:{card}")
        return True, 4242  # a worker is up (pid 4242); the task it is sent is queued through `apply_async`

    def ensure_model_loaded(endpoint_url: str, model_name: str) -> None:
        rec.reached.append(f"ensure_model_loaded:{model_name}")

    for name, fn in (("dispatch_gpu_task", dispatch_gpu_task), ("gpu_delay", gpu_delay),
                     ("ensure_steering_worker", ensure_steering_worker), ("ensure_model_loaded", ensure_model_loaded)):
        assert name in GPU_PRIMITIVES
        backstop._replace_everywhere(monkeypatch, backstop.GPU_FUNCTIONS[name], name, fn)  # noqa: SLF001
    # The outside paid calls answer as a local judge would: a client handle, and one verdict per call.
    def client_for(endpoint: str, api_key: Any = None, http_client: Any = None) -> Any:
        rec.reached.append(f"_client_for:{endpoint}")
        return object()

    def ask_judge(client: Any, model: str, spec: Any, messages: Any) -> str:
        rec.reached.append(f"ask_judge:{model}")
        return JUDGE_VERDICT

    for name, fn in (("_client_for", client_for), ("ask_judge", ask_judge)):
        backstop._replace_everywhere(monkeypatch, backstop.OUTSIDE_FUNCTIONS[name], name, fn)  # noqa: SLF001
    # `logit_lens_lease` stays a raising primitive: the logit lens runs the model in the request, so its covered case
    # is a refusal or a cached answer, never a model load (see OVERRIDES).
    return rec


async def run(client: Any, app: Any, case: Case, world: Any) -> Any:
    """Send one case through `client` (an ASGI client on the live app) and return the response."""
    req = request_for(app, case, world)
    kwargs: dict[str, Any] = {"params": req["query"], "headers": req["headers"]}
    if req["body"] is not None:
        kwargs["json"] = req["body"]
    return await client.request(req["method"], req["path"], **kwargs)


@dataclass
class Result:
    """One case's outcome: the status it got (or "SETUP" / "EXC" when its setup or the call raised), the start of the
    body, the answer middleware's failures recorded during the call, and the dispatch it reached."""
    case: Case
    got: Any
    text: str
    contract: list[str]
    reached: list[str]
    seconds: float
    body: bytes = b""     # the whole answer, for a check that does not go through the middleware (`assert_case`)
    media: Optional[str] = None
    checked: int = 0      # how many answers of the case's key the middleware checked during the call

    @property
    def ok(self) -> bool:
        return self.got == self.case.expect and not self.contract

    def describe(self) -> str:
        why = f"; off contract: {self.contract[0][:300]}" if self.contract else ""
        return f"{self.case.id}: expected {self.case.expect}, got {self.got}{why} :: {self.text[:300]}"


def is_read(case: Case) -> bool:
    """A case that may share a world with the router's other reads: a GET, and not one whose setup writes rows a
    later read would see (none do yet; `Override.setup` on a GET must leave the world as other reads expect it)."""
    return case.key.startswith("GET ")


def batches(keys: list[str]) -> list[list[Case]]:
    """`keys`' cases grouped for `run_cases`: every read of the list on one world (in key order), then each other
    case on its own world (a write changes rows the next case reads)."""
    all_cases = cases(keys)
    reads = [c for c in all_cases if is_read(c)]
    return ([reads] if reads else []) + [[c] for c in all_cases if not is_read(c)]


async def run_cases(cases_: list[Case], session: Any, monkeypatch: Any, data_dir: Path) -> list[Result]:
    """Run `cases_` in order on ONE world (`backstop_world.build_world` plus `extend_world`, the dispatch patched, the
    deployment's environment entered): each family's setup once, each key's own setup before its call. Returns one
    `Result` per case; an exception in a setup or a call is the case's result, never the batch's end. The answer
    middleware's failures during a call are moved into its `Result` (the autouse check sees none of them)."""
    import time
    import traceback

    import httpx

    from src.core.config import settings
    from src.main import app
    from tests.support import backstop, backstop_world
    from tests.support.answer_middleware import CHECKED, FAILURES

    monkeypatch.setattr(settings, "data_dir", data_dir)
    world = await backstop_world.build_world(session, data_dir, app)
    await extend_world(session, world)
    rec = patch_dispatch(monkeypatch)
    results: list[Result] = []
    families: set = set()
    with environment(monkeypatch, data_dir, world):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            for case in cases_:
                start, before, reached = time.monotonic(), len(FAILURES), len(rec.reached)
                try:
                    await prepare(Context(case, world, client, session, monkeypatch, data_dir), families)
                except Exception as exc:  # noqa: BLE001 - the case's result
                    results.append(Result(case, "SETUP", "".join(traceback.format_exception(exc))[-1500:],
                                          [f"setup: {f}" for f in FAILURES[before:]], [], time.monotonic() - start))
                    del FAILURES[before:]
                    continue
                setup_failures = [f"setup: {f}" for f in FAILURES[before:]]  # a setup's own call off its contract
                del FAILURES[before:]
                reached, checked = len(rec.reached), CHECKED.get(case.key, 0)
                try:
                    req = request_for(app, case, world)
                    response = await backstop.call(app, req["method"], req["path"], query=req["query"],
                                                   body=req["body"], headers=req["headers"])
                    results.append(Result(case, response.status_code, response.text[:1500],
                                          setup_failures + [str(f) for f in FAILURES[before:]], rec.reached[reached:],
                                          time.monotonic() - start, response.content,
                                          (response.headers.get("content-type") or "").split(";")[0] or None,
                                          CHECKED.get(case.key, 0) - checked))
                except Exception as exc:  # noqa: BLE001 - the case's result
                    results.append(Result(case, "EXC", "".join(traceback.format_exception(exc))[-1500:], [], [],
                                          time.monotonic() - start))
                finally:
                    del FAILURES[before:]
    return results


# ── the per-router modules (046 FTASKS 9.2, 9.3; FR-046.17 as amended by WT8) ──────────────────────────────────────


@functools.lru_cache(maxsize=1)
def walker_covered_keys() -> frozenset[str]:
    """The covered set derived WITHOUT the committed contract: the routes the live registry's tools call (the one
    route walker) plus the sibling apps' routes (`answers_contract.covered_keys`). The per-router and union tests
    compare their cases with this, so a contract file that lost a section cannot shrink both sides at once (046-2
    review round 1, finding 5)."""
    from src.api.answers_contract import covered_keys as walked, live_tool_names
    from src.main import app

    return frozenset(walked(live_tool_names(), app))


def registry_subset(router: str) -> frozenset[str]:
    """The covered keys whose endpoint's MODULE is `router` (a basename), read from the live app's registry by the
    endpoint's `__module__`: a second derivation, independent of `covered_for_router`'s source-file map and of the
    contract file, that a per-router module's cases must equal."""
    from src.api.answers_contract import operations
    from src.main import app

    covered = walker_covered_keys()
    return frozenset(f"{method} {path}" for method, path, route in operations(app)
                     if f"{method} {path}" in covered
                     and inspect.unwrap(route.endpoint).__module__.rsplit(".", 1)[-1] + ".py" == router)


def assert_router_set(router: str, cases_: list[Case]) -> None:
    """A per-router module's cases EQUAL its router's covered subset (FR-046.17 AC2, NC1): each key once as the
    operator, and a gated key also as an agent outside `MCP_POD_AGENTS` (expecting the gate's 202)."""
    keys = {c.key for c in cases_}
    want = registry_subset(router)
    assert keys == want, (f"{router}: cases for keys outside its covered subset {sorted(keys - want)}; "
                          f"covered keys with no case {sorted(want - keys)}")
    gated = gated_keys()
    for key in sorted(keys):
        callers = sorted((c.caller, c.expect) for c in cases_ if c.key == key)
        if key in gated:
            assert [c for c, _ in callers] == ["agent", "operator"] and dict(callers)["agent"] == 202, (key, callers)
        else:
            assert [c for c, _ in callers] == ["operator"], (key, callers)


#: (046-2 review round 1, finding 8) The empty lists each covered operator case's answer carries, by key: a list the
#: world leaves empty is a typed item model no served instance was compared with. The ratchet is exact per case: an
#: empty list not recorded here is red (a world change emptied it), and a recorded one now filled is red too (remove
#: it). It may only shrink: seed the world's rows to fill one.
EMPTY_LISTS_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "covered_route_empty_lists.json"
OBSERVED_EMPTY: dict[str, list[str]] = {}


def empty_lists(value: Any, path: str = "") -> list[str]:
    """The JSON paths (`a.b[].c`) of every empty list in `value`."""
    if isinstance(value, list):
        if not value:
            return [path or "."]
        return sorted({p for item in value for p in empty_lists(item, path + "[]")})
    if isinstance(value, dict):
        return sorted({p for k, v in value.items() for p in empty_lists(v, f"{path}.{k}" if path else k)})
    return []


@functools.lru_cache(maxsize=1)
def _recorded_empty() -> dict[str, list[str]]:
    return json.loads(EMPTY_LISTS_PATH.read_text(encoding="utf-8")) if EMPTY_LISTS_PATH.exists() else {}


async def assert_case(case: Case, session: Any, monkeypatch: Any, data_dir: Path) -> Result:
    """One case on its own world: it answers its expected status, the answer middleware checked that answer and found
    it on the contract, and the same body, checked again here against the committed contract without the middleware,
    is on it too. An agent's call to a gated route queues nothing (the approval is filed instead)."""
    from tests.support.answer_middleware import contract

    [result] = await run_cases([case], session, monkeypatch, data_dir)
    assert result.ok, result.describe()
    assert result.checked >= 1, f"{case.id}: the answer middleware checked no answer of this route"
    found = contract()
    if case.key in found.contract["responses"] and result.body:
        again = found.check_json(case.key, result.got, json.loads(result.body))
        assert not again, f"{case.id}: off contract (checked again): {again[0]}"
    else:
        again = found.check_other(case.key, result.got, result.media, len(result.body))
        assert not again, f"{case.id}: off contract (checked again): {again[0]}"
    if case.caller == "agent":
        assert not result.reached, f"{case.id}: an agent's call queued {result.reached}"
    elif case.key in found.contract["responses"] and result.body:
        observed = empty_lists(json.loads(result.body))
        OBSERVED_EMPTY[case.key] = observed
        recorded = _recorded_empty().get(case.key, [])
        assert observed == recorded, (f"{case.id}: the answer's empty lists moved (finding 8 ratchet, "
                                      f"{EMPTY_LISTS_PATH.name}): newly empty {sorted(set(observed) - set(recorded))}, "
                                      f"now filled (remove them) {sorted(set(recorded) - set(observed))}")
    return result
