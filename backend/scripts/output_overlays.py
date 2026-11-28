"""The declared overlays: what a tool adds to, wraps around or builds instead of its route's answer (046 WT7, MW5,
FR-046.20).

`scripts/generate_output_schemas.py` types a tool from the contract entry of each route it answers. A tool whose every
`return` is one route's body unchanged (`src/mcp_server/route_map.returns_unchanged`) needs nothing here. Every other
tool needs a row, or a reason from the closed list (`generate_output_schemas.REASONS`): `test_output_overlays.py` pins
the set equality both ways, so a tool that starts building its answer without a row is red, and so is a stale row.

A row says, for one tool:

* `base`: the routes whose body the tool answers (`"<METHOD> <path>"`, miWorkbench's under `/api/v1`, miServer's as
  its contract keys them);
* `wrapper`: a Pydantic model of the keys the tool adds BESIDE the body (a required field is always added; a field
  with a default is sometimes left out);
* `nest`: the key the body is placed under instead (`send` answers `{"result": <body>, "millm_headers"}`), with
  `wrapper` beside it;
* `variants`: answers the tool builds instead of a route's body (a not-found answer, a page it cuts from a file);
* `items_of`: a `variants` field whose items are the base route's list items, each plus `item_wrapper`'s keys;
* `code`: where the wrapping happens (`module:function`), or the tool itself when it has no separable function.

Refusals, argument errors, unavailable answers and pending approvals are typed once, as the schema's ALTERNATIVES
(MW4); a tool that only adds those needs `alternatives_only=True` and nothing else. The legacy `next` is never
constrained (045 A30), and `next_steps` and `approval` are the result language's, at the schema's root: a wrapper
declares neither.

`test_output_overlays.py` drives every row's tool through the SDK client with its base route answering a full body
(the real wrapper code runs) and checks the added keys EQUAL the wrapper's fields, and that the answer validates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Open(BaseModel):
    """Overlay models describe keys; the answer stays open (046 SH7: extra keys are allowed)."""

    model_config = ConfigDict(extra="allow")


# ── wrappers: keys beside the route's body ─────────────────────────────────────────────────────────────────────


class SendProvenance(_Open):
    """`miserver_send.send`: miServer's provenance headers, beside the body nested under `result`."""

    millm_headers: dict[str, str] = Field(description="The provenance headers the response carried, verbatim")


class ProbeVerdicts(SendProvenance):
    """`miserver_inference.lift_probe_verdicts`: the body's probe verdicts, lifted to the top."""

    millm_probe_verdicts: list[Any] = Field(description="miServer's probe verdicts for this request; [] when none")


class GenerateProvenance(ProbeVerdicts):
    """`miserver_generate`: also says, coarsely, whether the generation was steered (`miserver_send.steering_state`)."""

    steering_state: str = Field(description="unsteered | steered | unknown | not_reported")


class ProbeIdCopy(_Open):
    """`miserver_probes.with_probe_id`: the imported row's `id`, copied as `probe_id`."""

    probe_id: str = Field(description="miServer's probe id, the same value as `id`")


class ProgressNote(_Open):
    """`miserver_models._with_progress_note`: says when a download's progress is unknown, not zero."""

    progress_note: Optional[str] = Field(default=None, description="Present while downloading with no reported progress")


class ServingGuardSkipped(_Open):
    """`miserver_delete_circuit`: the delete ran without its serving check."""

    guard_skipped: Optional[str] = Field(default=None, description="serving_state_unreadable, when the check could not run")
    warning: Optional[str] = Field(default=None, description="What that means for live steering")


class LeaseHint(_Open):
    """`miserver_acquire_lease`: the lease nested under `lease`, with how to keep it."""

    hint: str = Field(description="The lease id is shown once; how to keep using it")


class ContextLengthNote(_Open):
    """`miserver_list_models`, per row: the resident model's window could not be read."""

    context_length_note: Optional[str] = Field(default=None, description="Present when the loaded model's window is unknown")
    progress_note: Optional[str] = Field(default=None, description="Present while downloading with no reported progress")


class JudgeGateNote(_Open):
    """`probe_monitors._with_gate_note`: why a judge run was not gated by miServer's queue."""

    judge_gate: Optional[str] = Field(default=None, description="The standalone note's code")
    judge_gate_reason: Optional[str] = Field(default=None, description="Its reason")


class TrainingForecast(_Open):
    """`start_probe_training_run`: the memory and time forecasts and the rule recommendation, beside the run."""

    memory_forecast: Optional[dict[str, Any]] = Field(default=None, description="The host-memory forecast (first call)")
    time_forecast: Optional[dict[str, Any]] = Field(default=None, description="The time forecast (first call)")
    rule_recommendation: Optional[dict[str, Any]] = Field(default=None, description="Which pooling rules fit")


class DetectorUnverified(_Open):
    """`miserver_recalibrate_probe`: pushed to a miServer copy whose detector could not be verified."""

    detector_unverified: Optional[bool] = Field(default=None, description="true when miServer states no detector digest")
    detector_unverified_reason: Optional[str] = Field(default=None, description="Why it could not be verified")


# ── variants: answers the tool builds ───────────────────────────────────────────────────────────────────────────


class ApprovalDescription(_Open):
    """`approvals.describe`: one miWorkbench approval row as an agent reads it."""

    approval_request_id: Optional[str] = Field(description="The approval request id")
    tool_name: Optional[str] = Field(description="The tool it gates")
    status: Optional[str] = Field(description="pending | approved | denied | expired | consumed")
    decided: bool = Field(description="false while pending")
    decided_by: Optional[str] = Field(description="Who decided, when decided")
    decided_at: Optional[str] = Field(description="When it was decided")
    consumed_at: Optional[str] = Field(description="When an approved request was used")
    created_at: Optional[str] = Field(description="When it was filed")
    requested_by: Optional[str] = Field(description="Who filed it")
    expires_at: Optional[str] = Field(description="When it expires")
    reason: Optional[str] = Field(description="The operator's or the expiry's reason")
    steering_task_id: Optional[str] = Field(description="The steering task it started, if any")
    payload: Optional[dict[str, Any]] = Field(description="The request it covers")


class TaskNotFound(_Open):
    """`get_task_status`: no task-queue record and no job row for the id."""

    status: Literal["not_found"]
    task_id: Optional[str] = Field(default=None, description="The Celery id asked about")
    task_queue_id: Optional[str] = Field(default=None, description="The task-queue id asked about")


class TaskPointer(_Open):
    """`jobs._answer_lookup`: where a Celery task is recorded, when it has no task-queue record."""

    task_id: str = Field(description="The Celery id asked about")
    status: Any = Field(description="The job row's status")
    found_in: str = Field(description="The table of the job row")
    row_id: str = Field(description="The job row's id")
    also_found: Optional[list[dict[str, Any]]] = Field(default=None, description="Other rows naming the id")


class NothingLeftToLabel(_Open):
    """`resume_labeling`: the extraction's coverage says nothing is left."""

    labeled: int
    reason: str
    coverage: dict[str, Any]


class SteeringSamplesOfCircuit(_Open):
    """`get_steering_samples` with `circuit_id`: the circuit's steering-sample manifests."""

    circuit_id: str
    manifests: list[dict[str, Any]]


class NothingToCapture(_Open):
    """`capture_wsp_heldout`: every held-out set is already captured or under way."""

    probe_id: str
    started: list[Any]
    sets: Any
    note: str


class CaptureRefusedBeforeFiling(_Open):
    """`capture_wsp_heldout`: the preflight's refusal (e.g. `gpu_memory_insufficient`), with the card check and the
    sets beside it; nothing is filed or spent."""

    refused: str
    reason: Optional[str]
    gpu: dict[str, Any]
    sets: list[dict[str, Any]]


class RemoteApprovalNotFound(_Open):
    """`miserver_get_approval_status`: miServer has no such approval."""

    approval_id: str
    status: Literal["not_found"]


class AnyLeaseHeld(_Open):
    """`miserver_lease_status` without a model: whether miServer holds any lease."""

    any_lease_held: bool
    lease: Optional[dict[str, Any]]
    model_loaded: Any
    model_name: Optional[str]
    scope: str


class LoadStarted(_Open):
    """`miserver_load_model`: miServer started the load (no approval was needed)."""

    load_started: Literal[True]
    model: Any = Field(description="miServer's model row")


class BatchResultsPage(_Open):
    """`miserver_batch_results`: one page of a batch's output or error file (JSON Lines rows)."""

    rows: list[dict[str, Any]]
    total: int
    next_offset: Optional[int]
    millm_headers: dict[str, str]


class ModelList(_Open):
    """`miserver_list_models`: every model row (each the route's row plus its notes), and how many."""

    models: list[dict[str, Any]]
    count: int


class GpuPreflight(_Open):
    """`gpu_preflight`: can a GPU job start on this card, from every product's view."""

    gpu: str
    verdict: Literal["clear", "busy", "uncertain"]
    busy_reasons: list[str]
    uncertain_reasons: list[str]
    mistudio: dict[str, Any] = Field(description="miWorkbench's card, active jobs and worker watch")  # p9: keep, frozen answer key (DR-53)
    millm: dict[str, Any] = Field(description="miServer's view of the card (configured, reachable, resident model)")  # p9: keep, frozen answer key (DR-53)
    miforge: dict[str, Any] = Field(description="miForge's pending and running runs")
    read_at: Any = Field(description="When the active-job list was read")


class GpuPreflightRefused(_Open):
    """`gpu_preflight`'s two refusals, which answer `refused: true` (a boolean, unlike the shared string code)."""

    refused: Literal[True]
    detail: Any
    gpu: Optional[str] = Field(default=None)
    status_code: Optional[int] = Field(default=None)


# ── the table ───────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Overlay:
    tool: str
    base: tuple[str, ...]
    code: str
    wrapper: Optional[type[BaseModel]] = None
    nest: Optional[str] = None
    variants: tuple[type[BaseModel], ...] = ()
    items_of: Optional[str] = None
    item_wrapper: Optional[type[BaseModel]] = None
    alternatives_only: bool = False
    #: Arguments that reach the base route's answer in the behavioural test (beyond the required ones).
    drive_args: dict[str, Any] = field(default_factory=dict)


def _alt(tool: str, *base: str, code: str = "", **kw: Any) -> Overlay:
    return Overlay(tool, tuple(base), code or f"tool:{tool}", alternatives_only=True, **kw)


T = "tool:"
SEND = "src.mcp_server.miserver_send:send"
RELAYED = "src.mcp_server.tools.miserver_approvals:relayed"

_ROWS: list[Overlay] = [
    # ── native tools (miWorkbench routes under /api/v1) ──
    _alt("cancel_steering_task", "DELETE /api/v1/steering/async/task/{}"),
    Overlay("capture_wsp_heldout", ("POST /api/v1/wsp/captures",), T + "capture_wsp_heldout",
            variants=(NothingToCapture, CaptureRefusedBeforeFiling)),
    _alt("delete_model", "DELETE /api/v1/models/{}", drive_args={"acknowledge_permanent": True}),
    _alt("delete_probe_training_run", "DELETE /api/v1/probe-monitors/runs/{}", drive_args={"acknowledge_permanent": True}),
    _alt("evaluate_probe_monitor", "POST /api/v1/probe-monitors/probes/{}/evaluate"),
    Overlay("get_approval_status", (), "src.mcp_server.tools.approvals:describe", variants=(ApprovalDescription,)),
    _alt("get_feature_by_index", "GET /api/v1/saes/{}/features/by-index/{}",
         "GET /api/v1/trainings/{}/features/by-index/{}", drive_args={"sae_id": "sae_1"}),
    _alt("get_steering_result", "GET /api/v1/steering/async/result/{}"),
    Overlay("get_steering_samples", ("GET /api/v1/validation-manifests/{}",), T + "get_steering_samples",
            variants=(SteeringSamplesOfCircuit,), drive_args={"manifest_id": "vman_1"}),
    Overlay("get_task_status", ("GET /api/v1/task-queue/{}", "GET /api/v1/task-queue/active", "GET /api/v1/task-queue"),
            "src.mcp_server.tools.jobs:_answer_lookup", variants=(TaskNotFound, TaskPointer),
            drive_args={"task_queue_id": "tq_1"}),
    Overlay("gpu_preflight", (), T + "gpu_preflight", variants=(GpuPreflight, GpuPreflightRefused),
            drive_args={"gpu": "GPU-00000000-0000-0000-0000-000000000000"}),
    _alt("list_trainings", "GET /api/v1/trainings"),
    _alt("recalibrate_probe_monitor", "POST /api/v1/probe-monitors/probes/{}/recalibrate"),
    Overlay("resume_labeling", ("POST /api/v1/labeling/panel",), T + "resume_labeling", variants=(NothingLeftToLabel,)),
    _alt("score_probe_family", "POST /api/v1/probe-monitors/behaviours/{}/family/score-batch", drive_args={"text": "t"}),
    _alt("score_probe_monitor", "POST /api/v1/probe-monitors/probes/{}/score", drive_args={"text": "t"}),
    _alt("score_probe_monitor_batch", "POST /api/v1/probe-monitors/probes/{}/score-batch",
         drive_args={"inputs": [{"text": "t"}]}),
    _alt("start_circuit_capture", "POST /api/v1/circuit-capture"),
    Overlay("start_probe_judge_run", ("POST /api/v1/probe-monitors/judge-runs",),
            "src.mcp_server.tools.probe_monitors:_with_gate_note", wrapper=JudgeGateNote),
    Overlay("start_probe_spot_check", ("POST /api/v1/probe-monitors/spot-checks",),
            "src.mcp_server.tools.probe_monitors:_with_gate_note", wrapper=JudgeGateNote),
    Overlay("start_probe_training_run", ("POST /api/v1/probe-monitors/runs",), T + "start_probe_training_run",
            wrapper=TrainingForecast),
    _alt("steer_combined", "POST /api/v1/steering/async/combined"),
    _alt("steer_compare", "POST /api/v1/steering/async/compare"),
    _alt("steer_sweep", "POST /api/v1/steering/async/sweep"),
    _alt("update_probe_dataset_columns", "PATCH /api/v1/probe-monitors/datasets/{}/columns",
         drive_args={"text_column": "text"}),
    # ── miServer relays (miServer's contract keys) ──
    Overlay("miserver_acquire_lease", ("POST /api/models/{}/lease",), T + "miserver_acquire_lease",
            wrapper=LeaseHint, nest="lease"),
    _alt("miserver_arm_probe", "POST /api/probes/{}/arm"),
    Overlay("miserver_batch_lease", ("POST /v1/batches/{}/lease",), SEND, wrapper=SendProvenance, nest="result"),
    Overlay("miserver_batch_results", (), T + "miserver_batch_results", variants=(BatchResultsPage,)),
    Overlay("miserver_batch_status", ("GET /v1/batches/{}",), SEND, wrapper=SendProvenance, nest="result"),
    Overlay("miserver_cancel_batch", ("POST /v1/batches/{}/cancel",), SEND, wrapper=SendProvenance, nest="result"),
    _alt("miserver_circuit_sensing_clear", "DELETE /api/circuit-sensing/events", drive_args={"circuit_id": "c"}),
    _alt("miserver_circuit_sensing_events", "GET /api/circuit-sensing/events"),
    _alt("miserver_clear_probe_events", "DELETE /api/probes/events", drive_args={"probe_id": "p"}),
    Overlay("miserver_delete_circuit", ("DELETE /api/circuits/{}",), T + "miserver_delete_circuit",
            wrapper=ServingGuardSkipped, drive_args={"acknowledge_serving": True}),
    _alt("miserver_delete_model", "DELETE /api/models/{}", code=RELAYED, drive_args={"acknowledge_permanent": True}),
    _alt("miserver_delete_probe", "DELETE /api/probes/{}", drive_args={"acknowledge_events_deleted": True}),
    _alt("miserver_delete_rule_set", "DELETE /api/rule-sets/{}", drive_args={"acknowledge_delete": True}),
    _alt("miserver_download_model", "POST /api/models"),
    Overlay("miserver_embeddings", ("POST /v1/embeddings",), SEND, wrapper=SendProvenance, nest="result"),
    Overlay("miserver_generate", ("POST /v1/chat/completions",),
            "src.mcp_server.tools.miserver_inference:lift_probe_verdicts", wrapper=GenerateProvenance, nest="result"),
    Overlay("miserver_get_approval_status", ("GET /api/approvals/{}",), T + "miserver_get_approval_status",
            variants=(RemoteApprovalNotFound,)),
    Overlay("miserver_get_model", ("GET /api/models/{}",), "src.mcp_server.tools.miserver_models:_with_progress_note",
            wrapper=ProgressNote),
    _alt("miserver_import_circuit", "POST /api/circuits/import", drive_args={"definition": {"x": 1}}),
    _alt("miserver_import_cluster", "POST /api/clusters/import", "POST /api/clusters/hub/import",
         drive_args={"definition": {"x": 1}}),
    Overlay("miserver_import_probe", ("POST /api/probes/import", "POST /api/probes/hub/import"),
            "src.mcp_server.tools.miserver_probes:with_probe_id", wrapper=ProbeIdCopy,
            drive_args={"definition": {"x": 1}}),
    _alt("miserver_import_rule_set", "POST /api/rule-sets/import", code=RELAYED),
    Overlay("miserver_lease_status", ("GET /api/models/{}/lease",), T + "miserver_lease_status",
            variants=(AnyLeaseHeld,), drive_args={"model_id": 1}),
    Overlay("miserver_list_models", ("GET /api/models",), T + "miserver_list_models", variants=(ModelList,),
            items_of="models", item_wrapper=ContextLengthNote),
    Overlay("miserver_load_model", (), T + "miserver_load_model", variants=(LoadStarted,)),
    _alt("miserver_measure_probe_capacity", "POST /api/models/{}/probe-capacity/measure", code=RELAYED),
    Overlay("miserver_recalibrate_probe", ("POST /api/probes/{}/recalibrate",), T + "miserver_recalibrate_probe",
            wrapper=DetectorUnverified),
    _alt("miserver_renew_lease", "POST /api/models/{}/lease/renew"),
    _alt("miserver_request_probe_limits", "PUT /api/models/{}/probe-limits", code=RELAYED),
    Overlay("miserver_score_chat", ("POST /v1/chat/completions",),
            "src.mcp_server.tools.miserver_inference:lift_probe_verdicts", wrapper=ProbeVerdicts, nest="result"),
    Overlay("miserver_score_completion", ("POST /v1/completions",), SEND, wrapper=SendProvenance, nest="result"),
    Overlay("miserver_score_probes", ("POST /api/probes/score",), SEND, wrapper=SendProvenance, nest="result",
            drive_args={"inputs": [{"text": "t"}]}),
    _alt("miserver_sensing_clear", "DELETE /api/sensing/events", drive_args={"profile_id": "p"}),
    _alt("miserver_sensing_config", "PUT /api/sensing/{}/config", drive_args={"min_k": 1}),
    _alt("miserver_sensing_events", "GET /api/sensing/events"),
    _alt("miserver_set_model_context_ceiling", "PUT /api/models/{}/context-ceiling", code=RELAYED,
         drive_args={"context_ceiling": 4096}),
    _alt("miserver_set_telemetry", "PUT /api/environment", code=RELAYED,
         drive_args={"values": {"TELEMETRY_TRACES_ENABLED": True}}),
    Overlay("miserver_submit_batch", ("POST /v1/batches",), SEND, wrapper=SendProvenance, nest="result"),
    _alt("miserver_unload_model", "POST /api/models/{}/unload", code=RELAYED),
    Overlay("miserver_upload_batch_file", ("POST /v1/files",), SEND, wrapper=SendProvenance, nest="result"),
]

OVERLAYS: dict[str, Overlay] = {row.tool: row for row in _ROWS}
assert len(OVERLAYS) == len(_ROWS), "a tool has two overlay rows"
