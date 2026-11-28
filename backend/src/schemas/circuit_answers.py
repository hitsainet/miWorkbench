"""The answers of the circuit routes (046 FTASKS 8.4 and 8.5; FTID §3.5): `circuits.py`, `circuit_discovery.py` and
`circuit_validation.py`.

Documentation only (FR-046.5, as amended 2026-11-26): each route keeps the serializer it had, and these models are
named in its `responses=` so the published contract (the answer contract, `answers_contract.DEFAULT_PATH`) states the shape.
Every key a route's helper writes is a declared field; a key some branch leaves out is `may_be_absent()`; a stored
free-form JSON value (members, edges, a run's config) is a declared map, `dict[str, Any]`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .answer_fields import history_form, may_be_absent

#: The `$id` of `docs/schemas/circuit-definition-v1.json`, which `GET /circuits/{id}/export` answers unchanged
#: (FR-046.11); `test_answer_errors_and_exports.py` checks it equals the file's.
CIRCUIT_DEFINITION_ID = "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/circuit-definition-v1.json"  # noqa: E501

# ── circuits.py ─────────────────────────────────────────────────────────────────────────────────────────────────


class CircuitSummaryAnswer(BaseModel):
    """One circuit in a list (`circuits._summary`): the slim row, no JSON bodies."""

    id: str
    name: str
    granularity: str = Field(description="feature or cluster")
    layers: list[Optional[int]] = Field(description="The members' layers, sorted, each once")
    member_count: int
    edge_count: int
    rung: int = Field(description="The evidence rung the circuit has earned (0 to 3)")
    rung_language: str = Field(description="What the rung licenses saying, in words")
    rung_next_step: str = Field(description="What earns the next rung")
    promoted: bool
    model_id: Optional[str]
    version: int = Field(description="The optimistic-concurrency token a PATCH may send back")
    updated_at: datetime


class CircuitAnswer(CircuitSummaryAnswer):
    """One circuit in full (`circuits._out`): get, create, import, patch and promote all answer it."""

    narrative: Optional[str]
    saes: list[dict[str, Any]] = Field(description="The SAEs the members come from (`DefinitionSAERef`, stored JSON)")
    members: list[dict[str, Any]] = Field(description="The members (`CircuitMember`, stored JSON)")
    edges: list[dict[str, Any]] = Field(description="The edges (`CircuitEdge`, stored JSON)")
    budget: Optional[dict[str, Any]]
    faithfulness: Optional[dict[str, Any]]
    faithfulness_status: Optional[str] = Field(description="pending, running, completed, failed or cancelled")
    calibration: Optional[dict[str, Any]]
    calibration_status: Optional[str]
    discovery: Optional[dict[str, Any]] = Field(description="Where the circuit was discovered (stored JSON)")
    discovery_run_id: Optional[str]
    model_hf_id: Optional[str]
    created_at: datetime


class CircuitListAnswer(BaseModel):
    """A page of circuits (`GET /circuits`)."""

    circuits: list[CircuitSummaryAnswer]
    total: int
    limit: int
    offset: int


class CircuitDeletedAnswer(BaseModel):
    deleted: str = Field(description="The deleted circuit's id")


class CircuitTaskQueued(BaseModel):
    """A circuit-scoped GPU stage queued (faithfulness, calibration)."""

    circuit_id: str
    task_id: str
    status: str = Field(description="queued")


class CalibrationReproductionQueued(BaseModel):
    reproduces: str = Field(description="The calibration manifest being reproduced")
    task_id: str
    status: str


class SteeringSamplesQueued(BaseModel):
    record_run_id: str
    task_id: str
    status: str


class CircuitSlicesAnswer(BaseModel):
    """`POST /circuits/{id}/export-slices`: per-layer cluster-definition/v1 renderings of the circuit."""

    parent: str
    parent_rung: int
    parent_rung_language: str
    slices: list[dict[str, Any]] = Field(description="Each a `mistudio.cluster-definition` document (partial)")


class CircuitStageCancelled(BaseModel):
    """A circuit-scoped stage cancelled (`circuits._cancel_circuit_stage`): the status field the stage keeps is named
    after it, so exactly one of the two is sent."""

    circuit_id: str
    faithfulness_status: Optional[str] = may_be_absent(description="cancelled, for a faithfulness run")
    calibration_status: Optional[str] = may_be_absent(description="cancelled, for a calibration run")
    was_running: bool
    message: str


class RunCancelled(BaseModel):
    """A run cancelled by id (the steering recorder, the discovery and validation stages)."""

    id: str
    status: str = Field(description="cancelled")
    was_running: bool
    message: str


# ── circuit_discovery.py ────────────────────────────────────────────────────────────────────────────────────────


class RunQueued(BaseModel):
    """A run queued on a worker: its id, the Celery task and `queued`."""

    id: str
    task_id: str
    status: str = Field(description="queued")


class CaptureQueued(RunQueued):
    """`POST /circuit-capture`: queued, and whether this was the confirmed full capture or the estimate."""

    confirmed: bool


class CaptureAnswer(BaseModel):
    """A capture run (`circuit_discovery._capture_out`)."""

    id: str
    status: str
    progress: Optional[float]
    error_message: Optional[str]
    corpus: Optional[dict[str, Any]] = Field(description="The corpus the capture reads (from its manifest)")
    model_id: Optional[str]
    layers: Optional[list[dict[str, Any]]] = Field(description="Each captured layer and its SAE (from the manifest)")
    split: dict[str, Any] = Field(description="The train/held-out split, with `heldout_count` in place of the list")
    estimate: Optional[dict[str, Any]]
    attention_capture: Optional[dict[str, Any]]
    counts: Optional[dict[str, Any]]
    bytes: Optional[int]
    events_total: Optional[int]
    stale: bool = Field(description="An SAE changed since the capture")
    gpu_request: Optional[str]
    gpu_uuid: Optional[str]
    created_at: datetime
    updated_at: datetime


class CaptureListAnswer(BaseModel):
    captures: list[CaptureAnswer]
    limit: int
    offset: int


class CaptureCancelled(BaseModel):
    id: str
    status: str = Field(description="cancelled")


class DiscoveryAnswer(BaseModel):
    """A discovery run (`circuit_discovery._discovery_out`); `candidates` only when asked for."""

    id: str
    capture_run_id: str
    status: str
    progress: Optional[float]
    error_message: Optional[str]
    params: dict[str, Any] = Field(description="The run's parameters, as submitted")
    report: Optional[dict[str, Any]] = Field(description="The trust surface: null method, FDR, replication, caps")
    candidate_count: int
    attribution_status: Optional[str]
    attribution_progress: Optional[float]
    attribution_error: Optional[str]
    validation_status: Optional[str]
    validation_progress: Optional[float]
    validation_error: Optional[str]
    created_at: datetime
    updated_at: datetime
    candidates: Optional[list[dict[str, Any]]] = may_be_absent(description="The candidate edges (include_candidates)")


class DiscoveryCircuitBuilt(BaseModel):
    """`POST /circuit-discovery/{id}/build-circuit`: the new circuit, unpromoted."""

    id: str
    discovery_run_id: str
    member_count: int
    edge_count: int
    rung: int


class AttributionCancelled(BaseModel):
    id: str
    attribution_status: str = Field(description="cancelled")


# ── circuit_validation.py ───────────────────────────────────────────────────────────────────────────────────────


class ValidationCancelled(BaseModel):
    id: str
    validation_status: str = Field(description="cancelled")


class ValidationManifestAnswer(BaseModel):
    """A validation or calibration manifest (`circuit_validation._manifest_out`)."""

    id: str
    kind: str
    discovery_run_id: Optional[str]
    circuit_id: Optional[str]
    parent_manifest_id: Optional[str]
    payload: dict[str, Any] = history_form(description="The manifest as recorded (stored JSON)")
    created_at: datetime


class ValidationManifestListAnswer(BaseModel):
    manifests: list[ValidationManifestAnswer]


class ValidationReproductionQueued(BaseModel):
    reproduce_of: str = Field(description="The manifest being reproduced")
    task_id: str
    status: str
