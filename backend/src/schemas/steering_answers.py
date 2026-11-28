"""The answers of the steering routes that declare no response model (046 FTASKS 8.6; FTID §3.5).

Documentation only (FR-046.5 as amended 2026-11-26): each route keeps its serializer; these models are named in its
`responses=`. A key one branch of a route writes and another does not is `may_be_absent()`.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .answer_fields import may_be_absent


class SteeringCardWorker(BaseModel):
    """One card's steering worker (`steering_workers.card_worker_status`)."""

    card: str = Field(description="The card's UUID")
    index: Optional[int]
    name: Optional[str]
    worker_pid: Optional[int]
    busy: bool
    task_id: Optional[str]


class SteeringModeAnswer(BaseModel):
    """`GET /steering/mode`: whether a steering worker runs, and (per-card mode) every card's."""

    active: bool
    worker_pid: Optional[int]
    gpu_memory_mb: Optional[int] = Field(description="GPU memory used across every card, in MB")
    workers: Optional[list[SteeringCardWorker]] = may_be_absent(description="Per-card mode only")


class SteeringWorkerStarted(BaseModel):
    """One card's worker as `enter-mode` left it."""

    card: str
    ok: bool
    worker_pid: Optional[int]
    already_active: bool


class SteeringEnterAnswer(BaseModel):
    """`POST /steering/enter-mode`."""

    success: bool
    message: str
    worker_pid: Optional[int]
    already_active: bool
    workers: Optional[list[SteeringWorkerStarted]] = may_be_absent(description="Per-card mode only")


class SteeringExitAnswer(BaseModel):
    """`POST /steering/exit-mode`."""

    success: bool
    message: str
    killed_pid: Optional[int]
    gpu_memory_before: Optional[int] = may_be_absent()
    gpu_memory_after: Optional[int] = may_be_absent()
    gpu_memory_freed_mb: int
    already_inactive: bool


class SteeringStatusAnswer(BaseModel):
    """`GET /steering/status`: the steering service's health."""

    status: str = Field(description="healthy or degraded")
    resilience: dict[str, Any] = Field(description="The circuit breaker's and the timeouts' state")
    cache: dict[str, Any] = Field(description="What the steering service holds loaded")
    timeout_seconds: float


class SteeringExperimentAnswer(BaseModel):
    """A saved steering experiment (`SteeringExperiment.to_dict`)."""

    id: str
    name: str
    description: Optional[str]
    sae_id: str
    model_id: str
    prompt: str
    selected_features: list[dict[str, Any]]
    generation_params: dict[str, Any]
    results: dict[str, Any]
    tags: list[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class SteeringPagination(BaseModel):
    skip: int
    limit: int
    total: int
    has_more: bool


class SteeringExperimentListAnswer(BaseModel):
    data: list[SteeringExperimentAnswer]
    pagination: SteeringPagination


class SteeringExperimentDeleted(BaseModel):
    message: str
