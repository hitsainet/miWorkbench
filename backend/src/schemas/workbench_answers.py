"""The answers of the smaller routers (046 FTASKS 8.9 and 8.10; FTID §3.5): labeling, features, feature groups,
models, operations, datasets, cluster profiles, the task queue, wsp and `GET /api/health`.

Documentation only (FR-046.5 as amended 2026-11-26): each route keeps its serializer; these models are named in its
`responses=`. A key one branch writes and another does not is `may_be_absent()`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .answer_fields import may_be_absent

# ── labeling.py: the prompt-template trials ─────────────────────────────────────────────────────────────────────


class LabelingTrialStarted(BaseModel):
    """`POST /labeling/trials`: the trial queued; it writes no labels."""

    trial_run_id: str
    labeling_job_id: str
    panel_id: Optional[str]
    status: str
    writes_labels: bool = Field(description="Always false: a trial labels nothing")


class LabelingTrialRow(BaseModel):
    trial_run_id: str
    panel_id: Optional[str]
    name: Optional[str]
    prompt_template_id: Optional[str]
    status: str
    extraction_job_id: str
    stats: Optional[dict[str, Any]] = Field(description="The trial's summary statistics, once it has them")
    created_at: datetime
    completed_at: Optional[datetime]


class LabelingTrialMeta(BaseModel):
    limit: int
    offset: int


class LabelingTrialListAnswer(BaseModel):
    data: list[LabelingTrialRow]
    meta: LabelingTrialMeta


class LabelingTrialAnswer(BaseModel):
    """One trial's full result."""

    trial_run_id: str
    panel_id: Optional[str]
    name: Optional[str]
    status: str
    error: Optional[str]
    prompt_template_id: Optional[str]
    extraction_job_id: str
    payload: Optional[dict[str, Any]] = Field(description="The trial's results as stored")
    created_at: datetime
    completed_at: Optional[datetime]


class LabelingTrialComparison(BaseModel):
    """`GET /labeling/trials/compare/{a}/{b}`: two trials over one panel (`LabelingTrialService.compare`; a comparison
    across panels is refused 409). The branch with no comparable features sends fewer keys."""

    run_a: str
    run_b: str
    comparable: bool
    verdict: Optional[str] = Field(description="identical, b_differs, inconclusive or null")
    compared: int
    detection_delta: Optional[dict[str, Any]]
    reason: Optional[str]
    errors: Optional[dict[str, int]] = may_be_absent(description="Per arm: the overlapping features that errored")
    panel_id: Optional[str] = may_be_absent()
    overlap: Optional[int] = may_be_absent()
    errored: Optional[int] = may_be_absent()
    label_change_rate: Optional[float] = may_be_absent()
    category_change_rate: Optional[float] = may_be_absent()
    per_feature: Optional[list[dict[str, Any]]] = may_be_absent()
    detection_delta_midrange: Optional[dict[str, Any]] = may_be_absent()
    two_ruler_verdict: Optional[dict[str, Any]] = may_be_absent(description="The verdict reading both rulers")


# ── features.py ─────────────────────────────────────────────────────────────────────────────────────────────────


class FeatureIdAnswer(BaseModel):
    """A feature's id, found by its index in a training or an SAE (null when no feature has that index)."""

    feature_id: Optional[str]


class TokenAnalysisSummary(BaseModel):
    total_examples: int
    original_token_count: int
    filtered_token_count: int
    junk_removed: int
    total_token_occurrences: int
    filtered_token_occurrences: int
    diversity_percent: float
    filter_stats: Optional[dict[str, int]] = may_be_absent(description="Tokens each filter removed (filters on)")


class TokenAnalysisRow(BaseModel):
    rank: int
    token: str
    count: int
    percentage: float


class TokenAnalysisAnswer(BaseModel):
    """`GET /features/{id}/token-analysis` (`utils.token_filters.analyze_feature_tokens`)."""

    summary: TokenAnalysisSummary
    tokens: list[TokenAnalysisRow]


# ── feature_groups.py and the other cancel routes ───────────────────────────────────────────────────────────────


class CancelRequested(BaseModel):
    """A run asked to stop (`core.cancellation.request_cancel`'s outcome, in words)."""

    id: str
    status: str = Field(description="cancelled")
    message: str


# ── models.py ───────────────────────────────────────────────────────────────────────────────────────────────────


class ModelArchitectureAnswer(BaseModel):
    """`GET /models/{id}/architecture` (`ModelService.get_model_architecture_info`)."""

    model_id: str
    name: str
    architecture: str
    params_count: int
    quantization: str
    architecture_config: Optional[dict[str, Any]]
    memory_required_bytes: Optional[int]
    disk_size_bytes: Optional[int]


class ModelDownloadCancelled(BaseModel):
    model_id: str
    status: str = Field(description="cancelled")
    message: str


# ── operations.py ───────────────────────────────────────────────────────────────────────────────────────────────


class ActiveJobsAnswer(BaseModel):
    """`GET /operations/active-jobs` (`services.active_jobs.list_active_jobs`, with the queue watch beside it)."""

    jobs: list[dict[str, Any]] = Field(description="Every queued or running job (rows of several kinds)")
    count: int
    kinds: list[str] = Field(description="The job kinds a `kind` filter accepts")
    errors: list[dict[str, Any]] = Field(description="Each source that could not be read, with the error")
    read_at: str
    note: str
    worker_watch: dict[str, Any] = Field(description="Whether GPU work waits on a stuck worker (`gpu_queue_state`)")
    gpu: Optional[str] = may_be_absent(description="The card asked for, when filtered by one")


class GpuStatusAnswer(BaseModel):
    """`GET /operations/gpu-status`: live per-card state."""

    worker_watch: dict[str, Any]
    cards: list[dict[str, Any]] = Field(description="Each card: identity, memory, leases and the jobs on it")
    card_count: int
    reason: Optional[str] = Field(description="Why the cards could not be read, when they could not")
    job_read_errors: list[dict[str, Any]]
    read_at: str
    attribution: str = Field(description="How a process on a card is attributed to miWorkbench")


# ── datasets.py, cluster_profiles.py, task_queue.py ─────────────────────────────────────────────────────────────

#: The `$id` of `docs/schemas/cluster-definition-v1.json`, which `GET /cluster-profiles/{id}/export` answers (FR-046.11).
CLUSTER_DEFINITION_ID = "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/cluster-definition-v1.json"  # noqa: E501


class DatasetDownloadCancelled(BaseModel):
    dataset_id: str
    status: str = Field(description="cancelled")
    message: str


class TaskLookupRow(BaseModel):
    table: str
    row_id: str
    status: Optional[str]


class TaskLookupAnswer(BaseModel):
    """`GET /task-queue/lookup/{task_id}`: the rows that name a Celery task id."""

    task_id: str
    found: list[TaskLookupRow]
    searched: list[str] = Field(description="The tables looked in")


# ── main.py: GET /api/health (the route miAssist calls) ─────────────────────────────────────────────────────────


def _legacy() -> tuple[type, type]:
    from .legacy_names import LegacyNamesBlock, LegacyNamesMeta

    return LegacyNamesBlock, LegacyNamesMeta


class HealthAnswer(BaseModel):
    """`GET /api/health`: the service's identity and the old-name evidence of ENH-009 (042 FR-042.9)."""

    status: str
    service: str
    version: str
    legacy_names: _legacy()[0]  # type: ignore[valid-type]
    legacy_names_meta: _legacy()[1]  # type: ignore[valid-type]


# ── wsp.py ──────────────────────────────────────────────────────────────────────────────────────────────────────


class WspCaptureRow(BaseModel):
    """One held-out capture (`wsp.capture_view`)."""

    id: str
    probe_id: str
    dataset_id: str
    role: str
    source_evaluation_id: Optional[str]
    layer: int
    scope: str
    rows: Optional[int]
    tokens: Optional[int]
    d_model: Optional[int]
    dtype: Optional[str]
    row_ids_sha256: Optional[str]
    parity: Optional[dict[str, Any]] = Field(description="The parity record against the stored evaluation")
    status: str
    bytes: Optional[int]
    error: Optional[str]
    created_at: Optional[str]
    finished_at: Optional[str]
    superseded_at: Optional[str]
    superseded_by: Optional[str]


class WspCaptureStarted(BaseModel):
    """`POST /wsp/captures`: the captures planned, the task (null when nothing needed running) and the card check."""

    probe_id: str
    task_id: Optional[str]
    task_queue_id: Optional[str] = Field(description="The capture's task-queue row (`task_type` wsp_heldout_capture, "
                                                     "2026-11-27); null when nothing needed running")
    started: list[str] = Field(description="The capture ids this call queued")
    captures: list[WspCaptureRow]
    gpu: Optional[dict[str, Any]] = Field(description="The memory check (`wsp_capture.capture_gpu_check`); null when "
                                                      "nothing needed running")


class WspPreflightSet(BaseModel):
    dataset_id: str
    role: str
    rows: int
    capture: Optional[str]
    status: Optional[str] = Field(description="The current capture's status; null when the set has none")
    would_run: bool


class WspCaptureRefusal(BaseModel):
    code: str
    message: Optional[str]


class WspCapturePreflight(BaseModel):
    """`GET /wsp/captures/preflight`: what a capture would do now, writing nothing."""

    probe_id: str
    sets: list[WspPreflightSet]
    to_run: int
    rows_to_run: int
    gpu: dict[str, Any]
    refusal: Optional[WspCaptureRefusal]


class WspCandidateImporting(BaseModel):
    """`POST /wsp/candidates/import`: the candidate checked and its import queued."""

    bundle_id: str
    parent_probe_id: str
    miforge_run_id: str
    score_call_id: str
    task_id: str
    task_queue_id: str = Field(description="The import's task-queue row (`task_type` wsp_candidate_import, "
                                           "2026-11-27): `get_task_status` answers it by `task_id`")
    status: str = Field(description="importing")
