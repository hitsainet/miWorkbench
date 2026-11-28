"""The answers of the probe-monitor routes (046 FTASKS 8.x; FTID §3.5): `probe_monitors.py`, `probe_spot_checks.py`,
`rule_sets.py` and `POST /code-rules/{sha}/approval-requests` (`code_rules.py`).

Documentation only (FR-046.5 as amended 2026-11-26): each route keeps its serializer, and its `response_model` where
it has one; these models are named in its `responses=`. A key one branch writes and another does not is
`may_be_absent()`; a value stored as JSON by an earlier release is `history_form()` (or a declared map); a stored
free-form JSON value (a run's config, its environment, an evaluation's metrics) is a declared map, `dict[str, Any]`.

Five routes answer an existing schema class (`ProbeDatasetResponse`, `ProbeDatasetPage`, `ProbeRunResponse`,
`ProbeRunPage`, `ProbeEvaluationRows`) and keep it as their `response_model`; their documentation models below subclass
it and override only the fields it leaves as `Any` or as a map that is really a fixed object.

`GET /probe-monitors/probes/{id}/definition` has no model here: it answers the cached `mistudio.probe-definition/v1`
file unchanged (`json.loads`), so its schema is a `$ref` to `PROBE_DEFINITION_ID` (FR-046.11), set in its decorator as
`GET /circuits/{id}/export` does.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel

from .answer_fields import JsonCell, history_form, may_be_absent
from .probe_monitor import (
    ProbeDatasetPage,
    ProbeDatasetResponse,
    ProbeDatasetSummary,
    ProbeEvaluationRow,
    ProbeEvaluationRows,
    ProbeRunPage,
    ProbeRunResponse,
    ProbeRunSummary,
)

#: The `$id` of `docs/schemas/probe-definition-v1.json`, which `GET /probe-monitors/probes/{id}/definition` answers
#: unchanged (FR-046.11); a test should check it equals the file's, as `CIRCUIT_DEFINITION_ID` is checked.
PROBE_DEFINITION_ID = "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/probe-definition-v1.json"  # noqa: E501



# ── shared pieces ───────────────────────────────────────────────────────────────────────────────────────────────


class ProbeNextCall(BaseModel):
    """One entry of a `next` list (`services/next_calls.NextCall`): the call that fixes or follows this answer."""

    tool: str
    arguments: dict[str, Any] = Field(description="The call's arguments (a map: keys are the tool's parameters)")
    why: str
    needs: Optional[list[str]] = may_be_absent(
        description="Arguments only the agent can supply (never filled in for it)")


class ProbeTaskQueued(BaseModel):
    """A probe GPU job queued: evaluate, score, the recalibrate re-cut arm and publish."""

    probe_id: str
    task_id: str
    status: Literal["queued"]


class ProbeEvaluationQueued(ProbeTaskQueued):
    """`POST /probes/{id}/evaluate` (202): the evaluation queued WITH its task-queue row (2026-11-27, train W5), so
    `get_task_status` and Active Operations follow it (view k of n, rows x of y)."""

    task_queue_id: str = Field(description="The evaluation's task-queue row (`task_type` probe_evaluation)")


class ProbeCancelling(BaseModel):
    """A cooperative cancel accepted: a probe run, a judge run, a spot-check."""

    id: str
    status: Literal["cancelling"]


# ── probe dataset views: the fixed objects inside `ProbeDatasetResponse` ───────────────────────────────────────────


class ProbeDatasetCounts(BaseModel):
    """`probe_monitor_datasets.counts` (`probe_monitor_service.describe_counts`): the five numbers plus the
    input-shape histogram."""

    positive: int
    negative: int
    excluded: int
    filtered_out: int
    unparseable: int
    kinds: dict[str, int] = Field(description="Rows per input shape (`plain`, chat kinds, ...): a map")


class KeywordFilterRecord(BaseModel):
    """The stored `keyword_filter` (`KeywordFilterSpec.model_dump()`)."""

    terms: list[str]
    mode: str = Field(description="any or all")
    case_sensitive: bool


class DatasetVersionRecord(BaseModel):
    """`dataset_version_manifest.ManifestFacts.record`: what a miDataworks manifest verified at registration."""

    kind: str
    version_id: Optional[str]
    version_number: Optional[int]
    dataset_name: Optional[str]
    split: Optional[str]
    manifest_rows: Optional[int]
    downloaded_rows: Optional[int]
    repository: Optional[str]
    commit: Optional[str]
    publication_state: Optional[str]
    manifest_sha256: str
    sensitivity: Optional[str] = Field(description="044 FR-7: recorded, read by nothing yet")
    wsp_role: Optional[str]
    wsp_set_kind: Optional[str]
    recorded_at: str


class DatasetVersionSummary(BaseModel):
    """`DATASET_VERSION_SUMMARY_KEYS` of the record, each only when the record has it (`ProbeDatasetSummary.of`)."""

    version_id: Optional[str] = may_be_absent()
    version_number: Optional[int] = may_be_absent()
    dataset_name: Optional[str] = may_be_absent()
    split: Optional[str] = may_be_absent()
    publication_state: Optional[str] = may_be_absent()


class PositiveLabelsConfirmation(BaseModel):
    """`behaviour_catalogue.confirmation_record` (D48): who confirmed that the positives ARE the behaviour."""

    key: str
    version: Optional[int]
    by: str
    at: str
    origin: str


class ColumnSnapshot(BaseModel):
    """One entry of `column_snapshots` (`view_columns._write_column`)."""

    path: str
    sha256: str
    n_rows: int
    n_values: int
    dtype: str


class RuleColumnsRecord(BaseModel):
    """The stored `rule_columns` (`RuleColumns.model_dump()`)."""

    rule_match: str
    rule_veto: Optional[str]
    rule_set_id: str
    rule_set_digest: str


class EvidenceSpansProfile(BaseModel):
    """`probe_rule_recommendation.profile_spans`: where the evidence sits in the view's rows."""

    column: str
    unit: str = Field(description="code_points")
    positives: int
    positives_with_span: int
    negatives: int
    negatives_with_span: int
    span_chars: dict[str, Optional[float]] = Field(description="median, p75, p90")
    ratio: dict[str, Optional[float]] = Field(description="median, p75, p90")
    position: dict[str, Optional[float]] = Field(description="p10, median, p90")
    spans_per_row: dict[str, Optional[float]] = Field(description="median, max")
    row_chars: dict[str, Optional[float]] = Field(description="p10, median, p90")
    length_spread: Optional[float]


class ProbeDatasetAnswer(ProbeDatasetResponse):
    """One probe dataset view (`ProbeDatasetResponse`): create, get, archive, unarchive, and a full page item."""

    dataset_id: str = Field(description="The downloaded dataset the view reads (a UUID)")
    keyword_filter: Optional[KeywordFilterRecord] = history_form()
    counts: ProbeDatasetCounts = history_form()
    dataset_version: Optional[DatasetVersionRecord] = history_form(default=None)
    positive_labels_confirmed: Optional[PositiveLabelsConfirmation] = history_form(default=None)
    column_snapshots: Optional[dict[str, ColumnSnapshot]] = history_form(
        default=None, description="Per snapshotted column (a map keyed by column name)")
    rule_columns: Optional[RuleColumnsRecord] = history_form(default=None)
    rule_set_listing: Optional[dict[str, Any]] = Field(
        default=None, description="The manifest's miDataworks `RuleSetListing`, as sent (a map)")
    evidence_spans: Optional[EvidenceSpansProfile] = history_form(default=None)


class ProbeDatasetSummaryAnswer(ProbeDatasetSummary):
    """One view in a `detail=summary` page (`ProbeDatasetSummary.of`)."""

    dataset_id: str
    counts: ProbeDatasetCounts = history_form()
    dataset_version: Optional[DatasetVersionSummary] = None


class ProbeDatasetPageAnswer(ProbeDatasetPage):
    """`GET /probe-monitors/datasets-page`: summaries, or with `detail=full` whole views."""

    items: list[Union[ProbeDatasetSummaryAnswer, ProbeDatasetAnswer]]


# ── nonsense text ───────────────────────────────────────────────────────────────────────────────────────────────


class NonsenseDatasetRecord(BaseModel):
    id: str
    name: str
    raw_path: str
    num_samples: int
    size_bytes: Optional[int]


class NonsenseTextManifest(BaseModel):
    """`nonsense_text.NonsenseSet.manifest`: what made the set (gap G-2)."""

    generator: str
    seed: int
    rows: int
    kinds: dict[str, int]
    skipped: dict[str, Any] = Field(description="Kinds skipped, with why (a map)")
    prose_dataset_id: Optional[str]
    prose_column: Optional[str]
    prose_rows: Optional[int]
    exclude_view_ids: list[str]
    dropped_by_exclusion: int
    dropped_by_exclusion_kinds: dict[str, int]


class NonsenseTextCreated(BaseModel):
    """`POST /probe-monitors/nonsense-text`: the dataset, its view (null with `register_view: false`) and manifest."""

    dataset: NonsenseDatasetRecord
    view: Optional[ProbeDatasetAnswer]
    manifest: NonsenseTextManifest
    next: str = Field(description="The next step, in words")


# ── a view's rows ───────────────────────────────────────────────────────────────────────────────────────────────


class ProbeDatasetRow(BaseModel):
    position: int
    outcome: str = Field(description="positive, negative, excluded, filtered_out or unparseable")
    mapped_label: Optional[str]
    raw_label: JsonCell
    input_kind: Optional[str]
    values: dict[str, JsonCell] = Field(description="Each returned column's value (a map keyed by column)")
    truncated: list[str] = Field(description="The columns whose value was cut at the cell cap")


class ProbeDatasetRowsPage(BaseModel):
    """`GET /probe-monitors/datasets/{id}/rows` (`probe_dataset_rows.browse` plus the route's view facts)."""

    view_id: str
    split: Optional[str]
    input_column: str
    label_column: str
    pair_column: Optional[str]
    other_columns: list[str]
    returned_columns: list[str]
    total_rows: int
    matched: int
    offset: int
    limit: int
    label: Optional[str]
    search: Optional[str]
    recount: Optional[dict[str, int]] = Field(description="The five outcome counts, when a label filter recounted")
    rows: list[ProbeDatasetRow]
    name: str
    role: str
    dataset_name: str
    counts: dict[str, Optional[int]] = Field(description="The stored five counts (null where the view has none)")
    counts_agree: Optional[bool]


# ── the rule recommendation ─────────────────────────────────────────────────────────────────────────────────────


class RuleRecommendationEvidence(BaseModel):
    """The numbers a recommendation is read from (`probe_rule_recommendation.recommend`)."""

    span_column: Optional[str]
    positives: int
    positives_with_span: int
    span_coverage: float
    evidence_share_median: Optional[float]
    evidence_share_p90: Optional[float]
    span_chars: Optional[dict[str, Any]]
    span_tokens: Optional[dict[str, Any]] = Field(description="n, median, p75, p90 in one model's tokens")
    position: Optional[dict[str, Any]]
    row_chars: Optional[dict[str, Any]]
    length_spread: Optional[float]
    negatives: int
    surface_only_negatives: int
    surface_only_negative_share: float


class RuleRecommendationAnswer(BaseModel):
    """`GET /probe-monitors/datasets/{id}/rule-recommendation` (`recommendation_for`)."""

    view_id: str
    model_id: Optional[str]
    thresholds: dict[str, float] = Field(description="The cuts the evidence is read against (a map)")
    recorded: bool
    shape: Optional[str] = Field(description="local, whole_text, mixed or unmeasured; null when nothing is recorded")
    rules: list[str]
    rolling_windows: Optional[list[int]]
    reason: str
    evidence: Optional[RuleRecommendationEvidence]
    cautions: list[str]


class RunRuleRecommendation(RuleRecommendationAnswer):
    """The forecast's `rule_recommendation`: the recommendation and the refusal `POST /runs` would give."""

    refusal: Optional[dict[str, Any]] = Field(
        description="`rule_choice_refusal`'s detail (code, message, missing, recommendation); null when none")


# ── runs ────────────────────────────────────────────────────────────────────────────────────────────────────────


class ProbeTimeForecast(BaseModel):
    """`probe_run_time_forecast.time_forecast_for_request`: how long, from comparable completed runs."""

    forecast: bool
    summary: str
    reason: Optional[str] = may_be_absent(description="Why there is no forecast")
    eval_rows: Optional[int] = may_be_absent()
    eval_rows_by_set: Optional[dict[str, Any]] = may_be_absent()
    eval_max_rows: Optional[int] = may_be_absent()
    complete: Optional[bool] = may_be_absent()
    seconds: Optional[dict[str, float]] = may_be_absent(description="low, typical, high")
    hours: Optional[dict[str, float]] = may_be_absent(description="low, typical, high")
    missing_stages: Optional[list[str]] = may_be_absent()
    least_certain: Optional[dict[str, Any]] = may_be_absent()
    caveats: Optional[list[str]] = may_be_absent()
    stages: Optional[dict[str, Any]] = may_be_absent(description="Per stage: unit, units, seconds, rates (a map)")
    training_device: Optional[dict[str, Any]] = may_be_absent()
    work: Optional[dict[str, Any]] = may_be_absent()
    basis: Optional[dict[str, Any]] = may_be_absent()


class ProbeRunForecast(BaseModel):
    """`POST /probe-monitors/runs/forecast`: the host-memory forecast (`probe_run_memory.forecast_for_request`), the
    time forecast and the training view's rule recommendation. `checked: false` carries only `reason`."""

    checked: bool
    fits: Optional[bool]
    reason: Optional[str] = may_be_absent(description="Why the forecast could not be made (checked: false)")
    based_on: Optional[str] = may_be_absent()
    needed_gib: Optional[float] = may_be_absent()
    needed_bytes: Optional[int] = may_be_absent()
    usable_gib: Optional[float] = may_be_absent()
    usable_bytes: Optional[int] = may_be_absent()
    available_gib: Optional[float] = may_be_absent()
    reserved_gib: Optional[float] = may_be_absent()
    worker_held_gib: Optional[float] = may_be_absent()
    worker_held_bytes: Optional[int] = may_be_absent()
    additional_gib: Optional[float] = may_be_absent()
    additional_bytes: Optional[int] = may_be_absent()
    batches_gib: Optional[float] = may_be_absent()
    batches_bytes: Optional[int] = may_be_absent()
    batches_on_card: Optional[bool] = may_be_absent()
    batches_element_bytes: Optional[int] = may_be_absent()
    batches_held: Optional[str] = may_be_absent()
    batches_held_reason: Optional[str] = may_be_absent()
    float32_batches_bytes: Optional[int] = may_be_absent()
    storage_bytes: Optional[int] = may_be_absent()
    batch_room_bytes: Optional[int] = may_be_absent()
    train_batches_gib: Optional[float] = may_be_absent()
    validation_batches_gib: Optional[float] = may_be_absent()
    pooled_gib: Optional[float] = may_be_absent()
    pooled_bytes: Optional[int] = may_be_absent()
    token_file_per_layer_gib: Optional[float] = may_be_absent()
    token_file_per_layer_bytes: Optional[int] = may_be_absent()
    rows: Optional[int] = may_be_absent()
    real_tokens: Optional[int] = may_be_absent()
    padded_slots: Optional[int] = may_be_absent()
    d_model: Optional[int] = may_be_absent()
    swept_layers: Optional[int] = may_be_absent()
    top_n_layers: Optional[int] = may_be_absent()
    by_top_n: Optional[dict[str, dict[str, float]]] = may_be_absent(
        description="Per top-n (a map keyed by n): needed, token files, resident if nothing is evicted, in GiB")
    arithmetic: Optional[str] = may_be_absent()
    message: Optional[str] = may_be_absent()
    template_prefix_skipped: Optional[int] = may_be_absent(
        description="G38: template-prefix positions the run skips; only when the rows were rendered with the skip")
    time_forecast: ProbeTimeForecast
    rule_recommendation: RunRuleRecommendation


class ProbeRunQueued(BaseModel):
    """`POST /probe-monitors/runs`: the row's id and the Celery task id."""

    id: str
    task_id: str
    status: Literal["queued"]


class CalibrationSizeAnswer(BaseModel):
    """`calibration_size.calibration_size`: the calibration's size against its floor (FR-19)."""

    view_id: Optional[str]
    threshold_source: Optional[str]
    n_negatives: Optional[int]
    n_negatives_source: Optional[str] = Field(description="recorded or array_length; null when unknown")
    floor: int
    floor_source: str = Field(description="behaviour or setting")
    few_negatives: Optional[bool]
    target_fpr: Optional[float]
    allowed_false_positives: Optional[int]
    sentence: Optional[str]


class ProbeRunAnswer(ProbeRunResponse):
    """One probe run (`ProbeRunResponse`): `GET /runs/{id}` and a `detail=full` page item."""

    layer_selection: Optional[dict[str, Any]] = Field(
        description="The layer sweep (`LayerSelection.as_dict`: grid, chosen, margin, poolings), stored JSON")
    calibration_size: Optional[CalibrationSizeAnswer] = None


class ProbeRunPageAnswer(ProbeRunPage):
    """`GET /probe-monitors/runs-page`: summaries, or with `detail=full` whole runs."""

    items: list[Union[ProbeRunSummary, ProbeRunAnswer]]


# ── evaluations ─────────────────────────────────────────────────────────────────────────────────────────────────


class ProbeEvaluationRowAnswer(ProbeEvaluationRow):
    text: JsonCell = Field(description="The view's input at `position`: a string, or a conversation as stored")


class ProbeEvaluationRowsAnswer(ProbeEvaluationRows):
    """`GET /probes/{id}/evaluations/{eid}/rows` (`ProbeEvaluationRows`), with each row's text typed."""

    rows: list[ProbeEvaluationRowAnswer]


class TruncationBackfillAnswer(BaseModel):
    """`POST /probes/{id}/evaluations/{eid}/backfill-truncation` (`probe_monitor_run.backfill_truncation`)."""

    evaluation_id: str
    probe_id: str
    status: Literal["already_recorded", "backfilled"]
    truncated_path: Optional[str] = may_be_absent(description="already_recorded: the mask already on disk")
    rows: Optional[int] = may_be_absent(description="backfilled: the rows matched")
    truncated: Optional[int] = may_be_absent(description="backfilled: the rows cut at max_length")
    truncation: Optional[dict[str, Any]] = may_be_absent(description="backfilled: the evaluation's truncation block")
    rung: Optional[dict[str, Any]] = may_be_absent(description="backfilled: `recompute_rung`'s outcome")


# ── recalibration ───────────────────────────────────────────────────────────────────────────────────────────────


class BarState(BaseModel):
    """A bar: the one in force (`current`), the one before a commit (`previous`)."""

    threshold: Optional[float]
    target_fpr: Optional[float]
    realised_fpr: Optional[float]
    threshold_source: Optional[str]
    revision: Optional[int]
    decision_digest: Optional[str]
    decision_revision: Optional[int]


class ProposedBar(BarState):
    """The candidate bar (`probe_recalibration.propose`)."""

    n_negatives: int
    truncated_excluded: Optional[int] = Field(description="Negatives left out as truncated; null = no mask (pre-G5)")
    fires_on_nothing: bool


class TransferSet(BaseModel):
    """One evaluation set in `threshold_transfer`."""

    name: Optional[str]
    auroc: Optional[float]
    own_threshold_at_1pct: Optional[float]
    max_score: Optional[float]
    recall_at_shipped: Optional[float] = may_be_absent(description="Only when there is a bar and a stored ROC")
    fpr_at_shipped: Optional[float] = may_be_absent()
    unreachable: Optional[bool] = may_be_absent()


class ThresholdTransferAnswer(BaseModel):
    """`probe_monitor_metrics.threshold_transfer`: how a bar behaves on each evaluated set."""

    shipped_threshold: Optional[float]
    threshold_source: Optional[str]
    per_set: list[TransferSet]
    own_threshold_spread: Optional[float]
    unreachable_sets: list[Optional[str]]
    caution: Optional[str]


class RecalibrationAnswer(BaseModel):
    """`probe_recalibration.recalibrate_probe`: a preview, an unchanged re-cut, or a commit."""

    probe_id: str
    current: BarState
    proposed: ProposedBar
    detector_digest: Optional[str]
    detector_unchanged: bool
    unchanged: bool
    window_decisions: Optional[dict[str, Any]] = Field(description="Per window, its re-cut bar (a map); null = none")
    length_bands: Optional[list[dict[str, Any]]]
    transfer_current: ThresholdTransferAnswer
    transfer_proposed: ThresholdTransferAnswer
    published_copies_go_stale: bool
    definition_rebuildable: bool
    definition_rebuild_refusal: Optional[str]
    render_note: Optional[str]
    commit_ends_exportability: bool
    committed: bool
    definition_invalidated: bool
    reason: Optional[str] = may_be_absent(description="unchanged: why nothing was committed")
    bookkeeping_updated: Optional[bool] = may_be_absent(description="unchanged: the GPU arm's arrays were kept")
    previous: Optional[BarState] = may_be_absent(description="committed: the bar before")
    millm_copies: Optional[Union[list[str], dict[str, str]]] = may_be_absent(
        description="committed: miServer's imported copies' ids, or {not_configured|unknown: reason}")
    next: Optional[list[ProbeNextCall]] = may_be_absent(description="unchanged and committed: the follow-up calls")


class RecalibrateAnswer(RootModel[Union[RecalibrationAnswer, ProbeTaskQueued]]):
    """`POST /probes/{id}/recalibrate` (200): the arithmetic re-cut, or with `recut_windows=true` the GPU re-cut queued."""


# ── scoring ─────────────────────────────────────────────────────────────────────────────────────────────────────


class ScoreTaskRef(BaseModel):
    index: int
    task_id: str


class ProbeScoreBatchQueued(BaseModel):
    """`POST /probes/{id}/score-batch`: one task per input."""

    probe_id: str
    tasks: list[ScoreTaskRef]
    submitted: int


class ProbeScoreResult(BaseModel):
    """`GET /probes/{id}/score/{task_id}`: the task's state and, on SUCCESS, its result."""

    task_id: str
    probe_id: str
    status: str = Field(description="The Celery state: PENDING, STARTED, SUCCESS, FAILURE, ...")
    result: Optional[dict[str, Any]] = Field(description="The score task's result (SUCCESS); null otherwise")
    error: Optional[str] = may_be_absent(description="FAILURE: the reason")


# ── judge runs ──────────────────────────────────────────────────────────────────────────────────────────────────


class JudgeRunQueued(BaseModel):
    """`POST /probe-monitors/judge-runs`."""

    id: str
    task_id: str
    status: Literal["queued"]
    prompt_id: str
    prompt_version: str
    prompt_hash: Optional[str]
    counts_toward_rung: Optional[bool]
    rung_note: Optional[str]


class JudgeRunAnswer(BaseModel):
    """One judge run (`probe_monitors.judge_run_entries`, with `judge_prompt_library.run_rung_view`)."""

    id: str
    endpoint: str = Field(description="The served model's URL")
    model: str
    prompt_version: str
    dataset_ids: list[str]
    probe_id: Optional[str]
    status: str
    progress: Optional[float]
    progress_detail: Optional[dict[str, Any]] = Field(description="The live reading (`JudgeProgress`), stored JSON")
    parse_failures: int
    max_rows_per_set: Optional[int]
    metrics: Optional[dict[str, Any]] = Field(description="The run's metrics, stored JSON (a map)")
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    prompt_id: Optional[str]
    prompt_hash: Optional[str]
    prompt_title: Optional[str]
    prompt_behaviour: Optional[str]
    counts_toward_rung: Optional[bool]
    rung_note: Optional[str]


class JudgeRunListAnswer(RootModel[list[JudgeRunAnswer]]):
    """`GET /probe-monitors/judge-runs`: a JSON list, newest first."""


# ── definitions ─────────────────────────────────────────────────────────────────────────────────────────────────


class DefinitionBuildQueued(BaseModel):
    """`POST /probes/{id}/definition`: the build queued, with the task-queue row `get_task_status` reads."""

    probe_id: str
    task_id: str
    task_queue_id: str
    status: Literal["queued"]


# ── spot-checks (probe_spot_checks.py) ──────────────────────────────────────────────────────────────────────────


class SpotCheckAvailability(BaseModel):
    """`GET /spot-checks/availability` (`miserver_read.spot_check_availability`)."""

    available: bool
    url: Optional[str]
    reason: Optional[str]


class SpotCheckStarted(BaseModel):
    """`POST /spot-checks`."""

    id: str
    status: str


class SpotCheckAnswer(BaseModel):
    """One spot-check (`probe_spot_check.check_out`)."""

    id: str
    behaviour: str
    behaviour_version: int
    prompt_id: str
    prompt_key: str
    prompt_hash: str
    endpoint: str = Field(description="The judge's URL")
    model: str
    sample_spec: dict[str, Any] = Field(description="What was asked for (seed, strata, window, probes), stored JSON")
    sample_returned: Optional[dict[str, Any]] = Field(description="What miServer returned, stored JSON")
    status: str
    progress: Optional[float]
    error_message: Optional[str]
    partial: bool
    parse_failures: int
    parse_failure_limit: float
    metrics: Optional[dict[str, Any]] = Field(description="The report, stored JSON")
    pinned: bool
    excerpts_expire_at: Optional[str]
    created_by: str
    created_at: Optional[str]
    completed_at: Optional[str]


class SpotCheckListAnswer(BaseModel):
    spot_checks: list[SpotCheckAnswer]


# ── rule sets and rules evaluations (rule_sets.py) ──────────────────────────────────────────────────────────────


class RuleSetSource(BaseModel):
    """Where the rule set came from (`rules_evaluation._choice`)."""

    kind: Literal["inline", "behaviour"]
    behaviour_key: Optional[str] = may_be_absent()
    version: Optional[int] = may_be_absent()


class RuleSetValidation(BaseModel):
    """`POST /rule-sets/validate` (`rules_evaluation.validate_request`). A set that fails is `valid: false` with
    `code`, `message` and the refusal's own details spread at the top level, so this object admits extra keys."""

    model_config = ConfigDict(extra="allow")

    rule_set_digest: str
    source: RuleSetSource
    engine_version: int
    engine_digest: str
    valid: bool
    code: Optional[str] = may_be_absent(description="valid: false: the refusal's code")
    message: Optional[str] = may_be_absent(description="valid: false")
    next: Optional[list[ProbeNextCall]] = may_be_absent(description="valid: false, when the refusal names a call")
    cases: Optional[dict[str, Any]] = may_be_absent(description="valid: passed and unverified")
    match_expressions: Optional[list[str]] = may_be_absent()
    match_source: Optional[str] = may_be_absent()
    has_vetoes: Optional[bool] = may_be_absent()
    pending_expressions: Optional[list[str]] = may_be_absent()
    provenance: Optional[dict[str, Any]] = may_be_absent(description="A derived set's recorded provenance")
    runner: Optional[str] = may_be_absent(description="With code rules: available or unavailable")
    code_sha256: Optional[Union[list[str], str]] = may_be_absent(
        description="Code rules, no runner: the hashes named; valid: false: the one hash a refusal names")
    state: Optional[str] = may_be_absent(description="valid: false (code_rule_not_approved): the hash's state")
    code_rules: Optional[dict[str, dict[str, Any]]] = may_be_absent(description="Code rules, a runner: per hash")


class RuleSignalsVerdict(BaseModel):
    """`POST /rule-signals/evaluate` (`rules_evaluation.evaluate_one`): `RuleVerdict` plus the set's identity."""

    rule_match: bool
    rule_veto: bool
    unknown: bool
    reason: Optional[str]
    fired: list[dict[str, Any]] = Field(description="The signals that fired, with their spans")
    expressions: dict[str, dict[str, Any]] = Field(description="Per expression id: fires, pending, count, form")
    vetoed: Optional[bool]
    missing: Optional[str]
    code_points: int
    code: dict[str, dict[str, Any]] = Field(description="Per code_rule signal: {count} or {reason}")
    rule_set_digest: str
    engine_digest: str
    scope: str
    source: RuleSetSource
    has_vetoes: bool
    match_expressions: list[str]
    match_source: str
    notes: Optional[list[str]] = may_be_absent(description="no_veto_signals, match_includes_veto_condition")
    runner: Optional[str] = may_be_absent()
    code_sha256: Optional[list[str]] = may_be_absent()
    code_rules: Optional[dict[str, dict[str, Any]]] = may_be_absent()


class RulesEvaluationRow(BaseModel):
    """One rules evaluation's row (`rules_evaluation._row_out`)."""

    evaluation_id: str
    status: str
    probe_id: str
    view_ids: list[str]
    behaviour_key: str
    behaviour_version: int
    rule_set_digest: Optional[str]
    engine_digest: str
    calibration: bool
    task_queue_id: Optional[str]
    progress: Optional[dict[str, Any]] = Field(description="The live reading, stored JSON")
    error: Optional[dict[str, Any]] = Field(description="code and message, stored JSON")
    created_by: str
    created_at: Optional[str]
    completed_at: Optional[str]
    body_sha256: Optional[str]


class RulesEvaluationRequested(RulesEvaluationRow):
    """`POST /probes/{id}/rules-evaluations`: 202 new or in flight, 200 a completed duplicate."""

    duplicate: bool
    next: list[ProbeNextCall]


class RulesEvaluationStaleness(BaseModel):
    """`rules_evaluation.staleness` (FR-38): which input moved since the result was computed."""

    stale: bool
    moved: list[dict[str, Any]] = Field(description="input, was, now (and vectors_now for the engine)")
    next: list[ProbeNextCall]


class RulesEvaluationAnswer(RulesEvaluationRow):
    """`GET /rules-evaluations/{id}` (`rules_evaluation.read`)."""

    result: Optional[dict[str, Any]] = Field(description="The stored result, byte for byte, once completed")
    stale: RulesEvaluationStaleness


class RulesEvaluationList(BaseModel):
    """`GET /behaviours/{key}/rules-evaluations` (`rules_evaluation.list_for`)."""

    behaviour_key: str
    rules_evaluations: list[RulesEvaluationRow]
    total: int
    next_cursor: Optional[str]


class RulesEvaluationCancel(BaseModel):
    """`POST /rules-evaluations/{id}/cancel` (`rules_evaluation.cancel`, `CancelOutcome`)."""

    evaluation_id: str
    accepted: bool
    was_running: bool
    message: str
    prior_status: Optional[str]


# ── code rules (code_rules.py) ──────────────────────────────────────────────────────────────────────────────────


class CodeRuleApprovalRequested(BaseModel):
    """`POST /code-rules/{sha}/approval-requests` (202): filed, already pending (`duplicate`), or already approved."""

    code_sha256: str
    status: Literal["approved", "pending_approval"]
    approval_request_id: Optional[str] = may_be_absent(description="pending_approval: the request to poll")
    duplicate: Optional[bool] = may_be_absent(description="True when a request for this hash was already pending")
    state: Optional[str] = may_be_absent(description="approved: the hash's state (`code_rules.states`)")
    approval_row_id: Optional[int] = may_be_absent(description="approved: the approval row")
