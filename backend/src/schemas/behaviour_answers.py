"""The answers of the behaviour catalogue and behaviour-detector routes (046 FTASKS 8.13; FTID §3.5):
`behaviours.py` and `behaviour_detectors.py`.

Documentation only (FR-046.5 as amended 2026-11-26): each route keeps its serializer; these models are named in its
`responses=`. A key one branch writes and another does not is `may_be_absent()`. A stored free-form JSON value (a
history row's old and new values, a stored report, a suggested blend's record, a rule set's validation) is a declared
map, `dict[str, Any]`, or a JSON value (`JsonCell`) where it need not be an object.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .answer_fields import JsonCell, may_be_absent

#: The `$id` of `docs/schemas/behaviour-verdict-v1.json`. `POST /behaviours/{key}/behaviour-verdict/export` answers
#: that document WRAPPED (`{definition, definition_sha256, bytes}`), so only its `definition` field references it.
BEHAVIOUR_VERDICT_ID = "https://hitsai.net/schemas/mistudio/behaviour-verdict-v1.json"


# ── shared pieces ───────────────────────────────────────────────────────────────────────────────────────────────


class BehaviourNextCall(BaseModel):
    """One suggested call (`detector_next._call`, `next_calls.build_probe_definition`)."""

    tool: str
    arguments: dict[str, Any]
    why: str
    needs: list[str] = may_be_absent(description="Arguments the caller must supply itself (e.g. an acknowledgement)")


class DefinitionIdentity(BaseModel):
    """The identity of a probe's built definition (`probe_decision.identity_fields`, the six `IDENTITY_KEYS`)."""

    detector_digest: Optional[str]
    decision_digest: Optional[str]
    decision_revision: Optional[int]
    identity_current: Optional[bool] = Field(description="Whether the built definition states the current decision; "
                                                         "null when it cannot be said")
    identity_stale_reason: Optional[str]
    identity_source: Optional[str] = Field(description="built, recomputed or not_recorded; null when never built")


# ── behaviours.py: the catalogue row (`behaviour_catalogue.head_out`, `version_out`, `detail`) ─────────────────


class BehaviourRuleSet(BaseModel):
    """A version's rule set (`rule_set_store.describe`): identity always; provenance and validation when the stored
    row is read (`detail` reads it; a list row and `versions` do not)."""

    digest: str
    digest_scheme: str
    source: Optional[str] = may_be_absent(description="Absent unless the stored rule-set row was read")
    source_ref: Optional[dict[str, Any]] = may_be_absent(
        description="Where a fetched set came from (stored JSON, a map: name, digest, version); null for a set given "
                    "inline; absent unless the stored rule-set row was read")
    behaviour_key: Optional[str] = may_be_absent(description="Absent unless the stored rule-set row was read")
    validation: dict[str, Any] = may_be_absent(
        description="This engine's validation of the set, computed on read (`validate_for_use`, or "
                    "{status: invalid, code, ...}); absent unless the stored row was read")


class BehaviourVersionAnswer(BaseModel):
    """One version of a behaviour's definition (`behaviour_catalogue.version_out`): `GET .../versions` lists them and
    every catalogue row carries the current one as `definition`. A detector field a version does not declare (null in
    the row) is ABSENT, so a version written before Feature 040 reads as it did."""

    version: int
    name: str
    description: Optional[str]
    positive_meaning: str
    negative_meaning: Optional[str]
    target_fpr: Optional[float]
    created_by: str
    created_at: str
    sub_questions: list[dict[str, Any]] = may_be_absent(description="Stored JSON; absent when not declared")
    success_criteria: list[dict[str, Any]] = may_be_absent(description="Stored JSON; absent when not declared")
    held_out_columns: list[str] = may_be_absent(description="Absent when not declared")
    models: list[dict[str, Any]] = may_be_absent(description="Stored JSON; absent when not declared")
    calibration_floor: int = may_be_absent(description="Absent when not declared")
    rule_set: BehaviourRuleSet = may_be_absent(description="Absent when the version names no rule set")


class BehaviourCounts(BaseModel):
    """How many records of each kind use the key (`usage_counts`, or `detail`'s count of `adopted`)."""

    prompts: int
    runs: int
    probes: int
    views: int


class BehaviourHead(BaseModel):
    """The fields every catalogue row has (`behaviour_catalogue.head_out`)."""

    key: str
    catalogued: bool = Field(description="Always true on a catalogue row")
    current_version: int
    archived: bool
    archived_at: Optional[str]
    archived_by: Optional[str]
    created_by: str
    created_at: str
    updated_at: str
    definition: BehaviourVersionAnswer = Field(description="The current version")


class BehaviourListRow(BehaviourHead):
    """One catalogued behaviour in `GET /behaviours` (`rows_for` plus the computed status)."""

    counts: BehaviourCounts
    status: str = Field(description="draft, validated, monitoring, unknown or archived, computed on read")


class UncataloguedKey(BaseModel):
    """A key in use with no catalogue row (`uncatalogued_page`)."""

    key: str
    catalogued: bool = Field(description="Always false")
    prompts: int
    runs: int
    probes: int
    views: int


class BehaviourListAnswer(BaseModel):
    """`GET /behaviours`: one page of the catalogue and of the uncatalogued keys."""

    behaviours: list[BehaviourListRow]
    uncatalogued: list[UncataloguedKey]
    next_cursor: Optional[str]
    total: Optional[int] = Field(description="Null when a status-filtered scan cannot say")
    total_exact: bool
    uncatalogued_total: int
    limit: int


class AdoptedPrompt(BaseModel):
    id: str
    key: str = Field(description="name/vN")
    title: str
    locked: bool
    archived: bool


class AdoptedRun(BaseModel):
    id: str
    status: str
    model_id: str
    train_dataset_id: str
    created_at: str


class AdoptedProbe(DefinitionIdentity):
    """A probe carrying the key (`adopted`), with its built definition's identity."""

    id: str
    run_id: str
    layer: int
    rule: str
    variant: str
    rung: int
    behaviour_label: Optional[str]
    definition_sha256: Optional[str]
    role: str = Field(description="primary, candidate or retired (candidate when none is recorded)")
    training_view_confirmed: bool
    created_at: str


class ViewDatasetVersion(BaseModel):
    """The miDataworks facts of a view registered from a dataset-version manifest."""

    version_id: Optional[str]
    version_number: Optional[int]
    dataset_name: Optional[str]
    split: Optional[str]


class BehaviourView(BaseModel):
    """A view that carries or links the key (`behaviour_catalogue.view_out`)."""

    id: str
    name: str
    role: str
    use: str
    linked_by: str = Field(description="key or link")
    behaviour: Optional[str]
    positive_meaning: Optional[str]
    positive_labels_confirmed: Optional[dict[str, Any]] = Field(description="The stored confirmation record")
    confirmed_for_this_behaviour: bool
    archived: bool
    dataset_version: Optional[ViewDatasetVersion]


class BehaviourAdopted(BaseModel):
    """Every record that uses the key (`behaviour_catalogue.adopted`)."""

    prompts: list[AdoptedPrompt]
    runs: list[AdoptedRun]
    probes: list[AdoptedProbe]
    views: list[BehaviourView]


class BehaviourAnswer(BehaviourHead):
    """One behaviour in full (`behaviour_catalogue.detail`): get, set a probe's role, link and unlink a view,
    archive and unarchive all answer it."""

    adopted: BehaviourAdopted
    counts: BehaviourCounts


class DefinitionWarning(BaseModel):
    """Something the new version names that may not exist yet (`behaviour_detector_definition.warnings`)."""

    code: str = Field(description="model_not_downloaded or column_not_on_any_view")
    field: str
    message: str
    next: list[BehaviourNextCall]
    column: str = may_be_absent(description="Only on column_not_on_any_view")


class BehaviourCreatedAnswer(BehaviourAnswer):
    """`POST /behaviours`: the new behaviour, with the warnings its version raised."""

    warnings: list[DefinitionWarning]


class FieldDiff(BaseModel):
    path: str = Field(description="The definition field that changed")
    before: JsonCell
    after: JsonCell


class BehaviourVersionedAnswer(BehaviourCreatedAnswer):
    """`POST /behaviours/{key}/versions`: the behaviour at its new version, with the diff against the base."""

    base_version: int
    diff: list[FieldDiff]


class BehaviourVersionsAnswer(BaseModel):
    """`GET /behaviours/{key}/versions`, newest first."""

    key: str
    versions: list[BehaviourVersionAnswer]


class BehaviourHistoryEntry(BaseModel):
    id: int
    subject_kind: str
    subject_id: Optional[str]
    field: str
    old_value: JsonCell = Field(description="Stored JSON, any shape an earlier release wrote")
    new_value: JsonCell = Field(description="Stored JSON, any shape an earlier release wrote")
    actor: str
    approval_id: Optional[str]
    reason: Optional[str]
    created_at: str


class BehaviourHistoryAnswer(BaseModel):
    """`GET /behaviours/{key}/history` (`behaviour_catalogue.history`), newest first."""

    key: str
    total: int
    entries: list[BehaviourHistoryEntry]


# ── behaviours.py: the checklist (`behaviour_checklist.checklist`) ──────────────────────────────────────────────


class ChecklistMember(BaseModel):
    probe_id: str
    model_id: str
    role: str
    state: str


class ChecklistModel(BaseModel):
    model_id: str
    members: int


class ChecklistGap(BaseModel):
    """A declared sub-question with no family member on a model."""

    model_id: str
    sub_question: str


class ChecklistIdentity(BaseModel):
    """The primary's built-definition identity on the `exported` step (`exported_step`)."""

    detector_digest: Optional[str]
    decision_digest: Optional[str]
    decision_revision: Optional[int]
    identity_current: Optional[bool]


class ChecklistStep(BaseModel):
    """One step (`behaviour_checklist._step`), plus the keys some steps add."""

    id: str
    label: str
    state: str = Field(description="done, not_done, partly_done, stale or unknown")
    reason: str
    evidence: list[str]
    link: dict[str, str] = Field(description="Where the step is done: {tab} in miWorkbench or {app}")
    next: list[BehaviourNextCall] = may_be_absent(description="Only on a step that is not done and has a remedy")
    members: list[ChecklistMember] = may_be_absent(
        description="Only on exported and armed, when the family has members")
    models: list[ChecklistModel] = may_be_absent(
        description="Only on exported and armed (not on armed when miServer cannot say)")
    identity: ChecklistIdentity = may_be_absent(description="Only on exported, when the primary's definition is built")
    identity_source: Optional[str] = may_be_absent(
        description="Only on exported, when the primary's definition is built or invalidated")
    missing: list[ChecklistGap] = may_be_absent(description="Only on family_trained with gaps")
    verdict: str = may_be_absent(description="Only on success_report when a report exists")
    behaviour_version: int = may_be_absent(description="Only on success_report when a report exists")


class BehaviourChecklistAnswer(BaseModel):
    """`GET /behaviours/{key}/checklist`: every step from definition to armed, and the computed status."""

    key: str
    steps: list[ChecklistStep]
    status: str = Field(description="draft, validated, monitoring, unknown or archived")


# ── behaviours.py: the probe family (`probe_family.family`) ─────────────────────────────────────────────────────


class FamilyModel(BaseModel):
    model_id: str
    members: int


class FamilyCoverage(BaseModel):
    """A declared sub-question's members, or the `main_question` group (`probe_family.coverage`)."""

    sub_question: Optional[str] = Field(description="Null on the main_question group")
    state: str = Field(description="covered, missing or (main_question) empty")
    members: list[str]
    group: str = may_be_absent(description="main_question; only on the last entry")
    next: list[BehaviourNextCall] = may_be_absent(description="Only on a missing sub-question")


class MemberDecision(BaseModel):
    """The decision in force, as `decide_many` takes it (`probe_breakdown.decision_inputs`)."""

    probe_bar: Optional[float]
    window_entry: Optional[dict[str, Any]] = Field(description="The stored window decision, when a window applies")
    probe_bands: JsonCell = Field(description="The probe's stored length bands, or null")
    probe_scope: str
    window: Optional[str]


class FamilyMember(BaseModel):
    """One member's facts (`probe_family.member_facts`), flags and default-blend mark."""

    probe_id: str
    run_id: str
    model_id: str
    role: str
    role_recorded: bool
    precision: Optional[str]
    precision_refusal: Optional[str]
    render_form: str
    scope: str
    layer: int
    rule: str
    variant: str
    sub_question: Optional[str]
    behaviour_version: Optional[int]
    rung: int
    rung_language: str
    threshold: Optional[float]
    decision_revision: Optional[int] = Field(description="The BUILT definition's revision")
    decision_digest: Optional[str]
    current_decision_revision: Optional[int] = Field(description="The revision of the decision in force")
    detector_digest: Optional[str]
    identity_current: Optional[bool]
    identity_stale_reason: Optional[str]
    training_view: str
    training_dataset_version: Optional[str]
    definition_built: bool
    created_at: Optional[str]
    decision: MemberDecision
    flags: list[dict[str, Any]] = Field(
        description="differs_* (flag, field, value, majority, majority_tie?), definition_stale (flag, reason) and "
                    "sub_question_not_declared (flag, sub_question, version)")
    default_blend: bool


class ProbeFamilyAnswer(BaseModel):
    """`GET /behaviours/{key}/family`: the family on one model, paged, with flags and coverage."""

    key: str
    version: int
    model: Optional[str] = Field(description="Null when the behaviour has no member on any model")
    models: list[FamilyModel]
    declared_sub_questions: list[str]
    coverage: list[FamilyCoverage]
    default_blend_set: list[str]
    flags_basis: str = Field(description="majority_of_all_members")
    members: list[FamilyMember]
    total: int
    offset: int
    limit: int
    next_offset: Optional[int]


class FamilyScoreTask(BaseModel):
    index: int = Field(description="The input's position in the request")
    task_id: str


class FamilyScoreQueued(BaseModel):
    """`POST /behaviours/{key}/family/score-batch`: one queued task per input."""

    key: str
    model: Optional[str]
    members: list[str]
    tasks: list[FamilyScoreTask]
    submitted: int


class FamilyScoreResult(BaseModel):
    """`GET /behaviours/{key}/family/score/{task_id}`: the task's state, and its result once it succeeded."""

    task_id: str
    key: str
    status: str = Field(description="The Celery state (PENDING, STARTED, SUCCESS, FAILURE, ...)")
    result: Optional[dict[str, Any]] = Field(description="The worker's result on SUCCESS; null otherwise")
    error: str = may_be_absent(description="Only on FAILURE")


class FamilyExportEntry(BaseModel):
    probe_id: str
    filename: str
    sha256: str
    size: int
    export_route: str


class FamilyExportIndex(BaseModel):
    """`GET /behaviours/{key}/family/export-index` (`probe_family_export.index`)."""

    key: str
    model: Optional[str]
    members: list[FamilyExportEntry]
    archive_route: str


class FamilyPublishQueued(BaseModel):
    """`POST /behaviours/{key}/family/publish`: the one Hugging Face commit queued."""

    key: str
    model: Optional[str]
    members: list[str]
    task_id: str
    status: str = Field(description="queued")


# ── behaviours.py: the suggested blend and the behaviour verdict ───────────────────────────────────────────────


class SuggestedBlendAnswer(BaseModel):
    """`GET` and `PUT /behaviours/{key}/suggested-blend` (`behaviours._suggested_envelope`)."""

    key: str
    model: str
    report_id: str
    candidate: str
    record: dict[str, Any] = Field(description="The stored advisory record, sliced from its report (stored JSON)")
    record_sha256: str
    history_id: str
    updated_at: Optional[str]
    stale: bool
    stale_reasons: list[dict[str, Any]] = Field(description="Each {code, ...}; empty when current")
    next: list[BehaviourNextCall]


class SuggestedBlendHistoryEntry(BaseModel):
    id: str
    report_id: str
    candidate: str
    record_sha256: str
    actor: str
    reason: Optional[str]
    created_at: Optional[str]


class SuggestedBlendHistoryAnswer(BaseModel):
    """`GET /behaviours/{key}/suggested-blend/history` (`probe_blend.service.suggested_history`)."""

    key: str
    model: str
    entries: list[SuggestedBlendHistoryEntry]
    offset: int
    limit: int
    next_offset: Optional[int]


class BehaviourVerdictExportAnswer(BaseModel):
    """`POST /behaviours/{key}/behaviour-verdict/export` (`behaviour_verdict.export`): the document, its sha256 and
    its length."""

    definition: dict[str, Any] = Field(
        description="The mistudio.behaviour-verdict/v1 document",
        json_schema_extra={"$ref": BEHAVIOUR_VERDICT_ID})
    definition_sha256: str = Field(description="sha256 of the document's canonical bytes")
    bytes: int = Field(description="The length of the canonical bytes")


# ── behaviour_detectors.py ──────────────────────────────────────────────────────────────────────────────────────


class ViewColumnsAnswer(BaseModel):
    """`PATCH /datasets/{dataset_id}/columns` (`view_columns.add`, `add_rule_fields`): each part answers only when
    the request sent it."""

    view_id: str
    added: list[str] = Field(description="Every column this request added")
    group_columns: list[str] = may_be_absent(description="Only when group_columns was sent")
    column_snapshots: dict[str, Any] = may_be_absent(description="Only when group_columns was sent; by column")
    text_column: Optional[str] = may_be_absent(description="Only when text_column or rule_columns was sent")
    rule_columns: Optional[dict[str, Any]] = may_be_absent(
        description="Only when text_column or rule_columns was sent (stored JSON)")


class BreakdownDecision(BaseModel):
    mode: str = Field(description="recorded or target_fpr")
    revision: Optional[int]
    scope: str
    window: Optional[str]
    rule_counts: Optional[dict[str, int]] = Field(description="Rows decided by each rule; null in target_fpr mode")
    target_fpr: Optional[float]
    threshold: Optional[float] = Field(description="The cut used in target_fpr mode; null in recorded mode")


class CalibrationSize(BaseModel):
    """`calibration_size.calibration_size`."""

    view_id: Optional[str]
    threshold_source: Optional[str]
    n_negatives: Optional[int]
    n_negatives_source: Optional[str]
    floor: int
    floor_source: str = Field(description="behaviour or setting")
    few_negatives: Optional[bool]
    target_fpr: Optional[float]
    allowed_false_positives: Optional[int]
    sentence: Optional[str]


class TruncationNote(BaseModel):
    """`probe_evaluation_scores.untruncated`'s note."""

    rows: str = Field(description="all or untruncated")
    excluded: Optional[int]
    reason: str = may_be_absent(description="Only when no truncation mask was recorded")
    policy: str = may_be_absent(description="Only when a truncation mask was recorded")


class BreakdownFilter(BaseModel):
    column: str
    value: str


class BreakdownPositives(BaseModel):
    n: int
    flagged: int
    recall: Optional[float] = Field(description="Null when n is 0")
    ci: Optional[list[float]] = Field(description="Wilson interval; null when n is 0")


class BreakdownNegatives(BaseModel):
    n: int
    flagged: int
    fpr: Optional[float] = Field(description="Null when n is 0")
    ci: Optional[list[float]] = Field(description="Wilson interval; null when n is 0")


class BreakdownGroup(BaseModel):
    """One subgroup (`probe_breakdown._groups`)."""

    value: JsonCell = Field(description="The column value")
    positives: BreakdownPositives
    negatives: BreakdownNegatives
    median_score: float
    rows: int
    too_small: bool


class BreakdownSettingsAnswer(BaseModel):
    """Every setting the breakdown read (`BreakdownSettingsAnswer.as_inputs`)."""

    breakdown_min_subgroup_rows: int
    breakdown_page_default: int
    breakdown_page_max: int
    breakdown_max_filters: int
    calibration_floor_negatives: int


class ProbeBreakdownAnswer(BaseModel):
    """`GET /probes/{probe_id}/evaluations/{evaluation_id}/breakdown` (`probe_breakdown.compute`)."""

    probe_id: str
    evaluation_id: str
    view_id: str
    decision: BreakdownDecision
    calibration: CalibrationSize
    column: str
    truncation: TruncationNote
    filters: list[BreakdownFilter]
    min_rows: int
    groups: list[BreakdownGroup]
    total: int
    next_cursor: Optional[str]
    inputs_digest: str
    settings: BreakdownSettingsAnswer


class ReportMove(BaseModel):
    input: str
    was: JsonCell
    now: JsonCell


class ReportStaleness(BaseModel):
    """`detector_success_report.staleness`, computed on read."""

    stale: bool
    moved: list[ReportMove]
    inputs_removed: dict[str, list[str]] = Field(description="Cited ids that are gone, by kind (probes, evaluations, "
                                                              "views); kinds with none are left out")
    next: list[BehaviourNextCall]


class SuccessReportAnswer(BaseModel):
    """One success report (`detector_success_report.envelope`): build and read answer it."""

    id: str
    created_at: Optional[str]
    created_by: str
    body_sha256: str
    report: dict[str, Any] = Field(
        description="The stored mistudio.detector-success-report/1 body (stored JSON; report_algorithm says which)")
    stale: ReportStaleness


class SuccessReportRow(BaseModel):
    id: str
    behaviour_version: int
    model_id: str
    overall_verdict: str
    report_algorithm: int
    dataset_identity: dict[str, Any] = Field(description="Stored JSON")
    created_by: str
    created_at: Optional[str]


class SuccessReportListAnswer(BaseModel):
    """`GET /behaviours/{key}/success-reports`, newest first."""

    key: str
    reports: list[SuccessReportRow]
    total: int
    next_cursor: Optional[str]


class ComparisonCriterion(BaseModel):
    id: str
    verdict: str
    recall: Optional[float]
    reasons: list[str] = Field(description="The reasons' codes")


class ComparisonBlend(BaseModel):
    verdict: str
    reasons: list[str]


class ModelComparisonRow(BaseModel):
    """One model's report, or why it has none (`model_comparison.compare`)."""

    model_id: str
    refused: dict[str, Any] = may_be_absent(description="Only when this model's report was refused: {code, message, "
                                                        "..., next}")
    report_id: str = may_be_absent(description="Absent on a refused row")
    probe_id: str = may_be_absent(description="Absent on a refused row")
    overall_verdict: str = may_be_absent(description="Absent on a refused row")
    criteria: list[ComparisonCriterion] = may_be_absent(description="Absent on a refused row")
    blend: ComparisonBlend = may_be_absent(description="Absent on a refused row")
    stale: bool = may_be_absent(description="Absent on a refused row")


class ModelComparisonAnswer(BaseModel):
    """`GET /behaviours/{key}/model-comparison`: one success report per model."""

    key: str
    behaviour_version: int
    dataset_version: Optional[str]
    models: list[ModelComparisonRow]
    total: int
    next_cursor: Optional[str]
