"""The answers of six J-lens routes (046 FTASKS 8.7; FTID §3.5): `jlens.py`.

Documentation only (FR-046.5 as amended 2026-11-26): each route keeps its serializer; these models are named in its
`responses=`. A key one branch writes and another does not is `may_be_absent()`.

`GET /jlens/artifacts/{slug}/document` answers a `mistudio.jlens-artifact/v1` document, a kind published in
`docs/schemas/jspace-contracts-v1.json`, so it gets no model here: its `responses=` is a `$ref` to that file's `$id`
(`JSPACE_CONTRACTS_ID`, FR-046.11), as `GET /circuits/{id}/export` is.
"""
from __future__ import annotations

from typing import Annotated, Optional, Union

from pydantic import BaseModel, Field, RootModel

from .answer_fields import history_form, may_be_absent
from .jspace_contracts import (
    JLensArtifactDocument,
    ReadoutRecordDocument,
    WatchlistDocument,
    WorkspaceAnnotationDocument,
)

#: The `$id` of `docs/schemas/jspace-contracts-v1.json` (the four J-space kinds, a `oneOf`), which
#: `GET /jlens/artifacts/{slug}/document` answers one kind of (`JLensArtifactDocument`) unchanged (FR-046.11).
JSPACE_CONTRACTS_ID = "https://raw.githubusercontent.com/hitsainet/miStudio/main/docs/schemas/jspace-contracts-v1.json"  # noqa: E501

# ── GET /jlens/template-lens/{task_id} ──────────────────────────────────────────────────────────────────────────


class TemplateLensLayer(BaseModel):
    """One phrase's direction at one layer."""

    direction: list[float] = Field(description="The unit direction, d_model values")
    separation_in_baseline_std: float
    raw_norm: float


class TemplateLensEntry(BaseModel):
    """One phrase's template (`jlens_probe_tasks._template_lens`): the `TemplateLensSpec` fields, computed."""

    phrase: str
    n_contexts: int
    baseline_set: str
    whitening: str
    direction_available: bool
    contexts_sha256: str
    layers: dict[str, TemplateLensLayer] = Field(description="By layer index (as a string)")


class TemplateLensPoll(BaseModel):
    """`GET /jlens/template-lens/{task_id}`: the build's Celery state and, once it succeeded, its result.

    Four forms, told apart by `status`: a Celery state still running (`stage`); `FAILURE` (`error`); `SUCCESS` with the
    templates; and a task that returned without building (`skipped`, `cancelled`, or `failed` when it lost its GPU
    lease), whose own `status` replaces `SUCCESS` and which carries `scope`, `target_id`, `reason` and `detail`."""

    task_id: str
    status: str = Field(description="A Celery state (PENDING, PROGRESS, SUCCESS, FAILURE, …), or the task's own "
                                    "status when it returned without building: skipped, cancelled or failed")
    stage: Optional[str] = may_be_absent(description="While running: loading_model, baseline or phrases")
    error: Optional[str] = may_be_absent(description="On FAILURE: the exception's text")
    templates: Optional[list[TemplateLensEntry]] = may_be_absent(description="On SUCCESS")
    mode: Optional[str] = may_be_absent(description="On SUCCESS: template_lens")
    model: Optional[str] = may_be_absent(description="On SUCCESS: the model's name")
    hook_target: Optional[str] = may_be_absent(description="On SUCCESS: where the residual was read")
    limitations: Optional[list[str]] = may_be_absent(description="On SUCCESS")
    evidence_rung: Optional[int] = may_be_absent(description="On SUCCESS: 0")
    scope: Optional[str] = may_be_absent(description="Returned without building: the job kind")
    target_id: Optional[str] = may_be_absent(description="Returned without building: the job's target")
    reason: Optional[str] = may_be_absent(description="skipped (already_<status>) or failed (the lease was lost)")
    detail: Optional[str] = may_be_absent(description="cancelled or failed: what stopped it")


# ── POST /jlens/attention-readout ───────────────────────────────────────────────────────────────────────────────


class AttentionHeadComponent(BaseModel):
    """One singular component of one head's projection, read as tokens."""

    head: int
    component: int = Field(description="The component's rank (0 is the largest)")
    singular_value: float
    energy_fraction: float
    toward_tokens: list[str]
    away_tokens: list[str] = Field(description="The other end: a singular vector has no sign")


class AttentionMatrixReadout(BaseModel):
    side: str = Field(description="read (Q/K/V) or write (O)")
    checkpoint_key: str
    n_heads: int
    heads: list[AttentionHeadComponent]


class AttentionReadoutAnswer(BaseModel):
    """`POST /jlens/attention-readout` (`jlens_attention_readout.readout`): weights only, rung 0."""

    layer: int
    head_dim: int
    lens_type: str = Field(description="LOGIT_LENS")
    matrices: dict[str, AttentionMatrixReadout] = Field(description="By projection name, as the model calls it")
    evidence_rung: int


# ── GET /jlens/annotations ──────────────────────────────────────────────────────────────────────────────────────


class JLensAnnotationRow(BaseModel):
    """One persisted annotation (`JLensFeatureAnnotation`, as `list_annotations` serves it)."""

    feature_index: int
    surface: str = Field(description="decoder or encoder")
    lens_type: str
    layer: int
    lens_kurtosis: Optional[float]
    workspace_class: str
    top_tokens: list[str] = Field(description="Decoded tokens (stored JSON)")
    compared_label: Optional[str]
    disagreement_score: Optional[float]
    has_disagreement: bool
    evidence_rung: int
    updated_at: Optional[str]


class JLensAnnotationListAnswer(BaseModel):
    """`GET /jlens/annotations`: the label-disagreement queue, one page."""

    sae_id: str
    total: int
    limit: int
    offset: int
    annotations: list[JLensAnnotationRow]


# ── POST /jlens/documents/validate ──────────────────────────────────────────────────────────────────────────────

#: The four kinds `jspace_documents.parse_document` reads, told apart by `kind`.
JSpaceDocument = Annotated[
    Union[JLensArtifactDocument, WorkspaceAnnotationDocument, ReadoutRecordDocument, WatchlistDocument],
    Field(discriminator="kind"),
]


class JSpaceDocumentValidated(BaseModel):
    """`POST /jlens/documents/validate`: the document's kind and the document as this build reads it."""

    kind: str = Field(description="The document's versioned kind")
    document: JSpaceDocument


# ── GET /jlens/reports/replication ──────────────────────────────────────────────────────────────────────────────


class ReplicationLensResult(BaseModel):
    """One lens's figures (`jlens_replication.ReplicationReport.to_dict`). Null means not measured, never zero."""

    lens: str
    pass_at_k_auc: dict[str, Optional[float]] = Field(description="By evaluation set")
    ablation_kl: Optional[float]
    swap_success_rate: Optional[float]
    coverage: list[str] = Field(description="The evaluation sets that produced a figure")


class ReplicationReportAnswer(BaseModel):
    """`GET /jlens/reports/replication`: the stored replication report (the route answers null when none is stored)."""

    model_id: str
    reference_commit: str
    complete: bool
    missing: dict[str, list[str]] = Field(description="By lens: the evaluation sets it has no figure for")
    evaluation_sets: list[str]
    notes: str
    results: list[ReplicationLensResult] = history_form(description="Per lens (stored JSON)")


class ReplicationReportOrNull(RootModel[Optional[ReplicationReportAnswer]]):
    """`GET /jlens/reports/replication`: the stored report, or null when none was recorded (FR-046.8)."""
