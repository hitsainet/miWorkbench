"""The answers of the blend-report routes (Feature 041; 046 FTID §3.5): `probe_blends.py`.

Documentation only (FR-046.5 as amended 2026-11-26): each route keeps its serializer; these models are named in its
`responses=`. A key one branch writes and another does not is `may_be_absent()`.

Two builders: `probe_blends._queue` (start and rerun) and `probe_blend.service.envelope` (list, get). The envelope's
`resolved` is read back from the report row's stored JSON (`ProbeBlendReport.resolved`), so it is `history_form()`;
the stored report document itself (`report`) and its `ranking` are declared maps.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .answer_fields import history_form, may_be_absent

# ── shared pieces ───────────────────────────────────────────────────────────────────────────────────────────────


class BlendTestView(BaseModel):
    id: str
    held_out: bool = Field(description="True for an out-of-distribution view")


class BlendCalibrationView(BaseModel):
    id: str
    sample_rows: int


class BlendReportViews(BaseModel):
    """The views a report reads, by role (`service.start`'s `resolved["views"]`)."""

    blend_fit: str = Field(description="The one view the blends are fitted on")
    tests: list[BlendTestView]
    calibration: BlendCalibrationView


class NextCallAnswer(BaseModel):
    """One suggested call (`services.next_calls.NextCall`, built by `detector_next._call`)."""

    tool: str
    arguments: dict[str, Any]
    why: str
    needs: Optional[list[str]] = may_be_absent(description="What the call needs first; absent when nothing")


# ── start and rerun (`probe_blends._queue`) ─────────────────────────────────────────────────────────────────────


class BlendQueuedResolved(BaseModel):
    """The inputs a new report fixed at start (six of `resolved`'s keys)."""

    views: BlendReportViews
    folds: int
    group_column: Optional[str]
    target_fpr: float
    seed: int
    flags: list[str] = Field(description="e.g. no_rule_set")


class BlendQueued(BaseModel):
    """`POST /probe-monitors/blend-reports` and `POST /blend-reports/{id}/rerun`: the report queued (`_queue`)."""

    report_id: str
    status: str = Field(description="queued")
    resolved: BlendQueuedResolved
    members: list[str] = Field(description="The member probes' ids")
    candidates: list[str] = Field(description="The candidate blends' names")
    task_id: str


# ── the envelope (`probe_blend.service.envelope`) ───────────────────────────────────────────────────────────────


class BlendStoredResolved(BaseModel):
    """`resolved`'s seven served keys, read with `.get` from the stored row, so each can be null."""

    views: Optional[BlendReportViews]
    candidates: Optional[list[str]]
    folds: Optional[int]
    group_column: Optional[str]
    target_fpr: Optional[float]
    seed: Optional[int]
    flags: Optional[list[str]]


class BlendFailure(BaseModel):
    code: Optional[str]
    message: Optional[str]


class BlendReportRow(BaseModel):
    """One blend report as a list serves it (`service.envelope(include_report=False)`)."""

    report_id: str
    behaviour_key: str
    behaviour_version: int
    model_id: str
    dataset_version: Optional[str]
    status: str = Field(description="queued, running, done, failed or cancelled")
    failure: Optional[BlendFailure] = Field(description="Set only when the status is failed")
    stale: bool = Field(description="Computed on read: an input the report recorded has moved")
    stale_reasons: list[dict[str, Any]] = Field(description="What moved, one entry per reason (`code` names it)")
    next: list[NextCallAnswer] = Field(description="The rerun call when the report is stale, else empty")
    resolved: BlendStoredResolved = history_form(description="The inputs fixed at start (stored JSON)")
    members: list[str]
    report_sha256: Optional[str]
    report_bytes: Optional[int]
    rows_pruned_at: Optional[str]
    rerun_of: Optional[str]
    rerun_by: Optional[str]
    has_rung: bool = Field(description="Always false: a blend report earns no rung")
    created_by: str
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


class BlendReportAnswer(BlendReportRow):
    """`GET /probe-monitors/blend-reports/{id}`: the envelope, with the parsed report once it is done."""

    ranking: Optional[dict[str, Any]] = may_be_absent(
        description="The report's ranking; absent until the report is done")
    report: Optional[dict[str, Any]] = may_be_absent(
        description="The stored report document, parsed; absent until the report is done")


class BlendReportListAnswer(BaseModel):
    """`GET /probe-monitors/blend-reports`: a behaviour's reports, newest first."""

    behaviour_key: str
    model: Optional[str] = Field(description="The model filter, as asked")
    reports: list[BlendReportRow]
    offset: int
    limit: int
    next_offset: Optional[int]


# ── scores and cancel ───────────────────────────────────────────────────────────────────────────────────────────


class BlendScoreRow(BaseModel):
    position: int
    value: float
    verdict: bool
    label: int


class BlendScoresAnswer(BaseModel):
    """`GET /probe-monitors/blend-reports/{id}/scores`: one candidate's per-row values on one view, paged."""

    report_id: str
    candidate: str
    view_id: str
    total: int
    offset: int
    next_offset: Optional[int]
    rows: list[BlendScoreRow]


class BlendCancelled(BaseModel):
    """`POST /probe-monitors/blend-reports/{id}/cancel`."""

    report_id: str
    status: str
    detail: str
