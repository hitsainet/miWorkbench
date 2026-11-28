"""Weight-space programming (ENH-010, feature 044): captures, bundles, baselines, score calls, the per-row head budget
and sensitivity marks (`<044a>`, FTDD §4.2); experiments (`<044b>`, phase 3).

miWorkbench never edits weights itself and never imports a WSP library (FR-10): it captures held-out activations,
builds a bundle in miForge's shape, scores heads miForge proposes on rows miForge never sees, and imports a verdicted
candidate as a NEW probe. These tables are that bookkeeping. Files live under the run's artifact directory
(`artifact_dir_for(run_id)/wsp/…`), so deleting the run reclaims them (FTDD §4.4); the rows carry paths, never tensors.

Two tables are APPEND-ONLY (an ORM guard refuses update and delete, the `PurgeTombstoneIsAppendOnly` pattern):
`wsp_sensitivity_marks` (an operator's decision is history) and `wsp_score_calls` (the audit and the reuse count).

⚠ THE ORM GUARD NEVER SEES A DATABASE CASCADE (R2 review fix OR2-044a-H2, 2026-11-17). `wsp_score_calls` and the
per-row head budget (`wsp_heldout_row_heads`) used to hang off the bundle by `ON DELETE CASCADE`, and the bundle off
the probe and the run, so deleting a run erased every score call and gave the held-out pool a fresh 64 heads — the
budget FR-5 calls the ONLY bound, reset by a cleanup. Neither table now has a foreign key to anything that can be
deleted: `bundle_id` and `first_call_id` keep the minted id they were written with (ids are never reused), so the
record still names its bundle and its first call after either is gone. `SET NULL` was rejected because it is an UPDATE
of an append-only row that erases the bundle FR-5 says the record holds; `RESTRICT` because score calls are never
deleted, so it would make every scored run undeletable and its files unreclaimable (FTDD §4.4).
"""
import uuid

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from ..core.database import Base
from ..core.clock import utc_now


def _wac_id() -> str:
    return f"wac_{uuid.uuid4().hex[:12]}"


def _wsb_id() -> str:
    return f"wsb_{uuid.uuid4().hex[:12]}"


def _wbl_id() -> str:
    return f"wbl_{uuid.uuid4().hex[:12]}"


def _wsc_id() -> str:
    return f"wsc_{uuid.uuid4().hex[:12]}"


def _wsm_id() -> str:
    return f"wsm_{uuid.uuid4().hex[:12]}"


def _wsx_id() -> str:
    return f"wsx_{uuid.uuid4().hex[:12]}"


class WspActivationCapture(Base):
    """One held-out set captured for one probe (FR-1): the token activations at the probe's layer, with parity."""

    __tablename__ = "wsp_activation_captures"

    id = Column(String(36), primary_key=True, default=_wac_id)
    probe_id = Column(String(36), ForeignKey("probe_monitors.id", ondelete="CASCADE"), nullable=False)
    dataset_id = Column(String(36), ForeignKey("probe_monitor_datasets.id", ondelete="CASCADE"), nullable=False)
    #: `eval`, `everyday_text`, `nonsense_text`, `calibration`, or `supplement` (FU-7a C2: practice, validation only).
    role = Column(String(16), nullable=False)
    #: Whose positions were captured; NULL for a calibration capture.
    source_evaluation_id = Column(
        String(36), ForeignKey("probe_monitor_evaluations.id", ondelete="SET NULL"), nullable=True
    )
    layer = Column(Integer, nullable=False)
    scope = Column(String(16), nullable=False)
    rows = Column(Integer, nullable=True)
    tokens = Column(BigInteger, nullable=True)
    d_model = Column(Integer, nullable=True)
    dtype = Column(String(8), nullable=True)
    dir_path = Column(String(1000), nullable=True)
    row_ids_sha256 = Column(String(64), nullable=True)
    #: `{auroc_stored, auroc_capture, count_stored, count_capture, controls: {permuted_w, rolled_rows}, passed}`.
    parity = Column(JSONB, nullable=True)
    #: `pending`, `running`, `ready`, `parity_failed`, `failed`.
    status = Column(String(16), nullable=False, default="pending")
    bytes = Column(BigInteger, nullable=True)
    #: Written by the capture task while it runs; the janitor fails a `running` row whose heartbeat is stale.
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    #: R4 review fix (C044a-M1), 2026-11-17: a READY capture is sealed evidence. A bundle names it by id
    #: (`wsp_bundles.capture_ids`) and is scored on it, so it is never reset. When a ready calibration capture no
    #: longer describes the probe's bar (`wsp_capture.calibration_capture_stale`), the next request makes a NEW row
    #: (and so a new directory) and marks this one superseded: `superseded_at` and the replacement's id. A superseded
    #: row is never offered to a new bundle, and every bundle that already names it keeps scoring on it. NO foreign key
    #: on `superseded_by`: it keeps the id it was written with, as the WSP audit columns do.
    superseded_at = Column(DateTime(timezone=True), nullable=True)
    superseded_by = Column(String(36), nullable=True)

    __table_args__ = (
        UniqueConstraint("probe_id", "dataset_id", "source_evaluation_id", name="uq_wsp_capture_source"),
        # R2 review fix (OR2-044a-L6), 2026-11-17: a calibration capture has no source evaluation (NULL), and NULLs
        # never collide in a unique key, so two concurrent POST /wsp/captures used to make two calibration rows (two
        # GPU captures). One CURRENT calibration capture per (probe, view); `role`, not the NULL, keys it, so an eval
        # capture whose evaluation was deleted (SET NULL) never collides here. R4 (C044a-M1): superseded rows are
        # history beside the current one, so the index covers only rows not superseded.
        Index("uq_wsp_capture_calibration", "probe_id", "dataset_id", unique=True,
              postgresql_where=text("role = 'calibration' AND superseded_at IS NULL")),
        Index("ix_wsp_activation_captures_probe_id", "probe_id"),
    )


class WspBundle(Base):
    """A `miforge.wsp-bundle/v1` built for one probe (FR-2): the manifest, its digest and its live sensitivity basis."""

    __tablename__ = "wsp_bundles"

    id = Column(String(36), primary_key=True, default=_wsb_id)
    probe_id = Column(String(36), ForeignKey("probe_monitors.id", ondelete="CASCADE"), nullable=False)
    #: `linear_probe` (phase 2), `sae` (phase 4).
    bundle_kind = Column(String(16), nullable=False, default="linear_probe")
    manifest = Column(JSONB, nullable=True)
    bundle_sha256 = Column(String(64), nullable=True, unique=True)
    #: `sensitive`, `not_sensitive`, `unknown` — a SNAPSHOT; `/sensitivity` recomputes it live (P41).
    sensitivity = Column(String(16), nullable=False, default="unknown")
    sensitivity_basis = Column(JSONB, nullable=True)
    capture_ids = Column(JSONB, nullable=False, default=list)
    #: R4 review fix (C044a-M1): what the bundle was sealed on, per capture id: the RFC 8785 sha256 of that capture's
    #: parity record at build time (`wsp_bundle.capture_parity_sha256`). The scorer and the import refuse
    #: `capture_changed` when a capture's parity, rows or row ids differ from the seal and the manifest's
    #: `heldout_ref`. NULL = not sealed, refused the same way.
    capture_seals = Column(JSONB, nullable=True)
    #: The parent's bar AS THE SCORER CUTS IT (`wsp_score.cut_head_bar`, at build time), shown beside
    #: `operating_point.threshold` (the bar as served) with their difference (FR-1; R2 review fix OR2-044a-M3).
    scorer_parent_bar = Column(JSONB, nullable=True)
    dir_path = Column(String(1000), nullable=True)
    #: `building`, `ready`, `failed`.
    status = Column(String(16), nullable=False, default="building")
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (Index("ix_wsp_bundles_probe_id", "probe_id"),)


class WspBaseline(Base):
    """One baseline fitted on a bundle's PRACTICE rows only (FR-4; one row per (bundle, kind))."""

    __tablename__ = "wsp_baselines"

    id = Column(String(36), primary_key=True, default=_wbl_id)
    bundle_id = Column(String(36), ForeignKey("wsp_bundles.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(32), nullable=False)
    #: NULL for the two operating-point kinds (`threshold_recut`, `affine_recalibration`).
    weights_path = Column(String(1000), nullable=True)
    bar = Column(JSONB, nullable=True)
    transform = Column(JSONB, nullable=True)
    fit = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (UniqueConstraint("bundle_id", "kind", name="uq_wsp_baseline_kind"),)


class WspScoreCall(Base):
    """One call of the scorer (FR-5): the audit, the reuse count and P23's persistence lookup. APPEND-ONLY."""

    __tablename__ = "wsp_score_calls"

    id = Column(String(36), primary_key=True, default=_wsc_id)
    #: The bundle scored, by its minted id. NO foreign key (OR2-044a-H2): the record outlives its bundle.
    bundle_id = Column(String(36), nullable=False)
    heldout_ref_sha256 = Column(String(64), nullable=False)
    labels = Column(JSONB, nullable=False, default=list)
    miforge_run_id = Column(String(64), nullable=True)
    #: Declared by the caller (P46).
    patch_sha256 = Column(String(64), nullable=True)
    #: The `candidate` head's `head_tensors_sha256` (P23's lookup; R1A-W8).
    candidate_tensors_sha256 = Column(String(64), nullable=True)
    #: Every tensor-bearing head's label and `head_tensors_sha256` (P64).
    heads = Column(JSONB, nullable=False, default=list)
    caller = Column(JSONB, nullable=True)
    request_sha256 = Column(String(64), nullable=True)
    response_sha256 = Column(String(64), nullable=True)
    candidate_scores_sha256 = Column(String(64), nullable=True)
    status = Column(String(16), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    #: The arithmetic the scores were computed with (`wsp_score.ARITHMETIC_FIELDS`: MKL kernel family, torch and numpy
    #: versions, the scorer's thread bound; a field unknown at scoring is null), so an import whose P23 check fails
    #: names the field that differs (B5a follow-up, 2026-11-24). NULL on a call recorded before revision
    #: `5e3c9a1d7b42`.
    arithmetic = Column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_wsp_score_calls_run_candidate", "miforge_run_id", "candidate_tensors_sha256"),
        Index("ix_wsp_score_calls_bundle_id", "bundle_id"),
    )


class WspHeldoutRowHead(Base):
    """The head budget, COUNTED PER ROW (P64 as amended by P70): one row per (held-out row content key, distinct
    tensor-bearing head) that has read it, so overlapping re-samples, re-registered views and re-downloads share it."""

    __tablename__ = "wsp_heldout_row_heads"

    model = Column(String(255), primary_key=True)
    layer = Column(Integer, primary_key=True)
    scope = Column(String(16), primary_key=True)
    #: The P67 text key of the row.
    row_content_key = Column(String(64), primary_key=True)
    head_tensors_sha256 = Column(String(64), primary_key=True)
    label = Column(String(32), nullable=False)
    #: The call that first counted this (row, head). NO foreign key (OR2-044a-H2): the budget outlives everything.
    first_call_id = Column(String(36), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class WspSensitivityMark(Base):
    """An operator's sensitivity mark on a downloaded dataset (FR-7, P11, P69). APPEND-ONLY."""

    __tablename__ = "wsp_sensitivity_marks"

    id = Column(String(36), primary_key=True, default=_wsm_id)
    #: The dataset the operator marked. NO foreign key (R3 review fix OR3-044a-L1, as OR2-044a-H2 for score calls):
    #: the table is append-only history, and a database cascade from a deleted dataset would erase every decision
    #: the operator made about it, beneath the ORM guard. A re-downloaded dataset has a new id and reads `unknown`.
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    #: `sensitive` / `not_sensitive`.
    value = Column(String(16), nullable=False)
    reason = Column(Text, nullable=False)
    decided_by = Column(String(32), nullable=False, default="operator")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (Index("ix_wsp_sensitivity_marks_dataset_id", "dataset_id"),)


class WspExperiment(Base):
    """One miForge WSP run miWorkbench started for one bundle (FR-12, FR-13; `<044b>`, FTDD §4.2; phase 3).

    miWorkbench relays miForge's pending answer (`approval`, decided in miForge), reads the run id from the approval's
    result, calls `evaluate` itself and stores the `verdict_id` it got back (P71), so the import (FR-14) takes THAT
    verdict even when a later one exists. `verdict`, `heldout_uses`, `program_sha256` and `reruns` are caches of
    miForge's answers (refreshed by `GET /wsp/experiments/{id}`), never miWorkbench's own judgement. A program-mode
    run is shown by its digest, never its source (P19).

    Schema only until the experiment driver (044 FTASKS 13.2) lands: nothing in `src/` writes this table yet. Ported
    from the unpushed `wsp/044-c` (`c0d8c8192`) as revision `408803691ec6` (suite plan B5, 2026-11-24)."""

    __tablename__ = "wsp_experiments"

    id = Column(String(36), primary_key=True, default=_wsx_id)
    #: The parent probe and the bundle. SET NULL, nullable (operator decision 2026-11-24, B5 Q2): the experiment row
    #: outlives a deleted probe or bundle, unlinked, as score calls keep their record (OR2-044a-H2) — it holds miForge's
    #: run id and the verdict id, and the run lives on in miForge.
    probe_id = Column(String(36), ForeignKey("probe_monitors.id", ondelete="SET NULL"), nullable=True)
    bundle_id = Column(String(36), ForeignKey("wsp_bundles.id", ondelete="SET NULL"), nullable=True)
    #: `inspect`, `create`, `repair` (the UI's "construct" is sent as `create`; R2B-18).
    kind = Column(String(16), nullable=False)
    #: `session`, `program`.
    mode = Column(String(16), nullable=False, default="session")
    #: NULL until the operator approves the start in miForge (the run id is read from the approval's result).
    miforge_run_id = Column(String(64), nullable=True)
    #: The verdict miWorkbench's own `evaluate` call returned (P71, R4B-1); the import fetches this one.
    verdict_id = Column(String(64), nullable=True)
    #: miForge's frozen-request digest (P24).
    request_sha256 = Column(String(64), nullable=True)
    #: `importing_bundle`, `pending_approval`, `running`, `evaluating`, `verdicted`, `imported`, `refused`, `failed`.
    status = Column(String(24), nullable=False, default="importing_bundle")
    approval = Column(JSONB, nullable=True)
    verdict = Column(JSONB, nullable=True)
    heldout_uses = Column(JSONB, nullable=True)
    program_sha256 = Column(String(64), nullable=True)
    reruns = Column(JSONB, nullable=True)
    imported_probe_id = Column(String(36), ForeignKey("probe_monitors.id", ondelete="SET NULL"), nullable=True)
    import_acknowledgement = Column(JSONB, nullable=True)
    error = Column(Text, nullable=True)
    last_synced_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        Index("ix_wsp_experiments_probe_id", "probe_id"),
        Index("ix_wsp_experiments_bundle_id", "bundle_id"),
        # One miForge run is one experiment: a second row naming the same run would double-count its verdict.
        UniqueConstraint("miforge_run_id", name="uq_wsp_experiments_miforge_run_id"),
    )


class WspRowIsAppendOnly(RuntimeError):
    """Raised when code tries to change or remove an append-only WSP row (a sensitivity mark or a score call)."""


def refuse_update(mapper, connection, target):  # noqa: ARG001
    raise WspRowIsAppendOnly(f"{target.__tablename__} is append-only: a row cannot be updated")


def refuse_delete(mapper, connection, target):  # noqa: ARG001
    raise WspRowIsAppendOnly(f"{target.__tablename__} is append-only: a row cannot be deleted")


#: The append-only WSP tables; each carries both guards on its live mapper.
APPEND_ONLY = (WspSensitivityMark, WspScoreCall)

for _model in APPEND_ONLY:
    event.listen(_model, "before_update", refuse_update)
    event.listen(_model, "before_delete", refuse_delete)
