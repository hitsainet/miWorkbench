"""The WSP scoring contract, `miworkbench-wsp-scoring-v1.json` (ENH-010 044 task 7.1; FR-5, FR-6; FTDD §5.7.1).

miWorkbench OWNS this contract: miForge's `evaluate` calls `POST /api/v1/wsp/score` and vendors the generated file
(016 13.2). FTDD §5.7.1 is the single definition; this module is its code. **Bars and metrics are per head and per
set** (P29). Every model is `extra="forbid"` with NO aliases (memory `pydantic-alias-renames-on-serialisation`: an
alias renames on output too). The refusal table is the WHOLE table from 044-A on, phase-3 and phase-4 codes included,
so 044-C changes no contract (R3B-16); 7.6 generates the published file from these models and `REFUSALS`.
"""
from __future__ import annotations

import base64
import binascii
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REQUEST_KIND = "miworkbench.wsp-score-request/v1"
RESPONSE_KIND = "miworkbench.wsp-score-result/v1"
HEX64 = r"^[0-9a-f]{64}$"

#: The baseline kinds a `baseline:<kind>` head may name (the `BASELINE_KINDS` registry, 6.1, keys these).
BASELINE_KINDS: Tuple[str, ...] = (
    "logistic_regression", "lda", "diff_of_means", "threshold_recut", "affine_recalibration",
)
#: `parent`, `candidate`, `control`, `control:<n>`, `baseline:<kind>`.
LABEL_PATTERN = r"^(parent|candidate|control(:(0|[1-9][0-9]{0,3}))?|baseline:(" + "|".join(BASELINE_KINDS) + r"))$"
#: The tensor names a probe head may carry (its safetensors keys).
TENSOR_NAMES = ("weight", "bias", "mean", "std", "attention_query")

SetRole = Literal["eval", "everyday_text", "nonsense_text", "calibration"]
BarSource = Literal["calibration", "practice", "practice_logit_zero"]


def _strict() -> ConfigDict:
    return ConfigDict(extra="forbid", allow_inf_nan=False, populate_by_name=False)


def carries_tensors(label: str) -> bool:
    """`candidate`, `control` and `control:<n>` carry tensors; `parent` and `baseline:*` never do."""
    return label == "candidate" or label == "control" or label.startswith("control:")


class RunRef(BaseModel):
    """The miForge run the call belongs to (P46, R1A-W8, R1B-X15)."""

    model_config = _strict()

    miforge_run_id: str = Field(min_length=1, max_length=64)
    patch_sha256: Optional[str] = Field(None, pattern=HEX64)
    #: The run's own limit; null takes the server ceiling (`wsp_score_max_changed_entries_ceiling`).
    max_changed_entries: Optional[int] = Field(None, ge=0)


class HeldoutSet(BaseModel):
    model_config = _strict()

    name: str = Field(min_length=1, max_length=128)
    view_id: str = Field(min_length=1, max_length=64)
    role: SetRole
    rows: int = Field(ge=0)
    row_ids_sha256: str = Field(pattern=HEX64)


class HeldoutRef(BaseModel):
    """Must equal the bundle's `heldout_ref` exactly (`heldout_ref_mismatch`); its RFC 8785 digest is
    `heldout_ref_sha256` in the response (R3B-13, P65)."""

    model_config = _strict()

    owner: Literal["miworkbench"]
    sets: List[HeldoutSet] = Field(min_length=1, max_length=64)


class HeadTensor(BaseModel):
    model_config = _strict()

    name: Literal["weight", "bias", "mean", "std", "attention_query"]
    dtype: Literal["float32", "float64"]
    shape: List[int] = Field(max_length=4)
    #: The tensor's little-endian bytes, base64.
    data_b64: str
    #: `tensor_sha256` of those bytes (016's definition).
    sha256: str = Field(pattern=HEX64)

    @field_validator("shape")
    @classmethod
    def _non_negative(cls, value: List[int]) -> List[int]:
        if any(n < 0 for n in value):
            raise ValueError("a shape holds no negative dimension")
        return value

    @field_validator("data_b64")
    @classmethod
    def _base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"data_b64 is not base64: {exc}") from exc
        return value


class ScoreHead(BaseModel):
    model_config = _strict()

    label: str = Field(pattern=LABEL_PATTERN)
    #: Required for `candidate`/`control*`, refused for `parent`/`baseline:*` — by the route, with its code
    #: (`parent_tensors_not_accepted`), not here, so the refusal carries `{code, message, next}`.
    tensors: Optional[List[HeadTensor]] = Field(None, max_length=len(TENSOR_NAMES))


class WspScoreRequest(BaseModel):
    model_config = _strict()

    kind: Literal["miworkbench.wsp-score-request/v1"]
    bundle_id: str = Field(pattern=r"^wsb_[0-9a-f]{12}$")
    bundle_sha256: str = Field(pattern=HEX64)
    run_ref: RunRef
    heldout_ref: HeldoutRef
    heads: List[ScoreHead] = Field(min_length=1, max_length=64)
    #: Echoed; miForge's verdict uses them.
    primary_sets: List[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def _unique_labels(self) -> "WspScoreRequest":
        labels = [h.label for h in self.heads]
        if len(labels) != len(set(labels)):
            raise ValueError("each head label appears once")
        names = {s.name for s in self.heldout_ref.sets}
        if len(names) != len(self.heldout_ref.sets):
            raise ValueError("each held-out set name appears once")
        unknown = sorted(set(self.primary_sets) - names)
        if unknown:
            raise ValueError(f"primary_sets names sets the request does not hold: {unknown}")
        return self


class FunctionInfo(BaseModel):
    model_config = _strict()

    rule: str
    params: Dict[str, float | int | str | bool | None]
    layer: int
    scope: str
    skip_template_prefix: bool


class Excluded(BaseModel):
    model_config = _strict()

    truncated: int = Field(ge=0)
    #: Per head label: how many of the truncated rows fire at that head's bar (G5).
    truncated_fired: Dict[str, int]


class Bar(BaseModel):
    model_config = _strict()

    threshold: Optional[float]
    source: BarSource
    n_negatives: int = Field(ge=0)
    realised_fpr: float
    target_fpr: float


class Metrics(BaseModel):
    """`false_alarms_at_bar` is a COUNT; the rate is `false_alarms_at_bar / negatives` (R1B-X3). Every metric is
    computed from exactly the float32 scores in the response."""

    model_config = _strict()

    auroc: Optional[float]
    recall_at_bar: Optional[float]
    false_alarms_at_bar: Optional[int]
    negatives: int = Field(ge=0)
    positives: int = Field(ge=0)


#: R3 review fix (OR3-044a-L5), 2026-11-17: "the calibration set carries no `scores`, `labels` or `metrics`" was prose
#: only, which miForge's validator cannot enforce. In the schema: a set whose `role` is `calibration` has each of them
#: absent or null. The model refuses the same (`SetResult._calibration_carries_no_rows`).
CALIBRATION_CARRIES_NO_ROWS = {
    "if": {"properties": {"role": {"const": "calibration"}}, "required": ["role"]},
    "then": {"properties": {"scores": {"type": "null"}, "labels": {"type": "null"}, "metrics": {"type": "null"}}},
}


class SetResult(BaseModel):
    """The calibration set (role `calibration`) carries no `scores`, `labels` or `metrics`: each is absent or null
    (it produces bars only)."""

    model_config = ConfigDict(**_strict(), json_schema_extra=CALIBRATION_CARRIES_NO_ROWS)

    name: str
    view_id: str
    role: SetRole
    row_ids_sha256: str = Field(pattern=HEX64)
    scored_row_ids_sha256: str = Field(pattern=HEX64)
    rows: int = Field(ge=0)
    excluded: Excluded
    labels: Optional[List[Literal[0, 1]]] = None
    scores: Optional[Dict[str, List[float]]] = None
    bars: Dict[str, Bar]
    metrics: Optional[Dict[str, Metrics]] = None

    @model_validator(mode="after")
    def _calibration_carries_no_rows(self) -> "SetResult":
        if self.role == "calibration" and any(v is not None for v in (self.scores, self.labels, self.metrics)):
            raise ValueError("the calibration set carries no scores, labels or metrics (it produces bars only)")
        return self


class WspScoreResponse(BaseModel):
    """No activation field and no text, ever (P46; 7.4's leak test pins the exact key sets)."""

    model_config = _strict()

    kind: Literal["miworkbench.wsp-score-result/v1"]
    bundle_sha256: str = Field(pattern=HEX64)
    heldout_ref_sha256: str = Field(pattern=HEX64)
    function: FunctionInfo
    sets: List[SetResult]
    scored_at: str
    call_id: str = Field(pattern=r"^wsc_[0-9a-f]{12}$")


#: **The whole refusal table** (code → HTTP status, when). 044-A's route raises every phase-2 code; the phase-3 and
#: phase-4 codes are declared now so 044-C and 044-D change no contract (R3B-16). The route's own refusal registry is
#: checked against this table by 7.6's `test_refusal_table_is_complete`.
REFUSALS: Dict[str, Tuple[int, str]] = {
    "operator_or_service_only": (403, "an agent caller (X-MiStudio-Agent) may not score"),
    "bundle_not_found": (404, "no bundle has that id"),
    "bundle_not_ready": (409, "the bundle is still building, or failed"),
    "bundle_sha256_mismatch": (409, "bundle_sha256 is not the bundle's"),
    "heldout_ref_mismatch": (422, "heldout_ref is not exactly the bundle's (a subset or a different set)"),
    "parent_tensors_not_accepted": (422, "a parent or baseline head carries tensors"),
    "head_tensors_required": (422, "a candidate or control head carries no tensors"),
    "head_shape_mismatch": (422, "a head's tensor dtype, shape, digest or byte count does not match the parent's"),
    "candidate_outside_patch_limits": (
        422, "a head differs from the parent outside the editable tensors, or in more entries than the clamp allows"),
    "too_many_heads": (422, "more heads than wsp_score_max_heads"),
    "capture_not_ready": (409, "a held-out capture of the bundle is not ready"),
    "capture_parity_failed": (409, "a held-out capture of the bundle failed parity"),
    "capture_changed": (409, "a held-out capture is not the one the bundle was sealed on: its rows, row ids or parity "
                             "record differ from the manifest's heldout_ref and the bundle's seal"),
    "heldout_set_head_budget_exhausted": (
        409, "a requested row has already been read by wsp_score_max_heads_per_set distinct tensor-bearing heads; "
             "the answer names the set, its row_ids_sha256 and rows_exhausted (a count)"),
    "scorer_busy": (429, "every scorer slot is in use; retry later"),
    "score_timeout": (503, "the call was still scoring after wsp_score_timeout_seconds; nothing was recorded"),
    "bundle_kind_unsupported": (422, "the bundle's kind is not scored by this route in this phase"),
    "baseline_not_available": (409, "a baseline:<kind> head names a kind the bundle has not fitted (pending or "
                                    "refused)"),
    # Phase 3 (044-C, P65, R3B-7, R4B-8): run-bound heads.
    "candidate_not_the_registered_patch": (
        409, "the candidate is not parent + the run's registered patch, or the run cannot be resolved (miForge unset "
             "or unreachable, an unknown run)"),
    "control_not_the_seeded_permutation": (409, "a control head is not the run's seeded permutation (016 §6.4)"),
    "control_index_out_of_range": (422, "control:<n> with n at or above the run's control count"),
}


def refusal_status(code: str) -> int:
    return REFUSALS[code][0]



#: Where the generated contract is published (7.6); miForge vendors it byte-identical (016 13.2).
CONTRACT_PATH = "docs/schemas/miworkbench-wsp-scoring-v1.json"


def scoring_contract() -> dict:
    """The published contract, generated from the two models and the WHOLE refusal table (FTDD §5.7.1, R3B-16).
    Regenerate with `python -m tests.unit.test_wsp_scoring_contract_generated --write` (from `backend/`)."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://hitsai.net/schemas/miworkbench-wsp-scoring-v1.json",
        "title": "miWorkbench WSP scoring contract (POST /api/v1/wsp/score)",
        "description": (
            "miWorkbench scores heads miForge proposes on held-out rows miForge never sees. Bars and metrics are "
            "per head and per set; every metric is computed from exactly the float32 scores in the response; "
            "false_alarms_at_bar is a count. The refusal table is complete for every phase (044-A on)."),
        "version": 1,
        "endpoint": {"method": "POST", "path": "/api/v1/wsp/score"},
        "request": WspScoreRequest.model_json_schema(),
        "response": WspScoreResponse.model_json_schema(),
        "refusals": {code: {"status": status, "when": when} for code, (status, when) in sorted(REFUSALS.items())},
    }


def contract_text() -> str:
    import json

    return json.dumps(scoring_contract(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
