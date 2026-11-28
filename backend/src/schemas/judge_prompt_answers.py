"""The answers of the judge-prompt library routes (046 FTASKS 8.x; FTID §3.5): `probe_judge_prompts.py`.

Documentation only (FR-046.5 as amended 2026-11-26): each route keeps its serializer; these models are named in its
`responses=`. A key one branch writes and another does not is `may_be_absent()`. A stored free-form JSON value (a
prompt's answer form, its provenance, a rung check's per-set rows) is a declared map, `dict[str, Any]`.

`judge_prompt_library.summary` is the ONE serializer for a library row, so one model (`JudgePromptSummary`) serves the
list, and `JudgePromptAnswer` (`library.detail`: the summary plus its versions) serves every route that answers one
prompt.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from .answer_fields import may_be_absent

# ── one library row (`judge_prompt_library.summary` / `detail`) ─────────────────────────────────────────────────


class JudgePromptSummary(BaseModel):
    """One judge-prompt version as every list serves it (`judge_prompt_library.summary`)."""

    id: str
    name: str
    version: int
    key: str = Field(description="`name/vN`, the string every judge run records as its prompt version")
    title: str
    description: Optional[str]
    behaviour: str = Field(description="The behaviour key this prompt judges")
    behaviour_label: Optional[str]
    system_prompt: str
    user_template: str
    answer_form: dict[str, Any] = Field(description="The answer form, normalised (rating, binary or labels; stored JSON)")
    answer_form_summary: str = Field(description="The answer form in one plain sentence")
    max_interaction_chars: int
    content_hash: str
    source: str = Field(description="seed, operator, agent:<name>, midataworks or import")
    source_ref: Optional[dict[str, Any]] = Field(description="Provenance detail (stored JSON): a rubric's id/ref/hash, "
                                                             "or the file an import came from")
    parent_id: Optional[str] = Field(description="The version or prompt this one was made from")
    created_by: str
    created_at: datetime
    updated_at: datetime
    locked: bool = Field(description="True once a judge run has used this version (its content is then frozen)")
    locked_at: Optional[datetime]
    archived: bool
    archived_at: Optional[datetime]
    used_by_runs: int = Field(description="How many judge runs name this version")
    is_default: bool = Field(description="The version a judge run uses when no prompt is named")
    warnings: list[str] = Field(description="Where the prompt does not ask for the reply shape the parser reads")


class JudgePromptVersionRow(BaseModel):
    """One version of a prompt's name, as `detail` lists them (newest first)."""

    id: str
    version: int
    key: str
    content_hash: str
    locked: bool
    archived: bool
    used_by_runs: int
    created_at: datetime
    created_by: str


class JudgePromptAnswer(JudgePromptSummary):
    """One version with every version of its name (`judge_prompt_library.detail`): get, create, import-rubric, patch,
    new version, clone, archive and unarchive all answer it."""

    versions: list[JudgePromptVersionRow]


class JudgePromptImported(JudgePromptAnswer):
    """`POST /probe-monitors/judge-prompts/import`: the stored version, and whether this import created it."""

    created: bool = Field(description="False when the same name, version and content were already stored")


class JudgePromptListAnswer(BaseModel):
    """`GET /probe-monitors/judge-prompts`: the library, and the placeholders a template may name."""

    prompts: list[JudgePromptSummary]
    placeholders: dict[str, str] = Field(description="Each placeholder a user template may name, with what it is "
                                                     "filled with (`TEMPLATE_PLACEHOLDERS`)")


# ── preview and try ─────────────────────────────────────────────────────────────────────────────────────────────


class JudgePromptPreviewRow(BaseModel):
    label: str = Field(description="positive or negative: the row's own label")
    user_message: str = Field(description="The judge's user message for this row, exactly as it would be sent")


class JudgePromptPreviewAnswer(BaseModel):
    """`POST /probe-monitors/judge-prompts/preview`: the judge's exact messages for a few rows; nothing is sent."""

    prompt_key: str = Field(description="`name/vN`, or `<name> (unsaved draft)` for a draft")
    content_hash: str
    dataset_name: str = Field(description="The view the rows were read from")
    system_prompt: str
    warnings: list[str]
    rows: list[JudgePromptPreviewRow]


class JudgeTryParsedAnswer(BaseModel):
    """One reply as the answer form read it (`judge_prompt_forms.ParsedAnswer`)."""

    value: Union[int, str] = Field(description="The rating, or the answer word or label")
    positive: bool = Field(description="The judge's own call under the answer form")
    score: float = Field(description="What the AUROC ranks")


class JudgeTryRow(BaseModel):
    """One row sent to the judge (`probe_judge_prompts._try_rows`)."""

    label: str = Field(description="positive or negative: the row's own label")
    user_message: str
    reply: Optional[str] = Field(description="The judge's reply (at most 2,000 characters); null when the call failed")
    answer: Optional[JudgeTryParsedAnswer] = Field(description="Null when the call failed or the reply was unreadable")
    error: Optional[str] = Field(description="Why the row has no answer (the call's error, or an unreadable reply)")
    agrees_with_label: Optional[bool] = may_be_absent(
        description="Whether the judge's call matches the row's label; null when the reply was unreadable. "
                    "Absent when the call to the judge failed")


class JudgePromptTryAnswer(BaseModel):
    """`POST /probe-monitors/judge-prompts/try` (gated): a few rows through the judge, each reply and how it was read."""

    rows: list[JudgeTryRow]
    parsed: int = Field(description="Rows whose reply was read")
    unparseable: int = Field(description="Rows with no answer (failed calls included)")
    stopped: Optional[str] = Field(description="Why the try stopped early (the judge endpoint is failing), or null")
    prompt_key: str
    dataset_name: str
    model: str = Field(description="The judge model the request named")


# ── miDataworks rubrics ─────────────────────────────────────────────────────────────────────────────────────────


class MidataworksRubricRow(BaseModel):
    """One miDataworks rubric as `judge_prompt_library.list_midataworks_rubrics` serves it.

    Every value except `importable` is copied from miDataworks' own row with `.get`, so each is null when that row
    lacks the key. The route checks no type; these are the types this app's own rubric reader
    (`judge_prompt_library.rubric_to_prompt`) treats them as."""

    id: Optional[str] = Field(description="miDataworks' rubric id")
    name: Optional[str]
    version: Optional[int]
    ref: Optional[str] = Field(description="`name@version`")
    style: Optional[str] = Field(description="The rubric's style; binary and pointwise can be imported")
    content_hash: Optional[str]
    importable: bool = Field(description="Whether this rubric's style can be imported as a judge prompt")
    input_fields: Optional[list[str]] = Field(description="The rubric body's `input_fields`")
    allowed_verdicts: Optional[list[str]] = Field(description="The rubric body's `allowed_verdicts`")


class MidataworksRubricListAnswer(BaseModel):
    """`GET /probe-monitors/judge-prompts/midataworks/rubrics`: miDataworks' rubrics, when it is connected."""

    rubrics: list[MidataworksRubricRow]


# ── export ──────────────────────────────────────────────────────────────────────────────────────────────────────


class JudgePromptDocument(BaseModel):
    """`GET /probe-monitors/judge-prompts/{id}/export`: a `mistudio.judge-prompt/v1` document
    (`judge_prompt_library.export_document`). No schema for this kind is published under `docs/schemas/`."""

    format: str = Field(description="mistudio.judge-prompt/v1")
    name: str
    version: int
    title: str
    description: Optional[str]
    behaviour: str
    behaviour_label: Optional[str]
    system_prompt: str
    user_template: str
    answer_form: dict[str, Any] = Field(description="The normalised answer form (stored JSON)")
    max_interaction_chars: int
    content_hash: str = Field(description="Reproduced from the other fields on any install")


# ── a probe's behaviour, and the rung it recomputes ─────────────────────────────────────────────────────────────


class RungTextCheck(BaseModel):
    """The everyday-text or nonsense-text false-alarm check (`probe_monitor_rung.EverydayCheck.as_dict`)."""

    kind: str = Field(description="everyday_text or nonsense_text: the view role it was measured on")
    status: str = Field(description="passed, not_checked or too_many_false_alarms")
    passed: bool
    reason: str
    next_step: Optional[str]
    target_fpr: Optional[float]
    factor: float
    min_rows: int
    allowed_rate: Optional[float] = Field(description="factor x target_fpr, or null with no target")
    sets: list[dict[str, Any]] = Field(description="Every set with its numbers, allowance and sentence")
    truncated_rows_count: bool = Field(description="Always false: truncated rows are never counted")
    verdict_changed_by_exclusion: bool


class RungCatchRateCheck(BaseModel):
    """The catch rate at the shipped bar on unseen tasks (`probe_monitor_rung.CatchRateCheck.as_dict`)."""

    status: str = Field(description="passed, no_sets, not_measured or too_few_caught")
    passed: bool
    reason: str
    next_step: Optional[str]
    min_recall: float
    sets: list[dict[str, Any]] = Field(description="Every out-of-distribution set with its positives and catches")
    truncated_rows_count: bool
    verdict_changed_by_exclusion: bool


class RungRecomputed(BaseModel):
    """A probe's rung, recomputed from the evaluations that exist (`probe_monitor_run.recompute_rung`, which
    returns `probe_monitor_rung.RungResult.as_dict`)."""

    everyday_text: Optional[RungTextCheck]
    nonsense_text: Optional[RungTextCheck]
    catch_rate: Optional[RungCatchRateCheck]
    rung: int = Field(description="The evidence rung (0 to 3)")
    language: str = Field(description="What the rung licenses saying, in words")
    next_step: str
    reasons: list[str]
    in_distribution_passed: list[str]
    in_distribution_failed: list[str]
    out_of_distribution_passed: list[str]
    out_of_distribution_failed: list[str]
    judge_sets_completed: list[str]


class ProbeBehaviourSet(BaseModel):
    """`PATCH /probe-monitors/probes/{id}/behaviour`: what the probe now detects, and its recomputed rung."""

    probe_id: str
    behaviour: Optional[str] = Field(description="The behaviour key, or null when it was cleared")
    behaviour_label: Optional[str]
    rung: RungRecomputed
