"""Ungated write routes the dynamic sweep can only see REFUSE (045 FTDD §16.4; R1 review fix OR1-045a0-3, 2026-11-16).

`test_rest_gpu_gate.py::test_no_other_write_route_dispatches_for_an_agent` sends every ungated POST, PUT, PATCH and
DELETE route its fixture request with the agent header. For the routes below the handler RUNS but answers non-2xx
(a 4xx the fixture world cannot satisfy, or a 500), so the sweep proves only that early refusal, not that a call
which passed it would dispatch nothing. For them the guarantee is the STATIC walk, which now fails closed on any
enqueue it cannot resolve (`test_the_walk_resolves_every_enqueue`), and
`test_routes_the_sweep_only_refuses_are_walked` requires each to be walked and to reach no primitive.

The list is pinned BOTH ways: a new route that refuses early is red until it is listed here (review it: give it a
fixture that reaches 2xx, or accept the static guarantee), and a listed route that now answers 2xx is red until it is
removed. Regenerate with `WRITE_SWEEP_REFUSES_EARLY=1` and review the diff; never edit it to make a run pass.
"""
from __future__ import annotations

from pathlib import Path

SWEEP_REFUSES_EARLY: frozenset[tuple[str, str]] = frozenset({
    ('DELETE', '/api/v1/circuit-capture/{run_id}'),
    ('DELETE', '/api/v1/cluster-profiles/{profile_id}'),
    ('DELETE', '/api/v1/datasets/{dataset_id}'),
    ('DELETE', '/api/v1/datasets/{dataset_id}/cancel'),
    ('DELETE', '/api/v1/datasets/{dataset_id}/tokenization'),
    ('DELETE', '/api/v1/datasets/{dataset_id}/tokenizations/{tokenization_id}'),
    ('DELETE', '/api/v1/extraction-templates/{template_id}'),
    ('DELETE', '/api/v1/labeling-prompt-templates/{template_id}'),
    ('DELETE', '/api/v1/labeling/{labeling_job_id}'),
    ('DELETE', '/api/v1/models/{model_id}'),
    ('DELETE', '/api/v1/models/{model_id}/cancel'),
    ('DELETE', '/api/v1/neuronpedia/export/{job_id}'),
    ('DELETE', '/api/v1/probe-monitors/behaviours/{key}'),
    ('DELETE', '/api/v1/probe-monitors/runs/{run_id}'),
    ('DELETE', '/api/v1/probe-monitors/spot-checks/{check_id}'),
    ('DELETE', '/api/v1/prompt-templates/{template_id}'),
    ('DELETE', '/api/v1/settings/{key}'),
    ('DELETE', '/api/v1/steering/experiments/{experiment_id}'),
    ('DELETE', '/api/v1/training-templates/{template_id}'),
    ('DELETE', '/api/v1/trainings/{training_id}'),
    ('DELETE', '/api/v1/trainings/{training_id}/checkpoints/{checkpoint_id}'),
    ('PATCH', '/api/v1/circuits/{circuit_id}'),
    ('PATCH', '/api/v1/cluster-profiles/{profile_id}'),
    ('PATCH', '/api/v1/extraction-templates/{template_id}'),
    ('PATCH', '/api/v1/labeling-prompt-templates/{template_id}'),
    ('PATCH', '/api/v1/prompt-templates/{template_id}'),
    ('PATCH', '/api/v1/training-templates/{template_id}'),
    ('POST', '/api/internal/steering/reconcile-worker'),
    ('POST', '/api/internal/ws/emit'),
    ('POST', '/api/v1/circuit-capture/{run_id}/cancel'),
    ('POST', '/api/v1/circuit-discovery'),
    ('POST', '/api/v1/circuit-discovery/refine-cluster-edge'),
    ('POST', '/api/v1/circuit-discovery/{run_id}/attribution/cancel'),
    ('POST', '/api/v1/circuit-discovery/{run_id}/build-circuit'),
    ('POST', '/api/v1/circuit-discovery/{run_id}/cancel'),
    ('POST', '/api/v1/circuit-discovery/{run_id}/validate/cancel'),
    ('POST', '/api/v1/circuits'),
    ('POST', '/api/v1/circuits/import'),
    ('POST', '/api/v1/circuits/steering-samples/{run_id}/cancel'),
    ('POST', '/api/v1/circuits/{circuit_id}/calibration/cancel'),
    ('POST', '/api/v1/circuits/{circuit_id}/export-slices'),
    ('POST', '/api/v1/circuits/{circuit_id}/faithfulness/cancel'),
    ('POST', '/api/v1/cluster-profiles/export-bundle'),
    ('POST', '/api/v1/cluster-profiles/import'),
    ('POST', '/api/v1/code-rules/{code_sha256}/approval-requests'),
    ('POST', '/api/v1/datasets/download'),
    ('POST', '/api/v1/datasets/tokenize-preview'),
    ('POST', '/api/v1/datasets/{dataset_id}/tokenizations/{tokenization_id}/cancel'),
    ('POST', '/api/v1/datasets/{dataset_id}/tokenize'),
    ('POST', '/api/v1/enhanced-labeling/{job_id}/cancel'),
    ('POST', '/api/v1/extraction-templates/import'),
    ('POST', '/api/v1/extraction-templates/{template_id}/favorite'),
    ('POST', '/api/v1/extractions/{extraction_id}/cancel-nlp'),
    ('POST', '/api/v1/extractions/{extraction_id}/feature-groups/compute'),
    ('POST', '/api/v1/feature-groups/runs/{run_id}/cancel'),
    ('POST', '/api/v1/features/{feature_id}/analyze-nlp'),
    ('POST', '/api/v1/jlens/acquire/preview'),
    ('POST', '/api/v1/jlens/annotate'),
    ('POST', '/api/v1/jlens/annotation-sweeps'),
    ('POST', '/api/v1/jlens/artifacts/{slug}/restore-superseded'),
    ('POST', '/api/v1/jlens/artifacts/{slug}/validate'),
    ('POST', '/api/v1/jlens/attention-readout'),
    ('POST', '/api/v1/jlens/documents/validate'),
    ('POST', '/api/v1/jlens/gate'),
    ('POST', '/api/v1/jlens/tasks/{task_id}/cancel'),
    ('POST', '/api/v1/jlens/token-check'),
    ('POST', '/api/v1/jlens/watchlists'),
    ('POST', '/api/v1/labeling-prompt-templates/{template_id}/clone'),
    ('POST', '/api/v1/labeling-prompt-templates/{template_id}/set-default'),
    ('POST', '/api/v1/labeling/models/openai'),
    ('POST', '/api/v1/labeling/resume-sweeps/{sweep_id}/cancel'),
    ('POST', '/api/v1/labeling/{labeling_job_id}/cancel'),
    ('POST', '/api/v1/mcp/approvals'),
    ('POST', '/api/v1/mcp/approvals/{request_id}/consume'),
    ('POST', '/api/v1/models/{model_id}/estimate-extraction'),
    ('POST', '/api/v1/models/{model_id}/extractions/{extraction_id}/cancel'),
    ('POST', '/api/v1/neuronpedia/export/{job_id}/cancel'),
    ('POST', '/api/v1/probe-monitors/behaviours'),
    ('POST', '/api/v1/probe-monitors/behaviours/{key}/behaviour-verdict/export'),
    ('POST', '/api/v1/probe-monitors/behaviours/{key}/versions'),
    ('POST', '/api/v1/probe-monitors/blend-reports'),
    ('POST', '/api/v1/probe-monitors/blend-reports/{report_id}/cancel'),
    ('POST', '/api/v1/probe-monitors/blend-reports/{report_id}/rerun'),
    ('POST', '/api/v1/probe-monitors/datasets/{dataset_id}/confirm-positive'),
    ('POST', '/api/v1/probe-monitors/judge-prompts/import'),
    ('POST', '/api/v1/probe-monitors/judge-prompts/import-rubric'),
    ('POST', '/api/v1/probe-monitors/judge-prompts/{prompt_id}/versions'),
    ('POST', '/api/v1/probe-monitors/judge-runs/{judge_run_id}/cancel'),
    ('POST', '/api/v1/probe-monitors/probes/{probe_id}/evaluations/{evaluation_id}/backfill-truncation'),
    ('POST', '/api/v1/probe-monitors/probes/{probe_id}/rules-evaluations'),
    ('POST', '/api/v1/probe-monitors/rule-sets/validate'),
    ('POST', '/api/v1/probe-monitors/rule-signals/evaluate'),
    ('POST', '/api/v1/probe-monitors/rules-evaluations/{evaluation_id}/cancel'),
    ('POST', '/api/v1/probe-monitors/runs/forecast'),
    ('POST', '/api/v1/probe-monitors/runs/{run_id}/cancel'),
    ('POST', '/api/v1/probe-monitors/spot-checks/{check_id}/cancel'),
    ('POST', '/api/v1/probe-monitors/spot-checks/{check_id}/pin'),
    ('POST', '/api/v1/prompt-templates/import'),
    ('POST', '/api/v1/prompt-templates/{template_id}/duplicate'),
    ('POST', '/api/v1/prompt-templates/{template_id}/favorite'),
    ('POST', '/api/v1/saes/download'),
    ('POST', '/api/v1/saes/hf/preview'),
    ('POST', '/api/v1/saes/import/file'),
    ('POST', '/api/v1/saes/import/training'),
    ('POST', '/api/v1/saes/{sae_id}/cancel-extraction'),
    ('POST', '/api/v1/settings/pin/remove'),
    ('POST', '/api/v1/settings/pin/set'),
    ('POST', '/api/v1/steering/cleanup'),
    ('POST', '/api/v1/steering/cluster-allocation'),
    ('POST', '/api/v1/steering/compare'),
    ('POST', '/api/v1/steering/experiments'),
    ('POST', '/api/v1/steering/experiments/delete'),
    ('POST', '/api/v1/steering/sweep'),
    ('POST', '/api/v1/system/legacy-names'),
    ('POST', '/api/v1/system/restart'),
    ('POST', '/api/v1/training-templates/import'),
    ('POST', '/api/v1/training-templates/{template_id}/favorite'),
    ('POST', '/api/v1/trainings/checkpoints/prune-all'),
    ('POST', '/api/v1/trainings/{training_id}/checkpoints/prune'),
    ('POST', '/api/v1/trainings/{training_id}/finalize'),
    ('POST', '/api/v1/wsp/bundles'),
    ('PUT', '/api/v1/probe-monitors/behaviours/{key}/probes/{probe_id}/role'),
    ('PUT', '/api/v1/probe-monitors/behaviours/{key}/suggested-blend'),
})
SIZE = 123


def write(early) -> None:
    """Rewrite the list above from one sweep's outcomes (`WRITE_SWEEP_REFUSES_EARLY=1`)."""
    path = Path(__file__)
    text = path.read_text()
    head, rest = text.split("SWEEP_REFUSES_EARLY: frozenset[tuple[str, str]] = frozenset({\n", 1)
    _, tail = rest.split("})\nSIZE = ", 1)
    tail = tail.split("\n", 1)[1]
    rows = "".join(f"    ({m!r}, {p!r}),\n" for m, p in sorted(set(early)))
    path.write_text(head + "SWEEP_REFUSES_EARLY: frozenset[tuple[str, str]] = frozenset({\n" + rows + "})\n"
                    + f"SIZE = {len(set(early))}\n" + tail)
