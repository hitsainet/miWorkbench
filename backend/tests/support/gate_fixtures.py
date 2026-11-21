"""The gated routes' design table and one valid agent request per route (045 FR-13, FR-14; FTDD §16.2, §16.2b).

`DESIGN` is FTDD §16.2's rows (and §16.2b's publishes) as live full path templates with the action FTID §11a's table
fixes for each (R2M-15: a route an MCP tool calls takes that tool's name; row 28 is `build_probe_definition`).
`DEFERRED` holds the design rows not yet gated on this branch, each with its reason; a route leaves the design only by a
recorded design change (A27).

`sample_request(route)` builds a request the route's OWN parsing accepts: required path, query and body values
generated from the route's dependant (the first value each schema admits), with per-route overrides where a schema
admits a value its validators refuse. It is the source of the published payload vectors.
"""
from __future__ import annotations

import re
from typing import Any

#: row → (full path template, action). Method is POST for every row.
DESIGN: dict[Any, tuple[str, str]] = {
    1: ("/api/v1/circuit-capture", "start_circuit_capture"),
    2: ("/api/v1/circuit-capture/{run_id}/confirm", "confirm_circuit_capture"),
    3: ("/api/v1/circuit-discovery/{run_id}/attribution", "run_attribution_pass"),
    4: ("/api/v1/circuit-discovery/{run_id}/validate", "validate_circuit_edges"),
    5: ("/api/v1/validation-manifests/{manifest_id}/reproduce", "reproduce_validation"),
    6: ("/api/v1/circuits/{circuit_id}/faithfulness", "run_circuit_faithfulness"),
    7: ("/api/v1/circuits/{circuit_id}/calibration", "calibrate_circuit_strength"),
    8: ("/api/v1/circuits/calibration-manifests/{manifest_id}/reproduce", "reproduce_calibration"),
    9: ("/api/v1/circuits/steering-samples", "record_steering_samples"),
    10: ("/api/v1/jlens/fit", "fit_jlens_artifact"),
    11: ("/api/v1/jlens/artifacts/revalidate", "revalidate_jlens_artifact"),
    12: ("/api/v1/jlens/band-report", "compute_jlens_band_report"),
    13: ("/api/v1/jlens/readout", "jlens_readout"),
    14: ("/api/v1/jlens/probe", "jlens_probe"),
    15: ("/api/v1/jlens/decompose", "jlens_decompose"),
    16: ("/api/v1/jlens/template-lens", "build_jlens_template_lens"),
    17: ("/api/v1/jlens/interventions", "run_jlens_intervention"),
    18: ("/api/v1/jlens/acquire", "acquire_jlens_artifact"),
    19: ("/api/v1/models/download", "download_model"),
    20: ("/api/v1/models/{model_id}/redownload", "redownload_model"),
    21: ("/api/v1/models/{model_id}/extract-activations", "extract_activations"),
    22: ("/api/v1/models/{model_id}/extractions/{extraction_id}/retry", "retry_extraction"),
    23: ("/api/v1/probe-monitors/runs", "start_probe_training_run"),
    24: ("/api/v1/probe-monitors/probes/{probe_id}/evaluate", "evaluate_probe_monitor"),
    25: ("/api/v1/probe-monitors/probes/{probe_id}/recalibrate", "recalibrate_probe_monitor"),
    26: ("/api/v1/probe-monitors/probes/{probe_id}/score", "score_probe_monitor"),
    27: ("/api/v1/probe-monitors/probes/{probe_id}/score-batch", "score_probe_monitor_batch"),
    28: ("/api/v1/probe-monitors/probes/{probe_id}/definition", "build_probe_definition"),
    29: ("/api/v1/probe-monitors/behaviours/{key}/family/score-batch", "score_probe_family_batch"),
    30: ("/api/v1/steering/async/compare", "steer_compare"),
    31: ("/api/v1/steering/async/sweep", "steer_sweep"),
    32: ("/api/v1/steering/async/combined", "steer_combined"),
    33: ("/api/v1/trainings", "train_sae"),
    34: ("/api/v1/trainings/{training_id}/evaluate", "evaluate_training"),
    35: ("/api/v1/trainings/{training_id}/control", "resume_training"),
    36: ("/api/v1/saes/{sae_id}/extract-features", "extract_features"),
    37: ("/api/v1/saes/batch-extract-features", "batch_extract_features"),
    38: ("/api/v1/labeling", "start_labeling"),
    39: ("/api/v1/labeling/{extraction_job_id}/resume-sweep", "start_labeling_resume_sweep"),
    40: ("/api/v1/neuronpedia/export", "neuronpedia_export"),
    41: ("/api/v1/neuronpedia/compute-dashboard-data", "compute_dashboard_data"),
    42: ("/api/v1/neuronpedia/push-local", "push_to_neuronpedia_local"),
    43: ("/api/v1/task-queue/{task_queue_id}/retry", "retry_task"),
    44: ("/api/v1/probe-monitors/judge-runs", "start_probe_judge_run"),
    45: ("/api/v1/probe-monitors/spot-checks", "start_probe_spot_check"),
    46: ("/api/v1/labeling/trials", "run_labeling_trial"),
    47: ("/api/v1/labeling/panel", "start_labeling"),
    48: ("/api/v1/extractions/{extraction_id}/label", "start_labeling"),
    49: ("/api/v1/extractions/{extraction_id}/analyze-nlp", "trigger_nlp_analysis"),
    50: ("/api/v1/features/{feature_id}/label/enhanced", "run_enhanced_labeling"),
    51: ("/api/v1/extractions/{extraction_id}/label/enhanced/batch", "run_enhanced_labeling_batch"),
    52: ("/api/v1/probe-monitors/judge-prompts/try", "try_judge_prompt"),
    # ENH-010 044 (FR-1; 045 NEEDS N-4): the WSP held-out capture, a GPU start, gated with the approval class
    # `capture_wsp_heldout` (FTASKS 4.4, 2026-11-19; DEFERRED on 044-A, when phase 2 refused every agent).
    53: ("/api/v1/wsp/captures", "capture_wsp_heldout"),
    "P1": ("/api/v1/probe-monitors/probes/{probe_id}/publish", "publish_probe_definition"),
    "P2": ("/api/v1/probe-monitors/behaviours/{key}/family/publish", "publish_probe_family"),
    "P3": ("/api/v1/saes/upload", "upload_sae"),
    "P4": ("/api/v1/jlens/publish", "publish_jlens_artifact"),
    # ENH-010 044 FTASKS 14.2 (FR-14): a candidate import registers a new probe (CPU, queue `wsp`), so the GPU walk
    # never derives it; gated by name like the publishes. `W` rows are 044's non-GPU gated routes.
    "W1": ("/api/v1/wsp/candidates/import", "import_wsp_candidate"),
}

#: Design rows not gated on this branch yet, with the reason. Each must be gated by the task named.
DEFERRED: dict[Any, str] = {}  # rows 30 – 32 left it with FTASKS 21.6 (steering is caller-executed); row 53 with
# 044 FTASKS 4.4 (2026-11-19). A deferred row must refuse EVERY agent before its handler (R2 review fix OR2-044a-H1):
# the GPU backstop's sweep proves it (`BACKSTOP_UNREACHED`), and `test_every_gpu_route_is_gated` requires both.

#: Values merged into a row's generated body where its schema admits a value a validator refuses (a one-of, a
#: format, an enum checked in a validator), so the route's own parsing accepts the call.
BODY_OVERRIDES: dict[Any, dict] = {
    10: {"prompts": ["The capital of France is"]},
    16: {"vocabulary": {"capital": [" Paris"]}},
    19: {"repo_id": "fixture-org/fixture-model"},
    21: {"hook_types": ["residual"]},
    26: {"text": "fixture text"},
    27: {"inputs": [{"text": "fixture text"}]},
    29: {"inputs": [{"text": "fixture text"}]},
    33: {"model_id": "m_fixture"},
    35: {"action": "resume"},  # the arm: only resume is a GPU start
    38: {"labeling_method": "local"},
    39: {"config": {"labeling_method": "local"}},
    47: {"labeling_method": "local"},
    48: {"labeling_method": "local"},
    52: {"prompt_id": "jp_fixture"},
    "P3": {"repo_id": "fixture-org/fixture-sae"},
}

#: Query values a row needs (its arm reads them).
QUERY_OVERRIDES: dict[Any, dict] = {
    25: {"recut_windows": True},  # the arm: only the GPU re-cut is gated
}


def _merge(base: Any, patch: dict) -> Any:
    if not isinstance(base, dict):
        return patch
    out = dict(base)
    for key, value in patch.items():
        out[key] = _merge(out.get(key), value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _sample(schema: dict, defs: dict) -> Any:
    """The first value a JSON schema admits (required fields only), through `$ref`, `anyOf` and `oneOf`."""
    if not isinstance(schema, dict):
        return "x"
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return _sample(defs.get(ref.split("/")[-1], {}), defs)
    for key in ("anyOf", "oneOf"):
        options = [s for s in schema.get(key) or [] if s.get("type") != "null"]
        if options:
            return _sample(options[0], defs)
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "string":
        return "x" * max(1, int(schema.get("minLength", 1)))
    if kind == "integer":
        return int(max(schema.get("minimum", 1), schema.get("exclusiveMinimum", 0) + 1))
    if kind == "number":
        low = schema.get("minimum", schema.get("exclusiveMinimum", 0))
        high = schema.get("maximum", schema.get("exclusiveMaximum", low + 1))
        return (low + high) / 2
    if kind == "boolean":
        return False
    if kind == "array":
        return [_sample(schema.get("items", {}), defs) for _ in range(int(schema.get("minItems", 0)))]
    if kind == "object":
        props = schema.get("properties") or {}
        return {k: _sample(props[k], defs) for k in schema.get("required", []) if k in props}
    return "x"


def _model_sample(annotation: Any) -> Any:
    """The first value a body annotation admits, through its JSON schema."""
    from pydantic import TypeAdapter

    schema = TypeAdapter(annotation).json_schema()
    return _sample(schema, schema.get("$defs") or {})


def _routes(app):
    from src.core.approval_gate import iter_routes

    return {(m, path): route for path, route in iter_routes(app) for m in route.methods}


def sample_request(app, row: Any) -> dict:
    """`{"path": concrete path, "path_params", "query", "body"}` for one design row."""
    from src.core.approval_gate import _flat

    template, _action = DESIGN[row]
    route = _routes(app)[("POST", template)]
    path_params = {name: f"fx_{name}" for name in re.findall(r"\{([^}]+)\}", template)}
    flat = _flat(route)
    query = {}
    for param in flat.query_params:
        if param.field_info.is_required():
            query[param.alias] = _model_sample(param.field_info.annotation)
    query.update(QUERY_OVERRIDES.get(row, {}))
    if not flat.body_params:
        body = None
    elif len(flat.body_params) == 1:
        body = _model_sample(flat.body_params[0].field_info.annotation)
    else:
        body = {p.alias: _model_sample(p.field_info.annotation) for p in flat.body_params}
    if row in BODY_OVERRIDES:
        body = _merge(body, BODY_OVERRIDES[row])
    concrete = template
    for name, value in path_params.items():
        concrete = concrete.replace("{" + name + "}", value)
    return {"path": concrete, "path_params": path_params, "query": query, "body": body}
