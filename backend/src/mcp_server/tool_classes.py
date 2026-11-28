"""Every tool's title and honest annotations, in ONE table (045 FR-1 – FR-4; PADR IDL-64).

A tool with no row stops `build_server`: `apply_tool_classes` raises, so a new tool (or a new category, such as 044's
`wsp`) cannot ship unannotated. `tests/unit/test_tool_classes.py` reads the LIVE registry and checks every row against
the wire and against FR-3's honesty rules; the generated contract's ⚠️ comes from here, never from docstring words.

Kinds are CHAIN_PLAN A13's:
  * read: changes nothing. It starts no job, GPU work or model call, files no approval, spends no paid call and
    writes no row, file or artifact. A read that issues a POST names why in `why` (rule 4).
  * destructive: deletes, purges, clears or cancels; changes served weights in place; or changes a setting that
    enables later deletion. A `delete`, `purge`, `clear` or `cancel` word in the name, or a recorded DELETE call,
    makes a tool destructive (rules 2 and 3).
  * write: everything else.
Proxied tools carry their source app's hints (A13): `dataworks_*` rows are checked against miDataworks' published
table (FR-17), `miserver_*` rows follow these rules.

`result` is the tool's output envelope (FR-4). Every row is `object` today: a tool's list-ness belongs to the backend
route it relays (292 of the 342 tools when the table was seeded, 2026-11-16, answer whatever the route answers, a dict or a list), so the only envelope true of
every answer is the object envelope, which admits a list wrapped as `{"result": [...]}` (OVN-mcp-workbench-4).

Leaf module: no import from the mcp_server package, none from `src.*` (the MCP pod has no database; memory
`mcp-server-is-a-separate-deployment`).
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from mcp.types import ToolAnnotations

Kind = Literal["read", "write", "destructive"]
Result = Literal["list", "object", "none"]
Approval = Optional[Literal["call_again_with", "server_runs"]]
DecideIn = Optional[Literal["miworkbench", "miserver", "midataworks"]]


@dataclass(frozen=True)
class ToolClass:
    title: str
    kind: Kind
    result: Result = "object"
    approval: Approval = None
    decide_in: DecideIn = None
    why: str = ""


def _r(title: str, **kw: Any) -> ToolClass:
    return ToolClass(title, "read", **kw)


def _w(title: str, **kw: Any) -> ToolClass:
    return ToolClass(title, "write", **kw)


def _d(title: str, **kw: Any) -> ToolClass:
    return ToolClass(title, "destructive", **kw)


#: One row per tool name, by category then name. Seeded from the live registry on 2026-11-16 (FTASKS 2.1, I45-D1)
#: and checked against FTID Appendix A; every difference is recorded in the 045 controls record.
TOOL_CLASSES: dict[str, ToolClass] = {
    # ── admin ──
    'delete_experiment': _d('Delete experiment'),
    'delete_extraction': _d('Delete extraction'),
    # ── circuits ──
    'build_circuit_from_discovery': _w('Build circuit from discovery'),
    'calibrate_circuit_strength': _w('Calibrate circuit strength'),
    'cancel_attribution_pass': _d('Cancel attribution pass'),
    'cancel_circuit_calibration': _d('Cancel circuit calibration'),
    'cancel_circuit_capture': _d('Cancel circuit capture'),
    'cancel_circuit_discovery': _d('Cancel circuit discovery'),
    'cancel_circuit_faithfulness': _d('Cancel circuit faithfulness'),
    'cancel_edge_validation': _d('Cancel edge validation'),
    'cancel_steering_samples': _d('Cancel steering samples'),
    'confirm_circuit_capture': _w('Confirm circuit capture'),
    'create_circuit': _w('Create circuit'),
    'delete_circuit': _d('Delete circuit'),
    'export_circuit_definition': _r('Export circuit definition'),
    'export_circuit_slices': _r(
        'Export circuit slices',
        why='a pure POST: renders per-layer slices of a stored circuit, writes nothing (endpoints/circuits.py export-slices)',
    ),
    'get_circuit': _r('Get circuit'),
    'get_discovery_results': _r('Get discovery results'),
    'get_steering_samples': _r('Get steering samples'),
    'get_validation_manifest': _r('Get validation manifest'),
    'import_circuit_definition': _w('Import circuit definition'),
    'list_circuit_captures': _r('List circuit captures'),
    'list_circuits': _r('List circuits'),
    'list_validation_manifests': _r('List validation manifests'),
    'promote_circuit': _w('Promote circuit'),
    'record_steering_samples': _w('Record steering samples'),
    'reproduce_calibration': _w('Reproduce calibration'),
    'reproduce_validation': _w('Reproduce validation'),
    'run_attribution_pass': _w('Run attribution pass'),
    'run_circuit_discovery': _w('Run circuit discovery'),
    'run_circuit_faithfulness': _w('Run circuit faithfulness'),
    'start_circuit_capture': _w('Start circuit capture'),
    'update_circuit': _w('Update circuit'),
    'validate_circuit_edges': _w('Validate circuit edges'),
    # ── dataworks ──
    'dataworks_build_version_files': _w('miDataworks: build version files'),
    'dataworks_cancel_job': _d('miDataworks: cancel job'),
    'dataworks_delete_version': _d('miDataworks: delete version', approval='server_runs', decide_in='midataworks'),
    'dataworks_get_approval_status': _r('miDataworks: get approval status'),
    'dataworks_get_detector_send': _r('miDataworks: get detector send'),
    'dataworks_get_detector_set': _r('miDataworks: get detector set'),
    'dataworks_get_probe_results': _r('miDataworks: get probe results'),
    'dataworks_get_version': _r('miDataworks: get version'),
    'dataworks_get_version_manifest': _r(
        'miDataworks: get version manifest',
        why='FR-17, R2A-6: the relay makes one GET /api/v1/versions/{version_id}/handoff-manifest; its source also POSTs '
            '/publish-builds and GETs /publish-builds/{build_id} when no files exist, so the sets differ and the relay '
            'is read-only by its own call',
    ),
    'dataworks_job_status': _r('miDataworks: job status'),
    'dataworks_list_datasets': _r('miDataworks: list datasets'),
    'dataworks_list_detector_sends': _r(
        'miDataworks: list detector sends',
        why='FR-17: the relay GETs /api/v1/detector-sets/{set_id}/sends, its source GETs /api/v1/detector-sends; '
            'classified by its own GET',
    ),
    'dataworks_list_detector_sets': _r('miDataworks: list detector sets'),
    'dataworks_publish_version': _w('miDataworks: publish version', approval='server_runs', decide_in='midataworks'),
    'dataworks_refresh_probe_results': _w('miDataworks: refresh probe results'),
    'dataworks_reread_behaviour_definition': _w('miDataworks: reread behaviour definition'),
    'dataworks_resume_detector_send': _w(
        'miDataworks: resume detector send',
        approval='server_runs',
        decide_in='midataworks',
    ),
    'dataworks_send_detector_set': _w(
        'miDataworks: send detector set',
        approval='server_runs',
        decide_in='midataworks',
        why='FR-17: the relay POSTs /api/v1/detector-sets/{set_id}/send; its source also GETs '
            '/publish-builds/{build_id}; a write either way',
    ),
    'dataworks_start_label_run': _w('miDataworks: start label run', approval='server_runs', decide_in='midataworks'),
    # ── experiments ──
    'get_experiment': _r('Get experiment'),
    'list_experiments': _r('List experiments'),
    'save_experiment': _w('Save experiment'),
    # ── groups ──
    'cancel_feature_grouping': _d('Cancel feature grouping'),
    'compute_feature_groups': _w('Compute feature groups'),
    'find_features_by_token': _r('Find features by token'),
    'find_related_features': _r('Find related features'),
    'get_feature_group_members': _r('Get feature group members'),
    'get_feature_groups': _r('Get feature groups'),
    'get_grouping_status': _r('Get grouping status'),
    # ── jlens ──
    'acquire_jlens_artifact': _w('Acquire J-lens artifact'),
    'annotate_jlens_feature': _w('Annotate J-lens feature'),
    'build_jlens_template_lens': _w('Build J-lens template lens'),
    'cancel_jlens_task': _d('Cancel J-lens task'),
    'compute_jlens_band_report': _w('Compute J-lens band report'),
    'create_jlens_watchlist': _w('Create J-lens watchlist'),
    'fit_jlens_artifact': _w('Fit J-lens artifact'),
    'get_jlens_artifact_document': _r('Get J-lens artifact document'),
    'get_jlens_band_report': _r('Get J-lens band report'),
    'get_jlens_decomposition': _r('Get J-lens decomposition'),
    'get_jlens_gate': _r('Get J-lens gate'),
    'get_jlens_interventions': _r('Get J-lens interventions'),
    'get_jlens_probe': _r('Get J-lens probe'),
    'get_jlens_readout': _r('Get J-lens readout'),
    'get_jlens_replication_report': _r('Get J-lens replication report'),
    'get_jlens_template_lens': _r('Get J-lens template lens'),
    'jlens_attention_readout': _w('J-lens attention readout'),
    'jlens_cost_estimate': _r('J-lens cost estimate'),
    'jlens_decompose': _w('J-lens decompose'),
    'jlens_probe': _w('J-lens probe'),
    'jlens_readout': _w('J-lens readout'),
    'jlens_token_check': _r(
        'J-lens token check',
        why="a pure POST: resolves strings against the model's tokenizer, stores nothing",
    ),
    'list_jlens_artifacts': _r('List J-lens artifacts'),
    'list_jlens_disagreements': _r('List J-lens disagreements'),
    'preview_jlens_repo': _r(
        'Preview J-lens repo',
        why="a pure POST (a token travels in the body, not the query): lists a Hub repository's lens files, stores nothing",
    ),
    'publish_jlens_artifact': _w('Publish J-lens artifact'),
    'record_jlens_gate': _w('Record J-lens gate'),
    'restore_jlens_artifact': _w('Restore J-lens artifact'),
    'revalidate_jlens_artifact': _w('Revalidate J-lens artifact'),
    'run_jlens_intervention': _w('Run J-lens intervention'),
    'start_jlens_annotation_sweep': _w('Start J-lens annotation sweep'),
    'validate_jlens_artifact': _w('Validate J-lens artifact'),
    'validate_jlens_document': _r(
        'Validate J-lens document',
        why='a pure POST: validates a document by its kind, stores NOTHING',
    ),
    # ── jobs ──
    'get_gpu_status': _r('Get GPU status'),
    'get_task_status': _r('Get task status'),
    'gpu_preflight': _r('GPU preflight'),
    'list_active_jobs': _r('List active jobs'),
    'list_gpus': _r('List GPUs'),
    # ── labeling ──
    'cancel_labeling_resume_sweep': _d('Cancel labeling resume sweep'),
    'compare_labeling_trials': _r('Compare labeling trials'),
    'get_enhanced_label': _r('Get enhanced label'),
    'get_labeling_coverage': _r('Get labeling coverage'),
    'get_labeling_resume_sweep': _r('Get labeling resume sweep'),
    'get_labeling_trial': _r('Get labeling trial'),
    'list_labeling_templates': _r('List labeling templates'),
    'list_labeling_trials': _r('List labeling trials'),
    'resume_labeling': _w('Resume labeling'),
    'run_enhanced_labeling': _w('Run enhanced labeling'),
    'run_labeling_trial': _w('Run labeling trial'),
    'start_labeling_resume_sweep': _w('Start labeling resume sweep'),
    'update_feature_label': _w('Update feature label'),
    # ── miserver_circuits ──
    'miserver_activate_circuit': _w('miServer: activate circuit'),
    'miserver_circuit_claims': _r('miServer: circuit claims'),
    'miserver_circuit_sensing_clear': _d('miServer: circuit sensing clear'),
    'miserver_circuit_sensing_disable': _w('miServer: circuit sensing disable'),
    'miserver_circuit_sensing_enable': _w('miServer: circuit sensing enable'),
    'miserver_circuit_sensing_event': _r('miServer: circuit sensing event'),
    'miserver_circuit_sensing_events': _r('miServer: circuit sensing events'),
    'miserver_circuit_sensing_status': _r('miServer: circuit sensing status'),
    'miserver_circuit_status': _r('miServer: circuit status'),
    'miserver_deactivate_circuit': _w('miServer: deactivate circuit'),
    'miserver_delete_circuit': _d('miServer: delete circuit'),
    'miserver_export_circuit': _r('miServer: export circuit'),
    'miserver_import_circuit': _w('miServer: import circuit'),
    'miserver_list_circuits': _r('miServer: list circuits'),
    'miserver_release_circuit_claims': _w('miServer: release circuit claims'),
    'miserver_set_circuit_intensity': _w('miServer: set circuit intensity'),
    # ── miserver_clusters ──
    'miserver_activate_cluster': _w('miServer: activate cluster'),
    'miserver_cluster_hub_definitions': _r('miServer: cluster hub definitions'),
    'miserver_deactivate_cluster': _w('miServer: deactivate cluster'),
    'miserver_export_cluster': _r('miServer: export cluster'),
    'miserver_hub_search': _r('miServer: hub search'),
    'miserver_import_cluster': _w('miServer: import cluster'),
    'miserver_list_clusters': _r('miServer: list clusters'),
    # ── miserver_inference ──
    'miserver_batch_lease': _w('miServer: batch lease'),
    'miserver_batch_results': _r('miServer: batch results'),
    'miserver_batch_status': _r('miServer: batch status'),
    'miserver_cancel_batch': _d('miServer: cancel batch'),
    'miserver_embeddings': _w('miServer: embeddings'),
    'miserver_generate': _w('miServer: generate'),
    'miserver_score_chat': _w('miServer: score chat'),
    'miserver_score_completion': _w('miServer: score completion'),
    'miserver_submit_batch': _w('miServer: submit batch'),
    'miserver_upload_batch_file': _w('miServer: upload batch file'),
    # ── miserver_models ──
    'miserver_acquire_lease': _w('miServer: acquire lease'),
    'miserver_cancel_download': _d('miServer: cancel download'),
    'miserver_delete_model': _d('miServer: delete model', approval='server_runs', decide_in='miserver'),
    'miserver_download_model': _w('miServer: download model'),
    'miserver_get_approval_status': _r('miServer: get approval status'),
    'miserver_get_model': _r('miServer: get model'),
    'miserver_lease_status': _r('miServer: lease status'),
    'miserver_list_approvals': _r('miServer: list approvals'),
    'miserver_list_models': _r('miServer: list models'),
    'miserver_load_model': _w('miServer: load model', approval='server_runs', decide_in='miserver'),
    'miserver_preview_model_repo': _r(
        'miServer: preview model repo',
        why="a pure POST: miServer lists a Hub repository's files, nothing is downloaded or stored",
    ),
    'miserver_release_lease': _d('miServer: release lease', why='rule 3: issues a DELETE (releases a lease)'),
    'miserver_renew_lease': _w('miServer: renew lease'),
    'miserver_set_model_context_ceiling': _w(
        'miServer: set model context ceiling', approval='server_runs', decide_in='miserver'),
    'miserver_unload_model': _w('miServer: unload model', approval='server_runs', decide_in='miserver'),
    # ── miserver_probes ──
    'miserver_arm_behaviour': _w('miServer: arm behaviour'),
    'miserver_arm_probe': _w('miServer: arm probe'),
    'miserver_clear_behaviour_rule_signals': _d('miServer: clear behaviour rule signals'),
    'miserver_clear_behaviour_verdict': _d('miServer: clear behaviour verdict'),
    'miserver_clear_probe_events': _d('miServer: clear probe events'),
    'miserver_delete_probe': _d('miServer: delete probe'),
    'miserver_delete_rule_set': _d('miServer: delete rule set'),
    'miserver_disarm_behaviour': _w('miServer: disarm behaviour'),
    'miserver_disarm_probe': _w('miServer: disarm probe'),
    'miserver_evaluate_rule_set': _r(
        'miServer: evaluate rule set',
        why="a pure POST: runs one stored rule set on the caller's text in miServer, nothing is stored",
    ),
    'miserver_get_behaviour_rule_signals': _r('miServer: get behaviour rule signals'),
    'miserver_get_behaviour_verdict': _r('miServer: get behaviour verdict'),
    'miserver_get_probe': _r('miServer: get probe'),
    'miserver_get_probe_event': _r('miServer: get probe event'),
    'miserver_get_probe_limit_recommendation': _r('miServer: get probe limit recommendation'),
    'miserver_get_rule_set': _r('miServer: get rule set'),
    'miserver_import_probe': _w('miServer: import probe'),
    'miserver_import_rule_set': _w('miServer: import rule set', approval='server_runs', decide_in='miserver'),
    'miserver_list_behaviours': _r('miServer: list behaviours'),
    'miserver_list_probes': _r('miServer: list probes'),
    'miserver_list_rule_sets': _r('miServer: list rule sets'),
    'miserver_list_rule_validators': _r('miServer: list rule validators'),
    'miserver_measure_probe_capacity': _w(
        'miServer: measure probe capacity',
        approval='server_runs',
        decide_in='miserver',
    ),
    'miserver_probe_capacity': _r('miServer: probe capacity'),
    'miserver_probe_events': _r('miServer: probe events'),
    'miserver_probe_hub_definitions': _r('miServer: probe hub definitions'),
    'miserver_probe_hub_search': _r('miServer: probe hub search'),
    'miserver_probe_parity': _w('miServer: probe parity'),
    'miserver_probe_status': _r('miServer: probe status'),
    'miserver_recalibrate_probe': _w('miServer: recalibrate probe'),
    'miserver_request_probe_limits': _w(
        'miServer: request probe limits',
        approval='server_runs',
        decide_in='miserver',
    ),
    'miserver_score_probes': _w('miServer: score probes'),
    'miserver_set_behaviour_rule_signals': _w('miServer: set behaviour rule signals'),
    'miserver_set_behaviour_verdict': _w('miServer: set behaviour verdict'),
    'miserver_validate_rule_set': _r(
        'miServer: validate rule set',
        why='a pure POST: miServer validates the document and its cases, writing nothing',
    ),
    # ── miserver_runtime ──
    'miserver_activate_profile': _w('miServer: activate profile'),
    'miserver_deactivate_profile': _w('miServer: deactivate profile'),
    'miserver_get_environment': _r('miServer: get environment'),
    'miserver_get_environment_history': _r('miServer: get environment history'),
    'miserver_list_profiles': _r('miServer: list profiles'),
    'miserver_set_intensity': _w('miServer: set intensity'),
    'miserver_set_telemetry': _w('miServer: set telemetry', approval='server_runs', decide_in='miserver'),
    'miserver_status': _r('miServer: what is steering right now'),
    # ── miserver_sensing ──
    'miserver_get_sensing_event': _r('miServer: get sensing event'),
    'miserver_sensing_clear': _d('miServer: sensing clear'),
    'miserver_sensing_config': _w('miServer: sensing config'),
    'miserver_sensing_disable': _w('miServer: sensing disable'),
    'miserver_sensing_enable': _w('miServer: sensing enable'),
    'miserver_sensing_events': _r('miServer: sensing events'),
    'miserver_sensing_status': _r('miServer: sensing status'),
    # ── models ──
    'cancel_model_download': _d('Cancel model download'),
    'delete_model': _d('Delete model'),
    'download_model': _w('Download model'),
    'get_model': _r('Get model'),
    'get_model_architecture': _r('Get model architecture'),
    'list_models': _r('List models'),
    # ── probe_monitors ──
    'archive_behaviour': _w('Archive behaviour'),
    'archive_judge_prompt': _w('Archive judge prompt'),
    'archive_probe_dataset': _w('Archive probe dataset'),
    'backfill_probe_truncation': _w('Backfill probe truncation'),
    'build_probe_definition': _w('Build probe definition'),
    'cancel_blend_evaluation': _d('Cancel blend evaluation'),
    'cancel_dataset_download': _d('Cancel dataset download'),
    'cancel_detector_rules_evaluation': _d('Cancel detector rules evaluation'),
    'cancel_probe_judge_run': _d('Cancel probe judge run'),
    'cancel_probe_spot_check': _d('Cancel probe spot check'),
    'cancel_probe_training_run': _d('Cancel probe training run'),
    'clone_judge_prompt': _w('Clone judge prompt'),
    'compare_detectors_across_models': _r('Compare detectors across models'),
    'create_behaviour': _w('Create behaviour'),
    'create_judge_prompt': _w('Create judge prompt'),
    'create_nonsense_text_dataset': _w('Create nonsense text dataset'),
    'delete_judge_prompt': _d('Delete judge prompt'),
    'delete_probe_training_run': _d('Delete probe training run'),
    'download_dataset': _w('Download dataset'),
    'evaluate_detector_with_rules': _w('Evaluate detector with rules'),
    'evaluate_probe_blends': _w('Evaluate probe blends'),
    'evaluate_probe_monitor': _w('Evaluate probe monitor', approval='call_again_with', decide_in='miworkbench'),
    'evaluate_rule_signals': _r(
        'Evaluate rule signals',
        why='a pure POST: one text through the rule engine, nothing is stored or logged',
    ),
    'export_behaviour_verdict': _r(
        'Export behaviour verdict',
        why='a pure POST: builds the verdict document, nothing is stored or started',
    ),
    'export_judge_prompt': _r('Export judge prompt'),
    'export_probe_definition': _r('Export probe definition'),
    'export_probe_family': _r('Export probe family'),
    'forecast_probe_training_run': _r(
        'Forecast probe training run',
        why='a pure POST: tokenises on the CPU to forecast host memory; no GPU, no row, no task',
    ),
    'get_behaviour': _r('Get behaviour'),
    'get_behaviour_checklist': _r('Get behaviour checklist'),
    'get_behaviour_history': _r('Get behaviour history'),
    'get_blend_report': _r('Get blend report'),
    'get_blend_report_scores': _r('Get blend report scores'),
    'get_detector_rules_evaluation': _r('Get detector rules evaluation'),
    'get_detector_success_report': _w(
        'Get detector success report',
        why='POSTs and stores its report immutably (endpoints/behaviour_detectors.py; R1A-15)',
    ),
    'get_judge_prompt': _r('Get judge prompt'),
    'get_probe_breakdown': _r('Get probe breakdown'),
    'get_probe_dataset': _r('Get probe dataset'),
    'get_probe_dataset_rows': _r('Get probe dataset rows'),
    'get_probe_evaluation_rows': _r('Get probe evaluation rows'),
    'get_probe_evaluation_scores': _r('Get probe evaluation scores'),
    'get_probe_family': _r('Get probe family'),
    'get_probe_family_score_result': _r('Get probe family score result'),
    'get_probe_monitor_report': _r('Get probe monitor report'),
    'get_probe_score_result': _r('Get probe score result'),
    'get_probe_spot_check': _r('Get probe spot check'),
    'get_probe_training_run': _r('Get probe training run'),
    'get_spot_check_availability': _r('Get spot check availability'),
    'get_suggested_blend': _r('Get suggested blend'),
    'import_judge_prompt': _w('Import judge prompt'),
    'import_judge_prompt_from_rubric': _w('Import judge prompt from rubric'),
    'link_behaviour_view': _w('Link behaviour view'),
    'list_behaviour_versions': _r('List behaviour versions'),
    'list_behaviours': _r('List behaviours'),
    'list_blend_reports': _r('List blend reports'),
    'list_detector_rules_evaluations': _r('List detector rules evaluations'),
    'list_detector_success_reports': _r('List detector success reports'),
    'list_judge_prompts': _r('List judge prompts'),
    'list_midataworks_rubrics_for_judging': _r('List midataworks rubrics for judging'),
    'list_probe_datasets': _r('List probe datasets'),
    'list_probe_judge_runs': _r('List probe judge runs'),
    'list_probe_monitors': _r('List probe monitors'),
    'list_probe_spot_checks': _r('List probe spot checks'),
    'list_probe_training_runs': _r('List probe training runs'),
    'new_behaviour_version': _w('New behaviour version'),
    'new_judge_prompt_version': _w('New judge prompt version'),
    'preview_judge_prompt': _r(
        'Preview judge prompt',
        why="a pure POST: renders the judge's messages for a few rows, nothing is sent to any model",
    ),
    'publish_probe_definition': _w('Publish probe definition'),
    'publish_probe_family': _w('Publish probe family'),
    'read_detector_success_report': _r('Read detector success report'),
    'recalibrate_probe_monitor': _w('Recalibrate probe monitor', approval='call_again_with', decide_in='miworkbench'),
    'recommend_probe_rules': _r('Recommend probe rules'),
    'register_probe_dataset': _w('Register probe dataset'),
    'request_code_rule_approval': _w('Request code rule approval', approval='call_again_with', decide_in='miworkbench'),
    'rerun_blend_report': _w('Rerun blend report'),
    'score_probe_family': _w('Score probe family'),
    'score_probe_family_batch': _w('Score probe family batch'),
    'score_probe_monitor': _w('Score probe monitor'),
    'score_probe_monitor_batch': _w('Score probe monitor batch'),
    'set_behaviour_probe_role': _w('Set behaviour probe role'),
    'set_probe_behaviour': _w('Set probe behaviour'),
    'set_suggested_blend': _w('Set suggested blend'),
    'start_probe_judge_run': _w('Start probe judge run', approval='call_again_with', decide_in='miworkbench'),
    'start_probe_spot_check': _w('Start probe spot check', approval='call_again_with', decide_in='miworkbench'),
    'start_probe_training_run': _w('Start probe training run', approval='call_again_with', decide_in='miworkbench'),
    'try_judge_prompt': _w('Try judge prompt'),
    'unarchive_behaviour': _w('Unarchive behaviour'),
    'unarchive_judge_prompt': _w('Unarchive judge prompt'),
    'unarchive_probe_dataset': _w('Unarchive probe dataset'),
    'unlink_behaviour_view': _d('Unlink behaviour view', why='rule 3: issues a DELETE (removes a view link)'),
    'update_judge_prompt': _w('Update judge prompt'),
    'update_probe_dataset_columns': _w('Update probe dataset columns'),
    'validate_rule_set': _w(
        'Validate rule set',
        why='not read-only: the validate route stores inline code documents ("Stores nothing but inline code documents")',
    ),
    # ── profiles ──
    'export_cluster_definition': _r('Export cluster definition'),
    'get_cluster_profile': _r('Get cluster profile'),
    'list_cluster_profiles': _r('List cluster profiles'),
    'save_cluster_profile': _w('Save cluster profile'),
    # ── read ──
    'get_approval_status': _r('Get approval status'),
    'get_dataset': _r('Get dataset'),
    'get_dataset_rows': _r('Get dataset rows'),
    'get_environment_settings': _r('Get environment settings'),
    'get_extraction_summary': _r('Get extraction summary'),
    'get_feature': _r('Get feature'),
    'get_feature_ablation': _w(
        'Get feature ablation',
        why='A13: writes an analysis cache row (services/analysis_service.py _cache_analysis)',
    ),
    'get_feature_by_index': _r('Get feature by index'),
    'get_feature_correlations': _w(
        'Get feature correlations',
        why='A13: writes an analysis cache row (services/analysis_service.py _cache_analysis)',
    ),
    'get_feature_examples': _r('Get feature examples'),
    'get_feature_logit_lens': _w(
        'Get feature logit lens',
        why='A13: writes an analysis cache row (services/analysis_service.py _cache_analysis) and runs the SAE decoder and model head on the CPU',
    ),
    'get_feature_nlp_analysis': _r('Get feature NLP analysis'),
    'get_feature_token_analysis': _r('Get feature token analysis'),
    'get_sae': _r('Get SAE'),
    'get_training': _r('Get training'),
    'list_datasets': _r('List datasets'),
    'list_extractions': _r('List extractions'),
    'list_saes': _r('List SAEs'),
    'list_trainings': _r('List trainings'),
    'miworkbench_howto': _r('miWorkbench guide: workflows and topics (start here)'),
    'search_features': _r('Search features'),
    # ── steering ──
    'cancel_steering_task': _d('Cancel steering task'),
    'compute_cluster_allocation': _r(
        'Compute cluster allocation',
        why='a pure POST: CPU weight-only SAE load and arithmetic, no row, no task (endpoints/steering.py cluster-allocation)',
    ),
    'enter_steering_mode': _w('Enter steering mode'),
    'exit_steering_mode': _w('Exit steering mode'),
    'get_steering_mode': _r('Get steering mode'),
    'get_steering_result': _r('Get steering result'),
    'steer_combined': _w('Steer combined', approval='call_again_with', decide_in='miworkbench'),
    'steer_compare': _w('Steer compare', approval='call_again_with', decide_in='miworkbench'),
    'steer_sweep': _w('Steer sweep', approval='call_again_with', decide_in='miworkbench'),
    'steering_status': _r('Steering status'),
    # ── wsp ──
    'capture_wsp_heldout': _w('Capture WSP held-out activations', approval='call_again_with', decide_in='miworkbench'),
    'import_wsp_candidate': _w('Import WSP candidate', approval='call_again_with', decide_in='miworkbench'),
}


def apply_tool_classes(mcp: Any) -> int:
    """Set every registered tool's `title` and annotations from `TOOL_CLASSES` (FR-2). Called by `build_server`
    after every category registered; removing that call fails
    `test_tool_classes.py::test_every_tool_is_annotated_on_the_wire`.

    Raises `RuntimeError` naming every tool with no row: a server with an unclassified tool does not build.
    Both hints are set explicitly (True or False, never None)."""
    manager = mcp._tool_manager  # noqa: SLF001 — no public hook for this in the SDK (as server.py)
    tools = manager.list_tools()
    missing = sorted(t.name for t in tools if t.name not in TOOL_CLASSES)
    if missing:
        raise RuntimeError(f"tools with no row in TOOL_CLASSES (mcp_server/tool_classes.py): {missing}")
    for tool in tools:
        row = TOOL_CLASSES[tool.name]
        tool.title = row.title
        tool.annotations = ToolAnnotations(
            title=row.title,
            read_only_hint=row.kind == "read",
            destructive_hint=row.kind == "destructive",
        )
    return len(tools)


#: The non-error shapes every tool may answer besides its resource (FR-4): `{"unavailable", "reason"}`,
#: `{"refused", …}`, and a pending approval in any of its three spellings.
_ENVELOPE_ALTERNATIVES: list[dict[str, Any]] = [
    {"required": ["unavailable"]},
    {"required": ["refused"]},
    {"required": ["approval_id"]},
    {"required": ["approval_request_id"]},
    {"required": ["approval"]},
]


def envelope_schema(kind: Result, contract: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """The output schema for a row's `result` kind (FTDD §6.1). `none` has none.

    An object envelope admits any object; a list envelope admits `{"result": [...]}` or one of the other non-error
    shapes. Phase B passes the vendored contract to add `approval` and `next_steps`. NOTHING EVER ADDS A `next`
    PROPERTY (A30; R1A-14): the contract defines none, and today's `next` is a string in some answers and an object
    or a list in others, so constraining it would make a client raise on a valid answer."""
    if kind == "none":
        return None
    if kind == "list":
        schema: dict[str, Any] = {
            "type": "object",
            "anyOf": [{"required": ["result"], "properties": {"result": {"type": "array"}}},
                      *_ENVELOPE_ALTERNATIVES],
        }
    else:
        schema = {"type": "object"}
    if contract is not None:
        schema.setdefault("properties", {}).update(result_language_properties(contract))
    return schema


#: The result-language keys every structured answer may carry (the vendored contract's `result_fields`).
RESULT_LANGUAGE_KEYS = ("approval", "next_steps")


def result_language_properties(contract: dict[str, Any]) -> dict[str, Any]:
    """`{"approval": …, "next_steps": …}` from the vendored contract's `result_fields`, inlined, with no `$ref` left
    (the SDK client raises on a reference it cannot resolve, `mcp/client/session.py:1150`), and WITHOUT the
    contract's `description` strings (046 WT6, BR-7): they repeated up to 2,266 bytes in every one of 346 tools'
    schemas, and the contract itself is where they are read. The one definition both the generic envelope and the
    generated typed schemas (`scripts/generate_output_schemas.py`) use."""
    defs = contract.get("$defs") or {}
    fields = (defs.get("result_fields") or {}).get("properties") or {}
    return {key: without_offset_formats(_without_descriptions(_inline(fields[key], defs)))
            for key in RESULT_LANGUAGE_KEYS if key in fields}


#: The `format` values an advertised output schema never carries (hotfix fix/046-1-datetime-format, 2026-11-26). Both
#: need an RFC 3339 offset (`Z` or `+hh:mm`), and the backend serialises naive datetimes without one
#: (`2026-11-26T10:15:00.123456`): an MCP client that asserts formats (the production JS client, ajv) then rejects the
#: WHOLE answer, as it did every `list_cluster_profiles` row ("data/data/0/created_at must match format date-time").
#: Removing the annotation changes no route's served bytes; emitting `Z` instead would change every REST answer for
#: every consumer. The value stays `{"type": "string"}`. The contracts themselves (`docs/schemas/`, the vendored
#: documents) keep their formats: only what a tool ADVERTISES is stripped.
OFFSET_FORMATS = frozenset({"date-time", "time"})


def without_offset_formats(schema: Any) -> Any:
    """`schema` with every `format` KEYWORD in `OFFSET_FORMATS` removed (a property NAMED `format` is kept)."""
    if isinstance(schema, list):
        return [without_offset_formats(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "format" and isinstance(value, str) and value in OFFSET_FORMATS:
            continue
        if key in ("properties", "$defs", "patternProperties", "dependentSchemas") and isinstance(value, dict):
            out[key] = {name: without_offset_formats(sub) for name, sub in value.items()}
        else:
            out[key] = without_offset_formats(value)
    return out


def _without_descriptions(schema: Any) -> Any:
    """`schema` with every `description` KEYWORD removed (a property NAMED `description` is kept)."""
    if isinstance(schema, list):
        return [_without_descriptions(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "description" and isinstance(value, str):
            continue
        if key in ("properties", "$defs", "patternProperties", "dependentSchemas") and isinstance(value, dict):
            out[key] = {name: _without_descriptions(sub) for name, sub in value.items()}
        else:
            out[key] = _without_descriptions(value)
    return out


def _inline(schema: Any, defs: dict[str, Any]) -> Any:
    """`schema` with every `#/$defs/<name>` reference replaced by a copy of that definition (the contract has no
    recursive definitions)."""
    if isinstance(schema, list):
        return [_inline(item, defs) for item in schema]
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        rest = {k: v for k, v in schema.items() if k != "$ref"}
        return {**_inline(defs[ref.split("/")[-1]], defs), **_inline(rest, defs)}
    return {key: _inline(value, defs) for key, value in schema.items()}


#: The generated per-tool output schemas (`backend/scripts/generate_output_schemas.py`; 045 follow-up, debt RL-1).
OUTPUT_SCHEMAS_PATH = Path(__file__).parent / "output_schemas.json"


@functools.lru_cache(maxsize=1)
def load_output_schemas() -> dict[str, dict[str, Any]]:
    """`{tool: schema}` of the tools typed by their route's body, from the generated file shipped in this package (it
    imports nothing from the backend). Raises when the file is missing: it ships in the image the MCP pod runs, so a
    missing copy is a packaging defect that must fail the build, never a run-time state."""
    tools: dict[str, dict[str, Any]] = json.loads(OUTPUT_SCHEMAS_PATH.read_text(encoding="utf-8"))["tools"]
    return tools


def output_schema(name: str, contract: Optional[dict[str, Any]] = None,
                  typed: Optional[dict[str, dict[str, Any]]] = None) -> Optional[dict[str, Any]]:
    """The `outputSchema` a tool advertises: its generated typed schema WHOLE, when it has one and its row's envelope
    is `object` (046 FR-046.23: the generator builds the answer, the alternatives and the result-language
    `approval` and `next_steps`, `scripts/generate_output_schemas.py`); else the envelope (045 FR-4)."""
    kind = TOOL_CLASSES[name].result
    envelope = envelope_schema(kind, contract)
    own = (load_output_schemas() if typed is None else typed).get(name)
    if own is None or envelope is None or kind != "object":
        return envelope
    return own


def apply_output_schemas(mcp: Any, contract: Optional[dict[str, Any]] = None,
                         typed: Optional[dict[str, dict[str, Any]]] = None) -> int:
    """Advertise each structured row's `outputSchema` (FR-4): its typed schema where the generated file has one (045
    follow-up, RL-1), else the envelope. Called by `build_server` AFTER `normalise_results`, which wraps only tools
    without an output schema: the other order would leave every tool unwrapped. A typed schema naming a tool with no
    row is a stale file and stops the build. Returns how many tools gained a schema."""
    typed = load_output_schemas() if typed is None else typed
    stale = sorted(set(typed) - set(TOOL_CLASSES))
    if stale:
        raise RuntimeError(f"output_schemas.json types tools with no row in TOOL_CLASSES: {stale}. Regenerate: "
                           "cd backend && python -m scripts.generate_output_schemas --write")
    manager = mcp._tool_manager  # noqa: SLF001
    count = 0
    for tool in manager.list_tools():
        schema = output_schema(tool.name, contract, typed)
        if schema is None:
            continue
        tool.fn_metadata.output_schema = schema
        tool.__dict__.pop("output_schema", None)  # the SDK caches it (`cached_property`, tools/base.py)
        count += 1
    return count
