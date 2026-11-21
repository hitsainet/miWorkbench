/**
 * The shared approvals vocabulary (ENH-008 §2.1, §2.2, §2.5), mapped onto miWorkbench's tools and
 * statuses. Every string the operator reads about a request's kind or outcome comes from here.
 */
import type { ApprovalRequest } from '../../types/featureGroups';

export const STEERING_TOOLS = new Set(['steer_compare', 'steer_sweep', 'steer_combined']);
/** Approving WRITES a durable record (an operator approval per code hash); never consumed. */
export const RECORD_TOOLS = new Set(['code_rule_approve']);
export const PROBE_RUN_TOOL = 'start_probe_training_run';
export const CODE_RULE_TOOL = 'code_rule_approve';
/** ENH-010 044: the two weight-space actions; the route spends each approval. */
export const WSP_CAPTURE_TOOL = 'capture_wsp_heldout';
export const WSP_IMPORT_TOOL = 'import_wsp_candidate';

/** Plain-words label per tool. Old `millm_*` rows (before miServer decided its own) stay readable in History. (p9: keep, rename history) */
export const ACTION_LABELS: Record<string, string> = {
  steer_compare: 'Compare steering on several features',
  steer_sweep: "Sweep a feature's steering strength",
  steer_combined: 'Steer several features together',
  start_probe_training_run: 'Train probes on a graphics card',
  evaluate_probe_monitor: 'Evaluate a probe on a graphics card',
  start_probe_judge_run: 'Judge probe datasets with miServer',
  start_probe_spot_check: 'Spot-check a behaviour with miServer',
  recalibrate_probe_monitor: "Re-score a probe's calibration on a graphics card",
  code_rule_approve: 'Run operator code in the isolated runner',
  // ENH-010 044 (FTASKS 4.4, 14.2): filed by the `wsp` tools or an agent's REST call, spent by the route.
  capture_wsp_heldout: "Capture a probe's held-out activations on a graphics card",
  import_wsp_candidate: 'Import a weight-space candidate as a new probe',
  // 045 FR-13: actions an agent's REST call files (FTID §11a's table). The card's route line says which route.
  start_circuit_capture: 'Capture circuit activations on a graphics card',
  confirm_circuit_capture: 'Run a confirmed circuit capture on a graphics card',
  run_attribution_pass: 'Run a circuit attribution pass on a graphics card',
  validate_circuit_edges: 'Validate circuit edges with interventions on a graphics card',
  reproduce_validation: 'Reproduce a circuit validation on a graphics card',
  run_circuit_faithfulness: "Measure a circuit's faithfulness on a graphics card",
  calibrate_circuit_strength: "Calibrate a circuit's strength on a graphics card",
  reproduce_calibration: 'Reproduce a circuit calibration on a graphics card',
  record_steering_samples: 'Record steering samples on a graphics card',
  fit_jlens_artifact: 'Fit a J-lens on a graphics card',
  revalidate_jlens_artifact: 'Re-validate a staged J-lens on a graphics card',
  compute_jlens_band_report: 'Compute a J-lens band report on a graphics card',
  jlens_readout: 'Run a J-lens readout on a graphics card',
  jlens_probe: 'Run a J-lens probe on a graphics card',
  jlens_decompose: 'Decompose a J-lens readout on a graphics card',
  build_jlens_template_lens: 'Build a J-lens template lens on a graphics card',
  run_jlens_intervention: 'Run a J-lens intervention on a graphics card',
  acquire_jlens_artifact: 'Download a J-lens from the Hugging Face Hub',
  download_model: 'Download a model',
  redownload_model: 'Download a model again',
  extract_activations: 'Extract activations on a graphics card',
  retry_extraction: 'Retry an activation extraction on a graphics card',
  score_probe_monitor: 'Score text with a probe on a graphics card',
  score_probe_monitor_batch: 'Score several texts with a probe on a graphics card',
  build_probe_definition: "Build a probe's portable definition on a graphics card",
  score_probe_family_batch: 'Score texts with a probe family on a graphics card',
  train_sae: 'Train an SAE on a graphics card',
  evaluate_training: 'Evaluate an SAE training on a graphics card',
  resume_training: 'Resume an SAE training on a graphics card',
  extract_features: "Extract an SAE's features on a graphics card",
  batch_extract_features: "Extract several SAEs' features on a graphics card",
  start_labeling: 'Label features with a judge model',
  start_labeling_resume_sweep: 'Resume labeling across several batches',
  neuronpedia_export: 'Export to Neuronpedia (uses a graphics card)',
  compute_dashboard_data: 'Compute Neuronpedia dashboard data on a graphics card',
  push_to_neuronpedia_local: 'Push features to the local Neuronpedia',
  retry_task: 'Retry a queued task',
  run_labeling_trial: 'Run a labeling prompt trial with a judge model',
  trigger_nlp_analysis: 'Analyse features and release queued extractions',
  run_enhanced_labeling: 'Label a feature with a judge model (enhanced)',
  run_enhanced_labeling_batch: 'Label several features with a judge model (enhanced)',
  try_judge_prompt: 'Try a judge prompt on a judge model',
  publish_probe_definition: 'Publish a probe to the Hugging Face Hub',
  publish_probe_family: 'Publish a probe family to the Hugging Face Hub',
  upload_sae: 'Upload an SAE to the Hugging Face Hub',
  publish_jlens_artifact: 'Publish a J-lens to the Hugging Face Hub',
  millm_load_model: 'Load a model in miServer',
  millm_request_probe_limits: "Change a model's probe limits in miServer",
  millm_measure_probe_capacity: "Measure a model's probe capacity in miServer",
  millm_import_rule_set: 'Import a rule set with operator code into miServer',
};

export function actionLabel(tool: string): string {
  return ACTION_LABELS[tool] ?? tool;
}

/** "Approve {action, lower case}" (§2.1 row 8). */
export function approveLabel(tool: string): string {
  const label = actionLabel(tool);
  return `Approve ${label.charAt(0).toLowerCase()}${label.slice(1)}`;
}

/** "asked by agent mcp" / "asked by an agent" / "asked by the operator" (§2.1 row 2). */
export function askedBy(requestedBy: string | null | undefined): string {
  if (!requestedBy) return 'asked by an agent';
  if (requestedBy === 'operator') return 'asked by the operator';
  return `asked by agent ${requestedBy}`;
}

/** The §2.2 outcome label for one of miWorkbench's rows. */
export function outcomeLabel(r: ApprovalRequest): string {
  switch (r.status) {
    case 'pending':
      return 'Waiting';
    case 'consumed':
      return 'Approved and used';
    case 'denied':
      return 'Rejected';
    case 'expired':
      return 'Expired';
    case 'failed':
      // 045 (A29): approved and used, then the call failed; never re-opened.
      return 'Approved, then failed';
    case 'approved':
      // 045 FR-13a (A16 D5): steering is caller-executed, so an approved steering request waits for the agent's
      // re-call like any other ("Approved and run" described the withdrawn backend self-call).
      if (RECORD_TOOLS.has(r.tool_name)) return 'Approved (recorded)';
      return 'Approved, waiting for the agent';
    default:
      return String(r.status);
  }
}

/** The payload keys that bind an approval to its route (045 A38); shown as the route line, not as facts. */
export const ROUTE_KEYS = new Set(['_method', '_route']);

/**
 * 045 (A38; V-9): the route an approval is bound to — `POST /api/v1/labeling/panel` — from the payload's reserved
 * `_method` and `_route`, or `null` for a row filed before 045. Three routes share `start_labeling`; the operator
 * tells them apart by this line.
 */
export function routeLine(r: Pick<ApprovalRequest, 'payload'>): string | null {
  const p = (r.payload ?? {}) as Record<string, unknown>;
  if (typeof p._method !== 'string' || typeof p._route !== 'string') return null;
  return `${p._method} ${p._route}`;
}

/** True when nobody decided: the request lapsed while pending (or was drained at switch-over). */
export function expiredUndecided(r: ApprovalRequest): boolean {
  if (r.status !== 'expired') return false;
  if (r.decided_by === 'system') return true;
  if (r.decided_by === 'operator') return false;
  return !(r.reason ?? '').startsWith('expired: approved');
}

/** "Decided by" (§2.2): "operator", or "nobody (expired)" for an expiry. */
export function decidedBy(r: ApprovalRequest): string {
  if (r.status === 'pending') return '—';
  if (expiredUndecided(r)) return 'nobody (expired)';
  return 'operator';
}

/** "Result" (§2.2): what the decision produced. */
export function resultText(r: ApprovalRequest): string {
  if (r.steering_task_id) return `Steering task ${r.steering_task_id}`;
  if (r.status === 'consumed') return 'used by the agent';
  if (r.status === 'failed') return r.reason ? `failed: ${r.reason}` : 'failed after it was used';
  if (r.status === 'expired') {
    return expiredUndecided(r) ? 'expired without a decision' : 'approved, then expired unused';
  }
  // ENH-006 D81: each request of an "Approve all" batch is its own row; the shared batch id ties them together.
  if (r.batch_id) return `approved together, batch ${r.batch_id}`;
  return '—';
}
