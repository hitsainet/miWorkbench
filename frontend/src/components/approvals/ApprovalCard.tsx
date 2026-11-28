/**
 * One waiting request as the shared card (ENH-008 §2.1), used by the banner and the Approvals page.
 *
 * Field order: action + Waiting pill; asked by; when (asked, expires) with the exact instant as
 * tooltips; facts; the extra card for kinds that need one (a probe run, operator code — D4); every
 * argument the agent sent, collapsed; the request id with a copy button; Approve and Reject. Reject
 * opens an inline reason (optional, at most 1000 characters).
 *
 * ⚠ EVERY FIELD FROM A REQUEST IS A TEXT NODE. Objects go through `JSON.stringify` and are still
 * rendered as text; nothing here uses `dangerouslySetInnerHTML` (§2.7; proven by
 * `ApprovalCard.hostile.test.tsx`, a render test, not a source grep).
 */
import { forwardRef, useEffect, useMemo, useState } from 'react';
import { Check, Copy, X } from 'lucide-react';
import { APIError } from '../../api/client';
import { SetList } from '../common/SetList';
import { setLine } from '../common/setLine';
import { getCodeRuleState } from '../../api/featureGroups';
import { getModel } from '../../api/models';
import { getGPUList } from '../../api/system';
import { useProbeViewLookup } from '../../hooks/useProbeViewLookup';
import { DEFAULT_REJECT_REASON, REJECT_REASON_MAX, useApprovalsStore } from '../../stores/approvalsStore';
import type { ApprovalRequest, ApprovalTimeForecast } from '../../types/featureGroups';
import type { ProbeDataset } from '../../types/probeMonitor';
import type { GPUInfo } from '../../types/system';
import { formatExpiresIn, formatLocal } from '../../utils/formatLocal';
import {
  actionLabel,
  approveLabel,
  askedBy,
  CODE_RULE_TOOL,
  PROBE_RUN_TOOL,
  WSP_CAPTURE_TOOL,
  WSP_IMPORT_TOOL,
  ROUTE_KEYS,
  routeLine,
  STEERING_TOOLS,
} from './approvalLabels';

/** Text, always: a string as is, anything else as JSON. */
export function asText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value === undefined) return '';
  return JSON.stringify(value);
}

/**
 * A one-line sentence for the kinds that have one, or `null` (the card then lists the facts and
 * every argument). Never cut: a request the operator cannot read in full is one they approve blind.
 */
export function renderSummary(request: ApprovalRequest): string | null {
  if (STEERING_TOOLS.has(request.tool_name)) {
    const payload = request.payload as { prompt?: unknown; selected_features?: unknown[]; strength_values?: unknown[] };
    const featureCount = payload.selected_features?.length ?? (payload.strength_values ? 1 : 0);
    return `${featureCount} feature${featureCount === 1 ? '' : 's'} · "${asText(payload.prompt ?? '')}"`;
  }
  return null;
}

const FIELD_LABELS: Record<string, string> = {
  model_id: 'Model',
  probe_id: 'Probe',
  dataset_ids: 'Probe datasets',
  model: 'Judge model',
  endpoint: 'Judge endpoint',
  prompt_id: 'Judge prompt',
  max_rows_per_set: 'Rows per set at most',
  behaviour: 'Behaviour',
  size: 'Turns',
  window: 'Window',
  target_fpr: 'Target false-positive rate',
  recut_windows: 'Re-score the windows',
  sample_rows: 'Rows sampled per set',
  prompt: 'Prompt',
  sae_id: 'Sparse autoencoder (SAE)',
};

/** Facts whose value is a list of sets: the card shows them one per line, collapsible (2026-11-27). */
const SET_LIST_FIELDS = new Set(['dataset_ids']);

/**
 * One fact: its label and its text, and for a list of sets (`SET_LIST_FIELDS`) each set's text too,
 * so the card can show them one per line. The text stays the whole value either way.
 */
export type Fact = [string, string] | [string, string, string[]];

/** The facts list (§2.1 row 4): known labels in this order, then unknown keys. */
export function factsFor(request: ApprovalRequest): Fact[] {
  const p = request.payload ?? {};
  if (request.tool_name === CODE_RULE_TOOL) {
    const doc = (p.document ?? {}) as { functions?: { name?: unknown }[] };
    return [
      ['Code hash', asText(p.code_sha256)],
      ['Functions', (doc.functions ?? []).map((f) => asText(f.name)).join(', ') || '—'],
    ];
  }
  if (request.tool_name === PROBE_RUN_TOOL) return [];
  if (request.tool_name === WSP_IMPORT_TOOL) return wspImportFacts(p);
  if (STEERING_TOOLS.has(request.tool_name)) {
    const summary = renderSummary(request);
    return summary === null ? [] : [['Steering', summary]];
  }
  const known = Object.keys(FIELD_LABELS).filter((k) => k in p);
  const unknown = Object.keys(p).filter((k) => !(k in FIELD_LABELS) && !ROUTE_KEYS.has(k));
  return [...known, ...unknown].map((k): Fact => {
    const value = p[k];
    const label = FIELD_LABELS[k] ?? k;
    if (SET_LIST_FIELDS.has(k) && Array.isArray(value)) return [label, asText(value), value.map(asText)];
    return [label, asText(value)];
  });
}

/**
 * ENH-010 044: an import's facts, read from the miForge run document it carries — never the documents and the
 * base64 patch themselves as facts (they are in "All arguments", in full).
 */
export function wspImportFacts(p: Record<string, unknown>): [string, string][] {
  const run = (p.run_document ?? {}) as {
    run_id?: unknown;
    candidate_id?: unknown;
    verdict?: { value?: unknown } | null;
    request?: { bundle?: { bundle_id?: unknown } | null } | null;
  };
  const ack = (p.acknowledgement ?? null) as { verdict?: unknown; reason?: unknown } | null;
  const patch = typeof p.edit_npz_b64 === 'string' ? p.edit_npz_b64 : '';
  const facts: [string, string][] = [
    ['miForge run', asText(run.run_id ?? '—')],
    ['Candidate', asText(run.candidate_id ?? '—')],
    ['Verdict', asText(run.verdict?.value ?? '—')],
    ['Bundle', asText(run.request?.bundle?.bundle_id ?? '—')],
    ['Patch size', `${Math.floor((patch.length * 3) / 4).toLocaleString()} bytes (edit.npz, base64 in the arguments)`],
    ['Result', 'a NEW probe that earns its rung at import; the parent probe is not changed'],
  ];
  if (ack) facts.push(['Acknowledged', `${asText(ack.verdict)}: ${asText(ack.reason)}`]);
  return facts;
}

/**
 * ENH-010 044: the GPU line and the sets of a capture request, as its tool read them when it filed.
 * Each set reads `<view name> (pmd_…) — role, rows` (operator 2026-11-27: name AND id on every line).
 */
export function WspCaptureForecast({ request, context }: { request: ApprovalRequest; context: ProbeRunContext }) {
  const gpu = request.forecast?.gpu ?? null;
  const sets = request.forecast?.sets ?? [];
  const lineClass = 'text-xs text-slate-700 dark:text-slate-300';
  let gpuLine = 'GPU: nothing was read when this request was filed.';
  if (gpu) {
    if (gpu.checked === false) gpuLine = `GPU: not checked — ${asText(gpu.summary ?? '')}`;
    else if (gpu.fits === false) gpuLine = `GPU: NO card has the memory now — ${asText(gpu.summary ?? '')}`;
    else gpuLine = `GPU: ${asText(gpu.summary ?? '')}`;
  }
  const running = sets.filter((s) => s.would_run !== false);
  const rows = running.reduce((n, s) => n + (typeof s.rows === 'number' ? s.rows : 0), 0);
  return (
    <div className="space-y-0.5" data-testid="approval-wsp-capture">
      <p className={lineClass} data-testid="approval-wsp-gpu">
        {gpuLine}
      </p>
      {sets.length === 0 ? (
        <p className={lineClass} data-testid="approval-wsp-sets">
          Sets: none were listed when this request was filed.
        </p>
      ) : (
        <SetList
          label="Sets to capture"
          detail={`${rows.toLocaleString()} rows`}
          items={running.map((s, i) => ({
            key: `${i}:${asText(s.dataset_id)}`,
            content: `${viewLine(asText(s.dataset_id), context)} — ${asText(s.role)}, ${asText(s.rows)} rows`,
          }))}
          className={lineClass}
          testId="approval-wsp-sets"
        />
      )}
      <p className="text-[11px] text-slate-500">
        Free memory is read again when the agent re-calls; a card that filled up since is refused then, not run.
      </p>
    </div>
  );
}

/** What the card looked up to name the things a probe run, and every probe view a request names, uses. */
export interface ProbeRunContext {
  modelNames: Record<string, string>;
  views: Record<string, ProbeDataset>;
  /** `null` = the card list could not be read. */
  gpus: GPUInfo[] | null;
  /**
   * True while the view lookup for the ids now on screen is still in flight: a line then reads
   * "(looking up name…)", never a false "(name not found)". Absent = the lookup is done (or none).
   */
  viewsPending?: boolean;
}

export const EMPTY_CONTEXT: ProbeRunContext = { modelNames: {}, views: {}, gpus: null };

/**
 * Every probe view a request names: a probe run's training, evaluation and calibration sets, any
 * `dataset_ids` (evaluate, capture, judge), and a capture's forecast sets.
 */
function referencedViewIds(request: ApprovalRequest): string[] {
  const p = (request.payload ?? {}) as Record<string, unknown>;
  const ids: unknown[] = [];
  if (request.tool_name === PROBE_RUN_TOOL) {
    ids.push(p.train_dataset_id, p.calibration_dataset_id);
    if (Array.isArray(p.eval_dataset_ids)) ids.push(...p.eval_dataset_ids);
  }
  if (Array.isArray(p.dataset_ids)) ids.push(...p.dataset_ids);
  for (const set of request.forecast?.sets ?? []) ids.push(set?.dataset_id);
  return ids.filter((id): id is string => typeof id === 'string' && id !== '');
}

/**
 * Resolve names for the waiting requests: models, cards and views for probe runs, and the view
 * names for every request that names probe views. A queue that names none of them makes no
 * extra requests. Every lookup may fail; a line then names the raw id and says so.
 *
 * ⚠ ONE VIEW LOOKUP FOR THE WHOLE QUEUE, NOT ONE PER LINE: the view list is read once
 * (`listDatasets`) and every card and every line reads its names from it. It is read again only
 * when a request names a view the last read did not have.
 */
export function useProbeRunContext(pending: ApprovalRequest[]): ProbeRunContext {
  const [runContext, setRunContext] = useState<Pick<ProbeRunContext, 'modelNames' | 'gpus'>>({
    modelNames: {},
    gpus: null,
  });
  const modelKey = pending
    .filter((r) => r.tool_name === PROBE_RUN_TOOL)
    .map((r) => String((r.payload as { model_id?: unknown }).model_id ?? ''))
    .sort()
    .join('|');
  const hasProbeRun = pending.some((r) => r.tool_name === PROBE_RUN_TOOL);
  const { views, pending: viewsPending } = useProbeViewLookup(pending.flatMap(referencedViewIds));

  useEffect(() => {
    if (!hasProbeRun) return;
    let cancelled = false;
    void (async () => {
      const [gpus, models] = await Promise.all([
        getGPUList().then((r) => r.gpus).catch(() => null),
        Promise.all(
          modelKey
            .split('|')
            .filter(Boolean)
            .map((id) =>
              getModel(id)
                .then((m) => [id, m.name] as const)
                .catch(() => null),
            ),
        ),
      ]);
      if (cancelled) return;
      setRunContext({ gpus, modelNames: Object.fromEntries(models.filter((m) => m !== null)) });
    })();
    return () => {
      cancelled = true;
    };
  }, [hasProbeRun, modelKey]);

  return useMemo(
    () => ({ ...runContext, views, viewsPending }),
    [runContext, views, viewsPending],
  );
}

function Time({ iso, prefix }: { iso: string | null | undefined; prefix: string }) {
  const t = formatLocal(iso);
  return (
    <span title={t.title || undefined}>
      {prefix} {t.text}
    </span>
  );
}

function decisionError(e: unknown): string {
  if (e instanceof APIError && e.status === 409) {
    if (/expired/i.test(e.detail)) return 'This request expired. Ask the agent to try again.';
    return `${e.detail} The list has been refreshed.`;
  }
  if (e instanceof APIError) return e.detail;
  if (e instanceof Error) return e.message;
  return String(e);
}

interface ApprovalCardProps {
  request: ApprovalRequest;
  context: ProbeRunContext;
  /** Called after a successful decision, so the list can move focus to the next card. */
  onDecided?: () => void;
}

export const ApprovalCard = forwardRef<HTMLDivElement, ApprovalCardProps>(function ApprovalCard(
  { request, context, onDecided },
  ref,
) {
  const approve = useApprovalsStore((s) => s.approve);
  const reject = useApprovalsStore((s) => s.reject);
  const [busy, setBusy] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const label = actionLabel(request.tool_name);
  const facts = factsFor(request);
  const route = routeLine(request);
  const expires = formatLocal(request.expires_at);

  const decide = async (kind: 'approve' | 'reject') => {
    setBusy(true);
    setError(null);
    try {
      if (kind === 'approve') await approve(request.id);
      else await reject(request.id, reason);
      onDecided?.();
    } catch (e) {
      setError(decisionError(e));
      // Focus stays on this card after a failed decision (§2.7).
      if (ref && typeof ref !== 'function') ref.current?.focus();
    } finally {
      setBusy(false);
    }
  };

  const copyId = async () => {
    try {
      await navigator.clipboard?.writeText(request.id);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div
      ref={ref}
      tabIndex={-1}
      className="bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded-control px-3 py-2 space-y-1 focus:outline-hidden focus:ring-2 focus:ring-amber-500"
      data-testid="approval-request"
      data-approval-id={request.id}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold text-slate-900 dark:text-slate-100" data-testid="approval-action">
          {label}
        </span>
        <span className="text-[11px] px-1.5 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/60 text-amber-800 dark:text-amber-200 border border-amber-300 dark:border-amber-700">
          Waiting
        </span>
      </div>
      <p className="text-xs text-slate-600 dark:text-slate-400" data-testid="approval-asked-by">
        {askedBy(request.requested_by)}
      </p>
      <p className="text-xs text-slate-600 dark:text-slate-400" data-testid="approval-when">
        <Time iso={request.created_at} prefix="asked" />
        {' · '}
        <span title={expires.title || undefined}>{formatExpiresIn(request.expires_at)}</span>
      </p>
      {route !== null && (
        <p className="text-xs font-mono text-slate-700 dark:text-slate-300 break-all" data-testid="approval-route">
          {route}
        </p>
      )}
      {facts.length > 0 && (
        <dl className="text-xs grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5" data-testid="approval-facts">
          {facts.map(([k, v, sets]) => (
            <div key={k} className="contents">
              <dt className="text-slate-500">{k}</dt>
              <dd className="text-slate-800 dark:text-slate-200 break-words whitespace-pre-wrap">
                {sets ? (
                  <SetList
                    label={k}
                    showLabel={false}
                    items={sets.map((s, i) => ({ key: `${i}:${s}`, content: viewLine(s, context) }))}
                    className="text-slate-800 dark:text-slate-200"
                    testId="approval-fact-sets"
                  />
                ) : (
                  v
                )}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {request.tool_name === PROBE_RUN_TOOL && <ProbeRunSummary payload={request.payload} context={context} />}
      {(request.tool_name === PROBE_RUN_TOOL || request.forecast?.time) && (
        <TimeForecast forecast={request.forecast?.time ?? null} />
      )}
      {request.tool_name === CODE_RULE_TOOL && <CodeRuleApprovalCard payload={request.payload} />}
      {request.tool_name === WSP_CAPTURE_TOOL && <WspCaptureForecast request={request} context={context} />}
      <AllArguments payload={request.payload} />
      <p className="text-[11px] text-slate-500 flex items-center gap-1">
        <span className="font-mono" data-testid="approval-id">{request.id}</span>
        <button
          type="button"
          onClick={() => void copyId()}
          className="p-0.5 rounded-control hover:bg-slate-200 dark:hover:bg-slate-700"
          aria-label={`Copy request id ${request.id}`}
          title="Copy request id"
        >
          <Copy className="w-3 h-3" />
        </button>
        {copied && <span>copied</span>}
      </p>
      {error && (
        <p className="text-xs text-red-600 dark:text-red-400" role="alert" data-testid="approval-error">
          {error}
        </p>
      )}
      {rejecting ? (
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="text"
            value={reason}
            maxLength={REJECT_REASON_MAX}
            onChange={(e) => setReason(e.target.value)}
            placeholder="Reason (optional)"
            aria-label="Reason (optional)"
            className="flex-1 min-w-48 text-xs px-2 py-1 rounded-control border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800"
          />
          <button
            type="button"
            disabled={busy}
            onClick={() => void decide('reject')}
            className="px-2.5 py-1 text-xs bg-red-700 hover:bg-red-600 text-white rounded-control disabled:opacity-50"
            title={`Left blank, the reason sent is "${DEFAULT_REJECT_REASON}"`}
          >
            Confirm reject
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              setRejecting(false);
              setReason('');
            }}
            className="px-2.5 py-1 text-xs bg-slate-200 dark:bg-slate-700 text-slate-800 dark:text-slate-200 rounded-control disabled:opacity-50"
          >
            Cancel
          </button>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            disabled={busy}
            onClick={() => void decide('approve')}
            className="flex items-center gap-1 px-2.5 py-1 text-xs bg-emerald-700 hover:bg-emerald-600 text-white rounded-control disabled:opacity-50"
          >
            <Check className="w-3 h-3" /> {approveLabel(request.tool_name)}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => setRejecting(true)}
            className="flex items-center gap-1 px-2.5 py-1 text-xs bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-800 dark:text-slate-200 rounded-control disabled:opacity-50"
          >
            <X className="w-3 h-3" /> Reject
          </button>
        </div>
      )}
    </div>
  );
});

/** Every argument, in full, wrapped, collapsed (§2.1 row 6) — never cut at a length. */
export function AllArguments({ payload }: { payload: Record<string, unknown> }) {
  const entries = Object.entries(payload ?? {});
  return (
    <details className="text-xs text-slate-600 dark:text-slate-400" data-testid="approval-arguments">
      <summary className="cursor-pointer">All arguments the agent sent ({entries.length})</summary>
      {entries.length === 0 ? (
        <p className="mt-1">No arguments.</p>
      ) : (
        <dl className="mt-1 space-y-0.5">
          {entries.map(([key, value]) => (
            <div key={key} className="flex gap-2">
              <dt className="font-mono text-slate-500 shrink-0">{key}</dt>
              <dd className="font-mono break-all whitespace-pre-wrap">{asText(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </details>
  );
}

function countsText(view: ProbeDataset): string {
  const c = view.counts;
  return `${c.positive.toLocaleString()} positive, ${c.negative.toLocaleString()} negative rows`;
}

/** A probe view by name AND id (`setLine`), or its id with "(name not found)". */
function viewLine(id: string, context: ProbeRunContext): string {
  return setLine(id, context.views[id]?.name, { pending: context.viewsPending });
}

/** A probe run's view: name AND id, then its row counts when the view was found. */
function viewText(id: string, context: ProbeRunContext): string {
  const view = context.views[id];
  const line = viewLine(id, context);
  return view ? `${line} — ${countsText(view)}` : line;
}

/** The card a probe run asks for, by name AND identifier (the identifier is what pins it). */
function gpuText(gpu: unknown, gpus: GPUInfo[] | null): string {
  const asked = gpu === undefined || gpu === null ? 'auto' : String(gpu);
  if (asked === 'auto') {
    return 'Automatic — whichever card has the most free memory when the job starts';
  }
  const card = gpus?.find((g) => g.uuid === asked || String(g.gpu_id) === asked);
  if (card) return `${card.name} (${card.uuid})`;
  if (gpus === null) return `${asked} — the card list could not be read to name it`;
  return `${asked} — not in this machine's current card list`;
}

const CONFIG_LABELS: Record<string, string> = {
  layers: 'Layers',
  stride: 'Every n-th layer',
  scope: 'Tokens read',
  rules: 'Pooling rules',
  max_length: 'Maximum tokens per input',
  target_fpr: 'Target false-positive rate',
  sae_variant: 'Also train sparse autoencoder (SAE) probes',
  sae_k: 'Sparse autoencoder (SAE) feature counts',
  top_n_layers: 'Best layers kept',
  batch_size: 'Batch size',
  val_fraction: 'Held-out validation fraction',
  seed: 'Random seed',
  calibration_max_rows: 'Calibration rows used at most',
  eval_max_rows: 'Rows scored per evaluation set at most',
  rolling_windows: 'Rolling windows',
};

/**
 * A probe training run, in plain words: WHAT IT DETECTS first (the behaviour, which is what tells
 * several waiting runs on one model apart), then the model, the graphics card BY NAME AND
 * IDENTIFIER, every dataset view by name AND id with its row counts, and the configuration.
 */
export function ProbeRunSummary({
  payload,
  context,
}: {
  payload: Record<string, unknown>;
  context: ProbeRunContext;
}) {
  const p = payload as {
    model_id?: unknown;
    train_dataset_id?: unknown;
    eval_dataset_ids?: unknown;
    calibration_dataset_id?: unknown;
    config?: unknown;
    gpu?: unknown;
    behaviour?: unknown;
    behaviour_label?: unknown;
  };
  const behaviour = typeof p.behaviour === 'string' && p.behaviour !== '' ? p.behaviour : null;
  const behaviourLabel =
    typeof p.behaviour_label === 'string' && p.behaviour_label !== '' ? p.behaviour_label : null;
  const modelId = String(p.model_id ?? '');
  const modelName = context.modelNames[modelId];
  const evalIds = Array.isArray(p.eval_dataset_ids) ? p.eval_dataset_ids.map(String) : [];
  const config =
    p.config && typeof p.config === 'object' ? (p.config as Record<string, unknown>) : null;
  const lineClass = 'text-xs text-slate-600 dark:text-slate-400';
  return (
    <div className="mt-1 space-y-0.5" data-testid="probe-run-summary">
      <p className="text-sm font-semibold text-slate-900 dark:text-slate-100" data-testid="probe-run-behaviour">
        {behaviourLabel ?? behaviour
          ? `Detects: ${behaviourLabel ?? behaviour}${behaviourLabel && behaviour ? ` (behaviour key “${behaviour}”)` : ''}`
          : 'Detects: no behaviour given — tell these probes apart by their training set below'}
      </p>
      <p className="text-xs text-slate-800 dark:text-slate-200">
        Train probes on {modelName ? `“${modelName}”` : `a model the banner could not look up (${modelId})`} —
        a graphics processing unit (GPU) job that runs for minutes to hours.
      </p>
      <p className={lineClass}>Graphics card: {gpuText(p.gpu, context.gpus)}</p>
      <p className={lineClass}>
        Training set: {p.train_dataset_id ? viewText(String(p.train_dataset_id), context) : 'none given'}
      </p>
      <SetList
        label="Evaluation sets"
        items={evalIds.map((id, i) => ({ key: `${i}:${id}`, content: viewText(id, context) }))}
        className={lineClass}
        testId="probe-run-eval-sets"
      />
      <p className={lineClass}>
        Calibration set:{' '}
        {p.calibration_dataset_id ? viewText(String(p.calibration_dataset_id), context) : 'none'}
      </p>
      <p className={lineClass}>
        Settings:{' '}
        {config && Object.keys(config).length > 0
          ? Object.entries(config)
              .map(([k, v]) => `${CONFIG_LABELS[k] ?? k}: ${asText(v)}`)
              .join(' · ')
          : 'the defaults'}
      </p>
    </div>
  );
}

/** "15,895" — counts as the operator reads them. */
function count(n: number): string {
  return n.toLocaleString('en-US');
}

/** What the evaluation rows rest on: the cap, or every set whole. */
export function evalRowsText(t: ApprovalTimeForecast): string {
  const sets = Object.keys(t.eval_rows_by_set ?? {}).length;
  if (sets === 0) return 'none (no evaluation sets named)';
  if (t.eval_rows == null) return `not known (the size of one of ${sets} sets could not be read)`;
  const basis =
    t.eval_max_rows != null
      ? `each set capped at ${count(t.eval_max_rows)} (eval_max_rows)`
      : 'every set scored whole (no eval_max_rows)';
  return `${count(t.eval_rows)} rows over ${sets} ${sets === 1 ? 'set' : 'sets'}, ${basis}`;
}

/**
 * ENH-006 G15: how long the run will take, as the agent's forecast said when it filed this request, and the
 * evaluation rows that forecast assumes. A request filed with none says so: silence would read as "short".
 */
export function TimeForecast({ forecast }: { forecast: ApprovalTimeForecast | null }) {
  const lineClass = 'text-xs text-slate-800 dark:text-slate-200';
  if (!forecast) {
    return (
      <p className={lineClass} data-testid="approval-time-forecast">
        Time forecast: none was filed with this request.
      </p>
    );
  }
  return (
    <div className="space-y-0.5" data-testid="approval-time-forecast-block">
      <p className={lineClass} data-testid="approval-time-forecast">
        Time forecast: {asText(forecast.summary)}
      </p>
      <p className="text-xs text-slate-600 dark:text-slate-400" data-testid="approval-time-forecast-eval-rows">
        Evaluation rows it assumes: {evalRowsText(forecast)}
      </p>
    </div>
  );
}

interface CodeRuleCaseResult {
  id: string;
  expect: Record<string, unknown>;
  got: Record<string, unknown> | null;
  same_twice: boolean;
}

type LiveState = { state: string; revoked_row_id?: number | null } | 'loading' | 'unreadable';

/**
 * Operator code (D4; Feature 040 S4): what the operator reads BEFORE letting it run — the hash, the
 * functions, the modules it imports (the runner's own inspection), its cases as run twice in the
 * runner, the FULL source, never cut, and the hash's CURRENT approval state, read when the card is
 * shown. The inspection and case results are a snapshot taken when the agent asked, and are
 * labelled so. Approving records a durable per-hash approval; revoking is on the behaviour's
 * rule-set screen. This card never says approving undoes a revocation: the revocation stays in the
 * hash's history whatever is decided here.
 */
export function CodeRuleApprovalCard({ payload }: { payload: Record<string, unknown> }) {
  const sha = asText(payload.code_sha256 ?? '');
  const document = (payload.document ?? {}) as {
    source?: unknown;
    functions?: { name?: unknown }[];
    derived_from?: { statement?: unknown };
  };
  const facts = (payload.facts ?? {}) as {
    inspect?: { functions?: unknown[]; imports?: unknown[]; reason?: unknown };
    case_results?: CodeRuleCaseResult[];
    midataworks_state?: unknown;
    requested_by?: unknown;
  };
  const imports = facts.inspect?.imports;
  const [live, setLive] = useState<LiveState>('loading');

  useEffect(() => {
    let cancelled = false;
    if (!/^[0-9a-f]{64}$/.test(sha)) {
      setLive('unreadable');
      return;
    }
    getCodeRuleState(sha)
      .then((s) => !cancelled && setLive({ state: s.state, revoked_row_id: s.revoked_row_id }))
      .catch(() => !cancelled && setLive('unreadable'));
    return () => {
      cancelled = true;
    };
  }, [sha]);

  let stateText: string;
  if (live === 'loading') stateText = 'reading…';
  else if (live === 'unreadable') stateText = 'could not be read';
  else if (live.state === 'pending' && live.revoked_row_id)
    stateText = 'waiting for this decision; this code was revoked earlier, and that revocation stays in its history';
  else if (live.state === 'revoked') stateText = 'revoked';
  else stateText = live.state;

  return (
    <div className="mt-1 space-y-1 text-xs text-slate-600 dark:text-slate-400" data-testid="code-rule-approval">
      <p>
        Run operator code <span className="font-mono break-all">{sha}</span> in miWorkbench&apos;s isolated runner
        {facts.requested_by ? ` (asked by ${asText(facts.requested_by)})` : ''}.
      </p>
      <p data-testid="code-rule-state">Current approval state of this code: {stateText}</p>
      <p>Functions: {(document.functions ?? []).map((f) => asText(f.name)).join(', ') || '—'}</p>
      <p className="text-slate-500">Snapshot taken when the agent asked:</p>
      <p>Imports: {Array.isArray(imports) ? (imports.length ? imports.map(asText).join(', ') : 'none') : 'not inspected'}</p>
      {document.derived_from?.statement ? <p>Derived from: {asText(document.derived_from.statement)}</p> : null}
      <p>miDataworks approval: {facts.midataworks_state ? asText(facts.midataworks_state) : 'not known'} (information only; this miWorkbench decides)</p>
      <ul data-testid="code-rule-cases">
        {(facts.case_results ?? []).map((c) => (
          <li key={asText(c.id)}>
            case {asText(c.id)}: expected {asText(c.expect)}, got {asText(c.got)}
            {c.same_twice ? ', same on two fresh instances' : ', DIFFERED between instances'}
          </li>
        ))}
      </ul>
      <details open>
        <summary className="cursor-pointer">Source (read it before approving)</summary>
        <pre className="whitespace-pre-wrap font-mono bg-slate-900 text-slate-100 p-2 rounded-control" data-testid="code-rule-source">
          {asText(document.source ?? '')}
        </pre>
      </details>
    </div>
  );
}
