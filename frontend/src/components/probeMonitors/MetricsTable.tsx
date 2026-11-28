/**
 * Per-evaluation-set metrics (032 FR-9).
 *
 * ⚠ A REFUSAL RENDERS ITS REASON. NEVER A 0, NEVER A BLANK, NEVER A DASH ALONE.
 * `scored: false` means the set could not be scored — below 20 examples of a class — and
 * the row shows that sentence with both counts. A 0 reads as "measured, and terrible"; a
 * blank reads as "not run yet"; a 0.5 reads as "measured, and chance". None of those is
 * what happened, and this estate has already paid for a metric that reported a
 * comfortable 0.5 instead of refusing.
 */
import type { ProbeEvaluation, ProbeMetrics } from '../../types/probeMonitor';
import { setLine } from '../common/setLine';

interface MetricsTableProps {
  evaluations: ProbeEvaluation[];
  /** id → display name, so the table names sets rather than showing ids. */
  datasetNames?: Record<string, string>;
}

/**
 * ENH-010 G5: the rows of this set cut at `max_length` — left out of the rung, still in the AUROC column (all rows).
 * The untruncated AUROC is the figure the rung reads; it is shown beside the count, or why there is none.
 */
function TruncationNote({ metrics }: { metrics: ProbeMetrics }) {
  const truncation = metrics.truncation;
  if (!truncation || !truncation.rows) return null;
  const fired =
    truncation.fired_at_bar === null ? 'an unknown number' : truncation.fired_at_bar.toLocaleString();
  const untruncated = metrics.untruncated;
  const figure =
    untruncated && untruncated.scored && untruncated.auroc !== null && untruncated.auroc !== undefined
      ? `AUROC without them ${untruncated.auroc.toFixed(3)}`
      : untruncated?.refused === 'too_few_untruncated'
        ? 'too few rows left to score without them'
        : null;
  return (
    <div className="mt-0.5 text-[11px] font-normal text-slate-400" data-testid="metrics-truncation">
      {truncation.rows.toLocaleString()} truncated row{truncation.rows === 1 ? '' : 's'} not counted ({fired} of
      them fire){figure ? `; ${figure}` : ''}
    </div>
  );
}

function formatAuroc(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return value.toFixed(3);
}

export function MetricsTable({ evaluations, datasetNames = {} }: MetricsTableProps) {
  if (evaluations.length === 0) {
    return (
      <p className="text-sm text-slate-400" data-testid="metrics-empty">
        No evaluation has been run for this probe yet.
      </p>
    );
  }
  return (
    <table className="w-full text-sm" data-testid="metrics-table">
      <thead>
        <tr className="border-b border-slate-700 text-left text-xs uppercase tracking-wide text-slate-400">
          <th className="py-2 pr-3">Set</th>
          <th className="py-2 pr-3">AUROC</th>
          <th className="py-2 pr-3">95% CI</th>
          <th className="py-2 pr-3">Recall @ 1% FPR</th>
          <th className="py-2 pr-3">n</th>
        </tr>
      </thead>
      <tbody>
        {evaluations.map((evaluation) => {
          const metrics = evaluation.metrics ?? { scored: false };
          // The set by name AND id (operator 2026-11-27), or its id with "(name not found)".
          const name = setLine(evaluation.dataset_id, metrics.name || datasetNames[evaluation.dataset_id]);
          if (!metrics.scored) {
            return (
              <tr
                key={evaluation.id}
                className="border-b border-slate-800"
                data-testid="metrics-refused"
              >
                <td className="py-2 pr-3 align-top font-medium text-slate-200">{name}</td>
                <td className="py-2 pr-3 align-top text-amber-300" colSpan={4}>
                  <span className="font-medium">Not scored.</span>{' '}
                  <span data-testid="refusal-reason">
                    {metrics.reason ?? 'no reason was recorded'}
                  </span>
                </td>
              </tr>
            );
          }
          const onePercent = (metrics.operating_points ?? []).find(
            (point) => Math.abs(point.target_fpr - 0.01) < 1e-9
          );
          return (
            <tr key={evaluation.id} className="border-b border-slate-800">
              <td className="py-2 pr-3 font-medium text-slate-200">
                {name}
                {metrics.out_of_distribution ? (
                  <span className="ml-2 rounded-full bg-slate-700/60 px-1.5 py-0.5 text-[10px] text-slate-300">
                    out-of-distribution
                  </span>
                ) : null}
                <TruncationNote metrics={metrics} />
              </td>
              <td className="py-2 pr-3 tabular-nums text-slate-100">
                {formatAuroc(metrics.auroc)}
              </td>
              <td className="py-2 pr-3 tabular-nums text-slate-400">
                {metrics.ci
                  ? `${metrics.ci.low.toFixed(3)}–${metrics.ci.high.toFixed(3)}`
                  : '—'}
              </td>
              <td className="py-2 pr-3 tabular-nums text-slate-100">
                {onePercent ? (
                  <>
                    {(onePercent.recall * 100).toFixed(1)}%
                    {/* THE REALISED RATE, not the target. With 100 negatives the
                        achievable rates are multiples of 1%, so the two differ often. */}
                    <span className="ml-1 text-[10px] text-slate-500">
                      at {(onePercent.realised_fpr * 100).toFixed(1)}% spent
                    </span>
                  </>
                ) : (
                  '—'
                )}
              </td>
              <td className="py-2 pr-3 tabular-nums text-slate-400">
                {metrics.n_positive ?? '—'}/{metrics.n_negative ?? '—'}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
