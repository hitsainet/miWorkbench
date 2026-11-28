/**
 * The words for a long probe GPU job's reading (`GpuJobProgress`, Active Operations): view k of n, rows x of y.
 * Every figure is the server's (`live_progress.ViewRowsProgress`).
 */
import type { ViewRowsProgressDetail } from '../../types/taskQueue';
import { formatRate, formatTimeLeft } from './LiveProgress';

/** The task types this renders, and the verb each one is doing. */
export const PROBE_GPU_JOB_VERBS: Record<string, string> = {
  probe_evaluation: 'Evaluating',
  wsp_heldout_capture: 'Capturing',
};

const count = (value: number) => Math.round(value).toLocaleString('en-US');

/** "view 3/10, 4,512/33,772 rows" — the line the operator asked for. */
export function describeViewRows(detail: ViewRowsProgressDetail): string {
  const parts: string[] = [];
  if (detail.set_index != null) {
    parts.push(`view ${detail.set_index + 1}/${detail.set_count}`);
  } else if (detail.phase === 'loading_model') {
    parts.push(`${detail.set_count} ${detail.set_count === 1 ? 'view' : 'views'}, loading the model`);
  }
  const total = detail.rows_estimated ? `about ${count(detail.rows_total)}` : count(detail.rows_total);
  parts.push(`${count(detail.rows_done)}/${total} rows`);
  return parts.join(', ');
}

/** "alpha: 812/1,370 rows · 12 rows/s · about 9 min left" — the view in hand, speed, time left. */
export function describeViewRowsDetail(detail: ViewRowsProgressDetail): string {
  const parts: string[] = [];
  if (detail.set_name) {
    if (detail.phase === 'preparing') {
      parts.push(`${detail.set_name}: preparing the rows`);
    } else if (detail.set_rows_done != null && detail.set_rows_total != null) {
      parts.push(`${detail.set_name}: ${count(detail.set_rows_done)}/${count(detail.set_rows_total)} rows`);
    } else {
      parts.push(detail.set_name);
    }
  }
  const rate = formatRate(detail.rate_per_second, 'rows');
  if (rate) parts.push(rate);
  const left = formatTimeLeft(detail.eta_seconds);
  if (left && detail.rows_done < detail.rows_total) {
    parts.push(detail.eta_provisional ? `${left} (provisional)` : left);
  }
  return parts.join(' · ');
}
