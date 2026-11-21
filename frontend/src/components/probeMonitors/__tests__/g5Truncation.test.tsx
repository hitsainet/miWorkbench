/** ENH-010 G5: the catch-rate line and the metrics table say how many truncated rows were left out, and how many fire. */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { CatchRateLine } from '../CatchRateLine';
import { MetricsTable } from '../MetricsTable';
import type { CatchRateCheck, ProbeEvaluation } from '../../../types/probeMonitor';

const catchRate = (over: Partial<CatchRateCheck>): CatchRateCheck => ({
  status: 'too_few_caught', passed: false, reason: 'catches 45% of positives', next_step: null, min_recall: 0.5,
  sets: [{ name: 'carriers (pmd_o)', positives: 20, caught: 9, threshold: 1, recall: 0.45, needed: 10, meets: false,
    note: null, sentence: 'On unseen tasks (carriers): catches 45% of positives at the shipped bar (9 of 20; needs ≥ 50%)',
    truncated_excluded: 10, truncated_caught: 10 }],
  verdict_changed_by_exclusion: true, truncated_rows_count: false, ...over,
});

const evaluation = (metrics: ProbeEvaluation['metrics']): ProbeEvaluation => ({
  id: 'pme_1', probe_id: 'pm_1', dataset_id: 'pmd_1', status: 'completed', n_positive: 30, n_negative: 30,
  metrics, created_at: '2026-11-16T00:00:00Z',
});

describe('G5 truncation on the report', () => {
  it('the catch-rate line names the truncated positives and the changed verdict', () => {
    render(<CatchRateLine check={catchRate({})} />);
    expect(screen.getByTestId('catch-rate-truncated')).toHaveTextContent(
      '10 truncated positives are not counted (10 of them fire)');
    expect(screen.getByTestId('catch-rate-verdict-changed')).toBeInTheDocument();
  });

  it('the metrics table shows the excluded count, the fired count and the untruncated AUROC', () => {
    render(<MetricsTable evaluations={[evaluation({
      scored: true, auroc: 0.801, n_positive: 30, n_negative: 30,
      untruncated: { scored: true, auroc: 0.912 },
      truncation: { rows: 3, positives: 0, negatives: 3, fired_at_bar: 3, threshold: 0.5, max_length: 4096,
        policy: 'excluded' },
    })]} />);
    expect(screen.getByTestId('metrics-truncation')).toHaveTextContent(
      '3 truncated rows not counted (3 of them fire); AUROC without them 0.912');
  });

  it('says when too few rows are left, and nothing when no row was truncated', () => {
    const { rerender } = render(<MetricsTable evaluations={[evaluation({
      scored: true, auroc: 0.8, untruncated: { scored: false, refused: 'too_few_untruncated' },
      truncation: { rows: 5, positives: 2, negatives: 3, fired_at_bar: null, threshold: null, max_length: 64,
        policy: 'excluded' },
    })]} />);
    expect(screen.getByTestId('metrics-truncation')).toHaveTextContent(
      '5 truncated rows not counted (an unknown number of them fire); too few rows left to score without them');
    rerender(<MetricsTable evaluations={[evaluation({ scored: true, auroc: 0.8,
      truncation: { rows: 0, positives: 0, negatives: 0, fired_at_bar: 0, threshold: 0.5, max_length: 64,
        policy: 'excluded' } })]} />);
    expect(screen.queryByTestId('metrics-truncation')).toBeNull();
  });
});
