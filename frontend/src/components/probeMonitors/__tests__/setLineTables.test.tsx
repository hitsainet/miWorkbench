/**
 * The probe report's per-set rows name each view by NAME AND ID (operator request 2026-11-27: every
 * set line shows the view's name and its raw `pmd_…` id): the metrics table and the judge run's
 * per-set table. A row whose view cannot be named keeps its id and says "(name not found)".
 *
 * NEGATIVE CONTROLS: drop the name or the id from `setLine` → red here.
 */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { JudgeRunResults } from '../JudgeRunResults';
import { MetricsTable } from '../MetricsTable';
import type { ProbeEvaluation, ProbeJudgeRun } from '../../../types/probeMonitor';

const evaluation = (id: string, dataset_id: string, name?: string): ProbeEvaluation => ({
  id, probe_id: 'pm_1', dataset_id, status: 'completed', n_positive: 30, n_negative: 30,
  metrics: { scored: false, reason: 'too few', name } as never, created_at: '2026-11-27T00:00:00Z',
});

describe('per-set report rows name the view and its id', () => {
  it('the metrics table: the stored name, else the page name, with the id; else the id and "(name not found)"', () => {
    render(
      <MetricsTable
        evaluations={[
          evaluation('pme_1', 'pmd_12c501631eae', 'ssn-d128-fu7a-fresh-id-test'),
          evaluation('pme_2', 'pmd_b', undefined),
          evaluation('pme_3', 'pmd_gone', undefined),
        ]}
        datasetNames={{ pmd_b: 'from the page' }}
      />,
    );
    const rows = screen.getAllByTestId('metrics-refused');
    expect(rows[0].querySelector('td')?.textContent).toBe('ssn-d128-fu7a-fresh-id-test (pmd_12c501631eae)');
    expect(rows[1].querySelector('td')?.textContent).toBe('from the page (pmd_b)');
    expect(rows[2].querySelector('td')?.textContent).toBe('pmd_gone (name not found)');
  });

  it("the judge run's per-set table", () => {
    const run = {
      id: 'pjr_1', probe_id: 'pm_1', status: 'completed', model: 'judge', dataset_ids: ['pmd_us', 'pmd_x', 'pmd_gone'],
      metrics: {
        per_set: {
          pmd_us: { set_name: 'US test', scored: false, reason: 'too few' },
          pmd_x: { scored: false, reason: 'too few' },
          pmd_gone: { scored: false, reason: 'too few' },
        },
      },
    } as unknown as ProbeJudgeRun;
    render(<JudgeRunResults run={run} datasetNames={{ pmd_x: 'page name' }} />);
    const cells = screen.getAllByTestId('judge-set-row').map((tr) => tr.querySelector('td')?.textContent);
    expect(cells).toEqual(['US test (pmd_us)', 'page name (pmd_x)', 'pmd_gone (name not found)']);
  });
});
