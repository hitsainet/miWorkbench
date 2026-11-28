/** ENH-006 D82: the report states the everyday-text result in the server's plain words. */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { EverydayTextLine } from '../EverydayTextLine';
import type { EverydayTextCheck, EverydayTextSet } from '../../../types/probeMonitor';

const set = (over: Partial<EverydayTextSet>): EverydayTextSet => ({
  name: 'prose (pmd_e)', rows: 1200, false_alarms: 0, threshold: 0.5, counts: true, allowed: 24, fpr: 0,
  within_allowance: true, note: null, sentence: 'On everyday text: 0 false alarms in 1,200 (allowed up to 24)', ...over,
});
const check = (over: Partial<EverydayTextCheck>): EverydayTextCheck => ({
  status: 'passed', passed: true, reason: 'few false alarms', next_step: null, target_fpr: 0.01, factor: 2,
  min_rows: 1000, allowed_rate: 0.02, sets: [set({})], ...over,
});

describe('EverydayTextLine', () => {
  it('states a passing set in plain words', () => {
    render(<EverydayTextLine check={check({})} />);
    expect(screen.getByTestId('everyday-text-set')).toHaveTextContent(
      'On everyday text: 0 false alarms in 1,200 (allowed up to 24)');
    expect(screen.getByTestId('everyday-text')).not.toHaveTextContent('Rung 2 not reached');
  });

  it('says a probe fired on all of it, and why rung 2 is not reached', () => {
    render(<EverydayTextLine check={check({
      status: 'too_many_false_alarms', passed: false,
      reason: 'false alarms on everyday text 100% at the shipped bar (allowed ≤ 2%)',
      sets: [set({ false_alarms: 1200, fpr: 1, within_allowance: false,
        sentence: 'Fired on 100% of everyday text (1,200 rows)' })],
    })} />);
    expect(screen.getByTestId('everyday-text-set')).toHaveTextContent('Fired on 100% of everyday text');
    expect(screen.getByTestId('everyday-text')).toHaveTextContent(
      'Rung 2 not reached: false alarms on everyday text 100% at the shipped bar (allowed ≤ 2%)');
  });

  it('says when it was never checked, and nothing without a verdict', () => {
    const { rerender } = render(<EverydayTextLine check={check({
      status: 'not_checked', passed: false, sets: [],
      reason: 'not yet checked on everyday text — evaluate on an everyday-text set' })} />);
    expect(screen.getByTestId('everyday-text')).toHaveTextContent('not yet checked on everyday text');
    expect(screen.getByTestId('everyday-text')).toHaveTextContent('at least 1,000 rows');
    rerender(<EverydayTextLine check={null} />);
    expect(screen.queryByTestId('everyday-text')).toBeNull();
  });
});

describe('ENH-010 G5: truncated rows', () => {
  it('shows truncated rows not counted', () => {
    render(<EverydayTextLine check={check({
      verdict_changed_by_exclusion: true, truncated_rows_count: false,
      sets: [set({ rows: 1000, false_alarms: 2, truncated_excluded: 100, truncated_fired: 100,
        sentence: 'On everyday text: 2 false alarms in 1,000 (allowed up to 2)' })],
    })} />);
    expect(screen.getByTestId('everyday-text-truncated')).toHaveTextContent(
      '100 truncated rows are not counted (100 of them fire)');
    expect(screen.getByTestId('everyday-text-verdict-changed')).toHaveTextContent(
      'Leaving out truncated rows changed this verdict');
  });

  it('does not repeat a note the server sentence already carries', () => {
    render(<EverydayTextLine check={check({
      sets: [set({ truncated_excluded: 3, truncated_fired: 1,
        sentence: 'On everyday text: 0 false alarms in 1,000 (allowed up to 20); 3 truncated rows are not counted (1 of them fire)' })],
    })} />);
    expect(screen.getByTestId('everyday-text-set')).toHaveTextContent('3 truncated rows are not counted (1 of them fire)');
    expect(screen.queryByTestId('everyday-text-truncated')).toBeNull();
    expect(screen.queryByTestId('everyday-text-verdict-changed')).toBeNull();
  });
});
