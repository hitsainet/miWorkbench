/**
 * Failed Operations offers Retry only where the retry route accepts it, and Delete wherever it did before.
 *
 * 2026-11-27: a failed `probe_evaluation` or `wsp_heldout_capture` row (and every J-Lens and probe
 * definition build row) offered Retry, and `POST /task-queue/{id}/retry` refused it with a 400. The
 * backend now sends `retryable` — the route's own decision (`task_queue.retry_refusal`) — and the row
 * shows Retry only when it is true. `can_retry` keeps meaning "this API owns the row", so Delete stays.
 *
 * MUTATION CONTROLS (0xcc/reviews/eval_card_and_retry_2026-11-27.md):
 *   F1  `retryable` gate dropped (Retry shown for every owned row)  -> "no Retry" tests red
 *   F2  Delete moved under the retryable gate                       -> "Delete stays" tests red
 *   F3  the hint span dropped                                       -> hint test red
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, render, within } from '@testing-library/react';
import { FailedOperationsSection } from './FailedOperationsSection';
import { useTaskQueueStore } from '../../stores/taskQueueStore';

vi.mock('../../stores/taskQueueStore');

function row(over: Record<string, unknown>) {
  return {
    id: `tq_${over.task_type}`,
    task_id: 'celery-1',
    entity_id: 'pm_80504153780f',
    entity_type: 'probe_monitor',
    status: 'failed',
    progress: 12,
    error_message: 'RuntimeError: CUDA out of memory',
    retry_params: null,
    retry_count: 0,
    can_retry: true,
    entity_info: { name: 'Probe pm_80504153780f' },
    completed_at: '2026-11-27T12:00:00Z',
    created_at: '2026-11-27T11:00:00Z',
    updated_at: null,
    started_at: null,
    ...over,
  };
}

function mountWith(tasks: unknown[]) {
  vi.mocked(useTaskQueueStore).mockReturnValue({
    failedTasks: tasks,
    failedLoading: false,
    failedError: null,
    fetchFailedTasks: vi.fn(),
    deleteTask: vi.fn(),
    dismissFailedTask: vi.fn(),
    dismissAllFailedTasks: vi.fn(),
  } as never);
  return render(<FailedOperationsSection />);
}

function card(name: string): HTMLElement {
  // The row's own box: the element holding its name, up to the bordered card.
  const el = screen.getByText(name).closest('.rounded-card') as HTMLElement | null;
  if (!el) throw new Error(`no card for ${name}`);
  return el;
}

describe('FailedOperationsSection: Retry only when retryable', () => {
  beforeEach(() => vi.clearAllMocks());

  it('shows Retry and Delete for a row the route accepts', () => {
    mountWith([row({ task_type: 'download', entity_type: 'model', retryable: true,
                     entity_info: { name: 'Model download' } })]);
    const box = card('Model download');
    expect(within(box).getByText('Retry')).toBeInTheDocument();
    expect(within(box).getByTitle('Remove from list')).toBeInTheDocument();
    expect(within(box).queryByTestId('rerun-hint')).not.toBeInTheDocument();
  });

  it.each([
    ['probe_evaluation', 'Re-run from the Probes page'],
    ['wsp_heldout_capture', 'Re-run from the Probes page'],
    ['probe_definition_build', 'Re-run from the Probes page'],
    ['jlens_fit', 'Re-run from the J-Lens page'],
    ['feature_grouping', 'Re-run from its own page'],
  ])('a %s row the route refuses: no Retry, Delete stays, and it says where to re-run', (taskType, hint) => {
    mountWith([row({ task_type: taskType, retryable: false })]);
    const box = card('Probe pm_80504153780f');
    expect(within(box).queryByText('Retry')).not.toBeInTheDocument();
    expect(within(box).getByTitle('Remove from list')).toBeInTheDocument();
    expect(within(box).getByTestId('rerun-hint')).toHaveTextContent(hint);
  });

  it('names a J-Lens row rather than showing its raw type', () => {
    mountWith([row({ task_type: 'jlens_band_report', retryable: false })]);
    expect(screen.getByText('J-Lens band report')).toBeInTheDocument();
  });

  it('an older backend that sends no flag offers no Retry (never inferred in the UI)', () => {
    mountWith([row({ task_type: 'download', entity_type: 'model', entity_info: { name: 'Old row' } })]);
    const box = card('Old row');
    expect(within(box).queryByText('Retry')).not.toBeInTheDocument();
    expect(within(box).getByTitle('Remove from list')).toBeInTheDocument();
  });

  it('mixed list: each row follows its own flag', () => {
    mountWith([
      row({ id: 'a', task_type: 'download', entity_type: 'dataset', retryable: true, entity_info: { name: 'A' } }),
      row({ id: 'b', task_type: 'probe_evaluation', retryable: false, entity_info: { name: 'B' } }),
    ]);
    expect(within(card('A')).getByText('Retry')).toBeInTheDocument();
    expect(within(card('B')).queryByText('Retry')).not.toBeInTheDocument();
    expect(screen.getAllByTitle('Remove from list')).toHaveLength(2);
  });
});
