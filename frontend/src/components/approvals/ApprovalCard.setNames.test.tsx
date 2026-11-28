/**
 * Operator request 2026-11-27: EVERY set line on an approval card shows the view's NAME AND its raw
 * `pmd_…` id — on the evaluate, capture and judge cards (the generic "Probe datasets" fact), which
 * listed bare ids, as well as on the training card.
 *
 * Rendered through the real Approvals page (`ApprovalsPanel` → `useProbeRunContext` → cards), so
 * these tests read the wiring, not a hand-built context:
 * - each line reads `<view name> (pmd_…)`; a view the lookup does not have (deleted) or a failed
 *   lookup reads `pmd_… (name not found)`; an in-flight lookup reads `pmd_… (looking up name…)`;
 * - ⚠ ONE `listDatasets` READ FOR THE WHOLE QUEUE, NOT ONE PER LINE OR PER CARD: four cards naming
 *   31 lines make exactly one request; a new card naming only known views makes none; a new card
 *   naming an unknown view makes exactly one more.
 *
 * NEGATIVE CONTROLS (2026-11-27): drop the name from `setLine` (id only), drop the id (name only),
 * and fetch per line instead of once → each goes red here.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ApprovalRequest } from '../../types/featureGroups';

vi.mock('../../api/featureGroups', () => ({
  listApprovals: vi.fn(),
  listDecidedApprovals: vi.fn(() => new Promise(() => {})),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
  getCodeRuleState: vi.fn(),
  HISTORY_PAGE: 200,
}));
vi.mock('../../api/models', () => ({ getModel: vi.fn(() => Promise.resolve({ id: 'm_1', name: 'Llama' })) }));
const listDatasets = vi.fn();
vi.mock('../../api/probeMonitors', () => ({
  probeMonitorsApi: { listDatasets: (...args: unknown[]) => listDatasets(...args) },
}));
vi.mock('../../api/system', () => ({ getGPUList: vi.fn(() => Promise.resolve({ gpus: [] })) }));

import { ApprovalsPanel } from '../panels/ApprovalsPanel';
import { useApprovalsStore } from '../../stores/approvalsStore';
import { useUIStore } from '../../stores/uiStore';

const IDS = Array.from({ length: 10 }, (_, i) => `pmd_${String(i).padStart(2, '0')}c501631eae`);
const NAMES = IDS.map((_, i) => `ssn-d128-fu7a-fresh-id-test-${i}`);
const VIEWS = IDS.map((id, i) => ({ id, name: NAMES[i], counts: { positive: 1, negative: 2 } }));
const LINE = (i: number) => `${NAMES[i]} (${IDS[i]})`;

function request(id: string, tool: string, payload: Record<string, unknown>, forecast?: unknown): ApprovalRequest {
  return {
    id, tool_name: tool, status: 'pending', reason: null, steering_task_id: null,
    created_at: '2026-11-27T13:00:00Z', resolved_at: null, requested_by: 'mcp', payload,
    forecast: (forecast ?? null) as ApprovalRequest['forecast'],
  };
}

const EVAL_A = request('apr_eval_a', 'evaluate_probe_monitor', { probe_id: 'pm_1', dataset_ids: IDS });
const EVAL_B = request('apr_eval_b', 'evaluate_probe_monitor', {
  probe_id: 'pm_2', dataset_ids: [...IDS.slice(0, 5), 'pmd_deleted00'],
});
const JUDGE = request('apr_judge', 'start_probe_judge_run', { probe_id: 'pm_1', dataset_ids: IDS.slice(0, 5) });
const CAPTURE = request(
  'apr_capture', 'capture_wsp_heldout', { probe_id: 'pm_1', dataset_ids: IDS.slice(0, 5) },
  { sets: IDS.slice(0, 5).map((id) => ({ dataset_id: id, role: 'eval', rows: 100, would_run: true })) },
);
const QUEUE = [EVAL_A, EVAL_B, JUDGE, CAPTURE];

function card(id: string) {
  const found = document.querySelector(`[data-approval-id="${id}"]`);
  if (!found) throw new Error(`no card ${id}`);
  return within(found as HTMLElement);
}

/** Expand every set list on a card and return its lines' text, list by list. */
async function linesOf(id: string, testId: string): Promise<string[]> {
  const root = card(id).getByTestId(testId);
  const toggle = within(root).getByRole('button');
  if (toggle.getAttribute('aria-expanded') === 'false') await userEvent.click(toggle);
  return within(root).getAllByRole('listitem').map((li) => li.textContent ?? '');
}

beforeEach(() => {
  listDatasets.mockReset();
  useUIStore.setState({ activePanel: 'approvals' });
  useApprovalsStore.setState({ pending: QUEUE, loaded: true, error: null });
});

describe('every set line on an approval card shows the view name AND its id', () => {
  it('evaluate, judge and capture cards: `<name> (pmd_…)` on every line, from ONE lookup for the queue', async () => {
    listDatasets.mockResolvedValue(VIEWS);
    render(<ApprovalsPanel />);
    await waitFor(() => expect(card('apr_eval_a').getByTestId('approval-fact-sets')).toHaveTextContent(NAMES[0]));

    expect(await linesOf('apr_eval_a', 'approval-fact-sets')).toEqual(IDS.map((_, i) => LINE(i)));
    expect(await linesOf('apr_eval_b', 'approval-fact-sets')).toEqual([
      ...[0, 1, 2, 3, 4].map(LINE),
      'pmd_deleted00 (name not found)',
    ]);
    expect(await linesOf('apr_judge', 'approval-fact-sets')).toEqual([0, 1, 2, 3, 4].map(LINE));
    expect(await linesOf('apr_capture', 'approval-fact-sets')).toEqual([0, 1, 2, 3, 4].map(LINE));
    expect(await linesOf('apr_capture', 'approval-wsp-sets')).toEqual(
      [0, 1, 2, 3, 4].map((i) => `${LINE(i)} — eval, 100 rows`),
    );

    // 31 lines over four cards, one request, with no arguments (the whole list, archived included).
    expect(listDatasets).toHaveBeenCalledTimes(1);
    expect(listDatasets).toHaveBeenCalledWith();
  });

  it('a FAILED lookup: every line keeps its id with "(name not found)" — none blank, none dropped', async () => {
    listDatasets.mockRejectedValue(new Error('backend down'));
    useApprovalsStore.setState({ pending: [EVAL_A], loaded: true, error: null });
    render(<ApprovalsPanel />);
    await waitFor(() =>
      expect(card('apr_eval_a').getByTestId('approval-fact-sets')).toHaveTextContent('(name not found)'),
    );
    const lines = await linesOf('apr_eval_a', 'approval-fact-sets');
    expect(lines).toEqual(IDS.map((id) => `${id} (name not found)`));
    expect(listDatasets).toHaveBeenCalledTimes(1);
  });

  it('a lookup still in flight reads "(looking up name…)", not a false "(name not found)"', async () => {
    listDatasets.mockReturnValue(new Promise(() => {}));
    useApprovalsStore.setState({ pending: [EVAL_A], loaded: true, error: null });
    render(<ApprovalsPanel />);
    const lines = await linesOf('apr_eval_a', 'approval-fact-sets');
    expect(lines).toEqual(IDS.map((id) => `${id} (looking up name…)`));
  });

  it('a new card naming only known views makes NO new request; one naming a new view makes exactly one', async () => {
    listDatasets.mockResolvedValue(VIEWS);
    useApprovalsStore.setState({ pending: [EVAL_A], loaded: true, error: null });
    render(<ApprovalsPanel />);
    await waitFor(() => expect(card('apr_eval_a').getByTestId('approval-fact-sets')).toHaveTextContent(NAMES[0]));
    expect(listDatasets).toHaveBeenCalledTimes(1);

    act(() => useApprovalsStore.setState({ pending: [EVAL_A, JUDGE] }));
    expect(await linesOf('apr_judge', 'approval-fact-sets')).toEqual([0, 1, 2, 3, 4].map(LINE));
    expect(listDatasets).toHaveBeenCalledTimes(1);

    listDatasets.mockResolvedValue([...VIEWS, { id: 'pmd_new0000000', name: 'made after the first read', counts: {} }]);
    const fresh = request('apr_fresh', 'evaluate_probe_monitor', { probe_id: 'pm_3', dataset_ids: ['pmd_new0000000'] });
    act(() => useApprovalsStore.setState({ pending: [EVAL_A, JUDGE, fresh] }));
    await waitFor(() =>
      expect(card('apr_fresh').getByRole('listitem')).toHaveTextContent(
        /^made after the first read \(pmd_new0000000\)$/,
      ),
    );
    expect(listDatasets).toHaveBeenCalledTimes(2);
  });

  it('a queue that names no probe view makes no lookup', async () => {
    useApprovalsStore.setState({
      pending: [request('apr_x', 'evaluate_probe_monitor', { probe_id: 'pm_1' })],
      loaded: true,
      error: null,
    });
    render(<ApprovalsPanel />);
    await screen.findByText('Evaluate a probe on a graphics card');
    expect(listDatasets).not.toHaveBeenCalled();
  });
});
