/**
 * The approval card's set lists use the ONE shared `SetList` (operator request 2026-11-27, after an
 * evaluation of 10 sets arrived as one joined line): a training run's "Evaluation sets", the generic
 * "Probe datasets" fact (`evaluate_probe_monitor`, `capture_wsp_heldout`, `start_probe_judge_run`),
 * and a capture's "Sets to capture". Each shows 10 sets as one summary line, expands to 10 lines, and
 * leaves Approve and Reject where they were.
 *
 * EVERY LINE NAMES THE VIEW AND ITS RAW `pmd_…` ID (operator request 2026-11-27, second pass):
 * `<view name> (pmd_…)`, then the training card's row counts; a view that cannot be named reads
 * `pmd_… (name not found)`; while the lookup is in flight, `pmd_… (looking up name…)`. The names come
 * from ONE `listDatasets` read for the whole queue, not one per line (`ApprovalCard.setNames.test.tsx`).
 *
 * NEGATIVE CONTROLS: put a site back to its joined line → its test here goes red; drop the name or
 * the id from `setLine` → the line tests go red.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ApprovalRequest } from '../../types/featureGroups';
import type { ProbeDataset } from '../../types/probeMonitor';

vi.mock('../../api/featureGroups', () => ({
  listApprovals: vi.fn(),
  listDecidedApprovals: vi.fn(() => new Promise(() => {})),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
  getCodeRuleState: vi.fn(),
  HISTORY_PAGE: 200,
}));
vi.mock('../../api/models', () => ({ getModel: vi.fn() }));
vi.mock('../../api/probeMonitors', () => ({ probeMonitorsApi: { listDatasets: vi.fn(() => Promise.resolve([])) } }));
vi.mock('../../api/system', () => ({ getGPUList: vi.fn(() => Promise.resolve({ gpus: [] })) }));

import { ApprovalCard, EMPTY_CONTEXT, factsFor, type ProbeRunContext } from './ApprovalCard';

const LINE = (i: number) => `held-out ${i} (pmd_set${i})`;

const IDS = Array.from({ length: 10 }, (_, i) => `pmd_set${i}`);
const VIEWS = Object.fromEntries(
  IDS.map((id, i) => [id, { id, name: `held-out ${i}`, counts: { positive: 1000 + i, negative: 2000 + i } } as unknown as ProbeDataset]),
);
const CONTEXT: ProbeRunContext = { modelNames: { m_1: 'Llama' }, views: VIEWS, gpus: [] };

function request(over: Partial<ApprovalRequest>): ApprovalRequest {
  return {
    id: 'apr_1', tool_name: 'x', status: 'pending', reason: null, steering_task_id: null,
    created_at: '2026-11-27T13:00:00Z', resolved_at: null, requested_by: 'mcp', payload: {}, ...over,
  };
}

async function expectSharedList(root: HTMLElement, name: RegExp, lines: number) {
  expect(root).toHaveAttribute('data-set-list', 'collapsed');
  const toggle = within(root).getByRole('button', { expanded: false });
  expect(toggle).toHaveAccessibleName(name);
  expect(within(root).queryAllByRole('listitem')).toHaveLength(0);
  await userEvent.click(toggle);
  const items = within(root).getAllByRole('listitem');
  expect(items).toHaveLength(lines);
  items.forEach((li) => expect(li).toHaveClass('block', 'text-left'));
  return items;
}

function expectDecisionButtons() {
  expect(screen.getByRole('button', { name: /^Reject$/ })).toBeEnabled();
  expect(screen.getAllByRole('button').some((b) => /Approve/.test(b.textContent ?? ''))).toBe(true);
}

describe('the approval card lists sets one per line, collapsible', () => {
  it("a training run's Evaluation sets: 10 collapse to one line and expand to name (id) — counts lines", async () => {
    render(
      <ApprovalCard
        request={request({
          tool_name: 'start_probe_training_run',
          payload: { model_id: 'm_1', train_dataset_id: IDS[0], eval_dataset_ids: IDS, gpu: 'auto' },
        })}
        context={CONTEXT}
      />,
    );
    const root = screen.getByTestId('probe-run-eval-sets');
    expect(within(root).getByRole('button')).toHaveTextContent(/^Evaluation sets: 10 sets\s*▸$/);
    expectDecisionButtons();
    const items = await expectSharedList(root, /^Evaluation sets: 10 sets$/, 10);
    expect(items[0]).toHaveTextContent(/^held-out 0 \(pmd_set0\) — 1,000 positive, 2,000 negative rows$/);
    expect(items[9]).toHaveTextContent(/^held-out 9 \(pmd_set9\) — 1,009 positive, 2,009 negative rows$/);
    // The training set's line too.
    expect(screen.getByTestId('probe-run-summary')).toHaveTextContent(
      'Training set: held-out 0 (pmd_set0) — 1,000 positive, 2,000 negative rows',
    );
    // The old joined line is gone.
    expect(root.textContent).not.toContain(';');
    expectDecisionButtons();
  });

  it('a set the card could not look up keeps its id on its own line, with "(name not found)"', async () => {
    render(
      <ApprovalCard
        request={request({ tool_name: 'start_probe_training_run', payload: { model_id: 'm_1', eval_dataset_ids: IDS } })}
        context={EMPTY_CONTEXT}
      />,
    );
    const items = await expectSharedList(screen.getByTestId('probe-run-eval-sets'), /10 sets/, 10);
    expect(items.map((li) => li.textContent)).toEqual(IDS.map((id) => `${id} (name not found)`));
  });

  it("an evaluation's Probe datasets fact (evaluate_probe_monitor): one name (id) per line", async () => {
    const req = request({
      tool_name: 'evaluate_probe_monitor',
      payload: { _method: 'POST', _route: '/api/v1/probe-monitors/probes/pm_1/evaluate', probe_id: 'pm_1', dataset_ids: IDS },
    });
    // The fact still carries the whole value as text, and each set for the list.
    const fact = factsFor(req).find(([k]) => k === 'Probe datasets');
    expect(fact).toEqual(['Probe datasets', JSON.stringify(IDS), IDS]);
    render(<ApprovalCard request={req} context={CONTEXT} />);
    const root = screen.getByTestId('approval-fact-sets');
    expect(within(root).getByRole('button')).toHaveTextContent(/^10 sets\s*▸$/);
    expectDecisionButtons();
    const items = await expectSharedList(root, /^Probe datasets: 10 sets$/, 10);
    expect(items.map((li) => li.textContent)).toEqual(IDS.map((_, i) => LINE(i)));
  });

  it('the Probe datasets fact: a deleted view keeps its id with "(name not found)"; none is blank', async () => {
    const req = request({ tool_name: 'evaluate_probe_monitor', payload: { probe_id: 'pm_1', dataset_ids: [...IDS, 'pmd_gone'] } });
    render(<ApprovalCard request={req} context={CONTEXT} />);
    const items = await expectSharedList(screen.getByTestId('approval-fact-sets'), /11 sets/, 11);
    expect(items[10]).toHaveTextContent(/^pmd_gone \(name not found\)$/);
    items.forEach((li) => expect(li.textContent?.trim()).not.toBe(''));
  });

  it('the Probe datasets fact while the lookup is in flight: the id, "(looking up name…)", never "not found"', async () => {
    const req = request({ tool_name: 'evaluate_probe_monitor', payload: { probe_id: 'pm_1', dataset_ids: IDS } });
    render(<ApprovalCard request={req} context={{ ...EMPTY_CONTEXT, viewsPending: true }} />);
    const items = await expectSharedList(screen.getByTestId('approval-fact-sets'), /10 sets/, 10);
    expect(items.map((li) => li.textContent)).toEqual(IDS.map((id) => `${id} (looking up name…)`));
  });

  it("a capture's Probe datasets fact (capture_wsp_heldout) uses it too", () => {
    render(
      <ApprovalCard
        request={request({ tool_name: 'capture_wsp_heldout', payload: { probe_id: 'pm_1', dataset_ids: IDS.slice(0, 2) } })}
        context={CONTEXT}
      />,
    );
    const root = screen.getByTestId('approval-fact-sets');
    // Two sets: expanded by default, one per line, each by name AND id.
    expect(root).toHaveAttribute('data-set-list', 'expanded');
    expect(within(root).getAllByRole('listitem').map((li) => li.textContent)).toEqual([LINE(0), LINE(1)]);
  });

  it("a capture's Sets to capture: the rows in the summary, one set per line", async () => {
    const sets = IDS.map((id) => ({ dataset_id: id, role: 'eval', rows: 100, would_run: true }));
    render(
      <ApprovalCard
        request={request({ tool_name: 'capture_wsp_heldout', payload: { probe_id: 'pm_1' }, forecast: { sets } as never })}
        context={CONTEXT}
      />,
    );
    const root = screen.getByTestId('approval-wsp-sets');
    const items = await expectSharedList(root, /^Sets to capture: 10 sets, 1,000 rows$/, 10);
    expect(items[0]).toHaveTextContent(/^held-out 0 \(pmd_set0\) — eval, 100 rows$/);
    expect(items[9]).toHaveTextContent(/^held-out 9 \(pmd_set9\) — eval, 100 rows$/);
  });

  it('a non-list fact is still plain text', () => {
    render(
      <ApprovalCard
        request={request({ tool_name: 'evaluate_probe_monitor', payload: { probe_id: 'pm_1', dataset_ids: 'pmd_x' } })}
        context={EMPTY_CONTEXT}
      />,
    );
    expect(screen.queryByTestId('approval-fact-sets')).toBeNull();
    expect(screen.getByTestId('approval-facts')).toHaveTextContent('pmd_x');
  });
});
