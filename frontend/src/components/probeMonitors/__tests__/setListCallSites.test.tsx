/**
 * Every probe-page place that lists evaluation sets uses the ONE shared `SetList` (operator request
 * 2026-11-27): one set per line, left-justified, collapsed to a summary line above 3 sets. Each test
 * mounts the real call site with 10 sets and reads the shared component's wiring (`data-set-list`,
 * the toggle's `aria-expanded`/`aria-controls`), then expands it and counts the lines.
 *
 * Every line names the view AND its raw `pmd_…` id (operator request 2026-11-27, second pass):
 * `<name> (pmd_…)`, or `pmd_… (name not found)` for a view the page cannot name. The name is the one
 * the site already showed (`config || name` where the config tells views of one repo apart).
 *
 * NEGATIVE CONTROLS: put a call site back to its old inline rendering (chips in a flex-wrap row, or
 * the joined line) → its test here goes red; drop the name or the id from `setLine` → red.
 */
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { JudgeRunForm } from '../JudgeRunForm';
import { ProbeActions } from '../ProbeActions';
import { ProbeRunForm } from '../ProbeRunForm';
import { RunTile } from '../RunTile';
import { TransferCaution } from '../TransferCaution';
import type { ProbeDataset, ProbeEvaluation, ProbeMonitorSummary, ThresholdTransfer } from '../../../types/probeMonitor';

const TEN = Array.from({ length: 10 }, (_, i) => ({
  id: `pmd_e${i}`,
  name: `eval view ${i}`,
  config: `set_${i}`,
  role: 'eval',
  distribution: i % 2 ? 'out_of_distribution' : null,
  counts: { positive: 10, negative: 20 },
})) as unknown as ProbeDataset[];

const probeState = {
  datasets: [{ id: 'pmd_train', name: 'train view', role: 'train', distribution: null }, ...TEN],
  loadDatasets: vi.fn(),
  submitRun: vi.fn(),
};

vi.mock('../../../stores/modelsStore', () => ({
  useModelsStore: () => ({ models: [{ id: 'm_1', name: 'Llama' }], fetchModels: vi.fn() }),
}));
vi.mock('../../../stores/probeMonitorsStore', () => ({
  useProbeMonitorsStore: (selector?: (s: typeof probeState) => unknown) =>
    selector ? selector(probeState) : probeState,
}));
vi.mock('../../common/GpuSelect', () => ({ GpuSelect: () => <div data-testid="gpu-select" /> }));
vi.mock('../../../api/probeMonitors', () => ({
  probeMonitorsApi: { listJudgePrompts: vi.fn(() => Promise.resolve({ prompts: [] })) },
}));

/** The shared component's root at this call site, collapsed, with its count — then expanded. */
async function expectSharedList(testId: string, summary: RegExp, lines: number) {
  const root = screen.getByTestId(testId);
  expect(root).toHaveAttribute('data-set-list', 'collapsed');
  const toggle = within(root).getByRole('button', { expanded: false });
  expect(toggle).toHaveAccessibleName(summary);
  const list = document.getElementById(toggle.getAttribute('aria-controls') ?? '');
  expect(list).not.toBeNull();
  expect(list).not.toBeVisible();
  await userEvent.click(toggle);
  expect(root).toHaveAttribute('data-set-list', 'expanded');
  expect(list).toBeVisible();
  const items = within(list as HTMLElement).getAllByRole('listitem');
  expect(items).toHaveLength(lines);
  items.forEach((li) => expect(li).toHaveClass('text-left', 'block'));
  return items;
}

describe('set lists on the probe page use the shared SetList', () => {
  it("the probe report's evaluate checklist (ProbeActions)", async () => {
    render(
      <ProbeActions
        probeId="pm_1"
        datasets={TEN}
        alreadyEvaluated={[]}
        onEvaluate={vi.fn()}
        onScore={vi.fn()}
        onPollScore={vi.fn()}
      />
    );
    const items = await expectSharedList('evaluate-sets', /^Evaluation sets: 10 sets, 0 chosen$/, 10);
    expect(items[0]).toHaveTextContent(/^set_0 \(pmd_e0\)$/);
    expect(items[1]).toHaveTextContent(/^set_1 \(pmd_e1\)\s*OOD$/);
    await userEvent.click(screen.getByTestId('evaluate-set-pmd_e3'));
    expect(screen.getByTestId('evaluate-sets-toggle')).toHaveTextContent('Evaluation sets: 10 sets, 1 chosen');
  });

  it('the training form\'s "Evaluation sets" (ProbeRunForm)', async () => {
    render(<ProbeRunForm />);
    const items = await expectSharedList('run-eval-sets', /^Evaluation sets: 10 sets, 0 chosen$/, 10);
    expect(within(items[9]).getByRole('checkbox', { name: /^eval view 9 \(pmd_e9\)/ })).toBeInTheDocument();
    expect(items[0]).toHaveTextContent(/^eval view 0 \(pmd_e0\)$/);
    // The training-view picker too.
    expect(screen.getByRole('option', { name: 'train view (pmd_train)' })).toBeInTheDocument();
  });

  it("the judge run form's evaluation sets (JudgeRunForm)", async () => {
    render(
      <JudgeRunForm
        datasets={TEN}
        probes={[{ id: 'pm_1', layer: 1, rule: 'mean', variant: 'dense' }] as unknown as ProbeMonitorSummary[]}
        onSubmit={vi.fn()}
      />
    );
    const items = await expectSharedList('judge-sets', /^Evaluation sets: 10 sets, 0 chosen$/, 10);
    expect(within(items[0]).getByTestId('judge-set-pmd_e0')).toBeInTheDocument();
    expect(items.map((li) => li.textContent)).toEqual(TEN.map((d) => `${d.config} (${d.id})`));
  });

  it("a run tile's Evaluate cell (RunTile)", async () => {
    // pmd_e9 has no name on this page (deleted since the run): its line keeps the id and says so.
    const names = Object.fromEntries(TEN.slice(0, 9).map((d) => [d.id, d.name]));
    render(
      <ul>
        <RunTile
          run={{ id: 'pmr_1', model_id: 'm_1', train_dataset_id: 'pmd_train', eval_dataset_ids: TEN.map((d) => d.id),
                 status: 'completed', created_at: '2026-11-27T10:00:00Z', completed_at: '2026-11-27T11:00:00Z' } as never}
          datasetNames={names}
          onCancel={vi.fn()}
          onDelete={vi.fn()}
        />
      </ul>
    );
    const items = await expectSharedList('run-evaluate-sets', /^Evaluation sets: 10 sets$/, 10);
    expect(items.map((li) => li.textContent)).toEqual([
      ...TEN.slice(0, 9).map((d) => `${d.name} (${d.id})`),
      'pmd_e9 (name not found)',
    ]);
    // The tile's training view too.
    expect(screen.getByTitle('pmd_train (name not found)')).toBeInTheDocument();
  });

  it("the probe report's unreachable sets (TransferCaution)", async () => {
    const transfer = { caution: null, unreachable_sets: TEN.map((d) => d.config) } as unknown as ThresholdTransfer;
    // The server names the sets only; the ids come from the report's evaluations. set_9 has none.
    const evaluations = TEN.slice(0, 9).map((d) => ({ dataset_id: d.id, metrics: { name: d.config } })) as unknown as ProbeEvaluation[];
    render(<TransferCaution transfer={transfer} evaluations={evaluations} />);
    const items = await expectSharedList('report-unreachable-set-list', /^Sets the threshold is above every score on: 10 sets$/, 10);
    expect(items.map((li) => li.textContent)).toEqual([
      ...TEN.slice(0, 9).map((d) => `${d.config} (${d.id})`),
      'set_9 (id not found)',
    ]);
    expect(screen.getByTestId('report-unreachable-sets')).toHaveTextContent(/recall there is zero/i);
  });
});
