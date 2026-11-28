/**
 * A long probe GPU job — an evaluation, a WSP held-out capture — shows view k of n and rows x of y while it runs
 * (operator, 2026-11-27: "what is this job? where can I see its progress in a UI?").
 *
 * The panel is mounted with only the network mocked (the probes API and `GET /task-queue/active`), so removing the
 * tile's block, the panel's polling hook, or the prop that hands the tile its jobs turns a test here red. Controls are
 * recorded in 0xcc/reviews/probe_gpu_job_progress_controls_2026-11-27.md:
 *   F1  <ProbeGpuJobProgress> removed from ProbeTile             → the tile and panel tests
 *   F2  `gpuJobs={gpuJobs[probe.id]}` removed from the panel      → the panel tests
 *   F3  the hook's interval removed                               → "the bar moves" test
 *   F4  the Active Operations reading removed                     → the Monitor test
 */
import { act, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { TaskQueueEntry, ViewRowsProgressDetail } from '../../../types/taskQueue';
import type { ProbeMonitorSummary } from '../../../types/probeMonitor';

const api = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  listRuns: vi.fn(),
  listProbes: vi.fn(),
  listJudgeRuns: vi.fn(),
  listJudgePrompts: vi.fn(),
  getReport: vi.fn(),
}));
const tasks = vi.hoisted(() => ({ getActiveTasks: vi.fn() }));

vi.mock('../../../api/probeMonitors', () => ({ probeMonitorsApi: api }));
vi.mock('../../../api/taskQueue', () => tasks);
vi.mock('../../../hooks/useProbeMonitorWebSocket', () => ({ useProbeMonitorWebSocket: () => undefined }));

import { ProbeMonitorsPanel } from '../../panels/ProbeMonitorsPanel';
import { ProbeGpuJobProgress } from '../GpuJobProgress';
import { describeViewRows, describeViewRowsDetail } from '../gpuJobText';
import { ProbeTile } from '../ProbeTile';
import { useProbeMonitorsStore } from '../../../stores/probeMonitorsStore';

function detail(overrides: Partial<ViewRowsProgressDetail> = {}): ViewRowsProgressDetail {
  return {
    kind: 'probe_evaluation', phase: 'scoring', rows_done: 4512, rows_total: 33772, rows_estimated: false,
    set_index: 2, set_count: 10, set_id: 'pmd_c', set_name: 'anthropic_hh', set_rows_done: 812,
    set_rows_total: 3370, rate_per_second: 12.4, eta_seconds: 2400, eta_provisional: false,
    updated_at: '2026-11-27T13:00:00Z', ...overrides,
  };
}

function job(overrides: Partial<TaskQueueEntry> = {}, d: Partial<ViewRowsProgressDetail> = {}): TaskQueueEntry {
  return {
    id: 'tq_eval', task_id: 'celery-eval', task_type: 'probe_evaluation', entity_id: 'pm_80504153780f',
    entity_type: 'probe_monitor', status: 'running', progress: 13.36, error_message: null,
    retry_params: { gpu: 'auto' }, retry_count: 0, can_retry: true, created_at: '2026-11-27T12:00:00Z',
    started_at: '2026-11-27T12:01:00Z', completed_at: null, updated_at: null,
    entity_info: { name: 'Probe pm_80504153780f · layer 11 · mean', progress_detail: detail(d) },
    ...overrides,
  } as TaskQueueEntry;
}

function probe(id = 'pm_80504153780f'): ProbeMonitorSummary {
  return {
    id, run_id: 'pmr_x', layer: 11, rule: 'mean', rule_params: {}, variant: 'dense', sae_id: null,
    sae_feature_indices: null, val_metrics: { val_auroc: 0.99 }, selected: false, threshold: 2.8,
    target_fpr: 0.01, realised_fpr: 0.01, threshold_source: 'validation_negatives', streamable: true, rung: 1,
    rung_reasons: [], rung_language: 'detects in distribution', rung_next_step: null,
    created_at: '2026-09-26T10:00:00Z',
  } as unknown as ProbeMonitorSummary;
}

const barWidth = (block: HTMLElement) =>
  (block.querySelector('.bg-emerald-500') as HTMLElement | null)?.style.width;

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset());
  tasks.getActiveTasks.mockReset();
  window.localStorage.clear();
  api.listDatasets.mockResolvedValue([]);
  api.listRuns.mockResolvedValue([]);
  api.listProbes.mockResolvedValue([probe()]);
  api.listJudgeRuns.mockResolvedValue([]);
  api.listJudgePrompts.mockResolvedValue({ prompts: [], placeholders: {} });
  tasks.getActiveTasks.mockResolvedValue({ data: [] });
});

afterEach(() => {
  vi.useRealTimers();
});

describe('the words', () => {
  it('says the view and the rows the way the operator asked', () => {
    expect(describeViewRows(detail())).toBe('view 3/10, 4,512/33,772 rows');
    expect(describeViewRowsDetail(detail())).toBe('anthropic_hh: 812/3,370 rows · 12 rows/s · about 40 min left');
  });

  it('says "about" while the total is an estimate, and what it is doing before a view begins', () => {
    expect(describeViewRows(detail({ rows_estimated: true, set_index: null, phase: 'loading_model', rows_done: 0 })))
      .toBe('10 views, loading the model, 0/about 33,772 rows');
    expect(describeViewRowsDetail(detail({ phase: 'preparing', rate_per_second: null, eta_seconds: null })))
      .toBe('anthropic_hh: preparing the rows');
  });

  it('marks an early estimate provisional', () => {
    expect(describeViewRowsDetail(detail({ eta_provisional: true }))).toMatch(/about 40 min left \(provisional\)$/);
  });
});

describe('the block', () => {
  it('a running evaluation: the line, the bar at the row\'s percentage, the view in hand', () => {
    render(<ProbeGpuJobProgress task={job()} />);
    const block = screen.getByTestId('probe-gpu-job');
    expect(screen.getByTestId('probe-gpu-job-line')).toHaveTextContent('Evaluating… view 3/10, 4,512/33,772 rows');
    expect(barWidth(block)).toBe('13.36%');
    expect(screen.getByTestId('probe-gpu-job-detail')).toHaveTextContent('anthropic_hh: 812/3,370 rows');
  });

  it('a capture says Capturing', () => {
    render(<ProbeGpuJobProgress task={job({ task_type: 'wsp_heldout_capture' as never }, { kind: 'wsp_heldout_capture' })} />);
    expect(screen.getByTestId('probe-gpu-job-line')).toHaveTextContent('Capturing… view 3/10');
  });

  it('queued and stopped-reporting are said, not drawn as a bar', () => {
    const { unmount } = render(<ProbeGpuJobProgress task={job({ status: 'queued' as never, entity_info: null })} />);
    expect(screen.getByTestId('probe-gpu-job')).toHaveTextContent('Evaluation queued, waiting for a GPU');
    unmount();
    render(<ProbeGpuJobProgress task={job({ status: 'orphaned' as never })} />);
    expect(screen.getByTestId('probe-gpu-job')).toHaveTextContent('Evaluating stopped reporting');
  });

  it('the tile renders each of its jobs', () => {
    render(
      <ul>
        <ProbeTile probe={probe()} expanded={false} onToggle={vi.fn()} gpuJobs={[job()]} />
      </ul>,
    );
    expect(within(screen.getByTestId('probe-row')).getByTestId('probe-gpu-job-line')).toHaveTextContent(
      'Evaluating… view 3/10, 4,512/33,772 rows',
    );
  });
});

describe('the Probes page', () => {
  it('finds a running evaluation from the task queue, on the right probe', async () => {
    api.listProbes.mockResolvedValue([probe(), probe('pm_other')]);
    tasks.getActiveTasks.mockResolvedValue({ data: [job(), { ...job(), id: 'tq_j', task_type: 'jlens_fit', entity_id: 'pm_other', entity_type: 'model' }] });
    useProbeMonitorsStore.setState({ activeTab: 'probes', probes: [] });
    render(<ProbeMonitorsPanel />);
    const line = await screen.findByTestId('probe-gpu-job-line');
    expect(line).toHaveTextContent('Evaluating… view 3/10, 4,512/33,772 rows');
    const rows = screen.getAllByTestId('probe-row');
    expect(within(rows[0]).queryByTestId('probe-gpu-job')).not.toBeNull();
    expect(within(rows[1]).queryByTestId('probe-gpu-job')).toBeNull();
  });

  it('the bar moves without a reload, and the probe reloads when the job ends', async () => {
    vi.useFakeTimers();
    tasks.getActiveTasks
      .mockResolvedValueOnce({ data: [job({ progress: 10 }, { rows_done: 3377, set_index: 1 })] })
      .mockResolvedValueOnce({ data: [job({ progress: 55.5 }, { rows_done: 18744, set_index: 5 })] })
      .mockResolvedValue({ data: [] });
    useProbeMonitorsStore.setState({ activeTab: 'probes', probes: [] });
    render(<ProbeMonitorsPanel />);
    await act(async () => { await vi.advanceTimersByTimeAsync(10); });
    expect(screen.getByTestId('probe-gpu-job-line')).toHaveTextContent('view 2/10, 3,377/33,772 rows');
    expect(barWidth(screen.getByTestId('probe-gpu-job'))).toBe('10%');

    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(screen.getByTestId('probe-gpu-job-line')).toHaveTextContent('view 6/10, 18,744/33,772 rows');
    expect(barWidth(screen.getByTestId('probe-gpu-job'))).toBe('55.5%');

    const loadsBefore = api.listProbes.mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(screen.queryByTestId('probe-gpu-job')).toBeNull();
    expect(api.listProbes.mock.calls.length).toBe(loadsBefore + 1);
  });

  it('asks nothing while another tab is open', async () => {
    vi.useFakeTimers();
    useProbeMonitorsStore.setState({ activeTab: 'runs', probes: [] });
    render(<ProbeMonitorsPanel />);
    await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
    expect(tasks.getActiveTasks).not.toHaveBeenCalled();
  });
});
