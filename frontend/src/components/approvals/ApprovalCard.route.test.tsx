/**
 * 045 (A38; V-9; FTASKS 21.3): three REST routes share the action `start_labeling` — `POST /labeling` (row 38),
 * `POST /labeling/panel` (row 47) and `POST /extractions/{extraction_id}/label` (row 48). An approval is bound to its
 * route by the payload's `_method` and `_route`, and the operator tells the three apart by the card's route line.
 * The requests are the PUBLISHED payload vectors (`docs/schemas/miworkbench-approval-payload-vectors-v1.json`), so the
 * card is tested on exactly what an agent files.
 *
 * NEGATIVE CONTROL (045 A0 controls record): hide the card's route line → every case here red.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, within } from '@testing-library/react';
import type { ApprovalRequest } from '../../types/featureGroups';
import published from '../../../../docs/schemas/miworkbench-approval-payload-vectors-v1.json';

vi.mock('../../api/featureGroups', () => ({
  listApprovals: vi.fn(),
  listDecidedApprovals: vi.fn(() => new Promise(() => {})),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
  getCodeRuleState: vi.fn(),
  HISTORY_PAGE: 200,
}));
vi.mock('../../api/models', () => ({ getModel: vi.fn(() => Promise.resolve({ id: 'm_1', name: 'Llama' })) }));
vi.mock('../../api/probeMonitors', () => ({ probeMonitorsApi: { listDatasets: vi.fn(() => Promise.resolve([])) } }));
vi.mock('../../api/system', () => ({ getGPUList: vi.fn(() => Promise.resolve({ gpus: [] })) }));

import { ApprovalsBanner } from '../layout/ApprovalsBanner';
import { useApprovalsStore } from '../../stores/approvalsStore';
import { useUIStore } from '../../stores/uiStore';

interface Vector { row: string; action: string; method: string; route: string; payload: Record<string, unknown> }

const VECTORS = published.vectors as unknown as Vector[];

function request(row: string): ApprovalRequest {
  const v = VECTORS.find((x) => x.row === row);
  if (!v) throw new Error(`no vector for row ${row}`);
  return {
    id: `apr_${row}`, tool_name: v.action, status: 'pending', reason: null, steering_task_id: null,
    created_at: '2026-11-16T08:00:00Z', resolved_at: null, requested_by: 'agent:miassist', payload: v.payload,
  } as ApprovalRequest;
}

const ROWS = ['38', '47', '48'];

beforeEach(() => {
  useUIStore.setState({ activePanel: 'datasets' });
  useApprovalsStore.setState({ pending: ROWS.map(request), loaded: true, error: null });
});

describe('the route line on the approval card', () => {
  it('shows each start_labeling request with its own route', () => {
    render(<ApprovalsBanner />);
    const lines = ROWS.map((row) => {
      const card = document.querySelector(`[data-approval-id="apr_${row}"]`) as HTMLElement;
      expect(card).not.toBeNull();
      return within(card).getByTestId('approval-route').textContent;
    });
    expect(lines).toEqual([
      'POST /api/v1/labeling',
      'POST /api/v1/labeling/panel',
      'POST /api/v1/extractions/{extraction_id}/label',
    ]);
    expect(new Set(ROWS.map((r) => request(r).tool_name))).toEqual(new Set(['start_labeling']));
  });
});
