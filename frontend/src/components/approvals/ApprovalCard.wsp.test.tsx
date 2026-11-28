/**
 * ENH-010 044 (FTASKS 4.4, 14.2): the two weight-space requests on the operator's card.
 *
 * * A capture names its action in words, the GPU line its tool read when it filed (and says plainly when NO card had
 *   the memory, or when nothing could be read), and the sets with their rows.
 * * An import's facts come from the miForge run document (run, candidate, verdict, bundle, patch size) — the
 *   documents and the base64 patch are never dumped into the facts list (they stay, in full, under "All arguments").
 *
 * NEGATIVE CONTROLS (record 0xcc/reviews/wsp_gated_capture_import_2026-11-19.md): delete the `<WspCaptureForecast …/>`
 * line in ApprovalCard → the capture tests red; delete the `WSP_IMPORT_TOOL` branch of `factsFor` → the import test red.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, waitFor, within } from '@testing-library/react';
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
// One view the card can name (pmd_eval) and one it cannot (pmd_cal, e.g. deleted since).
vi.mock('../../api/probeMonitors', () => ({
  probeMonitorsApi: { listDatasets: vi.fn(() => Promise.resolve([{ id: 'pmd_eval', name: 'ssn held-out eval' }])) },
}));
vi.mock('../../api/system', () => ({ getGPUList: vi.fn(() => Promise.resolve({ gpus: [] })) }));

import { ApprovalsPanel } from '../panels/ApprovalsPanel';
import { useApprovalsStore } from '../../stores/approvalsStore';
import { useUIStore } from '../../stores/uiStore';

const ROUTE = { _method: 'POST', _route: '/api/v1/wsp/captures' };
const SETS = [
  { dataset_id: 'pmd_eval', role: 'eval', rows: 1200, would_run: true },
  { dataset_id: 'pmd_cal', role: 'calibration', rows: 800, would_run: true },
];

function capture(id: string, forecast: ApprovalRequest['forecast']): ApprovalRequest {
  return {
    id, tool_name: 'capture_wsp_heldout', status: 'pending', reason: null, steering_task_id: null,
    created_at: '2026-11-19T08:00:00Z', resolved_at: null, requested_by: 'mcp',
    payload: { ...ROUTE, probe_id: 'pm_0123456789ab' }, forecast,
  };
}

const FITS = capture('apr_fits', {
  gpu: { checked: true, fits: true, required_mb: 18432, requested: 'auto',
         summary: 'GPU 0 (NVIDIA GeForce RTX 3090, 23,000 of 24,576 MB free) can take it: the capture needs ~18,432 MB' },
  sets: SETS,
});
const FULL = capture('apr_full', {
  gpu: { checked: true, fits: false, required_mb: 18432, requested: 'auto',
         summary: 'No single GPU has the ~18,432 MB this job needs.' },
  sets: SETS,
});
const UNREAD = capture('apr_unread', {
  gpu: { checked: false, fits: null, summary: 'no GPU is visible to the backend, so free memory cannot be read' },
  sets: [],
});

const BLOB = 'QUJD'.repeat(4000); // 16,000 base64 characters = 12,000 bytes
const IMPORT: ApprovalRequest = {
  id: 'apr_import', tool_name: 'import_wsp_candidate', status: 'pending', reason: null, steering_task_id: null,
  created_at: '2026-11-19T08:00:00Z', resolved_at: null, requested_by: 'mcp',
  payload: {
    _method: 'POST', _route: '/api/v1/wsp/candidates/import',
    run_document: { run_id: 'wsr_1', candidate_id: 'wsc_7', verdict: { value: 'not_significant' },
                    request: { bundle: { bundle_id: 'wsb_9' } } },
    patch_manifest: { patch_sha256: 'a'.repeat(64) },
    edit_npz_b64: BLOB,
    acknowledgement: { verdict: 'not_significant', reason: 'operator wants it measured' },
  },
};

function card(id: string): HTMLElement {
  const found = document.querySelector(`[data-approval-id="${id}"]`);
  if (!found) throw new Error(`no card ${id}`);
  return found as HTMLElement;
}

beforeEach(() => {
  useUIStore.setState({ activePanel: 'datasets' });
  useApprovalsStore.setState({ pending: [FITS, FULL, UNREAD, IMPORT], loaded: true, error: null });
});

describe('a WSP capture request', () => {
  it('names the action in words and shows the GPU line and the sets', async () => {
    render(<ApprovalsPanel />);
    const c = within(card('apr_fits'));
    expect(c.getByTestId('approval-action')).toHaveTextContent("Capture a probe's held-out activations on a graphics card");
    expect(c.getByTestId('approval-wsp-gpu')).toHaveTextContent('GPU: GPU 0 (NVIDIA GeForce RTX 3090');
    // One set per line (the shared SetList, 2026-11-27); two sets start expanded.
    const sets = within(c.getByTestId('approval-wsp-sets'));
    expect(sets.getByRole('button', { expanded: true })).toHaveAccessibleName('Sets to capture: 2 sets, 2,000 rows');
    // Each set by name AND id (2026-11-27); a view the lookup does not have keeps its id, said plainly.
    await waitFor(() =>
      expect(sets.getAllByRole('listitem').map((li) => li.textContent)).toEqual([
        'ssn held-out eval (pmd_eval) — eval, 1200 rows',
        'pmd_cal (name not found) — calibration, 800 rows',
      ]),
    );
  });

  it('says plainly when no card had the memory', () => {
    render(<ApprovalsPanel />);
    expect(within(card('apr_full')).getByTestId('approval-wsp-gpu')).toHaveTextContent(
      'GPU: NO card has the memory now — No single GPU has the ~18,432 MB this job needs.',
    );
  });

  it('says when free memory could not be read, rather than nothing', () => {
    render(<ApprovalsPanel />);
    const c = within(card('apr_unread'));
    expect(c.getByTestId('approval-wsp-gpu')).toHaveTextContent('GPU: not checked — no GPU is visible');
    expect(c.getByTestId('approval-wsp-sets')).toHaveTextContent('Sets: none were listed');
  });
});

describe('a WSP import request', () => {
  it('reads its facts from the run document and never lists the blobs as facts', () => {
    render(<ApprovalsPanel />);
    const c = within(card('apr_import'));
    expect(c.getByTestId('approval-action')).toHaveTextContent('Import a weight-space candidate as a new probe');
    const facts = c.getByTestId('approval-facts');
    for (const text of ['wsr_1', 'wsc_7', 'not_significant', 'wsb_9', '12,000 bytes', 'earns its rung at import',
                        'not_significant: operator wants it measured']) {
      expect(facts).toHaveTextContent(text);
    }
    // FU-5 (2026-11-25): an imported candidate is NOT a rung-0 probe; it earns its rung at import
    // (`pm_ae08dd79a26e` came in at rung 2), so the card never promises rung 0.
    expect(facts.textContent ?? '').not.toMatch(/rung[- ]0/i);
    expect(facts.textContent ?? '').not.toContain(BLOB.slice(0, 64));
    expect(facts.textContent ?? '').not.toContain('patch_sha256');
    // Every argument is still there, in full, collapsed.
    expect(c.getByTestId('approval-arguments').textContent ?? '').toContain(BLOB);
  });
});
