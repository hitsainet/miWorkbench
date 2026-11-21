/**
 * 045 FR-19 (A39; FTASKS 21.8): the settings page saves a provider key WITH the base URL it may be sent to.
 * The backend (`services/provider_keys.py`) sends the stored OpenAI key only to a base URL in
 * `openai_api_key_base_urls`, so the page must write that list with the key.
 *
 * Control: drop the binding upsert in `handleSave` → this test red.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { SettingsPanel, OPENAI_KEY_BASE_URLS, bindingCovers } from './SettingsPanel';
import { useSettingsStore } from '../../stores/settingsStore';
import { fetchAPI } from '../../api/client';

vi.mock('../../stores/settingsStore');
vi.mock('../../stores/trainingsStore');
vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>();
  return { ...actual, fetchAPI: vi.fn() };
});

const upsert = vi.fn().mockResolvedValue(undefined);

beforeEach(() => {
  vi.clearAllMocks();
  window.scrollTo = vi.fn() as never;
  (useSettingsStore as never as { mockReturnValue: (v: unknown) => void }).mockReturnValue({
    settings: [],
    isLoading: false,
    fetchAll: vi.fn().mockResolvedValue(undefined),
    getByCategory: vi.fn().mockReturnValue([]),
    upsert,
    remove: vi.fn().mockResolvedValue(undefined),
  });
  vi.mocked(fetchAPI).mockImplementation(async (path: string) => {
    if (path === '/settings/pin/status') return { configured: true, bypass_active: true } as never;
    return [] as never;
  });
});

describe('API key host binding', () => {
  it('saves the OpenAI key with the base URL it may be sent to', async () => {
    const user = userEvent.setup();
    render(<SettingsPanel />);
    await user.click(screen.getByRole('button', { name: 'API Keys' }));
    await user.click(await screen.findByPlaceholderText('Enter OpenAI API Key...'));
    await user.type(screen.getByPlaceholderText('sk-proj-...'), 'sk-test');
    const base = screen.getByTestId('openai_api_key-base-url');
    expect((base as HTMLInputElement).value).toBe('https://api.openai.com/v1');
    await user.clear(base);
    await user.type(base, 'http://judge.lan:8000/v1');
    await user.click(screen.getByRole('button', { name: 'Save OpenAI API Key' }));

    await waitFor(() => expect(upsert).toHaveBeenCalledTimes(2));
    expect(upsert.mock.calls[0][0]).toMatchObject({ key: 'openai_api_key', value: 'sk-test', is_sensitive: true });
    expect(upsert.mock.calls[1][0]).toEqual({
      key: OPENAI_KEY_BASE_URLS,
      value: JSON.stringify(['http://judge.lan:8000/v1']),
      is_sensitive: false,
      category: 'api_keys',
    });
  });

  it('keeps the stored compatible endpoint in the binding when the key is re-saved (OR1-045a0-8)', async () => {
    // R1 review fix (OR1-045a0-8), 2026-11-16. Control: write only the entered URL in `handleSave` → red.
    (useSettingsStore as never as { mockReturnValue: (v: unknown) => void }).mockReturnValue({
      settings: [
        { key: 'openai_api_key', value: 'sk-...TMKA', is_sensitive: true, category: 'api_keys' },
        { key: 'openai_compatible_endpoint', value: 'http://judge.lan:8000/v1', is_sensitive: false,
          category: 'endpoints' },
      ],
      isLoading: false,
      fetchAll: vi.fn().mockResolvedValue(undefined),
      getByCategory: vi.fn().mockReturnValue([]),
      upsert,
      remove: vi.fn().mockResolvedValue(undefined),
    });
    const user = userEvent.setup();
    render(<SettingsPanel />);
    await user.click(screen.getByRole('button', { name: 'API Keys' }));
    await user.click(await screen.findByRole('button', { name: 'Edit' }));
    await user.type(screen.getByPlaceholderText('sk-proj-...'), 'sk-new');
    await user.click(screen.getByRole('button', { name: 'Save OpenAI API Key' }));

    await waitFor(() => expect(upsert).toHaveBeenCalledTimes(2));
    expect(upsert.mock.calls[1][0]).toEqual({
      key: OPENAI_KEY_BASE_URLS,
      value: JSON.stringify(['https://api.openai.com/v1', 'http://judge.lan:8000/v1']),
      is_sensitive: false,
      category: 'api_keys',
    });
  });

  it('writes no binding for a key that has none (the Hugging Face token)', async () => {
    const user = userEvent.setup();
    render(<SettingsPanel />);
    await user.click(screen.getByRole('button', { name: 'API Keys' }));
    await user.click(await screen.findByPlaceholderText('Enter HuggingFace Token...'));
    await user.type(screen.getByPlaceholderText('hf_...'), 'hf_x');
    expect(screen.queryByTestId('hf_token-base-url')).toBeNull();
    await user.click(screen.getByRole('button', { name: 'Save HuggingFace Token' }));
    await waitFor(() => expect(upsert).toHaveBeenCalledTimes(1));
    expect(upsert.mock.calls[0][0].key).toBe('hf_token');
  });
});

function storeWith(settings: Array<Record<string, unknown>>) {
  (useSettingsStore as never as { mockReturnValue: (v: unknown) => void }).mockReturnValue({
    settings,
    isLoading: false,
    fetchAll: vi.fn().mockResolvedValue(undefined),
    getByCategory: vi.fn().mockReturnValue([]),
    upsert,
    remove: vi.fn().mockResolvedValue(undefined),
  });
}

describe('a compatible endpoint outside the saved key binding (R2 review fix OR2-045a0-4, 2026-11-16)', () => {
  const KEY = { key: 'openai_api_key', value: 'sk-...TMKA', is_sensitive: true, category: 'api_keys' };
  const BOUND = { key: OPENAI_KEY_BASE_URLS, value: JSON.stringify(['https://api.openai.com/v1', 'http://judge.lan:8000/v1']),
    is_sensitive: false, category: 'api_keys' };

  it('says plainly that the stored key is not sent there, and saves only the endpoint (never widens the binding)', async () => {
    // Control: drop the `compat-key-not-bound` note → red.
    storeWith([KEY, BOUND, { key: 'openai_compatible_endpoint', value: 'http://judge.lan:8000/v1',
      is_sensitive: false, category: 'endpoints' }, { key: 'openai_compatible_model', value: 'm',
      is_sensitive: false, category: 'endpoints' }]);
    const user = userEvent.setup();
    render(<SettingsPanel />);
    await user.click(screen.getByRole('button', { name: 'Endpoints' }));
    const input = await screen.findByDisplayValue('http://judge.lan:8000/v1');
    expect(screen.queryByTestId('compat-key-not-bound')).toBeNull();
    await user.clear(input);
    await user.type(input, 'http://other-judge.lan:9000/v1');
    expect(screen.getByTestId('compat-key-not-bound').textContent).toContain('not sent to this endpoint');
    await user.click(screen.getByRole('button', { name: /^Save$/ }));
    await waitFor(() => expect(upsert).toHaveBeenCalled());
    expect(upsert.mock.calls.map((c) => c[0].key)).not.toContain(OPENAI_KEY_BASE_URLS);
  });

  it('matches the backend rule: scheme, host and port equal, path equal or below; no saved list → nothing to say', () => {
    const list = JSON.stringify(['https://api.openai.com/v1', 'http://judge.lan:8000/v1']);
    expect(bindingCovers('http://judge.lan:8000/v1/', list)).toBe(true);
    expect(bindingCovers('http://JUDGE.lan:8000/v1/chat', list)).toBe(true);
    expect(bindingCovers('http://judge.lan:8000/v1x', list)).toBe(false);
    expect(bindingCovers('http://judge.lan:8001/v1', list)).toBe(false);
    expect(bindingCovers('https://api.openai.com:443/v1', list)).toBe(true);
    expect(bindingCovers('http://judge.lan:8000/v1', undefined)).toBeNull();
    expect(bindingCovers('http://judge.lan:8000/v1', 'not json')).toBe(false);
  });
});
