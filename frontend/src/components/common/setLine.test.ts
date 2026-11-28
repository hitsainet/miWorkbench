/**
 * `setLine`: every probe-view line reads `<view name> (pmd_…)` (operator request 2026-11-27). The
 * id is never dropped and the line is never blank.
 *
 * NEGATIVE CONTROLS (2026-11-27): return the id alone, or the name alone → red here and at every
 * call site's test.
 */
import { describe, expect, it } from 'vitest';

import { ID_NOT_FOUND, NAME_NOT_FOUND, NAME_PENDING, setLine } from './setLine';

describe('setLine', () => {
  it("reads `<view name> (pmd_…)` — the operator's example", () => {
    expect(setLine('pmd_12c501631eae', 'ssn-d128-fu7a-fresh-id-test')).toBe(
      'ssn-d128-fu7a-fresh-id-test (pmd_12c501631eae)',
    );
  });

  it('a view with no name (deleted, or the lookup failed) keeps its id and says "(name not found)"', () => {
    expect(NAME_NOT_FOUND).toBe('(name not found)');
    expect(setLine('pmd_gone', undefined)).toBe('pmd_gone (name not found)');
    expect(setLine('pmd_gone', null)).toBe('pmd_gone (name not found)');
    expect(setLine('pmd_gone', '   ')).toBe('pmd_gone (name not found)');
  });

  it('a "name" that is the id again (an upstream fallback) counts as no name, not `pmd_x (pmd_x)`', () => {
    expect(setLine('pmd_x', 'pmd_x')).toBe('pmd_x (name not found)');
  });

  it('while the lookup is in flight it says so, never a false "not found"', () => {
    expect(setLine('pmd_x', undefined, { pending: true })).toBe(`pmd_x ${NAME_PENDING}`);
    // A name already known wins over pending.
    expect(setLine('pmd_x', 'known', { pending: true })).toBe('known (pmd_x)');
  });

  it('a name with no id (a server field carrying names only) says "(id not found)"', () => {
    expect(setLine('', 'toolace_balanced')).toBe(`toolace_balanced ${ID_NOT_FOUND}`);
  });

  it('is never blank', () => {
    expect(setLine('', '').trim()).not.toBe('');
    expect(setLine(undefined, undefined)).toContain(NAME_NOT_FOUND);
  });
});
