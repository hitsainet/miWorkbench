/**
 * The brand guard's mode, frontend mirror of `BRAND_GUARD_MODE` in
 * `backend/tests/support/brand_surfaces.py` (042 FTDD §6.7; TPD-31; CF-21).
 *
 * `report` until S001 closes P4; flipped to `enforce` in this repository's first commit after that close
 * (CF-26). `backend/tests/unit/test_brand_guard_mode.py` and `brandMode.test.ts` hold the two constants
 * equal and tie them to the export header's `closed`.
 */
export type BrandGuardMode = 'report' | 'enforce';

export const BRAND_GUARD_MODE: BrandGuardMode = 'enforce';

/**
 * R23 (operator, 2026-11-11): the guard is ENFORCED where the full export exists (the private repository,
 * whose CI a commit must pass before it is mirrored) and only REPORTS on the public mirror. The mirror
 * carries the public export alone, which withholds the old host names (DR-29) and the private repository
 * names; a string those records allow would be a finding there, so an enforced mirror run would fail on
 * text the private run has already passed.
 *
 * `variant` is the export header's `variant` (`full` | `public`). Anything other than `full` reports.
 */
export function effectiveMode(mode: BrandGuardMode, variant: string): BrandGuardMode {
  return mode === 'enforce' && variant === 'full' ? 'enforce' : 'report';
}

/** What the real-tree test must assert empty: the findings when enforced, nothing when reporting. */
export function failuresFor(findings: readonly string[], mode: BrandGuardMode, variant: string): readonly string[] {
  return effectiveMode(mode, variant) === 'enforce' ? findings : [];
}

/**
 * ENH-009 P9 (043 FR-043.2): the backend guard's per-class mode, mirrored. Each class is report-only until its
 * residue group closes; `brandMode.test.ts` holds this map equal to `CLASS_CLOSED` in
 * `backend/tests/support/brand_surfaces.py`, where the P9 classes are scanned (frontend files included).
 */
export type P9Class = 'comments' | 'docs_text' | 'test_text' | 'test_values' | 'identifiers' | 'deploy_text';

export const CLASS_CLOSED: Readonly<Record<P9Class, boolean>> = {
  comments: true,
  docs_text: true,
  test_text: true,
  test_values: false,
  identifiers: true,
  deploy_text: true,
};
