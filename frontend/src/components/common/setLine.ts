/**
 * One probe view on one line: ITS NAME AND ITS RAW `pmd_…` ID, always both (operator request
 * 2026-11-27, after the 10-set evaluation card listed bare ids and the training card bare names).
 *
 *   setLine('pmd_12c501631eae', 'ssn-d128-fu7a-fresh-id-test')
 *     → 'ssn-d128-fu7a-fresh-id-test (pmd_12c501631eae)'
 *
 * The name is what a reader recognises; the id is what pins the view (two views can share a name,
 * and the id is what an agent, a log or the API uses). So neither is ever dropped:
 * - no name (the view was deleted, or the lookup failed) → `pmd_… (name not found)`;
 * - the lookup still in flight (`pending`) → `pmd_… (looking up name…)`, never a false "not found";
 * - a name with no id (a server field that carries names only) → `<name> (id not found)`;
 * - a "name" that is just the id again (a fallback upstream) counts as no name.
 * The line is never blank.
 *
 * Every place that lists probe views uses this one function; do not format the line per site.
 */
export const NAME_NOT_FOUND = '(name not found)';
export const NAME_PENDING = '(looking up name…)';
export const ID_NOT_FOUND = '(id not found)';

export function setLine(
  id: string | null | undefined,
  name: string | null | undefined,
  options: { pending?: boolean } = {},
): string {
  const i = typeof id === 'string' ? id.trim() : '';
  const raw = typeof name === 'string' ? name.trim() : '';
  const n = raw === i ? '' : raw;
  if (i && n) return `${n} (${i})`;
  if (i) return `${i} ${options.pending ? NAME_PENDING : NAME_NOT_FOUND}`;
  if (n) return `${n} ${ID_NOT_FOUND}`;
  return `a set with no name or id ${NAME_NOT_FOUND}`;
}
