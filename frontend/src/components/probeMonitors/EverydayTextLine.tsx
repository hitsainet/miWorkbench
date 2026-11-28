/**
 * The false-alarm checks rung 2 needs, in plain words: everyday text (ENH-006 D82) and nonsense text (D84).
 * "On everyday text: 0 false alarms in 1,200 (allowed up to 24)" / "Fired on 100% of nonsense text (2,000 rows)".
 *
 * Every sentence and number is the server's (`recompute_rung` → `everyday_text` / `nonsense_text` on the
 * report); this component only lays them out, so the panel and the MCP report cannot disagree.
 */
import type { EverydayTextCheck } from '../../types/probeMonitor';

/** ENH-010 G5: the counts the server reports for one set's left-out truncated rows, or null when there are none. */
export function truncatedNote(excluded: number | null | undefined, fired: number | null | undefined, noun: string): string | null {
  if (!excluded) return null;
  const firing = fired === null || fired === undefined ? 'an unknown number' : fired.toLocaleString();
  return `${excluded.toLocaleString()} truncated ${noun}${excluded === 1 ? ' is' : 's are'} not counted (${firing} of them fire)`;
}

/** ENH-010 G5: said when leaving truncated rows out changed a check's verdict. */
export const VERDICT_CHANGED_NOTE =
  'Leaving out truncated rows changed this verdict. Rows longer than the run\'s maximum length are not measured.';

interface TextCheckLineProps {
  check: EverydayTextCheck | null;
  /** "Everyday text" / "Nonsense text" — the label before a never-checked reason. */
  label: string;
  /** What staying quiet means for this kind of text, after a never-checked reason. */
  needs: string;
  testId: string;
}

export function TextCheckLine({ check, label, needs, testId }: TextCheckLineProps) {
  if (!check) return null;
  const tone = check.passed ? 'text-slate-400' : 'text-amber-300';
  return (
    <div className="mt-2 text-xs" data-testid={testId}>
      {check.sets.length === 0 ? (
        <p className={tone}>
          {label}: {check.reason}. {needs} (at least {check.min_rows.toLocaleString()} rows).
        </p>
      ) : (
        <>
          {check.sets.map((set) => {
            const note = truncatedNote(set.truncated_excluded, set.truncated_fired, 'row');
            return (
              <p
                key={set.name}
                className={set.counts && set.within_allowance ? 'text-slate-400' : 'text-amber-300'}
                data-testid={`${testId}-set`}
              >
                {set.sentence}
                {/* The server's sentence ends with this note; shown on its own only when it does not. */}
                {note && !(set.sentence ?? '').includes(note) ? (
                  <span data-testid={`${testId}-truncated`}>; {note}</span>
                ) : null}
              </p>
            );
          })}
          {!check.passed ? <p className={tone}>Rung 2 not reached: {check.reason}</p> : null}
          {check.verdict_changed_by_exclusion ? (
            <p className="text-slate-400" data-testid={`${testId}-verdict-changed`}>{VERDICT_CHANGED_NOTE}</p>
          ) : null}
        </>
      )}
    </div>
  );
}

export function EverydayTextLine({ check }: { check: EverydayTextCheck | null }) {
  return (
    <TextCheckLine
      check={check}
      label="Everyday text"
      needs="Rung 2 needs a probe to stay quiet on ordinary prose at its shipped bar"
      testId="everyday-text"
    />
  );
}

export function NonsenseTextLine({ check }: { check: EverydayTextCheck | null }) {
  return (
    <TextCheckLine
      check={check}
      label="Nonsense text"
      needs="Rung 2 needs a probe to stay quiet on text with no meaning (word salad, random characters, repeated tokens, garbled code) at its shipped bar"
      testId="nonsense-text"
    />
  );
}
