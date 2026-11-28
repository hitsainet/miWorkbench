/**
 * A list of probe datasets or evaluation sets: ONE SET PER LINE, left-justified, and the whole list
 * collapsible to a single summary line ("Evaluation sets: 10 sets ▸"). Operator request 2026-11-27:
 * an approval naming ten sets on one joined line ("a; b; c; …") could not be read at a glance.
 *
 * - Collapsed by default when there are MORE than `SET_LIST_COLLAPSE_ABOVE` (3) sets, expanded for 3
 *   or fewer. The default follows the count (a list that loads after mount still gets the right
 *   one) until the operator toggles it; their choice then holds.
 * - The toggle is a real `<button>` with `aria-expanded` and `aria-controls`, so Enter and Space work,
 *   and a visible focus ring. The summary always says how many sets there are.
 * - Collapsed items stay in the DOM under `hidden`: a checklist's checkboxes keep their state, and
 *   `aria-controls` always names an element that exists.
 *
 * Every call site uses this one component; do not copy it per site.
 */
import { useId, useState, type ReactNode } from 'react';

/** Collapsed by default above this many sets; 3 or fewer start expanded. */
export const SET_LIST_COLLAPSE_ABOVE = 3;

export interface SetListItem {
  key: string;
  content: ReactNode;
}

interface SetListProps {
  /** What the list is ("Evaluation sets"). Always in the toggle's accessible name. */
  label: string;
  /** False where a `<dt>` or heading beside the list already shows the label. */
  showLabel?: boolean;
  items: SetListItem[];
  /** Added to the summary after the count, e.g. "2 chosen" or "2,000 rows". */
  detail?: string;
  /** Shown instead of a toggle when there are no items. */
  emptyText?: string;
  /** Override the default threshold (tests, or a site that needs another). */
  collapseAbove?: number;
  /** Text classes for the summary line and the empty text. */
  className?: string;
  /** Classes for each item line. */
  itemClassName?: string;
  testId?: string;
}

function setCountText(n: number): string {
  return `${n.toLocaleString('en-US')} ${n === 1 ? 'set' : 'sets'}`;
}

export function SetList({
  label,
  showLabel = true,
  items,
  detail,
  emptyText = 'none',
  collapseAbove = SET_LIST_COLLAPSE_ABOVE,
  className = 'text-xs text-slate-600 dark:text-slate-400',
  itemClassName = '',
  testId = 'set-list',
}: SetListProps) {
  const listId = useId();
  const [chosenOpen, setChosenOpen] = useState<boolean | null>(null);
  const open = chosenOpen ?? items.length <= collapseAbove;

  if (items.length === 0) {
    return (
      <p className={`text-left ${className}`} data-testid={testId} data-set-list="empty">
        {showLabel ? `${label}: ` : ''}
        {emptyText}
      </p>
    );
  }

  const summary = `${setCountText(items.length)}${detail ? `, ${detail}` : ''}`;
  return (
    <div
      className={`text-left ${className}`}
      data-testid={testId}
      data-set-list={open ? 'expanded' : 'collapsed'}
    >
      <button
        type="button"
        aria-expanded={open}
        aria-controls={listId}
        aria-label={`${label}: ${summary}`}
        onClick={() => setChosenOpen(!open)}
        className="inline-flex items-center gap-1 text-left rounded-control hover:underline focus:outline-hidden focus-visible:ring-2 focus-visible:ring-sky-500"
        data-testid={`${testId}-toggle`}
      >
        <span>
          {showLabel ? `${label}: ` : ''}
          {summary}
        </span>
        <span aria-hidden="true">{open ? '▾' : '▸'}</span>
      </button>
      <ul
        id={listId}
        hidden={!open}
        className="mt-0.5 ml-3 list-none space-y-0.5 text-left"
        data-testid={`${testId}-items`}
      >
        {items.map(item => (
          <li
            key={item.key}
            className={`block text-left break-words ${itemClassName}`}
            data-testid="set-list-item"
          >
            {item.content}
          </li>
        ))}
      </ul>
    </div>
  );
}
