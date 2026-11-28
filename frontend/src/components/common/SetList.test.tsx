/**
 * The shared set list (operator request 2026-11-27): one set per line, left-justified, the whole list
 * collapsible to one summary line that says how many sets there are; collapsed by default above 3.
 *
 * NEGATIVE CONTROLS (record in the hand-back of feat/set-lists-one-per-line): (1) render the items
 * joined on one line ("a; b; c") instead of one `<li>` each → the one-per-line and alignment tests red;
 * (2) remove the collapse (always open, no `hidden`) → the collapsed and keyboard tests red.
 */
import { describe, it, expect } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SET_LIST_COLLAPSE_ABOVE, SetList } from './SetList';

const items = (n: number) =>
  Array.from({ length: n }, (_, i) => ({
    key: `pmd_${i}`,
    content: `“set ${i}” — ${i + 1} positive, ${i + 2} negative rows`,
  }));

function toggle() {
  return screen.getByTestId('set-list-toggle');
}

describe('SetList', () => {
  it('collapses by default above three sets, to one summary line that says how many', () => {
    expect(SET_LIST_COLLAPSE_ABOVE).toBe(3);
    render(<SetList label="Evaluation sets" items={items(10)} />);
    const t = toggle();
    expect(t.tagName).toBe('BUTTON');
    expect(t).toHaveAttribute('type', 'button');
    expect(t).toHaveAttribute('aria-expanded', 'false');
    expect(t).toHaveTextContent('Evaluation sets: 10 sets');
    expect(t).toHaveTextContent('▸');
    expect(t).toHaveAccessibleName('Evaluation sets: 10 sets');
    const list = screen.getByTestId('set-list-items');
    expect(t).toHaveAttribute('aria-controls', list.id);
    expect(list).not.toBeVisible();
    // Nothing of the sets is readable while collapsed: the summary is the only line.
    expect(screen.queryByRole('listitem')).toBeNull();
    expect(screen.getByTestId('set-list')).toHaveAttribute('data-set-list', 'collapsed');
  });

  it('expands to one set per line, left-justified, each with its own text', async () => {
    render(<SetList label="Evaluation sets" items={items(10)} />);
    await userEvent.click(toggle());
    expect(toggle()).toHaveAttribute('aria-expanded', 'true');
    expect(toggle()).toHaveTextContent('▾');
    const list = screen.getByRole('list');
    expect(list).toBeVisible();
    expect(list.tagName).toBe('UL');
    expect(list).toHaveClass('text-left');
    const lines = within(list).getAllByRole('listitem');
    expect(lines).toHaveLength(10);
    lines.forEach((li, i) => {
      expect(li.tagName).toBe('LI');
      expect(li).toHaveClass('block', 'text-left');
      expect(li).toHaveTextContent(`“set ${i}” — ${i + 1} positive, ${i + 2} negative rows`);
      // One set per line: no line carries a second set or a joining separator.
      expect(li.textContent).not.toMatch(/;|“set \d+”.*“set \d+”/);
    });
    expect(screen.getByTestId('set-list')).toHaveClass('text-left');
  });

  it('starts expanded for three or fewer sets, and can still be collapsed', async () => {
    render(<SetList label="Evaluation sets" items={items(3)} />);
    expect(toggle()).toHaveAttribute('aria-expanded', 'true');
    expect(toggle()).toHaveTextContent('Evaluation sets: 3 sets');
    expect(screen.getAllByRole('listitem')).toHaveLength(3);
    await userEvent.click(toggle());
    expect(toggle()).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('listitem')).toBeNull();
  });

  it('four sets is above the threshold: collapsed', () => {
    render(<SetList label="Evaluation sets" items={items(4)} />);
    expect(toggle()).toHaveAttribute('aria-expanded', 'false');
  });

  it('toggles from the keyboard: Enter and Space, with a visible focus ring', async () => {
    const user = userEvent.setup();
    render(<SetList label="Evaluation sets" items={items(10)} />);
    await user.tab();
    expect(toggle()).toHaveFocus();
    expect(toggle().className).toMatch(/focus-visible:ring-2/);
    await user.keyboard('{Enter}');
    expect(toggle()).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getAllByRole('listitem')).toHaveLength(10);
    await user.keyboard(' ');
    expect(toggle()).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('listitem')).toBeNull();
  });

  it('keeps the label in the accessible name when a heading beside it shows the label', () => {
    render(<SetList label="Probe datasets" showLabel={false} items={items(5)} detail="2 chosen" />);
    expect(toggle()).toHaveTextContent(/^5 sets, 2 chosen/);
    expect(toggle()).toHaveAccessibleName('Probe datasets: 5 sets, 2 chosen');
  });

  it('says one set, and shows the empty text with no toggle when there are none', () => {
    const { unmount } = render(<SetList label="Evaluation sets" items={items(1)} />);
    expect(toggle()).toHaveAccessibleName('Evaluation sets: 1 set');
    unmount();
    render(<SetList label="Evaluation sets" items={[]} />);
    expect(screen.queryByRole('button')).toBeNull();
    expect(screen.getByTestId('set-list')).toHaveTextContent('Evaluation sets: none');
  });

  it('a list that arrives after mount gets the default for its size; a toggle then holds', async () => {
    const { rerender } = render(<SetList label="Evaluation sets" items={[]} />);
    rerender(<SetList label="Evaluation sets" items={items(10)} />);
    expect(toggle()).toHaveAttribute('aria-expanded', 'false');
    await userEvent.click(toggle());
    rerender(<SetList label="Evaluation sets" items={items(11)} />);
    expect(toggle()).toHaveAttribute('aria-expanded', 'true');
  });
});
