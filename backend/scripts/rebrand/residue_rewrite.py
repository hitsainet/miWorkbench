"""The ENH-009 P9 residue rewrite (043 FR-043.11 to FR-043.17; FTDD §6.3; FTID §3.3). P9 only: deleted at P9 close.

Two steps, from `backend/`:

    python -m scripts.rebrand.residue_rewrite review <cls> <review.tsv> [--restrict POSITIONS] [--keep REGEX]
    python -m scripts.rebrand.residue_rewrite apply <review.tsv>

`review` runs the brand guard's OWN extractor for one P9 class and its OWN rules from the export
(`tests/support/brand_surfaces.py`), so the tool and the guard cannot disagree about what is allowed (memory
"derive from the registry, not the source"). Every old-brand stem the guard would report becomes one row:
`action, cls, path, row, col, before, after, token, line`. The action is

* `apply`   the stem maps by the fixed case-preserving table `MAP`;
* `keep`    the line matches `--keep` (a sentence about the rename itself, P9-D1);
* `restrict` `--restrict` names a file of `path:row:col` positions and this one is not among them (the census'
            fix groups: a position the census attributes to a frozen value, a register gap or history is never
            rewritten by this tool, even when the export does not yet cover it);
* `unmapped` the stem's spelling is not in `MAP` (reported, never guessed);
* `filename` a file NAME holding a stem: `apply` renames it with `git mv` (the `after` column is the new path).

A person reads the review file and edits actions before `apply`. `apply` refuses: a tree with uncommitted changes,
a row whose line no longer equals `line`, a path in a record tree, an applied migration or the MCP rename map,
and any stem the export allows (the rules are re-read at apply time). It edits only the `apply` rows.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tests.support import brand_surfaces as bs
from tests.support import legacy_export as le

#: Exact spellings only; any other spelling of a stem is reported as `unmapped`.
MAP: dict[str, str] = {
    "miLLM": "miServer", "MiLLM": "MiServer", "millm": "miserver", "MILLM": "MISERVER", "Millm": "Miserver",  # p9: keep, brand-guard plant or rule
    "miStudio": "miWorkbench", "MiStudio": "MiWorkbench", "mistudio": "miworkbench", "MISTUDIO": "MIWORKBENCH",  # p9: keep, brand-guard plant or rule
    "Mistudio": "Miworkbench", "MechInterp Studio": "miWorkbench", "MechInterp-Studio": "miWorkbench",  # p9: keep, brand-guard plant or rule
}

#: Never written by this tool, whatever a review file says (RG-8, RG-9; FTID §3.3).
REFUSED = re.compile(r"^(0xcc/|\.claude/context/|backend/alembic/versions/|backend/tests/support/mcp_rename_map\.json$"
                     r"|docs/legacy-identifiers)")

FIELDS = ("action", "cls", "path", "row", "col", "before", "after", "token", "line")


class Refused(RuntimeError):
    pass


@dataclass(frozen=True)
class Row:
    action: str
    cls: str
    path: str
    row: int
    col: int
    before: str
    after: str
    token: str
    line: str


def _where(item: bs.Item) -> tuple[str, int | None]:
    path, _, rest = item.where.rpartition(":")
    if item.where.endswith(" (file name)"):
        return item.where.removesuffix(" (file name)"), None
    return path, int(rest)


def uncovered(text: str, rules: bs.Rules) -> list[re.Match[str]]:
    """The stems of `text` the guard reports: `bs.scan`'s own rule, position by position."""
    spans = bs._allowed_spans(text, rules)
    return [m for m in bs.STEM.finditer(text) if not bs._covered(spans, *m.span())]


def load_positions(path: Path) -> set[tuple[str, int, int]]:
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p, r, c = line.rsplit(":", 2)
            out.add((p, int(r), int(c)))
    return out


def review(cls: str, root: Path = bs.REPO_ROOT, restrict: set[tuple[str, int, int]] | None = None,
           keep: re.Pattern[str] | None = None) -> list[Row]:
    if cls not in bs.P9_CLASSES:
        raise Refused(f"{cls!r} is not a P9 class {bs.P9_CLASSES}")
    rules = bs.rules_for(le.load(root))
    rows: list[Row] = []
    for item in bs.file_items(cls, root)[1]:
        path, row = _where(item)
        for m in uncovered(item.text, rules):
            token = bs._token(item.text, *m.span())
            if row is None:
                new_path = bs.STEM.sub(lambda s: MAP.get(s.group(0), s.group(0)), path)
                rows.append(Row("filename", cls, path, 0, m.start(), m.group(0), new_path, token, path))
                break
            col = item.col + m.start()
            line = (root / path).read_text(encoding="utf-8").split("\n")[row - 1]
            if m.group(0) not in MAP:
                action = "unmapped"
            elif restrict is not None and (path, row, col) not in restrict:
                action = "restrict"
            elif keep is not None and keep.search(line):
                action = "keep"
            else:
                action = "apply"
            rows.append(Row(action, cls, path, row, col, m.group(0), MAP.get(m.group(0), ""), token, line))
    return rows


def write_review(rows: list[Row], out: Path) -> None:
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(FIELDS)
        for r in rows:
            w.writerow([getattr(r, f) for f in FIELDS])


def read_review(path: Path) -> list[Row]:
    with path.open(encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        if tuple(header) != FIELDS:
            raise Refused(f"{path}: not a review file (header {header})")
        return [Row(a, c, p, int(r), int(col), b, af, t, ln) for a, c, p, r, col, b, af, t, ln in rd]


def _dirty(root: Path) -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def apply(rows: list[Row], root: Path = bs.REPO_ROOT) -> dict[str, int]:
    """Apply the `apply` rows (text first, then file renames). Returns counts per outcome."""
    if dirty := _dirty(root):
        raise Refused(f"uncommitted changes in the tree ({len(dirty)}): commit or stash first")
    rules = bs.rules_for(le.load(root))
    counts: dict[str, int] = defaultdict(int)
    by_file: dict[str, list[Row]] = defaultdict(list)
    renames: list[Row] = []
    for r in rows:
        if r.action not in ("apply",):
            counts[f"skipped:{r.action}"] += 1
            continue
        if REFUSED.match(r.path):
            raise Refused(f"{r.path}: a record, an applied migration or the rename map is never rewritten")
        if r.row == 0:
            renames.append(r)
        else:
            by_file[r.path].append(r)
    for path, file_rows in by_file.items():
        p = root / path
        text = p.read_text(encoding="utf-8")
        lines = text.split("\n")
        # The allowance is re-derived now, by the guard's own extractor and rules, on the file as it is.
        reported: set[tuple[str, int, int]] = set()
        for cls in {r.cls for r in file_rows}:
            for item in bs.EXTRACTORS[cls](path, text):
                _, row = _where(item)
                if row is not None:
                    reported.update((cls, row, item.col + m.start()) for m in uncovered(item.text, rules))
        edits: dict[int, list[Row]] = defaultdict(list)
        for r in file_rows:
            line = lines[r.row - 1] if r.row - 1 < len(lines) else None
            if line != r.line:
                counts["changed_line"] += 1
                print(f"skip (line changed): {path}:{r.row}", file=sys.stderr)
            elif line[r.col:r.col + len(r.before)] != r.before or MAP.get(r.before) != r.after:
                counts["mismatch"] += 1
                print(f"skip (stem not at its column, or not the table's mapping): {path}:{r.row}:{r.col}",
                      file=sys.stderr)
            elif (r.cls, r.row, r.col) not in reported:
                counts["allowed"] += 1
                print(f"skip (the export allows it): {path}:{r.row}:{r.col}", file=sys.stderr)
            else:
                edits[r.row].append(r)
        for row, row_edits in edits.items():
            line = lines[row - 1]
            for r in sorted({(e.col, e.before, e.after): e for e in row_edits}.values(), key=lambda e: -e.col):
                line = line[:r.col] + r.after + line[r.col + len(r.before):]
                counts["applied"] += 1
            lines[row - 1] = line
        p.write_text("\n".join(lines), encoding="utf-8")
    for r in renames:
        new = root / r.after
        new.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", r.path, r.after], cwd=root, check=True)
        counts["renamed"] += 1
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rv = sub.add_parser("review")
    rv.add_argument("cls", choices=bs.P9_CLASSES)
    rv.add_argument("out", type=Path)
    rv.add_argument("--restrict", type=Path)
    rv.add_argument("--keep")
    ap_ = sub.add_parser("apply")
    ap_.add_argument("review", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "review":
            rows = review(args.cls, restrict=load_positions(args.restrict) if args.restrict else None,
                          keep=re.compile(args.keep) if args.keep else None)
            write_review(rows, args.out)
            tally: dict[str, int] = defaultdict(int)
            for r in rows:
                tally[r.action] += 1
            print(f"{args.cls}: {len(rows)} row(s) {dict(tally)} -> {args.out}")
        else:
            print(apply(read_review(args.review)))
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
