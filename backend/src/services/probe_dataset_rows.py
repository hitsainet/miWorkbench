"""Browse a probe dataset VIEW's rows: paged, projected, filtered by mapped label and text.

Operator request 2026-10-21: "I want to be able to browse the rows and columns in each dataset."

⚠ ONE CLASSIFIER. A row's mapped label here comes from `probe_monitor_inputs.classify_row`, the
same function `build_examples` tallies into the counts stored at creation. So the label a row
shows and the counts above the table are one computation, not two that happen to agree
(`test_probe_dataset_rows.py` pins it on a fixture, and a mutation of either side goes red).

⚠ THE STORED COUNTS STAY THE AUTHORITY. They are what every run on this view was submitted
against. When a request has to classify the whole view anyway (a label filter), the response
also carries `recount` and `counts_agree`; a disagreement means the download changed under the
view, which is worth showing rather than hiding.

Cost: a plain page reads only its own rows (`Dataset.select` over a memory-mapped Arrow file), so
the first page of a million-row calibration set is cheap. A label filter classifies every row
once per view and caches the outcome codes (one byte a row) for the next request; a text search
scans the input column in batches.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .probe_monitor_inputs import ROW_OUTCOMES, classify_row
from .probe_monitor_service import open_split

logger = logging.getLogger(__name__)

#: Page sizes the browser offers; the route's bounds.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
#: The default page of `download_rows`, as `get_probe_evaluation_rows` defaults to 100.
DEFAULT_DOWNLOAD_PAGE = 100

#: A cell longer than this is cut on the server and flagged, so one 2 MB conversation cannot
#: make a page unreadable. The UI truncates further and expands up to this.
MAX_CELL_CHARS = 20_000

#: Rows read per batch when a whole column has to be scanned.
SCAN_BATCH = 4_096

#: How many views' outcome arrays stay in memory.
CACHE_VIEWS = 4

_OUTCOME_CODE = {name: code for code, name in enumerate(ROW_OUTCOMES)}


class RowsRefused(ValueError):
    """A request the view cannot answer, with a message that says what to change."""


class RowsMoved(RowsRefused):
    """The download no longer holds, at an evaluation's positions, the rows that evaluation scored.

    `out_of_range` are positions past the end of the split; `disagree` are positions whose row the view now maps to
    another label than the one the evaluation stored (or to no label). Either means the text at those positions is
    not the text that was scored, so none of the page is served."""

    def __init__(self, message: str, *, out_of_range: Sequence[int], disagree: Sequence[int]):
        super().__init__(message)
        self.out_of_range = [int(p) for p in out_of_range]
        self.disagree = [int(p) for p in disagree]


@dataclass(frozen=True)
class ViewSpec:
    """What the browser needs from a `ProbeMonitorDataset` row (kept ORM-free for tests)."""

    id: str
    split: Optional[str]
    input_column: str
    label_column: str
    pair_column: Optional[str]
    label_mapping: Dict[str, str]
    keyword_filter: Optional[Dict[str, Any]]

    @classmethod
    def of(cls, view: Any) -> "ViewSpec":
        return cls(
            id=view.id,
            split=view.split,
            input_column=view.input_column,
            label_column=view.label_column,
            pair_column=view.pair_column,
            label_mapping=dict(view.label_mapping or {}),
            keyword_filter=dict(view.keyword_filter) if view.keyword_filter else None,
        )


_cache: "OrderedDict[Tuple[Any, ...], np.ndarray]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(spec: ViewSpec, path: Path) -> Tuple[Any, ...]:
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = None
    return (
        spec.id,
        str(path),
        stamp,
        spec.split,
        spec.input_column,
        spec.label_column,
        json.dumps(spec.label_mapping, sort_keys=True),
        json.dumps(spec.keyword_filter, sort_keys=True, default=str),
    )


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def outcome_codes(data: Any, spec: ViewSpec, path: Path) -> np.ndarray:
    """Every row's outcome as a code into `ROW_OUTCOMES`, computed once per view and cached."""
    key = _cache_key(spec, path)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached
    codes = np.empty(len(data), dtype=np.int8)
    position = 0
    columns = data.select_columns([spec.input_column, spec.label_column])
    for batch in columns.iter(batch_size=SCAN_BATCH):
        for raw_input, raw_label in zip(batch[spec.input_column], batch[spec.label_column]):
            row = classify_row(raw_input, raw_label, spec.label_mapping, spec.keyword_filter)
            codes[position] = _OUTCOME_CODE[row.outcome]
            position += 1
    with _cache_lock:
        _cache[key] = codes
        while len(_cache) > CACHE_VIEWS:
            _cache.popitem(last=False)
    return codes


def tally(codes: np.ndarray) -> Dict[str, int]:
    """Outcome codes → the five stored-count keys."""
    counts = np.bincount(codes.astype(np.int64), minlength=len(ROW_OUTCOMES))
    return {name: int(counts[code]) for name, code in _OUTCOME_CODE.items()}


def _text_of(value: Any) -> str:
    """What a search matches: a conversation's message contents, or the value as text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", item.get("value", ""))))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(value)


def matching_positions(data: Any, column: str, needle: str) -> np.ndarray:
    """Positions whose `column` contains `needle`, ignoring case."""
    wanted = needle.casefold()
    hits: List[int] = []
    position = 0
    for batch in data.select_columns([column]).iter(batch_size=SCAN_BATCH):
        for value in batch[column]:
            if wanted in _text_of(value).casefold():
                hits.append(position)
            position += 1
    return np.asarray(hits, dtype=np.int64)


def _jsonable(value: Any) -> Tuple[Any, bool]:
    """A cell made safe for JSON and bounded in size. Returns (value, truncated)."""
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value), False
        return value, False
    if isinstance(value, str):
        if len(value) > MAX_CELL_CHARS:
            return value[:MAX_CELL_CHARS], True
        return value, False
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>", False
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > MAX_CELL_CHARS:
        return text[:MAX_CELL_CHARS], True
    try:
        return json.loads(text), False
    except ValueError:
        return text, False


def inputs_at(
    spec: ViewSpec,
    path: Path,
    positions: Sequence[int],
    labels: Sequence[int],
) -> List[Dict[str, Any]]:
    """The view's input at each of an evaluation's `positions`, in the order given (ENH-010 FU-7a gap G-1).

    `labels` are the labels the evaluation STORED at those positions (1 positive, 0 negative). Each row is classified
    again through `classify_row` — the function that made the evaluation's examples — and must map to its stored label;
    a position past the split's end or a row that maps differently raises `RowsMoved` for the whole page, because the
    text there is not the text that was scored. Returns `[{"text", "text_cut"}]`, one per position. Read-only; reads
    only the requested rows and the view's input and label columns.
    """
    if len(positions) != len(labels):
        raise ValueError(f"{len(positions)} positions against {len(labels)} labels")
    if len(positions) > MAX_PAGE_SIZE:
        raise RowsRefused(f"page size must be at most {MAX_PAGE_SIZE}; got {len(positions)}")
    if not positions:
        return []
    data = open_split(path, spec.split)
    all_columns = list(data.column_names)
    view_columns = [spec.input_column, spec.label_column]
    missing = [c for c in view_columns if c not in all_columns]
    if missing:
        raise RowsRefused(
            f"the download no longer has the view's columns {missing}; it has {sorted(all_columns)}"
        )
    total = len(data)
    wanted = [int(p) for p in positions]
    out_of_range = [p for p in wanted if not 0 <= p < total]
    if out_of_range:
        raise RowsMoved(
            f"{len(out_of_range)} of the evaluation's positions are past the end of the download "
            f"({total} rows), so it is not the download the evaluation scored",
            out_of_range=out_of_range, disagree=[],
        )
    page = data.select(wanted).select_columns(view_columns)
    rows: List[Dict[str, Any]] = []
    disagree: List[int] = []
    for position, stored, record in zip(wanted, labels, page, strict=True):
        row = classify_row(
            record[spec.input_column], record[spec.label_column], spec.label_mapping, spec.keyword_filter,
        )
        if row.outcome != ("positive" if int(stored) == 1 else "negative"):
            disagree.append(position)
        value, cut = _jsonable(record[spec.input_column])
        rows.append({"text": value, "text_cut": cut})
    if disagree:
        raise RowsMoved(
            f"{len(disagree)} of the page's rows no longer map to the label the evaluation stored, so the download "
            f"changed under the evaluation and its text cannot be paired with these scores",
            out_of_range=[], disagree=disagree,
        )
    return rows


def browse(
    spec: ViewSpec,
    path: Path,
    *,
    offset: int = 0,
    limit: int = DEFAULT_PAGE_SIZE,
    columns: Optional[Sequence[str]] = None,
    label: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """One page of a view's rows. Read-only.

    `columns` projects the OTHER columns of the download; the view's input, label and pair
    columns are always returned, because the mapped label cannot be shown without them.
    `label` keeps rows whose outcome is that one of `ROW_OUTCOMES`; `search` keeps rows whose
    input contains the text (case-insensitive). `offset` past the end is an empty page, not an
    error, so a client paging after a refresh never gets a refusal for a valid question.
    """
    if offset < 0:
        raise RowsRefused("offset cannot be negative")
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise RowsRefused(f"page size must be between 1 and {MAX_PAGE_SIZE}; got {limit}")
    if label is not None and label not in ROW_OUTCOMES:
        raise RowsRefused(f"unknown label filter {label!r}; use one of {list(ROW_OUTCOMES)}")

    data = open_split(path, spec.split)
    all_columns = list(data.column_names)
    view_columns = [c for c in (spec.input_column, spec.label_column, spec.pair_column) if c]
    missing = [c for c in view_columns if c not in all_columns]
    if missing:
        raise RowsRefused(
            f"the download no longer has the view's columns {missing}; it has {sorted(all_columns)}"
        )
    others = [c for c in all_columns if c not in view_columns]
    if columns is None:
        projected = others
    else:
        unknown = [c for c in columns if c not in all_columns]
        if unknown:
            raise RowsRefused(f"unknown columns {unknown}; this dataset has {sorted(all_columns)}")
        projected = [c for c in others if c in set(columns)]

    total = len(data)
    positions: Optional[np.ndarray] = None
    recount: Optional[Dict[str, int]] = None
    if label is not None:
        codes = outcome_codes(data, spec, path)
        recount = tally(codes)
        positions = np.flatnonzero(codes == _OUTCOME_CODE[label])
    needle = (search or "").strip()
    if needle:
        hits = matching_positions(data, spec.input_column, needle)
        positions = hits if positions is None else np.intersect1d(positions, hits)

    matched = total if positions is None else int(len(positions))
    if positions is None:
        page_positions = list(range(min(offset, total), min(offset + limit, total)))
    else:
        page_positions = [int(p) for p in positions[offset:offset + limit]]

    rows: List[Dict[str, Any]] = []
    if page_positions:
        page = data.select(page_positions).select_columns(view_columns + projected)
        for position, record in zip(page_positions, page):
            row = classify_row(
                record[spec.input_column],
                record[spec.label_column],
                spec.label_mapping,
                spec.keyword_filter,
            )
            truncated: List[str] = []
            values: Dict[str, Any] = {}
            for name in view_columns + projected:
                value, cut = _jsonable(record[name])
                values[name] = value
                if cut:
                    truncated.append(name)
            rows.append(
                {
                    "position": position,
                    "outcome": row.outcome,
                    "mapped_label": row.target,
                    "raw_label": values[spec.label_column],
                    "input_kind": row.parsed.kind if row.parsed is not None else None,
                    "values": values,
                    "truncated": truncated,
                }
            )

    return {
        "view_id": spec.id,
        "split": spec.split,
        "input_column": spec.input_column,
        "label_column": spec.label_column,
        "pair_column": spec.pair_column,
        "other_columns": others,
        "returned_columns": view_columns + projected,
        "total_rows": total,
        "matched": matched,
        "offset": offset,
        "limit": limit,
        "label": label,
        "search": needle or None,
        "recount": recount,
        "rows": rows,
    }


def download_rows(
    path: Path,
    *,
    split: Optional[str] = None,
    offset: int = 0,
    limit: int = DEFAULT_DOWNLOAD_PAGE,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """One page of a DOWNLOADED dataset's rows, by position, with no probe dataset view (FU-7a: read a new set to
    prove it disjoint before registering it). Read-only; reads only the page's rows of the projected columns.

    `split` is required when the download has several: one not given is refused, never guessed. A download with no
    split names answers `split: null`. `columns` projects (omit for every column; unknown names are refused). Each
    cell is cut at `MAX_CELL_CHARS` and its column listed in the row's `truncated`. A page holds at most
    `MAX_PAGE_SIZE` rows; `offset` past the end is an empty page with `next_offset: null`.
    """
    from datasets import DatasetDict, load_from_disk

    if offset < 0:
        raise RowsRefused("offset cannot be negative")
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise RowsRefused(f"page size must be between 1 and {MAX_PAGE_SIZE}; got {limit}")
    loaded = load_from_disk(str(path))
    splits: List[str] = list(loaded.keys()) if isinstance(loaded, DatasetDict) else []
    if splits:
        if split is None and len(splits) > 1:
            raise RowsRefused(f"this download has several splits; name one: {splits}")
        chosen: Optional[str] = split if split is not None else splits[0]
        if chosen not in splits:
            raise RowsRefused(f"split {chosen!r} is not in this download; available: {splits}")
        data = loaded[chosen]
    else:
        if split is not None:
            raise RowsRefused(f"this download has one split and no split names; omit split (got {split!r})")
        chosen, data = None, loaded
    all_columns = list(data.column_names)
    if columns is None:
        returned = all_columns
    else:
        unknown = [c for c in columns if c not in all_columns]
        if unknown:
            raise RowsRefused(f"unknown columns {unknown}; this dataset has {sorted(all_columns)}")
        returned = [c for c in all_columns if c in set(columns)]
    total = len(data)
    page_positions = list(range(min(offset, total), min(offset + limit, total)))
    rows: List[Dict[str, Any]] = []
    if page_positions and returned:
        page = data.select(page_positions).select_columns(returned)
        for position, record in zip(page_positions, page, strict=True):
            values: Dict[str, Any] = {}
            truncated: List[str] = []
            for name in returned:
                value, cut = _jsonable(record[name])
                values[name] = value
                if cut:
                    truncated.append(name)
            rows.append({"position": position, "values": values, "truncated": truncated})
    elif page_positions:
        rows = [{"position": p, "values": {}, "truncated": []} for p in page_positions]
    end = offset + len(page_positions)
    return {
        "split": chosen,
        "splits": splits,
        "columns": all_columns,
        "returned_columns": returned,
        "total": total,
        "offset": offset,
        "limit": limit,
        "next_offset": end if page_positions and end < total else None,
        "max_cell_chars": MAX_CELL_CHARS,
        "rows": rows,
    }
