"""ENH-010 G5: evaluation fixtures as the post-G5 writer stores them (a truncation mask and an untruncated block).

Since G5, rung 2 reads an evaluation without `truncated_path` as "not measured — truncation not recorded" (G5-D3),
so a fixture that stands for a CURRENT evaluation must carry what the writer now writes. `post_g5` adds it with
NO row truncated, so the untruncated block equals the all-rows one and every count the fixture was written around is
unchanged. Fixtures whose subject IS truncation build their own masks (`test_g5_*.py`).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

#: A mask path for a fixture with no files on disk: rung readers then fall back to the recorded untruncated count,
#: as they do for an evaluation whose files are gone (GR-4's gate).
NOT_ON_DISK = "/nonexistent/g5-fixture__truncated.npy"


def post_g5(metrics: Dict[str, Any], *, threshold: Optional[float] = None, stem: Any = None,
            rows: Optional[int] = None, max_length: int = 4096) -> Dict[str, Any]:
    """`metrics=` and `truncated_path=` kwargs for a `ProbeMonitorEvaluation`, with nothing truncated.

    With `stem` and `rows`, a zero mask of `rows` entries is written beside the fixture's other arrays."""
    stored = dict(metrics)
    stored["untruncated"] = dict(metrics)
    stored["truncation"] = {
        "rows": 0, "positives": 0, "negatives": 0,
        "fired_at_bar": 0 if threshold is not None else None,
        "threshold": float(threshold) if threshold is not None else None,
        "max_length": int(max_length), "policy": "excluded",
    }
    stored["not_scored"] = {"truncated": 0}
    path = NOT_ON_DISK
    if stem is not None and rows is not None:
        path = f"{stem}__truncated.npy"
        np.save(path, np.zeros(int(rows), dtype=np.int8))
    return {"metrics": stored, "truncated_path": path}
