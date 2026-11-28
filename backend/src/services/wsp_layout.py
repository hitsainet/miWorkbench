"""Where WSP files live on disk (ENH-010 044 task 3.3, FTDD §4.4, F44-D18).

Everything is under the parent probe's RUN directory (`artifact_dir_for(run_id)`), so `DELETE /runs/{id}`, which
removes that directory, reclaims every capture, bundle and baseline with it — no second janitor, no orphan bytes.
Every path is DERIVED from ids miWorkbench minted (`wac_…`, `wsb_…`), never taken from a request: these paths are
written, served by name from a manifest list, and removed.
"""
from __future__ import annotations

import re
from pathlib import Path

from .probe_monitor_run import artifact_dir_for

#: The ids this module will build a path from: the ORM's own prefixes and hex tails (`models/wsp.py`).
_ID = {"capture": re.compile(r"^wac_[0-9a-f]{12}$"), "bundle": re.compile(r"^wsb_[0-9a-f]{12}$")}
_RUN = re.compile(r"^pmr_[0-9a-f]{12}$")


def _checked(kind: str, value: str) -> str:
    pattern = _RUN if kind == "run" else _ID[kind]
    if not isinstance(value, str) or not pattern.match(value):
        raise ValueError(f"not a {kind} id: {value!r}")
    return value


def wsp_root(run_id: str) -> Path:
    """`<run dir>/wsp`."""
    return artifact_dir_for(_checked("run", run_id)) / "wsp"


def capture_dir(run_id: str, capture_id: str) -> Path:
    """`<run dir>/wsp/captures/<wac_id>`: tokens.f16, offsets/positions/labels/n_scored/truncated .npy."""
    return wsp_root(run_id) / "captures" / _checked("capture", capture_id)


def bundle_dir(run_id: str, bundle_id: str) -> Path:
    """`<run dir>/wsp/bundles/<wsb_id>`: manifest.json, heads.safetensors, practice_*.npy, baselines/."""
    return wsp_root(run_id) / "bundles" / _checked("bundle", bundle_id)


def baselines_dir(run_id: str, bundle_id: str) -> Path:
    """`<bundle dir>/baselines`: fitted baseline heads, never listed in the manifest's files."""
    return bundle_dir(run_id, bundle_id) / "baselines"
