"""B5a follow-up (operator approval 2026-11-24; review record `0xcc/reviews/b5a_import_determinism_2026-11-24.md`):
each WSP score call records the arithmetic it was computed with (`wsp_score_calls.arithmetic`, revision
`5e3c9a1d7b42`), so an import whose P23 bit check fails names the field that differs.

This file: the record's fields as this host answers them (`packed_scoring.mkl_kernel_family`,
`wsp_score.arithmetic_record`), the comparison (`wsp_import.arithmetic_differences`) and the migration. The wiring
(written at score time, read at import) is pinned in `test_wsp_import.py` on a real run.

The kernel family is MKL's OWN answer, through the service-layer entry points torch's statically linked MKL exports
(`mkl_serv_cpu_detect`, `mkl_serv_cbwr_get`, `MKL_Get_Version_String`). The child tests below prove that it follows
MKL's dispatch, not ATen's: on MKL's emulated Intel AVX-512 kernels it reads code path 7 (ATen reads "AVX512" on both),
and with MKL's conditional reproducibility on it says so (the bits change: measured 2026-11-24).
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from src.services import packed_scoring, wsp_import, wsp_score

BACKEND = Path(__file__).resolve().parents[2]
TORCH_CPU = Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"
needs_mkl = pytest.mark.skipif(not torch.backends.mkl.is_available() or not TORCH_CPU.exists(),
                               reason="SKIPPED LOUDLY: this torch has no statically linked MKL to ask")


# ── the kernel family ─────────────────────────────────────────────────────────


@needs_mkl
def test_the_kernel_family_is_mkls_own_code_path():
    """The family names the code path MKL's dispatcher answers in this very process, and the MKL build."""
    library = ctypes.CDLL(str(TORCH_CPU))
    detect = library.mkl_serv_cpu_detect
    detect.argtypes, detect.restype = [], ctypes.c_int
    family = packed_scoring.mkl_kernel_family()
    assert family is not None, "this torch has MKL, so its kernel family is known"
    assert family.startswith("MKL ") and " build " in family, family
    assert f", code path {detect()}" in family, family
    if "MKL_CBWR" not in os.environ:
        assert "CNR" not in family, f"conditional reproducibility is off here, yet the family says {family!r}"


def test_unknown_is_null_never_a_guess(monkeypatch):
    """No MKL (or its entry points missing): the family is None, and the record carries None for it."""
    monkeypatch.setattr(packed_scoring, "_MKL_FAMILY_FUNCTIONS", None)
    monkeypatch.setattr(torch.backends.mkl, "is_available", lambda: False)
    assert packed_scoring.mkl_kernel_family() is None
    assert wsp_score.arithmetic_record(3)["mkl_kernel_family"] is None
    monkeypatch.setattr(packed_scoring, "_MKL_FAMILY_FUNCTIONS", None)
    monkeypatch.setattr(torch.backends.mkl, "is_available", lambda: True)
    monkeypatch.setattr(packed_scoring, "MKL_CODE_PATH_SYMBOL", "mkl_serv_no_such_entry_point")
    assert packed_scoring.mkl_kernel_family() is None


def test_the_record_is_the_four_fields_of_this_process():
    record = wsp_score.arithmetic_record(5)
    assert tuple(record) == wsp_score.ARITHMETIC_FIELDS == (
        "mkl_kernel_family", "torch_version", "scorer_threads", "numpy_version")
    assert record == {"mkl_kernel_family": packed_scoring.mkl_kernel_family(), "torch_version": torch.__version__,
                      "scorer_threads": 5, "numpy_version": np.__version__}
    json.dumps(record)   # a JSONB value


_CHILD = textwrap.dedent("""
    import ctypes, json, sys
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    import torch
    from src.services.packed_scoring import mkl_kernel_family
    family = mkl_kernel_family()
    # MKL's own answer in THIS child, read beside the family, so a test compares the family with what MKL really took
    # here (its code path moves with MKL_CBWR on some CPUs: train W3, 2026-11-26).
    library = ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"))
    detect = library.mkl_serv_cpu_detect
    detect.argtypes, detect.restype = [], ctypes.c_int
    print(json.dumps({"family": family, "code_path": int(detect()), "aten": torch.backends.cpu.get_cpu_capability()}))
""")

#: MKL's Intel predicates answered "yes" (B5a's shim, without forcing the code path): MKL's own dispatcher then picks
#: its Intel AVX-512 kernels on any AVX-512 CPU. `libtorch_cpu.so` calls these through its PLT, so LD_PRELOAD interposes.
_PREDICATES_C = r"""
int mkl_serv_intel_cpu(void) { return 1; }
int mkl_serv_intel_cpu_true(void) { return 1; }
int mkl_serv_cpuisskl(void) { return 1; }
int mkl_serv_cpuiszen(void) { return 0; }
"""


def _child(tmp_path: Path, **env_extra: str) -> dict:
    (tmp_path / "child.py").write_text(_CHILD)
    # MKL's own dispatch variables are the caller's, not the test's: an inherited MKL_ENABLE_INSTRUCTIONS caps the code
    # path (AVX2 gives 5) and an inherited MKL_CBWR adds "CNR", so each child starts without them unless it asks.
    inherited = {k: v for k, v in os.environ.items()
                 if k not in ("MKL_ENABLE_INSTRUCTIONS", "MKL_CBWR", "MKL_DEBUG_CPU_TYPE")}
    env = {**inherited, "PYTHONWARNINGS": "ignore", **env_extra}
    if "MKL_CBWR" not in env_extra:
        env.pop("MKL_CBWR", None)
    done = subprocess.run([sys.executable, str(tmp_path / "child.py"), str(BACKEND)], env=env, capture_output=True,
                          text=True, timeout=300, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _avx512() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return all(f" {flag}" in flags for flag in ("avx512f", "avx512dq", "avx512bw", "avx512vl"))


def _intel_shim(tmp_path: Path) -> Path:
    """B5a's predicate shim, compiled for this test; skips loudly where MKL cannot take its Intel AVX-512 path."""
    compiler = shutil.which("cc") or shutil.which("gcc")
    if platform.system() != "Linux" or platform.machine() != "x86_64" or compiler is None:
        pytest.skip("SKIPPED LOUDLY: the Intel emulation needs Linux x86_64 and a C compiler")
    if not _avx512():
        pytest.skip("SKIPPED LOUDLY: no AVX-512 (F, DQ, BW, VL) here, so MKL cannot take its AVX-512 code path")
    (tmp_path / "shim.c").write_text(_PREDICATES_C)
    subprocess.run([compiler, "-shared", "-fPIC", "-O2", "-o", str(tmp_path / "shim.so"), str(tmp_path / "shim.c")],
                   check=True, capture_output=True)
    return tmp_path / "shim.so"


@needs_mkl
def test_on_mkls_intel_avx512_kernels_the_family_reads_code_path_7(tmp_path):
    """The defect the record exists for: B5a's AMD-scored hash does not reproduce on Intel AVX-512. With MKL's Intel
    checks answered yes, MKL itself takes code path 7, and so must the family; ATen's capability cannot tell."""
    emulated = _child(tmp_path, LD_PRELOAD=str(_intel_shim(tmp_path)))
    assert emulated["family"] is not None and emulated["family"].endswith(", code path 7"), emulated


_FAMILY = re.compile(r"(?P<build>MKL .+?), code path (?P<path>\d+)(?:, CNR (?P<cnr>\d+))?")


@needs_mkl
@pytest.mark.parametrize("cpu", ["native", "emulated_intel_avx512"])
def test_conditional_reproducibility_is_part_of_the_family(tmp_path, cpu):
    """`MKL_CBWR` changes gemv's bits (B5a trial 3; on the Zen 4 workstation every CNR setting gave other bits than
    off), so a family with it on is not the family with it off.

    The code path is NOT assumed equal across the two children: on an Intel AVX-512 CPU, `MKL_CBWR=COMPATIBLE` makes
    MKL dispatch code path 0 where it takes 7 without it (CI run 45893118361, shard 5, 2026-11-26: off `code path 7`,
    on `code path 0, CNR 3`). Each family must name the path MKL reports in its own child. The emulated case runs
    B5a's shim, which reproduces that runner's 7 to 0 move on any AVX-512 CPU."""
    extra = {} if cpu == "native" else {"LD_PRELOAD": str(_intel_shim(tmp_path))}
    off = _child(tmp_path, **extra)
    on = _child(tmp_path, MKL_CBWR="COMPATIBLE", **extra)
    parsed_off, parsed_on = _FAMILY.fullmatch(off["family"] or ""), _FAMILY.fullmatch(on["family"] or "")
    assert parsed_off and parsed_on, (off, on)
    assert parsed_on["build"] == parsed_off["build"], (off, on)
    assert "CNR" not in off["family"], off
    assert on["family"].endswith(", CNR 3"), on
    assert int(parsed_off["path"]) == off["code_path"], off
    assert int(parsed_on["path"]) == on["code_path"], on
    assert on["family"] != off["family"], (off, on)


# ── the comparison ────────────────────────────────────────────────────────────

_HERE = {"mkl_kernel_family": "MKL 2024.2 build 20240605, code path 0", "torch_version": "2.14.1+cpu",
         "scorer_threads": 8, "numpy_version": "2.5.4"}
_OTHER = {"mkl_kernel_family": "MKL 2024.2 build 20240605, code path 7", "torch_version": "2.15.0+cpu",
          "scorer_threads": 2, "numpy_version": "2.6.0"}


@pytest.mark.parametrize("field", wsp_score.ARITHMETIC_FIELDS)
def test_each_differing_field_is_named(field):
    at_scoring = {**_HERE, field: _OTHER[field]}
    assert wsp_import.arithmetic_differences(at_scoring, _HERE) == [
        {"field": field, "at_scoring": _OTHER[field], "now": _HERE[field]}]


def test_every_differing_field_is_named_in_the_fields_order():
    assert [d["field"] for d in wsp_import.arithmetic_differences(_OTHER, _HERE)] == list(wsp_score.ARITHMETIC_FIELDS)


@pytest.mark.parametrize("at_scoring", [None, {}, "not a record"])
def test_a_call_without_a_record_differs_in_nothing(at_scoring):
    assert wsp_import.arithmetic_differences(at_scoring, _HERE) == []


def test_a_field_unknown_on_either_side_is_not_named():
    assert wsp_import.arithmetic_differences({**_OTHER, "mkl_kernel_family": None}, _HERE)[0]["field"] == \
        "torch_version"
    assert wsp_import.arithmetic_differences(_OTHER, {**_HERE, "mkl_kernel_family": None})[0]["field"] == \
        "torch_version"
    assert wsp_import.arithmetic_differences(_HERE, dict(_HERE)) == []


# ── the migration ─────────────────────────────────────────────────────────────

REVISION, PARENT = "5e3c9a1d7b42", "408803691ec6"


def test_the_migration_round_trips():
    """`5e3c9a1d7b42` upgrades, downgrades and upgrades again on a fresh database built by the migrations (named after
    this run's own test database and xdist worker): the nullable JSONB column appears, disappears and reappears, and
    the downgrade leaves the parent's `wsp_score_calls` exactly as it was."""
    from urllib.parse import urlparse

    import psycopg2
    import sqlalchemy as sa

    from tests.support.test_databases import sync_test_url

    source = sync_test_url()  # never the dev database (OSD-57)
    parsed = urlparse(source.replace("+psycopg2", "").replace("+asyncpg", ""))
    assert parsed.hostname and parsed.username, f"DATABASE_URL_SYNC unparseable: {source!r}"
    own = (parsed.path or "/mistudio_test").lstrip("/").split("?")[0] or "mistudio_test"
    name = f"{own}_wsa_mig_{os.getenv('PYTEST_XDIST_WORKER', 'solo')}"[:63]
    admin_kw = dict(host=parsed.hostname, port=parsed.port or 5432, user=parsed.username,
                    password=parsed.password or "", dbname="postgres")

    def admin(sql):
        conn = psycopg2.connect(**admin_kw)
        conn.autocommit = True
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
        finally:
            conn.close()

    admin(f'DROP DATABASE IF EXISTS "{name}"')
    admin(f'CREATE DATABASE "{name}"')
    url = f"postgresql://{parsed.username}:{parsed.password or ''}@{parsed.hostname}:{parsed.port or 5432}/{name}"
    env = {**os.environ, "DATABASE_URL_SYNC": url, "DATABASE_URL": url.replace("postgresql://", "postgresql+asyncpg://")}

    def alembic(*args):
        done = subprocess.run([sys.executable, "-m", "alembic", *args], cwd=BACKEND, env=env, capture_output=True,
                              text=True)
        assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]

    def shape():
        engine = sa.create_engine(url)
        try:
            columns = {c["name"]: c for c in sa.inspect(engine).get_columns("wsp_score_calls")}
            with engine.connect() as conn:
                head = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one()
        finally:
            engine.dispose()
        column = columns.get("arithmetic")
        described = None if column is None else (type(column["type"]).__name__, column["nullable"], column["default"])
        return sorted(columns), described, head

    try:
        alembic("upgrade", PARENT)
        before, _, _ = shape()
        alembic("upgrade", REVISION)
        added = shape()
        assert added == (sorted(before + ["arithmetic"]), ("JSONB", True, None), REVISION)
        alembic("downgrade", PARENT)
        assert shape() == (before, None, PARENT)
        alembic("upgrade", REVISION)
        assert shape() == added
    finally:
        admin(f'DROP DATABASE IF EXISTS "{name}"')
