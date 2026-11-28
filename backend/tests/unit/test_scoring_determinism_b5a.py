"""B5a (2026-11-24): a head's score through `_aggregate_rows_many` does not depend on where its vectors sit in memory
or on the caller's thread count. The WSP import re-scores a re-loaded candidate and must reproduce the scorer's bits
(P23); private CI run 45467292953 refused a valid import (`persistence_mismatch`) because the two passes handed MKL's
gemv different addresses and thread counts. Review record: `0xcc/reviews/b5a_import_determinism_2026-11-24.md`.

The fixtures are the ones that measured a difference on the OLD code (`git show 118754639`):
* `ALIGN_FIXTURE` on the AMD Zen workstation (MKL's `def` kernels, which this host and the production node take): a
  weight 8 bytes off a 16-byte boundary moved 1 of 30 row scores;
* `THREAD_FIXTURE` under MKL's Intel AVX-512 kernels (emulated in a subprocess, below): 2 and 4 threads each moved 1
  of 60 row scores against 1 thread.

Controls (review record): drop `steady_head` from `_aggregate_rows_many` -> the address test red on this host; drop
`with one_blas_thread()` -> the AVX-512 subprocess test and the wiring tests red.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from src.ml.probe_monitor_model import ProbeHead
from src.services import packed_scoring
from src.services.probe_monitor_run import _aggregate_rows_many

BACKEND = Path(__file__).resolve().parents[2]

#: (d, rows, width): every row full width, seed d * 7 + rows (the trial harness's fixture).
ALIGN_FIXTURE = (32, 30, 11)
THREAD_FIXTURE = (32, 60, 11)


def _fixture(d: int, n: int, width: int):
    rng = np.random.default_rng(d * 7 + n)
    x = rng.standard_normal((n, width, d)).astype(np.float32)
    return [x[i] for i in range(n)], rng.standard_normal(d).astype(np.float32)


def _at_offset(values: np.ndarray, offset_bytes: int) -> torch.Tensor:
    """`values` in a torch tensor that starts `offset_bytes` past a 64-byte boundary."""
    buffer = torch.empty(values.shape[0] + 16, dtype=torch.float32)
    assert buffer.data_ptr() % 64 == 0
    view = buffer[offset_bytes // 4: offset_bytes // 4 + values.shape[0]]
    view.copy_(torch.from_numpy(values))
    assert view.data_ptr() % 64 == offset_bytes
    return view


def _scores(head, rule, rows):
    return np.asarray(_aggregate_rows_many([head], rule, {}, rows)[0], dtype=np.float32)


def _mkl_max_threads() -> int:
    library = ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so"))
    return int(library.MKL_Get_Max_Threads())


# ── the address ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rule", ["mean", "attention"])
def test_a_weights_address_does_not_reach_a_score(rule):
    rows, weight = _fixture(*ALIGN_FIXTURE)
    query = weight[::-1].copy()
    aligned = ProbeHead(weight=_at_offset(weight, 0), bias=0.25, attention_query=_at_offset(query, 0))
    expected = _scores(aligned, rule, rows)
    for offset in range(4, 64, 4):   # safetensors' `load_file` places a re-loaded head 8-byte aligned, anywhere
        moved = ProbeHead(weight=_at_offset(weight, offset), bias=0.25, attention_query=_at_offset(query, offset))
        got = _scores(moved, rule, rows)
        assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), (
            f"a weight {offset} bytes past a 64-byte boundary scored {int((got != expected).sum())} rows differently")


def test_steady_head_copies_the_matmul_vectors_into_fresh_allocations():
    weight, query = _at_offset(np.arange(32, dtype=np.float32), 8), _at_offset(np.ones(32, dtype=np.float32), 56)
    mean, std = torch.zeros(32), torch.ones(32)
    head = ProbeHead(weight=weight, bias=0.5, mean=mean, std=std, attention_query=query, layer=7)
    steady = packed_scoring.steady_head(head)
    assert type(steady) is ProbeHead and steady is not head
    assert steady.weight.data_ptr() % 64 == 0 and steady.attention_query.data_ptr() % 64 == 0
    assert torch.equal(steady.weight, weight) and torch.equal(steady.attention_query, query)
    assert (steady.bias, steady.layer, steady.eps) == (0.5, 7, head.eps)
    assert steady.mean is mean and steady.std is std   # the normalisation groups key on these bits, untouched
    assert packed_scoring.steady_head("not a head") == "not a head"


# ── the thread count ──────────────────────────────────────────────────────────


_LOCAL_CHILD = textwrap.dedent("""
    import ctypes, json, os, sys, threading
    import torch
    sys.path.insert(0, sys.argv[1])
    from src.services import packed_scoring
    mkl = ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so"))
    def here():
        return int(mkl.MKL_Get_Max_Threads())
    seen = {}
    def elsewhere(key):
        worker = threading.Thread(target=lambda: seen.setdefault(key, here()))
        worker.start()
        worker.join()
    torch.set_num_threads(4)
    seen["torch_set"] = here()
    elsewhere("other_before")
    with packed_scoring.one_blas_thread():
        seen["inside"] = here()
        torch.randn(64, 64) @ torch.randn(64)
        torch.randn(100000).sum()
        seen["inside_after_torch_ops"] = here()
        elsewhere("other_inside")
    seen["after"] = here()
    seen["torch_after"] = torch.get_num_threads()
    seen["found"] = packed_scoring.mkl_local_threads() is not None
    print(json.dumps(seen))
""")


def test_one_blas_thread_is_mkls_thread_local_count(tmp_path):
    """torch's `set_num_threads` sets MKL's THREAD-LOCAL count (ATen's OpenMP backend calls
    `mkl_set_num_threads_local`); `one_blas_thread` overrides it with 1 for the block, leaves every other thread's
    count alone, and puts torch's back. In a child whose PROCESS-WIDE MKL count is 3 (MKL_NUM_THREADS), so that "no
    local count" cannot pass for 1 as it would under xdist's MKL_NUM_THREADS=1, and whose thread runs its FIRST
    parallel torch ops inside the block (torch's lazy per-thread initialisation writes MKL's local count)."""
    assert torch.backends.mkl.is_available(), "the pinned torch build (requirements.txt) links MKL"
    (tmp_path / "child.py").write_text(_LOCAL_CHILD)
    env = {**os.environ, "MKL_NUM_THREADS": "3", "PYTHONWARNINGS": "ignore"}
    env.pop("OMP_NUM_THREADS", None)
    done = subprocess.run([sys.executable, str(tmp_path / "child.py"), str(BACKEND)], env=env, capture_output=True,
                          text=True, timeout=300, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-2000:]
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen["found"], (f"{packed_scoring.MKL_LOCAL_THREADS_SYMBOL} not found in libtorch_cpu.so: one_blas_thread "
                           "would fall back to torch's process-wide thread count")
    assert seen["torch_set"] == 4 and seen["other_before"] == 3, f"fixture: torch's count is thread-local: {seen}"
    assert seen["inside"] == 1 and seen["inside_after_torch_ops"] == 1, seen
    assert seen["after"] == 4 and seen["torch_after"] == 4, f"torch's count is put back: {seen}"
    assert seen["other_inside"] == 3, f"the count is this thread's only: {seen}"


class _Recording(ProbeHead):
    """A head `packed_scoring` cannot vouch for (it overrides `token_scores`), so it scores itself: it records the MKL
    thread count and its weight's address at the moment of its matmul."""

    def token_scores(self, activations):   # noqa: D401
        _Recording.seen.append((_mkl_max_threads(), self.weight.data_ptr() % 64))
        return super().token_scores(activations)


_Recording.seen = []


def test_every_matmul_runs_on_one_mkl_thread_from_a_steady_head():
    rows, weight = _fixture(*ALIGN_FIXTURE)
    rows = rows + [np.ones((3, 32), dtype=np.float32)] * 5   # two widths: more than one bucket is not required
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    _Recording.seen = []
    try:
        head = _Recording(weight=_at_offset(weight, 8))
        _aggregate_rows_many([head, head], "mean", {}, rows)
    finally:
        torch.set_num_threads(before)
    assert _Recording.seen, "the recording head was never scored"
    assert all(threads == 1 for threads, _ in _Recording.seen), _Recording.seen
    assert all(address == 0 for _, address in _Recording.seen), _Recording.seen


def test_one_blas_thread_wraps_each_heads_matmul_once_per_bucket(monkeypatch):
    """The wiring, by count: the context is entered once per head per bucket, and every vouched head's matmul runs
    inside it (the fast path, which `_Recording` cannot observe)."""
    import contextlib

    from src.services.probe_monitor_trainer import plan_length_buckets, slots_for

    entered = []
    real = packed_scoring.one_blas_thread

    @contextlib.contextmanager
    def counting():
        with real():
            entered.append(_mkl_max_threads())
            yield

    monkeypatch.setattr(packed_scoring, "one_blas_thread", counting)
    rows, weight = _fixture(*ALIGN_FIXTURE)
    rows = rows + [np.ones((3, 32), dtype=np.float32)] * 5
    heads = [ProbeHead(weight=torch.from_numpy(weight)), ProbeHead(weight=torch.from_numpy(weight[::-1].copy()))]
    buckets = len(plan_length_buckets([r.shape[0] for r in rows], budget_slots=slots_for(32)))
    _aggregate_rows_many(heads, "mean", {}, rows)
    assert entered == [1] * (len(heads) * buckets)


# ── the defect itself, on MKL's Intel AVX-512 kernels ─────────────────────────

#: Makes the MKL inside torch take its Intel AVX-512 kernels on any AVX-512 CPU (the AMD workstation included), by
#: answering MKL's own CPU checks. `libtorch_cpu.so` exports them and calls them through its PLT, so LD_PRELOAD
#: interposes. Code path numbers in torch 2.14.1+cpu's MKL 2024.2: 0 `def`, 5 AVX2, 7 AVX-512.
_SHIM_C = r"""
int mkl_serv_cpu_detect(void) { return 7; }
int mkl_serv_intel_cpu(void) { return 1; }
int mkl_serv_intel_cpu_true(void) { return 1; }
int mkl_serv_cpuisskl(void) { return 1; }
int mkl_serv_cpuiszen(void) { return 0; }
"""

_CHILD = textwrap.dedent("""
    import json, sys
    import numpy as np, torch
    sys.path.insert(0, sys.argv[1])
    from src.ml.probe_monitor_model import ProbeHead
    from src.services.probe_monitor_run import _aggregate_rows_many
    d, n, width = (int(v) for v in sys.argv[2:5])
    rng = np.random.default_rng(d * 7 + n)
    x = rng.standard_normal((n, width, d)).astype(np.float32)
    rows, w = [x[i] for i in range(n)], rng.standard_normal(d).astype(np.float32)
    raw, ours = {}, {}
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        raw[threads] = (torch.from_numpy(x).clone() @ torch.from_numpy(w).clone()).numpy().tobytes().hex()
        ours[threads] = np.asarray(_aggregate_rows_many([ProbeHead(weight=torch.from_numpy(w))], "mean", {}, rows)[0],
                                   dtype=np.float32).tobytes().hex()
    print(json.dumps({"raw_thread_dependent": len(set(raw.values())) > 1, "ours_distinct": len(set(ours.values()))}))
""")


def _avx512_skx() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return all(f" {flag}" in flags for flag in ("avx512f", "avx512dq", "avx512bw", "avx512vl"))


def test_on_mkls_avx512_kernels_the_thread_count_does_not_reach_a_score(tmp_path):
    compiler = shutil.which("cc") or shutil.which("gcc")
    torch_cpu = Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"
    if platform.system() != "Linux" or platform.machine() != "x86_64" or compiler is None:
        pytest.skip("SKIPPED LOUDLY: the AVX-512 emulation needs Linux x86_64 and a C compiler")
    if not _avx512_skx():
        pytest.skip("SKIPPED LOUDLY: this CPU has no AVX-512 (F, DQ, BW, VL), so MKL's AVX-512 kernels cannot run here; "
                    "the thread-count property is proven on an AVX-512 host")
    if not torch_cpu.exists() or not torch.backends.mkl.is_available():
        pytest.skip("SKIPPED LOUDLY: this torch has no statically linked MKL to emulate")
    (tmp_path / "shim.c").write_text(_SHIM_C)
    subprocess.run([compiler, "-shared", "-fPIC", "-O2", "-o", str(tmp_path / "shim.so"), str(tmp_path / "shim.c")],
                   check=True, capture_output=True)
    (tmp_path / "child.py").write_text(_CHILD)
    env = {**os.environ, "LD_PRELOAD": str(tmp_path / "shim.so"), "PYTHONWARNINGS": "ignore"}
    env.pop("OMP_NUM_THREADS", None)
    env.pop("MKL_NUM_THREADS", None)
    done = subprocess.run([sys.executable, str(tmp_path / "child.py"), str(BACKEND), *map(str, THREAD_FIXTURE)],
                          env=env, capture_output=True, text=True, timeout=300, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["raw_thread_dependent"], (
        "fixture: MKL's AVX-512 gemv no longer changes these rows with the thread count, so this test proves nothing")
    assert result["ours_distinct"] == 1, "a head's scores changed with the thread count on MKL's AVX-512 kernels"
