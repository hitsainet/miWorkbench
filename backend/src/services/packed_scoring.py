"""The memory side of `probe_monitor_run._aggregate_rows_many`: one set of bucket buffers per call, and heads grouped
by their normalisation so each group standardises a bucket once (2026-11-22, the D128 scorer defect).

THE DEFECT THIS REMOVES (production, 2026-11-22 01:26–02:22 UTC). miForge scored three candidates of the D128 probe
bundle; every `POST /wsp/score` ran past 600 s. The held-out sets hold 8.57 M tokens at d_model 4096 (70 GB of
float16), and for every one of 281 length buckets the old loop allocated a fresh float32 bucket, cast each row with
numpy into a temporary and copied it in, then had EACH of the six weighted heads run `ProbeHead.token_scores`, which
allocates four bucket-sized temporaries (`x - mean`, `zeros_like`, `out / divisor`, `where`). Measured on the D128
widths, standardising was 63 % of the time and the matmul 4.5 % (review record
`0xcc/reviews/wsp_score_d128_2026-11-22.md`).

WHAT IS THE SAME, BIT FOR BIT (the invariant `_aggregate_rows_many`'s docstring states and
`test_aggregate_rows_many.py` pins with `np.array_equal`):
* the buckets (`plan_length_buckets`, `slots_for`), the mask, the zero padding and the bucket's shape;
* the element-wise operations, each one IEEE-rounded per element, so where a slice starts cannot change a result:
  float16 → float32 is exact, `x - mean` and `/ divisor` are single correctly-rounded operations, a degenerate
  channel is +0.0 (`masked_fill_(…, 0.0)` writes what `torch.where(…, zeros_like, …)` wrote);
* THE MATMUL: one `(n, width, d) @ (d,)` per head over the WHOLE bucket, on a contiguous tensor at storage offset 0
  of a fresh allocation, exactly the call the old code made. ⚠ NEVER SLICE OR STACK IT. MKL's gemv result for a row
  depends on how many rows the call holds and on the row's alignment (measured on the Zen 4 workstation: 242 of 300
  random sub-slices differed from the same rows scored inside the full matrix, and a 3-float misaligned buffer
  differed too). On the Zen workstation it did not depend on the thread count (1–16 threads identical); on MKL's
  Intel kernels it does, so the matmul now runs on one MKL thread (B5a, below) and the scorer may bound torch's
  threads (`wsp_score.scorer_threads`) without reaching a score.

WHAT CHANGED: the bucket buffers are allocated once per call and reused; a float16 capture is copied into a float16
bucket (half the bytes) and widened inside the subtraction; heads whose `mean`, `std` and `eps` are bitwise equal
share one standardised bucket (a WSP call's candidate, control, parent, LDA and diff-of-means share the parent's
statistics); the standardisation runs in cache-sized slices.

A head this module cannot vouch for (not a `ProbeHead`, a subclass overriding its arithmetic, a non-float32 tensor,
a d_model it does not share, `attention` without a query) is scored by its own `token_scores` /
`attention_logits` on the float32 bucket, exactly as before, so it raises exactly what it raised.

TWO THINGS A CALLER USED TO HAND THE MATMUL, AND NO LONGER DOES (B5a, 2026-11-24; review record
`0xcc/reviews/b5a_import_determinism_2026-11-24.md`). The WSP import re-scores a re-loaded candidate and requires the
scorer's bits (P23). CI refused a valid import with `persistence_mismatch` because the two passes handed MKL's gemv
different inputs that are not part of the function:
* THE VECTOR'S ADDRESS. On the MKL path this torch build takes on AMD Zen (`mkl_serv_cpu_detect` = 0, the `def`
  kernels: the workstation, the production node, AMD CI runners), a weight vector that is not 16-byte aligned changes
  rows' bits (measured at d 32: every offset that is not a multiple of 16 bytes). `safetensors` `load_file` places a
  re-loaded head at the file's data offset, 8-byte aligned, and that offset moves with the header's length (the
  bias's text is in it); the scorer's head is a numpy copy, always 16-byte aligned. So every head is scored from a
  fresh torch allocation of its vectors (`steady_head`: 64-byte aligned, c10's CPU alignment).
* THE THREAD COUNT. On MKL's AVX-512 and AVX2 kernels for Intel CPUs (Intel CI runners), gemv's per-row result
  depends on the thread count (measured: 1 of 330 rows at 2 threads against 1, 4 of 660). The scorer ran at its bound
  (`wsp_score.scorer_threads`, 2 on a 2-vCPU runner) and the import at the process's count (1 under xdist). So the
  matmul always runs on ONE MKL thread (`one_blas_thread`, MKL's thread-local count; torch's own threads are
  untouched). On the Zen `def` kernels gemv does not thread at all (17.9 ms per head per 512 MiB bucket at 1 to 16
  threads), so this costs nothing there and changes no bits: a score stored before this change reproduces.
What still depends on the machine: the kernel MKL picks for the CPU (an AMD and an Intel host differ), and torch's
own sum over a single row longer than 32,768 tokens (it splits that row across threads), which is why the import
scores under the scorer's thread bound too (`wsp_import.import_candidate`).
"""
from __future__ import annotations

import concurrent.futures
import contextlib
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: Tokens per standardisation slice: 256 x 4096 x 4 B = 4 MiB, so the subtraction's output is still in cache when the
#: division reads it. Element-wise results do not depend on the slice (see the module docstring).
STANDARDISE_SLICE_TOKENS = 256
#: Tokens per row piece a pack thread copies (1,024 x 4096 x 2 B = 8 MiB of a float16 capture).
PACK_PIECE_TOKENS = 1024

#: Capture dtypes a bucket holds as they are (anything else is cast to float32 by numpy, as the old loop did).
_SOURCE_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))


#: Below this many bytes of rows a call packs on its own thread (a training batch, a parity block): no pool.
PACK_POOL_MIN_BYTES = 64 * 1024 ** 2


@contextlib.contextmanager
def pack_pool(total_bytes: int) -> Iterator[Optional[concurrent.futures.ThreadPoolExecutor]]:
    """A thread pool for `BucketBuffers.pack`, as wide as torch's intra-op threads (which the WSP scorer bounds:
    `wsp_score.scorer_threads`), or None for a small call or a one-thread process (the Celery workers set
    `OMP_NUM_THREADS=1`)."""
    import torch

    workers = int(torch.get_num_threads())
    if workers <= 1 or total_bytes < PACK_POOL_MIN_BYTES:
        yield None
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pack") as pool:
        yield pool


#: MKL's thread-local thread count, from the MKL torch links statically (`libtorch_cpu.so` exports it): `int (int)`,
#: returns the previous local count (0 = none set, the global count applies). None when this torch has no MKL.
#: ⚠ The C entry point is `MKL_Set_Num_Threads_Local` (mkl_service.h maps `mkl_set_num_threads_local` to it); the
#: lowercase symbol in the library is the FORTRAN one, which takes a pointer and crashed the process when called with
#: an int (measured on torch 2.14.1+cpu).
MKL_LOCAL_THREADS_SYMBOL = "MKL_Set_Num_Threads_Local"
_MKL_LOCAL_THREADS: Any = None
_MKL_LOOKED_UP = False


def mkl_local_threads() -> Any:
    """`MKL_Set_Num_Threads_Local` from the MKL inside this torch, or None (a torch built without MKL)."""
    global _MKL_LOCAL_THREADS, _MKL_LOOKED_UP
    if not _MKL_LOOKED_UP:
        import ctypes
        import os

        import torch

        _MKL_LOOKED_UP = True
        if torch.backends.mkl.is_available():
            library = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so")
            try:
                function = getattr(ctypes.CDLL(library), MKL_LOCAL_THREADS_SYMBOL)
            except (OSError, AttributeError):
                function = None
            if function is not None:
                function.argtypes, function.restype = [ctypes.c_int], ctypes.c_int
            _MKL_LOCAL_THREADS = function
    return _MKL_LOCAL_THREADS


#: MKL's own answers to "which kernels does this process run" (B5a follow-up, 2026-11-24), from the MKL inside torch:
#: the code path MKL's dispatcher chose for this CPU (`int (void)`; with torch 2.14.1+cpu's MKL 2024.2, 0 is the
#: generic `def` kernels an AMD CPU gets, 7 the Intel AVX-512 ones, 5 Intel AVX2), and its conditional-reproducibility
#: setting (`int (int)`; 1 = off). ⚠ Neither documented query answers this here: `MKL_Get_Version_String` carries no
#: CPU (the same string on the `def` and the emulated Intel AVX-512 kernels), `mkl_cbwr_get_auto_branch` and
#: `MKL_Get_Version` are not exported by `libtorch_cpu.so`, and `torch.backends.cpu.get_cpu_capability()` is ATen's own
#: dispatch ("AVX512" on the Zen 4 workstation, where MKL takes `def`). These `mkl_serv_*` symbols are the
#: service-layer entry points that MKL's own kernels call; they are exported, take nothing and change nothing.
MKL_CODE_PATH_SYMBOL = "mkl_serv_cpu_detect"
MKL_CBWR_GET_SYMBOL = "mkl_serv_cbwr_get"
MKL_VERSION_SYMBOL = "MKL_Get_Version_String"
#: `mkl_cbwr_get(MKL_CBWR_ALL)`: the branch with the STRICT bit; `MKL_CBWR_BRANCH_OFF` is 1.
_MKL_CBWR_ALL, _MKL_CBWR_OFF = -1, 1
_MKL_FAMILY_FUNCTIONS: Any = None


def _mkl_family_functions() -> Any:
    """(code path, CNR setting, version string) from the MKL inside this torch, or () when any is missing."""
    global _MKL_FAMILY_FUNCTIONS
    if _MKL_FAMILY_FUNCTIONS is None:
        import ctypes
        import os

        import torch

        found: Any = ()
        if torch.backends.mkl.is_available():
            try:
                library = ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so"))
                code_path, cbwr, version = (getattr(library, name) for name in (
                    MKL_CODE_PATH_SYMBOL, MKL_CBWR_GET_SYMBOL, MKL_VERSION_SYMBOL))
            except (OSError, AttributeError):
                pass
            else:
                code_path.argtypes, code_path.restype = [], ctypes.c_int
                cbwr.argtypes, cbwr.restype = [ctypes.c_int], ctypes.c_int
                version.argtypes, version.restype = [ctypes.c_char_p, ctypes.c_int], None
                found = (code_path, cbwr, version)
        _MKL_FAMILY_FUNCTIONS = found
    return _MKL_FAMILY_FUNCTIONS


def mkl_kernel_family() -> Optional[str]:
    """The MKL kernel family this process's matmuls run on, as MKL itself answers it: its version, the code path its
    dispatcher chose, and its conditional-reproducibility (CNR) setting when one is on, e.g.
    `"MKL 2024.2 build 20240605, code path 0"`. None when this torch has no MKL or the entry points are missing:
    unknown is recorded as unknown, never guessed. Raw numbers, no names: they are compared for equality only, and a
    name table would itself be a guess on another MKL build."""
    import ctypes
    import re

    functions = _mkl_family_functions()
    if not functions:
        return None
    code_path, cbwr, version = functions
    buffer = ctypes.create_string_buffer(256)
    version(buffer, len(buffer))
    text = buffer.value.decode("ascii", "replace").strip()
    parsed = re.search(r"Version (\S+?)-Product Build (\d+)", text)
    family = f"MKL {parsed.group(1)} build {parsed.group(2)}" if parsed else (text or "MKL")
    family += f", code path {int(code_path())}"
    setting = int(cbwr(_MKL_CBWR_ALL))
    if setting != _MKL_CBWR_OFF:
        family += f", CNR {setting}"
    return family


@contextlib.contextmanager
def one_blas_thread() -> Iterator[None]:
    """The matmuls inside run on ONE BLAS thread, whatever the caller's thread count (module docstring, B5a).

    MKL's thread-local count: this thread only, so a concurrent request's matmuls and torch's own intra-op threads
    (the standardisation, the rules, the pack pool's width) keep theirs. torch's `set_num_threads` sets this same
    thread-local count, and the previous value (torch's) is put back after. ⚠ torch sets it LAZILY too: a thread's
    first parallel op (or `torch.get_num_threads()`) writes torch's count into it (measured, torch 2.14.1: a fresh
    thread's first op inside the block put 4 back where this had put 1). So the lazy set is forced first, here.
    Without MKL, torch's process-wide count is set to 1 for the block and put back."""
    import torch

    local = mkl_local_threads()
    if local is not None:
        torch.get_num_threads()   # torch's once-per-thread initialisation, which writes MKL's local count: before ours
        previous = int(local(1))
        try:
            yield
        finally:
            local(previous)
        return
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def _fresh(tensor: Any) -> Any:
    """A copy of `tensor` in its own allocation (offset 0: 64-byte aligned on CPU), or None."""
    import torch

    return None if tensor is None else tensor.detach().clone(memory_format=torch.contiguous_format)


def steady_head(head: Any) -> Any:
    """`head` with its matmul vectors (`weight`, `attention_query`) in fresh allocations, so where the caller's
    tensors happen to sit in memory cannot reach a score (module docstring, B5a). Anything but a `ProbeHead` (or a
    subclass: `dataclasses.replace` keeps the class) is returned as it is."""
    import dataclasses

    import torch

    from ..ml.probe_monitor_model import ProbeHead

    if not isinstance(head, ProbeHead) or not isinstance(head.weight, torch.Tensor):
        return head
    query = head.attention_query if isinstance(head.attention_query, torch.Tensor) else None
    return dataclasses.replace(head, weight=_fresh(head.weight),
                               attention_query=_fresh(query) if query is not None else head.attention_query)


def _cpu_f32(tensor: Any, d_model: int) -> bool:
    import torch

    return (isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32 and tensor.device.type == "cpu"
            and tensor.ndim == 1 and int(tensor.shape[0]) == d_model)


def vouched(head: Any, rule: str, d_model: int) -> bool:
    """True when `head` is a plain `ProbeHead` whose arithmetic this module reproduces bit for bit."""
    from ..ml.probe_monitor_model import ProbeHead

    if not isinstance(head, ProbeHead):
        return False
    kind = type(head)
    if any(getattr(kind, name) is not getattr(ProbeHead, name)
           for name in ("standardise", "token_scores", "attention_logits")):
        return False
    if not _cpu_f32(head.weight, d_model):
        return False
    for name in ("mean", "std"):
        value = getattr(head, name)
        if value is not None and not _cpu_f32(value, d_model):
            return False
    if rule == "attention" and not _cpu_f32(head.attention_query, d_model):
        return False
    return True


def _key(tensor: Any) -> Optional[bytes]:
    return None if tensor is None else tensor.detach().contiguous().numpy().tobytes()


@dataclass
class Group:
    """Heads that share one standardised bucket. `fast` False: the heads score themselves on the float32 bucket."""

    members: List[int] = field(default_factory=list)
    fast: bool = True
    mean: Any = None
    divisor: Any = None
    #: The channels a `std` at or below `eps` zeroes; None when no channel is degenerate (nothing to fill).
    degenerate: Any = None

    @property
    def identity(self) -> bool:
        return self.mean is None and self.divisor is None


def normalisation_groups(heads: Sequence[Any], rule: str, d_model: int) -> List[Group]:
    """Group the heads by (mean, std, eps) BITS, in first-seen order; heads not `vouched` go in one slow group."""
    import torch

    groups: List[Group] = []
    by_key = {}
    slow = Group(fast=False)
    for index, head in enumerate(heads):
        if not vouched(head, rule, d_model):
            slow.members.append(index)
            continue
        key = (_key(head.mean), _key(head.std), float(head.eps))
        group = by_key.get(key)
        if group is None:
            group = Group(mean=head.mean)
            if head.std is not None:
                # `ProbeHead.standardise`'s own two lines, once per group instead of once per head per bucket.
                degenerate = head.std.abs() <= head.eps
                group.divisor = torch.where(degenerate, torch.ones_like(head.std), head.std)
                group.degenerate = degenerate if bool(degenerate.any()) else None
            by_key[key] = group
            groups.append(group)
        group.members.append(index)
    if slow.members:
        groups.append(slow)
    return groups


class BucketBuffers:
    """The three buffers one `_aggregate_rows_many` call reuses for every bucket: the rows as captured (float16 for a
    float16 capture), the standardised float32 bucket, and (only when a group needs it) the float32 bucket itself.

    Each is allocated once at the largest bucket's size, and every bucket is a view at storage offset 0 of it: a fresh
    allocation from torch's CPU allocator, so the matmul sees the alignment it always saw (module docstring)."""

    #: Every buffer allocated, as (name, bytes): the allocation guard (`test_wsp_score_buffers.py`) reads it.
    allocations: List[Tuple[str, int]]

    def __init__(self, arrays: Sequence[np.ndarray], widths: Sequence[int], plan: Sequence[Sequence[int]],
                 d_model: int):
        import torch

        self.arrays, self.widths, self.d_model = arrays, widths, d_model
        self.capacity = max(len(ix) * max(1, max(widths[i] for i in ix)) for ix in plan)
        dtypes = {np.dtype(a.dtype) for a in arrays}
        self.source_np_dtype = next(iter(dtypes)) if len(dtypes) == 1 and dtypes <= set(_SOURCE_DTYPES) \
            else np.dtype(np.float32)
        torch_dtype = torch.float16 if self.source_np_dtype == np.float16 else torch.float32
        self.allocations = []
        self.source = self._alloc("source", torch_dtype)
        self._source_np = self.source.numpy()
        self.mask = torch.empty(self.capacity, dtype=torch.bool)
        self._standardised: Any = None
        self._float32: Any = None
        self._float32_of: Any = None

    def _alloc(self, name: str, dtype: Any) -> Any:
        import torch

        buffer = torch.empty(self.capacity * self.d_model, dtype=dtype)
        self.allocations.append((name, int(buffer.element_size() * buffer.numel())))
        return buffer

    def pack(self, indices: Sequence[int], pool: Any = None) -> Tuple[Any, Any]:
        """The bucket's rows, zero-padded to its width, and its mask: (n, width, d) in the capture dtype, (n, width).

        With `pool` (a thread pool), the rows are copied in pieces of `PACK_PIECE_TOKENS` on its threads: numpy releases
        the GIL for the copy, so a capture not yet in the page cache is read with several requests in flight instead of
        one page fault at a time (measured cold on the D128 widths: one thread read 270 MB/s). A copy is exact on any
        thread, so the bucket's bytes are the same either way."""
        n = len(indices)
        width = max(1, max(self.widths[i] for i in indices))
        count = n * width * self.d_model
        rows = self._source_np[:count].reshape(n, width, self.d_model)
        mask = self.mask[: n * width].view(n, width)
        mask.fill_(False)
        pieces = []
        for position, row_index in enumerate(indices):
            length = self.widths[row_index]
            if length:
                for start in range(0, length, PACK_PIECE_TOKENS):
                    pieces.append((position, row_index, start, min(length, start + PACK_PIECE_TOKENS)))
                mask[position, :length] = True
            if length < width:
                rows[position, length:] = 0

        def copy(piece: Tuple[int, int, int, int]) -> None:
            position, row_index, start, stop = piece
            rows[position, start:stop] = self.arrays[row_index][start:stop]   # one read of the row; exact for float16

        if pool is not None and len(pieces) > 1:
            for _ in pool.map(copy, pieces):
                pass
        else:
            for piece in pieces:
                copy(piece)
        self._float32_of = None
        return self.source[:count].view(n, width, self.d_model), mask

    def float32(self, bucket: Any) -> Any:
        """The bucket as float32 (the old `packed`): itself for a float32 capture, else widened once per bucket."""
        import torch

        if bucket.dtype == torch.float32:
            return bucket
        if self._float32_of is not bucket:
            if self._float32 is None:
                self._float32 = self._alloc("float32", torch.float32)
            view = self._float32[: bucket.numel()].view(bucket.shape)
            view.copy_(bucket)
            self._float32_of = bucket
        return self._float32[: bucket.numel()].view(bucket.shape)

    def standardised(self, group: Group, bucket: Any) -> Any:
        """`ProbeHead.standardise(<the float32 bucket>)` for every head of `group`, bit for bit, in one buffer."""
        import torch

        if not group.fast or group.identity:
            return self.float32(bucket)
        if self._standardised is None:
            self._standardised = self._alloc("standardised", torch.float32)
        tokens = bucket.shape[0] * bucket.shape[1]
        source = bucket.reshape(tokens, self.d_model)
        out = self._standardised[: tokens * self.d_model].view(tokens, self.d_model)
        for start in range(0, tokens, STANDARDISE_SLICE_TOKENS):
            stop = min(tokens, start + STANDARDISE_SLICE_TOKENS)
            piece, into = source[start:stop], out[start:stop]
            into.copy_(piece)   # float16 → float32, exact; in place, so no temporary (a mixed-dtype `sub` makes one)
            if group.mean is not None:
                into.sub_(group.mean)
            if group.divisor is not None:
                into.div_(group.divisor)
                if group.degenerate is not None:
                    into.masked_fill_(group.degenerate, 0.0)
        return out.view(bucket.shape)
