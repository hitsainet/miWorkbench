"""Apply a `miforge.wsp-patch/v1` patch to a probe head, exactly as miForge writes it (ENH-010 044 task 8.1; FR-8).

miForge OWNS the patch format (`docs/schemas/miforge-wsp-patch-v1.json`, vendored byte-identical; 016 FTDD §3.2).
miWorkbench never imports miForge's code or a WSP library (FR-10): this module re-implements the published rules, and
`test_wsp_patch_golden_cross_repo.py` applies miForge's own golden patches to prove the two agree bit for bit.

The rules (wsplab's array layout, by the tensor's position `i` in `tensors[]`; R1B-X4):

* **A tensor is its raw bits.** numpy has no bfloat16, so every tensor is held as the unsigned-integer view of its
  bits (`uint16`/`uint32`/`uint64`); hashing, patching and comparison are bit-exact by construction.
* **`tensor_sha256`** is sha256 over the little-endian, C-contiguous bytes in the stored dtype (016's definition).
* **sparse:** `t<i>_idx` (int64 flat indices, strictly increasing) and `t<i>_val` (the new values' raw BITS):
  `e = base_bits.reshape(-1); e[idx] = val` — a bit view, never a float assignment.
* **lowrank:** `t<i>_a` (rows × r) and `t<i>_b` (r × cols), float64; `w = float64(base)`, then ONE RANK-ONE TERM AT A
  TIME `w = w + outer(a[:, k], b[k])`; cast to float32 (float64 for a float64 tensor), then to the tensor's dtype;
  then the `t<i>_idx`/`t<i>_val` fix-ups as for sparse. A fused `base + a @ b` rounds differently and is wrong.
* **dense:** `t<i>_bits` is the full edited tensor's bits — a replacement, not a delta.

**Order of checks** (each refuses before anything later runs; nothing is written by this module):
the manifest's own digests (`content_sha256`, `patch_sha256`, RFC 8785), the archive's digest (`arrays_sha256`,
checked FIRST over the raw bytes) and its size bound, every base tensor's hash, dtype and shape, then the rebuild,
then every edited hash and changed-entry count. The archive is read with `allow_pickle=False`, member names and
sizes bounded before any member is read.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import zipfile
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

#: The dtypes a WSP tensor may have (016 FTDD §3.1 `TensorEntry.dtype`), and the unsigned type of their bits.
BITS: dict[str, Any] = {"float16": np.uint16, "bfloat16": np.uint16, "float32": np.uint32, "float64": np.uint64}
_FLOAT: dict[str, Any] = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
#: safetensors dtype names (the parent head is a safetensors file).
_ST_DTYPE = {"F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64"}

PATCH_KIND = "miforge.wsp-patch/v1"
ARRAYS_FILE = "edit.npz"
NPY_HEADER_ALLOWANCE = 1024
_MEMBER = re.compile(r"^t(\d{1,6})_(idx|val|a|b|bits)\.npy$")


class WspPatchRefused(ValueError):
    """A patch miWorkbench will not apply: `code` is the refusal code the import route answers with."""

    def __init__(self, code: str, message: str, **details: Any):
        self.code = code
        self.details = details
        super().__init__(message)


def _malformed(message: str, **details: Any) -> WspPatchRefused:
    return WspPatchRefused("patch_malformed", message, **details)


@dataclass(frozen=True)
class WspTensor:
    """`(dtype name, raw bits)`; `bits` is C-contiguous, in the tensor's shape."""

    dtype: str
    bits: np.ndarray

    def __post_init__(self) -> None:
        if self.dtype not in BITS:
            raise _malformed(f"dtype {self.dtype!r} is not one of {sorted(BITS)}")
        if self.bits.dtype != np.dtype(BITS[self.dtype]):
            raise _malformed(f"a {self.dtype} tensor's bits must be {np.dtype(BITS[self.dtype]).name}")
        if not self.bits.flags.c_contiguous:
            object.__setattr__(self, "bits", np.ascontiguousarray(self.bits))

    @property
    def shape(self) -> list[int]:
        return [int(n) for n in self.bits.shape]

    @property
    def sha256(self) -> str:
        return tensor_sha256(self.bits)

    def values64(self) -> np.ndarray:
        """The values as float64 (exact for every dtype here)."""
        if self.dtype == "bfloat16":
            return (self.bits.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
        return self.bits.view(_FLOAT[self.dtype]).astype(np.float64)


def from_array(arr: np.ndarray) -> WspTensor:
    """Wrap a float16/32/64 array as its bits."""
    a = np.ascontiguousarray(arr)
    name = {np.dtype(np.float16): "float16", np.dtype(np.float32): "float32", np.dtype(np.float64): "float64"}.get(
        a.dtype.newbyteorder("="))
    if name is None:
        raise _malformed(f"cannot wrap a {a.dtype} array as a WSP tensor")
    return WspTensor(name, a.astype(a.dtype.newbyteorder("=")).view(BITS[name]).copy())


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    """float32 → bfloat16 bits, round to nearest even; NaN stays a quiet NaN."""
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounding = ((u >> np.uint64(16)) & np.uint64(1)) + np.uint64(0x7FFF)
    out = ((u + rounding) >> np.uint64(16)).astype(np.uint16)
    nan = np.isnan(np.asarray(values, dtype=np.float32))
    if nan.any():
        out = np.where(nan, ((u >> np.uint64(16)) | np.uint64(0x40)).astype(np.uint16), out)
    return np.ascontiguousarray(out.reshape(np.shape(values)))


def from_values(dtype: str, values: np.ndarray) -> WspTensor:
    """Cast float values to `dtype` (round to nearest even) and wrap the bits."""
    v = np.asarray(values)
    if dtype == "bfloat16":
        return WspTensor("bfloat16", _bf16_bits(v.astype(np.float32)))
    return WspTensor(dtype, np.ascontiguousarray(v.astype(_FLOAT[dtype])).view(BITS[dtype]).copy())


def tensor_sha256(arr: np.ndarray) -> str:
    """sha256 of the array's raw bytes: little-endian, C-contiguous, in its stored dtype (016's definition)."""
    a = np.ascontiguousarray(arr)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    return hashlib.sha256(a.reshape(-1).view(np.uint8).data).hexdigest()


def jcs_sha256(document: Any) -> str:
    """sha256 of the RFC 8785 canonical bytes (never `json.dumps(sort_keys=True)`: it writes 1e-05)."""
    import rfc8785

    return hashlib.sha256(bytes(rfc8785.dumps(document))).hexdigest()


def content_sha256(manifest: dict) -> str:
    """RFC 8785 over `{bundle_base_sha256, parent, tensors}` only (016 R1B-A1)."""
    return jcs_sha256({k: manifest[k] for k in ("bundle_base_sha256", "parent", "tensors")})


def patch_sha256(manifest: dict) -> str:
    """RFC 8785 over the manifest without `patch_sha256`."""
    return jcs_sha256({k: v for k, v in manifest.items() if k != "patch_sha256"})


def read_safetensors(data: bytes) -> dict[str, WspTensor]:
    """A safetensors file's tensors as bits (bf16 included, which numpy's own loader refuses)."""
    if len(data) < 8:
        raise _malformed("not a safetensors file")
    (length,) = struct.unpack("<Q", data[:8])
    if length > len(data) - 8:
        raise _malformed("safetensors header runs past the file")
    header = json.loads(data[8:8 + length].decode("utf-8"))
    body = data[8 + length:]
    out: dict[str, WspTensor] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype = _ST_DTYPE.get(entry["dtype"])
        if dtype is None:
            raise _malformed(f"tensor {name!r} has dtype {entry['dtype']}, not a WSP dtype")
        start, end = entry["data_offsets"]
        bits = np.frombuffer(body[start:end], dtype=np.dtype(BITS[dtype]).newbyteorder("<"))
        out[name] = WspTensor(dtype, bits.astype(np.dtype(BITS[dtype])).reshape(entry["shape"]).copy())
    return out


def validate_manifest(manifest: Any) -> dict:
    """Schema-valid against the VENDORED `miforge-wsp-patch-v1.json`, its two digests recomputed."""
    from jsonschema.exceptions import best_match
    from jsonschema.validators import validator_for

    schema = _patch_schema()
    error = best_match(validator_for(schema)(schema).iter_errors(manifest))
    if error is not None:
        path = "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in error.absolute_path)
        raise _malformed(f"patch.json is invalid at {path}: {error.message}", path=path)
    tensors = manifest["tensors"]
    if [t["index"] for t in tensors] != list(range(len(tensors))):
        raise _malformed("tensors[].index must be 0, 1, 2, ... in order")
    if len({t["name"] for t in tensors}) != len(tensors):
        raise _malformed("tensors[] names a tensor twice")
    if manifest["content_sha256"] != content_sha256(manifest):
        raise WspPatchRefused("patch_hash_mismatch", "content_sha256 does not match the manifest")
    if manifest["patch_sha256"] != patch_sha256(manifest):
        raise WspPatchRefused("patch_hash_mismatch", "patch_sha256 does not match the manifest")
    if manifest["changed_entries_total"] != sum(t["changed_entries"] for t in tensors):
        raise _malformed("changed_entries_total is not the sum of tensors[].changed_entries")
    return manifest


def _patch_schema() -> dict:
    from importlib import resources

    return json.loads(resources.files("src.schemas.vendored").joinpath("miforge-wsp-patch-v1.json").read_text("utf-8"))


def archive_bound(manifest: dict) -> int:
    """The most uncompressed bytes the archive can need for its tensors."""
    total = 0
    for pt in manifest["tensors"]:
        width = np.dtype(BITS[pt["dtype"]]).itemsize
        numel = int(np.prod(pt["edited_shape"], dtype=np.int64)) if pt["edited_shape"] else 1
        if pt["form"] == "dense":
            total += numel * width + NPY_HEADER_ALLOWANCE
        else:
            total += numel * (8 + width) + 2 * NPY_HEADER_ALLOWANCE
            if pt["form"] == "lowrank":
                rank = int(pt["rank"] or 0)
                rows, cols = (pt["shape"][0], pt["shape"][1]) if len(pt["shape"]) == 2 else (numel, 1)
                total += (rows * rank + rank * cols) * 8 + 2 * NPY_HEADER_ALLOWANCE
    return total


def read_archive(data: bytes, manifest: dict) -> dict[str, np.ndarray]:
    """The archive's arrays, names and sizes checked before any member is read; never unpickles."""
    expected: set[str] = set()
    for i, pt in enumerate(manifest["tensors"]):
        if pt["form"] == "sparse":
            expected |= {f"t{i}_idx", f"t{i}_val"}
        elif pt["form"] == "lowrank":
            expected |= {f"t{i}_idx", f"t{i}_val", f"t{i}_a", f"t{i}_b"}
        else:
            expected.add(f"t{i}_bits")
    bound = archive_bound(manifest)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise _malformed(f"{ARRAYS_FILE} is not a zip archive: {exc}") from exc
    with archive:
        infos = archive.infolist()
        names = []
        for info in infos:
            if _MEMBER.match(info.filename) is None or info.is_dir():
                raise _malformed(f"{ARRAYS_FILE} holds an unexpected member {info.filename!r}")
            names.append(info.filename[:-4])
        if len(names) != len(set(names)) or set(names) != expected:
            raise _malformed(f"{ARRAYS_FILE} members {sorted(names)} are not the ones the manifest needs")
        if sum(info.file_size for info in infos) > bound:
            raise _malformed(f"{ARRAYS_FILE} declares more uncompressed bytes than its tensors can need ({bound})")
        out: dict[str, np.ndarray] = {}
        remaining = bound
        for info in infos:
            with archive.open(info) as member:
                raw = member.read(remaining + 1)
            if len(raw) > remaining or len(raw) != info.file_size:
                raise _malformed(f"{ARRAYS_FILE} member {info.filename} expands beyond its declared size")
            remaining -= len(raw)
            try:
                out[info.filename[:-4]] = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
            except ValueError as exc:
                raise _malformed(f"{info.filename} is not a plain numpy array: {exc}") from exc
        return out


def lowrank_reconstruct(base: WspTensor, a: np.ndarray, b: np.ndarray) -> WspTensor:
    """`float64(base) + Σ_k outer(a[:, k], b[k])`, ONE TERM AT A TIME, cast to its dtype."""
    w = base.values64().copy()
    for k in range(a.shape[1]):
        w = w + np.outer(a[:, k], b[k])
    if base.dtype == "float64":
        return from_values("float64", w)
    return from_values(base.dtype, w.astype(np.float32))


def _rebuild(pt: dict, i: int, base: WspTensor, arrays: dict[str, np.ndarray]) -> WspTensor:
    uint = np.dtype(BITS[pt["dtype"]])
    numel = int(base.bits.size)
    if pt["edited_shape"] != pt["shape"] or pt.get("append") is not None:
        raise _malformed(f"tensor {pt['name']!r} changes shape or appends; refused for a probe head")

    def fixups(bits: np.ndarray) -> np.ndarray:
        idx, val = arrays[f"t{i}_idx"], arrays[f"t{i}_val"]
        if idx.dtype != np.dtype(np.int64) or idx.ndim != 1 or val.ndim != 1 or idx.shape != val.shape:
            raise _malformed(f"t{i}_idx/t{i}_val must be equal-length 1-D int64 and {uint.name} arrays")
        if val.dtype != uint:
            raise _malformed(f"t{i}_val is {val.dtype}; a {pt['dtype']} tensor's values are {uint.name} bits")
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= numel):
            raise _malformed(f"t{i}_idx holds an index outside 0..{numel - 1}")
        if idx.size > 1 and not bool(np.all(np.diff(idx) > 0)):
            raise _malformed(f"t{i}_idx is not strictly increasing")
        flat = np.array(bits, copy=True).reshape(-1)
        flat[idx] = val  # bits into bits: the values are never read as floats
        return flat

    if pt["form"] == "dense":
        bits = arrays[f"t{i}_bits"]
        if bits.dtype != uint or list(bits.shape) != pt["edited_shape"]:
            raise _malformed(f"t{i}_bits must be {uint.name} with shape {pt['edited_shape']}")
        flat = bits.reshape(-1)
    elif pt["form"] == "sparse":
        flat = fixups(base.bits)
    else:
        a, b = arrays[f"t{i}_a"], arrays[f"t{i}_b"]
        rank = pt["rank"]
        if (len(pt["shape"]) != 2 or a.dtype != np.float64 or b.dtype != np.float64
                or a.shape != (pt["shape"][0], rank) or b.shape != (rank, pt["shape"][1])):
            raise _malformed(f"t{i}_a/t{i}_b must be float64 factors of rank {rank} for shape {pt['shape']}")
        flat = fixups(lowrank_reconstruct(base, a, b).bits)
    return WspTensor(pt["dtype"], np.ascontiguousarray(flat).reshape(pt["edited_shape"]).astype(uint))


def apply_patch(manifest: Any, arrays_bytes: bytes,
                base: dict[str, WspTensor]) -> tuple[dict, dict[str, WspTensor]]:
    """`(manifest, tensors)`: `base` with the patch applied, in memory. Refuses before rebuilding anything:
    `patch_malformed`, `patch_hash_mismatch` (a digest), `patch_base_mismatch` (a base tensor's hash, dtype or shape)."""
    manifest = validate_manifest(manifest)
    if hashlib.sha256(arrays_bytes).hexdigest() != manifest["arrays_sha256"]:
        raise WspPatchRefused("patch_hash_mismatch", f"{ARRAYS_FILE} does not hash to arrays_sha256")
    for pt in manifest["tensors"]:
        held = base.get(pt["name"])
        if held is None or held.dtype != pt["dtype"] or held.shape != pt["shape"] or held.sha256 != pt["base_sha256"]:
            raise WspPatchRefused(
                "patch_base_mismatch",
                f"the patch was made against a different {pt['name']!r}: its base hash, dtype or shape does not "
                "match this probe's head", tensor=pt["name"])
    arrays = read_archive(arrays_bytes, manifest)
    rebuilt: dict[str, WspTensor] = {}
    for i, pt in enumerate(manifest["tensors"]):
        t = _rebuild(pt, i, base[pt["name"]], arrays)
        if t.sha256 != pt["edited_sha256"]:
            raise WspPatchRefused("patch_hash_mismatch", f"tensor {pt['name']!r} does not rebuild to its edited_sha256",
                                  tensor=pt["name"])
        changed = int(np.count_nonzero(base[pt["name"]].bits.reshape(-1) != t.bits.reshape(-1)))
        if changed != pt["changed_entries"]:
            raise _malformed(f"tensor {pt['name']!r} changes {changed} entries, not {pt['changed_entries']}")
        rebuilt[pt["name"]] = t
    return manifest, {**base, **rebuilt}


def check_limits(manifest: dict, *, editable: Iterable[str], max_changed_entries: Optional[int]) -> None:
    """Refuses `patch_outside_limits` when the patch changes a tensor that is not editable (a probe's `mean` and `std`
    never are, FTASKS 5.2) or more entries than the limit (`None` = every editable entry)."""
    allowed = set(editable)
    outside = sorted(t["name"] for t in manifest["tensors"] if t["name"] not in allowed)
    if outside:
        raise WspPatchRefused("patch_outside_limits", f"the patch changes tensors that are not editable: {outside}",
                              tensors=outside)
    total = int(manifest["changed_entries_total"])
    if max_changed_entries is not None and total > int(max_changed_entries):
        raise WspPatchRefused("patch_outside_limits",
                              f"the patch changes {total} entries; the limit is {max_changed_entries}",
                              changed_entries_total=total, limit=int(max_changed_entries))
