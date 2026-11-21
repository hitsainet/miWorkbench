"""ENH-010 044 task 8.1: `services/wsp_patch.py` applies each patch form exactly (R1B-X4), against hand-built expected
tensors, and refuses a pickled archive, a digest mismatch, a different base and an out-of-limits patch.

The cross-repo twin, `test_wsp_patch_golden_cross_repo.py`, applies miForge's own golden patches."""
from __future__ import annotations

import copy
import io
import zipfile

import numpy as np
import pytest

from src.services import wsp_patch as wp

H = "b" * 64


def _npz(arrays: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            body = io.BytesIO()
            np.lib.format.write_array(body, np.ascontiguousarray(arrays[name]), version=(1, 0), allow_pickle=False)
            archive.writestr(f"{name}.npy", body.getvalue())
    return buffer.getvalue()


def _patch(name: str, base: wp.WspTensor, edited: wp.WspTensor, form: str, arrays: dict, rank=None):
    """A schema-valid manifest for one tensor, its digests computed; returns (manifest, npz bytes)."""
    data = _npz(arrays)
    delta = edited.values64() - base.values64()
    entry = {"index": 0, "name": name, "dtype": base.dtype, "shape": base.shape, "edited_shape": edited.shape,
             "form": form, "rank": rank, "append": None, "base_sha256": base.sha256, "edited_sha256": edited.sha256,
             "changed_entries": int(np.count_nonzero(base.bits.reshape(-1) != edited.bits.reshape(-1))),
             "l2_delta": float(np.sqrt((delta ** 2).sum())), "max_abs_delta": float(np.abs(delta).max(initial=0.0))}
    manifest = {"kind": "miforge.wsp-patch/v1", "format_version": 1, "bundle_base_sha256": H,
                "parent": {"kind": "bundle", "ref": H, "patch_sha256": None}, "parent_patch_sha256": None,
                "arrays_file": "edit.npz", "arrays_sha256": wp.hashlib.sha256(data).hexdigest(), "tensors": [entry],
                "changed_entries_total": entry["changed_entries"],
                "created_by": {"provider": "weightspace", "provider_version": "0.3.0", "run_id": "wsp_0123456789ab"},
                "created_at": "2026-11-16T05:00:00Z"}
    manifest["content_sha256"] = wp.content_sha256(manifest)
    manifest["patch_sha256"] = wp.patch_sha256(manifest)
    return manifest, data


@pytest.fixture
def base():
    rng = np.random.default_rng(7)
    return wp.from_array(rng.standard_normal((6, 5)).astype(np.float32))


def test_sparse_assigns_bits_not_floats(base):
    # The new value at index 3 is the smallest subnormal (bits 0x00000001): read as a number it is 1, as bits 1e-45.
    idx = np.array([3, 11], dtype=np.int64)
    val = np.array([0x00000001, np.float32(2.5).view(np.uint32)], dtype=np.uint32)
    expected_bits = base.bits.reshape(-1).copy()
    expected_bits[idx] = val
    expected = wp.WspTensor("float32", expected_bits.reshape(6, 5))
    manifest, data = _patch("w", base, expected, "sparse", {"t0_idx": idx, "t0_val": val})
    _, out = wp.apply_patch(manifest, data, {"w": base})
    assert np.array_equal(out["w"].bits, expected.bits) and out["w"].sha256 == expected.sha256


def test_lowrank_is_one_rank_one_term_at_a_time_then_fixups(base):
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((6, 2)), rng.standard_normal((2, 5))
    w = base.values64().copy()
    for k in range(2):
        w = w + np.outer(a[:, k], b[k])
    recon = w.astype(np.float32)
    recon_bits = recon.view(np.uint32).reshape(-1).copy()
    fix_idx = np.array([0], dtype=np.int64)
    fix_val = np.array([np.float32(9.0).view(np.uint32)], dtype=np.uint32)
    recon_bits[fix_idx] = fix_val
    expected = wp.WspTensor("float32", recon_bits.reshape(6, 5))
    manifest, data = _patch("w", base, expected, "lowrank", {"t0_a": a, "t0_b": b, "t0_idx": fix_idx,
                                                             "t0_val": fix_val}, rank=2)
    _, out = wp.apply_patch(manifest, data, {"w": base})
    assert out["w"].sha256 == expected.sha256


def test_dense_is_a_full_replacement(base):
    edited = wp.from_array(np.full((6, 5), 0.25, dtype=np.float32))
    manifest, data = _patch("w", base, edited, "dense", {"t0_bits": edited.bits})
    _, out = wp.apply_patch(manifest, data, {"w": base})
    assert out["w"].sha256 == edited.sha256


def test_bfloat16_dense_round_trips_as_bits():
    base = wp.WspTensor("bfloat16", np.arange(8, dtype=np.uint16) + np.uint16(0x3F80))
    edited = wp.from_values("bfloat16", base.values64() * 2)
    manifest, data = _patch("w", base, edited, "dense", {"t0_bits": edited.bits})
    _, out = wp.apply_patch(manifest, data, {"w": base})
    assert out["w"].dtype == "bfloat16" and out["w"].sha256 == edited.sha256


def test_a_pickled_npz_is_refused(base):
    edited = wp.from_array(np.zeros((6, 5), dtype=np.float32))
    manifest, _ = _patch("w", base, edited, "dense", {"t0_bits": edited.bits})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        body = io.BytesIO()
        np.lib.format.write_array(body, np.array([{"x": 1}], dtype=object), allow_pickle=True)
        archive.writestr("t0_bits.npy", body.getvalue())
    data = buffer.getvalue()
    manifest = copy.deepcopy(manifest)
    manifest["arrays_sha256"] = wp.hashlib.sha256(data).hexdigest()
    manifest["patch_sha256"] = wp.patch_sha256(manifest)
    with pytest.raises(wp.WspPatchRefused) as refused:
        wp.apply_patch(manifest, data, {"w": base})
    assert refused.value.code == "patch_malformed"


def test_arrays_digest_is_checked_before_the_archive_is_read(base, monkeypatch):
    edited = wp.from_array(np.zeros((6, 5), dtype=np.float32))
    manifest, data = _patch("w", base, edited, "dense", {"t0_bits": edited.bits})
    monkeypatch.setattr(wp, "read_archive", lambda *a: pytest.fail("the archive was read before its digest"))
    with pytest.raises(wp.WspPatchRefused) as refused:
        wp.apply_patch(manifest, data + b"x", {"w": base})
    assert refused.value.code == "patch_hash_mismatch"


def test_a_different_base_is_refused(base):
    edited = wp.from_array(np.zeros((6, 5), dtype=np.float32))
    manifest, data = _patch("w", base, edited, "dense", {"t0_bits": edited.bits})
    other = wp.from_array(base.values64().astype(np.float32) + np.float32(1))
    with pytest.raises(wp.WspPatchRefused) as refused:
        wp.apply_patch(manifest, data, {"w": other})
    assert refused.value.code == "patch_base_mismatch"


def test_an_edited_digest_that_does_not_rebuild_is_refused(base):
    edited = wp.from_array(np.zeros((6, 5), dtype=np.float32))
    manifest, data = _patch("w", base, edited, "dense", {"t0_bits": edited.bits})
    manifest["tensors"][0]["edited_sha256"] = "0" * 64
    manifest["content_sha256"] = wp.content_sha256(manifest)
    manifest["patch_sha256"] = wp.patch_sha256(manifest)
    with pytest.raises(wp.WspPatchRefused) as refused:
        wp.apply_patch(manifest, data, {"w": base})
    assert refused.value.code == "patch_hash_mismatch"


def test_a_tampered_manifest_is_refused(base):
    edited = wp.from_array(np.zeros((6, 5), dtype=np.float32))
    manifest, data = _patch("w", base, edited, "dense", {"t0_bits": edited.bits})
    manifest["created_at"] = "2026-11-17T00:00:00Z"
    with pytest.raises(wp.WspPatchRefused) as refused:
        wp.apply_patch(manifest, data, {"w": base})
    assert refused.value.code == "patch_hash_mismatch"


def test_check_limits(base):
    edited = wp.from_array(np.zeros((6, 5), dtype=np.float32))
    manifest, _ = _patch("weight", base, edited, "dense", {"t0_bits": edited.bits})
    wp.check_limits(manifest, editable={"weight", "bias"}, max_changed_entries=None)
    with pytest.raises(wp.WspPatchRefused) as refused:
        wp.check_limits(manifest, editable={"bias"}, max_changed_entries=None)
    assert refused.value.code == "patch_outside_limits"
    with pytest.raises(wp.WspPatchRefused) as refused:
        wp.check_limits(manifest, editable={"weight"}, max_changed_entries=manifest["changed_entries_total"] - 1)
    assert refused.value.code == "patch_outside_limits"


def test_tensor_sha256_is_little_endian_bytes():
    arr = np.array([1.0, -2.0], dtype=np.float32)
    assert wp.tensor_sha256(arr) == wp.hashlib.sha256(arr.astype("<f4").tobytes()).hexdigest()
    assert wp.tensor_sha256(arr.astype(">f4")) == wp.tensor_sha256(arr)


def test_lowrank_order_matters_and_the_sequential_sum_wins(base):
    """The fixture the golden lacks: miForge's lowrank golden is rank 1, where `a @ b` and the term-by-term sum are
    the same bits, so it cannot tell them apart (044-C8.1-NC-fused survived against it). Here the first term swamps
    the base and the second cancels it: term by term gives 0, fused gives the base back."""
    a = np.stack([np.full(6, 1e20), np.ones(6)], axis=1)
    b = np.stack([np.ones(5), np.full(5, -1e20)], axis=0)
    w = base.values64().copy()
    for k in range(2):  # built here, not by the module under test
        w = w + np.outer(a[:, k], b[k])
    sequential = wp.from_values("float32", w.astype(np.float32))
    fused = wp.from_values("float32", (base.values64() + a @ b).astype(np.float32))
    assert sequential.sha256 != fused.sha256, "fixture too weak: the two orders must give different bits"
    manifest, data = _patch("w", base, sequential, "lowrank", {"t0_a": a, "t0_b": b,
                                                               "t0_idx": np.zeros(0, dtype=np.int64),
                                                               "t0_val": np.zeros(0, dtype=np.uint32)}, rank=2)
    _, out = wp.apply_patch(manifest, data, {"w": base})
    assert out["w"].sha256 == sequential.sha256
    assert np.array_equal(out["w"].values64(), np.zeros((6, 5)))
