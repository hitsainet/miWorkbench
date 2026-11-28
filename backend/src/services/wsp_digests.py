"""Row-id and head digests shared by the WSP capture, bundle and scorer (ENH-010 044; FTDD §5.7.3, P64).

Every digest hashes canonical JSON (`rfc8785`, RFC 8785 JCS) or raw tensor bytes, never `json.dumps` output.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .wsp_patch import jcs_sha256, tensor_sha256

ROW_IDS_ALGORITHM = "miworkbench.row-ids/1"


def row_ids_sha256(view_id: str, positions: Iterable[int]) -> str:
    """`sha256(JCS({"algorithm", "view_id", "positions"}))`, positions in stored order (§5.7.3)."""
    return jcs_sha256({"algorithm": ROW_IDS_ALGORITHM, "view_id": str(view_id),
                       "positions": [int(p) for p in positions]})


def head_tensors_sha256(tensors: Mapping[str, np.ndarray]) -> str:
    """sha256 over each tensor's `tensor_sha256`, in NAME order (P64): a head's identity in the budget."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(tensor_sha256(np.asarray(tensors[name])).encode())
    return digest.hexdigest()


__all__: Sequence[str] = ("ROW_IDS_ALGORITHM", "row_ids_sha256", "head_tensors_sha256", "jcs_sha256",
                          "tensor_sha256")
