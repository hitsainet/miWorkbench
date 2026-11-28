"""The SAE weights hash a circuit or cluster definition names (ENH-011 X-2, #12).

A definition binds on its consumer by `mistudio_sae_id`, `layer`, `hook_type`, `n_features` and `d_model`, which a
different SAE of the same shape (another training run, another release of one repository) matches just as well. The
probe definition has carried `sae.weights_sha256` since 033; circuit and cluster definitions now carry the same value,
`DefinitionSAERef.weights_sha256`, computed by the SAME function (`probe_definition_builder._sae_weights_sha256`), so
a consumer that checks one kind of document checks them all the same way.

The field is OPTIONAL (planner default E11-6). Where the probe export REFUSES an SAE it cannot hash, a circuit or
cluster export goes out without the field and says why in the log: those exports worked before the field existed,
and a refusal would turn every SAE with no local copy into a failed export. Absent means "not recorded", never a guess.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from .probe_definition_builder import ProbeExportRefused, _sae_weights_sha256

logger = logging.getLogger(__name__)


async def definition_weights_sha256(sae_row: Any, memo: Optional[Dict[str, Optional[str]]] = None) -> Optional[str]:
    """The sha256 of the weights file `sae_row` names, or None when it cannot be computed (logged, with the reason).

    Hashing reads the whole file, so it runs off the event loop. `memo` (keyed by SAE id) lets one export that names
    an SAE several times (a bundle) read its file once."""
    key = str(getattr(sae_row, "id", ""))
    if memo is not None and key in memo:
        return memo[key]
    try:
        digest: Optional[str] = await asyncio.to_thread(_sae_weights_sha256, sae_row)
    except (ProbeExportRefused, OSError) as exc:
        logger.warning("definition exported without weights_sha256 for SAE %s: %s", key, exc)
        digest = None
    if memo is not None:
        memo[key] = digest
    return digest
