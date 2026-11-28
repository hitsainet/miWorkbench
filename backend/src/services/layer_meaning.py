"""The `layer_meaning_changed` flag: an artefact recorded at the indices of the wrong layer stack.

ENH-011 #2. Before X-1, `discover_transformer_structure` took the FIRST layer stack in a model's
module tree. On a composite model whose vision tower is registered before its language model
(Gemma 3 and 4, LLaVA, Qwen2.5-VL) that was the vision encoder, so an SAE, a probe or a circuit
recorded there names layer indices into a stack a text forward never runs. After the fix, the same
index means a text decoder layer: the artefact's layer numbers no longer mean what they meant when
it was recorded.

Operator decision Q1 (2026-11-15): each such artefact is marked and REFUSED for serving or arming
until it is re-validated on the corrected resolver; nothing is deleted. OQ-3 (2026-11-26, answer B):
the flag is built even though the live inventory found nothing to mark
(`0xcc/reviews/ENH-011_X-1_inventory_2026-11-26.md`).

The flag is a nullable JSONB column, `layer_meaning_changed`, on `external_saes`, `probe_monitors`
and `circuits`, set by migration `c7d2e4f9a1b3` and never by a request:
`{old_layers_path, new_layers_path, model_type, flagged_at, flagged_by, reason}`. NULL means the
artefact was recorded on the corrected resolver, or on a model whose stacks never disagreed. A
cluster profile has no layer of its own; it reads its SAE's, so it is refused through its SAE.

RE-VALIDATION is re-recording: train, fit or validate again on the corrected resolver, which makes
a NEW row. The flagged row stays, readable, and stays refused for serving.

Refused here ("serving or arming" as miWorkbench meets it; miServer serves and arms what these
produce):
  - a probe's definition: build, read, export and publish;
  - a circuit's definition: export and its per-layer slices;
  - a cluster profile's definition (single and bundle) whose SAE is flagged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

CODE = "layer_meaning_changed"


class LayerMeaningChanged(Exception):
    """The artefact is flagged; HTTP 409 with `detail.code == "layer_meaning_changed"`."""

    status = 409

    def __init__(self, kind: str, artefact_id: str, flag: Dict[str, Any], *, via: Optional[str] = None):
        self.kind = kind
        self.artefact_id = artefact_id
        self.flag = dict(flag)
        self.via = via
        old = self.flag.get("old_layers_path")
        new = self.flag.get("new_layers_path")
        recorded_at = f"layer indices into {old!r}" if old else "layer indices into a stack the old resolver chose"
        subject = f"{kind} {artefact_id}" + (f" (through {via})" if via else "")
        super().__init__(
            f"{subject} was recorded at {recorded_at}; on this model the text decoder's layers are "
            f"{new!r}, so its layer numbers no longer mean what they meant when it was recorded "
            f"(ENH-011 #2). Refused for serving and arming until it is re-validated on the corrected "
            f"resolver: record it again, which makes a new {kind}. This row is kept, never deleted."
        )

    def detail(self) -> Dict[str, Any]:
        return {
            "code": CODE,
            "message": str(self),
            "kind": self.kind,
            "id": self.artefact_id,
            "old_layers_path": self.flag.get("old_layers_path"),
            "new_layers_path": self.flag.get("new_layers_path"),
        }


def flag_of(row: Any) -> Optional[Dict[str, Any]]:
    """The row's flag, or None. Anything but a non-empty dict reads as not flagged."""
    flag = getattr(row, "layer_meaning_changed", None) if row is not None else None
    return flag if isinstance(flag, dict) and flag else None


def refuse_if_changed(kind: str, row: Any, *, via: Optional[str] = None,
                      artefact_id: Optional[str] = None) -> None:
    """Raise `LayerMeaningChanged` when `row` is flagged; return None otherwise."""
    flag = flag_of(row)
    if flag is not None:
        raise LayerMeaningChanged(kind, artefact_id or str(getattr(row, "id", "?")), flag, via=via)


def refuse_probe(probe: Any) -> None:
    """A probe is refused when it is flagged.

    Its own flag is enough: the migration flags a probe through its run's model, and the SAE a
    k-sparse probe reads is on the same model, so it is flagged with it. The builder's gate, which
    has the SAE row loaded anyway, checks both.
    """
    refuse_if_changed("probe", probe)


def refuse_circuit(circuit: Any) -> None:
    """A circuit is refused when it is flagged (through its `model_id`, like its SAEs)."""
    refuse_if_changed("circuit", circuit)


async def layer_meaning_handler(request: Any, exc: LayerMeaningChanged):
    """409 with `detail.code == "layer_meaning_changed"`, registered on the app in `main.py`."""
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=exc.status, content={"detail": exc.detail()})
