"""miForge's held-out uses, read through miForge's API: the authority on which held-out sets were scored (FU-7a C1).

miForge counts every use of a held-out set inside the verdict that used it (`GET /api/v1/wsp/heldout-uses`, one row
per owner, set name and `row_ids_sha256`; FR-016.29). A bundle request is refused `heldout_reused` when a held-out set
it would carry already has a use there, and `heldout_overlaps_practice` when a supplement it would carry does (miForge
would then read rows its own verdicts were scored on). The calibration capture is exempt: it is the parent's bar
basis, kept across rebuilds by decision (FU-7a D4). The rows miForge counts as used travel to the build
(`used_by_us`), which refuses a new set sharing any row key with one of them (`wsp_bundle.check_heldout_unused`).

NEVER ASSUMED. miForge not configured (`MIFORGE_API_URL` empty) is `miforge_not_configured`, and an answer that cannot
be read (unreachable, not 200, not the expected shape) is `heldout_uses_unavailable`, both 503 with nothing queued:
"no answer" is never read as "no uses". Read by the route only (`POST /wsp/bundles`): the `wsp` worker that builds
the bundle has no miForge address; the build repeats the local half (`wsp_bundle.check_heldout_unused`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import httpx

from .wsp_bundle import BundleRefused, heldout_set_name

USES_PATH = "/api/v1/wsp/heldout-uses"
TIMEOUT_S = 15.0
OWNER = "miworkbench"


def read_heldout_uses(base_url: str, *, transport: Optional[httpx.BaseTransport] = None) -> List[Dict[str, Any]]:
    """Every `{owner, set_name, row_ids_sha256, heldout_uses, last_used_at, last_candidate_id}` row miForge holds."""
    if not base_url:
        raise BundleRefused("miforge_not_configured",
                            "MIFORGE_API_URL is not set, so miForge's held-out uses cannot be read; a bundle is built "
                            "only when every held-out set it would carry is shown unused there", status=503)
    url = base_url.rstrip("/") + USES_PATH
    try:
        with httpx.Client(timeout=TIMEOUT_S, transport=transport) as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        raise _unavailable(f"miForge is unreachable on {url}: {type(exc).__name__}: {exc}")
    if response.status_code != 200:
        raise _unavailable(f"miForge answered {response.status_code} on {url}")
    try:
        body = response.json()
    except ValueError:
        raise _unavailable(f"miForge answered no JSON on {url}")
    rows = body.get("heldout_uses") if isinstance(body, dict) else None
    if not isinstance(rows, list) or not all(_well_formed(r) for r in rows):
        raise _unavailable(f"miForge's answer on {url} is not a list of held-out uses")
    return rows


def _well_formed(row: Any) -> bool:
    return (isinstance(row, dict) and all(isinstance(row.get(k), str) for k in ("owner", "set_name", "row_ids_sha256"))
            and isinstance(row.get("heldout_uses"), int) and not isinstance(row.get("heldout_uses"), bool)
            and row["heldout_uses"] >= 0)


def _unavailable(message: str) -> BundleRefused:
    return BundleRefused("heldout_uses_unavailable",
                         f"{message}. miForge's held-out uses decide whether a held-out set is fresh, so nothing was "
                         f"queued; retry when miForge answers", status=503)


def used_by_us(uses: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """miWorkbench's sets with at least one use: `{set_name, row_ids_sha256, heldout_uses}` each. The route passes
    these to the build, which compares their rows with the new sets (`wsp_bundle.earlier_bundle_keys`)."""
    return [{"set_name": u["set_name"], "row_ids_sha256": u["row_ids_sha256"], "heldout_uses": int(u["heldout_uses"])}
            for u in uses if u["owner"] == OWNER and int(u["heldout_uses"]) > 0]


def refuse_used(captures: Sequence[Any], supplements: Sequence[Any], uses: Sequence[Dict[str, Any]]) -> None:
    """`heldout_reused` for a held-out capture miForge has scored (by its set name or its row-id digest), and
    `heldout_overlaps_practice` for a supplement capture whose rows were scored as a held-out set."""
    used = [u for u in uses if u["owner"] == OWNER and int(u["heldout_uses"]) > 0]
    by_name = {u["set_name"]: u for u in used}
    by_rows = {u["row_ids_sha256"]: u for u in used}
    for capture in captures:
        if capture.role == "calibration":
            continue
        name = heldout_set_name(capture)
        hit = by_name.get(name) or by_rows.get(capture.row_ids_sha256)
        if hit:
            raise BundleRefused(
                "heldout_reused",
                f"held-out set {name} (view {capture.dataset_id}) was scored {hit['heldout_uses']} time(s) in miForge "
                f"(as {hit['set_name']}, last {hit.get('last_used_at')}); a held-out set is scored in one bundle's "
                f"history only. Capture fresh held-out rows and name them (heldout_view_ids).",
                set=name, heldout_view=capture.dataset_id, heldout_uses=int(hit["heldout_uses"]),
                miforge_set=hit["set_name"], last_used_at=hit.get("last_used_at"))
    for capture in supplements:
        hit = by_rows.get(capture.row_ids_sha256)
        if hit:
            raise BundleRefused(
                "heldout_overlaps_practice",
                f"supplement view {capture.dataset_id} holds exactly the rows of held-out set {hit['set_name']}, "
                f"scored {hit['heldout_uses']} time(s) in miForge; a held-out row is never practice.",
                practice_view=capture.dataset_id, miforge_set=hit["set_name"], key="miforge_heldout_use")
