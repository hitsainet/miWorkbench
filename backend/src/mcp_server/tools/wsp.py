"""Weight-space programming tools (category: wsp) — ENH-010 044 FR-15, FTASKS 4.4, 14.2, 15.x.

Two of the phase-3 tools, the two that were operator-only until now (the operator pasted curl commands for them):

* `capture_wsp_heldout` — `POST /wsp/captures`: capture a probe's held-out activations on a GPU (the probe's model is
  loaded once on one card), each set checked for parity with its stored evaluation.
* `import_wsp_candidate` — `POST /wsp/candidates/import`: import a verdicted miForge candidate as a NEW probe (the
  parent is never changed). The new probe earns its rung at import, from the held-out evaluations the import runs: it
  is created at rung 0 and can come in at rung 2 (`pm_ae08dd79a26e` did; FU-5, 2026-11-25). It takes the candidate's
  documents verbatim, or a candidate id (`wspc_…`) that this module resolves through miForge's API to the run
  document, the patch manifest and `edit.npz` (FU-4, 2026-11-25); a document given beside the id that is not that
  candidate's is refused by name before anything is filed.

BOTH ARE ALWAYS GATED BY THE OPERATOR, and the ROUTE spends the approval (FTDD §5.4, §5.6; P66, R3B-4): the first
call files one approval (`_gate`: the FTDD's `approval_gate(..., enabled=True, consume=False)`) and returns
`{approval_request_id, status: "pending_approval"}` — nothing is queued; the re-call with the same arguments and that
id, once the operator approved it in miWorkbench's Approvals view, sends the SAME body to the route with the id in the
approval-request header, and the route consumes it (once, for exactly that payload) before its handler runs. Not keyed
on `MCP_PROBE_GPU_APPROVAL`: the route refuses the MCP pod without an approved id whatever that setting says.

The rest of FR-15's seven (bundles, experiments, the experiment import) land with 044-C's experiment driver
(FTASKS 13.x, 14.1); the scorer, the bundle files and sensitivity marks never get a tool (FTDD §5.4).

Every call goes over HTTP through the client this module is handed (`test_mcp_tools_stay_on_http.py`).
"""

import base64
import binascii
import hashlib
import json
from typing import Annotated, Any, Dict, List, Literal, Optional

import httpx
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from src.core import approval_payload

from ..client import BackendError, MiWorkbenchClient
from ..config import MCPSettings

#: The backend's API prefix: an approval names the route's FULL template, as the REST gate reads it (045 A38).
API_PREFIX = "/api/v1"

APPROVAL_ID_DESCRIPTION = ("The id a first call returned with status pending_approval; pass it, with the SAME "
                           "arguments, once the operator approved it in miWorkbench")


class ImportAcknowledgement(BaseModel):
    """Importing a verdict other than `improved`, or an `improved` that is `improved_unconfirmed` (FU-7b), needs a
    named acceptance (the route's `ImportAcknowledgement`)."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["not_significant", "regressed", "unmeasured", "improved_unconfirmed"] = Field(
        description="The verdict you accept, exactly as the run document states it; for an `improved` verdict no "
        "fresh primary held-out set confirms (each read `improved` with `heldout_uses` above 1), `improved_unconfirmed`")
    reason: str = Field(min_length=1, max_length=2000, description="Why this candidate is imported anyway")


#: miForge's candidate ids (`wsp/runs.py::new_candidate_id`, the run schema's `candidate_id` pattern).
CANDIDATE_ID_PATTERN = r"^wspc_[0-9a-f]{12}$"

#: The three documents the import route takes, in the order the tool fetches the missing ones.
DOCUMENTS = ("run_document", "patch_manifest", "edit_npz_b64")

#: The miForge reads a candidate id resolves through (miForge `api/v1/endpoints/wsp_candidates.py`, under `/api/v1`):
#: its summary, then each document the caller did not give. `test_wsp_import_by_candidate_cross_repo.py` holds each
#: one to miForge's live router.
FORGE_SUMMARY = "/wsp/candidates/{candidate_id}"
FORGE_DOCUMENTS = {
    "run_document": FORGE_SUMMARY + "/run-document",
    "patch_manifest": FORGE_SUMMARY + "/patch/patch.json",
    "edit_npz_b64": FORGE_SUMMARY + "/patch/edit.npz",
}

#: How long a read of miForge may take (one of four per resolution; `edit.npz` is tens of KB).
FORGE_TIMEOUT_S = 30.0

#: What every candidate-id path answers when miForge is not configured on this MCP server.
NO_FORGE = ("MIFORGE_API_URL is not set on this MCP server, so a candidate id cannot be resolved; pass the "
            "candidate's run_document, patch_manifest and edit_npz_b64 instead")


def _forge_client(url: str) -> httpx.AsyncClient:
    """The miForge HTTP client a candidate id is resolved with. A function so a test can hand in a transport."""
    return httpx.AsyncClient(base_url=url, timeout=FORGE_TIMEOUT_S)


class _Stop(Exception):
    """A resolution that cannot go on: `answer` is what the tool returns (a refusal or `unavailable`)."""

    def __init__(self, answer: Dict[str, Any]):
        super().__init__(answer.get("reason"))
        self.answer = answer


def _refused(code: str, reason: str, **extra: Any) -> Dict[str, Any]:
    return {"refused": code, "reason": reason, **extra, "filed": False,
            "note": "Nothing was filed, so no approval was asked for or spent."}


async def _forge_get(forge: httpx.AsyncClient, path: str) -> httpx.Response:
    """One read of miForge: a 4xx is miForge's own refusal, relayed by its code; anything else that is not a 2xx is
    `unavailable: miforge` (never read as an answer)."""
    try:
        response = await forge.get(f"/api/v1{path}")  # an f-string, so the one route walker reads it (046 R1-5)
    except httpx.HTTPError as exc:
        raise _Stop({"unavailable": "miforge", "reason": f"miForge unreachable on {path}: {type(exc).__name__}: {exc}"})
    if 400 <= response.status_code < 500:
        try:
            error = (response.json() or {}).get("error") or {}
        except ValueError:
            error = {}
        code = error.get("code") if isinstance(error, dict) else None
        raise _Stop(_refused(str(code or f"miforge_{response.status_code}"),
                             f"miForge refused {path}: {error.get('message') if isinstance(error, dict) else None}",
                             miforge_status=response.status_code))
    if response.status_code >= 300:
        raise _Stop({"unavailable": "miforge", "reason": f"miForge answered {response.status_code} on {path}"})
    return response


def _forge_json(response: httpx.Response, path: str) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise _Stop({"unavailable": "miforge", "reason": f"miForge returned no JSON object from {path} (a misrouted "
                                                         "ingress or proxy page), so nothing was resolved"})
    return body


def _forge_bytes(response: httpx.Response, path: str) -> bytes:
    """A served file, checked against the `X-Content-SHA256` miForge computes over exactly the bytes it sent."""
    data = response.content
    stated = response.headers.get("X-Content-SHA256")
    if stated is None or hashlib.sha256(data).hexdigest() != stated:
        raise _Stop({"unavailable": "miforge", "reason": f"{path} arrived without a matching X-Content-SHA256 "
                                                         f"(stated {stated}), so it was not used"})
    return data


async def resolve_candidate(url: str, candidate_id: str, given: Dict[str, Any]) -> Dict[str, Any]:
    """FU-4 (2026-11-25): a candidate id → its summary (`GET /wsp/candidates/{id}`) and every document the caller did
    not give (`…/run-document`, `…/patch/patch.json`, `…/patch/edit.npz`), from miForge. Raises `_Stop`."""
    base = FORGE_SUMMARY.format(candidate_id=candidate_id)
    forge = _forge_client(url)
    try:
        summary = _forge_json(await _forge_get(forge, base), base)
        if summary.get("candidate_id") != candidate_id:
            raise _Stop({"unavailable": "miforge", "reason": f"miForge answered {base} with candidate "
                                                             f"{summary.get('candidate_id')!r}, so nothing was used"})
        if not isinstance(summary.get("verdict"), dict):
            raise _Stop(_refused("run_not_evaluated", f"candidate {candidate_id} has no verdict in miForge; only an "
                                                      "evaluated candidate can be imported (miForge's evaluate "
                                                      "scores it)", candidate_id=candidate_id))
        docs = dict(given)
        if docs.get("run_document") is None:
            path = FORGE_DOCUMENTS["run_document"].format(candidate_id=candidate_id)
            docs["run_document"] = _forge_json(await _forge_get(forge, path), path)
        if docs.get("patch_manifest") is None:
            path = FORGE_DOCUMENTS["patch_manifest"].format(candidate_id=candidate_id)
            try:
                manifest = json.loads(_forge_bytes(await _forge_get(forge, path), path))
            except ValueError:
                manifest = None
            if not isinstance(manifest, dict):
                raise _Stop({"unavailable": "miforge", "reason": f"{path} is not a JSON object"})
            docs["patch_manifest"] = manifest
        if docs.get("edit_npz_b64") is None:
            path = FORGE_DOCUMENTS["edit_npz_b64"].format(candidate_id=candidate_id)
            docs["edit_npz_b64"] = base64.b64encode(_forge_bytes(await _forge_get(forge, path), path)).decode("ascii")
    finally:
        await forge.aclose()
    return {"summary": summary, **docs}


def candidate_mismatch(candidate_id: str, summary: Dict[str, Any], run_document: Dict[str, Any],
                       patch_manifest: Dict[str, Any], edit_npz_b64: str) -> Optional[Dict[str, Any]]:
    """The refusal, by name, when a document is not the candidate's; `None` when every one is.

    * `candidate_run_mismatch` — the run document is another run's, or another candidate's;
    * `candidate_bundle_mismatch` — the run's bundle or the patch's base bundle is not the candidate's;
    * `candidate_patch_mismatch` — the patch manifest is not the candidate's patch, or `edit.npz` is not its arrays.
    """
    run_id, bundle = summary.get("run_id"), summary.get("bundle_base_sha256")
    if run_document.get("run_id") != run_id or run_document.get("candidate_id") != candidate_id:
        return _refused("candidate_run_mismatch", f"the run document is run {run_document.get('run_id')}'s candidate "
                        f"{run_document.get('candidate_id')}, not candidate {candidate_id} of run {run_id}",
                        candidate_id=candidate_id, run_id=run_id)
    request = run_document.get("request") if isinstance(run_document.get("request"), dict) else {}
    run_bundle = (request.get("bundle") or {}).get("bundle_sha256") if isinstance(request.get("bundle"), dict) \
        else None
    if run_bundle != bundle or patch_manifest.get("bundle_base_sha256") != bundle:
        return _refused("candidate_bundle_mismatch", f"candidate {candidate_id} is on bundle {bundle}; the run "
                        f"document names bundle {run_bundle} and the patch names {patch_manifest.get('bundle_base_sha256')}",
                        candidate_id=candidate_id, bundle_sha256=bundle)
    if patch_manifest.get("patch_sha256") != summary.get("patch_sha256"):
        return _refused("candidate_patch_mismatch", f"the patch manifest is patch {patch_manifest.get('patch_sha256')}, "
                        f"not candidate {candidate_id}'s {summary.get('patch_sha256')}", candidate_id=candidate_id)
    try:
        arrays = base64.b64decode(edit_npz_b64, validate=True)
    except (binascii.Error, ValueError):
        arrays = None
    if arrays is None or hashlib.sha256(arrays).hexdigest() != patch_manifest.get("arrays_sha256"):
        return _refused("candidate_patch_mismatch", f"edit_npz_b64 is not candidate {candidate_id}'s arrays (the "
                        "manifest's arrays_sha256)", candidate_id=candidate_id)
    return None


def filed_payload(route: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """The canonical payload of `POST /api/v1<route>` with this body (`core.approval_payload`): what the operator
    approves and what the route's gate rebuilds from the re-call and compares."""
    return approval_payload.canonical_payload("POST", API_PREFIX + route, {}, {}, body)


def _gpu_card(preflight: Dict[str, Any]) -> Dict[str, Any]:
    """What the operator's card shows about the GPU and the sets (filed BESIDE the payload; never approved)."""
    gpu = preflight.get("gpu") or {}
    return {
        "gpu": {"summary": gpu.get("message"), "checked": gpu.get("checked"), "fits": gpu.get("fits"),
                "required_mb": gpu.get("required_mb"), "requested": gpu.get("requested"), "card": gpu.get("card"),
                "model_id": gpu.get("model_id"), "quantization": gpu.get("quantization")},
        "sets": [{k: s.get(k) for k in ("dataset_id", "role", "rows", "would_run")}
                 for s in (preflight.get("sets") or [])][:64],
    }


def _status_hint(tool_name: str, status: str) -> str:
    return {
        "pending": "The operator has not decided yet. Re-call later with the same approval_request_id.",
        "consumed": "This approval was already used. Request a new one by calling without approval_request_id.",
    }.get(status, f"Request a new approval by calling {tool_name} without approval_request_id.")


def _not_found(tool_name: str, approval_request_id: str) -> Dict[str, Any]:
    return {"approval_request_id": approval_request_id, "status": "not_found",
            "hint": f"No such approval. Call {tool_name} without approval_request_id to request one."}


def _mismatch(approval_request_id: str, reason: Any) -> Dict[str, Any]:
    return {"approval_request_id": approval_request_id, "status": "mismatch", "reason": reason,
            "hint": "That approval is for different arguments. An approval covers exactly the payload it names."}


def route_recall_refusal(tool_name: str, approval_request_id: str, exc: BackendError) -> Optional[Dict[str, Any]]:
    """The structured answer to an approved re-call the ROUTE's gate refused (`require_consumed_approval`): 409
    `{status, reason}` (not approved, already spent, failed), 404 for an unknown id, 422 `mismatch`. `None` for
    anything else, which is the handler's own answer and reaches the agent verbatim — a 404 or 409 carrying a `code`
    (`probe_not_found`, a capture or import refusal) is the handler's, never read as the approval's."""
    detail = exc.detail
    if exc.status_code == 409 and isinstance(detail, dict) and detail.get("status") and "code" not in detail:
        status = str(detail["status"])
        return {"approval_request_id": approval_request_id, "status": status, "reason": detail.get("reason"),
                "hint": _status_hint(tool_name, status)}
    if exc.status_code == 404 and isinstance(detail, str) and detail.startswith("Approval request"):
        return _not_found(tool_name, approval_request_id)
    if exc.status_code == 422 and isinstance(detail, dict) and detail.get("code") in ("mismatch",
                                                                                      "approval_not_applicable"):
        return _mismatch(approval_request_id, detail)
    return None


#: The tool an accepted import's `task_id` resolves through (its `wsp_candidate_import` task-queue row, 2026-11-27).
IMPORT_POLL_TOOL = "get_task_status"


def with_import_poll(answer: Any) -> Any:
    """An accepted import (`status: importing`) names its poll as a call: `get_task_status {task_id}`. Before
    2026-11-27 the prose said to poll it and it answered `not_found` for the id (the round agent read the worker
    log)."""
    from ..result_language import next_step

    if isinstance(answer, dict) and answer.get("status") == "importing" and answer.get("task_id"):
        step = next_step(IMPORT_POLL_TOOL, {"task_id": answer["task_id"]},
                         "the import's task-queue record: running, then completed as the new probe (its rung in "
                         "entity_info) or failed with the refusal's code")
        return {**answer, "next_steps": [step] + list(answer.get("next_steps") or [])}
    return answer


def register(mcp: MCPServer, client: MiWorkbenchClient, settings: MCPSettings) -> None:
    async def _gate(tool_name: str, payload: Dict[str, Any], approval_request_id: Optional[str], *,
                    reason: str, forecast: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """`None` when the route may be called now (with the id); otherwise the answer the tool returns.

        FTDD §5.4's `approval_gate(…, enabled=True, consume=False)`. ALWAYS enabled — never keyed on
        `MCP_PROBE_GPU_APPROVAL`, because the route refuses the MCP pod without an approved id whatever that setting
        says. NEVER consumes: the route does, exactly once, so a consume here would make the route's answer 409.

        * no id: files `POST /mcp/approvals {tool_name, payload[, forecast]}` and answers `pending_approval`.
          `forecast` rides BESIDE the payload for the operator's card; only the payload is approved.
        * an id: READS the request (`GET /mcp/approvals/{id}`) and answers `None` only when it is approved, for this
          tool and exactly this payload. Pending, denied, expired, spent or failed, or another tool's or payload's,
          is answered here and the route is never called.
        """
        if approval_request_id is None:
            body: Dict[str, Any] = {"tool_name": tool_name, "payload": payload}
            if forecast is not None:
                body["forecast"] = forecast
            created = await client.post("/mcp/approvals", json_body=body)
            return {
                "approval_request_id": created["id"],
                "status": "pending_approval",
                "hint": (f"{tool_name} {reason}, so it needs operator approval in miWorkbench's Approvals view. "
                         "Poll get_approval_status; once it is approved, re-call "
                         f"{tool_name} with the same arguments and this approval_request_id."),
            }
        try:
            row = await client.get(f"/mcp/approvals/{approval_request_id}")
        except BackendError as exc:
            if exc.status_code == 404:
                return _not_found(tool_name, approval_request_id)
            raise
        if row.get("tool_name") != tool_name:
            return _mismatch(approval_request_id,
                             f"approval {approval_request_id} is for {row.get('tool_name')}, not {tool_name}")
        if row.get("payload") != payload:
            return _mismatch(approval_request_id, "the payload differs from the one the operator approved")
        status = row.get("status")
        if status != "approved":
            return {"approval_request_id": approval_request_id, "status": status, "reason": row.get("reason"),
                    "hint": _status_hint(tool_name, str(status))}
        return None

    async def _submit_approved(tool_name: str, route: str, body: Dict[str, Any], approval_request_id: str) -> Any:
        """The approved re-call: the SAME body to the route, the id in the approval-request header; the route's gate
        spends it. A refusal by that gate is a structured status; anything else is the handler's answer."""
        try:
            return await client.post(route, json_body=body,
                                     headers={approval_payload.APPROVAL_REQUEST_HEADER: approval_request_id})
        except BackendError as exc:
            refused = route_recall_refusal(tool_name, approval_request_id, exc)
            if refused is None:
                raise
            return refused

    @mcp.tool()
    async def capture_wsp_heldout(
        probe_id: Annotated[str, Field(min_length=1, max_length=64, description="miWorkbench probe id (pm_…)")],
        dataset_ids: Annotated[Optional[List[str]], Field(max_length=64, description="Only these held-out views (and/or the probe's calibration view, and/or its `supplement` views: validation-only practice rows); omitted = every held-out and supplement set the probe was evaluated on")] = None,
        approval_request_id: Annotated[Optional[str], Field(description=APPROVAL_ID_DESCRIPTION)] = None,
    ) -> Any:
        """Capture a probe's held-out activations for weight-space programming (ENH-010): a GPU job that loads the
        probe's model once, on ONE card, and captures each held-out set's scored-token activations at the probe's
        layer, then checks each set reproduces its stored evaluation (`parity`). miForge never sees a held-out row;
        the capture is what lets miWorkbench score miForge's candidates on them.

        ALWAYS waits for the operator: the first call files an approval in miWorkbench and returns
        `{approval_request_id, status: "pending_approval"}` — nothing is queued. Poll `get_approval_status`; once it
        is approved, re-call with the SAME arguments plus `approval_request_id`, which starts it (the route spends the
        approval once). Every call first reads `GET /wsp/captures/preflight`: the sets and their rows, which would run,
        and whether a card has the memory the model needs NOW. When no card does, the answer is
        `{"refused": "gpu_memory_insufficient", ...}` and no approval is filed or spent (free the card — another job,
        or miServer's resident model — and call again). When every set is already captured or under way, nothing is
        filed and the current captures are returned. Poll the task with `get_task_status`; each set's `parity`
        appears on its capture."""
        body: Dict[str, Any] = {"probe_id": probe_id}
        if dataset_ids is not None:
            body["dataset_ids"] = list(dataset_ids)
        query: Dict[str, Any] = {"probe_id": probe_id}
        if dataset_ids:
            query["dataset_ids"] = list(dataset_ids)
        preflight = await client.get("/wsp/captures/preflight", **query)
        if preflight.get("refusal"):
            return {"refused": preflight["refusal"]["code"], "reason": preflight["refusal"]["message"],
                    "gpu": preflight.get("gpu"), "sets": preflight.get("sets")}
        if not preflight.get("to_run"):
            return {"probe_id": probe_id, "started": [], "sets": preflight.get("sets"),
                    "note": "Every set is already captured, pending or running; nothing to start, nothing filed."}
        held = await _gate("capture_wsp_heldout", filed_payload("/wsp/captures", body), approval_request_id,
                           reason="holds a GPU", forecast=_gpu_card(preflight))
        if held is not None:
            return {**held, "sets": preflight.get("sets"), "gpu": preflight.get("gpu")}
        return await _submit_approved("capture_wsp_heldout", "/wsp/captures", body, approval_request_id)

    @mcp.tool()
    async def import_wsp_candidate(
        candidate_id: Annotated[Optional[str], Field(pattern=CANDIDATE_ID_PATTERN, description="miForge's candidate id (`wspc_…`). Given alone, miForge's API resolves it to the candidate's run document, patch manifest and `edit.npz` (MIFORGE_API_URL on this server). Given beside any of them, each one given must be that candidate's, or the call is refused by name")] = None,
        run_document: Annotated[Optional[Dict[str, Any]], Field(description="miForge's run document for the candidate's run, with its verdict (`miforge.wsp-run/v1`), verbatim. Fetched from miForge when only candidate_id is given")] = None,
        patch_manifest: Annotated[Optional[Dict[str, Any]], Field(description="The candidate patch's manifest (`patch.json`, `miforge.wsp-patch/v1`), verbatim. Fetched from miForge when only candidate_id is given")] = None,
        edit_npz_b64: Annotated[Optional[str], Field(min_length=1, max_length=1_048_576, description="The patch's `edit.npz`, base64 (the request body is capped at 1 MiB in all). Fetched from miForge when only candidate_id is given")] = None,
        acknowledgement: Annotated[Optional[ImportAcknowledgement], Field(description="Required when the verdict is not `improved`, or is an `improved` no fresh primary held-out set confirms (`improved_unconfirmed`: every primary set that read improved had been scored before): the verdict you accept and why")] = None,
        approval_request_id: Annotated[Optional[str], Field(description=APPROVAL_ID_DESCRIPTION)] = None,
    ) -> Any:
        """Import a verdicted miForge weight-space candidate as a NEW probe that EARNS ITS RUNG AT IMPORT (ENH-010
        FR-8): the import calibrates and evaluates it from the held-out captures and sets its rung from those
        evaluations, so it can come in at rung 2 (it holds rung 0 only while the import runs). The parent probe and
        its file are never changed; the new probe records `derived_from` and is never auto-selected.

        Name the candidate by `candidate_id` (`wspc_…`) and miForge's API supplies its run document, patch manifest
        and `edit.npz`; or pass those three verbatim. With both, each document given must be the candidate's:
        `candidate_run_mismatch` (another run or candidate), `candidate_bundle_mismatch` (another bundle),
        `candidate_patch_mismatch` (another patch or arrays) is answered `{"refused": code, ...}` and nothing is filed.
        A candidate miForge has not evaluated is refused `run_not_evaluated` the same way; with no MIFORGE_API_URL the
        answer is `{"unavailable": "miforge", ...}`.

        ALWAYS waits for the operator: the first call files an approval and returns `{approval_request_id, status:
        "pending_approval"}`; nothing is written yet. Once approved, re-call with the SAME arguments plus
        `approval_request_id` (a candidate id is resolved again and must give the same documents): the route spends
        the approval, runs every check (schemas and digests, the bundle, the verdict, the patch's base and limits) and
        refuses `{code, message}` (`nothing_to_import`, `acknowledgement_required`, `patch_base_mismatch`,
        `patch_hash_mismatch`, `patch_outside_limits`, `run_sha256_mismatch`, …) — a refused call spends the approval
        (it reads `failed`). Accepted, it answers `{task_id, task_queue_id, status: "importing", parent_probe_id, …}`
        and `next_steps` naming the poll: call `get_task_status` with that `task_id`. It answers the import's
        task-queue record: `running` (naming the parent probe), then `completed` naming the NEW probe (`entity_id`,
        with its `rung` in `entity_info`), or `failed` with `error_message` `refused <code>: <message>` (the worker
        repeats every check)."""
        docs: Dict[str, Any] = {"run_document": run_document, "patch_manifest": patch_manifest,
                                "edit_npz_b64": edit_npz_b64}
        resolved: Optional[Dict[str, Any]] = None
        if candidate_id is None:
            missing = [name for name in DOCUMENTS if docs[name] is None]
            if missing:
                return _refused("candidate_or_documents_required", "pass candidate_id, or all of run_document, "
                                f"patch_manifest and edit_npz_b64 (missing: {', '.join(missing)})", missing=missing)
        else:
            if not settings.miforge_api_url:
                return {"unavailable": "miforge", "reason": NO_FORGE, "filed": False}
            try:
                found = await resolve_candidate(settings.miforge_api_url, candidate_id, docs)
            except _Stop as stop:
                return stop.answer
            summary = found.pop("summary")
            docs = found
            mismatch = candidate_mismatch(candidate_id, summary, docs["run_document"], docs["patch_manifest"],
                                          docs["edit_npz_b64"])
            if mismatch is not None:
                return mismatch
            resolved = {"candidate_id": candidate_id, "run_id": summary.get("run_id"),
                        "bundle_base_sha256": summary.get("bundle_base_sha256"),
                        "patch_sha256": summary.get("patch_sha256"),
                        "verdict": (summary.get("verdict") or {}).get("value")}
        body: Dict[str, Any] = dict(docs)
        if acknowledgement is not None:
            body["acknowledgement"] = acknowledgement.model_dump()
        held = await _gate("import_wsp_candidate", filed_payload("/wsp/candidates/import", body), approval_request_id,
                           reason="registers a new probe from a miForge candidate")
        if held is not None:
            return held if resolved is None else {**held, "candidate": resolved}
        answer = await _submit_approved("import_wsp_candidate", "/wsp/candidates/import", body, approval_request_id)
        return with_import_poll(answer)
