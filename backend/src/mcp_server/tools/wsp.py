"""Weight-space programming tools (category: wsp) — ENH-010 044 FR-15, FTASKS 4.4, 14.2, 15.x.

Two of the phase-3 tools, the two that were operator-only until now (the operator pasted curl commands for them):

* `capture_wsp_heldout` — `POST /wsp/captures`: capture a probe's held-out activations on a GPU (the probe's model is
  loaded once on one card), each set checked for parity with its stored evaluation.
* `import_wsp_candidate` — `POST /wsp/candidates/import`: import a verdicted miForge candidate as a NEW probe at
  rung 0 (the parent is never changed).

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

from typing import Annotated, Any, Dict, List, Literal, Optional

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
    """Importing a verdict other than `improved` needs a named acceptance (the route's `ImportAcknowledgement`)."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["not_significant", "regressed", "unmeasured"] = Field(
        description="The verdict you accept, exactly as the run document states it")
    reason: str = Field(min_length=1, max_length=2000, description="Why this candidate is imported anyway")


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
        dataset_ids: Annotated[Optional[List[str]], Field(max_length=64, description="Only these held-out views (and/or the probe's calibration view); omitted = every held-out set the probe was evaluated on")] = None,
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
        run_document: Annotated[Dict[str, Any], Field(description="miForge's run document for the candidate's run, with its verdict (`miforge.wsp-run/v1`), verbatim")],
        patch_manifest: Annotated[Dict[str, Any], Field(description="The candidate patch's manifest (`patch.json`, `miforge.wsp-patch/v1`), verbatim")],
        edit_npz_b64: Annotated[str, Field(min_length=1, max_length=1_048_576, description="The patch's `edit.npz`, base64 (the request body is capped at 1 MiB in all)")],
        acknowledgement: Annotated[Optional[ImportAcknowledgement], Field(description="Required when the verdict is not `improved`: the verdict you accept and why")] = None,
        approval_request_id: Annotated[Optional[str], Field(description=APPROVAL_ID_DESCRIPTION)] = None,
    ) -> Any:
        """Import a verdicted miForge weight-space candidate as a NEW probe at rung 0 (ENH-010 FR-8): the parent probe
        and its file are never changed, the new probe records `derived_from`, is calibrated and evaluated from the
        held-out captures, and is never auto-selected. Its rung rises only from evaluations.

        ALWAYS waits for the operator: the first call files an approval and returns `{approval_request_id, status:
        "pending_approval"}`; nothing is checked or written yet. Once approved, re-call with the SAME arguments plus
        `approval_request_id`: the route spends the approval, runs every check (schemas and digests, the bundle, the
        verdict, the patch's base and limits) and refuses `{code, message}` (`nothing_to_import`,
        `acknowledgement_required`, `patch_base_mismatch`, `patch_hash_mismatch`, `patch_outside_limits`,
        `run_sha256_mismatch`, …) — a refused call spends the approval (it reads `failed`). Accepted, it answers
        `{task_id, status: "importing", parent_probe_id, …}`; poll `get_task_status`."""
        body: Dict[str, Any] = {"run_document": run_document, "patch_manifest": patch_manifest,
                                "edit_npz_b64": edit_npz_b64}
        if acknowledgement is not None:
            body["acknowledgement"] = acknowledgement.model_dump()
        held = await _gate("import_wsp_candidate", filed_payload("/wsp/candidates/import", body), approval_request_id,
                           reason="registers a new probe from a miForge candidate")
        if held is not None:
            return held
        return await _submit_approved("import_wsp_candidate", "/wsp/candidates/import", body, approval_request_id)
