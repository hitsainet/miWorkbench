"""Read one miWorkbench approval request (category: read) — MCP gap G10, 2026-10-21.

THE CHECK MUST COME BEFORE THE ACT. The caller-executed GPU tools (`start_probe_training_run`,
`evaluate_probe_monitor`; `miserver_load_model` until ENH-008 moved its approval into miServer) answer a re-call with the approval id by CONSUMING
the approval and starting the job — so before this tool, the only way an agent could learn
whether an approval had been granted was the call that spends it. During the end-to-end probe
run of 2026-10-20 the pre-GPU safety checks had to be run before knowing whether the approval existed at all.

UNGATED AND IN `read`, the category every deployment enables. It used to be registered only
when `MCP_STEERING_APPROVAL=true` (inside `steering.py`), so with steering approval off — the
deployed setting — the tool did not exist even though probe and model-load approvals did.

WHO DECIDED: the suite has no login (operator decision 2026-10-06), so no approval records a
person's name. An approval or denial is the operator's act in miWorkbench's approvals banner; an
expiry is the TTL's (`mcp_approval_ttl_hours`). The answer says exactly that, never a name.
"""

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import BackendError, MiWorkbenchClient
from ..config import MCPSettings
from ..result_language import STORED_ACTIONS, next_step

#: What an agent should do next, per status. A caller-executed approval is spent by RE-CALLING the
#: original tool with the same arguments plus the id; nothing here consumes it.
NEXT_STEP = {
    "pending": "The operator has not decided. Poll again later; do not re-submit.",
    "approved": "Approved and unspent. Re-call the original tool with the SAME arguments plus "
                "approval_request_id to use it (once); steering included (045 FR-13a).",
    "denied": "Denied. Do not retry the same request; ask the operator why (see reason).",
    "expired": "Expired unused. Call the original tool without approval_request_id to file a "
               "new request.",
    "consumed": "Already used once; the job it approved was submitted. A new request is needed "
                "to run it again.",
    "failed": "Used once, and the call it approved then failed (see reason). It is never re-opened: "
              "call the original tool without approval_request_id to file a new request.",
}

#: An approved code rule (stored `tool_name` `code_rule_approve`): the operator approved a HASH; nothing is re-called
#: with the id (R1 review fix OR1-045-1, 2026-11-16).
CODE_RULE_APPROVED = ("Approved: the operator approved this code rule's hash. Retry the call that was refused "
                      "code_rule_not_approved; nothing is called with approval_request_id.")

OPERATOR = ("operator (miWorkbench approvals banner; the suite has no login, so no name is "
            "recorded)")


def next_steps_for(row: dict[str, Any]) -> list[dict[str, Any]]:
    """045 FR-8: the status's next call, beside the legacy prose `next` (FTDD §8: poll the status tool; re-call the
    original tool; none). A steering approval the backend already submitted is polled by its task id. An approved
    code rule has none. The result-language wrapper then keeps only tools this server registers, sent only the
    arguments they declare (OR1-045-1)."""
    status, approval_id, tool = row.get("status"), row.get("id"), row.get("tool_name")
    if status == "pending":
        return [next_step("get_approval_status", {"approval_request_id": approval_id},
                          "the operator has not decided; poll again later and never re-submit")]
    if status == "approved" and tool in STORED_ACTIONS:
        return []  # a stored action is not a tool; the agent retries the call that was refused (OR1-045-1)
    if status == "approved" and row.get("steering_task_id"):
        return [next_step("get_steering_result", {"task_id": row["steering_task_id"]},
                          "the backend already submitted this steering task; poll its result")]
    if status == "approved" and isinstance(tool, str) and tool:
        return [next_step(tool, {"approval_request_id": approval_id},
                          f"re-call {tool} with the SAME arguments you first sent plus this approval_request_id; "
                          "it runs once")]
    return []


def describe(row: dict[str, Any]) -> dict[str, Any]:
    """The agent-facing view of one approval row: status, who decided, when, and what next."""
    status = row.get("status")
    reason = row.get("reason") or ""
    if status in ("approved", "denied", "consumed"):
        decided_by = OPERATOR
    elif status == "expired":
        # Two expiries (mcp_approvals.expire_stale): pending too long (the operator never
        # decided), or approved and never used (the operator DID decide, then time ran out).
        decided_by = (OPERATOR if reason.startswith("expired: approved")
                      else "expiry (mcp_approval_ttl_hours elapsed while pending)")
    else:
        decided_by = None
    return {
        "approval_request_id": row.get("id"),
        "tool_name": row.get("tool_name"),
        "status": status,
        "decided": status != "pending",
        "decided_by": decided_by,
        "decided_at": row.get("resolved_at"),
        "consumed_at": row.get("consumed_at"),
        "created_at": row.get("created_at"),
        "requested_by": row.get("requested_by"),
        "expires_at": row.get("expires_at"),
        "reason": row.get("reason"),
        "steering_task_id": row.get("steering_task_id"),
        "payload": row.get("payload"),
        "next": (CODE_RULE_APPROVED if status == "approved" and row.get("tool_name") in STORED_ACTIONS
                 else NEXT_STEP.get(status, "Unknown status; read the approvals banner.")),
        "next_steps": next_steps_for(row),
    }


def register(mcp: MCPServer, client: MiWorkbenchClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def get_approval_status(
        approval_request_id: Annotated[str, Field(description="The approval_request_id a gated miWorkbench tool returned (start_probe_training_run, evaluate_probe_monitor, capture_wsp_heldout, import_wsp_candidate, steer_* under approval mode, request_code_rule_approval)", min_length=1)],
    ) -> Any:
        """Read a miWorkbench operator-approval request WITHOUT spending it: status (pending |
        approved | denied | expired | consumed), who decided and when, when it was consumed,
        the reason, and what to do next.

        Use it BEFORE re-calling a gated GPU tool: that re-call consumes an approved request and
        starts the job at once. Read-only; never approves, denies or consumes. miServer and
        miDataworks decide their own requests — read those with miserver_get_approval_status and
        dataworks_get_approval_status."""
        try:
            row = await client.get(f"/mcp/approvals/{approval_request_id}")
        except BackendError as exc:
            if exc.status_code == 404:
                return {"approval_request_id": approval_request_id, "status": "not_found",
                        "next": "No such miWorkbench approval. A miServer approval id is read with "
                                "miserver_get_approval_status, a miDataworks one (apr_…) with "
                                "dataworks_get_approval_status.",
                        "next_steps": []}
            raise
        return describe(row)
