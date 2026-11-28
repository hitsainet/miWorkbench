"""The suite's result language on every miWorkbench tool answer (045 FR-7 – FR-9; phase B; CHAIN_PLAN A6, A30).

miAssist OWNS `mistudio.result-language/v1`; this server reads the vendored PACKAGE copy (`vendored/…`, byte-identical
to `docs/schemas/…`, pinned by `test_vendored_result_language.py`) and speaks it in three places:

* `approval` (FR-7): a gated tool's pending or decided approval, recognised from the fields it answers today (which
  stay exactly as they are) and its `TOOL_CLASSES` row (`then`, `decide_in`); the three status tools say the same.
  A relay whose source already carries `approval` keeps the source's object unchanged (A30; R1A-13).
* `next_steps` (FR-8): a NEW key beside the legacy `next`, which is never touched. A legacy list of
  `{tool, arguments, why}` converts generically; a prose or tool-name `next` is converted at its own site. On a relay
  (`dataworks_*`, `miserver_*`) items naming a tool this server does not register are dropped (A30; R1A-13). A gated
  tool's answer whose `approval` still waits names the deciding app's status tool first, with the id (`add_poll_step`,
  045 follow-up 2026-11-24).
* errors (FR-9): `teach` writes the contract's error text (code, fix, next, then the original text unchanged);
  `server.guard_and_teach` raises it.

`speak_result_language` wraps each tool's `fn` and is called by `build_server` after `apply_tool_classes` and before
`normalise_results` (FTDD §2), so it edits a plain dict. Removing that call fails every test in
`test_result_language_approval.py` and `test_result_language_next.py`.

Leaf module (FR-10): the standard library, pydantic and this package only (`.tool_classes`, and `.client` for
`BackendError`); never the backend's `src.*` layers, never a miAssist URL (`test_mcp_tools_stay_on_http.py`). The
vendored file ships in the backend image the MCP pod runs, so a missing copy fails the build (a packaging defect, never
a run-time state).
"""
from __future__ import annotations

import functools
import inspect
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

NAME = "mistudio-result-language-v1.json"  # p9: keep, the suite contract's file name (miAssist owns it)
CONTRACT_PATH = Path(__file__).parent / "vendored" / NAME

#: Every native or product status → the contract's `approval_status` (FPRD FR-7; A6 amended 2026-11-15 adds
#: `executing` and `failed`). Any other status (`not_found`, `mismatch`, …) is not an approval: no `approval`.
STATUS_MAP: dict[str, str] = {
    "pending_approval": "pending",
    "pending": "pending",
    "approved": "approved",
    "denied": "rejected",
    "rejected": "rejected",
    "consumed": "executed",
    "executed": "executed",
    "executing": "executing",
    "failed": "failed",
    "expired": "expired",
}

#: The three status tools and the queue each reads (their rows carry `approval: None`).
STATUS_TOOLS: dict[str, str] = {
    "get_approval_status": "miworkbench",
    "miserver_get_approval_status": "miserver",
    "dataworks_get_approval_status": "midataworks",
}

#: Tools whose answers relay another app's (their `next_steps` are filtered to this server's tools).
RELAY_PREFIXES = ("dataworks_", "miserver_")

#: A stored approval's `tool_name` that names no tool → the `TOOL_CLASSES` row that files it. A code-rule request is
#: stored as `code_rule_approve` (`services/code_rules.APPROVAL_TOOL`), which no server registers (R1 review fix
#: OR1-045-1, 2026-11-16).
STORED_ACTIONS: dict[str, str] = {"code_rule_approve": "request_code_rule_approval"}

#: Gated rows whose approval is never spent by re-calling with its id: the operator approves a code rule's HASH, and
#: the agent then retries the call that was refused `code_rule_not_approved` (C3; OR1-045-1).
NO_RECALL = frozenset({"request_code_rule_approval"})

#: Where each app's operator decides, for `summary`.
_WHERE = {
    "miworkbench": "miWorkbench's approvals banner",
    "miserver": "miServer's Approvals page",
    "midataworks": "miDataworks' Approvals page",
}
_STATUS_TOOL_OF = {queue: tool for tool, queue in STATUS_TOOLS.items()}

#: The contract's limits (`next_step.why` ≤ 300, `result_fields.next_steps` ≤ 5 items, `approval.summary` ≤ 500).
WHY_MAX, NEXT_STEPS_MAX, SUMMARY_MAX = 300, 5, 500

#: When true, an answer that keeps a legacy `next` and gains no `next_steps` raises (the drive test sets it, so a
#: site left unconverted is found); in production it is logged once per tool.
STRICT = False
_REPORTED: set[Any] = set()


class UnconvertedNext(AssertionError):
    """A tool answered a legacy `next` with no `next_steps` beside it (FTDD §8's guard)."""


@functools.lru_cache(maxsize=1)
def load_contract() -> dict[str, Any]:
    """The vendored contract, read from the package copy. Raises when it is missing or unreadable (FR-10)."""
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    import pydantic_core

    return pydantic_core.to_jsonable_python(value, fallback=str)


def call_args(kwargs: dict[str, Any], context_kwarg: Optional[str]) -> dict[str, Any]:
    """The arguments a tool was called with, JSON-able, without the SDK context argument (`None`-valued optional
    arguments kept: they are part of the payload the operator approves)."""
    return _jsonable({k: v for k, v in kwargs.items() if k != context_kwarg})


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def next_step(tool: str, args: Optional[dict] = None, why: str = "") -> dict[str, Any]:
    """One contract `next_step` item."""
    return {"tool": tool, "args": dict(args or {}), "why": _clip(why or f"call {tool}", WHY_MAX)}


# ── approval (FR-7) ──────────────────────────────────────────────────────────────────────────────────────────────


def _summary(tool: str, status: str, then: str, decide_in: str, recall: bool, tool_known: bool = True) -> str:
    where = _WHERE.get(decide_in, decide_in)
    poll = _STATUS_TOOL_OF.get(decide_in, "get_approval_status")
    if status == "pending":
        if then == "server_runs":
            return (f"{tool} waits for the operator in {where}; {decide_in} runs it once when approved. "
                    f"Poll {poll}; never re-submit.")
        if recall:
            return (f"{tool} waits for the operator in {where}; once approved, call it again with call_again_with "
                    "(the same arguments plus approval_request_id). Poll get_approval_status; never re-submit.")
        return f"{tool} waits for the operator in {where}. Poll {poll}; never re-submit."
    if status == "approved":
        if tool in NO_RECALL:
            return (f"The operator approved {tool}; nothing is called with approval_request_id. Retry the call that "
                    "was refused code_rule_not_approved.")
        if recall or then == "call_again_with":
            return (f"The operator approved {tool}; it runs when you call it again with the same arguments plus "
                    "approval_request_id (once).")
        return f"The operator approved {tool}; {decide_in} runs it once."
    return {
        "executing": f"The operator approved {tool}; it is running now. Poll {poll}.",
        "executed": f"The operator approved {tool} and it ran once; the approval is spent.",
        "failed": f"The operator approved {tool} and it failed when it ran; ask again only if it still needs doing.",
        "rejected": f"The operator rejected {tool}. Do not retry the same request.",
        "expired": (f"The request for {tool} expired unused. Call {tool} again to file a new one." if tool_known
                    # an app's action name is not a tool here (R3 review fix OR3-045b-4)
                    else f"The request for {tool} expired unused. Call the original tool again to file a new one."),
    }.get(status, f"{tool}: {status}.")


def _approval(tool: str, approval_id: Any, raw_status: Any, then: str, decide_in: str, *, value: dict,
              recall_args: Optional[dict] = None, action: Any = None, tool_known: bool = True) -> Optional[dict[str, Any]]:
    status = STATUS_MAP.get(str(raw_status)) if raw_status is not None else None
    if status is None or approval_id in (None, "") or not decide_in:
        return None
    out: dict[str, Any] = {"id": str(approval_id)[:128], "status": status, "decide_in": decide_in, "then": then}
    # call_again_with only where a re-call can still run it (pending or approved) and the tool takes the id: an agent
    # is never told to send an argument the tool would refuse (unknown-argument guard).
    recall = then == "call_again_with" and recall_args is not None and status in ("pending", "approved")
    out["summary"] = _clip(_summary(tool, status, then, decide_in, recall, tool_known), SUMMARY_MAX)
    if recall:
        out["call_again_with"] = {"tool": tool, "args": {**recall_args, "approval_request_id": str(approval_id)}}
    if isinstance(action, str) and action:
        out["action"] = action
    if isinstance(value.get("expires_at"), str):
        out["expires_at"] = value["expires_at"]
    reason = value.get("reason")
    if isinstance(reason, str) and reason and status in ("rejected", "failed", "expired"):
        out["reason"] = reason
    return out


def add_approval(name: str, row: Any, args: dict[str, Any], value: dict[str, Any], *,
                 accepts_id: bool = False, classes: Optional[dict] = None,
                 source_status: Optional[str] = None) -> dict[str, Any]:
    """`value` plus `approval` when it is a pending or decided approval (FTDD §7's recognition table); `value`
    unchanged otherwise, and ALWAYS unchanged when it already carries `approval` (a relay whose source adopted the
    contract keeps the source's object, A30; R1A-13). Every existing key keeps its name and value.

    `source_status`: the contract status of a source `approval` that `conform` could not keep. It wins over the legacy
    status, so a request the source calls `expired` is never rebuilt as `pending` from a legacy column that lags
    (miDataworks 022-D9; R2 review fix OR2-045-1)."""
    if "approval" in value:
        return value
    raw_status = source_status or value.get("status")
    built: Optional[dict[str, Any]] = None
    queue = STATUS_TOOLS.get(name)
    if queue == "miworkbench":
        stored = value.get("tool_name")
        tool = STORED_ACTIONS.get(stored, stored) if isinstance(stored, str) else None
        source = (classes or {}).get(tool) if tool else None
        if source is not None and source.approval:
            built = _approval(tool, value.get("approval_request_id"), raw_status, source.approval,
                              source.decide_in or "miworkbench", value=value, action=stored)
    elif queue is not None:
        approval_id = value.get("id") if queue == "midataworks" else value.get("approval_id")
        built = _approval(str(value.get("action") or name), approval_id or value.get("approval_id") or value.get("id"),
                          raw_status, "server_runs", queue, value=value, action=value.get("action"),
                          tool_known=False)  # the action (or this status tool) names no tool to call again
    elif row is not None and row.approval == "call_again_with" and "approval_request_id" in value:
        built = _approval(name, value.get("approval_request_id"), raw_status, "call_again_with",
                          row.decide_in or "miworkbench", value=value,
                          recall_args=args if accepts_id else None, action=name)
    elif row is not None and row.approval == "server_runs" and row.decide_in == "miserver":
        if value.get("decide_in") == "miserver" and value.get("approval_id"):
            built = _approval(name, value["approval_id"], raw_status, "server_runs", "miserver",
                              value=value, action=value.get("action"))
    elif row is not None and row.approval == "server_runs" and row.decide_in == "midataworks":
        if value.get("approval_id") and raw_status:
            built = _approval(name, value["approval_id"], raw_status, "server_runs", "midataworks",
                              value=value, action=value.get("action"))
    return value if built is None else {**value, "approval": built}


# ── next_steps (FR-8) ────────────────────────────────────────────────────────────────────────────────────────────


def from_legacy(legacy: Any) -> Optional[list[dict[str, Any]]]:
    """A legacy `next` list of `{tool, arguments, why[, needs]}` → `next_step` items (`args` = `arguments`; `needs`
    stays only in the legacy list). `None` for any other shape (a tool name or prose is converted at its site)."""
    if not isinstance(legacy, list) or not all(isinstance(i, dict) and isinstance(i.get("tool"), str) for i in legacy):
        return None
    return [next_step(i["tool"], i.get("arguments") if isinstance(i.get("arguments"), dict) else {},
                      str(i.get("why") or "")) for i in legacy]


def _callable_here(step: Any, registered: frozenset[str], accepts: Optional[dict[str, frozenset[str]]]) -> bool:
    """`step` names a tool this server registers and (when `accepts` is known) sends only arguments that tool
    declares: an agent is never told to make a call the unknown-argument guard refuses (OVN-b-2; OR1-045-1)."""
    if not isinstance(step, dict) or step.get("tool") not in registered:
        return False
    if accepts is None:
        return True
    args = step.get("args")
    return not isinstance(args, dict) or set(args) <= accepts.get(step["tool"], frozenset())


#: The id argument of each status tool, for the poll step a status answer that still waits must keep.
_POLL_ARG = {"get_approval_status": "approval_request_id", "miserver_get_approval_status": "approval_id",
             "dataworks_get_approval_status": "approval_id"}
#: Legacy statuses of a request that is still waiting or running (poll again).
_STILL_WAITING = frozenset({"pending", "pending_approval", "executing"})


def _keep_the_poll(name: str, value: dict[str, Any], kept: list, registered: frozenset[str],
                   accepts: Optional[dict[str, frozenset[str]]]) -> list:
    """After a status answer lost a step to the filter: when the request still waits and no kept step polls this
    status tool, put the poll back, so a source's bad step costs that step only, never the poll (R2 review fix
    OR2-045-3)."""
    if str(value.get("status")) not in _STILL_WAITING or any(s.get("tool") == name for s in kept):
        return kept
    arg = _POLL_ARG[name]
    approval_id = value.get(arg) or value.get("approval_request_id") or value.get("approval_id") or value.get("id")
    if not approval_id:
        return kept
    poll = next_step(name, {arg: str(approval_id)}, "the request still waits; poll again later and never re-submit")
    return [poll, *kept][:NEXT_STEPS_MAX] if _callable_here(poll, registered, accepts) else kept


def add_next_steps(value: dict[str, Any], name: str, registered: frozenset[str],
                   accepts: Optional[dict[str, frozenset[str]]] = None) -> dict[str, Any]:
    """`value` plus `next_steps` beside its legacy `next` (FTDD §8). The legacy `next` is NEVER modified.

    * an answer that already has `next_steps` keeps it, except on a relay or one of the three status tools, where
      items naming a tool this server does not register (or an argument that tool does not declare) are dropped
      (A30; R1A-13; R1 review fix OR1-045-1: a status tool named a stored action, `code_rule_approve`, and tools of
      disabled categories);
    * a legacy list converts generically, keeping only registered tools (a disabled category's tool is not here);
    * a legacy `next` of `None` → `[]`;
    * any other legacy `next` (prose, a bare name) must have been converted at its site: `STRICT` raises, else logged.
    At most five items (the contract's `maxItems`)."""
    if "next_steps" in value:
        steps = value["next_steps"]
        if (name.startswith(RELAY_PREFIXES) or name in STATUS_TOOLS) and isinstance(steps, list):
            kept = [s for s in steps if _callable_here(s, registered, accepts)][:NEXT_STEPS_MAX]
            if name in STATUS_TOOLS and len(kept) < len(steps):
                _report(name, "next_steps item")
                kept = _keep_the_poll(name, value, kept, registered, accepts)
            if kept != steps:
                return {**value, "next_steps": kept}
        return value
    if "next" not in value:
        return value
    legacy = value["next"]
    if legacy is None:
        return {**value, "next_steps": []}
    steps = from_legacy(legacy)
    if steps is None:
        if STRICT:
            raise UnconvertedNext(f"{name} answered a legacy `next` ({type(legacy).__name__}) with no next_steps")
        if name not in _REPORTED:
            _REPORTED.add(name)
            logger.warning("result language: %s answered a legacy `next` with no next_steps", name)
        return value
    return {**value, "next_steps": [s for s in steps if s["tool"] in registered][:NEXT_STEPS_MAX]}


# ── a backend's own `approval` / `next_steps` (R1 review fix OR1-045-5) ──────────────────────────────────────────


@functools.lru_cache(maxsize=2)
def _field_validator(key: str) -> Any:
    """A draft 2020-12 validator for the envelope's `key` property: the very schema the tool advertises (and the SDK
    client validates against), built the same way (`tool_classes.envelope_schema`)."""
    import jsonschema

    from .tool_classes import envelope_schema

    return jsonschema.Draft202012Validator(envelope_schema("object", load_contract())["properties"][key])


def _conforms(key: str, field: Any) -> bool:
    return _field_validator(key).is_valid(field)


def repair_approval(field: Any) -> Optional[dict[str, Any]]:
    """A source `approval` cut to what this vendored contract knows (R2 review fix OR2-045-1): unknown keys left out
    (a later minor version may add one: the contract's `approval` is `additionalProperties: false`), `id` and `summary`
    clipped to their limits, an optional key of the wrong shape left out. The source's own `status` is kept. `None`
    when even that does not validate (a required field missing or unusable)."""
    import jsonschema

    if not isinstance(field, dict):
        return None
    schema = _field_validator("approval").schema
    props = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    out = {k: v for k, v in field.items() if k in props}
    if isinstance(out.get("id"), str):
        out["id"] = out["id"][:128]
    if isinstance(out.get("summary"), str) and out["summary"].strip():
        out["summary"] = _clip(out["summary"], SUMMARY_MAX)
    for key in [k for k in out if k not in required]:
        if not jsonschema.Draft202012Validator(props[key]).is_valid(out[key]):
            del out[key]
    return out if _conforms("approval", out) else None


def contract_statuses() -> frozenset[str]:
    """The contract's `approval_status` values."""
    return frozenset(load_contract()["$defs"]["approval_status"]["enum"])


def conform(name: str, value: dict[str, Any]) -> dict[str, Any]:
    """`value` with a top-level `approval` or `next_steps` that the advertised schema would refuse repaired or taken
    out, so a backend answering `approval: null` or a `next_steps` of another shape costs that one field, never the
    whole answer (the SDK client raises on a non-conforming structured answer). An `approval` object is REPAIRED first
    (`repair_approval`: its known fields, its own status) and dropped only when that fails; `speak` then refuses a
    gated, relayed or status answer left with no `approval` (OR2-045-1). A `next_steps` list keeps its conforming
    items. Logged once per tool and key; `value` unchanged when both conform."""
    out = value
    if "approval" in out and not _conforms("approval", out["approval"]):
        repaired = repair_approval(out["approval"])
        out = ({**out, "approval": repaired} if repaired is not None
               else {k: v for k, v in out.items() if k != "approval"})
        _report(name, "approval")
    if "next_steps" in out and not _conforms("next_steps", out["next_steps"]):
        steps = out["next_steps"]
        item = _field_validator("next_steps").schema.get("items") or {}
        import jsonschema

        check = jsonschema.Draft202012Validator(item)
        kept = [s for s in steps if check.is_valid(s)][:NEXT_STEPS_MAX] if isinstance(steps, list) else None
        out = ({**out, "next_steps": kept} if kept is not None
               else {k: v for k, v in out.items() if k != "next_steps"})
        _report(name, "next_steps")
    return out


def _report(name: str, key: str) -> None:
    if (name, key) not in _REPORTED:
        _REPORTED.add((name, key))
        logger.warning("result language: %s answered a %s the contract refuses; it was left out", name, key)


# ── the wrapper (build_server) ───────────────────────────────────────────────────────────────────────────────────


def speak(name: str, row: Any, kwargs: dict[str, Any], context_kwarg: Optional[str], value: Any, *,
          accepts_id: bool, registered: frozenset[str], classes: dict,
          accepts: Optional[dict[str, frozenset[str]]] = None) -> Any:
    """One answer in the result language; anything but a dict passes through. The call's arguments are made JSON-able
    only for a re-call row whose tool takes the id (the one place they are used): a 16 MiB batch upload is never
    copied for nothing."""
    if not isinstance(value, dict):
        return value
    source = value.get("approval")
    value = conform(name, value)
    dropped = source is not None and "approval" not in value
    claimed = source.get("status") if dropped and isinstance(source, dict) else None
    # A status this vendored contract does not know (a later minor version's) is never replaced by the legacy column,
    # which may lag: the answer is refused instead (R3 review fix OR3-045b-2).
    foreign = isinstance(claimed, str) and claimed not in contract_statuses()
    source_status = claimed if isinstance(claimed, str) and not foreign else None
    # The refusal's three branches: a gated row; a relay; a status tool. The status branch is defence in depth: today
    # the two relayed status tools are relays, and `get_approval_status` builds its own dict, so a source `approval`
    # never reaches here through it alone (review-045-a0r2-br3 S7).
    must_carry = ((row is not None and row.approval) or name.startswith(RELAY_PREFIXES) or name in STATUS_TOOLS)
    if dropped and must_carry and foreign:
        raise unreadable_approval(name, row, source, value, registered, accepts)
    recall = accepts_id and row is not None and row.approval == "call_again_with" and "approval_request_id" in value
    args = call_args(kwargs, context_kwarg) if recall else {}
    value = add_approval(name, row, args, value, accepts_id=accepts_id, classes=classes, source_status=source_status)
    if dropped and "approval" not in value and must_carry:
        raise unreadable_approval(name, row, source, value, registered, accepts)
    value = _no_poll_once_decided(name, value)
    value = add_next_steps(value, name, registered, accepts)
    return add_poll_step(name, value, registered, accepts)


#: Contract statuses of a request that still waits for its operator or is running: its answer names the poll.
_POLL_STATUSES = frozenset({"pending", "executing"})


def add_poll_step(name: str, value: dict[str, Any], registered: frozenset[str],
                  accepts: Optional[dict[str, frozenset[str]]] = None) -> dict[str, Any]:
    """`value` with the deciding app's status tool FIRST in `next_steps`, with the approval's id, when a gated tool's
    answer carries an `approval` that still waits (045 follow-up, 2026-11-24; miDataworks 022's shape,
    `{tool: dataworks_get_approval_status, args: {approval_id}, why}`). It follows the `approval` object, not the
    row: every gated row (`TOOL_CLASSES` with an `approval`) answers one, and a relay whose source app gates a call
    this table does not yet know still names its poll (control P8, 2026-11-24). The three status tools keep their own
    poll (`_keep_the_poll`). No step when the status tool is not registered on this server (its category is off) or
    would refuse the argument, or when a step already names it; at most five steps (the contract's `maxItems`)."""
    if name in STATUS_TOOLS:
        return value
    approval = value.get("approval")
    if not isinstance(approval, dict) or approval.get("status") not in _POLL_STATUSES or not approval.get("id"):
        return value
    poll = _STATUS_TOOL_OF.get(str(approval.get("decide_in")))
    if poll is None:
        return value
    steps = value.get("next_steps") if isinstance(value.get("next_steps"), list) else []
    if any(isinstance(s, dict) and s.get("tool") == poll for s in steps):
        return value
    where = _WHERE.get(approval["decide_in"], approval["decide_in"])
    step = next_step(poll, {_POLL_ARG[poll]: str(approval["id"])},
                     f"the request waits for the operator in {where}; poll until it is decided and never re-submit")
    if not _callable_here(step, registered, accepts):
        return value
    return {**value, "next_steps": [step, *steps][:NEXT_STEPS_MAX]}


#: Contract statuses with nothing left to poll.
_DECIDED = frozenset({"executed", "failed", "rejected", "expired"})


def _no_poll_once_decided(name: str, value: dict[str, Any]) -> dict[str, Any]:
    """A status answer whose `approval` (the source's shared status) is decided while its legacy status still waits:
    the site derived its poll step from the lagging legacy column, so that step is taken out; any other step stays
    (A30). R3 review fix OR3-045b-4."""
    approval = value.get("approval")
    if name not in STATUS_TOOLS or not isinstance(approval, dict) or approval.get("status") not in _DECIDED:
        return value
    if STATUS_MAP.get(str(value.get("status"))) == approval["status"] or not isinstance(value.get("next_steps"), list):
        return value
    kept = [s for s in value["next_steps"] if not (isinstance(s, dict) and s.get("tool") == name)]
    return value if len(kept) == len(value["next_steps"]) else {**value, "next_steps": kept}


def unreadable_approval(name: str, row: Any = None, source: Any = None, value: Optional[dict] = None,
                        registered: frozenset[str] = frozenset(),
                        accepts: Optional[dict[str, frozenset[str]]] = None) -> Exception:
    """The error for a gated, relayed or status answer whose source `approval` could neither be kept nor rebuilt: it
    must never read as a plain, finished result (R2 review fix OR2-045-1). It names the approval id the answer carried
    and, when the deciding app's status tool is registered here (and this is not that tool), a poll step for line 3
    (R3 review fix OR3-045b-3)."""
    from mcp.server.mcpserver.exceptions import ToolError

    value = value or {}
    src = source if isinstance(source, dict) else {}
    approval_id = next((v for v in (src.get("id"), value.get("approval_id"), value.get("approval_request_id"),
                                    value.get("id")) if isinstance(v, (str, int)) and str(v)), None)
    queue = (src.get("decide_in") if src.get("decide_in") in _STATUS_TOOL_OF
             else getattr(row, "decide_in", None) if getattr(row, "decide_in", None) in _STATUS_TOOL_OF
             else "midataworks" if name.startswith("dataworks_")
             else "miserver" if name.startswith("miserver_") else "miworkbench")
    steps: list[dict[str, Any]] = []
    poll = _STATUS_TOOL_OF[queue]
    if approval_id is not None and name not in STATUS_TOOLS:
        step = next_step(poll, {_POLL_ARG[poll]: str(approval_id)},
                         "read the request where it is decided; nothing here confirms it waits, ran or was refused")
        if _callable_here(step, registered, accepts):
            steps.append(step)
    which = f" (approval {approval_id})" if approval_id is not None else ""
    error = ToolError(f"{name}: the app answered an approval{which} this server cannot read, so nothing here says "
                      "whether the action waits, ran or was refused.")
    error.code = "approval_unreadable"  # type: ignore[attr-defined]
    error.next_steps = steps  # type: ignore[attr-defined]
    return error


def speak_result_language(mcp: Any) -> int:
    """Wrap every registered tool's `fn` so its dict answer gains `approval` and `next_steps` (FR-7, FR-8). Called
    by `build_server` after `apply_tool_classes` (every tool has a row) and BEFORE `normalise_results` (FTDD §2).
    Reads the vendored contract first, so a missing package copy fails the build (FR-10). Returns the count."""
    from .tool_classes import TOOL_CLASSES

    load_contract()
    manager = mcp._tool_manager  # noqa: SLF001 — no public hook for this in the SDK (as server.py)
    tools = manager.list_tools()
    registered = frozenset(t.name for t in tools)
    accepts = {t.name: frozenset(((t.parameters or {}).get("properties") or {})) for t in tools}
    count = 0
    for tool in tools:
        original = tool.fn
        if getattr(original, "speaks_result_language", False):
            continue
        bound = dict(name=tool.name, row=TOOL_CLASSES.get(tool.name), ctx=tool.context_kwarg,
                     accepts_id="approval_request_id" in ((tool.parameters or {}).get("properties") or {}))

        if inspect.iscoroutinefunction(original):
            @functools.wraps(original)
            async def wrapper(*args, _fn=original, _b=bound, **kwargs):
                value = await _fn(*args, **kwargs)
                return speak(_b["name"], _b["row"], kwargs, _b["ctx"], value, accepts_id=_b["accepts_id"],
                             registered=registered, classes=TOOL_CLASSES, accepts=accepts)
        else:
            @functools.wraps(original)
            def wrapper(*args, _fn=original, _b=bound, **kwargs):  # type: ignore[misc]
                value = _fn(*args, **kwargs)
                return speak(_b["name"], _b["row"], kwargs, _b["ctx"], value, accepts_id=_b["accepts_id"],
                             registered=registered, classes=TOOL_CLASSES, accepts=accepts)
        wrapper.speaks_result_language = True  # type: ignore[attr-defined]
        tool.fn = wrapper
        count += 1
    return count


# ── errors that teach (FR-9) ─────────────────────────────────────────────────────────────────────────────────────

#: Each code family's fix (FTID I45-D3). `backend`, `miserver` and `dataworks` are the families of
#: `<product>_<status>[_<code>]`; every code an error can carry has a row here (`test_every_code_has_a_fix`).
ERROR_CODES: dict[str, str] = {
    "unknown_arguments": "send only the arguments the tool declares (listed below); a field of an object parameter "
                         "goes inside that parameter.",
    "invalid_arguments": "correct the named argument to the type and range the tool's input schema gives, then call "
                         "again.",
    "steering_guardrail": "poll get_steering_result for a task in flight (that releases its slot) or cancel one with "
                          "cancel_steering_task, then call again.",
    "backend": "read miWorkbench's detail below and correct the request or the state it names before calling again.",
    "miserver": "read miServer's detail below and correct the request or the state it names before calling again.",
    "dataworks": "read miDataworks' detail below and correct the request or the state it names before calling again.",
    "tool_error": "read the message below and correct the call.",
    "internal_error": "the server failed unexpectedly; the operator can read its log. Report it; do not retry in a "
                      "loop.",
    "approval_unreadable": "treat the action as NOT done and do not re-submit it: read the request with the next "
                           "step below, or ask the operator in the deciding app when there is none; report the "
                           "mismatch.",
}

#: Who refused, for line 1's message.
_PRODUCT_LABEL = {"backend": "miWorkbench", "miserver": "miServer", "dataworks": "miDataworks"}

_NOT_CODE = re.compile(r"[^a-z0-9_]+")
#: The contract's `error_code` length limit.
CODE_MAX = 64


def code_part(raw: Any) -> str:
    """A backend's own code for line 1: lower-cased, every non-`[a-z0-9_]` run becomes `_` (FTID §8)."""
    return _NOT_CODE.sub("_", str(raw).lower()).strip("_")


def _causes(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__  # the explicit chain only: an error raised while handling another is not caused by it


def _backend_code(detail: Any) -> Optional[str]:
    if not isinstance(detail, dict):
        return None
    if isinstance(detail.get("code"), str):
        return detail["code"]
    inner = detail.get("error")
    if isinstance(inner, dict) and isinstance(inner.get("code"), str):
        return inner["code"]
    if isinstance(inner, str):
        return inner
    inner = detail.get("detail")
    if isinstance(inner, dict) and isinstance(inner.get("code"), str):
        return inner["code"]
    return None


def classify(exc: BaseException) -> str:
    """The contract `error_code` for a tool's error (FTDD §9): a pydantic `ValidationError` cause →
    `invalid_arguments`; a `BackendError` in the cause chain → `<product>_<status>[_<code>]`; a `ToolError` carrying a
    known `code` (the steering guardrail) → that code; else `tool_error`."""
    from pydantic import ValidationError

    from .client import BackendError

    if isinstance(exc.__cause__, ValidationError):
        return "invalid_arguments"
    for cause in _causes(exc):
        if isinstance(cause, BackendError):
            product = getattr(cause, "product", "backend") or "backend"
            status = "unreachable" if cause.status_code == 0 else str(cause.status_code)
            code = f"{product}_{status}"
            own = _backend_code(cause.detail)
            if own:
                part = code_part(own)
                if part:
                    code = f"{code}_{part}"
            return code[:CODE_MAX].rstrip("_")
        own = getattr(cause, "code", None)
        if isinstance(own, str) and own in ERROR_CODES:
            return own
    return "tool_error"


def family(code: str) -> str:
    """The `ERROR_CODES` row a code reads: `backend_409_x` → `backend`."""
    if code in ERROR_CODES:
        return code
    head = code.split("_", 1)[0]
    return head if head in ERROR_CODES else "tool_error"


def error_next_steps(exc: Optional[BaseException], registered: frozenset[str]) -> list[dict[str, Any]]:
    """A product error's own hint (`next` in its detail: a tool name, or a list of `{tool, arguments, why}`), as
    `next_step` items naming registered tools only; `[]` when there is none."""
    from .client import BackendError

    for cause in _causes(exc) if exc is not None else ():
        own = getattr(cause, "next_steps", None)  # an error this server raised with its own steps (OR3-045b-3)
        if isinstance(own, list):
            return [s for s in own if isinstance(s, dict) and s.get("tool") in registered][:NEXT_STEPS_MAX]
        if isinstance(cause, BackendError) and isinstance(cause.detail, dict):
            detail = cause.detail
            hint = detail.get("next")
            for inner in ("error", "detail"):  # an OpenAI-style `{"error": {…}}` or FastAPI's `{"detail": {…}}`
                if hint is None and isinstance(detail.get(inner), dict):
                    hint = detail[inner].get("next")
            if isinstance(hint, str) and hint in registered:
                return [next_step(hint, {}, f"{_PRODUCT_LABEL.get(cause.product, cause.product)}'s hint for this "
                                            "error")]
            steps = from_legacy(hint) or []
            return [s for s in steps if s["tool"] in registered][:NEXT_STEPS_MAX]
    return []


def teach(code: str, tool: str, original: str, next_steps: Optional[list] = None) -> str:
    """The contract's error text: `<code>: <message>`, `fix: …`, `next: <JSON list>`, then the original unchanged."""
    fam = family(code)
    if fam in _PRODUCT_LABEL:
        message = f"{_PRODUCT_LABEL[fam]} refused {tool}."
    else:
        message = {
            "unknown_arguments": f"{tool} was called with arguments it does not declare; nothing was run.",
            "invalid_arguments": f"{tool}'s arguments failed validation; nothing was run.",
            "steering_guardrail": f"{tool} was refused by the steering concurrency guardrail.",
            "internal_error": f"{tool} failed unexpectedly.",
            "approval_unreadable": f"{tool}'s approval could not be read; nothing is confirmed done.",
        }.get(fam, f"{tool} returned an error.")
    steps = json.dumps(list(next_steps or []), separators=(", ", ": "))
    return f"{code}: {message}\nfix: {ERROR_CODES[fam]}\nnext: {steps}\n{original}"


__all__ = [
    "ERROR_CODES", "STATUS_MAP", "STATUS_TOOLS", "add_approval", "add_next_steps", "add_poll_step", "call_args",
    "classify",
    "error_next_steps", "load_contract", "next_step", "speak_result_language", "teach",
]
