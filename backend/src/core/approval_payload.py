"""The ONE canonical approval payload (045 FR-13; CHAIN_PLAN A29, A38; FTDD §16.3).

Every filer builds the payload an operator approves with this function: the REST gate (`core.approval_gate`) from the
route's own request, and the MCP filers (`_approval_gate`'s callers, steering's `_submit`) from the arguments they send
to that route. So an approval filed by an agent over REST and one filed by the MCP pod for the same call are the same
object, and the consume compares like with like.

The payload is FLAT — the route's path parameters, its parsed query values and the raw JSON body's keys at the top level
(a non-object body keyed by the handler's body parameter name) — plus two reserved keys, `_method` and `_route` (the
route's path template), so an approval binds the route it was filed for (A38): an approval filed on
`POST /labeling/panel` is a `422 mismatch` on `POST /labeling`, although both are the action `start_labeling`.

A LEAF module: the standard library only, so the MCP pod (no database, no `Settings`) may import it
(`test_mcp_tools_stay_on_http.py::STDLIB_LEAVES`). Published as one vector per gated route in
`docs/schemas/miworkbench-approval-payload-vectors-v1.json`, which miAssist vendors.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

#: Keys the payload reserves for the route binding. A route whose parameters use either name cannot be gated
#: (`test_approval_payload.py::test_no_route_has_colliding_names`).
RESERVED_KEYS = frozenset({"_method", "_route"})

#: The key an id may travel under in a body (044's routes); never part of what is approved.
APPROVAL_ID_KEY = "approval_request_id"

#: The header an approved re-call carries its id in (A37). Here, in the leaf, so the MCP pod's steer tools send the
#: very name the REST gate (`core.approval_gate`) reads.
APPROVAL_REQUEST_HEADER = "X-MiStudio-Approval-Request"  # p9: keep, a 045 wire header fixed by CHAIN_PLAN A37 and pinned by miAssist


class CollidingPayloadKey(ValueError):
    """A body key that names a reserved key, a path parameter or a query value of the same call."""

    def __init__(self, key: str):
        super().__init__(f"the body key {key!r} collides with the route binding, a path parameter or a query value")
        self.key = key


def canonical_payload(method: str, route_template: str, path_params: Mapping[str, Any], query: Mapping[str, Any],
                      body: Any, *, body_param: Optional[str] = None) -> dict[str, Any]:
    """The flat payload an operator approves for one call of one gated route.

    R1 review fix (OR1-045a0-10), 2026-11-16: a body key never overwrites `_method`, `_route`, a path parameter or a
    query value. Such a key is REFUSED (`CollidingPayloadKey`), because letting either side win would make the card
    show one value while the handler reads the other."""
    out: dict[str, Any] = {"_method": method.upper(), "_route": route_template, **dict(path_params), **dict(query)}
    if isinstance(body, dict):
        clash = sorted(k for k in body if k in out and k != APPROVAL_ID_KEY)
        if clash:
            raise CollidingPayloadKey(clash[0])
        out.update(body)
    elif body is not None:
        if not body_param:
            raise ValueError(f"a non-object body on {method.upper()} {route_template} needs its parameter name")
        out[body_param] = body
    out.pop(APPROVAL_ID_KEY, None)
    return out
