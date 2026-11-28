"""Generate the normative MCP contract from the LIVE tool registry.

WHY GENERATED
-------------
miServer's `docs/mcp-contract.md` is hand-maintained, and for an entire increment
it carried ✅ marks for 16 tools that were never registered with the server.
miWorkbench had no contract at all: 32 `millm_*` rows existed there, zero rows for  (p9: keep, rename history)
the 58 native tools.

Writing a second hand-maintained document would repeat a pattern that has now
failed four times in this codebase (the copy audit's SURFACES list, the
reachability harness's category list, the contract row scraper, and the
hand-written tool map). So this derives the document, and a test regenerates it
and diffs — a stale contract fails the build instead of quietly lying.

WHAT IT IS NOT
--------------
Not a replacement for docstrings. This is the inventory an operator or reviewer
reads; an AGENT should call `miworkbench_howto` instead, which carries the
workflows and the failure modes a table cannot express.
"""

import ast
import inspect
from typing import Any

HEADER = """<!-- GENERATED FILE — DO NOT EDIT BY HAND.

Regenerate with:
    python -c "from src.mcp_server.contract import write_contract; write_contract()"

`backend/tests/unit/test_mcp_contract_generated.py` regenerates this and fails
if it differs from what is committed, so an edit here is reverted by the next
run rather than silently kept.
-->

# miWorkbench MCP contract

Every tool the miWorkbench MCP server registers, its category, and the backend
endpoint it calls. Derived from the live registry — see
`src/mcp_server/contract.py`.

**Agents: call `miworkbench_howto` instead of reading this.** A table cannot carry
ordering constraints, GPU-lock contention, id namespaces, or the failure modes
that mislead. This document is the inventory; that tool is the guidance.

Categories are gated by `MCP_TOOL_CATEGORIES`. The `miserver_*` categories also
require `MISERVER_API_URL`, the `dataworks` category `MIDATAWORKS_API_URL`; none of them is enabled by default.
The `millm_*` tool and category names were renamed `miserver_*` at ENH-009 P3 <!-- p9: keep, rename history -->
(`mistudio_howto` → `miworkbench_howto`); since P7 an old category key is an unknown category. <!-- p9: keep, rename history -->
"""


def _endpoints(tool: str) -> list[str]:
    """HTTP calls a tool issues, as `METHOD /path`, from the one route walker (`route_map`, 046 WT1).

    Until 046 this module had its own walker (`_endpoints_for`), which read only `_submit*` helpers and differed from
    the parity guards' on others; a route one walker saw could be invisible to the other. An f-string parameter is
    shown as `{…}`, as the contract always has."""
    from .route_map import tool_routes

    return sorted({f"{method} {path.replace('{}', '{…}')}" for _product, method, path in tool_routes().get(tool, ())})


def _row(name: str, summary: str, endpoints: list[str]) -> dict[str, Any]:
    """One contract row; title, kind, approval and ⚠️ come from `TOOL_CLASSES` (045 FR-3, FR-11).

    Until 045 the ⚠️ came from docstring words ("DESTRUCTIVE", "permanently", …), which flagged 9 of the 16 tools that
    delete or clear and none of the 21 cancels. The table is the one authority; `apply_tool_classes` refuses to build
    a server with a tool that has no row, so a missing row cannot reach this function."""
    from .tool_classes import TOOL_CLASSES, load_output_schemas

    row = TOOL_CLASSES[name]
    return {"name": name, "summary": summary, "endpoints": endpoints, "title": row.title, "kind": row.kind,
            "approval": row.approval, "decide_in": row.decide_in,
            "destructive": row.kind == "destructive",
            # 045 follow-up (RL-1): the tool advertises its route's fields, not only the generic envelope.
            "typed": name in load_output_schemas()}


def collect() -> dict[str, list[dict[str, Any]]]:
    """`{category: [{name, summary, endpoints, title, kind, approval, decide_in, destructive}]}` from the registry."""
    from .tools import all_categories

    # The REGISTRY is the authority on what exists; the AST pass only enriches
    # rows with endpoints, which `list_tools()` cannot report. An earlier
    # version derived membership from the AST alone and an adversarial pass
    # walked straight through it.
    from .tools.howto import _all_tools

    index = _all_tools()

    out: dict[str, list[dict[str, Any]]] = {}
    for category, modules in all_categories().items():
        served = {n: d for n, d in index.get(category, [])}
        seen_in_source: set[str] = set()
        rows: list[dict[str, Any]] = []
        for module in modules:
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    continue
                is_tool = any(
                    isinstance(d.func if isinstance(d, ast.Call) else d, ast.Attribute)
                    and (d.func if isinstance(d, ast.Call) else d).attr == "tool"
                    for d in node.decorator_list
                )
                if not is_tool:
                    continue
                doc = ast.get_docstring(node) or ""
                summary = " ".join(doc.split())
                cut = summary.find(". ")
                if 0 < cut < 150:
                    summary = summary[: cut + 1]
                if node.name not in served:
                    # Registered-by-helper tools are invisible to this AST
                    # pass; they are picked up from `served` below. A tool in
                    # the source but NOT served (conditional registration) is
                    # correctly omitted — the contract records what an agent
                    # can actually call.
                    continue
                seen_in_source.add(node.name)
                rows.append(_row(node.name, summary[:160], _endpoints(node.name)))
        # Anything the server serves that the AST could not see still gets a
        # row — without an endpoint, but never omitted.
        for name, summary in served.items():
            if name not in seen_in_source:
                # 045 FR-3: the table answers for these too, so a helper-registered delete can no longer be
                # reported `destructive: False` by default.
                rows.append(_row(name, summary, []))
        if rows:
            out[category] = sorted(rows, key=lambda r: r["name"])
    return out


def render() -> str:
    """The full contract document."""
    index = collect()
    total = sum(len(v) for v in index.values())
    typed = sum(1 for rows in index.values() for r in rows if r["typed"])
    lines = [HEADER, f"\n**{total} tools across {len(index)} categories.**\n",
             f"**{typed} of {total} tools advertise a typed output schema** (their routes' answers, from each "
             "route's answer contract); the rest advertise the generic envelope. Each untyped tool and its reason, "
             "from a closed list, are in `backend/src/mcp_server/output_schemas.json` (`untyped`, with `coverage` "
             "and `sources`), generated by `backend/scripts/generate_output_schemas.py`.\n"]
    for category in sorted(index):
        rows = index[category]
        lines.append(f"\n## `{category}` ({len(rows)} tools)\n")
        lines.append("| Tool | Title | R/W/D | Approval | Typed | Endpoint | Summary |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            eps = "<br>".join(f"`{e}`" for e in r["endpoints"]) or "—"
            name = f"`{r['name']}`"
            if r["destructive"]:
                name += " ⚠️"
            summary = r["summary"].replace("|", "\\|")
            kind = {"read": "R", "write": "W", "destructive": "D"}[r["kind"]]
            approval = "—" if not r["approval"] else (
                r["approval"] + (f" ({r['decide_in']})" if r["decide_in"] else ""))
            typed_cell = "yes" if r["typed"] else "no"
            lines.append(f"| {name} | {r['title']} | {kind} | {approval} | {typed_cell} | {eps} | {summary} |")
    lines.append(
        "\n⚠️ = destructive: deletes, purges, clears or cancels (CHAIN_PLAN A13). R/W/D = read-only, write, "
        "destructive, from `src/mcp_server/tool_classes.py`, the same table that sets each tool's `readOnlyHint` and "
        "`destructiveHint`. Approval: `call_again_with` = re-call with the approval id once the operator approves; "
        "`server_runs` = the named app runs the request itself once approved. Typed: the tool's `outputSchema` "
        "carries its routes' answers from their contracts (`required` = always sent; extra keys allowed), as "
        "alternatives beside the shared pending, unavailable, refused and error answers and the result language's "
        "`approval` and `next_steps`.\n")
    return "\n".join(lines)


def contract_path():
    from pathlib import Path

    return Path(__file__).resolve().parents[3] / "docs" / "mcp-contract.md"


def write_contract() -> str:
    path = contract_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render()
    path.write_text(text)
    return str(path)
