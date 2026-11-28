"""Which HTTP route each MCP tool calls, read from the tool modules' source (046 WT1: the one route walker).

Until 046 there were three walkers: this one (then `tests/support/mcp_route_map.py`, feeding the route-parity
guards), `contract.py`'s `_endpoints_for` (the generated `docs/mcp-contract.md`) and the output-schema generator's
`routes_returned`. They disagreed on cases, so a route one of them saw could be invisible to another. Every
consumer now reads this module: the contract, the generator, the answer contract's coverage and the parity guards.

MEMBERSHIP COMES FROM THE REGISTRY, NEVER FROM THIS WALK. `tool_routes()` names every `@mcp.tool` function in
`tools/`, whether or not a server registers it; a consumer joins it to the LIVE registry (`list_tools()` of a built
server: `tests/support/live_registry.registered_tools`, `scripts.generate_output_schemas.live_tools`,
`contract.collect`'s `_all_tools`). The walk only ENRICHES a registered tool with the routes it calls, which
`list_tools()` cannot report. A tool whose registration line is deleted therefore drops out of the covered set and
its routes go uncovered, which is the red the parity guards exist for (memory: derive-from-the-registry-not-the-source,
source-scrape-guards-fail-open).

What the walk reads, per `@mcp.tool` function:

* `client.<verb>("/path")`, `miserver.<verb>(...)`, `dw.<verb>(...)` and `ctx.miworkbench.<verb>(...)`
  with a literal or f-string path — `{…}` parts become `{}`;
* calls to a helper defined in the same module, followed transitively, including a helper that
  receives the path as an argument (`_submit("steer_sweep", "/steering/async/sweep", body)`).

`raw_get` is a GET. Calls whose path the walk cannot read are reported by `unreadable_calls()`,
which the guards require to be empty — an unread call is a route the guard cannot see.

`returns_unchanged(fn)` is the generator's return classifier (moved here from `routes_returned`): a tool answers a
route's body unchanged only when EVERY `return` of its own body is `return await <receiver>.<verb>("<path>", ...)`.

Standard library only: the MCP pod imports this package and must not import the backend (memory
`mcp-server-is-a-separate-deployment`; `test_mcp_tools_stay_on_http.py`).
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

TOOLS = Path(__file__).parent / "tools"

VERBS = {"get", "post", "put", "patch", "delete", "raw_get"}

#: receiver name → the product the call goes to. `forge` is miForge's run-list client (`gpu_ops`, `wsp`).
RECEIVERS = {"client": "miworkbench", "miserver": "miserver", "dw": "dataworks", "forge": "miforge"}


def normalise(path: str) -> str:
    """`/a/{x}/b?q=1` → `/a/{}/b`; a `{repo_id:path}` converter is a parameter like any other."""
    return re.sub(r"\{[^}]*\}", "{}", path.split("?")[0]).rstrip("/") or "/"


def _product(receiver: ast.AST) -> str | None:
    if isinstance(receiver, ast.Name):
        return RECEIVERS.get(receiver.id)
    if isinstance(receiver, ast.Attribute) and receiver.attr == "miworkbench":
        return "miworkbench"
    return None


def _path_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(part.value) if isinstance(part, ast.Constant) else "{}" for part in node.values
        )
    return None


def _is_tool(fn: ast.AST) -> bool:
    for dec in getattr(fn, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr == "tool":
            return True
    return False


Route = tuple[str, str, str]  # (product, METHOD, path as written)


def _module_routes(path: Path) -> tuple[dict[str, set[Route]], list[str]]:  # noqa: C901
    tree = ast.parse(path.read_text())
    funcs = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    unreadable: set[str] = set()

    def direct(fn: Any) -> tuple[set[Route], set[tuple[str, str, str]], list[tuple[str, ast.Call]]]:
        routes: set[Route] = set()
        by_param: set[tuple[str, str, str]] = set()  # (product, METHOD, parameter name)
        helpers: list[tuple[str, ast.Call]] = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in VERBS:
                product = _product(func.value)
                if product is None and node.args:
                    # review round 1, finding 5: a verb with a route path on a receiver this walk does not know is
                    # a route nobody would see; it is reported, never silently skipped.
                    literal = _path_of(node.args[0])
                    if literal is not None and literal.startswith("/"):
                        unreadable.add(f"{path.name}:{node.lineno} (receiver {ast.unparse(func.value)})")
                if product and node.args:
                    method = "GET" if func.attr == "raw_get" else func.attr.upper()
                    literal = _path_of(node.args[0])
                    if literal is not None and literal.startswith("/"):
                        routes.add((product, method, literal))
                    elif isinstance(node.args[0], ast.Name):
                        by_param.add((product, method, node.args[0].id))
                    else:
                        unreadable.add(f"{path.name}:{node.lineno}")
            if isinstance(func, ast.Name) and func.id in funcs and funcs[func.id] is not fn:
                helpers.append((func.id, node))
        return routes, by_param, helpers

    def resolve(fn: Any, seen: frozenset[str]) -> set[Route]:
        routes, _, helpers = direct(fn)
        for name, call in helpers:
            if name in seen:
                continue
            helper = funcs[name]
            routes |= resolve(helper, seen | {name})
            _, by_param, _ = direct(helper)
            params = [a.arg for a in helper.args.args]
            for product, method, param in by_param:
                if param not in params:
                    unreadable.add(f"{path.name}:{helper.lineno} ({param})")
                    continue
                values = []
                index = params.index(param)
                if index < len(call.args):
                    values.append(call.args[index])
                values += [kw.value for kw in call.keywords if kw.arg == param]
                for value in values:
                    literal = _path_of(value)
                    if literal is not None and literal.startswith("/"):
                        routes.add((product, method, literal))
        return routes

    out = {}
    for name, fn in funcs.items():
        if isinstance(fn, ast.AsyncFunctionDef) and _is_tool(fn):
            out[name] = resolve(fn, frozenset({name}))
            # A tool's OWN call with a variable path has no call site to resolve it from.
            for _product_name, _method, param in direct(fn)[1]:
                unreadable.add(f"{path.name}:{fn.lineno} ({param})")
    return out, sorted(unreadable)


@lru_cache(maxsize=1)
def tool_routes() -> dict[str, frozenset[Route]]:
    """`{tool function name: {(product, METHOD, path)}}` for every `@mcp.tool` in tools/."""
    out: dict[str, frozenset[Route]] = {}
    for module in sorted(TOOLS.glob("*.py")):
        routes, _ = _module_routes(module)
        for name, found in routes.items():
            assert name not in out, f"tool function {name} is defined twice"
            out[name] = frozenset(found)
    return out


def unreadable_calls() -> list[str]:
    """Client calls whose path is neither a literal, an f-string nor a helper parameter."""
    found: list[str] = []
    for module in sorted(TOOLS.glob("*.py")):
        found += _module_routes(module)[1]
    return found


# ── which tools answer one route's body unchanged (moved from the generator's `routes_returned`) ─────────────────


def _own_returns(fn: ast.AST) -> list[ast.Return]:
    """The `return` statements of `fn` itself, not of a function or lambda nested in it."""
    found: list[ast.Return] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return):
                found.append(child)
            walk(child)

    walk(fn)
    return found


def _route_of(ret: ast.Return) -> Optional[Route]:
    """`(product, METHOD, normalised path)` when `ret` is `return await <receiver>.<verb>("<path>", ...)`.

    `full=True` makes a miServer verb answer a `MiServerResponse` (status, body, headers), not the body, so such a
    return is not the route's body unchanged."""
    value = ret.value
    if not (isinstance(value, ast.Await) and isinstance(value.value, ast.Call)):
        return None
    call = value.value
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr in VERBS) or not call.args:
        return None
    product = _product(func.value)
    if product is None:
        return None
    if any(kw.arg == "full" for kw in call.keywords):
        return None
    path = _path_of(call.args[0])
    if path is None or not path.startswith("/"):
        return None
    method = "GET" if func.attr == "raw_get" else func.attr.upper()
    return product, method, normalise(path)


def _function_node(fn: Any) -> ast.AST:
    original = inspect.unwrap(fn)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    node = tree.body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)), node
    return node


def returns_unchanged(fn: Any) -> tuple[Optional[set[Route]], str]:
    """The routes a tool answers unchanged (`{(product, METHOD, normalised path)}`), or `None` and why it cannot be
    typed from one route's body."""
    returns = _own_returns(_function_node(fn))
    if not returns:
        return None, "the tool returns no route's body"
    routes = {_route_of(r) for r in returns}
    if None in routes:
        return None, "the tool builds or changes its answer (not one route's body unchanged)"
    return routes, ""  # type: ignore[return-value]
