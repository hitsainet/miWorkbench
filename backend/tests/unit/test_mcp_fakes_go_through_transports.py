"""Every MCP test's backend fake answers through a transport, where the fake check judges it (046 FR-046.24, WT3;
FTASKS 7.6, FTDD §6.4, FTID §8.3).

A mock bound to a client verb hands a tool its answer without crossing a transport, so the fake check
(`tests/support/contract_fakes.py`) never sees it, and the fixture can agree with the tool by construction. This guard
walks the AST of every `tests/unit/**/*.py` that imports `src.mcp_server` (or drives it through `mcp_drive`) and fails
on either form:

1. a mock (`AsyncMock`, `MagicMock`, `Mock`) bound to a client verb (`get`, `post`, `put`, `patch`, `delete`,
   `request`, `request_full`, `raw_get`) by assignment, `setattr`, `patch.object` or a keyword;
2. a mock object assigned to a variable named with `client`, `miserver`, `dw` or `backend`, or handed to a tool
   module's `register(...)`;
3. a PLAIN-OBJECT fake handed to `register(...)`: an instance of a class the test file defines itself that is not a
   real client (`MiWorkbenchClient`, `MiServerClient`, `DataworksClient` or a subclass), directly or through a name
   it was assigned to (046-2 review round 1, finding 6). The files that still do this are tracked debt
   (`PLAIN_FAKES`, each with its reason); the list may only shrink, and a new file is red.

(A mock of something else that has a `get`, a database session for instance, or a request's `.client` attribute, is
neither form.)

Forms 1 and 2 have no exemption list: a file the guard flags is migrated (`RouteBackend`,
`fake_clients.workbench_over`), not exempted. A walk over source fails open (memory `source-scrape-guards-fail-open`), so the guard's own test plants
each form in a temporary file and requires it to be found, and requires the walk to have read a real number of files.

Control: M22 (the guard stops matching one form: its planted case is red).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

UNIT = Path(__file__).resolve().parent
VERBS = frozenset({"get", "post", "put", "patch", "delete", "request", "request_full", "raw_get"})
NAMES = ("client", "miserver", "dw", "backend")
MOCKS = frozenset({"AsyncMock", "MagicMock", "Mock"})
MCP = re.compile(r"src\.mcp_server|mcp_drive")
REAL_CLIENTS = frozenset({"MiWorkbenchClient", "MiServerClient", "DataworksClient"})
#: Form 3's tracked debt (record `046_controls_2026-11-25.md` §P6 7.5 and §review round 1): it may only shrink.
PLAIN_FAKES = {
    "test_reachability.py": "RecordingClient, one fake for every product's ~300 tools answering one envelope; needs "
                            "per-product adapters and a route-shaped answer per tool (P6 7.5)",
    "test_miserver_inference_tools.py": "chat bodies on management routes, JSON Lines batches; each needs its route's "
                                        "shape (P6 7.5)",
    "test_dataworks_tools.py": "`_Dw`: miDataworks' contract arrives at 046-3 (FTASKS 11.2)",
    "test_probe_live_progress.py": "a progress-polling fake whose answers are the test's sequence, not a route's",
    "test_miserver_approvals_relay.py": "`ApprovalQueueCheck(_Client(), gate)`: the deliberately off-contract cases "
                                        "feed bare lists to prove WT10 fails closed (P6 7.5)",
}


def _is_mock(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        return name in MOCKS
    return False


def _target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _verb_keywords(node: ast.AST) -> list[str]:
    return [k.arg for k in getattr(node, "keywords", []) if k.arg in VERBS and _is_mock(k.value)] if _is_mock(node) \
        else []


def findings(source: str, name: str = "<planted>") -> list[str]:
    """Each place in `source` that fakes a client without a transport, as `<name>:<line> <form>`."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is None or not _is_mock(node.value):
                continue
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in VERBS:
                    out.append(f"{name}:{node.lineno} a mock bound to the verb {target.attr!r}")
                elif isinstance(target, ast.Name) and any(word in target.id.lower() for word in NAMES):
                    out.append(f"{name}:{node.lineno} a mock client {target.id!r}")
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if called in ("setattr", "object") and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                    and node.args[1].value in VERBS:
                values = [*node.args[2:], *(k.value for k in node.keywords)]
                if any(_is_mock(v) for v in values) or (called == "object" and not values):
                    out.append(f"{name}:{node.lineno} a mock bound to the verb {node.args[1].value!r} by {called}")
            if called == "register":
                for arg in [*node.args, *(k.value for k in node.keywords)]:
                    if _is_mock(arg):
                        out.append(f"{name}:{node.lineno} a mock client handed to register()")
    return out


def plain_fakes(source: str, name: str = "<planted>") -> list[str]:
    """Form 3: an instance of a class `source` defines that is no real client, handed as the client to a tool module's
    `register(mcp, client, ...)`, or as any argument to a callable `source` imports from `src.mcp_server` (a relay's
    `ApprovalQueueCheck(client, gate)`, say). Followed: a name or a `self.` attribute it was assigned to (an annotated
    assignment too), and `x or Fake()` (review round 2, finding 6)."""
    tree = ast.parse(source)
    local = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = {b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else "" for b in node.bases}
            local[node.name] = bases
    # a class whose base chain (within the file) reaches a real client is a client
    def is_client(cls: str, seen: frozenset = frozenset()) -> bool:
        bases = local.get(cls, set())
        return bool(bases & REAL_CLIENTS) or any(is_client(b, seen | {cls}) for b in bases if b in local and b not in seen)
    fakes = {c for c in local if not is_client(c)}
    if not fakes:
        return []
    mcp_names = {alias.asname or alias.name for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src.mcp_server")
                 for alias in node.names}

    def instance_of_local(value: ast.AST) -> bool:
        if isinstance(value, ast.BoolOp):
            return any(instance_of_local(v) for v in value.values)
        return isinstance(value, ast.Call) and _target_name(value.func) in fakes

    def target_key(target: ast.AST) -> str:
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
            return "self." + target.attr
        return ""

    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and instance_of_local(node.value):
            bound |= {target_key(t) for t in node.targets} - {""}
        elif isinstance(node, ast.AnnAssign) and node.value is not None and instance_of_local(node.value):
            bound |= {target_key(node.target)} - {""}

    def is_fake(arg: ast.AST) -> bool:
        return instance_of_local(arg) or (target_key(arg) in bound and target_key(arg) != "")

    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if _target_name(func) == "register":
            # the client is a tool module's second argument (a gate or context later is not a client)
            candidates = [*node.args[1:2],
                          *(k.value for k in node.keywords if k.arg in ("client", "miserver", "dw", "backend"))]
            what = "register()"
        elif isinstance(func, ast.Name) and func.id in mcp_names:  # (a module alias's call is too loose: `mod` is
            # rebound to services and workers in the same files)
            candidates = [*node.args, *(k.value for k in node.keywords)]
            what = f"{ast.unparse(func)}()"
        else:
            continue
        if any(is_fake(arg) for arg in candidates):
            out.append(f"{name}:{node.lineno} a plain-object fake handed to {what}")
    return out


def _mcp_test_files() -> list[Path]:
    return sorted(p for p in UNIT.rglob("*.py") if p != Path(__file__) and MCP.search(p.read_text(encoding="utf-8")))


def test_no_mcp_test_fakes_a_client_without_a_transport():
    files = _mcp_test_files()
    assert len(files) >= 100, len(files)  # a blind walk would read nothing and pass
    found = [f for p in files for f in findings(p.read_text(encoding="utf-8"), str(p.relative_to(UNIT)))]
    assert not found, "MCP test fakes that answer without a transport (migrate them to RouteBackend):\n" + \
        "\n".join(found)


def test_no_new_file_hands_a_tool_a_plain_object_fake_and_the_debt_only_shrinks():
    found: dict[str, list[str]] = {}
    for p in _mcp_test_files():
        hits = plain_fakes(p.read_text(encoding="utf-8"), str(p.relative_to(UNIT)))
        if hits:
            found[str(p.relative_to(UNIT))] = hits
    new = {f: h for f, h in found.items() if f not in PLAIN_FAKES}
    assert not new, "MCP tests handing a tool a plain-object fake (use RouteBackend or workbench_over):\n" + \
        "\n".join(x for h in new.values() for x in h)
    gone = sorted(set(PLAIN_FAKES) - set(found))
    assert not gone, f"migrated: remove from PLAIN_FAKES {gone}"


def test_a_plain_object_fake_is_found_in_a_planted_file():
    planted = ("class FakeClient:\n    async def get(self, path):\n        return {}\n"
               "jlens.register(mcp, FakeClient(), settings)\n")
    assert plain_fakes(planted)
    assert plain_fakes(planted.replace("jlens.register(mcp, FakeClient(), settings)",
                                       "fake = FakeClient()\njlens.register(mcp, fake)"))
    real = ("from src.mcp_server.client import MiWorkbenchClient\nclass Sub(MiWorkbenchClient):\n    pass\n"
            "class SubSub(Sub):\n    pass\njlens.register(mcp, Sub('http://x'))\njlens.register(mcp, SubSub('x'))\n")
    assert plain_fakes(real) == []
    fake = "class Fake:\n    pass\n"
    for form in ("jlens.register(mcp, client or Fake())",
                 "c: Fake = Fake()\njlens.register(mcp, c)",
                 "self.client = Fake()\njlens.register(mcp, self.client)",
                 "jlens.register(mcp, backend=Fake())",
                 "from src.mcp_server.tools.miserver_approvals import ApprovalQueueCheck\nApprovalQueueCheck(Fake(), gate)"):
        assert plain_fakes(fake + form + "\n"), form


def test_each_form_is_found_in_a_planted_file(tmp_path):
    """The guard's own control: each form, planted, is found (a walk over source fails open)."""
    planted = {
        "assign_verb": "from unittest.mock import AsyncMock\nclient.get = AsyncMock(return_value={})\n",
        "setattr_verb": "from unittest.mock import AsyncMock\nsetattr(client, 'post', AsyncMock())\n",
        "patch_object": "from unittest.mock import patch, AsyncMock\npatch.object(Client, 'request', AsyncMock())\n",
        "keyword_verb": "from unittest.mock import AsyncMock, MagicMock\nclient = MagicMock(get=AsyncMock())\n",
        "registered": "from unittest.mock import AsyncMock, MagicMock\njlens.register(mcp, MagicMock(get=AsyncMock()))\n",
        "mock_client": "from unittest.mock import MagicMock\nmiserver = MagicMock()\n",
        "mock_backend": "from unittest.mock import MagicMock\nfake_backend = MagicMock()\n",
    }
    for form, source in planted.items():
        path = tmp_path / f"test_{form}.py"
        path.write_text(source, encoding="utf-8")
        assert findings(path.read_text(encoding="utf-8"), path.name), form
    clean = ("from unittest.mock import AsyncMock, MagicMock\ngate = MagicMock()\nresult = {'get': 1}\n"
             "session = MagicMock(get=AsyncMock())\nhttp_request.client = MagicMock(host='h')\n")
    assert findings(clean) == []
