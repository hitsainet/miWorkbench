"""miForge 018 FU-5 (2026-11-25): an imported WSP candidate is NOT "a new rung-0 probe". It EARNS ITS RUNG AT IMPORT.

The MCP instructions, the `wsp` howto topic and `import_wsp_candidate`'s description told an agent the candidate comes
in as "a new rung-0 probe". It does not: the import calibrates the new probe and evaluates it on every held-out
capture, then `recompute_rung` sets its rung (`services/wsp_import.py::import_candidate`); `pm_ae08dd79a26e` came in at
rung 2 on 2026-11-23. The probe holds rung 0 only while the import runs (`create_derived_probe`, pinned in
`test_wsp_import.py`).

Each surface is read where an agent or the operator reads it: the instructions the LIVE server sends, the howto
topics the live `miworkbench_howto` tool answers, every tool description in the live registry (every category), the
backend's OpenAPI document, the operator's approval summary, the generated MCP contract and the manual. No paragraph
about a WSP candidate may call it "a rung-0 probe" or "a probe at rung 0" (a probe TRAINED with no evaluation set
still is one); the import's own sentence in each says it earns its rung at import.

Controls (044 FTASKS, FU-5 note): put "a new rung-0 probe" back in `server.py`'s instructions → red; put "a NEW
probe at rung 0" back in the tool's docstring, the howto topic or the approval summary → red.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

#: The wrong claim, in each spelling it has had: "a new rung-0 probe", "a NEW probe at rung 0", "a NEW probe (rung 0".
WRONG = re.compile(r"rung[- ]0 probe|probe\s+(?:at\s+|\(\s*)rung[- ]0", re.IGNORECASE)
RIGHT = re.compile(r"earns\s+its\s+rung\s+at\s+import", re.IGNORECASE)


def _server():
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server
    from src.mcp_server.tools import all_categories

    mp = pytest.MonkeyPatch()
    try:
        from src.mcp_server.products import PRODUCTS

        for spec in PRODUCTS:
            mp.setenv(spec.url_env, f"http://{spec.name}.test")
        mcp, client = build_server(MCPSettings(tool_categories=",".join(sorted(all_categories())),
                                               allow_anonymous=True), stdio=True)
    finally:
        mp.undo()
    return mcp


#: A paragraph (or table row) about a WSP candidate: the wrong claim is about the IMPORT. A probe TRAINED with no
#: evaluation set really is "a rung-0 probe" (manual `core-workflow/probe-monitors.md`), and stays allowed.
ABOUT_IMPORT = re.compile(r"candidate|import_wsp|weight-space|\bwsp\b", re.IGNORECASE)


def _wrong(text: str) -> list[str]:
    found = []
    for block in re.split(r"\n\s*\n|\n(?=\|)", text):
        if ABOUT_IMPORT.search(block):
            found += [block[max(0, m.start() - 80):m.end() + 40] for m in WRONG.finditer(block)]
    return found


def test_the_live_instructions_say_the_import_earns_its_rung():
    mcp = _server()
    instructions = mcp.instructions or ""
    assert not _wrong(instructions), _wrong(instructions)
    sentence = next(s for s in instructions.split("\n\n") if "import_wsp_candidate" in s)
    assert RIGHT.search(sentence), sentence


def test_no_live_tool_description_calls_the_import_a_rung_0_probe():
    mcp = _server()
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) > 100  # every category, not an empty registry
    found = {t.name: _wrong(t.description or "") for t in tools if _wrong(t.description or "")}
    assert not found, found
    tool = next(t for t in tools if t.name == "import_wsp_candidate")
    assert RIGHT.search(tool.description), tool.description[:300]


def _strings(node) -> list[str]:
    """Every string in a tool's answer (its text content or structured fields), depth first."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, (list, tuple)):
        return [s for v in node for s in _strings(v)]
    for attr in ("text", "content", "structured_content"):
        if hasattr(node, attr):
            return _strings(getattr(node, attr))
    return []


def test_the_live_howto_topics_say_the_import_earns_its_rung():
    from src.mcp_server.tools import howto

    mcp = _server()
    for topic in howto.TOPICS:
        answer = asyncio.run(mcp._tool_manager.call_tool("miworkbench_howto", {"topic": topic},  # noqa: SLF001
                                                         context=None))
        text = "\n\n".join(_strings(answer))  # the answer's own text, its newlines real (never JSON-escaped)
        assert len(text) > 200, (topic, text[:200])
        assert not _wrong(text), (topic, _wrong(text))
        if topic in ("wsp", "tool_map"):
            assert RIGHT.search(text), topic


def test_the_operator_approval_summary_says_the_import_earns_its_rung():
    from src.api.v1.endpoints.mcp_approvals import _summarize

    payload = {"run_document": {"run_id": "wsp_1", "candidate_id": "wspc_1", "verdict": {"value": "improved"},
                                "request": {"bundle": {"bundle_id": "wsb_1"}}},
               "patch_manifest": {}, "edit_npz_b64": "QUJD"}
    text = _summarize("import_wsp_candidate", payload)
    assert not _wrong(text) and RIGHT.search(text), text


def test_the_backend_openapi_never_calls_the_import_a_rung_0_probe():
    from src.main import app

    found = [w for text in _strings(app.openapi()) for w in _wrong(text)]  # each description, its newlines real
    assert not found, found
    route = app.openapi()["paths"]["/api/v1/wsp/candidates/import"]["post"]
    assert RIGHT.search(route.get("description", "").replace("\n", " ")), route.get("description")


def test_the_contract_and_the_manual_never_call_the_import_a_rung_0_probe():
    contract = (REPO / "docs" / "mcp-contract.md").read_text()
    row = next(line for line in contract.splitlines() if line.startswith("| `import_wsp_candidate`"))
    assert RIGHT.search(row), row
    pages = sorted((REPO / "manual" / "docs").rglob("*.md"))
    assert pages, "manual/docs has no pages"
    found = {str(p.relative_to(REPO)): _wrong(p.read_text()) for p in [REPO / "docs" / "mcp-contract.md", *pages]
             if _wrong(p.read_text())}
    assert not found, found
    manual = (REPO / "manual" / "docs" / "advanced" / "mcp-server.md").read_text()
    assert RIGHT.search(next(line for line in manual.splitlines() if "`import_wsp_candidate` ⚙️" in line))
