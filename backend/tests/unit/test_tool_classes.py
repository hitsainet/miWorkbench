"""045 FR-1 – FR-3: one classification table, applied to the live registry, and honest.

Every assertion reads the WIRE (`tools/list` through the SDK's in-memory client) of servers built the way production
builds them (`build_server`, one per category, placeholder product URLs), never the table alone. The honesty rules
read each tool's HTTP verbs from two recorders: `test_reachability.EXPECTED_CALLS` (payload-asserted) and the verbs
the tool actually sends through the real clients when driven over the SDK client (`tests/support/mcp_drive.py`).

Controls (record `0xcc/reviews/045_implementation_controls_2026-11-16.md`, restored from saved bytes):
  delete the `apply_tool_classes` call            → test_every_tool_is_annotated_on_the_wire red
  delete one row / add a stale row                → test_every_live_tool_has_a_row red
  skip `destructive_hint`                         → test_every_tool_is_annotated_on_the_wire red
  turn the raise into a warning                   → test_an_unclassified_tool_refuses_to_build red
  `delete_judge_prompt` → write                   → test_rules_hold[rule2] red
  `miserver_score_probes` → read                  → test_rules_hold[rule4] red
  `get_detector_success_report` → read            → test_rules_hold[rule4] red
  a gated tool → read                             → test_rules_hold[rule5] red
  drop a read tool's driven-call override         → test_rules_hold[rule6] red
  record sent verbs for read tools only           → test_rule3_reads_the_drive red (OR1-045-4)
"""
from __future__ import annotations

import re

import pytest

from tests.support import mcp_drive

#: Arguments for read tools whose generated arguments (required fields only) reach no HTTP call: a one-of choice
#: or a validated GPU UUID. Used only to record the verbs the tool sends (rule 6).
DRIVE_ARGUMENTS: dict[str, dict] = {
    "get_steering_samples": {"manifest_id": "vman_test"},
    "gpu_preflight": {"gpu": "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"},
}

#: Read-only tools that make no HTTP call at all, with the reason. Each is ALSO driven, and the test requires that
#: it really sends nothing (so a pure read that starts calling a route must earn an entry elsewhere).
NO_HTTP_READS: dict[str, str] = {
    "miworkbench_howto": "builds throwaway servers in-process and reads their registry; guidance text otherwise",
}

_DESTRUCTIVE_WORDS = ("delete", "purge", "clear", "cancel")
_READ_PREFIXES = ("get_", "list_", "search_", "find_", "read_")


def _bare(name: str) -> str:
    for prefix in ("miserver_", "dataworks_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


#: Tools that send a DELETE with no `EXPECTED_CALLS` entry: rule 3 sees their DELETE only because every tool is
#: driven (R1 review fix OR1-045-4, 2026-11-16). `test_rule3_reads_the_drive` requires each to be seen.
DELETE_WITHOUT_AN_ENTRY = ("cancel_steering_task", "delete_circuit", "cancel_model_download", "delete_experiment",
                           "delete_extraction")


@pytest.fixture(scope="module")
def world():
    """Every category built, its wire tools listed, and EVERY tool's sent verbs recorded (OR1-045-4: rule 3 reads the
    verbs each tool really sends, not only the tools that have an `EXPECTED_CALLS` entry)."""
    from src.mcp_server.tool_classes import TOOL_CLASSES

    with pytest.MonkeyPatch.context() as mp:
        # 046: the drive records the verbs each tool SENDS; its `{}` / `[]` answers are arbitrary, so it is an
        # off-contract fake, and a typed tool's client refusing them (the typing biting) does not stop the record.
        with mcp_drive.patched_http(mp, off_contract="the drive records each tool's sent verbs over arbitrary "
                                                     "answers") as backend:
            servers = mcp_drive.build_every_category()
            wire = {}
            sent: dict[str, set[str]] = {}
            for category, mcp in servers.items():
                for tool in mcp_drive.wire_tools(mcp):
                    wire[tool.name] = tool
                    if TOOL_CLASSES.get(tool.name) is None:
                        continue
                    verbs: set[str] = set()
                    for body in (mcp_drive.EMPTY_OBJECT, mcp_drive.EMPTY_LIST):
                        backend.body = body
                        backend.requests.clear()
                        args = DRIVE_ARGUMENTS.get(tool.name) or mcp_drive.sample_arguments(tool)
                        try:
                            mcp_drive.call(mcp, tool.name, args)
                        except Exception:  # noqa: BLE001 - a typed tool's client refuses `{}`; the verbs are sent
                            pass
                        verbs |= {f"{q.method} {q.url.path}" for q in backend.requests
                                  if not q.url.path.endswith("/health")}
                    sent[tool.name] = verbs
    return {"wire": wire, "sent": sent}


def _expected_verbs() -> dict[str, set[str]]:
    from tests.unit.test_reachability import EXPECTED_CALLS

    return {name: {"GET" if method == "RAW_GET" else method}
            for name, (method, _path, _kwargs, _payload) in EXPECTED_CALLS.items()}


def _verbs(world, name: str) -> set[str]:
    recorded = set(_expected_verbs().get(name, set()))
    recorded |= {v.split(" ", 1)[0] for v in world["sent"].get(name, set())}
    return recorded


def test_every_live_tool_has_a_row(world):
    """FR-1: the live names equal the table's keys, both directions (a stale row fails too)."""
    from src.mcp_server.tool_classes import TOOL_CLASSES

    live = set(world["wire"])
    assert len(live) > 300, f"the registry was not built: {len(live)} tools"
    assert not live - set(TOOL_CLASSES), f"live tools with no row: {sorted(live - set(TOOL_CLASSES))}"
    assert not set(TOOL_CLASSES) - live, f"rows for no live tool: {sorted(set(TOOL_CLASSES) - live)}"


def test_every_tool_is_annotated_on_the_wire(world):
    """FR-2: per tool, a title and BOTH hints as booleans, equal to the table, as a client reads them."""
    from src.mcp_server.tool_classes import TOOL_CLASSES

    wrong = []
    for name, tool in world["wire"].items():
        row = TOOL_CLASSES[name]
        ann = tool.annotations
        if (not tool.title or tool.title != row.title or ann is None or ann.title != row.title
                or ann.read_only_hint is not (row.kind == "read")
                or ann.destructive_hint is not (row.kind == "destructive")):
            wrong.append((name, tool.title, ann))
    assert not wrong, f"{len(wrong)} tools not annotated as the table says, e.g. {wrong[:3]}"
    dumped = next(iter(world["wire"].values())).model_dump(by_alias=True, exclude_none=True)
    assert {"readOnlyHint", "destructiveHint", "title"} <= set(dumped["annotations"]), dumped


def test_an_unclassified_tool_refuses_to_build(monkeypatch):
    """FR-2: `build_server` raises, naming the tool, when a registered tool has no row."""
    from src.mcp_server import tool_classes

    monkeypatch.delitem(tool_classes.TOOL_CLASSES, "list_models")
    with pytest.raises(RuntimeError, match="list_models"):
        mcp_drive.build_category("models")

    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("t")

    @mcp.tool()
    async def throwaway_tool_with_no_row() -> dict:
        return {}

    with pytest.raises(RuntimeError, match="throwaway_tool_with_no_row"):
        tool_classes.apply_tool_classes(mcp)


def test_titles_are_well_formed():
    """I45-D2: non-empty, at most 60 characters, unique, not the tool name, no trailing period."""
    from src.mcp_server.tool_classes import TOOL_CLASSES

    titles = [row.title for row in TOOL_CLASSES.values()]
    assert len(titles) == len(set(titles)), "duplicate titles"
    bad = [(n, r.title) for n, r in TOOL_CLASSES.items()
           if not r.title.strip() or len(r.title) > 60 or r.title == n or r.title.endswith(".")]
    assert not bad, bad


def test_rule1_no_tool_is_both_read_only_and_destructive(world):
    bad = [n for n, t in world["wire"].items()
           if t.annotations and t.annotations.read_only_hint and t.annotations.destructive_hint]
    assert not bad, bad


def test_rule2_a_destructive_word_makes_a_tool_destructive():
    from src.mcp_server.tool_classes import TOOL_CLASSES

    bad = [n for n, r in TOOL_CLASSES.items()
           if any(w in _bare(n).split("_") for w in _DESTRUCTIVE_WORDS) and r.kind != "destructive"]
    assert not bad, f"named like a delete, purge, clear or cancel but not destructive: {bad}"


def test_rule2b_a_read_named_tool_that_is_not_read_says_why():
    from src.mcp_server.tool_classes import TOOL_CLASSES

    bad = [n for n, r in TOOL_CLASSES.items()
           if _bare(n).startswith(_READ_PREFIXES) and r.kind != "read" and not r.why]
    assert not bad, f"named like a read but classified {[(n, TOOL_CLASSES[n].kind) for n in bad]} with no why"


def test_rule3_a_recorded_delete_makes_a_tool_destructive(world):
    from src.mcp_server.tool_classes import TOOL_CLASSES

    bad = [n for n, r in TOOL_CLASSES.items() if "DELETE" in _verbs(world, n) and r.kind != "destructive"]
    assert not bad, f"issue a DELETE but are not destructive: {bad}"


def test_rule3_reads_the_drive(world):
    """OR1-045-4: the drive records the verbs of every tool, so a DELETE-sending tool with no `EXPECTED_CALLS` entry
    is seen by rule 3 (not only by the name rule). Control: record sent verbs for read tools only → red."""
    from tests.unit.test_reachability import EXPECTED_CALLS

    sends_delete = {n for n, verbs in world["sent"].items() if any(v.startswith("DELETE ") for v in verbs)}
    assert len(sends_delete) >= 13, sorted(sends_delete)
    for name in DELETE_WITHOUT_AN_ENTRY:
        assert name not in EXPECTED_CALLS, f"{name} gained an EXPECTED_CALLS entry; drop it from this list"
        assert name in sends_delete, f"the drive did not see {name}'s DELETE"


def test_rule4_a_read_only_tool_only_gets_unless_it_says_why(world):
    from src.mcp_server.tool_classes import TOOL_CLASSES

    bad = {n: sorted(_verbs(world, n)) for n, r in TOOL_CLASSES.items()
           if r.kind == "read" and _verbs(world, n) - {"GET"} and "pure POST" not in r.why}
    assert not bad, f"read-only tools that send more than GET with no 'pure POST' why: {bad}"


def test_rule5_a_gated_tool_is_never_read_only():
    from src.mcp_server.tool_classes import TOOL_CLASSES

    gated = {n for n, r in TOOL_CLASSES.items() if r.approval}
    assert len(gated) == 24, sorted(gated)
    assert not [n for n in gated if TOOL_CLASSES[n].kind == "read"]


def test_rule6_every_read_only_tool_has_recorded_calls(world):
    """R1A-15: a read-only tool must have its HTTP verbs recorded (an `EXPECTED_CALLS` entry, or the verbs it sends
    when driven through the real clients), or be a declared no-HTTP read that really sends nothing."""
    from src.mcp_server.tool_classes import TOOL_CLASSES

    reads = [n for n, r in TOOL_CLASSES.items() if r.kind == "read"]
    unproven = [n for n in reads if not _verbs(world, n) and n not in NO_HTTP_READS]
    assert not unproven, f"read-only tools with no recorded call: {unproven}"
    for name in NO_HTTP_READS:
        assert not world["sent"].get(name), f"{name} is declared no-HTTP but sent {world['sent'][name]}"


def test_the_proposed_counts():
    """The re-derived classification (FTASKS 2.1; FTID Appendix A plus the recorded differences). 2026-11-26: +1 read,
    `get_probe_evaluation_rows` (ENH-010 FU-7a G-1); +1 read, `get_dataset_rows` (ENH-010 FU-7a)."""
    from collections import Counter

    from src.mcp_server.tool_classes import TOOL_CLASSES

    assert Counter(r.kind for r in TOOL_CLASSES.values()) == {"read": 166, "write": 143, "destructive": 39}
    pattern = re.compile(r"^[a-z][a-z0-9_]*$")
    assert all(pattern.match(n) for n in TOOL_CLASSES)


def test_the_deployed_shape_lists_every_row_with_its_payload():
    """B4 port wiring control (2026-11-24): ONE server with every category enabled, the shape the pod runs, read
    through the SDK client as an agent reads it. Its live `tools/list` equals the table in both directions, the count
    `initialize` states is the count `tools/list` returns, and each tool's annotation payload is exactly its row.

    Removing any registration line of `build_server` (the native and the external-product `module.register` calls),
    any module from a category map, or `apply_tool_classes` turns this red (controls record, B4 port section)."""
    from mcp import Client

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server
    from src.mcp_server.tool_classes import TOOL_CLASSES
    from src.mcp_server.tools import all_categories

    categories = sorted(all_categories())
    mcp, _client = build_server(
        MCPSettings(tool_categories=",".join(categories), allow_anonymous=True, _env_file=None,
                    **mcp_drive.placeholder_overrides()),
        stdio=True,
    )

    async def go():
        async with Client(mcp) as client:
            return client.instructions, (await client.list_tools()).tools

    init, tools = mcp_drive.run(go())
    names = [t.name for t in tools]
    assert len(names) == len(set(names)) == len(TOOL_CLASSES), (len(names), len(TOOL_CLASSES))
    assert set(names) == set(TOOL_CLASSES), {
        "live, no row": sorted(set(names) - set(TOOL_CLASSES)),
        "row, not live": sorted(set(TOOL_CLASSES) - set(names)),
    }
    assert f"{len(names)} tools in {len(categories)} categories here" in (init or ""), (init or "")[:300]
    wrong = []
    for tool in tools:
        row = TOOL_CLASSES[tool.name]
        payload = tool.model_dump(by_alias=True, exclude_none=True)
        expected = {"title": row.title, "readOnlyHint": row.kind == "read",
                    "destructiveHint": row.kind == "destructive"}
        got = {k: payload.get("annotations", {}).get(k) for k in expected}
        # mistudio.result-language/v1 AR-5: idempotentHint and openWorldHint stay unset.
        unset = {"idempotentHint", "openWorldHint"} & set(payload.get("annotations", {}))
        if got != expected or unset or payload.get("title") != row.title or payload.get("outputSchema", {}).get(
                "type") != "object":
            wrong.append((tool.name, got, payload.get("outputSchema", {}).get("type")))
    assert not wrong, f"{len(wrong)} tools whose payload is not their row, e.g. {wrong[:3]}"
