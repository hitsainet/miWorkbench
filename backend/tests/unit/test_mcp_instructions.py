"""045 FR-5: the instructions state THIS server's tool count, fit in 4,096 characters, and their first 512 stand alone.

Controls (record `0xcc/reviews/045_implementation_controls_2026-11-16.md`):
  count `all_categories()` instead of the registry      → test_the_count_is_this_servers red (the subset server)
  delete the `instructions = render_instructions` line   → test_the_count_is_this_servers red
  append the old probe-monitor paragraph                 → test_length_and_first_512 red
"""
from __future__ import annotations

import asyncio

import pytest

from tests.support import mcp_drive


def _built(categories: str):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(tool_categories=categories, allow_anonymous=True, _env_file=None,
                                      **mcp_drive.placeholder_overrides()), stdio=True)
    return mcp


def _initialize_instructions(mcp) -> str:
    """What a client reads at `initialize`, through the SDK's in-memory client."""
    from mcp import Client

    async def go():
        async with Client(mcp) as client:
            return client.instructions

    return asyncio.run(go())


@pytest.mark.parametrize("categories", ["read", "read,jobs,models,circuits", "probe_monitors,miserver_probes,dataworks"])
def test_the_count_is_this_servers(categories):
    mcp = _built(categories)
    count = len(asyncio.run(mcp.list_tools()))
    text = _initialize_instructions(mcp)
    assert f"{count} tools in {len(categories.split(','))} categories here." in text, text[:200]
    assert "Up to" not in text and "up to" not in text.split("\n\n")[0]


def test_length_and_first_512():
    from src.mcp_server.server import INSTRUCTION_HEAD_CHARS, MAX_INSTRUCTION_CHARS
    from src.mcp_server.tools import all_categories

    for categories in (None, ",".join(sorted(all_categories()))):
        mcp = _built(categories) if categories else _built("")
        text = _initialize_instructions(mcp)
        assert len(text) <= MAX_INSTRUCTION_CHARS == 4096, len(text)
        head = text[:INSTRUCTION_HEAD_CHARS]
        assert "miworkbench_howto" in head
        assert "unavailable" in head and "refused" in head
        assert "approval" in head or "pending" in head
        assert "never" in head and "approve" in head


def test_the_probe_chain_moved_to_its_topic():
    from src.mcp_server.server import SERVER_INSTRUCTIONS
    from src.mcp_server.tools.howto import TOPICS

    assert "acknowledge_rule_choice" not in SERVER_INSTRUCTIONS and "nonsense_text" not in SERVER_INSTRUCTIONS
    assert 'miworkbench_howto("probe_monitors")' in SERVER_INSTRUCTIONS
    for tool in ("recommend_probe_rules", "forecast_probe_training_run", "start_probe_training_run",
                 "evaluate_probe_monitor", "build_probe_definition", "register_probe_dataset"):
        assert tool in TOPICS["probe_monitors"], tool
