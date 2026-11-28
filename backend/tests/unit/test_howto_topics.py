"""045 FR-5: the guide tool's description lists every topic (built from `TOPICS`), the two new topics resolve, and
its answer reports this server's own tool count.

Controls (record `0xcc/reviews/045_implementation_controls_2026-11-16.md`):
  hand-write the old docstring as the description        → test_the_description_lists_every_topic red
  report `tool_count` as `enabled_tool_count`             → test_enabled_tool_count_is_this_servers red
"""
from __future__ import annotations

import asyncio
import json

from tests.support import mcp_drive


def _jobs_server():
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    mcp, _ = build_server(MCPSettings(tool_categories="read,jobs", allow_anonymous=True, _env_file=None,
                                      **mcp_drive.placeholder_overrides()), stdio=True)
    return mcp


def test_the_description_lists_every_topic():
    from src.mcp_server.tools.howto import TOPICS

    tool = next(t for t in mcp_drive.wire_tools(_jobs_server()) if t.name == "miworkbench_howto")
    for topic in [*TOPICS, "tools"]:
        assert topic in tool.description, topic
    assert "probe_monitors" in TOPICS and "results" in TOPICS and "model_management" in TOPICS


def test_new_topics_resolve():
    mcp = _jobs_server()
    for topic in ("probe_monitors", "results"):
        result = mcp_drive.call(mcp, "miworkbench_howto", {"topic": topic})
        assert not result.is_error
        assert result.structured_content["topic"] == topic and result.structured_content["guidance"]


def test_enabled_tool_count_is_this_servers():
    mcp = _jobs_server()
    result = mcp_drive.call(mcp, "miworkbench_howto", {})
    body = result.structured_content
    assert body["enabled_tool_count"] == len(asyncio.run(mcp.list_tools()))
    assert body["tool_count"] > body["enabled_tool_count"], "tool_count stays every category's"
    assert json.loads(result.content[0].text)["enabled_tool_count"] == body["enabled_tool_count"]
