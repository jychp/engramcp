"""MCP protocol-level integration tests.

Verifies tool registration, roundtrip send/get, and error handling
at the MCP protocol layer via ``fastmcp.Client``.
"""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

from engramcp.server import _reset_working_memory
from engramcp.server import configure
from engramcp.server import mcp


@pytest.fixture(autouse=True)
async def _clean(redis_container):
    await configure(redis_url=redis_container)
    await _reset_working_memory()
    yield
    await _reset_working_memory()


def _parse(result) -> dict:
    return json.loads(result.content[0].text)


class TestMcpProtocol:
    """Protocol-level checks for the EngraMCP server."""

    async def test_list_tools_includes_three_tools(self):
        async with Client(mcp) as client:
            tools = await client.list_tools()
            names = {t.name for t in tools}
            assert names == {"send_memory", "get_memory", "correct_memory"}

    async def test_send_then_get_roundtrip(self):
        async with Client(mcp) as client:
            # Send
            send_result = await client.call_tool(
                "send_memory",
                {
                    "content": "Epstein flew to St. Thomas on March 15, 2005",
                    "source": {
                        "type": "flight_log",
                        "ref": "https://example.com/log.pdf",
                        "citation": "page 3",
                    },
                    "confidence_hint": "B",
                    "agent_id": "test_agent",
                },
            )
            send_data = _parse(send_result)
            memory_id = send_data["memory_id"]
            assert memory_id.startswith("mem_")

            # Get
            get_result = await client.call_tool(
                "get_memory",
                {"query": "Epstein St. Thomas"},
            )
            get_data = _parse(get_result)
            assert get_data["meta"]["total_found"] >= 1

            memories = get_data["memories"]
            assert any(m["id"] == memory_id for m in memories)

            found = next(m for m in memories if m["id"] == memory_id)
            assert "Epstein" in found["content"]
            assert found["type"] == "Fact"
            assert len(found["sources"]) == 1
            assert found["sources"][0]["type"] == "flight_log"

    async def test_tool_error_on_invalid_input(self):
        async with Client(mcp) as client:
            # send_memory requires content — omitting it should error
            with pytest.raises(Exception):
                await client.call_tool("send_memory", {})
