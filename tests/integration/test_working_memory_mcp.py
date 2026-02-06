"""MCP integration tests for working memory.

Verify that the MCP tools interact correctly with the Redis-backed
working memory via the full MCP protocol (``fastmcp.Client``).
"""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

from engramcp.server import configure
from engramcp.server import mcp


def _parse(result) -> dict:
    return json.loads(result.content[0].text)


@pytest.fixture(autouse=True)
async def _setup_server(redis_container):
    """Configure the server with the test Redis before each test."""
    await configure(redis_url=redis_container)
    yield


class TestWorkingMemoryMCP:
    """MCP roundtrip tests with real working memory."""

    async def test_send_memory_stores_in_working_memory(self):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "send_memory",
                {
                    "content": "Alice met Bob on March 15",
                    "source": {
                        "type": "report",
                        "ref": "https://example.com/report.pdf",
                        "citation": "page 7",
                    },
                    "confidence_hint": "B",
                    "agent_id": "test_agent",
                },
            )
            data = _parse(result)
            assert data["status"] == "accepted"
            assert data["memory_id"].startswith("mem_")

            # Verify it can be retrieved
            get_result = await client.call_tool(
                "get_memory", {"query": "Alice Bob"}
            )
            get_data = _parse(get_result)
            assert get_data["meta"]["total_found"] >= 1

    async def test_get_memory_checks_working_memory_first(self):
        async with Client(mcp) as client:
            # Store a memory
            send_result = await client.call_tool(
                "send_memory", {"content": "The project uses Redis"}
            )
            mem_id = _parse(send_result)["memory_id"]

            # Query should find it in working memory
            get_result = await client.call_tool(
                "get_memory", {"query": "Redis"}
            )
            get_data = _parse(get_result)
            assert get_data["meta"]["working_memory_hits"] >= 1
            ids = [m["id"] for m in get_data["memories"]]
            assert mem_id in ids
