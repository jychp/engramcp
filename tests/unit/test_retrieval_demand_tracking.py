"""Unit tests for retrieval-demand tracking wiring in get_memory."""

from __future__ import annotations

import json


def _parse(result) -> dict:
    return json.loads(result.content[0].text)


class TestRetrievalDemandTracking:
    async def test_tracks_default_get_memory_shape(self, mcp_client):
        from engramcp.server import _get_query_demand_count

        await mcp_client.call_tool("send_memory", {"content": "A flew to St. Thomas"})
        await mcp_client.call_tool("get_memory", {"query": "flew"})

        count = _get_query_demand_count(
            node_types=["Fact"],
            properties=[
                "content",
                "confidence",
                "participants",
                "causal_chain",
                "sources",
                "contradictions",
                "min_confidence",
            ],
        )
        assert count == 1

    async def test_tracks_compact_shape_without_sources(self, mcp_client):
        from engramcp.server import _get_query_demand_count

        await mcp_client.call_tool("send_memory", {"content": "A flew to St. Thomas"})
        await mcp_client.call_tool(
            "get_memory",
            {"query": "flew", "compact": True, "include_sources": False},
        )

        count = _get_query_demand_count(
            node_types=["Fact"],
            properties=["content", "confidence", "contradictions", "min_confidence"],
        )
        assert count == 1

    async def test_counts_repeated_calls_for_same_shape(self, mcp_client):
        from engramcp.server import _get_query_demand_count

        await mcp_client.call_tool("send_memory", {"content": "A flew to St. Thomas"})
        await mcp_client.call_tool("get_memory", {"query": "flew"})
        await mcp_client.call_tool("get_memory", {"query": "flew"})

        count = _get_query_demand_count(
            node_types=["Fact"],
            properties=[
                "content",
                "confidence",
                "participants",
                "causal_chain",
                "sources",
                "contradictions",
                "min_confidence",
            ],
        )
        assert count == 2

    async def test_emits_concept_candidate_when_threshold_reached(self, mcp_client):
        from engramcp.server import _get_concept_candidate_count

        await mcp_client.call_tool("send_memory", {"content": "A flew to St. Thomas"})
        await mcp_client.call_tool("get_memory", {"query": "flew"})
        await mcp_client.call_tool("get_memory", {"query": "flew"})
        await mcp_client.call_tool("get_memory", {"query": "flew"})

        assert _get_concept_candidate_count() == 1
