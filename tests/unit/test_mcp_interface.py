"""MCP interface contract tests.

All tests use ``fastmcp.Client`` to exercise the full MCP protocol
(serialization, validation).  Backend logic is mocked via a module-level
``_working_memory`` dict in ``server.py``.
"""

from __future__ import annotations

import json

import pytest


def _parse(result) -> dict:
    """Extract the JSON payload from a CallToolResult."""
    return json.loads(result.content[0].text)


# -----------------------------------------------------------------------
# send_memory
# -----------------------------------------------------------------------


class TestSendMemory:
    """Contract tests for the send_memory tool."""

    async def test_accepts_valid_payload(self, mcp_client):
        result = await mcp_client.call_tool(
            "send_memory", {"content": "A met B on March 15"}
        )
        data = _parse(result)
        assert data["status"] == "accepted"

    async def test_returns_memory_id(self, mcp_client):
        result = await mcp_client.call_tool(
            "send_memory", {"content": "A met B on March 15"}
        )
        data = _parse(result)
        assert "memory_id" in data
        assert isinstance(data["memory_id"], str)
        assert len(data["memory_id"]) > 0

    async def test_rejects_missing_content(self, mcp_client):
        with pytest.raises(Exception):
            await mcp_client.call_tool("send_memory", {})

    async def test_accepts_optional_source(self, mcp_client):
        result = await mcp_client.call_tool(
            "send_memory",
            {
                "content": "A met B",
                "source": {
                    "type": "court_document",
                    "ref": "https://example.com/doc.pdf",
                    "citation": "page 3, line 7",
                },
            },
        )
        data = _parse(result)
        assert data["status"] == "accepted"

    async def test_accepts_optional_confidence_hint(self, mcp_client):
        result = await mcp_client.call_tool(
            "send_memory",
            {"content": "A met B", "confidence_hint": "B"},
        )
        data = _parse(result)
        assert data["status"] == "accepted"

    async def test_accepts_optional_agent_id(self, mcp_client):
        result = await mcp_client.call_tool(
            "send_memory",
            {"content": "A met B", "agent_id": "analyst_1"},
        )
        data = _parse(result)
        assert data["status"] == "accepted"


# -----------------------------------------------------------------------
# get_memory
# -----------------------------------------------------------------------


class TestGetMemory:
    """Contract tests for the get_memory tool."""

    async def test_accepts_query_string(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory", {"query": "Who is A?"}
        )
        data = _parse(result)
        assert "memories" in data

    async def test_accepts_optional_max_depth(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory", {"query": "Who is A?", "max_depth": 5}
        )
        data = _parse(result)
        assert "memories" in data

    async def test_accepts_optional_min_confidence(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory", {"query": "Who is A?", "min_confidence": "B2"}
        )
        data = _parse(result)
        assert "memories" in data

    async def test_accepts_optional_include_contradictions(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory",
            {"query": "Who is A?", "include_contradictions": False},
        )
        data = _parse(result)
        assert "memories" in data

    async def test_accepts_optional_include_sources(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory",
            {"query": "Who is A?", "include_sources": False},
        )
        data = _parse(result)
        assert "memories" in data

    async def test_accepts_optional_limit(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory", {"query": "Who is A?", "limit": 5}
        )
        data = _parse(result)
        assert "memories" in data

    async def test_accepts_optional_compact_flag(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory", {"query": "Who is A?", "compact": True}
        )
        data = _parse(result)
        assert "memories" in data

    async def test_response_has_memories_array(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory", {"query": "anything"}
        )
        data = _parse(result)
        assert isinstance(data["memories"], list)

    async def test_response_has_contradictions_array(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory", {"query": "anything"}
        )
        data = _parse(result)
        assert isinstance(data["contradictions"], list)

    async def test_response_has_meta_object(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory", {"query": "anything"}
        )
        data = _parse(result)
        assert isinstance(data["meta"], dict)
        assert data["meta"]["query"] == "anything"

    async def test_memory_has_id_type_content_confidence(self, mcp_client):
        # Send a memory first so get_memory returns something
        await mcp_client.call_tool(
            "send_memory", {"content": "A flew to St. Thomas"}
        )
        result = await mcp_client.call_tool(
            "get_memory", {"query": "flew"}
        )
        data = _parse(result)
        assert len(data["memories"]) > 0
        mem = data["memories"][0]
        assert "id" in mem
        assert "type" in mem
        assert "content" in mem
        assert "confidence" in mem

    async def test_memory_has_dynamic_type_when_exists(self, mcp_client):
        await mcp_client.call_tool(
            "send_memory", {"content": "A flew to St. Thomas"}
        )
        result = await mcp_client.call_tool(
            "get_memory", {"query": "flew"}
        )
        data = _parse(result)
        mem = data["memories"][0]
        # dynamic_type may be None but the key must exist
        assert "dynamic_type" in mem

    async def test_memory_has_participants_inline(self, mcp_client):
        await mcp_client.call_tool(
            "send_memory", {"content": "A flew to St. Thomas"}
        )
        result = await mcp_client.call_tool(
            "get_memory", {"query": "flew"}
        )
        data = _parse(result)
        mem = data["memories"][0]
        assert "participants" in mem
        assert isinstance(mem["participants"], list)

    async def test_memory_has_causal_chain_with_summaries(self, mcp_client):
        await mcp_client.call_tool(
            "send_memory", {"content": "A flew to St. Thomas"}
        )
        result = await mcp_client.call_tool(
            "get_memory", {"query": "flew"}
        )
        data = _parse(result)
        mem = data["memories"][0]
        assert "causal_chain" in mem
        assert isinstance(mem["causal_chain"], list)

    async def test_memory_has_sources_with_citation(self, mcp_client):
        await mcp_client.call_tool(
            "send_memory",
            {
                "content": "A flew to St. Thomas",
                "source": {
                    "type": "flight_log",
                    "ref": "https://example.com/log.pdf",
                    "citation": "page 3",
                },
            },
        )
        result = await mcp_client.call_tool(
            "get_memory", {"query": "flew"}
        )
        data = _parse(result)
        mem = data["memories"][0]
        assert "sources" in mem
        assert isinstance(mem["sources"], list)

    async def test_meta_has_total_found_and_returned(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory", {"query": "anything"}
        )
        data = _parse(result)
        meta = data["meta"]
        assert "total_found" in meta
        assert "returned" in meta
        assert isinstance(meta["total_found"], int)
        assert isinstance(meta["returned"], int)

    async def test_meta_indicates_truncation(self, mcp_client):
        result = await mcp_client.call_tool(
            "get_memory", {"query": "anything"}
        )
        data = _parse(result)
        assert "truncated" in data["meta"]
        assert isinstance(data["meta"]["truncated"], bool)

    async def test_include_sources_false_omits_sources(self, mcp_client):
        await mcp_client.call_tool(
            "send_memory",
            {
                "content": "A flew to St. Thomas",
                "source": {
                    "type": "flight_log",
                    "ref": "https://example.com/log.pdf",
                    "citation": "page 3",
                },
            },
        )
        result = await mcp_client.call_tool(
            "get_memory", {"query": "flew", "include_sources": False}
        )
        data = _parse(result)
        assert len(data["memories"]) > 0
        mem = data["memories"][0]
        assert mem["sources"] == []
        # Other fields should still be present
        assert "participants" in mem
        assert "causal_chain" in mem

    async def test_compact_mode_omits_sources_and_chains(self, mcp_client):
        await mcp_client.call_tool(
            "send_memory",
            {
                "content": "A flew to St. Thomas",
                "source": {
                    "type": "flight_log",
                    "ref": "https://example.com/log.pdf",
                    "citation": "page 3",
                },
            },
        )
        result = await mcp_client.call_tool(
            "get_memory", {"query": "flew", "compact": True}
        )
        data = _parse(result)
        assert len(data["memories"]) > 0
        mem = data["memories"][0]
        # In compact mode, sources and causal_chain should be empty
        assert mem.get("sources", []) == []
        assert mem.get("causal_chain", []) == []
        assert mem.get("participants", []) == []
        # Contradictions block should be empty
        assert data["contradictions"] == []

    async def test_respects_min_confidence_filter(self, mcp_client):
        # Send a memory with low confidence hint
        await mcp_client.call_tool(
            "send_memory",
            {"content": "Unconfirmed rumor", "confidence_hint": "E"},
        )
        # Query with a high min_confidence filter
        result = await mcp_client.call_tool(
            "get_memory", {"query": "rumor", "min_confidence": "A1"}
        )
        data = _parse(result)
        # Should filter out the low-confidence memory (E3 fails both A and 1)
        assert data["meta"]["total_found"] == 0
        assert len(data["memories"]) == 0

    async def test_confidence_filter_uses_and_semantics(self, mcp_client):
        # A3: reliable source (A), but uncorroborated info (3)
        await mcp_client.call_tool(
            "send_memory",
            {"content": "Reliable uncorroborated claim", "confidence_hint": "A"},
        )
        # Filter D1: letter A passes (A <= D), but number 3 fails (3 > 1).
        # AND semantics → excluded.
        result = await mcp_client.call_tool(
            "get_memory",
            {"query": "Reliable uncorroborated", "min_confidence": "D1"},
        )
        data = _parse(result)
        assert data["meta"]["total_found"] == 0

    async def test_confidence_filter_passes_when_both_ok(self, mcp_client):
        # A3: reliable source (A), uncorroborated info (3)
        await mcp_client.call_tool(
            "send_memory",
            {"content": "Reliable uncorroborated claim", "confidence_hint": "A"},
        )
        # Filter D3: letter A passes (A <= D), number 3 passes (3 <= 3).
        # AND semantics → included.
        result = await mcp_client.call_tool(
            "get_memory",
            {"query": "Reliable uncorroborated", "min_confidence": "D3"},
        )
        data = _parse(result)
        assert data["meta"]["total_found"] == 1


# -----------------------------------------------------------------------
# correct_memory
# -----------------------------------------------------------------------


class TestCorrectMemory:
    """Contract tests for the correct_memory tool."""

    async def _send_and_get_id(self, mcp_client, content="test fact"):
        """Helper: send a memory and return its id."""
        result = await mcp_client.call_tool(
            "send_memory", {"content": content}
        )
        return _parse(result)["memory_id"]

    async def test_contest_node(self, mcp_client):
        mem_id = await self._send_and_get_id(mcp_client)
        result = await mcp_client.call_tool(
            "correct_memory",
            {
                "target_id": mem_id,
                "action": "contest",
                "payload": {"reason": "Source unreliable"},
            },
        )
        data = _parse(result)
        assert data["action"] == "contest"
        assert data["status"] == "applied"

    async def test_annotate_node(self, mcp_client):
        mem_id = await self._send_and_get_id(mcp_client)
        result = await mcp_client.call_tool(
            "correct_memory",
            {
                "target_id": mem_id,
                "action": "annotate",
                "payload": {"note": "Additional context"},
            },
        )
        data = _parse(result)
        assert data["action"] == "annotate"
        assert data["status"] == "applied"

    async def test_merge_entities(self, mcp_client):
        id_a = await self._send_and_get_id(mcp_client, "A is a person")
        id_b = await self._send_and_get_id(mcp_client, "B is a person")
        result = await mcp_client.call_tool(
            "correct_memory",
            {
                "target_id": id_a,
                "action": "merge_entities",
                "payload": {"merge_with": id_b},
            },
        )
        data = _parse(result)
        assert data["action"] == "merge_entities"
        assert data["status"] == "applied"

    async def test_split_entity(self, mcp_client):
        mem_id = await self._send_and_get_id(mcp_client)
        result = await mcp_client.call_tool(
            "correct_memory",
            {
                "target_id": mem_id,
                "action": "split_entity",
                "payload": {"split_into": ["part_a", "part_b"]},
            },
        )
        data = _parse(result)
        assert data["action"] == "split_entity"
        assert data["status"] == "applied"

    async def test_reclassify_node(self, mcp_client):
        mem_id = await self._send_and_get_id(mcp_client)
        result = await mcp_client.call_tool(
            "correct_memory",
            {
                "target_id": mem_id,
                "action": "reclassify",
                "payload": {"new_type": "Event"},
            },
        )
        data = _parse(result)
        assert data["action"] == "reclassify"
        assert data["status"] == "applied"

    async def test_rejects_invalid_target_id(self, mcp_client):
        result = await mcp_client.call_tool(
            "correct_memory",
            {
                "target_id": "nonexistent_id_xyz",
                "action": "contest",
            },
        )
        data = _parse(result)
        assert data["status"] == "not_found"
