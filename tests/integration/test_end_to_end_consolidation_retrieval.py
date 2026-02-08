"""End-to-end integration tests for consolidation to retrieval flow."""

from __future__ import annotations

import asyncio
import json
import re
import time

import pytest
from fastmcp import Client

from engramcp.config import AuditConfig
from engramcp.config import ConsolidationConfig
from engramcp.engine import LLMAdapter
from engramcp.server import _get_wm
from engramcp.server import configure
from engramcp.server import mcp
from engramcp.server import shutdown


def _parse(result) -> dict:
    return json.loads(result.content[0].text)


async def _wait_until(predicate, *, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(interval)
    return False


class _PromptEchoLLMAdapter(LLMAdapter):
    """Build deterministic extraction claims from prompt fragment contents."""

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        del temperature, max_tokens, timeout_seconds
        contents = [
            line.strip()
            for line in re.findall(r"^Content:\s*(.+)$", prompt, flags=re.MULTILINE)
        ]
        claims = [{"content": content, "claim_type": "Fact"} for content in contents]
        return json.dumps(
            {
                "entities": [],
                "relations": [],
                "claims": claims,
                "fragment_ids_processed": [],
                "errors": [],
            }
        )


@pytest.fixture(autouse=True)
async def _setup_server(redis_container, neo4j_container):
    await configure(
        redis_url=redis_container,
        enable_consolidation=True,
        neo4j_url=neo4j_container,
        llm_adapter=_PromptEchoLLMAdapter(),
        consolidation_config=ConsolidationConfig(fragment_threshold=1),
        audit_config=AuditConfig(enabled=False),
    )
    yield
    await shutdown()


class TestEndToEndConsolidationRetrieval:
    async def test_send_consolidate_get_returns_graph_memory(self):
        async with Client(mcp) as client:
            await client.call_tool(
                "send_memory", {"content": "Storm disrupted route SXM-STT"}
            )

            wm = _get_wm()
            async def _wm_empty() -> bool:
                return await wm.count() == 0

            flushed = await _wait_until(_wm_empty)
            assert flushed

            result = await client.call_tool("get_memory", {"query": "storm"})
            data = _parse(result)

        assert data["meta"]["working_memory_hits"] == 0
        assert data["meta"]["graph_hits"] >= 1
        assert any(
            "Storm disrupted route SXM-STT" in memory["content"]
            for memory in data["memories"]
        )

    async def test_send_consolidate_get_surfaces_contradictions(self):
        async with Client(mcp) as client:
            await client.call_tool(
                "send_memory", {"content": "Alice traveled to Paris"}
            )
            await client.call_tool(
                "send_memory", {"content": "Alice not traveled to Paris"}
            )

            wm = _get_wm()
            async def _wm_empty() -> bool:
                return await wm.count() == 0

            flushed = await _wait_until(_wm_empty)
            assert flushed

            result = await client.call_tool("get_memory", {"query": "Paris"})
            data = _parse(result)

        assert data["meta"]["working_memory_hits"] == 0
        assert data["meta"]["graph_hits"] >= 1
        assert len(data["contradictions"]) >= 1
        assert any(
            "Paris" in contra["contradicting_memory"]["content"]
            for contra in data["contradictions"]
        )
