"""Integration tests for consolidation idempotency and overlap reliability."""

from __future__ import annotations

import asyncio
import json
import re
import time

import pytest
from fastmcp import Client

import engramcp.server as server_module
from engramcp.config import AuditConfig
from engramcp.config import ConsolidationConfig
from engramcp.engine import LLMAdapter
from engramcp.graph import GraphStore
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


@pytest.fixture
async def idempotency_client(redis_container, neo4j_container):
    await configure(
        redis_url=redis_container,
        enable_consolidation=True,
        neo4j_url=neo4j_container,
        llm_adapter=_PromptEchoLLMAdapter(),
        consolidation_config=ConsolidationConfig(fragment_threshold=100),
        audit_config=AuditConfig(enabled=False),
    )
    yield
    await shutdown()


@pytest.fixture
async def overlap_client(redis_container, neo4j_container):
    await configure(
        redis_url=redis_container,
        enable_consolidation=True,
        neo4j_url=neo4j_container,
        llm_adapter=_PromptEchoLLMAdapter(),
        consolidation_config=ConsolidationConfig(fragment_threshold=2),
        audit_config=AuditConfig(enabled=False),
    )
    yield
    await shutdown()


class TestReliabilityIdempotencyConcurrency:
    async def test_repeated_consolidation_run_is_idempotent(
        self,
        idempotency_client,
        graph_store: GraphStore,
    ):
        async with Client(mcp) as client:
            await client.call_tool(
                "send_memory", {"content": "Idempotency probe fact alpha"}
            )
            await client.call_tool(
                "send_memory", {"content": "Idempotency probe fact beta"}
            )

        wm = _get_wm()
        fragments = await wm.get_recent(limit=10)
        assert len(fragments) == 2

        pipeline = server_module._consolidation_pipeline
        assert pipeline is not None

        await pipeline.run(fragments)
        await pipeline.run(fragments)

        claims = await graph_store.find_claim_nodes()
        contents = [
            getattr(node, "content", "")
            for node in claims
            if str(getattr(node, "content", "")).startswith("Idempotency probe fact")
        ]
        assert sorted(contents) == [
            "Idempotency probe fact alpha",
            "Idempotency probe fact beta",
        ]

    async def test_flush_overlap_retries_without_loss_or_duplication(
        self,
        overlap_client,
        monkeypatch,
        graph_store: GraphStore,
    ):
        pipeline = server_module._consolidation_pipeline
        assert pipeline is not None

        original_run = pipeline.run

        async def slow_run(fragments):
            await asyncio.sleep(0.12)
            return await original_run(fragments)

        monkeypatch.setattr(pipeline, "run", slow_run)

        async with Client(mcp) as client:
            await client.call_tool("send_memory", {"content": "Overlap fact one"})
            await client.call_tool("send_memory", {"content": "Overlap fact two"})
            await asyncio.sleep(0.02)
            await asyncio.gather(
                client.call_tool("send_memory", {"content": "Overlap fact three"}),
                client.call_tool("send_memory", {"content": "Overlap fact four"}),
            )

        wm = _get_wm()

        async def _is_empty() -> bool:
            return await wm.count() == 0

        empty = await _wait_until(_is_empty)
        assert empty

        claims = await graph_store.find_claim_nodes()
        overlap_contents = [
            getattr(node, "content", "")
            for node in claims
            if str(getattr(node, "content", "")).startswith("Overlap fact")
        ]
        assert sorted(overlap_contents) == [
            "Overlap fact four",
            "Overlap fact one",
            "Overlap fact three",
            "Overlap fact two",
        ]
