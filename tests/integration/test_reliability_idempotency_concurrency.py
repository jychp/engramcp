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
from engramcp.engine import LLMError
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
        fragment_ids = re.findall(r"^--- Fragment (.+?) ---$", prompt, flags=re.MULTILINE)
        contents = [
            line.strip()
            for line in re.findall(r"^Content:\s*(.+)$", prompt, flags=re.MULTILINE)
        ]
        claims = []
        for idx, content in enumerate(contents):
            source_fragment_id = fragment_ids[idx] if idx < len(fragment_ids) else ""
            claim = {"content": content, "claim_type": "Fact"}
            if source_fragment_id:
                claim["source_fragment_ids"] = [source_fragment_id]
            claims.append(claim)
        return json.dumps(
            {
                "entities": [],
                "relations": [],
                "claims": claims,
                "fragment_ids_processed": [],
                "errors": [],
            }
        )


class _FailOncePromptEchoLLMAdapter(_PromptEchoLLMAdapter):
    """Fail on first call, then behave like prompt-echo adapter."""

    def __init__(self) -> None:
        self._failed_once = False

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        if not self._failed_once:
            self._failed_once = True
            raise LLMError("transient upstream failure")
        return await super().complete(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )


class _FailFirstBatchThenEchoLLMAdapter(_PromptEchoLLMAdapter):
    """Fail first extraction batch, then succeed for subsequent batches."""

    def __init__(self) -> None:
        self._call_count = 0

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        self._call_count += 1
        if self._call_count == 1:
            raise LLMError("first batch failed")
        return await super().complete(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
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


@pytest.fixture
async def llm_fail_once_client(redis_container, neo4j_container):
    await configure(
        redis_url=redis_container,
        enable_consolidation=True,
        neo4j_url=neo4j_container,
        llm_adapter=_FailOncePromptEchoLLMAdapter(),
        consolidation_config=ConsolidationConfig(fragment_threshold=1),
        audit_config=AuditConfig(enabled=False),
    )
    yield
    await shutdown()


@pytest.fixture
async def partial_batch_fail_client(redis_container, neo4j_container):
    await configure(
        redis_url=redis_container,
        enable_consolidation=True,
        neo4j_url=neo4j_container,
        llm_adapter=_FailFirstBatchThenEchoLLMAdapter(),
        consolidation_config=ConsolidationConfig(
            fragment_threshold=2,
            extraction_batch_size=1,
        ),
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

    async def test_llm_transient_failure_keeps_fragments_until_retry_succeeds(
        self,
        llm_fail_once_client,
        graph_store: GraphStore,
    ):
        wm = _get_wm()

        async with Client(mcp) as client:
            await client.call_tool("send_memory", {"content": "Transient fail fact one"})

            async def _has_one_fragment() -> bool:
                return await wm.count() == 1

            assert await _wait_until(_has_one_fragment)

            await client.call_tool("send_memory", {"content": "Transient fail fact two"})

        async def _is_empty() -> bool:
            return await wm.count() == 0

        assert await _wait_until(_is_empty)

        claims = await graph_store.find_claim_nodes()
        contents = [
            getattr(node, "content", "")
            for node in claims
            if str(getattr(node, "content", "")).startswith("Transient fail fact")
        ]
        assert sorted(contents) == [
            "Transient fail fact one",
            "Transient fail fact two",
        ]

    async def test_partial_graph_write_retry_completes_without_duplicate_claims(
        self,
        idempotency_client,
        monkeypatch,
        graph_store: GraphStore,
    ):
        wm = _get_wm()
        async with Client(mcp) as client:
            await client.call_tool("send_memory", {"content": "Partial write probe fact"})

        fragments = await wm.get_recent(limit=10)
        assert len(fragments) == 1

        pipeline = server_module._consolidation_pipeline
        assert pipeline is not None

        original_create_relationship = pipeline._graph.create_relationship
        should_fail = True

        async def fail_once_on_sourced_from(from_id, to_id, rel):
            nonlocal should_fail
            if should_fail and rel.rel_type == "SOURCED_FROM":
                should_fail = False
                raise RuntimeError("simulated graph failure")
            return await original_create_relationship(from_id, to_id, rel)

        monkeypatch.setattr(
            pipeline._graph,
            "create_relationship",
            fail_once_on_sourced_from,
        )

        with pytest.raises(RuntimeError, match="simulated graph failure"):
            await pipeline.run(fragments)

        claim_id = next(
            node.id
            for node in await graph_store.find_claim_nodes()
            if getattr(node, "content", "") == "Partial write probe fact"
        )
        partial_sourced_from = await graph_store.get_relationships(
            claim_id,
            rel_type="SOURCED_FROM",
            direction="outgoing",
        )
        assert partial_sourced_from == []

        await pipeline.run(fragments)

        matching_claims = [
            node
            for node in await graph_store.find_claim_nodes()
            if getattr(node, "content", "") == "Partial write probe fact"
        ]
        assert len(matching_claims) == 1

        sourced_from = await graph_store.get_relationships(
            claim_id,
            rel_type="SOURCED_FROM",
            direction="outgoing",
        )
        assert len(sourced_from) == 1

    async def test_partial_extraction_error_with_mutation_clears_batch(
        self,
        partial_batch_fail_client,
        graph_store: GraphStore,
    ):
        wm = _get_wm()
        async with Client(mcp) as client:
            await client.call_tool("send_memory", {"content": "Partial batch fact one"})
            await client.call_tool("send_memory", {"content": "Partial batch fact two"})

            async def _first_batch_cleared() -> bool:
                return await wm.count() == 0

            assert await _wait_until(_first_batch_cleared)

            await client.call_tool("send_memory", {"content": "Partial batch fact three"})
            await client.call_tool("send_memory", {"content": "Partial batch fact four"})

        async def _is_empty() -> bool:
            return await wm.count() == 0

        assert await _wait_until(_is_empty)

        claims = await graph_store.find_claim_nodes()
        contents = [
            getattr(node, "content", "")
            for node in claims
            if str(getattr(node, "content", "")).startswith("Partial batch fact")
        ]
        assert len(contents) == 3
        assert "Partial batch fact three" in contents
        assert "Partial batch fact four" in contents
        assert (
            "Partial batch fact one" in contents
            or "Partial batch fact two" in contents
        )
