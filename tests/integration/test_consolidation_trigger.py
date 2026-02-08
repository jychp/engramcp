"""Integration tests for consolidation threshold trigger behavior."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastmcp import Client

import engramcp.server as server_module
from engramcp.config import AuditConfig
from engramcp.config import ConsolidationConfig
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


@pytest.fixture(autouse=True)
async def _setup_server(redis_container, neo4j_container):
    await configure(
        redis_url=redis_container,
        enable_consolidation=True,
        neo4j_url=neo4j_container,
        consolidation_config=ConsolidationConfig(fragment_threshold=2),
        audit_config=AuditConfig(enabled=False),
    )
    yield
    await shutdown()


class TestConsolidationTriggerIntegration:
    async def test_threshold_trigger_flushes_working_memory(self):
        async with Client(mcp) as client:
            await client.call_tool("send_memory", {"content": "trigger fact one"})
            await client.call_tool("send_memory", {"content": "trigger fact two"})

        wm = _get_wm()

        async def _is_empty() -> bool:
            return await wm.count() == 0

        flushed = await _wait_until(_is_empty)
        assert flushed

    async def test_concurrent_sends_retrigger_consolidation(self, monkeypatch):
        pipeline = server_module._consolidation_pipeline
        assert pipeline is not None

        original_run = pipeline.run
        run_calls = 0

        async def slow_run(fragments):
            nonlocal run_calls
            run_calls += 1
            await asyncio.sleep(0.12)
            return await original_run(fragments)

        monkeypatch.setattr(pipeline, "run", slow_run)

        async with Client(mcp) as client:
            await client.call_tool("send_memory", {"content": "concurrent fact 1"})
            await client.call_tool("send_memory", {"content": "concurrent fact 2"})
            await asyncio.sleep(0.02)
            await asyncio.gather(
                client.call_tool("send_memory", {"content": "concurrent fact 3"}),
                client.call_tool("send_memory", {"content": "concurrent fact 4"}),
            )

        wm = _get_wm()

        async def _is_empty() -> bool:
            return await wm.count() == 0

        empty = await _wait_until(_is_empty)
        assert empty
        assert run_calls >= 2
