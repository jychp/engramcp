"""Integration guardrail for sustained-load CPU and memory usage."""

from __future__ import annotations

import json
import tracemalloc
from time import perf_counter
from time import process_time

import pytest
from fastmcp import Client

from engramcp.observability import latency_metrics_snapshot
from engramcp.observability import reset_latency_metrics
from engramcp.server import _get_wm
from engramcp.server import configure
from engramcp.server import mcp
from engramcp.server import shutdown

_INGEST_OPERATIONS = 600
_RETRIEVAL_OPERATIONS = 200
_MAX_WORKING_SET = 120
_MEMORY_GROWTH_BUDGET_BYTES = 30 * 1024 * 1024
_CPU_BUDGET_SECONDS = 30.0
_CPU_PER_RETRIEVAL_BUDGET_SECONDS = 0.2
_WALL_BUDGET_SECONDS = 30.0
_WALL_PER_RETRIEVAL_BUDGET_SECONDS = 0.2


def _parse(result) -> dict:
    return json.loads(result.content[0].text)


@pytest.fixture(autouse=True)
async def _setup_server(redis_container):
    reset_latency_metrics()
    await configure(
        redis_url=redis_container,
        enable_consolidation=False,
        max_size=_MAX_WORKING_SET,
        ttl=3600,
    )
    yield
    await shutdown()


class TestBoundedResourceUsage:
    async def test_sustained_send_and_get_stays_within_cpu_and_memory_budgets(self):
        async with Client(mcp) as client:
            for idx in range(_INGEST_OPERATIONS):
                await client.call_tool(
                    "send_memory",
                    {"content": f"Sustained load signal {idx} route telemetry"},
                )

            wm = _get_wm()
            assert await wm.count() <= _MAX_WORKING_SET

            warmup_result = await client.call_tool(
                "get_memory",
                {"query": "sustained route", "limit": 20},
            )
            assert _parse(warmup_result)["status"] == "ok"

            tracemalloc.start()
            baseline_current, _ = tracemalloc.get_traced_memory()
            wall_start = perf_counter()
            cpu_start = process_time()
            for _ in range(_RETRIEVAL_OPERATIONS):
                result = await client.call_tool(
                    "get_memory",
                    {
                        "query": "sustained route",
                        "max_depth": 1,
                        "include_sources": False,
                        "include_contradictions": False,
                        "limit": 20,
                        "compact": True,
                    },
                )
                parsed = _parse(result)
                assert parsed["status"] == "ok"
                assert parsed["meta"]["returned"] <= 20
            cpu_elapsed = process_time() - cpu_start
            wall_elapsed = perf_counter() - wall_start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        metrics = latency_metrics_snapshot()
        assert metrics["mcp.get_memory"]["count"] >= (_RETRIEVAL_OPERATIONS + 1)
        assert (peak - baseline_current) <= _MEMORY_GROWTH_BUDGET_BYTES
        assert cpu_elapsed <= _CPU_BUDGET_SECONDS
        assert (cpu_elapsed / _RETRIEVAL_OPERATIONS) <= _CPU_PER_RETRIEVAL_BUDGET_SECONDS
        assert wall_elapsed <= _WALL_BUDGET_SECONDS
        assert (wall_elapsed / _RETRIEVAL_OPERATIONS) <= _WALL_PER_RETRIEVAL_BUDGET_SECONDS
