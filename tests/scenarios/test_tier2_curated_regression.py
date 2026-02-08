"""Tier 2 curated scenario regression tests (deterministic, CI-safe)."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import pytest
from fastmcp import Client

from engramcp.config import AuditConfig
from engramcp.config import ConsolidationConfig
from engramcp.evaluation import THRESHOLDS
from engramcp.engine import LLMAdapter
from engramcp.server import _get_wm
from engramcp.server import configure
from engramcp.server import mcp
from engramcp.server import shutdown
from tests.scenarios.helpers.metrics import emit_scenario_metric
from tests.scenarios.helpers.reporting import build_failure_context

pytestmark = [pytest.mark.tier2]

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "tier2_curated_travel_conflict.json"
)


def _load_fixture() -> dict:
    with _FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    """Deterministic adapter: each prompt fragment becomes one Fact claim."""

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


class TestTier2CuratedRegression:
    async def test_curated_conflict_is_retrieved_with_contradiction_signal(self):
        fixture = _load_fixture()
        scenario_name = fixture["scenario_name"]
        fragments: list[str] = fixture["fragments"]
        query: str = fixture["query"]

        async with Client(mcp) as client:
            for fragment in fragments:
                await client.call_tool("send_memory", {"content": fragment})

            wm = _get_wm()

            async def _wm_empty() -> bool:
                return await wm.count() == 0

            assert await _wait_until(_wm_empty)
            result = await client.call_tool("get_memory", {"query": query})
            data = _parse(result)

        ctx = build_failure_context(
            scenario=scenario_name,
            query=query,
            response=data,
            fragments=fragments,
        )

        emit_scenario_metric(
            scenario=scenario_name,
            tier="tier2",
            metric_class="contradiction_coverage",
            values={
                "working_memory_hits": data["meta"]["working_memory_hits"],
                "graph_hits": data["meta"]["graph_hits"],
                "returned_memories": len(data["memories"]),
                "contradictions": len(data["contradictions"]),
            },
        )

        assert data["meta"]["working_memory_hits"] == 0, ctx
        assert data["meta"]["graph_hits"] >= THRESHOLDS.min_graph_hits, ctx
        assert len(data["memories"]) >= 2, ctx
        assert any("John Doe" in memory["content"] for memory in data["memories"]), ctx
        assert len(data["contradictions"]) >= THRESHOLDS.min_contradictions, ctx
