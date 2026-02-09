"""Tier 2 semi-real scenario: corporate timeline evolution tracking."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import pytest
from fastmcp import Client

from engramcp.config import AuditConfig
from engramcp.config import scenario_eval_consolidation_config
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
    Path(__file__).resolve().parent / "fixtures" / "tier2_curated_corporate_timeline.json"
)
_FRAGMENT_PATTERN = re.compile(
    r"^\[(?P<date>\d{4}-\d{2}-\d{2})\]\s*(?P<speaker>[A-Za-z]+):\s*(?P<message>.+)$"
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


class _CorporateTimelineAdapter(LLMAdapter):
    """Deterministic extraction from timeline-style fragments."""

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

        entities: dict[str, dict] = {}
        claims: list[dict] = []
        for index, content in enumerate(contents):
            match = _FRAGMENT_PATTERN.match(content)
            speaker = match.group("speaker") if match else "Unknown"
            message = match.group("message") if match else content
            fragment_id = fragment_ids[index] if index < len(fragment_ids) else ""

            entities.setdefault(
                speaker,
                {
                    "name": speaker,
                    "type": "Agent",
                    "source_fragment_ids": [fragment_id] if fragment_id else [],
                },
            )
            if fragment_id and fragment_id not in entities[speaker]["source_fragment_ids"]:
                entities[speaker]["source_fragment_ids"].append(fragment_id)

            claims.append(
                {
                    "content": f"{speaker}: {message}",
                    "claim_type": "Fact",
                    "involved_entities": [speaker],
                    "source_fragment_ids": [fragment_id] if fragment_id else [],
                }
            )

        return json.dumps(
            {
                "entities": list(entities.values()),
                "relations": [],
                "claims": claims,
                "fragment_ids_processed": fragment_ids,
                "errors": [],
            }
        )


@pytest.fixture(autouse=True)
async def _setup_server(redis_container, neo4j_container):
    await configure(
        redis_url=redis_container,
        enable_consolidation=True,
        neo4j_url=neo4j_container,
        llm_adapter=_CorporateTimelineAdapter(),
        consolidation_config=scenario_eval_consolidation_config(),
        audit_config=AuditConfig(enabled=False),
    )
    yield
    await shutdown()


class TestTier2CorporateTimeline:
    async def test_tracks_deadline_position_changes_without_false_contradictions(self):
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

        memories = data["memories"]
        contents = [memory["content"].lower() for memory in memories]
        changed_agents = set()
        if any("alice: the deadline is march 15" in content for content in contents) and any(
            "alice: march 20 is fine" in content for content in contents
        ):
            changed_agents.add("alice")
        if any("bob: i agree with the march 15 deadline" in content for content in contents) and any(
            "bob: march 20 is more realistic" in content for content in contents
        ) and any("bob: let's target march 25 now" in content for content in contents):
            changed_agents.add("bob")

        timeline_statement_count = sum(
            "deadline" in content or "march 20" in content or "march 25" in content
            for content in contents
        )
        carol_consistency_hits = sum("carol: scope concerns" in content for content in contents)

        emit_scenario_metric(
            scenario=scenario_name,
            tier="tier2",
            metric_class="timeline_change_tracking",
            values={
                "graph_hits": data["meta"]["graph_hits"],
                "returned_memories": len(memories),
                "changed_agents_count": len(changed_agents),
                "timeline_statement_count": timeline_statement_count,
                "carol_consistency_hits": carol_consistency_hits,
                "contradictions": len(data["contradictions"]),
            },
        )

        assert data["meta"]["working_memory_hits"] == 0, ctx
        assert data["meta"]["graph_hits"] >= THRESHOLDS.min_graph_hits, ctx
        assert len(memories) >= THRESHOLDS.min_memories, ctx
        assert len(changed_agents) >= THRESHOLDS.min_changed_agents_in_timeline, ctx
        assert (
            timeline_statement_count >= THRESHOLDS.min_timeline_statements
        ), ctx
        assert (
            carol_consistency_hits >= THRESHOLDS.min_consistent_agent_hits
        ), ctx
        assert (
            len(data["contradictions"])
            <= THRESHOLDS.max_contradictions_for_temporal_evolution
        ), ctx
