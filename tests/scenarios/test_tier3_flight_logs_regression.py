"""Tier 3 real-data style scenario: flight-log subset ingestion and retrieval."""

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

pytestmark = [pytest.mark.tier3]

_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "tier3_flight_logs_subset_runtime.json"
)
_FRAGMENT_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+\|\s+(?P<origin>[A-Z]{3})\s+->\s+(?P<destination>[A-Z]{3})\s+\|\s+passengers:\s*(?P<passengers>.+)$"
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


class _Tier3FlightLogAdapter(LLMAdapter):
    """Deterministic extraction with alias normalization for known variants."""

    _ALIAS_MAP = {
        "epstein, jeffrey": "Jeffrey Epstein",
        "jeff epstein": "Jeffrey Epstein",
        "gm": "Ghislaine Maxwell",
    }

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
            if not match:
                continue

            flight_date = match.group("date")
            origin = match.group("origin")
            destination = match.group("destination")
            passengers_raw = [
                name.strip() for name in match.group("passengers").split(";") if name.strip()
            ]
            passengers = [
                self._ALIAS_MAP.get(name.lower(), name) for name in passengers_raw
            ]
            fragment_id = fragment_ids[index] if index < len(fragment_ids) else ""

            for name in passengers:
                entities.setdefault(
                    name,
                    {
                        "name": name,
                        "type": "Agent",
                        "source_fragment_ids": [fragment_id] if fragment_id else [],
                    },
                )
                if fragment_id and fragment_id not in entities[name]["source_fragment_ids"]:
                    entities[name]["source_fragment_ids"].append(fragment_id)

            claims.append(
                {
                    "content": (
                        f"Flight on {flight_date} from {origin} to {destination} included "
                        f"{', '.join(passengers)}."
                    ),
                    "claim_type": "Fact",
                    "involved_entities": passengers,
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
        llm_adapter=_Tier3FlightLogAdapter(),
        consolidation_config=ConsolidationConfig(fragment_threshold=1),
        audit_config=AuditConfig(enabled=False),
    )
    yield
    await shutdown()


class TestTier3FlightLogsRegression:
    async def test_flight_log_subset_metrics_and_retrieval_quality(self, neo4j_driver):
        fixture = _load_fixture()
        scenario_name = fixture["scenario_name"]
        fragments: list[str] = fixture["fragments"]
        query: str = fixture["query"]
        expected_fact_snippets: list[str] = fixture["expected_fact_snippets"]
        expected_core_agents = {str(name).lower() for name in fixture["expected_core_agents"]}
        alias_variants = {str(name).lower() for name in fixture["alias_variants"]}

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

        memory_contents = [str(memory["content"]) for memory in data["memories"]]
        normalized_contents = [content.lower() for content in memory_contents]
        citation_hits = sum(1 for memory in data["memories"] if memory.get("sources"))
        contradictions_count = len(data["contradictions"])

        matched_snippets = sum(
            1
            for snippet in expected_fact_snippets
            if any(snippet.lower() in content for content in normalized_contents)
        )
        precision_proxy = matched_snippets / len(memory_contents) if memory_contents else 0.0
        recall_proxy = (
            matched_snippets / len(expected_fact_snippets) if expected_fact_snippets else 0.0
        )

        async with neo4j_driver.session() as session:
            agent_rows = await session.run("MATCH (a:Agent) RETURN toLower(a.name) AS name")
            agent_names = {
                str(row["name"]).strip().lower() for row in await agent_rows.data() if row.get("name")
            }

        false_split_count = sum(1 for alias in alias_variants if alias in agent_names)
        false_merge_count = (
            1 if len(agent_names.intersection(expected_core_agents)) < len(expected_core_agents) else 0
        )

        emit_scenario_metric(
            scenario=scenario_name,
            tier="tier3",
            metric_class="extraction_precision_recall_proxy",
            values={
                "precision_proxy": precision_proxy,
                "recall_proxy": recall_proxy,
            },
        )
        emit_scenario_metric(
            scenario=scenario_name,
            tier="tier3",
            metric_class="entity_merge_precision",
            values={
                "false_merge_count": false_merge_count,
                "false_split_count": false_split_count,
            },
        )
        emit_scenario_metric(
            scenario=scenario_name,
            tier="tier3",
            metric_class="retrieval_usefulness",
            values={
                "graph_hits": data["meta"]["graph_hits"],
                "returned_memories": len(data["memories"]),
                "citation_hits": citation_hits,
                "contradictions": contradictions_count,
            },
        )

        assert data["meta"]["working_memory_hits"] == 0, ctx
        assert data["meta"]["graph_hits"] >= THRESHOLDS.min_graph_hits, ctx
        assert len(data["memories"]) >= THRESHOLDS.min_memories, ctx
        assert citation_hits >= THRESHOLDS.min_citation_hits, ctx
        assert (
            contradictions_count <= THRESHOLDS.max_contradictions_for_tier3_retrieval
        ), ctx

        assert precision_proxy >= THRESHOLDS.min_extraction_precision_proxy, ctx
        assert recall_proxy >= THRESHOLDS.min_extraction_recall_proxy, ctx
        assert false_merge_count <= THRESHOLDS.max_false_merges, ctx
        assert false_split_count <= THRESHOLDS.max_false_splits, ctx
