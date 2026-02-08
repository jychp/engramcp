"""Tier 2 curated scenario tests for corroboration and derivation traceability."""

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
from engramcp.models.confidence import Credibility
from engramcp.models.relations import Cites
from engramcp.models.relations import SourcedFrom
from engramcp.server import _get_wm
from engramcp.server import configure
from engramcp.server import mcp
from engramcp.server import shutdown
from tests.scenarios.helpers.metrics import emit_scenario_metric
from tests.scenarios.helpers.reporting import build_failure_context

pytestmark = [pytest.mark.tier2]

_CORROBORATION_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "tier2_curated_corroboration_sources.json"
)
_DERIVATION_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "tier2_curated_derivation_traceability.json"
)


def _load_fixture(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
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


class _ScenarioStructuredAdapter(LLMAdapter):
    """Deterministic adapter with canonicalization and causal abstraction signal."""

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
        for index, content in enumerate(contents):
            normalized = content
            if "AC123" in content and "JFK" in content and "09:00" in content:
                normalized = "Flight AC123 departed JFK at 09:00."

            claim = {
                "content": normalized,
                "claim_type": "Fact",
                "source_fragment_ids": [fragment_ids[index]] if index < len(fragment_ids) else [],
            }
            claims.append(claim)

        relations = []
        entities = []
        # Emit one deterministic causal relation to unlock Rule abstraction when
        # concept formation conditions are satisfied.
        if len(claims) >= 4:
            entities = [
                {
                    "name": "Signal Alpha",
                    "type": "Agent",
                    "source_fragment_ids": [fragment_ids[0]] if fragment_ids else [],
                },
                {
                    "name": "Signal Beta",
                    "type": "Agent",
                    "source_fragment_ids": [fragment_ids[-1]] if fragment_ids else [],
                },
            ]
            relations = [
                {
                    "from_entity": "Signal Alpha",
                    "to_entity": "Signal Beta",
                    "relation_type": "LEADS_TO",
                    "source_fragment_ids": [fragment_ids[0]] if fragment_ids else [],
                }
            ]

        return json.dumps(
            {
                "entities": entities,
                "relations": relations,
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
        llm_adapter=_ScenarioStructuredAdapter(),
        consolidation_config=scenario_eval_consolidation_config(),
        audit_config=AuditConfig(enabled=False),
    )
    yield
    await shutdown()


class TestTier2CorroborationDerivation:
    async def test_curated_sources_show_corroboration_signal(self):
        fixture = _load_fixture(_CORROBORATION_FIXTURE)
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

        assert data["meta"]["working_memory_hits"] == 0, ctx
        assert data["meta"]["graph_hits"] >= 1, ctx

        corroborated = [
            memory
            for memory in data["memories"]
            if memory["content"] == "Flight AC123 departed JFK at 09:00."
        ]
        assert corroborated, ctx
        unique_source_ids = {
            source["id"]
            for memory in corroborated
            for source in memory.get("sources", [])
            if source.get("id")
        }
        emit_scenario_metric(
            scenario=scenario_name,
            tier="tier2",
            metric_class="corroboration",
            values={
                "graph_hits": data["meta"]["graph_hits"],
                "returned_memories": len(data["memories"]),
                "corroborated_candidates": len(corroborated),
                "unique_source_ids": len(unique_source_ids),
            },
        )
        assert len(unique_source_ids) >= THRESHOLDS.min_unique_sources_for_corroboration, ctx

    async def test_confidence_progression_reflects_source_independence(
        self,
        confidence_engine,
        graph_store,
    ):
        fixture = _load_fixture(_CORROBORATION_FIXTURE)
        scenario_name = f"{fixture['scenario_name']}_confidence_progression"
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

        corroborated = [
            memory
            for memory in data["memories"]
            if memory["content"] == "Flight AC123 departed JFK at 09:00."
            and memory.get("sources")
        ]
        assert len(corroborated) >= 2, ctx

        first = corroborated[0]
        second = next(
            (memory for memory in corroborated[1:] if memory["id"] != first["id"]),
            None,
        )
        assert second is not None, ctx

        fact_id = first["id"]
        first_source_id = first["sources"][0]["id"]
        second_source_id = second["sources"][0]["id"]
        assert first_source_id and second_source_id, ctx
        assert first_source_id != second_source_id, ctx

        await graph_store.create_relationship(
            fact_id,
            second_source_id,
            SourcedFrom(credibility=Credibility.SIX),
        )

        independent_count, _ = await confidence_engine.check_corroboration(fact_id)
        independent_assessment = await confidence_engine.assess_credibility(fact_id)

        await graph_store.create_relationship(second_source_id, first_source_id, Cites())

        dependent_count, _ = await confidence_engine.check_corroboration(fact_id)
        dependent_assessment = await confidence_engine.assess_credibility(fact_id)

        emit_scenario_metric(
            scenario=scenario_name,
            tier="tier2",
            metric_class="confidence_progression",
            values={
                "independent_source_count": independent_count,
                "independent_credibility": independent_assessment.credibility.value,
                "dependent_source_count": dependent_count,
                "dependent_credibility": dependent_assessment.credibility.value,
            },
        )

        assert (
            independent_count
            >= THRESHOLDS.min_independent_sources_for_confidence_upgrade
        ), ctx
        assert (
            independent_assessment.credibility.value
            == THRESHOLDS.expected_corroborated_credibility
        ), ctx
        assert dependent_count == THRESHOLDS.expected_dependent_independent_sources, ctx
        assert (
            dependent_assessment.credibility.value
            == THRESHOLDS.expected_dependent_credibility
        ), ctx
        assert int(independent_assessment.credibility.value) < int(
            dependent_assessment.credibility.value
        ), ctx

    async def test_curated_derivation_is_traceable_in_get_memory_output(self):
        fixture = _load_fixture(_DERIVATION_FIXTURE)
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

        assert data["meta"]["working_memory_hits"] == 0, ctx
        assert data["meta"]["graph_hits"] >= 1, ctx

        rule_entries = [
            memory for memory in data["memories"] if memory["type"] == "Rule"
        ]
        derivation_depth = rule_entries[0]["properties"].get("derivation_depth") if rule_entries else None
        emit_scenario_metric(
            scenario=scenario_name,
            tier="tier2",
            metric_class="derivation_traceability",
            values={
                "graph_hits": data["meta"]["graph_hits"],
                "returned_memories": len(data["memories"]),
                "rule_entries": len(rule_entries),
                "rule_derivation_depth": derivation_depth,
                "has_derivation_run_id": bool(
                    rule_entries and rule_entries[0]["properties"].get("derivation_run_id")
                ),
            },
        )
        assert rule_entries, ctx
        assert len(rule_entries) >= THRESHOLDS.min_rule_entries_for_derivation, ctx
        assert derivation_depth == THRESHOLDS.expected_rule_derivation_depth, ctx
        assert rule_entries[0]["properties"].get("derivation_run_id"), ctx
