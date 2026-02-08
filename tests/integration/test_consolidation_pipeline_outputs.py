"""Integration tests for consolidation contradiction and abstraction outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from engramcp.audit import AuditLogger
from engramcp.config import AuditConfig
from engramcp.config import ConsolidationConfig
from engramcp.engine.consolidation import ConsolidationPipeline
from engramcp.engine.schemas import ExtractedClaim
from engramcp.engine.schemas import ExtractedEntity
from engramcp.engine.schemas import ExtractedRelation
from engramcp.engine.schemas import ExtractionResult
from engramcp.graph import EntityResolver
from engramcp.graph import MergeExecutor
from engramcp.memory.schemas import MemoryFragment
from engramcp.models.nodes import Fact


@dataclass
class _StaticExtractionEngine:
    result: ExtractionResult

    async def extract(self, fragments: list[MemoryFragment]) -> ExtractionResult:
        del fragments
        return self.result


def _make_pipeline(
    *,
    graph_store,
    extraction_result: ExtractionResult,
    config: ConsolidationConfig | None = None,
) -> ConsolidationPipeline:
    return ConsolidationPipeline(
        extraction_engine=_StaticExtractionEngine(extraction_result),
        entity_resolver=EntityResolver(),
        merge_executor=MergeExecutor(graph_store),
        graph_store=graph_store,
        audit_logger=AuditLogger(AuditConfig(enabled=False)),
        config=config,
    )


class TestConsolidationPipelineOutputs:
    async def test_detects_contradiction_against_existing_claim(self, graph_store):
        existing_id = await graph_store.create_node(
            Fact(content="Alice not traveled to Paris")
        )
        extraction = ExtractionResult(
            claims=[
                ExtractedClaim(
                    content="Alice traveled to Paris",
                    claim_type="Fact",
                )
            ]
        )
        pipeline = _make_pipeline(graph_store=graph_store, extraction_result=extraction)

        result = await pipeline.run([MemoryFragment(content="new fragment")])

        assert result.contradictions_detected == 1
        facts = await graph_store.find_by_label("Fact")
        new_fact = next(f for f in facts if f.content == "Alice traveled to Paris")
        rels = await graph_store.get_relationships(
            new_fact.id, rel_type="CONTRADICTS", direction="outgoing"
        )
        assert any(rel["to_id"] == existing_id for rel in rels)

    async def test_creates_pattern_concept_rule_abstractions(self, graph_store):
        extraction = ExtractionResult(
            entities=[
                ExtractedEntity(name="Alice", type="Agent"),
                ExtractedEntity(name="Bob", type="Agent"),
            ],
            relations=[
                ExtractedRelation(
                    from_entity="Alice",
                    to_entity="Bob",
                    relation_type="CAUSED_BY",
                )
            ],
            claims=[
                ExtractedClaim(content="Alice traveled to Paris", claim_type="Fact"),
                ExtractedClaim(content="Alice traveled to Paris", claim_type="Fact"),
                ExtractedClaim(content="Bob visited London", claim_type="Fact"),
                ExtractedClaim(content="Bob visited London", claim_type="Fact"),
            ],
        )
        pipeline = _make_pipeline(
            graph_store=graph_store,
            extraction_result=extraction,
            config=ConsolidationConfig(pattern_min_occurrences=2),
        )

        result = await pipeline.run([MemoryFragment(content="batch fragment")])

        assert result.patterns_created == 2
        assert result.concepts_created == 1
        assert result.rules_created == 1

        patterns = await graph_store.find_by_label("Pattern")
        concepts = await graph_store.find_by_label("Concept")
        rules = await graph_store.find_by_label("Rule")
        assert len(patterns) == 2
        assert len(concepts) == 1
        assert len(rules) == 1

        rule_to_concept = await graph_store.get_relationships(
            rules[0].id, rel_type="DERIVED_FROM", direction="outgoing"
        )
        assert any(rel["to_id"] == concepts[0].id for rel in rule_to_concept)
