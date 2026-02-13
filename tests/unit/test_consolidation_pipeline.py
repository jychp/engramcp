"""Tests for the consolidation pipeline orchestrator (Layer 4).

All pure unit tests — GraphStore, ExtractionEngine, EntityResolver,
MergeExecutor, and AuditLogger are fully mocked (AsyncMock). No
containers needed.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from engramcp.audit.schemas import AuditEventType
from engramcp.engine.consolidation import ConsolidationPipeline
from engramcp.engine.consolidation import ConsolidationRunResult
from engramcp.engine.schemas import ExtractedClaim
from engramcp.engine.schemas import ExtractedEntity
from engramcp.engine.schemas import ExtractedRelation
from engramcp.engine.schemas import ExtractionResult
from engramcp.engine.schemas import TemporalInfo
from engramcp.graph.entity_resolution import ExistingEntity
from engramcp.graph.entity_resolution import MergeResult
from engramcp.graph.entity_resolution import ResolutionAction
from engramcp.graph.entity_resolution import ResolutionCandidate
from engramcp.memory.schemas import MemoryFragment
from engramcp.models.nodes import Agent
from engramcp.models.nodes import AgentType
from engramcp.models.nodes import Fact


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def extraction_engine():
    engine = AsyncMock()
    engine.extract.return_value = ExtractionResult()
    return engine


@pytest.fixture()
def entity_resolver():
    return AsyncMock()


@pytest.fixture()
def merge_executor():
    return AsyncMock()


@pytest.fixture()
def graph_store():
    store = AsyncMock()
    store.find_by_label.return_value = []
    store.find_claim_nodes.return_value = []
    store.create_node.return_value = "node-id"
    store.create_relationship.return_value = True
    return store


@pytest.fixture()
def audit_logger():
    return AsyncMock()


@pytest.fixture()
def pipeline(extraction_engine, entity_resolver, merge_executor, graph_store, audit_logger):
    return ConsolidationPipeline(
        extraction_engine=extraction_engine,
        entity_resolver=entity_resolver,
        merge_executor=merge_executor,
        graph_store=graph_store,
        audit_logger=audit_logger,
    )


def _make_fragment(**overrides) -> MemoryFragment:
    defaults = {"content": "Test fragment content"}
    defaults.update(overrides)
    return MemoryFragment(**defaults)


def _make_entity(**overrides) -> ExtractedEntity:
    defaults = {"name": "Alice", "type": "Agent"}
    defaults.update(overrides)
    return ExtractedEntity(**defaults)


def _make_claim(**overrides) -> ExtractedClaim:
    defaults = {"content": "Some fact", "claim_type": "Fact"}
    defaults.update(overrides)
    return ExtractedClaim(**defaults)


def _make_relation(**overrides) -> ExtractedRelation:
    defaults = {
        "from_entity": "Alice",
        "to_entity": "Bob",
        "relation_type": "CONCERNS",
    }
    defaults.update(overrides)
    return ExtractedRelation(**defaults)


# ===================================================================
# Basic flow
# ===================================================================


class TestBasicFlow:
    async def test_empty_fragments_returns_early(self, pipeline, extraction_engine):
        result = await pipeline.run([])
        assert isinstance(result, ConsolidationRunResult)
        assert result.fragments_processed == 0
        extraction_engine.extract.assert_not_called()

    async def test_runs_extraction_on_fragments(self, pipeline, extraction_engine):
        fragments = [_make_fragment()]
        await pipeline.run(fragments)
        extraction_engine.extract.assert_awaited_once_with(fragments)

    async def test_run_id_is_unique_per_call(self, pipeline):
        r1 = await pipeline.run([_make_fragment()])
        r2 = await pipeline.run([_make_fragment()])
        assert r1.run_id != r2.run_id


# ===================================================================
# Entity resolution + graph integration
# ===================================================================


class TestEntityResolution:
    async def test_creates_new_entity_nodes_in_graph(
        self, pipeline, extraction_engine, entity_resolver, graph_store
    ):
        entity = _make_entity(name="Alice", type="Agent")
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[entity],
        )
        entity_resolver.resolve.return_value = ResolutionCandidate(
            entity_name="Alice",
            existing_node_id=None,
            existing_name=None,
            score=0.0,
            action=ResolutionAction.create_new,
            method="level_1",
        )

        result = await pipeline.run([_make_fragment()])
        assert result.entities_created == 1
        graph_store.create_node.assert_called()

    async def test_merges_matching_entities(
        self, pipeline, extraction_engine, entity_resolver, merge_executor, graph_store
    ):
        entity = _make_entity(name="Alice", type="Agent")
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[entity],
        )
        entity_resolver.resolve.return_value = ResolutionCandidate(
            entity_name="Alice",
            existing_node_id="existing-123",
            existing_name="Alice Smith",
            score=0.95,
            action=ResolutionAction.merge,
            method="level_1",
        )
        merge_executor.execute_merge.return_value = MergeResult(
            survivor_id="existing-123",
            absorbed_id="new-node-id",
            aliases_added=["Alice"],
            relations_transferred=0,
        )
        # The pipeline creates a new node first, then merges it
        graph_store.create_node.return_value = "new-node-id"

        result = await pipeline.run([_make_fragment()])
        assert result.entities_merged == 1
        merge_executor.execute_merge.assert_awaited_once()

    async def test_creates_possibly_same_as_for_links(
        self, pipeline, extraction_engine, entity_resolver, graph_store
    ):
        entity = _make_entity(name="Alice", type="Agent")
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[entity],
        )
        entity_resolver.resolve.return_value = ResolutionCandidate(
            entity_name="Alice",
            existing_node_id="existing-456",
            existing_name="Alice Jones",
            score=0.6,
            action=ResolutionAction.link,
            method="level_2",
        )
        graph_store.create_node.return_value = "new-node-id"

        result = await pipeline.run([_make_fragment()])
        assert result.entities_linked == 1
        # Should create POSSIBLY_SAME_AS relationship
        graph_store.create_relationship.assert_called()

    async def test_entity_name_to_node_id_mapping_used_for_relations(
        self, pipeline, extraction_engine, entity_resolver, graph_store
    ):
        """Entity names resolved during entity resolution are used when creating relations."""
        alice = _make_entity(name="Alice", type="Agent")
        bob = _make_entity(name="Bob", type="Agent")
        relation = _make_relation(
            from_entity="Alice", to_entity="Bob", relation_type="CONCERNS"
        )
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[alice, bob],
            relations=[relation],
        )

        call_count = 0

        async def resolve_side_effect(entity, existing):
            nonlocal call_count
            call_count += 1
            return ResolutionCandidate(
                entity_name=entity.name,
                existing_node_id=None,
                existing_name=None,
                score=0.0,
                action=ResolutionAction.create_new,
                method="level_1",
            )

        entity_resolver.resolve.side_effect = resolve_side_effect
        node_ids = iter(["alice-id", "bob-id"])
        graph_store.create_node.side_effect = lambda node: next(node_ids)

        result = await pipeline.run([_make_fragment()])
        assert result.relations_created == 1

    async def test_unknown_entity_type_skipped_with_error(
        self, pipeline, extraction_engine, entity_resolver, graph_store
    ):
        entity = _make_entity(name="Something", type="UnknownType")
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[entity],
        )

        result = await pipeline.run([_make_fragment()])
        assert result.entities_created == 0
        assert any("UnknownType" in e for e in result.errors)
        entity_resolver.resolve.assert_not_called()

    async def test_claim_like_entity_type_is_skipped_without_error(
        self, pipeline, extraction_engine, entity_resolver, graph_store
    ):
        entity = _make_entity(name="1889", type="Fact")
        extraction_engine.extract.return_value = ExtractionResult(entities=[entity])

        result = await pipeline.run([_make_fragment()])
        assert result.entities_created == 0
        assert result.errors == []
        entity_resolver.resolve.assert_not_called()


# ===================================================================
# Claim integration
# ===================================================================


class TestClaimIntegration:
    async def test_creates_fact_node_for_claim(
        self, pipeline, extraction_engine, graph_store
    ):
        claim = _make_claim(content="Water is wet", claim_type="Fact")
        extraction_engine.extract.return_value = ExtractionResult(claims=[claim])

        result = await pipeline.run([_make_fragment()])
        assert result.claims_created == 1
        graph_store.create_node.assert_called()

    async def test_creates_event_node_for_temporal_claim(
        self, pipeline, extraction_engine, graph_store
    ):
        claim = _make_claim(
            content="Meeting happened",
            claim_type="Event",
            temporal_info=TemporalInfo(
                occurred_at="2024-06-01T10:00:00Z", precision="exact"
            ),
        )
        extraction_engine.extract.return_value = ExtractionResult(claims=[claim])

        result = await pipeline.run([_make_fragment()])
        assert result.claims_created == 1

    async def test_links_claims_to_entities_via_concerns(
        self, pipeline, extraction_engine, entity_resolver, graph_store
    ):
        entity = _make_entity(name="Alice", type="Agent")
        claim = _make_claim(
            content="Alice did something",
            involved_entities=["Alice"],
        )
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[entity],
            claims=[claim],
        )
        entity_resolver.resolve.return_value = ResolutionCandidate(
            entity_name="Alice",
            existing_node_id=None,
            existing_name=None,
            score=0.0,
            action=ResolutionAction.create_new,
            method="level_1",
        )
        graph_store.create_node.return_value = "alice-id"

        result = await pipeline.run([_make_fragment()])
        assert result.claims_created == 1
        # At least one CONCERNS relationship should be created
        assert graph_store.create_relationship.call_count >= 1

    async def test_existing_claim_still_links_concerns_on_retry(
        self, pipeline, extraction_engine, entity_resolver, graph_store
    ):
        entity = _make_entity(name="Alice", type="Agent")
        claim = _make_claim(
            content="Alice did something",
            involved_entities=["Alice"],
        )
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[entity],
            claims=[claim],
        )
        entity_resolver.resolve.return_value = ResolutionCandidate(
            entity_name="Alice",
            existing_node_id=None,
            existing_name=None,
            score=0.0,
            action=ResolutionAction.create_new,
            method="level_1",
        )

        claim_id = "claim-existing"

        async def _get_node(node_id: str):
            if node_id == claim_id:
                return Fact(id=claim_id, content="Alice did something")
            return None

        graph_store.get_node.side_effect = _get_node
        graph_store.get_relationships.return_value = []
        graph_store.create_node.return_value = "alice-id"

        with patch("engramcp.engine.consolidation._stable_claim_id", return_value=claim_id):
            result = await pipeline.run([_make_fragment()])

        assert result.claims_created == 0
        assert graph_store.create_node.call_count == 1
        assert any(
            call.args[:2] == (claim_id, "alice-id")
            for call in graph_store.create_relationship.call_args_list
        )

    async def test_unknown_claim_type_skipped_with_error(
        self, pipeline, extraction_engine, graph_store
    ):
        claim = _make_claim(content="Mystery", claim_type="UnknownClaim")
        extraction_engine.extract.return_value = ExtractionResult(claims=[claim])

        result = await pipeline.run([_make_fragment()])
        assert result.claims_created == 0
        assert any("UnknownClaim" in e for e in result.errors)


# ===================================================================
# Source traceability
# ===================================================================


class TestSourceTraceability:
    async def test_creates_source_nodes_from_fragments(
        self, pipeline, extraction_engine, graph_store
    ):
        fragment = _make_fragment()
        claim = _make_claim(source_fragment_ids=[fragment.id])
        extraction_engine.extract.return_value = ExtractionResult(claims=[claim])

        await pipeline.run([fragment])
        # Source node created (in addition to the claim node)
        assert graph_store.create_node.call_count >= 2

    async def test_creates_sourced_from_relations(
        self, pipeline, extraction_engine, graph_store
    ):
        fragment = _make_fragment()
        claim = _make_claim(source_fragment_ids=[fragment.id])
        extraction_engine.extract.return_value = ExtractionResult(claims=[claim])

        await pipeline.run([fragment])
        # Should have SOURCED_FROM relationship
        assert graph_store.create_relationship.call_count >= 1


# ===================================================================
# Relation integration
# ===================================================================


class TestRelationIntegration:
    async def test_creates_extracted_relations_between_entities(
        self, pipeline, extraction_engine, entity_resolver, graph_store
    ):
        alice = _make_entity(name="Alice", type="Agent")
        bob = _make_entity(name="Bob", type="Agent")
        relation = _make_relation(
            from_entity="Alice", to_entity="Bob", relation_type="SUPPORTS"
        )
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[alice, bob],
            relations=[relation],
        )

        async def resolve_side_effect(entity, existing):
            return ResolutionCandidate(
                entity_name=entity.name,
                existing_node_id=None,
                existing_name=None,
                score=0.0,
                action=ResolutionAction.create_new,
                method="level_1",
            )

        entity_resolver.resolve.side_effect = resolve_side_effect
        node_ids = iter(["alice-id", "bob-id"])
        graph_store.create_node.side_effect = lambda node: next(node_ids)

        result = await pipeline.run([_make_fragment()])
        assert result.relations_created == 1

    async def test_unknown_relation_type_skipped_with_error(
        self, pipeline, extraction_engine, entity_resolver, graph_store
    ):
        alice = _make_entity(name="Alice", type="Agent")
        bob = _make_entity(name="Bob", type="Agent")
        relation = _make_relation(
            from_entity="Alice", to_entity="Bob", relation_type="UNKNOWN_REL"
        )
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[alice, bob],
            relations=[relation],
        )

        async def resolve_side_effect(entity, existing):
            return ResolutionCandidate(
                entity_name=entity.name,
                existing_node_id=None,
                existing_name=None,
                score=0.0,
                action=ResolutionAction.create_new,
                method="level_1",
            )

        entity_resolver.resolve.side_effect = resolve_side_effect
        node_ids = iter(["alice-id", "bob-id"])
        graph_store.create_node.side_effect = lambda node: next(node_ids)

        result = await pipeline.run([_make_fragment()])
        assert result.relations_created == 0
        assert any("UNKNOWN_REL" in e for e in result.errors)

    async def test_skips_relation_when_entity_not_resolved(
        self, pipeline, extraction_engine, graph_store
    ):
        # No entities extracted, but a relation references them
        relation = _make_relation(
            from_entity="Unknown1", to_entity="Unknown2", relation_type="SUPPORTS"
        )
        extraction_engine.extract.return_value = ExtractionResult(
            relations=[relation],
        )

        result = await pipeline.run([_make_fragment()])
        assert result.relations_created == 0
        assert any("Unknown1" in e or "Unknown2" in e for e in result.errors)

    @pytest.mark.parametrize("relation_type", ["INSTANCE_OF", "GENERALIZES"])
    async def test_accepts_abstraction_relation_types_from_extraction(
        self, pipeline, extraction_engine, entity_resolver, graph_store, relation_type
    ):
        alice = _make_entity(name="Alice", type="Agent")
        bob = _make_entity(name="Bob", type="Agent")
        relation = _make_relation(
            from_entity="Alice", to_entity="Bob", relation_type=relation_type
        )
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[alice, bob],
            relations=[relation],
        )

        async def resolve_side_effect(entity, existing):
            return ResolutionCandidate(
                entity_name=entity.name,
                existing_node_id=None,
                existing_name=None,
                score=0.0,
                action=ResolutionAction.create_new,
                method="level_1",
            )

        entity_resolver.resolve.side_effect = resolve_side_effect
        node_ids = iter(["alice-id", "bob-id"])
        graph_store.create_node.side_effect = lambda node: next(node_ids)

        result = await pipeline.run([_make_fragment()])
        assert result.relations_created == 1
        assert not any(f"Unknown relation type '{relation_type}'" in e for e in result.errors)


# ===================================================================
# Contradictions
# ===================================================================


class TestContradictions:
    async def test_detects_contradiction_with_existing_claim(
        self, pipeline, extraction_engine, graph_store
    ):
        claim = _make_claim(content="Alice is in Paris", claim_type="Fact")
        extraction_engine.extract.return_value = ExtractionResult(claims=[claim])
        graph_store.find_claim_nodes.return_value = [Fact(content="Alice is not in Paris")]
        graph_store.create_node.return_value = "new-claim-id"

        await pipeline.run([_make_fragment()])

        contradiction_calls = [
            call
            for call in graph_store.create_relationship.call_args_list
            if call.args[2].rel_type == "CONTRADICTS"
        ]
        assert contradiction_calls


# ===================================================================
# Abstraction
# ===================================================================


class TestAbstraction:
    async def test_repeated_claims_create_pattern(self, pipeline, extraction_engine, graph_store):
        claims = [
            _make_claim(content="Alice filed a report"),
            _make_claim(content="Alice filed a report"),
            _make_claim(content="Alice filed a report"),
        ]
        extraction_engine.extract.return_value = ExtractionResult(claims=claims)
        graph_store.find_claim_nodes.return_value = []
        graph_store.create_node.side_effect = [
            "claim-1",
            "claim-2",
            "claim-3",
            "pattern-1",
        ]

        result = await pipeline.run([_make_fragment()])

        assert result.patterns_created == 1

    async def test_causal_relations_create_rule(self, pipeline, extraction_engine, graph_store):
        claims = [
            _make_claim(content="Pattern one repeated"),
            _make_claim(content="Pattern one repeated"),
            _make_claim(content="Pattern one repeated"),
            _make_claim(content="Pattern two repeated"),
            _make_claim(content="Pattern two repeated"),
            _make_claim(content="Pattern two repeated"),
        ]
        relation = _make_relation(relation_type="CAUSED_BY")
        extraction_engine.extract.return_value = ExtractionResult(
            claims=claims, relations=[relation]
        )
        graph_store.find_claim_nodes.return_value = []
        graph_store.create_node.side_effect = [
            "claim-1",
            "claim-2",
            "claim-3",
            "claim-4",
            "claim-5",
            "claim-6",
            "pattern-1",
            "pattern-2",
            "concept-1",
            "rule-1",
        ]

        result = await pipeline.run([_make_fragment()])

        assert result.patterns_created == 2
        assert result.concepts_created == 1
        assert result.rules_created == 1


# ===================================================================
# Audit
# ===================================================================


class TestAudit:
    async def test_logs_consolidation_run_to_audit(
        self, pipeline, extraction_engine, audit_logger
    ):
        await pipeline.run([_make_fragment()])
        logged_events = [
            call.args[0]
            for call in audit_logger.log.call_args_list
        ]
        event_types = {e.event_type for e in logged_events}
        assert AuditEventType.CONSOLIDATION_RUN in event_types

    async def test_logs_node_creation_to_audit(
        self, pipeline, extraction_engine, entity_resolver, graph_store, audit_logger
    ):
        entity = _make_entity(name="Alice", type="Agent")
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[entity],
        )
        entity_resolver.resolve.return_value = ResolutionCandidate(
            entity_name="Alice",
            existing_node_id=None,
            existing_name=None,
            score=0.0,
            action=ResolutionAction.create_new,
            method="level_1",
        )

        await pipeline.run([_make_fragment()])
        logged_events = [
            call.args[0]
            for call in audit_logger.log.call_args_list
        ]
        event_types = [e.event_type for e in logged_events]
        assert AuditEventType.NODE_CREATED in event_types

    async def test_logs_relation_creation_to_audit(
        self, pipeline, extraction_engine, entity_resolver, graph_store, audit_logger
    ):
        alice = _make_entity(name="Alice", type="Agent")
        bob = _make_entity(name="Bob", type="Agent")
        relation = _make_relation(
            from_entity="Alice", to_entity="Bob", relation_type="SUPPORTS"
        )
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[alice, bob],
            relations=[relation],
        )

        async def resolve_side_effect(entity, existing):
            return ResolutionCandidate(
                entity_name=entity.name,
                existing_node_id=None,
                existing_name=None,
                score=0.0,
                action=ResolutionAction.create_new,
                method="level_1",
            )

        entity_resolver.resolve.side_effect = resolve_side_effect
        node_ids = iter(["alice-id", "bob-id"])
        graph_store.create_node.side_effect = lambda node: next(node_ids)

        await pipeline.run([_make_fragment()])
        logged_events = [
            call.args[0]
            for call in audit_logger.log.call_args_list
        ]
        event_types = [e.event_type for e in logged_events]
        assert AuditEventType.RELATION_CREATED in event_types


# ===================================================================
# Result
# ===================================================================


class TestResult:
    async def test_run_result_counts_correct(
        self, pipeline, extraction_engine, entity_resolver, graph_store
    ):
        entity = _make_entity(name="Alice", type="Agent")
        claim = _make_claim(content="Water is wet")
        extraction_engine.extract.return_value = ExtractionResult(
            entities=[entity],
            claims=[claim],
        )
        entity_resolver.resolve.return_value = ResolutionCandidate(
            entity_name="Alice",
            existing_node_id=None,
            existing_name=None,
            score=0.0,
            action=ResolutionAction.create_new,
            method="level_1",
        )

        result = await pipeline.run([_make_fragment()])
        assert result.fragments_processed == 1
        assert result.entities_created == 1
        assert result.claims_created == 1
        assert result.run_id  # non-empty

    async def test_extraction_errors_propagated_to_result(
        self, pipeline, extraction_engine
    ):
        extraction_engine.extract.return_value = ExtractionResult(
            errors=["LLM call failed: timeout"],
        )

        result = await pipeline.run([_make_fragment()])
        assert "LLM call failed: timeout" in result.errors
