"""Tests for the confidence engine (Layer 3 — engine/confidence.py).

2 pure logic tests + 22 Neo4j tests = 24 total.
"""

from __future__ import annotations

import pytest

from engramcp.engine.confidence import (
    CascadeResult,
    ConfidenceConfig,
    ConfidenceEngine,
    CredibilityAssessment,
    PropagatedRating,
)
from engramcp.models import (
    Cites,
    Contradicts,
    Credibility,
    DerivedFrom,
    DerivedStatus,
    Fact,
    FactStatus,
    Pattern,
    Concept,
    Reliability,
    Rule,
    Source,
    SourcedFrom,
    Supports,
)


# ===================================================================
# TestDimension1Reliability
# ===================================================================


class TestDimension1Reliability:
    def test_reliability_from_confidence_hint(self, confidence_engine):
        """Parse 'B' from a confidence hint string."""
        assert confidence_engine.reliability_from_hint("B2") == Reliability.B
        assert confidence_engine.reliability_from_hint("A") == Reliability.A

    def test_default_reliability_when_no_hint(self, confidence_engine):
        """No hint returns the config default (F)."""
        assert confidence_engine.reliability_from_hint(None) == Reliability.F
        assert confidence_engine.reliability_from_hint("") == Reliability.F

    async def test_reliability_override_via_correct_memory(
        self, graph_store, confidence_engine
    ):
        """Source reliability is read from the Source node itself."""
        src = Source(type="testimony", reliability=Reliability.A)
        await graph_store.create_node(src)

        fact = Fact(content="Claim with A-rated source")
        await graph_store.create_node(fact)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.TWO)
        )

        # The source's reliability is A (not the default F)
        retrieved = await graph_store.get_node(src.id)
        assert retrieved.reliability == Reliability.A


# ===================================================================
# TestDimension2Credibility
# ===================================================================


class TestDimension2Credibility:
    async def test_initial_credibility_uncorroborated(
        self, graph_store, confidence_engine
    ):
        """A single-source fact gets initial credibility (3)."""
        fact = Fact(content="Uncorroborated claim")
        src = Source(type="testimony", reliability=Reliability.B)
        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.THREE)
        )

        assessment = await confidence_engine.assess_credibility(fact.id)
        assert assessment.credibility == Credibility.THREE

    async def test_credibility_upgrade_on_corroboration(
        self, graph_store, confidence_engine
    ):
        """Two independent sources upgrade credibility toward 1."""
        fact = Fact(content="Corroborated claim")
        src_a = Source(type="testimony_a", reliability=Reliability.B)
        src_b = Source(type="testimony_b", reliability=Reliability.B)
        await graph_store.create_node(fact)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_relationship(
            fact.id, src_a.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            fact.id, src_b.id, SourcedFrom(credibility=Credibility.THREE)
        )

        assessment = await confidence_engine.assess_credibility(fact.id)
        # Two independent sources → upgrade from 3 to 2
        assert assessment.credibility == Credibility.TWO
        assert assessment.supporting_count >= 2

    async def test_credibility_downgrade_on_contradiction(
        self, graph_store, confidence_engine
    ):
        """A contradicted fact is downgraded toward 4."""
        fact = Fact(content="Contradicted claim")
        src = Source(type="testimony", reliability=Reliability.B)
        contra_fact = Fact(content="Opposing claim")
        contra_src = Source(type="other_testimony", reliability=Reliability.B)

        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_node(contra_fact)
        await graph_store.create_node(contra_src)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            contra_fact.id, contra_src.id,
            SourcedFrom(credibility=Credibility.THREE),
        )
        await graph_store.create_relationship(
            fact.id, contra_fact.id,
            Contradicts(detection_run_id="run-1"),
        )

        assessment = await confidence_engine.assess_credibility(fact.id)
        assert assessment.credibility == Credibility.FOUR
        assert assessment.contradicting_count >= 1

    async def test_credibility_improbable_contradicted_stronger(
        self, graph_store, confidence_engine
    ):
        """Contradicted by a stronger source → credibility 5."""
        fact = Fact(content="Weak claim")
        weak_src = Source(type="rumor", reliability=Reliability.E)
        contra_fact = Fact(content="Strong opposing claim")
        strong_src = Source(type="court_document", reliability=Reliability.A)

        await graph_store.create_node(fact)
        await graph_store.create_node(weak_src)
        await graph_store.create_node(contra_fact)
        await graph_store.create_node(strong_src)
        await graph_store.create_relationship(
            fact.id, weak_src.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            contra_fact.id, strong_src.id,
            SourcedFrom(credibility=Credibility.ONE),
        )
        await graph_store.create_relationship(
            fact.id, contra_fact.id,
            Contradicts(detection_run_id="run-2"),
        )

        assessment = await confidence_engine.assess_credibility(fact.id)
        assert assessment.credibility == Credibility.FIVE

    async def test_credibility_unknown_insufficient_data(
        self, graph_store, confidence_engine
    ):
        """A fact with no sources gets credibility 6 (cannot be judged)."""
        fact = Fact(content="Sourceless claim")
        await graph_store.create_node(fact)

        assessment = await confidence_engine.assess_credibility(fact.id)
        assert assessment.credibility == Credibility.SIX


# ===================================================================
# TestCorroboration
# ===================================================================


class TestCorroboration:
    async def test_independent_sources_boost_credibility(
        self, graph_store, confidence_engine
    ):
        """Independent sources are counted for corroboration."""
        fact = Fact(content="Well-supported claim")
        src_a = Source(type="testimony_a", reliability=Reliability.B)
        src_b = Source(type="testimony_b", reliability=Reliability.B)
        src_c = Source(type="testimony_c", reliability=Reliability.B)
        await graph_store.create_node(fact)
        for src in (src_a, src_b, src_c):
            await graph_store.create_node(src)
            await graph_store.create_relationship(
                fact.id, src.id, SourcedFrom(credibility=Credibility.THREE)
            )

        count, source_ids = await confidence_engine.check_corroboration(fact.id)
        assert count == 3
        assert set(source_ids) == {src_a.id, src_b.id, src_c.id}

    async def test_non_independent_sources_no_boost(
        self, graph_store, confidence_engine
    ):
        """Sources sharing a common ancestor count as one."""
        fact = Fact(content="Echoed claim")
        root = Source(type="original", reliability=Reliability.A)
        src_a = Source(type="copy_a", reliability=Reliability.C)
        src_b = Source(type="copy_b", reliability=Reliability.C)
        await graph_store.create_node(fact)
        await graph_store.create_node(root)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_relationship(src_a.id, root.id, Cites())
        await graph_store.create_relationship(src_b.id, root.id, Cites())
        await graph_store.create_relationship(
            fact.id, src_a.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            fact.id, src_b.id, SourcedFrom(credibility=Credibility.THREE)
        )

        count, source_ids = await confidence_engine.check_corroboration(fact.id)
        # src_a and src_b are not independent — only 1 independent group
        assert count == 1

    async def test_corroboration_requires_minimum_2_independent(
        self, graph_store, confidence_engine
    ):
        """A single source (even strong) doesn't count as corroborated."""
        fact = Fact(content="Single-source claim")
        src = Source(type="court_document", reliability=Reliability.A)
        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.ONE)
        )

        count, _ = await confidence_engine.check_corroboration(fact.id)
        assert count == 1  # Not enough for corroboration boost


# ===================================================================
# TestPropagationUpward
# ===================================================================


class TestPropagationUpward:
    async def test_pattern_letter_is_weakest_source(
        self, graph_store, confidence_engine
    ):
        """Pattern reliability = worst reliability among contributing facts' sources."""
        fact_a = Fact(content="Fact A")
        fact_b = Fact(content="Fact B")
        src_a = Source(type="testimony", reliability=Reliability.A)
        src_b = Source(type="rumor", reliability=Reliability.D)
        pattern = Pattern(content="Recurring pattern", derivation_run_id="run-1")

        await graph_store.create_node(fact_a)
        await graph_store.create_node(fact_b)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_node(pattern)
        await graph_store.create_relationship(
            fact_a.id, src_a.id, SourcedFrom(credibility=Credibility.TWO)
        )
        await graph_store.create_relationship(
            fact_b.id, src_b.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            pattern.id, fact_a.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )
        await graph_store.create_relationship(
            pattern.id, fact_b.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )

        rating = await confidence_engine.propagate_upward(pattern.id)
        assert rating.reliability == Reliability.D  # Worst of A, D

    async def test_pattern_number_computed_from_convergence(
        self, graph_store, confidence_engine
    ):
        """Pattern credibility is derived from the convergence of its sources."""
        fact_a = Fact(content="Fact A")
        fact_b = Fact(content="Fact B")
        src_a = Source(type="testimony_a", reliability=Reliability.B)
        src_b = Source(type="testimony_b", reliability=Reliability.B)
        pattern = Pattern(content="Strong pattern", derivation_run_id="run-1")

        await graph_store.create_node(fact_a)
        await graph_store.create_node(fact_b)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_node(pattern)
        await graph_store.create_relationship(
            fact_a.id, src_a.id, SourcedFrom(credibility=Credibility.TWO)
        )
        await graph_store.create_relationship(
            fact_b.id, src_b.id, SourcedFrom(credibility=Credibility.TWO)
        )
        await graph_store.create_relationship(
            pattern.id, fact_a.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )
        await graph_store.create_relationship(
            pattern.id, fact_b.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )

        rating = await confidence_engine.propagate_upward(pattern.id)
        # Two independent facts with cred 2, then degraded by depth_decay (1 step)
        # Best source credibility = 2, degraded by 1 = 3
        assert rating.credibility == Credibility.THREE

    async def test_concept_inherits_degraded_from_patterns(
        self, graph_store, confidence_engine
    ):
        """Concept credibility is further degraded from its source patterns."""
        fact = Fact(content="Base fact")
        src = Source(type="testimony", reliability=Reliability.B)
        pattern = Pattern(content="Base pattern", derivation_run_id="run-1")
        concept = Concept(content="Abstract concept", derivation_run_id="run-2")

        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_node(pattern)
        await graph_store.create_node(concept)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.TWO)
        )
        await graph_store.create_relationship(
            pattern.id, fact.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )
        await graph_store.create_relationship(
            concept.id, pattern.id,
            DerivedFrom(derivation_run_id="run-2", derivation_method="clustering"),
        )

        rating = await confidence_engine.propagate_upward(concept.id)
        # Source cred 2, pattern at depth 1 = 2+1 = 3, concept at depth 2 = 3+1 = 4
        assert rating.credibility == Credibility.FOUR

    async def test_rule_inherits_degraded_from_concepts(
        self, graph_store, confidence_engine
    ):
        """Rule credibility is degraded through the full derivation chain."""
        fact = Fact(content="Base fact")
        src = Source(type="testimony", reliability=Reliability.B)
        pattern = Pattern(content="Base pattern", derivation_run_id="run-1")
        concept = Concept(content="Base concept", derivation_run_id="run-2")
        rule = Rule(content="Causal rule", derivation_run_id="run-3")

        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_node(pattern)
        await graph_store.create_node(concept)
        await graph_store.create_node(rule)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.TWO)
        )
        await graph_store.create_relationship(
            pattern.id, fact.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )
        await graph_store.create_relationship(
            concept.id, pattern.id,
            DerivedFrom(derivation_run_id="run-2", derivation_method="clustering"),
        )
        await graph_store.create_relationship(
            rule.id, concept.id,
            DerivedFrom(derivation_run_id="run-3", derivation_method="causal"),
        )

        rating = await confidence_engine.propagate_upward(rule.id)
        # Source cred 2, +1 per layer (3 layers) = 5
        assert rating.credibility == Credibility.FIVE

    async def test_depth_decay_applied_per_layer(
        self, graph_store, confidence_engine
    ):
        """Depth decay is applied once per abstraction layer."""
        fact = Fact(content="Base fact")
        src = Source(type="court_document", reliability=Reliability.A)
        pattern = Pattern(content="Pattern", derivation_run_id="run-1")

        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_node(pattern)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.ONE)
        )
        await graph_store.create_relationship(
            pattern.id, fact.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )

        rating = await confidence_engine.propagate_upward(pattern.id)
        # Source cred 1, degraded by 1 step for pattern layer = 2
        assert rating.credibility == Credibility.TWO
        assert rating.reliability == Reliability.A


# ===================================================================
# TestPropagationDownward
# ===================================================================


class TestPropagationDownward:
    async def test_contest_fact_cascades_to_pattern(
        self, graph_store, confidence_engine
    ):
        """Contesting a fact cascades to patterns derived from it."""
        fact = Fact(content="Contested fact", status=FactStatus.contested)
        src = Source(type="testimony", reliability=Reliability.B)
        pattern = Pattern(content="Dependent pattern", derivation_run_id="run-1")

        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_node(pattern)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            pattern.id, fact.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )

        result = await confidence_engine.cascade_contest(fact.id)
        assert pattern.id in result.affected_nodes

    async def test_contest_fact_dissolves_weak_pattern(
        self, graph_store, confidence_engine
    ):
        """A pattern with all sources contested is dissolved."""
        fact = Fact(content="Only source", status=FactStatus.contested)
        src = Source(type="testimony", reliability=Reliability.B)
        pattern = Pattern(content="Weak pattern", derivation_run_id="run-1")

        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_node(pattern)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            pattern.id, fact.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )

        await confidence_engine.cascade_contest(fact.id)

        updated = await graph_store.get_node(pattern.id)
        assert updated.status == DerivedStatus.dissolved

    async def test_cascade_propagates_to_concept(
        self, graph_store, confidence_engine
    ):
        """Contest cascades from fact → pattern → concept."""
        fact = Fact(content="Contested fact", status=FactStatus.contested)
        src = Source(type="testimony", reliability=Reliability.B)
        pattern = Pattern(content="Pattern", derivation_run_id="run-1")
        concept = Concept(content="Concept", derivation_run_id="run-2")

        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_node(pattern)
        await graph_store.create_node(concept)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            pattern.id, fact.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )
        await graph_store.create_relationship(
            concept.id, pattern.id,
            DerivedFrom(derivation_run_id="run-2", derivation_method="clustering"),
        )

        result = await confidence_engine.cascade_contest(fact.id)
        assert concept.id in result.affected_nodes

    async def test_cascade_propagates_to_rule(
        self, graph_store, confidence_engine
    ):
        """Contest cascades through the full hierarchy: fact → pattern → concept → rule."""
        fact = Fact(content="Contested fact", status=FactStatus.contested)
        src = Source(type="testimony", reliability=Reliability.B)
        pattern = Pattern(content="Pattern", derivation_run_id="run-1")
        concept = Concept(content="Concept", derivation_run_id="run-2")
        rule = Rule(content="Rule", derivation_run_id="run-3")

        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_node(pattern)
        await graph_store.create_node(concept)
        await graph_store.create_node(rule)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            pattern.id, fact.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )
        await graph_store.create_relationship(
            concept.id, pattern.id,
            DerivedFrom(derivation_run_id="run-2", derivation_method="clustering"),
        )
        await graph_store.create_relationship(
            rule.id, concept.id,
            DerivedFrom(derivation_run_id="run-3", derivation_method="causal"),
        )

        result = await confidence_engine.cascade_contest(fact.id)
        assert rule.id in result.affected_nodes

    async def test_cascade_recalculates_not_deletes(
        self, graph_store, confidence_engine
    ):
        """Cascade changes status but never deletes nodes."""
        fact = Fact(content="Contested fact", status=FactStatus.contested)
        src = Source(type="testimony", reliability=Reliability.B)
        pattern = Pattern(content="Preserved pattern", derivation_run_id="run-1")

        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_node(pattern)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            pattern.id, fact.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )

        await confidence_engine.cascade_contest(fact.id)

        # Node still exists (not deleted)
        node = await graph_store.get_node(pattern.id)
        assert node is not None
        # Status changed to dissolved
        assert node.status == DerivedStatus.dissolved


# ===================================================================
# TestEdgeCases
# ===================================================================


class TestEdgeCases:
    async def test_single_source_fact_confidence(
        self, graph_store, confidence_engine
    ):
        """A fact with a single strong source gets a reasonable rating."""
        fact = Fact(content="Single-source claim")
        src = Source(type="court_document", reliability=Reliability.A)
        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.ONE)
        )

        assessment = await confidence_engine.assess_credibility(fact.id)
        # Single source, no corroboration, not contradicted → initial (THREE)
        assert assessment.credibility == Credibility.THREE

    async def test_all_sources_contested(
        self, graph_store, confidence_engine
    ):
        """When all contributing facts are contested, pattern dissolves."""
        fact_a = Fact(content="Fact A", status=FactStatus.contested)
        fact_b = Fact(content="Fact B", status=FactStatus.contested)
        src_a = Source(type="testimony_a", reliability=Reliability.B)
        src_b = Source(type="testimony_b", reliability=Reliability.B)
        pattern = Pattern(content="Doomed pattern", derivation_run_id="run-1")

        await graph_store.create_node(fact_a)
        await graph_store.create_node(fact_b)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_node(pattern)
        await graph_store.create_relationship(
            fact_a.id, src_a.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            fact_b.id, src_b.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            pattern.id, fact_a.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )
        await graph_store.create_relationship(
            pattern.id, fact_b.id,
            DerivedFrom(derivation_run_id="run-1", derivation_method="frequency"),
        )

        # Cascade from both contested facts
        await confidence_engine.cascade_contest(fact_a.id)
        await confidence_engine.cascade_contest(fact_b.id)

        updated = await graph_store.get_node(pattern.id)
        assert updated.status == DerivedStatus.dissolved

    async def test_circular_support_does_not_infinite_loop(
        self, graph_store, confidence_engine
    ):
        """Circular SUPPORTS relationships don't cause infinite recursion."""
        fact_a = Fact(content="Fact A")
        fact_b = Fact(content="Fact B")
        src_a = Source(type="testimony_a", reliability=Reliability.B)
        src_b = Source(type="testimony_b", reliability=Reliability.B)

        await graph_store.create_node(fact_a)
        await graph_store.create_node(fact_b)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_relationship(
            fact_a.id, src_a.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(
            fact_b.id, src_b.id, SourcedFrom(credibility=Credibility.THREE)
        )
        # Circular support
        await graph_store.create_relationship(
            fact_a.id, fact_b.id, Supports()
        )
        await graph_store.create_relationship(
            fact_b.id, fact_a.id, Supports()
        )

        # Should complete without hanging
        assessment = await confidence_engine.assess_credibility(fact_a.id)
        assert assessment.credibility is not None
