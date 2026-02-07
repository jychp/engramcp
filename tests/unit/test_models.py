"""Unit tests for Layer 0 models (confidence, nodes, relations).

Pure Python tests — no Neo4j or Redis required.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone

import pytest

from engramcp.models import (
    Agent,
    AgentType,
    Artifact,
    ArtifactType,
    CausedBy,
    Concerns,
    Concept,
    Contradicts,
    Credibility,
    Decision,
    DerivedFrom,
    DerivedStatus,
    Event,
    Fact,
    FactStatus,
    LABEL_TO_MODEL,
    NATORating,
    Observation,
    Outcome,
    ParticipatedIn,
    Pattern,
    PossiblySameAs,
    Reliability,
    ResolutionStatus,
    Rule,
    Source,
    SourcedFrom,
    TemporalPrecision,
    credibility_from_int,
    degrade_credibility,
    worst_reliability,
)


# ===================================================================
# NATORating
# ===================================================================


class TestNATORating:
    def test_str_representation(self):
        rating = NATORating(reliability=Reliability.B, credibility=Credibility.TWO)
        assert str(rating) == "B2"

    def test_repr(self):
        rating = NATORating(reliability=Reliability.A, credibility=Credibility.ONE)
        assert repr(rating) == "NATORating('A1')"

    def test_from_str_valid(self):
        rating = NATORating.from_str("B2")
        assert rating.reliability == Reliability.B
        assert rating.credibility == Credibility.TWO

    def test_from_str_lowercase(self):
        rating = NATORating.from_str("c3")
        assert rating.reliability == Reliability.C
        assert rating.credibility == Credibility.THREE

    def test_from_str_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid NATO rating"):
            NATORating.from_str("")

    def test_from_str_invalid_letter(self):
        with pytest.raises(ValueError):
            NATORating.from_str("Z1")

    def test_from_str_invalid_number(self):
        with pytest.raises(ValueError):
            NATORating.from_str("A9")

    def test_is_better_or_equal_same(self):
        r = NATORating.from_str("B2")
        assert r.is_better_or_equal(r) is True

    def test_is_better_or_equal_better(self):
        better = NATORating.from_str("A1")
        worse = NATORating.from_str("C3")
        assert better.is_better_or_equal(worse) is True

    def test_is_better_or_equal_worse(self):
        better = NATORating.from_str("A1")
        worse = NATORating.from_str("C3")
        assert worse.is_better_or_equal(better) is False

    def test_is_better_or_equal_mixed(self):
        # Better letter but worse number — not better overall
        r1 = NATORating.from_str("A6")
        r2 = NATORating.from_str("C1")
        assert r1.is_better_or_equal(r2) is False
        assert r2.is_better_or_equal(r1) is False

    def test_frozen(self):
        rating = NATORating.from_str("B2")
        with pytest.raises(Exception):
            rating.reliability = Reliability.A


# ===================================================================
# Enums
# ===================================================================


class TestEnums:
    def test_reliability_values(self):
        assert [r.value for r in Reliability] == ["A", "B", "C", "D", "E", "F"]

    def test_credibility_values(self):
        assert [c.value for c in Credibility] == ["1", "2", "3", "4", "5", "6"]

    def test_temporal_precision_values(self):
        assert "exact" in [p.value for p in TemporalPrecision]
        assert "unknown" in [p.value for p in TemporalPrecision]

    def test_fact_status_values(self):
        assert set(s.value for s in FactStatus) == {"active", "contested", "retracted"}

    def test_derived_status_values(self):
        assert set(s.value for s in DerivedStatus) == {
            "active",
            "contested",
            "dissolved",
        }

    def test_resolution_status_values(self):
        assert "unresolved" in [s.value for s in ResolutionStatus]
        assert "acknowledged" in [s.value for s in ResolutionStatus]


# ===================================================================
# Node models
# ===================================================================


class TestNodes:
    def test_fact_defaults(self):
        fact = Fact(content="The sky is blue")
        assert fact.id  # auto-generated
        assert fact.content == "The sky is blue"
        assert fact.status == FactStatus.active
        assert fact.node_labels == ("Memory", "Fact")
        assert fact.created_at is not None
        assert fact.ingested_at is not None

    def test_event_labels_and_temporal(self):
        now = datetime.now(timezone.utc)
        event = Event(content="Something happened", occurred_at=now)
        assert event.node_labels == ("Memory", "Temporal", "Event")
        assert event.occurred_at == now
        assert event.temporal_precision == TemporalPrecision.exact
        assert event.occurred_until is None

    def test_observation_labels(self):
        obs = Observation(content="I noticed X")
        assert obs.node_labels == ("Memory", "Observation")
        assert obs.observed_at is not None

    def test_decision_labels(self):
        now = datetime.now(timezone.utc)
        dec = Decision(content="Chose option A", occurred_at=now)
        assert dec.node_labels == ("Memory", "Temporal", "Decision")

    def test_outcome_labels(self):
        now = datetime.now(timezone.utc)
        out = Outcome(content="Result was positive", occurred_at=now)
        assert out.node_labels == ("Memory", "Temporal", "Outcome")

    def test_agent(self):
        agent = Agent(name="Alice", type=AgentType.person)
        assert agent.node_labels == ("Memory", "Agent")
        assert agent.aliases == []
        agent_with_aliases = Agent(
            name="Bob",
            type=AgentType.person,
            aliases=["Robert", "Bobby"],
        )
        assert agent_with_aliases.aliases == ["Robert", "Bobby"]

    def test_artifact(self):
        art = Artifact(
            name="Report.pdf",
            type=ArtifactType.document,
            ref="https://example.com/report.pdf",
        )
        assert art.node_labels == ("Memory", "Artifact")
        assert art.ref == "https://example.com/report.pdf"

    def test_source(self):
        src = Source(type="court_document", reliability=Reliability.A)
        assert src.node_labels == ("Memory", "Source")
        assert src.reliability == Reliability.A
        assert src.ref is None

    def test_pattern_labels(self):
        pat = Pattern(
            content="Recurring meetings",
            derivation_run_id="run-1",
        )
        assert pat.node_labels == ("Memory", "Derived", "Pattern")
        assert pat.derivation_depth == 1

    def test_concept_labels(self):
        con = Concept(
            content="Power dynamics",
            derivation_run_id="run-2",
        )
        assert con.node_labels == ("Memory", "Derived", "Concept")
        assert con.derivation_depth == 2

    def test_rule_labels(self):
        rule = Rule(
            content="If X then Y",
            derivation_run_id="run-3",
        )
        assert rule.node_labels == ("Memory", "Derived", "Rule")
        assert rule.derivation_depth == 3

    def test_label_to_model_completeness(self):
        """Every node type is in the label-to-model mapping."""
        expected_types = {
            Fact,
            Event,
            Observation,
            Decision,
            Outcome,
            Agent,
            Artifact,
            Source,
            Pattern,
            Concept,
            Rule,
        }
        assert set(LABEL_TO_MODEL.values()) == expected_types

    def test_label_to_model_roundtrip(self):
        """Creating a node and looking up its labels returns the correct model."""
        fact = Fact(content="test")
        labels = frozenset(fact.node_labels)
        assert LABEL_TO_MODEL[labels] is Fact


# ===================================================================
# Relationship models
# ===================================================================


class TestRelations:
    def test_sourced_from(self):
        rel = SourcedFrom(credibility=Credibility.TWO)
        assert rel.rel_type == "SOURCED_FROM"
        assert rel.created_at is not None

    def test_derived_from(self):
        rel = DerivedFrom(
            derivation_run_id="run-1",
            derivation_method="frequency_detection",
        )
        assert rel.rel_type == "DERIVED_FROM"
        assert rel.weight == 1.0

    def test_caused_by(self):
        rel = CausedBy(mechanism="direct influence")
        assert rel.rel_type == "CAUSED_BY"

    def test_contradicts(self):
        rel = Contradicts(detection_run_id="run-42")
        assert rel.rel_type == "CONTRADICTS"
        assert rel.resolution_status == ResolutionStatus.unresolved
        assert rel.resolved_at is None

    def test_participated_in(self):
        rel = ParticipatedIn(role="witness")
        assert rel.rel_type == "PARTICIPATED_IN"

    def test_concerns(self):
        rel = Concerns(role="subject")
        assert rel.rel_type == "CONCERNS"

    def test_possibly_same_as(self):
        rel = PossiblySameAs(similarity_score=0.85)
        assert rel.rel_type == "POSSIBLY_SAME_AS"
        assert rel.detection_method == "name_similarity"


# ===================================================================
# Confidence helpers
# ===================================================================


class TestWorstReliability:
    def test_single_value(self):
        assert worst_reliability(Reliability.A) == Reliability.A

    def test_multiple_returns_worst(self):
        assert worst_reliability(Reliability.A, Reliability.C, Reliability.B) == Reliability.C

    def test_all_same(self):
        assert worst_reliability(Reliability.D, Reliability.D) == Reliability.D

    def test_f_is_worst(self):
        assert worst_reliability(Reliability.A, Reliability.F) == Reliability.F

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one argument"):
            worst_reliability()


class TestDegradeCredibility:
    def test_single_step(self):
        assert degrade_credibility(Credibility.TWO) == Credibility.THREE

    def test_multiple_steps(self):
        assert degrade_credibility(Credibility.ONE, steps=3) == Credibility.FOUR

    def test_clamped_at_six(self):
        assert degrade_credibility(Credibility.FIVE, steps=5) == Credibility.SIX

    def test_zero_steps_no_change(self):
        assert degrade_credibility(Credibility.THREE, steps=0) == Credibility.THREE

    def test_already_at_six(self):
        assert degrade_credibility(Credibility.SIX) == Credibility.SIX


class TestCredibilityFromInt:
    def test_all_valid_values(self):
        assert credibility_from_int(1) == Credibility.ONE
        assert credibility_from_int(2) == Credibility.TWO
        assert credibility_from_int(3) == Credibility.THREE
        assert credibility_from_int(4) == Credibility.FOUR
        assert credibility_from_int(5) == Credibility.FIVE
        assert credibility_from_int(6) == Credibility.SIX

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="1-6"):
            credibility_from_int(0)

    def test_seven_raises(self):
        with pytest.raises(ValueError, match="1-6"):
            credibility_from_int(7)
