"""Unit tests for concept candidate registry (Layer 5 bootstrap)."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

from engramcp.engine.concepts import CandidateStatus
from engramcp.engine.concepts import ConceptRegistry
from engramcp.engine.demand import DemandSignal
from engramcp.engine.demand import QueryPattern


class TestConceptRegistry:
    def test_creates_candidate_from_demand_signal(self):
        emitted_at = datetime(2026, 2, 8, 12, 0, tzinfo=UTC)
        registry = ConceptRegistry()
        pattern = QueryPattern.from_parts(
            node_types=["Fact"], properties=["content", "participants"]
        )
        signal = DemandSignal(
            pattern=pattern, count=3, threshold=3, emitted_at=emitted_at
        )

        candidate = registry.observe_signal(signal)

        assert candidate.pattern == pattern
        assert candidate.status == CandidateStatus.detected
        assert candidate.demand_count == 3
        assert candidate.threshold == 3
        assert candidate.first_seen_at == emitted_at
        assert candidate.last_seen_at == emitted_at
        assert registry.candidate_count() == 1

    def test_reuses_existing_candidate_for_same_pattern(self):
        registry = ConceptRegistry()
        pattern = QueryPattern.from_parts(node_types=["Fact"], properties=["content"])

        c1 = registry.observe_signal(
            DemandSignal(
                pattern=pattern,
                count=3,
                threshold=3,
                emitted_at=datetime(2026, 2, 8, 12, 0, tzinfo=UTC),
            )
        )
        c2 = registry.observe_signal(
            DemandSignal(
                pattern=pattern,
                count=5,
                threshold=3,
                emitted_at=datetime(2026, 2, 8, 12, 10, tzinfo=UTC),
            )
        )

        assert c1.id == c2.id
        assert c2.demand_count == 5
        assert registry.candidate_count() == 1

    def test_can_progress_candidate_lifecycle(self):
        registry = ConceptRegistry()
        pattern = QueryPattern.from_parts(node_types=["Fact"], properties=["content"])
        candidate = registry.observe_signal(
            DemandSignal(
                pattern=pattern,
                count=3,
                threshold=3,
                emitted_at=datetime(2026, 2, 8, 12, 0, tzinfo=UTC),
            )
        )

        candidate = registry.set_status(candidate.id, CandidateStatus.under_review)
        assert candidate.status == CandidateStatus.under_review

        candidate = registry.set_status(candidate.id, CandidateStatus.materialized)
        assert candidate.status == CandidateStatus.materialized

    def test_get_by_pattern_returns_none_for_unknown(self):
        registry = ConceptRegistry()
        pattern = QueryPattern.from_parts(node_types=["Fact"], properties=["content"])
        assert registry.get_by_pattern(pattern) is None
