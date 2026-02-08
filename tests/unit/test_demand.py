"""Unit tests for retrieval-demand tracking (Layer 5 bootstrap)."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

from engramcp.engine.demand import QueryDemandTracker
from engramcp.engine.demand import QueryPattern


class TestQueryPattern:
    def test_from_parts_normalizes_case_sorts_and_deduplicates(self):
        pattern = QueryPattern.from_parts(
            node_types=["Person", "event", "person", "  Event  "],
            properties=["Name", "timestamp", "name", "  TIMESTAMP "],
        )

        assert pattern.node_types == ("event", "person")
        assert pattern.properties == ("name", "timestamp")

    def test_from_parts_drops_empty_values(self):
        pattern = QueryPattern.from_parts(
            node_types=["  ", "", "Person"],
            properties=["", "content"],
        )

        assert pattern.node_types == ("person",)
        assert pattern.properties == ("content",)


class TestQueryDemandTracker:
    def test_records_pattern_count(self):
        tracker = QueryDemandTracker(threshold=3)
        pattern = QueryPattern.from_parts(node_types=["Person"], properties=["name"])

        assert tracker.count(pattern) == 0
        tracker.record(pattern)
        tracker.record(pattern)
        assert tracker.count(pattern) == 2

    def test_emits_signal_when_threshold_reached(self):
        emitted_at = datetime(2026, 2, 8, 12, 0, tzinfo=UTC)
        tracker = QueryDemandTracker(threshold=2, clock=lambda: emitted_at)
        pattern = QueryPattern.from_parts(node_types=["Fact"], properties=["content"])

        assert tracker.record(pattern) is None
        signal = tracker.record(pattern)

        assert signal is not None
        assert signal.pattern == pattern
        assert signal.count == 2
        assert signal.threshold == 2
        assert signal.emitted_at == emitted_at

    def test_emits_only_once_per_pattern(self):
        tracker = QueryDemandTracker(threshold=2)
        pattern = QueryPattern.from_parts(node_types=["Fact"], properties=["content"])

        assert tracker.record(pattern) is None
        assert tracker.record(pattern) is not None
        assert tracker.record(pattern) is None
        assert tracker.record(pattern) is None

    def test_record_call_with_parts_tracks_shape(self):
        tracker = QueryDemandTracker(threshold=2)

        tracker.record_call(node_types=["Person"], properties=["name"])
        signal = tracker.record_call(node_types=["person"], properties=["Name"])

        assert signal is not None
        assert signal.pattern == QueryPattern(
            node_types=("person",), properties=("name",)
        )
        assert tracker.count(signal.pattern) == 2

    def test_invalid_threshold_raises_value_error(self):
        try:
            QueryDemandTracker(threshold=0)
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "threshold must be >= 1" in str(exc)
