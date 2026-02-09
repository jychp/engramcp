"""Unit tests for in-process latency observability helpers."""

from __future__ import annotations

from engramcp.observability import latency_metrics_snapshot
from engramcp.observability import record_latency
from engramcp.observability import reset_latency_metrics


class TestObservabilityLatency:
    def setup_method(self):
        reset_latency_metrics()

    def teardown_method(self):
        reset_latency_metrics()

    def test_records_latency_aggregates(self):
        record_latency(operation="mcp.send_memory", duration_ms=10.0, ok=True)
        record_latency(operation="mcp.send_memory", duration_ms=30.0, ok=False)

        metrics = latency_metrics_snapshot()["mcp.send_memory"]
        assert metrics["count"] == 2
        assert metrics["error_count"] == 1
        assert metrics["total_ms"] == 40.0
        assert metrics["avg_ms"] == 20.0
        assert metrics["min_ms"] == 10.0
        assert metrics["max_ms"] == 30.0
        assert metrics["last_ms"] == 30.0

    def test_reset_clears_all_metrics(self):
        record_latency(operation="consolidation.run", duration_ms=12.0, ok=True)
        assert "consolidation.run" in latency_metrics_snapshot()
        reset_latency_metrics()
        assert latency_metrics_snapshot() == {}
