"""Lightweight in-process observability helpers for latency metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class LatencySummary:
    """Aggregated latency metrics for one operation."""

    count: int = 0
    error_count: int = 0
    total_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0


class _LatencyRecorder:
    def __init__(self) -> None:
        self._lock = Lock()
        self._stats: dict[str, LatencySummary] = {}

    def record(self, *, operation: str, duration_ms: float, ok: bool) -> None:
        normalized = max(float(duration_ms), 0.0)
        with self._lock:
            summary = self._stats.setdefault(operation, LatencySummary())
            summary.count += 1
            if not ok:
                summary.error_count += 1
            summary.total_ms += normalized
            summary.last_ms = normalized
            if summary.count == 1:
                summary.min_ms = normalized
                summary.max_ms = normalized
            else:
                summary.min_ms = min(summary.min_ms, normalized)
                summary.max_ms = max(summary.max_ms, normalized)

        logger.info(
            "latency operation=%s duration_ms=%.3f ok=%s",
            operation,
            normalized,
            ok,
        )

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                operation: {
                    "count": summary.count,
                    "error_count": summary.error_count,
                    "total_ms": round(summary.total_ms, 3),
                    "avg_ms": round(
                        summary.total_ms / summary.count if summary.count else 0.0,
                        3,
                    ),
                    "min_ms": round(summary.min_ms, 3),
                    "max_ms": round(summary.max_ms, 3),
                    "last_ms": round(summary.last_ms, 3),
                }
                for operation, summary in sorted(self._stats.items())
            }

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()


_RECORDER = _LatencyRecorder()


def record_latency(*, operation: str, duration_ms: float, ok: bool = True) -> None:
    """Record one latency sample."""
    _RECORDER.record(operation=operation, duration_ms=duration_ms, ok=ok)


def latency_metrics_snapshot() -> dict[str, dict[str, float | int]]:
    """Return current in-process latency aggregates."""
    return _RECORDER.snapshot()


def reset_latency_metrics() -> None:
    """Clear all latency aggregates (test helper)."""
    _RECORDER.reset()
