"""Query demand tracking for concept-emergence bootstrap (Layer 5)."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from datetime import UTC


@dataclass(frozen=True)
class QueryPattern:
    """Normalized structural shape extracted from retrieval demand."""

    node_types: tuple[str, ...] = ()
    properties: tuple[str, ...] = ()

    @classmethod
    def from_parts(
        cls,
        node_types: Sequence[str] | None = None,
        properties: Sequence[str] | None = None,
    ) -> QueryPattern:
        """Build a normalized immutable pattern from raw shape parts."""

        return cls(
            node_types=_normalize_tokens(node_types),
            properties=_normalize_tokens(properties),
        )


@dataclass(frozen=True)
class DemandSignal:
    """Signal emitted when a query pattern crosses recurrence threshold."""

    pattern: QueryPattern
    count: int
    threshold: int
    emitted_at: datetime


class QueryDemandTracker:
    """Tracks recurring query patterns and emits one signal per pattern."""

    def __init__(
        self,
        threshold: int = 3,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self._threshold = threshold
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        self._counts: dict[QueryPattern, int] = {}
        self._emitted: set[QueryPattern] = set()

    def record(self, pattern: QueryPattern) -> DemandSignal | None:
        """Record a retrieval pattern occurrence and emit signal on threshold."""

        count = self._counts.get(pattern, 0) + 1
        self._counts[pattern] = count

        if count >= self._threshold and pattern not in self._emitted:
            self._emitted.add(pattern)
            return DemandSignal(
                pattern=pattern,
                count=count,
                threshold=self._threshold,
                emitted_at=self._clock(),
            )
        return None

    def record_call(
        self,
        *,
        node_types: Sequence[str] | None = None,
        properties: Sequence[str] | None = None,
    ) -> DemandSignal | None:
        """Convenience helper to record a retrieval call shape directly."""

        pattern = QueryPattern.from_parts(node_types=node_types, properties=properties)
        return self.record(pattern)

    def count(self, pattern: QueryPattern) -> int:
        """Return current occurrence count for a pattern."""

        return self._counts.get(pattern, 0)


def _normalize_tokens(tokens: Sequence[str] | None) -> tuple[str, ...]:
    if not tokens:
        return ()

    normalized = {token.strip().lower() for token in tokens if token.strip()}
    return tuple(sorted(normalized))
