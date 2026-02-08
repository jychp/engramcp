"""Concept candidate registry for Layer 5 concept-emergence bootstrap."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from engramcp.engine.demand import DemandSignal
from engramcp.engine.demand import QueryPattern


class CandidateStatus(StrEnum):
    """Lifecycle states for concept candidates."""

    detected = "detected"
    under_review = "under_review"
    materialized = "materialized"
    rejected = "rejected"


@dataclass(frozen=True)
class ConceptCandidate:
    """A candidate concept inferred from recurring retrieval demand."""

    id: str
    pattern: QueryPattern
    status: CandidateStatus
    demand_count: int
    threshold: int
    first_seen_at: datetime
    last_seen_at: datetime


class ConceptRegistry:
    """In-memory registry for concept candidates and lifecycle updates."""

    def __init__(self) -> None:
        self._by_id: dict[str, ConceptCandidate] = {}
        self._by_pattern: dict[QueryPattern, str] = {}

    def observe_signal(self, signal: DemandSignal) -> ConceptCandidate:
        """Create or refresh a candidate for an emitted demand signal."""
        existing = self.get_by_pattern(signal.pattern)
        if existing is None:
            candidate = ConceptCandidate(
                id=f"cc_{uuid.uuid4().hex[:16]}",
                pattern=signal.pattern,
                status=CandidateStatus.detected,
                demand_count=signal.count,
                threshold=signal.threshold,
                first_seen_at=signal.emitted_at,
                last_seen_at=signal.emitted_at,
            )
            self._by_id[candidate.id] = candidate
            self._by_pattern[candidate.pattern] = candidate.id
            return candidate

        candidate = ConceptCandidate(
            id=existing.id,
            pattern=existing.pattern,
            status=existing.status,
            demand_count=max(existing.demand_count, signal.count),
            threshold=existing.threshold,
            first_seen_at=existing.first_seen_at,
            last_seen_at=signal.emitted_at,
        )
        self._by_id[candidate.id] = candidate
        return candidate

    def set_status(
        self, candidate_id: str, status: CandidateStatus
    ) -> ConceptCandidate:
        """Update status for an existing candidate."""
        candidate = self._by_id[candidate_id]
        updated = ConceptCandidate(
            id=candidate.id,
            pattern=candidate.pattern,
            status=status,
            demand_count=candidate.demand_count,
            threshold=candidate.threshold,
            first_seen_at=candidate.first_seen_at,
            last_seen_at=candidate.last_seen_at,
        )
        self._by_id[candidate_id] = updated
        return updated

    def get_by_pattern(self, pattern: QueryPattern) -> ConceptCandidate | None:
        candidate_id = self._by_pattern.get(pattern)
        if candidate_id is None:
            return None
        return self._by_id[candidate_id]

    def candidate_count(self) -> int:
        return len(self._by_id)
