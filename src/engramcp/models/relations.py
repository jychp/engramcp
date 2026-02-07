"""Pydantic models for all Neo4j relationship types in the EngraMCP ontology.

Each model exposes a ``rel_type`` property returning the Neo4j relationship
type string (e.g. ``SOURCED_FROM``, ``CONTRADICTS``).
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from enum import Enum

from pydantic import BaseModel
from pydantic import Field

from engramcp.models.confidence import Credibility

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ResolutionStatus(str, Enum):
    """Resolution status for contradictions."""

    unresolved = "unresolved"
    resolved_in_favor_of_source = "resolved_in_favor_of_source"
    resolved_in_favor_of_target = "resolved_in_favor_of_target"
    acknowledged = "acknowledged"


class ExtractionMethod(str, Enum):
    """How a fact was extracted from a source."""

    llm_extraction = "llm_extraction"
    direct_input = "direct_input"
    manual = "manual"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class RelationshipBase(BaseModel):
    """Fields shared by every relationship."""

    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When this relationship was created.",
    )


# ---------------------------------------------------------------------------
# 3.1 Traceability relations
# ---------------------------------------------------------------------------


class SourcedFrom(RelationshipBase):
    """Links a knowledge node to its source (carries NATO credibility)."""

    credibility: Credibility = Field(
        description="NATO credibility number (1-6).",
    )
    extracted_at: datetime = Field(
        default_factory=_utcnow,
        description="When the extraction happened.",
    )
    extraction_method: ExtractionMethod = Field(
        default=ExtractionMethod.llm_extraction,
        description="Method used to extract the information.",
    )

    @property
    def rel_type(self) -> str:
        return "SOURCED_FROM"


class DerivedFrom(RelationshipBase):
    """Links a derived node to nodes it was derived from."""

    derivation_run_id: str = Field(
        description="ID of the consolidation run.",
    )
    derived_at: datetime = Field(
        default_factory=_utcnow,
        description="When the derivation happened.",
    )
    derivation_method: str = Field(
        description="Method used (frequency_detection, semantic_clustering, etc.).",
    )
    weight: float = Field(
        default=1.0,
        description="How much this source contributed to the derivation (0.0-1.0).",
    )

    @property
    def rel_type(self) -> str:
        return "DERIVED_FROM"


class Cites(RelationshipBase):
    """Links one Source to another Source (citation chain)."""

    citation: str | None = Field(
        default=None,
        description="Exact citation reference.",
    )

    @property
    def rel_type(self) -> str:
        return "CITES"


# ---------------------------------------------------------------------------
# 3.2 Causal relations
# ---------------------------------------------------------------------------


class CausedBy(RelationshipBase):
    """Asserts a causal link: source node was caused by target node."""

    credibility: Credibility | None = Field(
        default=None,
        description="NATO credibility number (1-6).",
    )
    mechanism: str | None = Field(
        default=None,
        description="Description of the causal mechanism.",
    )

    @property
    def rel_type(self) -> str:
        return "CAUSED_BY"


class LeadsTo(RelationshipBase):
    """Inverse direction of causality: source leads to target."""

    credibility: Credibility | None = Field(
        default=None,
        description="NATO credibility number (1-6).",
    )
    mechanism: str | None = Field(
        default=None,
        description="Description of the causal mechanism.",
    )

    @property
    def rel_type(self) -> str:
        return "LEADS_TO"


# ---------------------------------------------------------------------------
# 3.3 Temporal relations
# ---------------------------------------------------------------------------


class Preceded(RelationshipBase):
    """Strict temporal ordering: source occurred before target."""

    confidence: str | None = Field(
        default=None,
        description="How certain is the ordering.",
    )
    gap_seconds: float | None = Field(
        default=None,
        description="Duration between the two events in seconds.",
    )

    @property
    def rel_type(self) -> str:
        return "PRECEDED"


class Followed(RelationshipBase):
    """Semantic inverse of PRECEDED."""

    confidence: str | None = Field(
        default=None,
        description="How certain is the ordering.",
    )
    gap_seconds: float | None = Field(
        default=None,
        description="Duration between the two events in seconds.",
    )

    @property
    def rel_type(self) -> str:
        return "FOLLOWED"


# ---------------------------------------------------------------------------
# 3.4 Epistemic relations
# ---------------------------------------------------------------------------


class Supports(RelationshipBase):
    """One piece of knowledge corroborates another."""

    strength: float = Field(
        default=1.0,
        description="How strongly it supports (0.0-1.0).",
    )

    @property
    def rel_type(self) -> str:
        return "SUPPORTS"


class Contradicts(RelationshipBase):
    """One piece of knowledge conflicts with another."""

    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="When the contradiction was detected.",
    )
    detection_run_id: str = Field(
        description="ID of the run that detected this.",
    )
    resolution_status: ResolutionStatus = Field(
        default=ResolutionStatus.unresolved,
        description="Current resolution state.",
    )
    resolved_at: datetime | None = Field(
        default=None,
        description="When the contradiction was resolved.",
    )
    resolved_by: str | None = Field(
        default=None,
        description="Who resolved it (agent_id or 'system').",
    )

    @property
    def rel_type(self) -> str:
        return "CONTRADICTS"


# ---------------------------------------------------------------------------
# 3.5 Participation relations
# ---------------------------------------------------------------------------


class ParticipatedIn(RelationshipBase):
    """An agent was involved in an event."""

    role: str | None = Field(
        default=None,
        description="Role played (organizer, attendee, witness, etc.).",
    )
    credibility: Credibility | None = Field(
        default=None,
        description="NATO credibility number (1-6).",
    )

    @property
    def rel_type(self) -> str:
        return "PARTICIPATED_IN"


class DecidedBy(RelationshipBase):
    """A decision was made by a specific agent."""

    credibility: Credibility | None = Field(
        default=None,
        description="NATO credibility number (1-6).",
    )

    @property
    def rel_type(self) -> str:
        return "DECIDED_BY"


class ObservedBy(RelationshipBase):
    """An observation was made by a specific agent."""

    credibility: Credibility | None = Field(
        default=None,
        description="NATO credibility number (1-6).",
    )

    @property
    def rel_type(self) -> str:
        return "OBSERVED_BY"


# ---------------------------------------------------------------------------
# 3.6 Reference relations
# ---------------------------------------------------------------------------


class Mentions(RelationshipBase):
    """A knowledge node or artifact references an entity."""

    context: str | None = Field(
        default=None,
        description="Brief context of the mention.",
    )

    @property
    def rel_type(self) -> str:
        return "MENTIONS"


class Concerns(RelationshipBase):
    """A fact, event, or observation is about a specific entity."""

    role: str | None = Field(
        default=None,
        description="Role of the entity (subject, object, location, etc.).",
    )

    @property
    def rel_type(self) -> str:
        return "CONCERNS"


# ---------------------------------------------------------------------------
# 3.7 Abstraction relations
# ---------------------------------------------------------------------------


class Generalizes(RelationshipBase):
    """A derived node abstracts from a lower-level node."""

    @property
    def rel_type(self) -> str:
        return "GENERALIZES"


class InstanceOf(RelationshipBase):
    """A lower-level node is an instance of a higher abstraction."""

    @property
    def rel_type(self) -> str:
        return "INSTANCE_OF"


# ---------------------------------------------------------------------------
# 3.8 Entity resolution
# ---------------------------------------------------------------------------


class PossiblySameAs(RelationshipBase):
    """Candidate entity resolution link (unresolved)."""

    similarity_score: float = Field(
        description="Score indicating how likely the two nodes are the same (0.0-1.0).",
    )
    detection_method: str = Field(
        default="name_similarity",
        description="Method used to detect potential match.",
    )

    @property
    def rel_type(self) -> str:
        return "POSSIBLY_SAME_AS"


# ---------------------------------------------------------------------------
# Union type for generic dispatch
# ---------------------------------------------------------------------------

Relationship = (
    SourcedFrom
    | DerivedFrom
    | Cites
    | CausedBy
    | LeadsTo
    | Preceded
    | Followed
    | Supports
    | Contradicts
    | ParticipatedIn
    | DecidedBy
    | ObservedBy
    | Mentions
    | Concerns
    | Generalizes
    | InstanceOf
    | PossiblySameAs
)
