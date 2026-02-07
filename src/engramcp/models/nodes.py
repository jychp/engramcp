"""Pydantic models for all Neo4j node types in the EngraMCP ontology.

Every node carries the ``Memory`` base label. Concrete types add
additional labels (e.g. ``Fact``, ``Event``, ``Agent``).  Temporal nodes
also carry the ``Temporal`` label.  Derived nodes carry ``Derived``.

Each model exposes a ``node_labels`` property returning the ordered tuple
of Neo4j labels to apply when creating the node.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from datetime import timezone
from enum import Enum

from pydantic import BaseModel
from pydantic import Field

from engramcp.models.confidence import Reliability

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TemporalPrecision(str, Enum):
    """Granularity of a temporal timestamp."""

    exact = "exact"
    day = "day"
    month = "month"
    year = "year"
    approximate = "approximate"
    unknown = "unknown"


class FactStatus(str, Enum):
    """Lifecycle status of a knowledge node."""

    active = "active"
    contested = "contested"
    retracted = "retracted"


class DerivedStatus(str, Enum):
    """Lifecycle status of a derived node."""

    active = "active"
    contested = "contested"
    dissolved = "dissolved"


class AgentType(str, Enum):
    """Kind of agent."""

    person = "person"
    organization = "organization"
    system = "system"
    ai_agent = "ai_agent"


class ArtifactType(str, Enum):
    """Kind of artifact."""

    document = "document"
    file = "file"
    recording = "recording"
    physical_object = "physical_object"


# ---------------------------------------------------------------------------
# Base models
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class MemoryBase(BaseModel):
    """Fields shared by every node in the graph (the ``Memory`` label)."""

    id: str = Field(
        default_factory=_new_id,
        description="Unique node identifier (UUID hex).",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the node was created.",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of the last update.",
    )


class TemporalMixin(BaseModel):
    """Fields for nodes that carry the ``Temporal`` label."""

    occurred_at: datetime = Field(
        description="When the event/decision/outcome occurred.",
    )
    occurred_until: datetime | None = Field(
        default=None,
        description="End of the time range (if a period, not a point).",
    )
    temporal_precision: TemporalPrecision = Field(
        default=TemporalPrecision.exact,
        description="Granularity of the timestamp.",
    )


# ---------------------------------------------------------------------------
# Core knowledge nodes (ingested)
# ---------------------------------------------------------------------------


class Fact(MemoryBase):
    """An atomic assertion about the world."""

    content: str = Field(description="Textual content of the fact.")
    ingested_at: datetime = Field(
        default_factory=_utcnow,
        description="When this fact was ingested into the system.",
    )
    status: FactStatus = Field(
        default=FactStatus.active,
        description="Lifecycle status.",
    )

    @property
    def node_labels(self) -> tuple[str, ...]:
        return ("Memory", "Fact")


class Event(MemoryBase, TemporalMixin):
    """Something that happened at a specific time or period."""

    content: str = Field(description="Textual description of the event.")
    ingested_at: datetime = Field(
        default_factory=_utcnow,
        description="When this event was ingested into the system.",
    )
    status: FactStatus = Field(
        default=FactStatus.active,
        description="Lifecycle status.",
    )

    @property
    def node_labels(self) -> tuple[str, ...]:
        return ("Memory", "Temporal", "Event")


class Observation(MemoryBase):
    """A subjective perception by an agent."""

    content: str = Field(description="Textual content of the observation.")
    observed_at: datetime = Field(
        default_factory=_utcnow,
        description="When the observation was made.",
    )
    ingested_at: datetime = Field(
        default_factory=_utcnow,
        description="When this observation was ingested.",
    )
    status: FactStatus = Field(
        default=FactStatus.active,
        description="Lifecycle status.",
    )

    @property
    def node_labels(self) -> tuple[str, ...]:
        return ("Memory", "Observation")


class Decision(MemoryBase, TemporalMixin):
    """A choice made by an agent at a point in time."""

    content: str = Field(description="Description of the decision.")
    ingested_at: datetime = Field(
        default_factory=_utcnow,
        description="When this decision was ingested.",
    )
    status: FactStatus = Field(
        default=FactStatus.active,
        description="Lifecycle status.",
    )

    @property
    def node_labels(self) -> tuple[str, ...]:
        return ("Memory", "Temporal", "Decision")


class Outcome(MemoryBase, TemporalMixin):
    """An observed result of a decision or event."""

    content: str = Field(description="Description of the outcome.")
    ingested_at: datetime = Field(
        default_factory=_utcnow,
        description="When this outcome was ingested.",
    )
    status: FactStatus = Field(
        default=FactStatus.active,
        description="Lifecycle status.",
    )

    @property
    def node_labels(self) -> tuple[str, ...]:
        return ("Memory", "Temporal", "Outcome")


# ---------------------------------------------------------------------------
# Entity nodes
# ---------------------------------------------------------------------------


class Agent(MemoryBase):
    """Anything that can act: a person, organization, system, or AI agent."""

    name: str = Field(description="Primary display name of the agent.")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names or identifiers.",
    )
    type: AgentType = Field(description="Kind of agent.")

    @property
    def node_labels(self) -> tuple[str, ...]:
        return ("Memory", "Agent")


class Artifact(MemoryBase):
    """A document, file, piece of evidence, or tangible object."""

    name: str = Field(description="Name or title of the artifact.")
    type: ArtifactType = Field(description="Kind of artifact.")
    ref: str | None = Field(
        default=None,
        description="URL or external reference.",
    )

    @property
    def node_labels(self) -> tuple[str, ...]:
        return ("Memory", "Artifact")


# ---------------------------------------------------------------------------
# Source node
# ---------------------------------------------------------------------------


class Source(MemoryBase):
    """Traceability anchor — every piece of knowledge traces back here."""

    type: str = Field(
        description="Kind of source (court_document, testimony, news_article, etc.).",
    )
    ref: str | None = Field(
        default=None,
        description="URL, document ID, or file path.",
    )
    citation: str | None = Field(
        default=None,
        description="Exact location: page, line, paragraph, timestamp.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Which calling agent submitted this.",
    )
    reliability: Reliability = Field(
        description="NATO source reliability letter (A-F).",
    )
    ingested_at: datetime = Field(
        default_factory=_utcnow,
        description="When this source was ingested.",
    )

    @property
    def node_labels(self) -> tuple[str, ...]:
        return ("Memory", "Source")


# ---------------------------------------------------------------------------
# Derived nodes (emergent, via consolidation)
# ---------------------------------------------------------------------------


class Pattern(MemoryBase):
    """A recurrence detected across multiple base knowledge nodes."""

    content: str = Field(description="Human-readable description of the pattern.")
    derivation_depth: int = Field(
        default=1,
        description="Always 1 for Pattern.",
    )
    derivation_run_id: str = Field(
        description="ID of the consolidation run that produced this.",
    )
    derived_at: datetime = Field(
        default_factory=_utcnow,
        description="When this pattern was derived.",
    )
    status: DerivedStatus = Field(
        default=DerivedStatus.active,
        description="Lifecycle status.",
    )

    @property
    def node_labels(self) -> tuple[str, ...]:
        return ("Memory", "Derived", "Pattern")


class Concept(MemoryBase):
    """An abstraction generalizing multiple patterns."""

    content: str = Field(description="Human-readable description of the concept.")
    derivation_depth: int = Field(
        default=2,
        description="Always 2 for Concept.",
    )
    derivation_run_id: str = Field(
        description="ID of the consolidation run that produced this.",
    )
    derived_at: datetime = Field(
        default_factory=_utcnow,
        description="When this concept was derived.",
    )
    status: DerivedStatus = Field(
        default=DerivedStatus.active,
        description="Lifecycle status.",
    )

    @property
    def node_labels(self) -> tuple[str, ...]:
        return ("Memory", "Derived", "Concept")


class Rule(MemoryBase):
    """A causal principle derived from concepts."""

    content: str = Field(description="Human-readable description of the rule.")
    derivation_depth: int = Field(
        default=3,
        description="Always 3 for Rule.",
    )
    derivation_run_id: str = Field(
        description="ID of the consolidation run that produced this.",
    )
    derived_at: datetime = Field(
        default_factory=_utcnow,
        description="When this rule was derived.",
    )
    status: DerivedStatus = Field(
        default=DerivedStatus.active,
        description="Lifecycle status.",
    )

    @property
    def node_labels(self) -> tuple[str, ...]:
        return ("Memory", "Derived", "Rule")


# ---------------------------------------------------------------------------
# Union type for generic dispatch
# ---------------------------------------------------------------------------

MemoryNode = (
    Fact
    | Event
    | Observation
    | Decision
    | Outcome
    | Agent
    | Artifact
    | Source
    | Pattern
    | Concept
    | Rule
)

# Label-to-model mapping for deserialization
LABEL_TO_MODEL: dict[frozenset[str], type[MemoryBase]] = {
    frozenset(("Memory", "Fact")): Fact,
    frozenset(("Memory", "Temporal", "Event")): Event,
    frozenset(("Memory", "Observation")): Observation,
    frozenset(("Memory", "Temporal", "Decision")): Decision,
    frozenset(("Memory", "Temporal", "Outcome")): Outcome,
    frozenset(("Memory", "Agent")): Agent,
    frozenset(("Memory", "Artifact")): Artifact,
    frozenset(("Memory", "Source")): Source,
    frozenset(("Memory", "Derived", "Pattern")): Pattern,
    frozenset(("Memory", "Derived", "Concept")): Concept,
    frozenset(("Memory", "Derived", "Rule")): Rule,
}
