"""Extraction result models.

Pydantic schemas for the output of LLM-based extraction. These are
intermediate representations — entity resolution and graph integration
happen downstream in the consolidation pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class TemporalInfo(BaseModel):
    """Temporal information extracted from text.

    Uses strings (not datetime) — parsing to datetime + TemporalPrecision
    happens in the consolidation pipeline.
    """

    occurred_at: str | None = Field(
        default=None,
        description="When it happened (ISO-8601 string).",
    )
    occurred_until: str | None = Field(
        default=None,
        description="End of time range.",
    )
    precision: str = Field(
        default="unknown",
        description="Granularity hint (exact, day, month, year, approximate, unknown).",
    )


class ExtractedEntity(BaseModel):
    """An entity extracted by the LLM.

    Entities reference by **name**, not ID. Entity resolution (Sprint 6c)
    maps names to graph nodes.
    """

    name: str = Field(description="Primary name of the entity.")
    type: str = Field(description="Ontology node type (Agent, Artifact, etc.).")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names or identifiers.",
    )
    properties: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata.",
    )
    disambiguating_context: str | None = Field(
        default=None,
        description="Context to help entity resolution.",
    )
    source_fragment_ids: list[str] = Field(
        default_factory=list,
        description="Which fragments mentioned this entity.",
    )


class ExtractedRelation(BaseModel):
    """A relation extracted by the LLM."""

    from_entity: str = Field(description="Source entity name.")
    to_entity: str = Field(description="Target entity name.")
    relation_type: str = Field(description="Ontology relation type.")
    properties: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata.",
    )
    source_fragment_ids: list[str] = Field(
        default_factory=list,
        description="Which fragments mentioned this relation.",
    )


class ExtractedClaim(BaseModel):
    """A claim (fact, event, observation, etc.) extracted by the LLM."""

    content: str = Field(description="The claim text.")
    claim_type: str = Field(
        default="Fact",
        description="Ontology node type (Fact, Event, Observation, etc.).",
    )
    confidence_hint: str | None = Field(
        default=None,
        description="NATO confidence hint (e.g. 'B2').",
    )
    temporal_info: TemporalInfo | None = Field(
        default=None,
        description="When the claim occurred.",
    )
    properties: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata.",
    )
    involved_entities: list[str] = Field(
        default_factory=list,
        description="Entity names involved in this claim.",
    )
    source_fragment_ids: list[str] = Field(
        default_factory=list,
        description="Which fragments support this claim.",
    )


class ExtractionResult(BaseModel):
    """Aggregate output of an extraction run.

    The ``errors`` field enables partial success: failed batches
    record their error but do not abort the pipeline.
    """

    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Extracted entities.",
    )
    relations: list[ExtractedRelation] = Field(
        default_factory=list,
        description="Extracted relations.",
    )
    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        description="Extracted claims.",
    )
    fragment_ids_processed: list[str] = Field(
        default_factory=list,
        description="IDs of fragments that were processed.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages from failed batches.",
    )
