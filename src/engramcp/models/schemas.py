"""Pydantic models for the MCP interface (send_memory, get_memory, correct_memory).

These models define the frozen API contract.
Input models validate tool arguments; output models shape responses.
FastMCP v2 serializes Pydantic models automatically.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel
from pydantic import Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CorrectionAction(str, Enum):
    """Actions available via correct_memory."""

    contest = "contest"
    annotate = "annotate"
    merge_entities = "merge_entities"
    split_entity = "split_entity"
    reclassify = "reclassify"


class ContradictionNature(str, Enum):
    """Types of contradiction detected between memories."""

    temporal_impossibility = "temporal_impossibility"
    factual_conflict = "factual_conflict"
    source_disagreement = "source_disagreement"
    logical_inconsistency = "logical_inconsistency"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class SourceInput(BaseModel):
    """Source reference attached to a memory."""

    type: str | None = Field(
        default=None,
        description="Kind of source (e.g. court_document, article, testimony).",
    )
    ref: str | None = Field(
        default=None,
        description="URI or locator for the original document.",
    )
    citation: str | None = Field(
        default=None,
        description="Precise location within the source (page, paragraph, etc.).",
    )


class SendMemoryInput(BaseModel):
    """Input for send_memory tool."""

    content: str = Field(
        description="The affirmation, fact, or observation as free text.",
    )
    source: SourceInput | None = Field(
        default=None,
        description="Optional provenance reference for the memory.",
    )
    confidence_hint: str | None = Field(
        default=None,
        description="Source reliability hint, single letter A-F.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Identifier of the calling agent.",
    )


class GetMemoryInput(BaseModel):
    """Input for get_memory tool."""

    query: str = Field(
        description="Natural language search query.",
    )
    max_depth: int = Field(
        default=3,
        description="Maximum causal chain traversal depth.",
    )
    min_confidence: str = Field(
        default="F6",
        description="Minimum NATO rating filter, e.g. 'B2'.",
    )
    include_contradictions: bool = Field(
        default=True,
        description="Whether to include contradicting memories.",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include full source chains.",
    )
    limit: int = Field(
        default=20,
        description="Maximum number of memories to return.",
    )
    compact: bool = Field(
        default=False,
        description="Compact mode: omit sources, chains, and participants.",
    )


class CorrectMemoryInput(BaseModel):
    """Input for correct_memory tool."""

    target_id: str = Field(
        description="ID of the memory or node to correct.",
    )
    action: CorrectionAction = Field(
        description="Correction action to apply.",
    )
    payload: dict | None = Field(
        default=None,
        description="Action-specific data (reason, new labels, merge targets, etc.).",
    )


# ---------------------------------------------------------------------------
# Output models — send_memory
# ---------------------------------------------------------------------------


class SendMemoryResult(BaseModel):
    """Response from send_memory."""

    memory_id: str = Field(
        description="ID assigned to the newly created memory.",
    )
    status: str = Field(
        default="accepted",
        description="Ingestion status (accepted, rejected).",
    )


# ---------------------------------------------------------------------------
# Output models — get_memory
# ---------------------------------------------------------------------------


class Participant(BaseModel):
    """An entity participating in a memory."""

    id: str = Field(
        description="Unique identifier of the participant node.",
    )
    type: str = Field(
        description="Ontology type of the participant (Person, Organization, etc.).",
    )
    dynamic_type: str | None = Field(
        default=None,
        description="LLM-inferred subtype refined during consolidation.",
    )
    name: str = Field(
        description="Display name of the participant.",
    )
    role: str | None = Field(
        default=None,
        description="Role played in this memory (witness, suspect, etc.).",
    )
    confidence: str | None = Field(
        default=None,
        description="NATO rating for this participant link.",
    )


class CausalLink(BaseModel):
    """A single causal relation from a memory."""

    relation: str = Field(
        description="Type of causal relation (caused_by, led_to, etc.).",
    )
    target_id: str = Field(
        description="ID of the target memory in the causal chain.",
    )
    target_summary: str = Field(
        description="Short summary of the target memory.",
    )
    confidence: str | None = Field(
        default=None,
        description="NATO rating for this causal link.",
    )


class SourceEntry(BaseModel):
    """A source backing a memory."""

    id: str = Field(
        description="Unique identifier of the source record.",
    )
    type: str = Field(
        description="Kind of source (court_document, article, testimony, etc.).",
    )
    ref: str | None = Field(
        default=None,
        description="URI or locator for the original document.",
    )
    citation: str | None = Field(
        default=None,
        description="Precise location within the source.",
    )
    reliability: str | None = Field(
        default=None,
        description="Source reliability letter (A-F).",
    )
    credibility: str | None = Field(
        default=None,
        description="Information credibility number (1-6).",
    )


class MemoryEntry(BaseModel):
    """A single memory in the get_memory response."""

    id: str = Field(
        description="Unique identifier of the memory.",
    )
    type: str = Field(
        description="Ontology node type (Fact, Event, Entity, etc.).",
    )
    dynamic_type: str | None = Field(
        default=None,
        description="LLM-inferred subtype refined during consolidation.",
    )
    content: str = Field(
        description="Textual content of the memory.",
    )
    confidence: str | None = Field(
        default=None,
        description="NATO two-dimensional rating, e.g. 'B2'.",
    )
    properties: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata extracted from content.",
    )
    participants: list[Participant] = Field(
        default_factory=list,
        description="Entities involved in this memory.",
    )
    causal_chain: list[CausalLink] = Field(
        default_factory=list,
        description="Ordered causal links to related memories.",
    )
    sources: list[SourceEntry] = Field(
        default_factory=list,
        description="Provenance records tracing back to original documents.",
    )


class Contradiction(BaseModel):
    """A detected contradiction between memories."""

    id: str = Field(
        description="Unique identifier of the contradiction.",
    )
    memory_id: str = Field(
        description="ID of the first memory in the contradiction pair.",
    )
    contradicting_memory: MemoryEntry = Field(
        description="The second memory that contradicts the first.",
    )
    nature: ContradictionNature = Field(
        description="Classification of the contradiction type.",
    )
    resolution_status: str = Field(
        default="unresolved",
        description="Current resolution state (unresolved, resolved, dismissed).",
    )
    detected_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when the contradiction was detected.",
    )


class MetaInfo(BaseModel):
    """Metadata about the get_memory response."""

    query: str = Field(
        description="Original search query submitted by the caller.",
    )
    total_found: int = Field(
        description="Total number of matching memories before truncation.",
    )
    returned: int = Field(
        description="Number of memories actually returned.",
    )
    truncated: bool = Field(
        default=False,
        description="Whether results were truncated by the limit.",
    )
    max_depth_used: int | None = Field(
        default=None,
        description="Causal chain depth actually traversed.",
    )
    min_confidence_applied: str | None = Field(
        default=None,
        description="NATO rating filter that was applied.",
    )
    retrieval_ms: int | None = Field(
        default=None,
        description="Wall-clock retrieval time in milliseconds.",
    )
    working_memory_hits: int = Field(
        default=0,
        description="Number of matches from working memory.",
    )
    graph_hits: int = Field(
        default=0,
        description="Number of matches from the knowledge graph.",
    )


class GetMemoryResult(BaseModel):
    """Response from get_memory."""

    memories: list[MemoryEntry] = Field(
        default_factory=list,
        description="Matching memories ordered by relevance.",
    )
    contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="Contradictions detected among returned memories.",
    )
    meta: MetaInfo = Field(
        description="Response metadata (counts, filters, timing).",
    )


# ---------------------------------------------------------------------------
# Output models — correct_memory
# ---------------------------------------------------------------------------


class CorrectMemoryResult(BaseModel):
    """Response from correct_memory."""

    target_id: str = Field(
        description="ID of the memory or node that was corrected.",
    )
    action: CorrectionAction = Field(
        description="Correction action that was applied.",
    )
    status: str = Field(
        default="applied",
        description="Outcome status (applied, not_found, rejected).",
    )
    details: dict = Field(
        default_factory=dict,
        description="Action-specific result data.",
    )
