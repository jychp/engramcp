"""Pydantic models for the MCP interface (send_memory, get_memory, correct_memory).

These models define the frozen API contract for Sprint 1.
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

    type: str | None = None
    ref: str | None = None
    citation: str | None = None


class SendMemoryInput(BaseModel):
    """Input for send_memory tool."""

    content: str
    source: SourceInput | None = None
    confidence_hint: str | None = None
    agent_id: str | None = None


class GetMemoryInput(BaseModel):
    """Input for get_memory tool."""

    query: str
    max_depth: int = 3
    min_confidence: str = "F6"
    include_contradictions: bool = True
    include_sources: bool = True
    limit: int = 20
    compact: bool = False


class CorrectMemoryInput(BaseModel):
    """Input for correct_memory tool."""

    target_id: str
    action: CorrectionAction
    payload: dict | None = None


# ---------------------------------------------------------------------------
# Output models — send_memory
# ---------------------------------------------------------------------------


class SendMemoryResult(BaseModel):
    """Response from send_memory."""

    memory_id: str
    status: str = "accepted"


# ---------------------------------------------------------------------------
# Output models — get_memory
# ---------------------------------------------------------------------------


class Participant(BaseModel):
    """An entity participating in a memory."""

    id: str
    type: str
    dynamic_type: str | None = None
    name: str
    role: str | None = None
    confidence: str | None = None


class CausalLink(BaseModel):
    """A single causal relation from a memory."""

    relation: str
    target_id: str
    target_summary: str
    confidence: str | None = None


class SourceEntry(BaseModel):
    """A source backing a memory."""

    id: str
    type: str
    ref: str | None = None
    citation: str | None = None
    reliability: str | None = None
    credibility: str | None = None


class MemoryEntry(BaseModel):
    """A single memory in the get_memory response."""

    id: str
    type: str
    dynamic_type: str | None = None
    content: str
    confidence: str | None = None
    properties: dict = Field(default_factory=dict)
    participants: list[Participant] = Field(default_factory=list)
    causal_chain: list[CausalLink] = Field(default_factory=list)
    sources: list[SourceEntry] = Field(default_factory=list)


class Contradiction(BaseModel):
    """A detected contradiction between memories."""

    id: str
    memory_id: str
    contradicting_memory: MemoryEntry
    nature: ContradictionNature
    resolution_status: str = "unresolved"
    detected_at: str | None = None


class MetaInfo(BaseModel):
    """Metadata about the get_memory response."""

    query: str
    total_found: int
    returned: int
    truncated: bool = False
    max_depth_used: int | None = None
    min_confidence_applied: str | None = None
    retrieval_ms: int | None = None
    working_memory_hits: int = 0
    graph_hits: int = 0


class GetMemoryResult(BaseModel):
    """Response from get_memory."""

    memories: list[MemoryEntry] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    meta: MetaInfo


# ---------------------------------------------------------------------------
# Output models — correct_memory
# ---------------------------------------------------------------------------


class CorrectMemoryResult(BaseModel):
    """Response from correct_memory."""

    target_id: str
    action: CorrectionAction
    status: str = "applied"
    details: dict = Field(default_factory=dict)
