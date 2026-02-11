"""Memory domain data models."""

from __future__ import annotations

import uuid

from pydantic import BaseModel
from pydantic import Field


class MemoryFragment(BaseModel):
    """A single memory held in working memory."""

    id: str = Field(
        default_factory=lambda: f"mem_{uuid.uuid4().hex}",
        description="Unique identifier, auto-generated as mem_{uuid4_hex}.",
    )
    content: str = Field(
        description="Raw textual content of the memory.",
    )
    type: str = Field(
        default="Fact",
        description="Ontology node type (Fact, Event, Entity, etc.).",
    )
    dynamic_type: str | None = Field(
        default=None,
        description="LLM-inferred subtype refined during consolidation.",
    )
    confidence: str | None = Field(
        default=None,
        description="NATO two-dimensional rating, e.g. 'B2'.",
    )
    properties: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata extracted from content.",
    )
    participants: list = Field(
        default_factory=list,
        description="Entities involved in this memory.",
    )
    causal_chain: list = Field(
        default_factory=list,
        description="Ordered causal links to related memories.",
    )
    sources: list[dict] = Field(
        default_factory=list,
        description="Provenance records tracing back to original documents.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Identifier of the agent that submitted this memory.",
    )
    agent_fingerprint: str | None = Field(
        default=None,
        description="SHA-256[:16] deterministic hash of agent_id.",
    )
    timestamp: float = Field(
        default_factory=lambda: __import__("time").time(),
        description="Unix epoch when the fragment was created.",
    )
