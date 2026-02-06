"""Memory domain data models."""

from __future__ import annotations

import uuid

from pydantic import BaseModel
from pydantic import Field


class MemoryFragment(BaseModel):
    """A single memory held in working memory."""

    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    content: str
    type: str = "Fact"
    dynamic_type: str | None = None
    confidence: str | None = None
    properties: dict = Field(default_factory=dict)
    participants: list = Field(default_factory=list)
    causal_chain: list = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    agent_id: str | None = None
    agent_fingerprint: str | None = None
    timestamp: float = Field(default_factory=lambda: __import__("time").time())
