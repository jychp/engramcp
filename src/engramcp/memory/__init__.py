"""Memory domain — working memory buffer and fragment lifecycle."""

from __future__ import annotations

import uuid

from engramcp.memory.schemas import MemoryFragment
from engramcp.memory.store import WorkingMemory
from engramcp.models import agent_fingerprint

__all__ = ["MemoryFragment", "WorkingMemory", "create_memory_fragment"]


def create_memory_fragment(
    content: str,
    *,
    source: dict | None = None,
    confidence_hint: str | None = None,
    agent_id: str | None = None,
) -> MemoryFragment:
    """Factory for creating a MemoryFragment with all domain invariants.

    Encapsulates confidence rating construction, source assembly, and
    agent fingerprint computation.
    """
    memory_id = f"mem_{uuid.uuid4().hex}"

    # Default confidence: hint letter + uncorroborated number
    confidence = f"{confidence_hint or 'F'}3"

    sources: list[dict] = []
    if source:
        sources.append(
            {
                "id": f"src_{uuid.uuid4().hex}",
                "type": source.get("type", "unknown"),
                "ref": source.get("ref"),
                "citation": source.get("citation"),
                "reliability": confidence_hint,
                "credibility": "3",
            }
        )

    return MemoryFragment(
        id=memory_id,
        content=content,
        type="Fact",
        confidence=confidence,
        sources=sources,
        agent_id=agent_id,
        agent_fingerprint=agent_fingerprint(agent_id),
    )
