"""EngraMCP — FastMCP v2 server with three MCP tools.

Mock implementation for Sprint 1: backend logic uses a module-level dict
as working memory.  Real stores replace this in Sprint 2+.
"""

from __future__ import annotations

import uuid

from fastmcp import FastMCP

from engramcp.models.mcp import CorrectionAction
from engramcp.models.mcp import CorrectMemoryResult
from engramcp.models.mcp import GetMemoryResult
from engramcp.models.mcp import MemoryEntry
from engramcp.models.mcp import MetaInfo
from engramcp.models.mcp import SendMemoryResult
from engramcp.models.mcp import SourceEntry

mcp = FastMCP("EngraMCP")

# ---------------------------------------------------------------------------
# Mock working memory (replaced in Sprint 2)
# ---------------------------------------------------------------------------

_working_memory: dict[str, dict] = {}


def _reset_working_memory() -> None:
    """Clear working memory — exposed for test cleanup."""
    _working_memory.clear()


# ---------------------------------------------------------------------------
# Confidence helpers
# ---------------------------------------------------------------------------

_RELIABILITY_ORDER = "ABCDEF"


def _confidence_passes(confidence: str | None, min_confidence: str) -> bool:
    """Check whether *confidence* meets or exceeds *min_confidence*.

    The NATO rating is ``<letter><number>`` (e.g. ``B2``).
    Letter A is best (index 0), F is worst (index 5).
    Number 1 is best, 6 is worst.

    A memory passes the filter when its letter is <= the filter letter
    **or** its number is <= the filter number.  The loosest filter is
    ``F6`` which lets everything through.
    """
    if min_confidence == "F6":
        return True
    if not confidence or len(confidence) < 2:
        return False

    mem_letter = confidence[0].upper()
    mem_number = confidence[1:]

    flt_letter = min_confidence[0].upper()
    flt_number = min_confidence[1:]

    try:
        letter_ok = _RELIABILITY_ORDER.index(mem_letter) <= _RELIABILITY_ORDER.index(
            flt_letter
        )
        number_ok = int(mem_number) <= int(flt_number)
    except (ValueError, IndexError):
        return False

    return letter_ok and number_ok


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
def send_memory(
    content: str,
    source: dict | None = None,
    confidence_hint: str | None = None,
    agent_id: str | None = None,
) -> SendMemoryResult:
    """Ingest a memory into the EngraMCP system.

    Args:
        content: The affirmation / fact / observation as text.
        source: Optional source reference (type, ref, citation).
        confidence_hint: Source reliability hint (letter A-F).
        agent_id: Identifier of the calling agent.
    """
    memory_id = f"mem_{uuid.uuid4().hex[:8]}"

    # Default confidence: hint letter + uncorroborated number
    confidence = f"{confidence_hint or 'F'}3"

    entry: dict = {
        "id": memory_id,
        "type": "Fact",
        "dynamic_type": None,
        "content": content,
        "confidence": confidence,
        "properties": {},
        "participants": [],
        "causal_chain": [],
        "sources": [],
        "agent_id": agent_id,
    }

    if source:
        entry["sources"].append(
            {
                "id": f"src_{uuid.uuid4().hex[:8]}",
                "type": source.get("type", "unknown"),
                "ref": source.get("ref"),
                "citation": source.get("citation"),
                "reliability": confidence_hint,
                "credibility": "3",
            }
        )

    _working_memory[memory_id] = entry
    return SendMemoryResult(memory_id=memory_id)


@mcp.tool
def get_memory(
    query: str,
    max_depth: int = 3,
    min_confidence: str = "F6",
    include_contradictions: bool = True,
    include_sources: bool = True,
    limit: int = 20,
    compact: bool = False,
) -> GetMemoryResult:
    """Retrieve relevant memories from the EngraMCP system.

    Args:
        query: Natural language query.
        max_depth: Max causal chain traversal depth.
        min_confidence: Minimum NATO rating (e.g. "B2", default "F6").
        include_contradictions: Include contradicting memories.
        include_sources: Include full source chains.
        limit: Max memories returned.
        compact: Compact mode — omit sources, chains, participants.
    """
    # Simple keyword matching on working memory (mock)
    query_words = set(query.lower().split())
    matches: list[dict] = []
    for entry in _working_memory.values():
        content_words = set(entry["content"].lower().split())
        if query_words & content_words:
            if _confidence_passes(entry.get("confidence"), min_confidence):
                matches.append(entry)

    total_found = len(matches)
    truncated = total_found > limit
    matches = matches[:limit]

    memories: list[MemoryEntry] = []
    for m in matches:
        sources = []
        if not compact and m.get("sources"):
            sources = [SourceEntry(**s) for s in m["sources"]]

        memories.append(
            MemoryEntry(
                id=m["id"],
                type=m["type"],
                dynamic_type=m.get("dynamic_type"),
                content=m["content"],
                confidence=m.get("confidence"),
                properties=m.get("properties", {}),
                participants=[] if compact else m.get("participants", []),
                causal_chain=[] if compact else m.get("causal_chain", []),
                sources=sources,
            )
        )

    meta = MetaInfo(
        query=query,
        total_found=total_found,
        returned=len(memories),
        truncated=truncated,
        max_depth_used=max_depth,
        min_confidence_applied=min_confidence,
        working_memory_hits=len(memories),
        graph_hits=0,
    )

    return GetMemoryResult(
        memories=memories,
        contradictions=[],
        meta=meta,
    )


@mcp.tool
def correct_memory(
    target_id: str,
    action: str,
    payload: dict | None = None,
) -> CorrectMemoryResult:
    """Correct or annotate knowledge in the EngraMCP system.

    Args:
        target_id: ID of the memory/node to correct.
        action: One of: contest, annotate, merge_entities,
                split_entity, reclassify.
        payload: Action-specific data.
    """
    # Validate action
    action_enum = CorrectionAction(action)

    # Check target exists
    if target_id not in _working_memory:
        return CorrectMemoryResult(
            target_id=target_id,
            action=action_enum,
            status="not_found",
        )

    # Mock: apply correction (real logic in later sprints)
    details: dict = {}
    if payload:
        details = payload

    return CorrectMemoryResult(
        target_id=target_id,
        action=action_enum,
        status="applied",
        details=details,
    )
