"""EngraMCP — FastMCP v2 server with three MCP tools.

Tools delegate to ``WorkingMemory`` (Redis-backed) for storage and
search.  Call ``configure(redis_url=...)`` before using the server.
"""

from __future__ import annotations

from fastmcp import FastMCP
from redis.asyncio import Redis  # type: ignore[import-untyped]

from engramcp.memory import create_memory_fragment
from engramcp.memory import WorkingMemory
from engramcp.models.schemas import CorrectionAction
from engramcp.models.schemas import CorrectMemoryResult
from engramcp.models.schemas import GetMemoryResult
from engramcp.models.schemas import MemoryEntry
from engramcp.models.schemas import MetaInfo
from engramcp.models.schemas import SendMemoryResult
from engramcp.models.schemas import SourceEntry

mcp = FastMCP("EngraMCP")

# ---------------------------------------------------------------------------
# Working memory instance (set via configure())
# ---------------------------------------------------------------------------

_wm: WorkingMemory | None = None


async def configure(
    redis_url: str = "redis://localhost:6379",
    *,
    ttl: int = 3600,
    max_size: int = 1000,
    flush_threshold: int | None = None,
    on_flush=None,
) -> None:
    """Initialize the working memory backend.

    Must be called before the MCP tools can function.
    """
    global _wm
    client = Redis.from_url(redis_url)
    _wm = WorkingMemory(
        client,
        ttl=ttl,
        max_size=max_size,
        flush_threshold=flush_threshold,
        on_flush=on_flush,
    )


async def _reset_working_memory() -> None:
    """Clear working memory — exposed for test cleanup."""
    if _wm is not None:
        await _wm.clear()


def _get_wm() -> WorkingMemory:
    """Return the working memory instance or raise."""
    if _wm is None:
        raise RuntimeError("Working memory not configured. Call configure() first.")
    return _wm


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


_VALID_RELIABILITY_LETTERS = set("ABCDEF")


@mcp.tool
async def send_memory(
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
    wm = _get_wm()

    # Validate confidence_hint
    if confidence_hint is not None:
        hint = confidence_hint.upper()
        if hint not in _VALID_RELIABILITY_LETTERS:
            return SendMemoryResult(
                memory_id="",
                status="rejected",
            )
        confidence_hint = hint

    fragment = create_memory_fragment(
        content=content,
        source=source,
        confidence_hint=confidence_hint,
        agent_id=agent_id,
    )

    await wm.store(fragment)
    return SendMemoryResult(memory_id=fragment.id)


@mcp.tool
async def get_memory(
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
        max_depth: Reserved for Layer 6 (graph traversal depth). Currently unused.
        min_confidence: Minimum NATO rating (e.g. "B2", default "F6").
        include_contradictions: Include contradicting memories.
        include_sources: Include full source chains.
        limit: Max memories returned.
        compact: Compact mode — omit sources, chains, participants.
    """
    wm = _get_wm()
    matches = await wm.search(query, min_confidence=min_confidence)

    total_found = len(matches)
    truncated = total_found > limit
    matches = matches[:limit]

    memories: list[MemoryEntry] = []
    for m in matches:
        sources = []
        if not compact and include_sources and m.sources:
            sources = [SourceEntry(**s) for s in m.sources]

        memories.append(
            MemoryEntry(
                id=m.id,
                type=m.type,
                dynamic_type=m.dynamic_type,
                content=m.content,
                confidence=m.confidence,
                properties=m.properties,
                participants=[] if compact else m.participants,
                causal_chain=[] if compact else m.causal_chain,
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
async def correct_memory(
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
    wm = _get_wm()

    # Validate action
    try:
        action_enum = CorrectionAction(action)
    except ValueError:
        return CorrectMemoryResult(
            target_id=target_id,
            action=CorrectionAction.contest,  # placeholder
            status="rejected",
            details={"error": f"Invalid action: {action}"},
        )

    # Check target exists
    if not await wm.exists(target_id):
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
