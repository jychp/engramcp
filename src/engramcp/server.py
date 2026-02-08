"""EngraMCP — FastMCP v2 server with three MCP tools.

Tools delegate to ``WorkingMemory`` (Redis-backed) for storage and
search.  Call ``configure(redis_url=...)`` before using the server.
"""

from __future__ import annotations

from fastmcp import FastMCP
from neo4j import AsyncDriver
from neo4j import AsyncGraphDatabase
from pydantic import ValidationError
from redis.asyncio import Redis  # type: ignore[import-untyped]

from engramcp.audit import AuditLogger
from engramcp.config import AuditConfig
from engramcp.config import ConsolidationConfig
from engramcp.config import EntityResolutionConfig
from engramcp.engine import ConsolidationPipeline
from engramcp.engine import ExtractionEngine
from engramcp.engine import LLMAdapter
from engramcp.graph import EntityResolver
from engramcp.graph import GraphStore
from engramcp.graph import init_schema
from engramcp.graph import MergeExecutor
from engramcp.memory import create_memory_fragment
from engramcp.memory import WorkingMemory
from engramcp.memory.schemas import MemoryFragment
from engramcp.models.schemas import CorrectionAction
from engramcp.models.schemas import CorrectMemoryInput
from engramcp.models.schemas import CorrectMemoryResult
from engramcp.models.schemas import GetMemoryInput
from engramcp.models.schemas import GetMemoryResult
from engramcp.models.schemas import MemoryEntry
from engramcp.models.schemas import MetaInfo
from engramcp.models.schemas import SendMemoryInput
from engramcp.models.schemas import SendMemoryResult
from engramcp.models.schemas import SourceEntry

mcp = FastMCP("EngraMCP")

# ---------------------------------------------------------------------------
# Working memory instance (set via configure())
# ---------------------------------------------------------------------------

_wm: WorkingMemory | None = None
_graph_driver: AsyncDriver | None = None
_consolidation_pipeline: ConsolidationPipeline | None = None


class _NoopLLMAdapter(LLMAdapter):
    """Fallback LLM adapter returning an empty extraction payload."""

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        del prompt, temperature, max_tokens, timeout_seconds
        return (
            '{"entities":[],"relations":[],"claims":[],'
            '"fragment_ids_processed":[],"errors":[]}'
        )


async def _run_consolidation(fragments: list[MemoryFragment]) -> None:
    """Run one consolidation pass and clear processed fragments from working memory."""
    pipeline = _consolidation_pipeline
    wm = _wm
    if pipeline is None or wm is None or not fragments:
        return

    await pipeline.run(fragments)
    for fragment in fragments:
        await wm.delete(fragment.id)


async def configure(
    redis_url: str = "redis://localhost:6379",
    *,
    ttl: int = 3600,
    max_size: int = 1000,
    flush_threshold: int | None = None,
    on_flush=None,
    enable_consolidation: bool = False,
    neo4j_url: str | None = None,
    consolidation_config: ConsolidationConfig | None = None,
    entity_resolution_config: EntityResolutionConfig | None = None,
    audit_config: AuditConfig | None = None,
) -> None:
    """Initialize the working memory backend.

    Must be called before the MCP tools can function.
    """
    global _wm, _graph_driver, _consolidation_pipeline
    if _wm is not None:
        await _wm.close()
    if _graph_driver is not None:
        await _graph_driver.close()
        _graph_driver = None
        _consolidation_pipeline = None

    consolidation_callback = on_flush
    threshold = flush_threshold
    if enable_consolidation:
        if neo4j_url is None:
            raise ValueError("neo4j_url is required when enable_consolidation=True")

        cfg = consolidation_config or ConsolidationConfig()
        _graph_driver = AsyncGraphDatabase.driver(neo4j_url)
        await init_schema(_graph_driver)

        graph_store = GraphStore(_graph_driver)
        extraction_engine = ExtractionEngine(
            llm=_NoopLLMAdapter(),
            consolidation_config=cfg,
        )
        resolver = EntityResolver(config=entity_resolution_config)
        merger = MergeExecutor(graph_store)
        audit_logger = AuditLogger(audit_config or AuditConfig())
        _consolidation_pipeline = ConsolidationPipeline(
            extraction_engine=extraction_engine,
            entity_resolver=resolver,
            merge_executor=merger,
            graph_store=graph_store,
            audit_logger=audit_logger,
            config=cfg,
        )
        consolidation_callback = _run_consolidation
        if threshold is None:
            threshold = cfg.fragment_threshold

    client = Redis.from_url(redis_url)
    _wm = WorkingMemory(
        client,
        ttl=ttl,
        max_size=max_size,
        flush_threshold=threshold,
        on_flush=consolidation_callback,
    )


async def shutdown() -> None:
    """Close backend clients and release server resources."""
    global _wm, _graph_driver, _consolidation_pipeline
    if _wm is not None:
        await _wm.close()
        _wm = None
    if _graph_driver is not None:
        await _graph_driver.close()
        _graph_driver = None
    _consolidation_pipeline = None


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


def _validation_message(exc: ValidationError) -> str:
    err = exc.errors()[0] if exc.errors() else {}
    return str(err.get("msg", "Invalid input"))


def _send_rejected(error_code: str, message: str) -> SendMemoryResult:
    return SendMemoryResult(
        memory_id="",
        status="rejected",
        error_code=error_code,
        message=message,
    )


def _get_error(
    *,
    query: str,
    max_depth: int,
    min_confidence: str,
    error_code: str,
    message: str,
) -> GetMemoryResult:
    return GetMemoryResult(
        status="error",
        error_code=error_code,
        message=message,
        memories=[],
        contradictions=[],
        meta=MetaInfo(
            query=query,
            total_found=0,
            returned=0,
            truncated=False,
            max_depth_used=max_depth,
            min_confidence_applied=min_confidence,
            working_memory_hits=0,
            graph_hits=0,
        ),
    )


def _correct_rejected(
    target_id: str,
    *,
    action: CorrectionAction = CorrectionAction.contest,
    error_code: str,
    message: str,
) -> CorrectMemoryResult:
    return CorrectMemoryResult(
        target_id=target_id,
        action=action,
        status="rejected",
        error_code=error_code,
        message=message,
        details={},
    )


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

    try:
        validated = SendMemoryInput.model_validate(
            {
                "content": content,
                "source": source,
                "confidence_hint": confidence_hint,
                "agent_id": agent_id,
            }
        )
    except ValidationError as exc:
        return _send_rejected("validation_error", _validation_message(exc))

    # Validate confidence_hint
    if validated.confidence_hint is not None:
        hint = validated.confidence_hint.upper()
        if hint not in _VALID_RELIABILITY_LETTERS:
            return _send_rejected(
                "invalid_confidence_hint",
                "confidence_hint must be a letter between A and F.",
            )
        validated.confidence_hint = hint

    fragment = create_memory_fragment(
        content=validated.content,
        source=(
            validated.source.model_dump(exclude_none=True) if validated.source else None
        ),
        confidence_hint=validated.confidence_hint,
        agent_id=validated.agent_id,
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
    try:
        validated = GetMemoryInput.model_validate(
            {
                "query": query,
                "max_depth": max_depth,
                "min_confidence": min_confidence,
                "include_contradictions": include_contradictions,
                "include_sources": include_sources,
                "limit": limit,
                "compact": compact,
            }
        )
    except ValidationError as exc:
        return _get_error(
            query=query,
            max_depth=max_depth,
            min_confidence=min_confidence,
            error_code="validation_error",
            message=_validation_message(exc),
        )

    matches = await wm.search(validated.query, min_confidence=validated.min_confidence)

    total_found = len(matches)
    truncated = total_found > validated.limit
    matches = matches[: validated.limit]

    memories: list[MemoryEntry] = []
    for m in matches:
        sources = []
        if not validated.compact and validated.include_sources and m.sources:
            sources = [SourceEntry(**s) for s in m.sources]

        memories.append(
            MemoryEntry(
                id=m.id,
                type=m.type,
                dynamic_type=m.dynamic_type,
                content=m.content,
                confidence=m.confidence,
                properties=m.properties,
                participants=[] if validated.compact else m.participants,
                causal_chain=[] if validated.compact else m.causal_chain,
                sources=sources,
            )
        )

    meta = MetaInfo(
        query=validated.query,
        total_found=total_found,
        returned=len(memories),
        truncated=truncated,
        max_depth_used=validated.max_depth,
        min_confidence_applied=validated.min_confidence,
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
    try:
        validated = CorrectMemoryInput.model_validate(
            {
                "target_id": target_id,
                "action": action,
                "payload": payload,
            }
        )
    except ValidationError as exc:
        return _correct_rejected(
            target_id,
            error_code="validation_error",
            message=_validation_message(exc),
        )

    # Validate action
    try:
        action_enum = CorrectionAction(validated.action)
    except ValueError:
        return _correct_rejected(
            validated.target_id,
            error_code="invalid_action",
            message=f"Invalid action: {validated.action}",
        )

    # Check target exists
    if not await wm.exists(validated.target_id):
        return CorrectMemoryResult(
            target_id=validated.target_id,
            action=action_enum,
            status="not_found",
        )

    # Mock: apply correction (real logic in later sprints)
    details: dict = {}
    if validated.payload:
        details = validated.payload

    return CorrectMemoryResult(
        target_id=validated.target_id,
        action=action_enum,
        status="applied",
        details=details,
    )
