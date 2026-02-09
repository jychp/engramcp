"""EngraMCP — FastMCP v2 server with three MCP tools.

Tools delegate to ``WorkingMemory`` (Redis-backed) for storage and
search.  Call ``configure(redis_url=...)`` before using the server.
"""

from __future__ import annotations

import json
import time
from time import perf_counter

from fastmcp import FastMCP
from neo4j import AsyncDriver
from neo4j import AsyncGraphDatabase
from pydantic import ValidationError
from redis.asyncio import Redis  # type: ignore[import-untyped]

from engramcp.audit import AuditEvent
from engramcp.audit import AuditEventType
from engramcp.audit import AuditLogger
from engramcp.config import AuditConfig
from engramcp.config import ConsolidationConfig
from engramcp.config import EntityResolutionConfig
from engramcp.config import LLMConfig
from engramcp.engine import build_llm_adapter
from engramcp.engine import ConceptRegistry
from engramcp.engine import ConsolidationPipeline
from engramcp.engine import ExtractionEngine
from engramcp.engine import LLMAdapter
from engramcp.engine import QueryDemandTracker
from engramcp.engine import RetrievalEngine
from engramcp.graph import EntityResolver
from engramcp.graph import GraphStore
from engramcp.graph import init_schema
from engramcp.graph import MergeExecutor
from engramcp.memory import create_memory_fragment
from engramcp.memory import WorkingMemory
from engramcp.memory.schemas import MemoryFragment
from engramcp.models.nodes import Agent
from engramcp.models.nodes import MemoryNode
from engramcp.models.schemas import AnnotatePayload
from engramcp.models.schemas import ContestPayload
from engramcp.models.schemas import CorrectionAction
from engramcp.models.schemas import CorrectMemoryInput
from engramcp.models.schemas import CorrectMemoryResult
from engramcp.models.schemas import GetMemoryInput
from engramcp.models.schemas import GetMemoryResult
from engramcp.models.schemas import MergeEntitiesPayload
from engramcp.models.schemas import MetaInfo
from engramcp.models.schemas import ReclassifyPayload
from engramcp.models.schemas import SendMemoryInput
from engramcp.models.schemas import SendMemoryResult
from engramcp.models.schemas import SplitEntityPayload
from engramcp.observability import record_latency

mcp = FastMCP("EngraMCP")

# ---------------------------------------------------------------------------
# Working memory instance (set via configure())
# ---------------------------------------------------------------------------

_wm: WorkingMemory | None = None
_graph_driver: AsyncDriver | None = None
_consolidation_pipeline: ConsolidationPipeline | None = None
_retrieval_engine: RetrievalEngine | None = None
_audit_logger: AuditLogger | None = None


async def _run_consolidation(fragments: list[MemoryFragment]) -> None:
    """Run one consolidation pass and clear processed fragments from working memory."""
    start = perf_counter()
    pipeline = _consolidation_pipeline
    wm = _wm
    if pipeline is None or wm is None or not fragments:
        return

    ok = False
    try:
        run_result = await pipeline.run(fragments)
        had_mutation = any(
            (
                run_result.entities_created,
                run_result.entities_merged,
                run_result.entities_linked,
                run_result.claims_created,
                run_result.relations_created,
                run_result.contradictions_detected,
                run_result.patterns_created,
                run_result.concepts_created,
                run_result.rules_created,
            )
        )
        if run_result.errors and not had_mutation:
            msg = "; ".join(run_result.errors[:3])
            raise RuntimeError(
                "Consolidation produced no mutations and reported errors; "
                f"skipping fragment deletion for retry: {msg}"
            )

        for fragment in fragments:
            await wm.delete(fragment.id)
        ok = True
    finally:
        record_latency(
            operation="consolidation.run",
            duration_ms=(perf_counter() - start) * 1000,
            ok=ok,
        )


async def configure(
    redis_url: str = "redis://localhost:6379",
    *,
    ttl: int = 3600,
    max_size: int = 1000,
    flush_threshold: int | None = None,
    on_flush=None,
    enable_consolidation: bool = False,
    neo4j_url: str | None = None,
    llm_config: LLMConfig | None = None,
    llm_adapter: LLMAdapter | None = None,
    consolidation_config: ConsolidationConfig | None = None,
    entity_resolution_config: EntityResolutionConfig | None = None,
    audit_config: AuditConfig | None = None,
) -> None:
    """Initialize the working memory backend.

    Must be called before the MCP tools can function.
    """
    global _wm, _graph_driver, _consolidation_pipeline, _retrieval_engine, _audit_logger
    if _wm is not None:
        try:
            await _wm.close()
        except RuntimeError:
            # Tests may reconfigure across event loops.
            pass
    if _graph_driver is not None:
        try:
            await _graph_driver.close()
        except RuntimeError:
            # Tests may reconfigure across event loops.
            pass
        _graph_driver = None
        _consolidation_pipeline = None

    _audit_logger = AuditLogger(audit_config or AuditConfig())

    consolidation_callback = on_flush
    threshold = flush_threshold
    if enable_consolidation:
        if neo4j_url is None:
            raise ValueError("neo4j_url is required when enable_consolidation=True")

        cfg = consolidation_config or ConsolidationConfig()
        llm_cfg = llm_config or LLMConfig()
        adapter = llm_adapter or build_llm_adapter(llm_cfg)

        _graph_driver = AsyncGraphDatabase.driver(neo4j_url)
        await init_schema(_graph_driver)

        graph_store = GraphStore(_graph_driver)
        extraction_engine = ExtractionEngine(
            llm=adapter,
            llm_config=llm_cfg,
            consolidation_config=cfg,
        )
        resolver = EntityResolver(config=entity_resolution_config)
        merger = MergeExecutor(graph_store)
        _consolidation_pipeline = ConsolidationPipeline(
            extraction_engine=extraction_engine,
            entity_resolver=resolver,
            merge_executor=merger,
            graph_store=graph_store,
            audit_logger=_audit_logger,
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
    _retrieval_engine = RetrievalEngine(
        _wm,
        graph_retriever=(
            GraphStore(_graph_driver) if _graph_driver is not None else None
        ),
        demand_tracker=QueryDemandTracker(),
        concept_registry=ConceptRegistry(),
    )


async def shutdown() -> None:
    """Close backend clients and release server resources."""
    global _wm, _graph_driver, _consolidation_pipeline, _retrieval_engine, _audit_logger
    if _wm is not None:
        await _wm.close()
        _wm = None
    if _graph_driver is not None:
        await _graph_driver.close()
        _graph_driver = None
    _consolidation_pipeline = None
    _retrieval_engine = None
    _audit_logger = None


async def _reset_working_memory() -> None:
    """Clear working memory — exposed for test cleanup."""
    if _wm is not None:
        await _wm.clear()


def _get_wm() -> WorkingMemory:
    """Return the working memory instance or raise."""
    if _wm is None:
        raise RuntimeError("Working memory not configured. Call configure() first.")
    return _wm


def _get_query_demand_count(
    *,
    node_types: list[str] | None = None,
    properties: list[str] | None = None,
) -> int:
    """Return tracked count for a normalized retrieval shape (test helper)."""
    if _retrieval_engine is None:
        return 0
    return _retrieval_engine.query_demand_count(
        node_types=node_types, properties=properties
    )


def _get_concept_candidate_count() -> int:
    """Return current number of tracked concept candidates (test helper)."""
    if _retrieval_engine is None:
        return 0
    return _retrieval_engine.concept_candidate_count()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


_VALID_RELIABILITY_LETTERS = set("ABCDEF")
_ALLOWED_GRAPH_REL_TYPES = {
    "SOURCED_FROM",
    "DERIVED_FROM",
    "CITES",
    "CAUSED_BY",
    "LEADS_TO",
    "PRECEDED",
    "FOLLOWED",
    "SUPPORTS",
    "CONTRADICTS",
    "PARTICIPATED_IN",
    "DECIDED_BY",
    "OBSERVED_BY",
    "MENTIONS",
    "CONCERNS",
    "GENERALIZES",
    "INSTANCE_OF",
    "POSSIBLY_SAME_AS",
    "MERGED_FROM",
}


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


async def _log_correct_memory_event(payload: dict) -> None:
    """Write a CORRECT_MEMORY audit event when audit logging is configured."""
    if _audit_logger is None:
        return
    await _audit_logger.log(
        AuditEvent(event_type=AuditEventType.CORRECT_MEMORY, payload=payload)
    )


def _downgrade_confidence(confidence: str | None) -> str:
    """Downgrade a NATO rating by one conservative step, capped at F6."""
    if not confidence or len(confidence) < 2:
        return "F6"

    letters = "ABCDEF"
    letter = confidence[0].upper()
    number_raw = confidence[1:]
    try:
        letter_idx = letters.index(letter)
        number = int(number_raw)
    except (ValueError, IndexError):
        return "F6"

    number = min(max(number, 1), 6)
    if number < 6:
        return f"{letter}{number + 1}"

    if letter_idx < len(letters) - 1:
        return f"{letters[letter_idx + 1]}6"
    return "F6"


def _contest_cascade_hook(target_id: str) -> dict:
    """Placeholder for future graph confidence cascade wiring."""
    return {
        "triggered": False,
        "reason": "cascade_not_configured",
        "target_id": target_id,
    }


async def _replace_fragment(wm: WorkingMemory, fragment: MemoryFragment) -> None:
    """Replace an existing fragment while preserving ID and index consistency."""
    await wm.delete(fragment.id)
    await wm.store(fragment)


def _confidence_sort_key(confidence: str | None) -> tuple[int, int]:
    """Return sortable quality key (smaller is better)."""
    if not confidence or len(confidence) < 2:
        return (99, 99)
    letters = "ABCDEF"
    try:
        return (letters.index(confidence[0].upper()), int(confidence[1:]))
    except (ValueError, IndexError):
        return (99, 99)


def _best_confidence(*values: str | None) -> str | None:
    candidates = [value for value in values if value is not None]
    if not candidates:
        return None
    return min(candidates, key=_confidence_sort_key)


def _normalize_reclassify_history_entry(entry: object) -> dict[str, object] | None:
    """Normalize one reclassify history entry to ``{from,to,at}`` shape.

    Supports:
    - Dict entries from WM path
    - JSON-string encoded dict entries from graph path
    - Legacy ``from->to@timestamp`` string format
    """
    if isinstance(entry, dict):
        src = entry.get("from")
        dst = entry.get("to")
        at = entry.get("at")
        if src is None or dst is None:
            return None
        try:
            at_value = float(at or 0.0)
        except (TypeError, ValueError):
            at_value = 0.0
        return {"from": str(src), "to": str(dst), "at": at_value}

    if isinstance(entry, str):
        # Preferred graph representation: JSON string entry
        try:
            decoded = json.loads(entry)
            if isinstance(decoded, dict):
                return _normalize_reclassify_history_entry(decoded)
        except json.JSONDecodeError:
            pass

        # Backward compatibility with legacy "A->B@ts" format
        if "->" in entry and "@" in entry:
            pair, _, ts = entry.partition("@")
            src, _, dst = pair.partition("->")
            if src and dst:
                try:
                    at_val = float(ts)
                except ValueError:
                    at_val = 0.0
                return {"from": src, "to": dst, "at": at_val}

    return None


def _reclassify_history_record(
    old_type: str, new_type: str, at: float
) -> dict[str, object]:
    return {"from": old_type, "to": new_type, "at": at}


def _build_split_graph_node(target_node: MemoryNode, split_value: str) -> MemoryNode:
    """Clone a graph node for split operation with content/name overridden."""
    payload = target_node.model_dump(exclude={"id", "created_at", "updated_at"})
    if "name" in payload:
        payload["name"] = split_value
        if isinstance(target_node, Agent):
            payload["aliases"] = list(
                dict.fromkeys([*target_node.aliases, target_node.name])
            )
    elif "content" in payload:
        payload["content"] = split_value
    else:
        msg = f"Node type {type(target_node).__name__} cannot be split"
        raise ValueError(msg)
    return type(target_node)(**payload)


async def _split_entity_in_graph(target_id: str, split_into: list[str]) -> dict | None:
    """Split a graph node into multiple nodes and duplicate its relationships."""
    if _graph_driver is None:
        return None

    graph_store = GraphStore(_graph_driver)
    target_node = await graph_store.get_node(target_id)
    if target_node is None:
        return None

    created_memory_ids: list[str] = []
    for split_value in split_into:
        new_node = _build_split_graph_node(target_node, split_value)
        created_memory_ids.append(await graph_store.create_node(new_node))

    outgoing: list[dict] = []
    incoming: list[dict] = []
    async with _graph_driver.session() as session:
        outgoing_result = await session.run(
            "MATCH (n:Memory {id: $id})-[r]->(m:Memory) "
            "RETURN type(r) AS rel_type, properties(r) AS props, m.id AS other_id",
            id=target_id,
        )
        outgoing = [record.data() async for record in outgoing_result]

        incoming_result = await session.run(
            "MATCH (m:Memory)-[r]->(n:Memory {id: $id}) "
            "RETURN type(r) AS rel_type, properties(r) AS props, m.id AS other_id",
            id=target_id,
        )
        incoming = [record.data() async for record in incoming_result]

        redistributed = 0
        for child_id in created_memory_ids:
            for rel in outgoing:
                rel_type = str(rel.get("rel_type", ""))
                if rel_type not in _ALLOWED_GRAPH_REL_TYPES:
                    continue
                await session.run(
                    "MATCH (a:Memory {id: $from_id}), (b:Memory {id: $to_id}) "
                    f"CREATE (a)-[r:{rel_type}]->(b) "
                    "SET r = $props",
                    from_id=child_id,
                    to_id=rel.get("other_id"),
                    props=rel.get("props") or {},
                )
                redistributed += 1
            for rel in incoming:
                rel_type = str(rel.get("rel_type", ""))
                if rel_type not in _ALLOWED_GRAPH_REL_TYPES:
                    continue
                await session.run(
                    "MATCH (a:Memory {id: $from_id}), (b:Memory {id: $to_id}) "
                    f"CREATE (a)-[r:{rel_type}]->(b) "
                    "SET r = $props",
                    from_id=rel.get("other_id"),
                    to_id=child_id,
                    props=rel.get("props") or {},
                )
                redistributed += 1

    await graph_store.delete_node(target_id)
    return {
        "split_into": split_into,
        "created_memory_ids": created_memory_ids,
        "redistributed_relationships": redistributed,
        "storage": "graph",
    }


async def _merge_entities_in_graph(
    target_id: str,
    merge_with_id: str,
) -> dict | None:
    """Merge entities in graph storage when Neo4j is configured.

    Returns merge details when both nodes are present in graph, else ``None``.
    """
    if _graph_driver is None:
        return None

    graph_store = GraphStore(_graph_driver)
    target_node = await graph_store.get_node(target_id)
    merge_with_node = await graph_store.get_node(merge_with_id)
    if target_node is None or merge_with_node is None:
        return None

    merge_executor = MergeExecutor(graph_store)
    merge_result = await merge_executor.execute_merge(
        survivor_id=target_id,
        absorbed_id=merge_with_id,
        merge_run_id=f"correct_memory_{int(time.time() * 1000)}",
    )
    return {
        "merged_into": merge_result.survivor_id,
        "merged_from": merge_result.absorbed_id,
        "aliases_added": merge_result.aliases_added,
        "relations_transferred": merge_result.relations_transferred,
        "storage": "graph",
    }


async def _reclassify_in_graph(target_id: str, new_type: str) -> dict | None:
    """Reclassify a graph node through lifecycle updates when Neo4j is configured."""
    if _graph_driver is None:
        return None

    async with _graph_driver.session() as session:
        result = await session.run(
            "MATCH (n:Memory {id: $id}) "
            "RETURN labels(n) AS labels, properties(n) AS props",
            id=target_id,
        )
        record = await result.single()
        if record is None:
            return None

        labels = set(record["labels"])
        props = record["props"] or {}
        known_types = (
            "Fact",
            "Event",
            "Observation",
            "Decision",
            "Outcome",
            "Agent",
            "Artifact",
            "Source",
            "Pattern",
            "Concept",
            "Rule",
        )
        old_type = next((label for label in known_types if label in labels), "Memory")

        history_raw = props.get("reclassify_history", [])
        if not isinstance(history_raw, list):
            history_raw = []
        history: list[dict[str, object]] = []
        for item in history_raw:
            normalized = _normalize_reclassify_history_entry(item)
            if normalized is not None:
                history.append(normalized)
        now = time.time()
        history.append(_reclassify_history_record(old_type, new_type, now))

        updates: dict = {
            # Neo4j properties cannot store nested maps; keep structured entries
            # encoded as JSON strings for parity with WM history shape.
            "reclassify_history": [
                json.dumps(item, separators=(",", ":")) for item in history
            ],
            "reclassified_to": new_type,
            "reclassified_at": now,
            "updated_at": now,
        }
        details: dict = {
            "old_type": old_type,
            "new_type": new_type,
            "storage": "graph",
        }
        if "Derived" in labels:
            updates["status"] = "dissolved"
            updates["dissolved_at"] = now
            updates["dissolved_reason"] = f"reclassified_to_{new_type}"
            details["lifecycle"] = {"target_status": "dissolved"}

        await session.run(
            "MATCH (n:Memory {id: $id}) SET n += $updates",
            id=target_id,
            updates=updates,
        )

    if "Derived" in labels:
        from engramcp.engine.confidence import ConfidenceEngine
        from engramcp.graph.traceability import SourceTraceability

        confidence_engine = ConfidenceEngine(
            GraphStore(_graph_driver),
            SourceTraceability(_graph_driver),
        )
        cascade = await confidence_engine.cascade_contest(target_id)
        details["lifecycle"]["cascade"] = {
            "affected_nodes": cascade.affected_nodes,
            "reason": cascade.reason,
        }

    return details


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
    start = perf_counter()
    ok = False
    try:
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
                validated.source.model_dump(exclude_none=True)
                if validated.source
                else None
            ),
            confidence_hint=validated.confidence_hint,
            agent_id=validated.agent_id,
        )

        await wm.store(fragment)
        ok = True
        return SendMemoryResult(memory_id=fragment.id)
    finally:
        record_latency(
            operation="mcp.send_memory",
            duration_ms=(perf_counter() - start) * 1000,
            ok=ok,
        )


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
    start = perf_counter()
    ok = False
    try:
        _get_wm()
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

        if _retrieval_engine is None:
            return _get_error(
                query=validated.query,
                max_depth=validated.max_depth,
                min_confidence=validated.min_confidence,
                error_code="retrieval_engine_not_configured",
                message="Retrieval engine not configured. Call configure() first.",
            )

        result = await _retrieval_engine.retrieve(validated)
        ok = result.status == "ok"
        return result
    finally:
        record_latency(
            operation="mcp.get_memory",
            duration_ms=(perf_counter() - start) * 1000,
            ok=ok,
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

    if action_enum in (
        CorrectionAction.contest,
        CorrectionAction.annotate,
    ) and not await wm.exists(validated.target_id):
        return CorrectMemoryResult(
            target_id=validated.target_id,
            action=action_enum,
            status="not_found",
        )

    if action_enum == CorrectionAction.split_entity:
        try:
            split_payload = SplitEntityPayload.model_validate(validated.payload or {})
        except ValidationError as exc:
            return _correct_rejected(
                validated.target_id,
                action=action_enum,
                error_code="validation_error",
                message=_validation_message(exc),
            )
        if not split_payload.split_into:
            return _correct_rejected(
                validated.target_id,
                action=action_enum,
                error_code="validation_error",
                message="split_into must contain at least one item.",
            )

        graph_details = await _split_entity_in_graph(
            validated.target_id, split_payload.split_into
        )
        if graph_details is not None:
            await _log_correct_memory_event(
                {
                    "target_id": validated.target_id,
                    "action": action_enum.value,
                    "status": "applied",
                    "created_memory_ids": graph_details["created_memory_ids"],
                    "split_into": graph_details["split_into"],
                    "storage": graph_details["storage"],
                    "redistributed_relationships": graph_details[
                        "redistributed_relationships"
                    ],
                }
            )
            return CorrectMemoryResult(
                target_id=validated.target_id,
                action=action_enum,
                status="applied",
                details=graph_details,
            )

        target = await wm.get(validated.target_id)
        if target is None:
            return CorrectMemoryResult(
                target_id=validated.target_id,
                action=action_enum,
                status="not_found",
            )

        source_input: dict | None = None
        if target.sources:
            source = target.sources[0]
            source_input = {
                "type": source.get("type"),
                "ref": source.get("ref"),
                "citation": source.get("citation"),
            }
        confidence_hint = target.confidence[0] if target.confidence else None

        created_memory_ids: list[str] = []
        for split_item in split_payload.split_into:
            fragment = create_memory_fragment(
                content=split_item,
                source=source_input,
                confidence_hint=confidence_hint,
                agent_id=target.agent_id,
            )
            await wm.store(fragment)
            created_memory_ids.append(fragment.id)

        await wm.delete(validated.target_id)
        split_details = {
            "split_into": split_payload.split_into,
            "created_memory_ids": created_memory_ids,
            "storage": "working_memory",
        }
        await _log_correct_memory_event(
            {
                "target_id": validated.target_id,
                "action": action_enum.value,
                "status": "applied",
                "created_memory_ids": created_memory_ids,
                "split_into": split_payload.split_into,
                "storage": "working_memory",
            }
        )
        return CorrectMemoryResult(
            target_id=validated.target_id,
            action=action_enum,
            status="applied",
            details=split_details,
        )

    if action_enum == CorrectionAction.contest:
        try:
            contest_payload = ContestPayload.model_validate(validated.payload or {})
        except ValidationError as exc:
            return _correct_rejected(
                validated.target_id,
                action=action_enum,
                error_code="validation_error",
                message=_validation_message(exc),
            )

        target = await wm.get(validated.target_id)
        if target is None:
            return CorrectMemoryResult(
                target_id=validated.target_id,
                action=action_enum,
                status="not_found",
            )

        old_confidence = target.confidence
        new_confidence = _downgrade_confidence(old_confidence)
        updated_properties = dict(target.properties)
        updated_properties["status"] = "contested"
        updated_properties["contest_reason"] = contest_payload.reason
        updated_properties["contested_at"] = time.time()
        updated = target.model_copy(
            update={
                "confidence": new_confidence,
                "properties": updated_properties,
                "timestamp": time.time(),
            }
        )
        await _replace_fragment(wm, updated)

        cascade = _contest_cascade_hook(validated.target_id)
        details = {
            "reason": contest_payload.reason,
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "cascade": cascade,
        }
        await _log_correct_memory_event(
            {
                "target_id": validated.target_id,
                "action": action_enum.value,
                "status": "applied",
                "reason": contest_payload.reason,
                "old_confidence": old_confidence,
                "new_confidence": new_confidence,
                "cascade": cascade,
            }
        )
        return CorrectMemoryResult(
            target_id=validated.target_id,
            action=action_enum,
            status="applied",
            details=details,
        )

    if action_enum == CorrectionAction.annotate:
        try:
            annotate_payload = AnnotatePayload.model_validate(validated.payload or {})
        except ValidationError as exc:
            return _correct_rejected(
                validated.target_id,
                action=action_enum,
                error_code="validation_error",
                message=_validation_message(exc),
            )

        target = await wm.get(validated.target_id)
        if target is None:
            return CorrectMemoryResult(
                target_id=validated.target_id,
                action=action_enum,
                status="not_found",
            )

        updated_properties = dict(target.properties)
        existing_annotations = updated_properties.get("annotations", [])
        if not isinstance(existing_annotations, list):
            existing_annotations = []
        existing_annotations.append(
            {
                "note": annotate_payload.note,
                "created_at": time.time(),
            }
        )
        updated_properties["annotations"] = existing_annotations
        updated = target.model_copy(
            update={
                "properties": updated_properties,
                "timestamp": time.time(),
            }
        )
        await _replace_fragment(wm, updated)

        details = {
            "note": annotate_payload.note,
            "annotation_count": len(existing_annotations),
        }
        await _log_correct_memory_event(
            {
                "target_id": validated.target_id,
                "action": action_enum.value,
                "status": "applied",
                "note": annotate_payload.note,
                "annotation_count": len(existing_annotations),
            }
        )
        return CorrectMemoryResult(
            target_id=validated.target_id,
            action=action_enum,
            status="applied",
            details=details,
        )

    if action_enum == CorrectionAction.merge_entities:
        try:
            merge_payload = MergeEntitiesPayload.model_validate(validated.payload or {})
        except ValidationError as exc:
            return _correct_rejected(
                validated.target_id,
                action=action_enum,
                error_code="validation_error",
                message=_validation_message(exc),
            )

        if merge_payload.merge_with == validated.target_id:
            return _correct_rejected(
                validated.target_id,
                action=action_enum,
                error_code="invalid_merge_target",
                message="merge_with must be different from target_id.",
            )

        graph_details = await _merge_entities_in_graph(
            validated.target_id,
            merge_payload.merge_with,
        )
        if graph_details is not None:
            await _log_correct_memory_event(
                {
                    "target_id": validated.target_id,
                    "action": action_enum.value,
                    "status": "applied",
                    **graph_details,
                }
            )
            return CorrectMemoryResult(
                target_id=validated.target_id,
                action=action_enum,
                status="applied",
                details=graph_details,
            )

        target = await wm.get(validated.target_id)
        if target is None:
            return CorrectMemoryResult(
                target_id=validated.target_id,
                action=action_enum,
                status="not_found",
            )
        merge_with = await wm.get(merge_payload.merge_with)
        if merge_with is None:
            return CorrectMemoryResult(
                target_id=validated.target_id,
                action=action_enum,
                status="not_found",
            )

        merged_content_parts = [target.content]
        if merge_with.content != target.content:
            merged_content_parts.append(merge_with.content)
        merged_properties = dict(target.properties)
        merged_properties["merged_from"] = merge_with.id
        merged_properties["merged_at"] = time.time()

        merged_sources = list(target.sources)
        existing_source_ids = {src.get("id") for src in merged_sources}
        for source in merge_with.sources:
            source_id = source.get("id")
            if source_id not in existing_source_ids:
                merged_sources.append(source)
                existing_source_ids.add(source_id)

        merged = target.model_copy(
            update={
                "content": " | ".join(merged_content_parts),
                "confidence": _best_confidence(
                    target.confidence, merge_with.confidence
                ),
                "sources": merged_sources,
                "properties": merged_properties,
                "timestamp": time.time(),
            }
        )
        await wm.delete(merge_with.id)
        await _replace_fragment(wm, merged)

        details = {
            "merged_into": validated.target_id,
            "merged_from": merge_with.id,
            "confidence": merged.confidence,
            "storage": "working_memory",
        }
        await _log_correct_memory_event(
            {
                "target_id": validated.target_id,
                "action": action_enum.value,
                "status": "applied",
                "merged_into": validated.target_id,
                "merged_from": merge_with.id,
                "confidence": merged.confidence,
            }
        )
        return CorrectMemoryResult(
            target_id=validated.target_id,
            action=action_enum,
            status="applied",
            details=details,
        )

    if action_enum == CorrectionAction.reclassify:
        try:
            reclass_payload = ReclassifyPayload.model_validate(validated.payload or {})
        except ValidationError as exc:
            return _correct_rejected(
                validated.target_id,
                action=action_enum,
                error_code="validation_error",
                message=_validation_message(exc),
            )

        graph_details = await _reclassify_in_graph(
            validated.target_id,
            reclass_payload.new_type,
        )
        if graph_details is not None:
            await _log_correct_memory_event(
                {
                    "target_id": validated.target_id,
                    "action": action_enum.value,
                    "status": "applied",
                    **graph_details,
                }
            )
            return CorrectMemoryResult(
                target_id=validated.target_id,
                action=action_enum,
                status="applied",
                details=graph_details,
            )

        target = await wm.get(validated.target_id)
        if target is None:
            return CorrectMemoryResult(
                target_id=validated.target_id,
                action=action_enum,
                status="not_found",
            )

        old_type = target.type
        updated_properties = dict(target.properties)
        history_raw = updated_properties.get("reclassify_history", [])
        if not isinstance(history_raw, list):
            history_raw = []
        history: list[dict[str, object]] = []
        for item in history_raw:
            normalized = _normalize_reclassify_history_entry(item)
            if normalized is not None:
                history.append(normalized)
        history.append(
            _reclassify_history_record(
                old_type,
                reclass_payload.new_type,
                time.time(),
            )
        )
        updated_properties["reclassify_history"] = history

        updated = target.model_copy(
            update={
                "type": reclass_payload.new_type,
                "properties": updated_properties,
                "timestamp": time.time(),
            }
        )
        await _replace_fragment(wm, updated)

        details = {
            "old_type": old_type,
            "new_type": reclass_payload.new_type,
            "storage": "working_memory",
        }
        await _log_correct_memory_event(
            {
                "target_id": validated.target_id,
                "action": action_enum.value,
                "status": "applied",
                "old_type": old_type,
                "new_type": reclass_payload.new_type,
            }
        )
        return CorrectMemoryResult(
            target_id=validated.target_id,
            action=action_enum,
            status="applied",
            details=details,
        )

    # Mock: apply correction (real logic in later sprints)
    fallback_details = {}
    if validated.payload:
        fallback_details = validated.payload

    return CorrectMemoryResult(
        target_id=validated.target_id,
        action=action_enum,
        status="applied",
        details=fallback_details,
    )
