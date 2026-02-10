"""Graph-backed correction operations used by the MCP application layer."""

from __future__ import annotations

import json
import time

from neo4j import AsyncDriver

from engramcp.engine.confidence import ConfidenceEngine
from engramcp.graph.entity_resolution import MergeExecutor
from engramcp.graph.store import GraphStore
from engramcp.graph.traceability import SourceTraceability
from engramcp.models.nodes import Agent
from engramcp.models.nodes import MemoryNode

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


def _normalize_reclassify_history_entry(entry: object) -> dict[str, object] | None:
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
        try:
            decoded = json.loads(entry)
            if isinstance(decoded, dict):
                return _normalize_reclassify_history_entry(decoded)
        except json.JSONDecodeError:
            pass
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


async def split_entity_in_graph(
    driver: AsyncDriver, target_id: str, split_into: list[str]
) -> dict | None:
    """Split a graph node into multiple nodes and duplicate its relationships."""
    graph_store = GraphStore(driver)
    target_node = await graph_store.get_node(target_id)
    if target_node is None:
        return None

    created_memory_ids: list[str] = []
    for split_value in split_into:
        new_node = _build_split_graph_node(target_node, split_value)
        created_memory_ids.append(await graph_store.create_node(new_node))

    outgoing: list[dict] = []
    incoming: list[dict] = []
    async with driver.session() as session:
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


async def merge_entities_in_graph(
    driver: AsyncDriver, target_id: str, merge_with_id: str
) -> dict | None:
    """Merge entities in graph storage when both nodes exist."""
    graph_store = GraphStore(driver)
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


async def reclassify_in_graph(
    driver: AsyncDriver, target_id: str, new_type: str
) -> dict | None:
    """Reclassify a graph node through lifecycle updates."""
    async with driver.session() as session:
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
        confidence_engine = ConfidenceEngine(
            GraphStore(driver),
            SourceTraceability(driver),
        )
        cascade = await confidence_engine.cascade_contest(target_id)
        details["lifecycle"]["cascade"] = {
            "affected_nodes": cascade.affected_nodes,
            "reason": cascade.reason,
        }

    return details
