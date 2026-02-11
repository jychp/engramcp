"""Neo4j-backed graph store — CRUD operations on the EngraMCP ontology.

The ``GraphStore`` operates directly on ``models.nodes`` and
``models.relations`` Pydantic types.  It uses label-based dispatch for
node creation and type-based dispatch for relationships.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum

from neo4j import AsyncDriver
from neo4j import time as neo4j_time

from engramcp.models.confidence import Reliability
from engramcp.models.nodes import LABEL_TO_MODEL
from engramcp.models.nodes import MemoryBase
from engramcp.models.nodes import MemoryNode
from engramcp.models.relations import Relationship
from engramcp.models.relations import RelationshipBase
from engramcp.models.relations import ResolutionStatus

# ---------------------------------------------------------------------------
# Query safety guards
# ---------------------------------------------------------------------------

_ALLOWED_NODE_LABELS = {label for labels in LABEL_TO_MODEL for label in labels}
_ALLOWED_NODE_LABELS.add("Memory")
_ALLOWED_REL_TYPES = {
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
_ALLOWED_DIRECTIONS = {"outgoing", "incoming", "both"}
_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_RETRIEVAL_MEMORY_LABELS = (
    "Fact|Event|Observation|Decision|Outcome|Pattern|Concept|Rule"
)


def _require_allowed(value: str, allowed: set[str], *, field_name: str) -> str:
    if value not in allowed:
        msg = f"Invalid {field_name}: {value!r}"
        raise ValueError(msg)
    return value


def _tokenize_query(query: str) -> list[str]:
    """Normalize a free-text query into lowercase search tokens."""
    tokens = [token.casefold() for token in _QUERY_TOKEN_RE.findall(query)]
    filtered = [token for token in tokens if len(token) >= 3]
    if filtered:
        return list(dict.fromkeys(filtered))
    if tokens:
        return list(dict.fromkeys(tokens))
    return []


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _neo4j_to_python(value: object) -> object:
    """Convert Neo4j temporal types to Python stdlib equivalents."""
    if isinstance(value, neo4j_time.DateTime):
        return value.to_native()
    if isinstance(value, neo4j_time.Date):
        return value.to_native()
    if isinstance(value, neo4j_time.Duration):
        return value.hours_minutes_seconds_nanoseconds
    return value


def _convert_props(props: dict) -> dict:
    """Convert all Neo4j types in a property dict to Python types."""
    return {k: _neo4j_to_python(v) for k, v in props.items()}


def _serialize_props(model: MemoryBase | RelationshipBase) -> dict:
    """Convert a Pydantic model to a Neo4j property dict.

    - Drops ``None`` values (Neo4j doesn't store nulls).
    - Converts enums to their ``.value``.
    - Passes ``datetime`` objects through (the Neo4j driver handles them).
    """
    data = model.model_dump(exclude_none=True)
    result: dict = {}
    for key, value in data.items():
        if isinstance(value, Enum):
            result[key] = value.value
        elif isinstance(value, list):
            result[key] = [v.value if isinstance(v, Enum) else v for v in value]
        else:
            result[key] = value
    return result


def _deserialize_node(props: dict, labels: frozenset[str]) -> MemoryNode:
    """Reconstruct a Pydantic node model from Neo4j properties and labels."""
    model_cls = LABEL_TO_MODEL.get(labels)
    if model_cls is None:
        msg = f"Unknown label combination: {labels}"
        raise ValueError(msg)
    return model_cls.model_validate(_convert_props(props))


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------


class GraphStore:
    """Async CRUD interface to the Neo4j knowledge graph."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    # ----- Node CRUD -----

    async def create_node(self, node: MemoryNode) -> str:
        """Create a node in the graph and return its ID.

        Labels are derived from ``node.node_labels``.
        """
        labels_tuple = tuple(
            _require_allowed(label, _ALLOWED_NODE_LABELS, field_name="label")
            for label in node.node_labels
        )
        labels = ":".join(labels_tuple)
        props = _serialize_props(node)
        query = f"CREATE (n:{labels} $props) RETURN n.id AS id"
        async with self._driver.session() as session:
            result = await session.run(query, props=props)
            record = await result.single()
            return record["id"]

    async def get_node(self, node_id: str) -> MemoryNode | None:
        """Retrieve a node by its ID, or ``None`` if not found."""
        query = "MATCH (n:Memory {id: $id}) RETURN properties(n) AS props, labels(n) AS labels"
        async with self._driver.session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            if record is None:
                return None
            return _deserialize_node(
                record["props"],
                frozenset(record["labels"]),
            )

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its relationships. Return ``True`` if found."""
        query = "MATCH (n:Memory {id: $id}) DETACH DELETE n RETURN count(n) AS cnt"
        async with self._driver.session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            return record["cnt"] > 0

    async def update_node(self, node_id: str, **updates: object) -> MemoryNode | None:
        """Update properties on an existing node. Return the updated node."""
        # Serialize enum values in updates
        serialized: dict = {}
        for k, v in updates.items():
            serialized[k] = v.value if isinstance(v, Enum) else v
        query = "MATCH (n:Memory {id: $id}) SET n += $props RETURN properties(n) AS props, labels(n) AS labels"
        async with self._driver.session() as session:
            result = await session.run(query, id=node_id, props=serialized)
            record = await result.single()
            if record is None:
                return None
            return _deserialize_node(
                record["props"],
                frozenset(record["labels"]),
            )

    # ----- Relationship CRUD -----

    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel: Relationship,
    ) -> bool:
        """Create a relationship between two nodes. Return ``True`` if both nodes exist."""
        rel_type = _require_allowed(
            rel.rel_type, _ALLOWED_REL_TYPES, field_name="relationship type"
        )
        props = _serialize_props(rel)
        query = (
            f"MATCH (a:Memory {{id: $from_id}}), (b:Memory {{id: $to_id}}) "
            f"CREATE (a)-[r:{rel_type} $props]->(b) "
            f"RETURN type(r) AS t"
        )
        async with self._driver.session() as session:
            result = await session.run(query, from_id=from_id, to_id=to_id, props=props)
            record = await result.single()
            return record is not None

    async def get_relationships(
        self,
        node_id: str,
        rel_type: str | None = None,
        direction: str = "both",
    ) -> list[dict]:
        """Return relationships attached to a node.

        Each result is a dict with ``type``, ``props``, ``from_id``, ``to_id``.
        """
        if rel_type:
            _require_allowed(
                rel_type, _ALLOWED_REL_TYPES, field_name="relationship type"
            )
            rel_match = f"[r:{rel_type}]"
        else:
            rel_match = "[r]"

        direction = _require_allowed(
            direction, _ALLOWED_DIRECTIONS, field_name="direction"
        )

        if direction == "outgoing":
            pattern = f"(n:Memory {{id: $id}})-{rel_match}->(b)"
        elif direction == "incoming":
            pattern = f"(b)-{rel_match}->(n:Memory {{id: $id}})"
        else:
            pattern = f"(n:Memory {{id: $id}})-{rel_match}-(b)"

        query = (
            f"MATCH {pattern} "
            f"RETURN type(r) AS type, properties(r) AS props, "
            f"startNode(r).id AS from_id, endNode(r).id AS to_id"
        )
        async with self._driver.session() as session:
            result = await session.run(query, id=node_id)
            records = [record.data() async for record in result]
            return records

    # ----- Query methods -----

    async def find_by_id(self, node_id: str) -> MemoryNode | None:
        """Alias for ``get_node``."""
        return await self.get_node(node_id)

    async def find_by_label(self, label: str) -> list[MemoryNode]:
        """Find all nodes with a specific label (e.g. ``'Agent'``, ``'Artifact'``)."""
        _require_allowed(label, _ALLOWED_NODE_LABELS, field_name="label")
        query = (
            f"MATCH (n:{label}) " "RETURN properties(n) AS props, labels(n) AS labels"
        )
        return await self._run_multi_node_query(query)

    async def find_facts_by_agent(self, agent_name: str) -> list[MemoryNode]:
        """Find all Fact nodes linked to an Agent via CONCERNS."""
        query = (
            "MATCH (f:Fact)-[:CONCERNS]->(a:Agent {name: $name}) "
            "RETURN properties(f) AS props, labels(f) AS labels"
        )
        return await self._run_multi_node_query(query, name=agent_name)

    async def find_events_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> list[MemoryNode]:
        """Find all Event nodes within a time range."""
        query = (
            "MATCH (e:Event) "
            "WHERE e.occurred_at >= $start AND e.occurred_at <= $end "
            "RETURN properties(e) AS props, labels(e) AS labels "
            "ORDER BY e.occurred_at"
        )
        return await self._run_multi_node_query(query, start=start, end=end)

    async def find_sources_by_reliability(
        self,
        reliability: Reliability,
    ) -> list[MemoryNode]:
        """Find all Source nodes with a given reliability letter."""
        query = (
            "MATCH (s:Source {reliability: $r}) "
            "RETURN properties(s) AS props, labels(s) AS labels"
        )
        return await self._run_multi_node_query(query, r=reliability.value)

    async def find_claim_nodes(self) -> list[MemoryNode]:
        """Return all retrieval-eligible memory nodes."""
        query = (
            f"MATCH (n:{_RETRIEVAL_MEMORY_LABELS}) "
            "RETURN properties(n) AS props, labels(n) AS labels"
        )
        return await self._run_multi_node_query(query)

    async def find_claim_nodes_by_content(
        self, query: str, *, limit: int = 20
    ) -> list[MemoryNode]:
        """Find retrieval-eligible nodes whose ``content`` matches query tokens."""
        tokens = _tokenize_query(query)
        if not tokens:
            return []
        cypher = (
            f"MATCH (n:{_RETRIEVAL_MEMORY_LABELS}) "
            "WHERE ANY(token IN $search_tokens WHERE toLower(n.content) CONTAINS token) "
            "RETURN properties(n) AS props, labels(n) AS labels "
            "ORDER BY n.updated_at DESC "
            "LIMIT $limit"
        )
        return await self._run_multi_node_query(
            cypher,
            search_tokens=tokens,
            limit=limit,
        )

    async def find_claim_context_by_content(
        self,
        query: str,
        *,
        limit: int = 20,
        max_depth: int = 3,
        include_sources: bool = True,
        include_contradictions: bool = True,
    ) -> list[dict]:
        """Find claims and enrich each hit with bounded graph context."""
        if max_depth < 1:
            max_depth = 1
        if max_depth > 10:
            max_depth = 10
        tokens = _tokenize_query(query)
        if not tokens:
            return []

        node_query = (
            f"MATCH (n:{_RETRIEVAL_MEMORY_LABELS}) "
            "WHERE ANY(token IN $search_tokens WHERE toLower(n.content) CONTAINS token) "
            "RETURN properties(n) AS props, labels(n) AS labels "
            "ORDER BY n.updated_at DESC "
            "LIMIT $limit"
        )
        async with self._driver.session() as session:
            node_result = await session.run(
                node_query,
                search_tokens=tokens,
                limit=limit,
            )
            nodes = [record.data() async for record in node_result]
            node_ids = [
                str(_convert_props(record["props"]).get("id"))
                for record in nodes
                if _convert_props(record["props"]).get("id")
            ]
            if not node_ids:
                return []

            causal_map = await self._fetch_causal_chains_for_nodes(
                session=session,
                node_ids=node_ids,
                max_depth=max_depth,
            )
            sources_map: dict[str, list[dict]] = {}
            if include_sources:
                sources_map = await self._fetch_source_trails_for_nodes(
                    session=session, node_ids=node_ids
                )
            contradictions_map: dict[str, list[dict]] = {}
            if include_contradictions:
                contradictions_map = await self._fetch_contradictions_for_nodes(
                    session=session, node_ids=node_ids
                )

            contexts: list[dict] = []
            for node_record in nodes:
                props = _convert_props(node_record["props"])
                node_id = props.get("id")
                if not node_id:
                    continue
                node_id_str = str(node_id)
                contexts.append(
                    {
                        "node": {
                            "id": node_id_str,
                            "content": props.get("content", ""),
                            "labels": list(node_record["labels"]),
                            "confidence": props.get("confidence"),
                            "properties": props,
                        },
                        "causal_chain": causal_map.get(node_id_str, []),
                        "sources": sources_map.get(node_id_str, []),
                        "contradictions": contradictions_map.get(node_id_str, []),
                    }
                )
            return contexts

    async def find_contradictions_unresolved(self) -> list[dict]:
        """Find all unresolved CONTRADICTS relationships.

        Returns dicts with ``from_node``, ``to_node``, and ``rel_props``.
        """
        query = (
            "MATCH (a:Memory)-[c:CONTRADICTS {resolution_status: $status}]->(b:Memory) "
            "RETURN properties(a) AS a_props, labels(a) AS a_labels, "
            "properties(b) AS b_props, labels(b) AS b_labels, "
            "properties(c) AS rel_props"
        )
        async with self._driver.session() as session:
            result = await session.run(query, status=ResolutionStatus.unresolved.value)
            results = []
            async for record in result:
                results.append(
                    {
                        "from_node": _deserialize_node(
                            record["a_props"], frozenset(record["a_labels"])
                        ),
                        "to_node": _deserialize_node(
                            record["b_props"], frozenset(record["b_labels"])
                        ),
                        "rel_props": record["rel_props"],
                    }
                )
            return results

    async def find_agent_by_alias(self, alias: str) -> list[MemoryNode]:
        """Find Agent nodes matching a name or alias."""
        query = (
            "MATCH (a:Agent) "
            "WHERE a.name = $alias OR ANY(x IN a.aliases WHERE x = $alias) "
            "RETURN properties(a) AS props, labels(a) AS labels"
        )
        return await self._run_multi_node_query(query, alias=alias)

    async def find_possibly_same_as_unresolved(self) -> list[dict]:
        """Find all POSSIBLY_SAME_AS relationships.

        Returns dicts with ``from_node``, ``to_node``, ``similarity_score``.
        """
        query = (
            "MATCH (a:Memory)-[r:POSSIBLY_SAME_AS]->(b:Memory) "
            "RETURN properties(a) AS a_props, labels(a) AS a_labels, "
            "properties(b) AS b_props, labels(b) AS b_labels, "
            "properties(r) AS rel_props"
        )
        async with self._driver.session() as session:
            result = await session.run(query)
            results = []
            async for record in result:
                results.append(
                    {
                        "from_node": _deserialize_node(
                            record["a_props"], frozenset(record["a_labels"])
                        ),
                        "to_node": _deserialize_node(
                            record["b_props"], frozenset(record["b_labels"])
                        ),
                        "similarity_score": record["rel_props"]["similarity_score"],
                    }
                )
            return results

    # ----- Internal helpers -----

    async def _run_multi_node_query(
        self, query: str, **params: object
    ) -> list[MemoryNode]:
        """Run a Cypher query that returns ``props`` and ``labels`` columns."""
        async with self._driver.session() as session:
            result = await session.run(query, **params)
            nodes = []
            async for record in result:
                node = _deserialize_node(
                    record["props"],
                    frozenset(record["labels"]),
                )
                nodes.append(node)
            return nodes

    async def _fetch_causal_chains_for_nodes(
        self,
        *,
        session,
        node_ids: list[str],
        max_depth: int,
    ) -> dict[str, list[dict]]:
        rel_types = ["CAUSED_BY", "LEADS_TO", "PRECEDED", "FOLLOWED"]
        query = (
            "MATCH (n:Memory) "
            "WHERE n.id IN $node_ids "
            "OPTIONAL MATCH p = (n)-[rels*1.."
            f"{max_depth}"
            "]-(m:Memory) "
            "WHERE ALL(rel IN rels WHERE type(rel) IN $rel_types) "
            "WITH n.id AS node_id, m, p "
            "ORDER BY node_id, length(p) ASC "
            "WITH node_id, collect(DISTINCT CASE "
            "  WHEN m IS NULL OR p IS NULL THEN NULL "
            "  ELSE {"
            "    relation: type(last(relationships(p))), "
            "    target_id: m.id, "
            "    target_summary: coalesce(m.content, ''), "
            "    confidence: coalesce(last(relationships(p)).credibility, "
            "                         last(relationships(p)).confidence, null)"
            "  } "
            "END) AS raw_links "
            "RETURN node_id, [item IN raw_links WHERE item IS NOT NULL][0..20] AS links"
        )
        result = await session.run(query, node_ids=node_ids, rel_types=rel_types)
        by_node: dict[str, list[dict]] = {}
        async for record in result:
            by_node[str(record["node_id"])] = list(record["links"] or [])
        return by_node

    async def _fetch_source_trails_for_nodes(
        self, *, session, node_ids: list[str]
    ) -> dict[str, list[dict]]:
        query = (
            "MATCH (n:Memory) "
            "WHERE n.id IN $node_ids "
            "OPTIONAL MATCH (n)-[r:SOURCED_FROM]->(s:Source) "
            "WITH n.id AS node_id, s, r "
            "ORDER BY node_id, s.ingested_at DESC "
            "WITH node_id, collect(CASE "
            "  WHEN s IS NULL THEN NULL "
            "  ELSE {"
            "    id: s.id, "
            "    type: coalesce(s.type, 'unknown'), "
            "    ref: s.ref, "
            "    citation: s.citation, "
            "    reliability: s.reliability, "
            "    credibility: toString(r.credibility)"
            "  } "
            "END) AS raw_sources "
            "RETURN node_id, [item IN raw_sources WHERE item IS NOT NULL][0..20] AS sources"
        )
        result = await session.run(query, node_ids=node_ids)
        by_node: dict[str, list[dict]] = {}
        async for record in result:
            by_node[str(record["node_id"])] = list(record["sources"] or [])
        return by_node

    async def _fetch_contradictions_for_nodes(
        self, *, session, node_ids: list[str]
    ) -> dict[str, list[dict]]:
        query = (
            "MATCH (n:Memory) "
            "WHERE n.id IN $node_ids "
            "OPTIONAL MATCH (n)-[r:CONTRADICTS]-(m:Memory) "
            "WHERE r.resolution_status = $status "
            "WITH n.id AS node_id, r, m, labels(m) AS mem_labels "
            "ORDER BY node_id, r.detected_at DESC "
            "WITH node_id, collect(CASE "
            "  WHEN r IS NULL OR m IS NULL THEN NULL "
            "  ELSE {"
            "    id: coalesce(r.id, ''), "
            "    memory_id: node_id, "
            "    nature: 'factual_conflict', "
            "    resolution_status: coalesce(r.resolution_status, $status), "
            "    detected_at: toString(r.detected_at), "
            "    memory: {"
            "      id: coalesce(m.id, ''), "
            "      content: coalesce(m.content, ''), "
            "      labels: mem_labels, "
            "      confidence: m.confidence, "
            "      properties: properties(m)"
            "    }"
            "  } "
            "END) AS raw_contradictions "
            "RETURN node_id, [item IN raw_contradictions WHERE item IS NOT NULL][0..20] AS contradictions"
        )
        result = await session.run(
            query,
            node_ids=node_ids,
            status=ResolutionStatus.unresolved.value,
        )
        by_node: dict[str, list[dict]] = {}
        async for record in result:
            node_id = str(record["node_id"])
            raw_items: list[dict] = list(record["contradictions"] or [])
            normalized: list[dict] = []
            for idx, item in enumerate(raw_items, start=1):
                contradiction_id = item.get("id") or f"{node_id}:contra:{idx}"
                contradictory = item.get("memory", {}) or {}
                normalized.append(
                    {
                        "id": contradiction_id,
                        "memory_id": node_id,
                        "nature": item.get("nature", "factual_conflict"),
                        "resolution_status": item.get(
                            "resolution_status", ResolutionStatus.unresolved.value
                        ),
                        "detected_at": item.get("detected_at"),
                        "memory": {
                            "id": contradictory.get("id", ""),
                            "content": contradictory.get("content", ""),
                            "labels": contradictory.get("labels", []),
                            "confidence": contradictory.get("confidence"),
                            "properties": _convert_props(
                                contradictory.get("properties", {}) or {}
                            ),
                        },
                    }
                )
            by_node[node_id] = normalized
        return by_node
