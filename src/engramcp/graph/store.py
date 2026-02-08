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
        """Return all claim nodes (Fact/Event/Observation/Decision/Outcome)."""
        query = (
            "MATCH (n:Fact|Event|Observation|Decision|Outcome) "
            "RETURN properties(n) AS props, labels(n) AS labels"
        )
        return await self._run_multi_node_query(query)

    async def find_claim_nodes_by_content(
        self, query: str, *, limit: int = 20
    ) -> list[MemoryNode]:
        """Find claim nodes whose textual ``content`` matches a query substring."""
        tokens = _tokenize_query(query)
        if not tokens:
            return []
        cypher = (
            "MATCH (n:Fact|Event|Observation|Decision|Outcome) "
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
            "MATCH (n:Fact|Event|Observation|Decision|Outcome) "
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
            contexts: list[dict] = []
            for node_record in nodes:
                props = _convert_props(node_record["props"])
                node_id = props.get("id")
                if not node_id:
                    continue

                context = {
                    "node": {
                        "id": props["id"],
                        "content": props.get("content", ""),
                        "labels": list(node_record["labels"]),
                        "confidence": props.get("confidence"),
                        "properties": props,
                    },
                    "causal_chain": await self._fetch_causal_chain(
                        session=session, node_id=node_id, max_depth=max_depth
                    ),
                    "sources": [],
                    "contradictions": [],
                }

                if include_sources:
                    context["sources"] = await self._fetch_source_trail(
                        session=session, node_id=node_id
                    )

                if include_contradictions:
                    context["contradictions"] = await self._fetch_contradictions(
                        session=session, node_id=node_id
                    )

                contexts.append(context)
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

    async def _fetch_causal_chain(
        self,
        *,
        session,
        node_id: str,
        max_depth: int,
    ) -> list[dict]:
        rel_types = ["CAUSED_BY", "LEADS_TO", "PRECEDED", "FOLLOWED"]
        query = (
            "MATCH p = (n:Memory {id: $node_id})-[rels*1.."
            f"{max_depth}"
            "]-(m:Memory) "
            "WHERE ALL(rel IN rels WHERE type(rel) IN $rel_types) "
            "WITH m, p "
            "ORDER BY length(p) ASC "
            "WITH m, last(relationships(p)) AS terminal_rel "
            "RETURN DISTINCT "
            "type(terminal_rel) AS relation, "
            "m.id AS target_id, "
            "coalesce(m.content, '') AS target_summary, "
            "coalesce(terminal_rel.credibility, terminal_rel.confidence, null) AS confidence "
            "LIMIT 20"
        )
        result = await session.run(query, node_id=node_id, rel_types=rel_types)
        links: list[dict] = []
        async for record in result:
            links.append(
                {
                    "relation": record["relation"],
                    "target_id": record["target_id"],
                    "target_summary": record["target_summary"],
                    "confidence": record["confidence"],
                }
            )
        return links

    async def _fetch_source_trail(self, *, session, node_id: str) -> list[dict]:
        query = (
            "MATCH (n:Memory {id: $node_id})-[r:SOURCED_FROM]->(s:Source) "
            "RETURN properties(s) AS source_props, properties(r) AS rel_props "
            "ORDER BY s.ingested_at DESC "
            "LIMIT 20"
        )
        result = await session.run(query, node_id=node_id)
        sources: list[dict] = []
        async for record in result:
            source_props = _convert_props(record["source_props"])
            rel_props = _convert_props(record["rel_props"])
            credibility_raw = rel_props.get("credibility")
            if credibility_raw is not None:
                credibility_raw = str(credibility_raw)
            sources.append(
                {
                    "id": source_props.get("id"),
                    "type": source_props.get("type", "unknown"),
                    "ref": source_props.get("ref"),
                    "citation": source_props.get("citation"),
                    "reliability": source_props.get("reliability"),
                    "credibility": credibility_raw,
                }
            )
        return sources

    async def _fetch_contradictions(self, *, session, node_id: str) -> list[dict]:
        query = (
            "MATCH (n:Memory {id: $node_id})-[r:CONTRADICTS]-(m:Memory) "
            "WHERE r.resolution_status = $status "
            "RETURN properties(r) AS rel_props, properties(m) AS mem_props, labels(m) AS mem_labels "
            "ORDER BY r.detected_at DESC "
            "LIMIT 20"
        )
        result = await session.run(
            query,
            node_id=node_id,
            status=ResolutionStatus.unresolved.value,
        )
        contradictions: list[dict] = []
        index = 0
        async for record in result:
            index += 1
            rel_props = _convert_props(record["rel_props"])
            mem_props = _convert_props(record["mem_props"])
            contradiction_id = rel_props.get("id")
            if not contradiction_id:
                contradiction_id = f"{node_id}:contra:{index}"
            contradictions.append(
                {
                    "id": contradiction_id,
                    "memory_id": node_id,
                    "nature": "factual_conflict",
                    "resolution_status": rel_props.get(
                        "resolution_status", ResolutionStatus.unresolved.value
                    ),
                    "detected_at": (
                        rel_props["detected_at"].isoformat()
                        if rel_props.get("detected_at") is not None
                        else None
                    ),
                    "memory": {
                        "id": mem_props.get("id", ""),
                        "content": mem_props.get("content", ""),
                        "labels": list(record["mem_labels"]),
                        "confidence": mem_props.get("confidence"),
                        "properties": mem_props,
                    },
                }
            )
        return contradictions
