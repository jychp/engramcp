"""Source chain traversal and independence detection.

Uses variable-length Cypher paths (``[:CITES*0..N]``) to walk citation
chains — this is why ``SourceTraceability`` takes an ``AsyncDriver``
directly instead of going through ``GraphStore``.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import AsyncDriver

from engramcp.graph.store import _convert_props
from engramcp.models.nodes import Source

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceChain:
    """An ordered chain from an immediate source to its root."""

    sources: list[Source]
    root: Source


@dataclass(frozen=True)
class IndependenceResult:
    """Result of checking whether two sources are independent."""

    independent: bool
    common_ancestor: str | None
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_to_source(props: dict) -> Source:
    """Convert a Neo4j node property dict to a ``Source`` model."""
    return Source.model_validate(_convert_props(props))


def _validate_max_depth(max_depth: int) -> int:
    if max_depth < 1 or max_depth > 50:
        msg = f"max_depth must be between 1 and 50, got {max_depth}"
        raise ValueError(msg)
    return max_depth


# ---------------------------------------------------------------------------
# SourceTraceability
# ---------------------------------------------------------------------------


class SourceTraceability:
    """Source chain traversal and independence detection."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    async def get_citation_chain(
        self, source_id: str, *, max_depth: int = 10
    ) -> list[Source]:
        """Follow the ``CITES`` chain from *source_id* to the root.

        Returns an ordered list from the starting source to the root
        (the terminal source with no outgoing ``CITES``).
        """
        depth = _validate_max_depth(max_depth)
        query = (
            f"MATCH p = (s:Source {{id: $id}})-[:CITES*0..{depth}]->(root:Source) "
            "WHERE NOT (root)-[:CITES]->(:Source) "
            "RETURN [n IN nodes(p) | properties(n)] AS chain "
            "ORDER BY length(p) DESC "
            "LIMIT 1"
        )
        async with self._driver.session() as session:
            result = await session.run(query, id=source_id)
            record = await result.single()
            if record is None:
                return []
            return [_node_to_source(props) for props in record["chain"]]

    async def find_root_source(self, source_id: str) -> Source | None:
        """Return the terminal source in the citation chain."""
        chain = await self.get_citation_chain(source_id)
        if not chain:
            return None
        return chain[-1]

    async def check_independence(
        self,
        source_a_id: str,
        source_b_id: str,
        *,
        max_depth: int = 10,
    ) -> IndependenceResult:
        """Check whether two sources are independent.

        Two sources are **not** independent if they share any common
        ancestor in their ``CITES`` chains (including each other).

        Conservative: if either source is not found, returns not independent.
        """
        depth = _validate_max_depth(max_depth)
        # First verify both sources exist
        exists_query = (
            "OPTIONAL MATCH (a:Source {id: $a_id}) "
            "OPTIONAL MATCH (b:Source {id: $b_id}) "
            "RETURN a IS NOT NULL AS a_exists, b IS NOT NULL AS b_exists"
        )
        async with self._driver.session() as session:
            result = await session.run(exists_query, a_id=source_a_id, b_id=source_b_id)
            record = await result.single()
            if not record["a_exists"] or not record["b_exists"]:
                return IndependenceResult(
                    independent=False,
                    common_ancestor=None,
                    reason="One or both sources not found — conservative default",
                )

        # Check for common ancestors (including direct citation)
        ancestor_query = (
            f"MATCH (a:Source {{id: $a_id}})-[:CITES*0..{depth}]->(ancestor:Source),"
            f"      (b:Source {{id: $b_id}})-[:CITES*0..{depth}]->(ancestor) "
            "RETURN ancestor.id AS common_ancestor "
            "LIMIT 1"
        )
        async with self._driver.session() as session:
            result = await session.run(
                ancestor_query, a_id=source_a_id, b_id=source_b_id
            )
            record = await result.single()
            if record is not None:
                return IndependenceResult(
                    independent=False,
                    common_ancestor=record["common_ancestor"],
                    reason=f"Common ancestor: {record['common_ancestor']}",
                )

        return IndependenceResult(
            independent=True,
            common_ancestor=None,
            reason="No common ancestor found in citation chains",
        )

    async def trace_fact_to_sources(self, fact_id: str) -> list[Source]:
        """Return all ``Source`` nodes linked to *fact_id* via ``SOURCED_FROM``."""
        query = (
            "MATCH (f:Memory {id: $id})-[:SOURCED_FROM]->(s:Source) "
            "RETURN properties(s) AS props"
        )
        async with self._driver.session() as session:
            result = await session.run(query, id=fact_id)
            return [_node_to_source(record["props"]) async for record in result]

    async def trace_fact_to_root_sources(
        self, fact_id: str, *, max_depth: int = 10
    ) -> list[SourceChain]:
        """Trace a fact through its sources to root sources.

        ``Fact → SOURCED_FROM → Source → CITES* → root``

        Returns one ``SourceChain`` per immediate source.
        """
        sources = await self.trace_fact_to_sources(fact_id)
        chains: list[SourceChain] = []
        for src in sources:
            chain = await self.get_citation_chain(src.id, max_depth=max_depth)
            if chain:
                chains.append(SourceChain(sources=chain, root=chain[-1]))
        return chains

    async def find_dependent_pairs(
        self, source_ids: list[str], *, max_depth: int = 10
    ) -> list[tuple[str, str]]:
        """Return source ID pairs that share at least one citation ancestor."""
        depth = _validate_max_depth(max_depth)
        if len(source_ids) < 2:
            return []

        query = (
            "MATCH (s:Source)-[:CITES*0.."
            f"{depth}"
            "]->(ancestor:Source) "
            "WHERE s.id IN $source_ids "
            "WITH ancestor, collect(DISTINCT s.id) AS ids "
            "WHERE size(ids) > 1 "
            "UNWIND ids AS a_id "
            "UNWIND ids AS b_id "
            "WITH a_id, b_id WHERE a_id < b_id "
            "RETURN DISTINCT a_id, b_id"
        )
        async with self._driver.session() as session:
            result = await session.run(query, source_ids=source_ids)
            return [(record["a_id"], record["b_id"]) async for record in result]
