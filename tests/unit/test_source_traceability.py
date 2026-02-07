"""Tests for source traceability (Layer 3 — graph/traceability.py).

All tests require a Neo4j testcontainer (session-scoped via conftest).
"""

from __future__ import annotations

import pytest

from engramcp.models import (
    Cites,
    Credibility,
    Fact,
    Reliability,
    Source,
    SourcedFrom,
)


# ===================================================================
# TestSourceCreation
# ===================================================================


class TestSourceCreation:
    async def test_source_with_all_fields(self, graph_store):
        src = Source(
            type="court_document",
            ref="https://example.com/doc.pdf",
            citation="page 42, paragraph 3",
            agent_id="agent-001",
            reliability=Reliability.A,
        )
        await graph_store.create_node(src)
        retrieved = await graph_store.get_node(src.id)
        assert isinstance(retrieved, Source)
        assert retrieved.type == "court_document"
        assert retrieved.ref == "https://example.com/doc.pdf"
        assert retrieved.citation == "page 42, paragraph 3"
        assert retrieved.agent_id == "agent-001"
        assert retrieved.reliability == Reliability.A

    async def test_source_with_minimal_fields(self, graph_store):
        src = Source(type="testimony", reliability=Reliability.C)
        await graph_store.create_node(src)
        retrieved = await graph_store.get_node(src.id)
        assert isinstance(retrieved, Source)
        assert retrieved.ref is None
        assert retrieved.citation is None
        assert retrieved.agent_id is None

    async def test_source_reliability_validation(self):
        with pytest.raises(ValueError):
            Source(type="testimony", reliability="Z")


# ===================================================================
# TestSourceChains
# ===================================================================


class TestSourceChains:
    async def test_cites_relationship(self, graph_store):
        """Source A cites Source B — the CITES relationship is created."""
        src_a = Source(type="news_article", reliability=Reliability.C)
        src_b = Source(type="court_document", reliability=Reliability.A)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_relationship(src_a.id, src_b.id, Cites())

        rels = await graph_store.get_relationships(
            src_a.id, rel_type="CITES", direction="outgoing"
        )
        assert len(rels) == 1
        assert rels[0]["to_id"] == src_b.id

    async def test_chain_depth_traversal(self, graph_store, traceability):
        """A -> B -> C chain is traversed in order."""
        src_a = Source(type="article", reliability=Reliability.C)
        src_b = Source(type="report", reliability=Reliability.B)
        src_c = Source(type="original", reliability=Reliability.A)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_node(src_c)
        await graph_store.create_relationship(src_a.id, src_b.id, Cites())
        await graph_store.create_relationship(src_b.id, src_c.id, Cites())

        chain = await traceability.get_citation_chain(src_a.id)
        assert len(chain) == 3
        assert chain[0].id == src_a.id
        assert chain[1].id == src_b.id
        assert chain[2].id == src_c.id

    async def test_find_root_source(self, graph_store, traceability):
        """Root source is the terminal node with no outgoing CITES."""
        src_a = Source(type="article", reliability=Reliability.C)
        src_b = Source(type="report", reliability=Reliability.B)
        src_c = Source(type="original", reliability=Reliability.A)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_node(src_c)
        await graph_store.create_relationship(src_a.id, src_b.id, Cites())
        await graph_store.create_relationship(src_b.id, src_c.id, Cites())

        root = await traceability.find_root_source(src_a.id)
        assert root.id == src_c.id


# ===================================================================
# TestIndependence
# ===================================================================


class TestIndependence:
    async def test_independent_sources_detected(self, graph_store, traceability):
        """Two sources with no shared ancestor are independent."""
        src_a = Source(type="testimony", reliability=Reliability.B)
        src_b = Source(type="document", reliability=Reliability.A)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)

        result = await traceability.check_independence(src_a.id, src_b.id)
        assert result.independent is True
        assert result.common_ancestor is None

    async def test_common_ancestor_detected(self, graph_store, traceability):
        """Two sources citing the same root share a common ancestor."""
        root = Source(type="original", reliability=Reliability.A)
        src_a = Source(type="article_a", reliability=Reliability.C)
        src_b = Source(type="article_b", reliability=Reliability.C)
        await graph_store.create_node(root)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_relationship(src_a.id, root.id, Cites())
        await graph_store.create_relationship(src_b.id, root.id, Cites())

        result = await traceability.check_independence(src_a.id, src_b.id)
        assert result.independent is False
        assert result.common_ancestor == root.id

    async def test_transitive_citation_detected(self, graph_store, traceability):
        """A cites B — they are not independent (B is ancestor of A)."""
        src_a = Source(type="article", reliability=Reliability.C)
        src_b = Source(type="original", reliability=Reliability.A)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_relationship(src_a.id, src_b.id, Cites())

        result = await traceability.check_independence(src_a.id, src_b.id)
        assert result.independent is False

    async def test_conservative_default_when_unknown(self, graph_store, traceability):
        """When a source doesn't exist, return not independent (conservative)."""
        src_a = Source(type="testimony", reliability=Reliability.B)
        await graph_store.create_node(src_a)

        result = await traceability.check_independence(src_a.id, "nonexistent-id")
        assert result.independent is False
        assert "unknown" in result.reason.lower() or "not found" in result.reason.lower()


# ===================================================================
# TestTraceability
# ===================================================================


class TestTraceability:
    async def test_fact_traces_to_source(self, graph_store, traceability):
        """A fact linked via SOURCED_FROM traces back to its source."""
        fact = Fact(content="The treaty was signed in 1648")
        src = Source(type="textbook", reliability=Reliability.B)
        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.TWO)
        )

        sources = await traceability.trace_fact_to_sources(fact.id)
        assert len(sources) == 1
        assert sources[0].id == src.id

    async def test_fact_traces_to_citation(self, graph_store, traceability):
        """A fact with multiple sources returns all of them."""
        fact = Fact(content="GDP grew by 3%")
        src_a = Source(type="report_a", reliability=Reliability.B)
        src_b = Source(type="report_b", reliability=Reliability.C)
        await graph_store.create_node(fact)
        await graph_store.create_node(src_a)
        await graph_store.create_node(src_b)
        await graph_store.create_relationship(
            fact.id, src_a.id, SourcedFrom(credibility=Credibility.TWO)
        )
        await graph_store.create_relationship(
            fact.id, src_b.id, SourcedFrom(credibility=Credibility.THREE)
        )

        sources = await traceability.trace_fact_to_sources(fact.id)
        assert len(sources) == 2
        source_ids = {s.id for s in sources}
        assert src_a.id in source_ids
        assert src_b.id in source_ids

    async def test_full_chain_fact_to_root_source(self, graph_store, traceability):
        """Fact → SOURCED_FROM → Source → CITES → Root."""
        fact = Fact(content="Inflation reached 5%")
        src = Source(type="news", reliability=Reliability.C)
        root = Source(type="central_bank", reliability=Reliability.A)
        await graph_store.create_node(fact)
        await graph_store.create_node(src)
        await graph_store.create_node(root)
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.THREE)
        )
        await graph_store.create_relationship(src.id, root.id, Cites())

        chains = await traceability.trace_fact_to_root_sources(fact.id)
        assert len(chains) == 1
        chain = chains[0]
        assert chain.root.id == root.id
        assert len(chain.sources) == 2
        assert chain.sources[0].id == src.id
        assert chain.sources[1].id == root.id
