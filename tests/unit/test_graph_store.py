"""Tests for the graph store (Layer 2) and schema initialization.

These tests require a Neo4j testcontainer (session-scoped via conftest).
"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest

from engramcp.models import (
    Agent,
    AgentType,
    Artifact,
    ArtifactType,
    Concept,
    Concerns,
    Contradicts,
    Credibility,
    Decision,
    DerivedStatus,
    Event,
    Fact,
    FactStatus,
    Observation,
    Outcome,
    ParticipatedIn,
    Pattern,
    PossiblySameAs,
    Reliability,
    ResolutionStatus,
    Rule,
    Source,
    SourcedFrom,
    TemporalPrecision,
)


# ===================================================================
# Schema initialization
# ===================================================================


class TestSchema:
    async def test_init_schema_creates_constraints(self, neo4j_driver, _graph_schema_initialized):
        """The uniqueness constraint on Memory.id exists."""
        async with neo4j_driver.session() as session:
            result = await session.run("SHOW CONSTRAINTS")
            constraints = [record.data() async for record in result]
        names = [c["name"] for c in constraints]
        assert "mem_unique_id" in names

    async def test_init_schema_creates_indexes(self, neo4j_driver, _graph_schema_initialized):
        """Key indexes exist after schema init."""
        async with neo4j_driver.session() as session:
            result = await session.run("SHOW INDEXES")
            indexes = [record.data() async for record in result]
        names = {idx["name"] for idx in indexes}
        assert "agent_name" in names
        assert "fact_status" in names
        assert "source_reliability" in names

    async def test_init_schema_idempotent(self, neo4j_driver, _graph_schema_initialized):
        """Running init_schema twice does not raise."""
        from engramcp.graph.schema import init_schema

        await init_schema(neo4j_driver)  # Second call should be fine


# ===================================================================
# Node CRUD
# ===================================================================


class TestNodes:
    async def test_create_and_get_fact(self, graph_store):
        fact = Fact(content="The sky is blue")
        node_id = await graph_store.create_node(fact)
        assert node_id == fact.id

        retrieved = await graph_store.get_node(node_id)
        assert retrieved is not None
        assert isinstance(retrieved, Fact)
        assert retrieved.content == "The sky is blue"
        assert retrieved.status == FactStatus.active

    async def test_create_and_get_event(self, graph_store):
        now = datetime.now(timezone.utc)
        event = Event(
            content="Meeting at 10am",
            occurred_at=now,
            temporal_precision=TemporalPrecision.exact,
        )
        await graph_store.create_node(event)
        retrieved = await graph_store.get_node(event.id)
        assert isinstance(retrieved, Event)
        assert retrieved.content == "Meeting at 10am"
        assert retrieved.temporal_precision == TemporalPrecision.exact

    async def test_create_and_get_observation(self, graph_store):
        obs = Observation(content="The room was cold")
        await graph_store.create_node(obs)
        retrieved = await graph_store.get_node(obs.id)
        assert isinstance(retrieved, Observation)

    async def test_create_and_get_decision(self, graph_store):
        now = datetime.now(timezone.utc)
        dec = Decision(content="Chose plan B", occurred_at=now)
        await graph_store.create_node(dec)
        retrieved = await graph_store.get_node(dec.id)
        assert isinstance(retrieved, Decision)

    async def test_create_and_get_outcome(self, graph_store):
        now = datetime.now(timezone.utc)
        out = Outcome(content="Revenue increased 20%", occurred_at=now)
        await graph_store.create_node(out)
        retrieved = await graph_store.get_node(out.id)
        assert isinstance(retrieved, Outcome)

    async def test_create_and_get_agent(self, graph_store):
        agent = Agent(
            name="Alice",
            type=AgentType.person,
            aliases=["Al", "A"],
        )
        await graph_store.create_node(agent)
        retrieved = await graph_store.get_node(agent.id)
        assert isinstance(retrieved, Agent)
        assert retrieved.name == "Alice"
        assert retrieved.aliases == ["Al", "A"]

    async def test_create_and_get_artifact(self, graph_store):
        art = Artifact(
            name="report.pdf",
            type=ArtifactType.document,
            ref="https://example.com/report.pdf",
        )
        await graph_store.create_node(art)
        retrieved = await graph_store.get_node(art.id)
        assert isinstance(retrieved, Artifact)
        assert retrieved.ref == "https://example.com/report.pdf"

    async def test_create_and_get_source(self, graph_store):
        src = Source(type="court_document", reliability=Reliability.A)
        await graph_store.create_node(src)
        retrieved = await graph_store.get_node(src.id)
        assert isinstance(retrieved, Source)
        assert retrieved.reliability == Reliability.A

    async def test_create_derived_nodes(self, graph_store):
        pat = Pattern(content="Recurring pattern", derivation_run_id="run-1")
        con = Concept(content="Abstract concept", derivation_run_id="run-2")
        rule = Rule(content="If X then Y", derivation_run_id="run-3")

        for node in (pat, con, rule):
            await graph_store.create_node(node)

        p = await graph_store.get_node(pat.id)
        assert isinstance(p, Pattern)
        assert p.derivation_depth == 1

        c = await graph_store.get_node(con.id)
        assert isinstance(c, Concept)
        assert c.derivation_depth == 2

        r = await graph_store.get_node(rule.id)
        assert isinstance(r, Rule)
        assert r.derivation_depth == 3

    async def test_get_nonexistent_node(self, graph_store):
        result = await graph_store.get_node("nonexistent-id")
        assert result is None

    async def test_delete_node(self, graph_store):
        fact = Fact(content="Temporary fact")
        await graph_store.create_node(fact)
        assert await graph_store.get_node(fact.id) is not None

        deleted = await graph_store.delete_node(fact.id)
        assert deleted is True
        assert await graph_store.get_node(fact.id) is None

    async def test_delete_nonexistent_node(self, graph_store):
        deleted = await graph_store.delete_node("nonexistent")
        assert deleted is False

    async def test_update_node(self, graph_store):
        fact = Fact(content="Original content")
        await graph_store.create_node(fact)

        updated = await graph_store.update_node(
            fact.id,
            content="Updated content",
            status=FactStatus.contested,
        )
        assert updated is not None
        assert isinstance(updated, Fact)
        assert updated.content == "Updated content"
        assert updated.status == FactStatus.contested

    async def test_datetime_roundtrip(self, graph_store):
        """Python datetime objects survive the Neo4j round-trip."""
        now = datetime.now(timezone.utc)
        event = Event(content="Time test", occurred_at=now)
        await graph_store.create_node(event)
        retrieved = await graph_store.get_node(event.id)
        assert isinstance(retrieved, Event)
        # Datetimes should be within 1 second (driver precision)
        diff = abs((retrieved.occurred_at - now).total_seconds())
        assert diff < 1.0

    async def test_uniqueness_constraint(self, graph_store):
        """Creating two nodes with the same ID raises an error."""
        fact1 = Fact(id="duplicate-id", content="First")
        fact2 = Fact(id="duplicate-id", content="Second")
        await graph_store.create_node(fact1)
        with pytest.raises(Exception):
            await graph_store.create_node(fact2)


# ===================================================================
# Relationship CRUD
# ===================================================================


class TestRelationships:
    async def test_create_sourced_from(self, graph_store):
        fact = Fact(content="Claim X")
        src = Source(type="testimony", reliability=Reliability.B)
        await graph_store.create_node(fact)
        await graph_store.create_node(src)

        rel = SourcedFrom(credibility=Credibility.TWO)
        created = await graph_store.create_relationship(fact.id, src.id, rel)
        assert created is True

    async def test_create_relationship_missing_node(self, graph_store):
        fact = Fact(content="Orphan")
        await graph_store.create_node(fact)

        rel = SourcedFrom(credibility=Credibility.ONE)
        created = await graph_store.create_relationship(
            fact.id, "nonexistent", rel
        )
        assert created is False

    async def test_create_concerns(self, graph_store):
        fact = Fact(content="Alice did X")
        agent = Agent(name="Alice", type=AgentType.person)
        await graph_store.create_node(fact)
        await graph_store.create_node(agent)

        rel = Concerns(role="subject")
        created = await graph_store.create_relationship(fact.id, agent.id, rel)
        assert created is True

    async def test_create_participated_in(self, graph_store):
        now = datetime.now(timezone.utc)
        event = Event(content="Company meeting", occurred_at=now)
        agent = Agent(name="Bob", type=AgentType.person)
        await graph_store.create_node(event)
        await graph_store.create_node(agent)

        rel = ParticipatedIn(role="attendee")
        created = await graph_store.create_relationship(agent.id, event.id, rel)
        assert created is True

    async def test_create_contradicts(self, graph_store):
        f1 = Fact(content="Statement A")
        f2 = Fact(content="Statement B")
        await graph_store.create_node(f1)
        await graph_store.create_node(f2)

        rel = Contradicts(detection_run_id="run-42")
        created = await graph_store.create_relationship(f1.id, f2.id, rel)
        assert created is True

    async def test_get_relationships_outgoing(self, graph_store):
        fact = Fact(content="Test")
        agent = Agent(name="Carol", type=AgentType.person)
        await graph_store.create_node(fact)
        await graph_store.create_node(agent)

        rel = Concerns(role="subject")
        await graph_store.create_relationship(fact.id, agent.id, rel)

        rels = await graph_store.get_relationships(fact.id, direction="outgoing")
        assert len(rels) == 1
        assert rels[0]["type"] == "CONCERNS"

    async def test_get_relationships_incoming(self, graph_store):
        fact = Fact(content="Test")
        agent = Agent(name="Dave", type=AgentType.person)
        await graph_store.create_node(fact)
        await graph_store.create_node(agent)

        rel = Concerns(role="subject")
        await graph_store.create_relationship(fact.id, agent.id, rel)

        rels = await graph_store.get_relationships(agent.id, direction="incoming")
        assert len(rels) == 1
        assert rels[0]["type"] == "CONCERNS"

    async def test_get_relationships_by_type(self, graph_store):
        fact = Fact(content="Multi-rel test")
        agent = Agent(name="Eve", type=AgentType.person)
        src = Source(type="testimony", reliability=Reliability.C)
        await graph_store.create_node(fact)
        await graph_store.create_node(agent)
        await graph_store.create_node(src)

        await graph_store.create_relationship(
            fact.id, agent.id, Concerns(role="subject")
        )
        await graph_store.create_relationship(
            fact.id, src.id, SourcedFrom(credibility=Credibility.THREE)
        )

        concerns_only = await graph_store.get_relationships(
            fact.id, rel_type="CONCERNS", direction="outgoing"
        )
        assert len(concerns_only) == 1

    async def test_create_possibly_same_as(self, graph_store):
        a1 = Agent(name="John Smith", type=AgentType.person)
        a2 = Agent(name="J. Smith", type=AgentType.person)
        await graph_store.create_node(a1)
        await graph_store.create_node(a2)

        rel = PossiblySameAs(similarity_score=0.85)
        created = await graph_store.create_relationship(a1.id, a2.id, rel)
        assert created is True

    async def test_get_relationships_rejects_invalid_direction(self, graph_store):
        fact = Fact(content="Direction test")
        await graph_store.create_node(fact)
        with pytest.raises(ValueError):
            await graph_store.get_relationships(fact.id, direction="sideways")


# ===================================================================
# Query methods
# ===================================================================


class TestQueries:
    async def test_find_by_id(self, graph_store):
        fact = Fact(content="Find me")
        await graph_store.create_node(fact)
        result = await graph_store.find_by_id(fact.id)
        assert result is not None
        assert result.content == "Find me"

    async def test_find_facts_by_agent(self, graph_store):
        agent = Agent(name="TestAgent", type=AgentType.person)
        f1 = Fact(content="Fact about TestAgent")
        f2 = Fact(content="Another fact about TestAgent")
        f3 = Fact(content="Unrelated fact")

        await graph_store.create_node(agent)
        await graph_store.create_node(f1)
        await graph_store.create_node(f2)
        await graph_store.create_node(f3)

        await graph_store.create_relationship(f1.id, agent.id, Concerns(role="subject"))
        await graph_store.create_relationship(f2.id, agent.id, Concerns(role="subject"))

        results = await graph_store.find_facts_by_agent("TestAgent")
        assert len(results) == 2
        ids = {r.id for r in results}
        assert f1.id in ids
        assert f2.id in ids
        assert f3.id not in ids

    async def test_find_events_in_range(self, graph_store):
        base = datetime(2024, 6, 1, tzinfo=timezone.utc)
        e1 = Event(content="Early event", occurred_at=base)
        e2 = Event(
            content="Mid event",
            occurred_at=base + timedelta(days=15),
        )
        e3 = Event(
            content="Late event",
            occurred_at=base + timedelta(days=60),
        )

        for e in (e1, e2, e3):
            await graph_store.create_node(e)

        results = await graph_store.find_events_in_range(
            start=base,
            end=base + timedelta(days=30),
        )
        assert len(results) == 2
        contents = [r.content for r in results]
        assert "Early event" in contents
        assert "Mid event" in contents
        assert "Late event" not in contents

    async def test_find_sources_by_reliability(self, graph_store):
        s1 = Source(type="testimony", reliability=Reliability.A)
        s2 = Source(type="news_article", reliability=Reliability.A)
        s3 = Source(type="rumor", reliability=Reliability.E)

        for s in (s1, s2, s3):
            await graph_store.create_node(s)

        results = await graph_store.find_sources_by_reliability(Reliability.A)
        assert len(results) == 2

    async def test_find_contradictions_unresolved(self, graph_store):
        f1 = Fact(content="Claim A")
        f2 = Fact(content="Claim B")
        await graph_store.create_node(f1)
        await graph_store.create_node(f2)

        rel = Contradicts(
            detection_run_id="run-1",
            resolution_status=ResolutionStatus.unresolved,
        )
        await graph_store.create_relationship(f1.id, f2.id, rel)

        results = await graph_store.find_contradictions_unresolved()
        assert len(results) == 1
        assert results[0]["from_node"].id == f1.id
        assert results[0]["to_node"].id == f2.id

    async def test_find_agent_by_alias(self, graph_store):
        agent = Agent(
            name="John Smith",
            type=AgentType.person,
            aliases=["Johnny", "J.S."],
        )
        await graph_store.create_node(agent)

        # Find by primary name
        results = await graph_store.find_agent_by_alias("John Smith")
        assert len(results) == 1

        # Find by alias
        results = await graph_store.find_agent_by_alias("Johnny")
        assert len(results) == 1
        assert results[0].name == "John Smith"

        # No match
        results = await graph_store.find_agent_by_alias("Unknown")
        assert len(results) == 0

    async def test_find_possibly_same_as_unresolved(self, graph_store):
        a1 = Agent(name="Jane Doe", type=AgentType.person)
        a2 = Agent(name="J. Doe", type=AgentType.person)
        await graph_store.create_node(a1)
        await graph_store.create_node(a2)

        rel = PossiblySameAs(similarity_score=0.9)
        await graph_store.create_relationship(a1.id, a2.id, rel)

        results = await graph_store.find_possibly_same_as_unresolved()
        assert len(results) == 1
        assert results[0]["similarity_score"] == 0.9

    async def test_find_by_label_returns_matching_nodes(self, graph_store):
        a1 = Agent(name="Alpha", type=AgentType.person)
        a2 = Agent(name="Beta", type=AgentType.organization)
        fact = Fact(content="Unrelated fact")
        await graph_store.create_node(a1)
        await graph_store.create_node(a2)
        await graph_store.create_node(fact)

        results = await graph_store.find_by_label("Agent")
        ids = {r.id for r in results}
        assert a1.id in ids
        assert a2.id in ids
        assert fact.id not in ids

    async def test_find_by_label_empty_result(self, graph_store):
        results = await graph_store.find_by_label("Pattern")
        assert results == []

    async def test_find_by_label_rejects_invalid_label(self, graph_store):
        with pytest.raises(ValueError):
            await graph_store.find_by_label("Agent) DETACH DELETE n //")
