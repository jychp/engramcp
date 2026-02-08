"""Unit tests for Layer 6 retrieval engine."""

from __future__ import annotations

from dataclasses import dataclass

from engramcp.engine.retrieval import RetrievalEngine
from engramcp.memory.schemas import MemoryFragment
from engramcp.models.nodes import Fact
from engramcp.models.schemas import GetMemoryInput


@dataclass
class _FakeWorkingMemory:
    results: list[MemoryFragment]

    async def search(self, query: str, *, min_confidence: str = "F6") -> list[MemoryFragment]:
        del query, min_confidence
        return list(self.results)


@dataclass
class _FakeGraphRetriever:
    nodes: list[Fact]
    calls: int = 0
    last_query: str | None = None
    last_limit: int | None = None

    async def find_claim_nodes(self) -> list[Fact]:
        self.calls += 1
        return list(self.nodes)

    async def find_claim_nodes_by_content(self, query: str, *, limit: int) -> list[Fact]:
        self.calls += 1
        self.last_query = query
        self.last_limit = limit
        normalized = query.casefold()
        filtered = [node for node in self.nodes if normalized in node.content.casefold()]
        return filtered[:limit]


@dataclass
class _LegacyGraphRetriever:
    nodes: list[Fact]
    calls: int = 0

    async def find_claim_nodes(self) -> list[Fact]:
        self.calls += 1
        return list(self.nodes)


class TestRetrievalEngine:
    async def test_working_memory_first(self):
        fragment = MemoryFragment(
            id="mem_1",
            content="The project uses Redis",
            confidence="B2",
            timestamp=100.0,
        )
        wm = _FakeWorkingMemory(results=[fragment])
        graph = _FakeGraphRetriever(nodes=[Fact(content="Graph fact should not be used")])
        engine = RetrievalEngine(wm, graph_retriever=graph)

        result = await engine.retrieve(GetMemoryInput(query="Redis"))

        assert result.meta.total_found == 1
        assert result.meta.working_memory_hits == 1
        assert result.meta.graph_hits == 0
        assert result.memories[0].id == "mem_1"
        assert graph.calls == 0

    async def test_falls_through_to_graph(self):
        wm = _FakeWorkingMemory(results=[])
        graph = _FakeGraphRetriever(
            nodes=[
                Fact(id="fact_1", content="Neo4j graph fallback result"),
                Fact(id="fact_2", content="Other content"),
            ]
        )
        engine = RetrievalEngine(wm, graph_retriever=graph)

        result = await engine.retrieve(GetMemoryInput(query="Neo4j"))

        assert result.meta.total_found == 1
        assert result.meta.working_memory_hits == 0
        assert result.meta.graph_hits == 1
        assert result.memories[0].id == "fact_1"
        assert result.memories[0].content == "Neo4j graph fallback result"
        assert graph.calls == 1
        assert graph.last_query == "Neo4j"
        assert graph.last_limit == 20

    async def test_falls_back_to_legacy_graph_retriever_contract(self):
        wm = _FakeWorkingMemory(results=[])
        graph = _LegacyGraphRetriever(
            nodes=[
                Fact(id="fact_1", content="Neo4j graph fallback result"),
                Fact(id="fact_2", content="Other content"),
            ]
        )
        engine = RetrievalEngine(wm, graph_retriever=graph)

        result = await engine.retrieve(GetMemoryInput(query="Neo4j"))

        assert result.meta.total_found == 1
        assert result.meta.graph_hits == 1
        assert result.memories[0].id == "fact_1"
        assert graph.calls == 1

    async def test_tracks_query_pattern_shape(self):
        fragment = MemoryFragment(id="mem_1", content="A flew to St. Thomas")
        engine = RetrievalEngine(_FakeWorkingMemory(results=[fragment]))

        await engine.retrieve(GetMemoryInput(query="flew"))
        await engine.retrieve(GetMemoryInput(query="flew"))
        await engine.retrieve(GetMemoryInput(query="flew"))

        assert (
            engine.query_demand_count(
                node_types=["Fact"],
                properties=[
                    "content",
                    "confidence",
                    "participants",
                    "causal_chain",
                    "sources",
                    "contradictions",
                    "min_confidence",
                ],
            )
            == 3
        )
        assert engine.concept_candidate_count() == 1
