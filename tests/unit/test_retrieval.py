"""Unit tests for Layer 6 retrieval engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timezone

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


@dataclass
class _RichGraphRetriever:
    calls: int = 0
    last_query: str | None = None
    last_limit: int | None = None
    last_max_depth: int | None = None
    last_include_sources: bool | None = None
    last_include_contradictions: bool | None = None

    async def find_claim_context_by_content(
        self,
        query: str,
        *,
        limit: int,
        max_depth: int,
        include_sources: bool,
        include_contradictions: bool,
    ) -> list[dict]:
        self.calls += 1
        self.last_query = query
        self.last_limit = limit
        self.last_max_depth = max_depth
        self.last_include_sources = include_sources
        self.last_include_contradictions = include_contradictions
        return [
            {
                "node": {
                    "id": "fact_1",
                    "content": "Storm impacted flight routes",
                    "labels": ["Memory", "Fact", "FlightDisruption"],
                    "confidence": "B2",
                    "properties": {"route": "SXM-STT"},
                },
                "causal_chain": [
                    {
                        "relation": "CAUSED_BY",
                        "target_id": "fact_2",
                        "target_summary": "Severe weather event",
                        "confidence": "C3",
                    }
                ],
                "contradictions": [
                    {
                        "id": "contra_1",
                        "memory_id": "fact_1",
                        "nature": "factual_conflict",
                        "resolution_status": "unresolved",
                        "detected_at": datetime.now(timezone.utc).isoformat(),
                        "memory": {
                            "id": "fact_9",
                            "content": "No weather issue reported",
                            "labels": ["Memory", "Fact", "OpsReport"],
                            "confidence": "D4",
                            "properties": {},
                        },
                    }
                ],
                "sources": [
                    {
                        "id": "src_1",
                        "type": "flight_log",
                        "ref": "https://example.test/log",
                        "citation": "p.3",
                        "reliability": "B",
                        "credibility": "2",
                    }
                ],
            }
        ]


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
        assert isinstance(result.meta.retrieval_ms, int)
        assert result.meta.retrieval_ms >= 0
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

    async def test_uses_graph_path_context_when_available(self):
        wm = _FakeWorkingMemory(results=[])
        graph = _RichGraphRetriever()
        engine = RetrievalEngine(wm, graph_retriever=graph)

        result = await engine.retrieve(
            GetMemoryInput(query="storm", max_depth=4, include_sources=True)
        )

        assert graph.calls == 1
        assert graph.last_query == "storm"
        assert graph.last_limit == 20
        assert graph.last_max_depth == 4
        assert graph.last_include_sources is True
        assert graph.last_include_contradictions is True

        assert result.meta.total_found == 1
        assert result.meta.graph_hits == 1
        assert result.memories[0].type == "Fact"
        assert result.memories[0].dynamic_type == "FlightDisruption"
        assert result.memories[0].causal_chain[0].relation == "caused_by"
        assert result.memories[0].causal_chain[0].target_id == "fact_2"
        assert result.memories[0].sources[0].id == "src_1"
        assert result.contradictions[0].id == "contra_1"
        assert result.contradictions[0].contradicting_memory.id == "fact_9"

    async def test_compact_mode_omits_graph_heavy_fields(self):
        wm = _FakeWorkingMemory(results=[])
        graph = _RichGraphRetriever()
        engine = RetrievalEngine(wm, graph_retriever=graph)

        result = await engine.retrieve(
            GetMemoryInput(query="storm", compact=True, include_contradictions=True)
        )

        assert result.memories[0].participants == []
        assert result.memories[0].causal_chain == []
        assert result.memories[0].sources == []
        assert result.contradictions == []
