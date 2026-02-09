"""Integration perf smoke tests for bounded retrieval on deep/branching graphs."""

from __future__ import annotations

import json
from time import perf_counter

import pytest
from fastmcp import Client

from engramcp.config import AuditConfig
from engramcp.config import ConsolidationConfig
from engramcp.engine.llm_adapters import NoopLLMAdapter
from engramcp.server import configure
from engramcp.server import mcp
from engramcp.server import shutdown

_RETRIEVAL_LATENCY_BUDGET_MS = 5000


def _parse(result) -> dict:
    return json.loads(result.content[0].text)


@pytest.fixture(autouse=True)
async def _setup_server(redis_container, neo4j_container):
    await configure(
        redis_url=redis_container,
        enable_consolidation=True,
        neo4j_url=neo4j_container,
        llm_adapter=NoopLLMAdapter(),
        consolidation_config=ConsolidationConfig(fragment_threshold=9999),
        audit_config=AuditConfig(enabled=False),
    )
    yield
    await shutdown()


class TestRetrievalPerformance:
    async def test_graph_retrieval_is_bounded_on_deep_branching_topology(
        self, neo4j_driver
    ):
        # Build one query-matching root plus a branching causal graph.
        async with neo4j_driver.session() as session:
            await session.run(
                "CREATE (:Memory:Fact {id: $id, content: $content, updated_at: timestamp()})",
                id="root_fact",
                content="Storm disrupted route SXM-STT",
            )

            # Create 4-level branching (1 + 4 + 16 + 64 = 85 fact nodes)
            # with causal edges that retrieval traverses.
            previous_level = ["root_fact"]
            counter = 0
            for depth in range(1, 5):
                current_level: list[str] = []
                for parent in previous_level:
                    for branch in range(4):
                        counter += 1
                        child = f"fact_d{depth}_{counter}_{branch}"
                        current_level.append(child)
                        await session.run(
                            """
                            MATCH (p:Memory {id: $parent_id})
                            CREATE (c:Memory:Fact {
                              id: $child_id,
                              content: $content,
                              updated_at: timestamp()
                            })
                            CREATE (p)-[:CAUSED_BY]->(c)
                            """,
                            parent_id=parent,
                            child_id=child,
                            content=f"Derived weather signal depth {depth} branch {branch}",
                        )
                previous_level = current_level

        async with Client(mcp) as client:
            start = perf_counter()
            result = await client.call_tool(
                "get_memory",
                {
                    "query": "storm route",
                    "max_depth": 5,
                    "include_sources": False,
                    "include_contradictions": False,
                    "limit": 10,
                },
            )
            elapsed_ms = (perf_counter() - start) * 1000
            data = _parse(result)

        assert data["meta"]["graph_hits"] >= 1
        assert data["meta"]["retrieval_ms"] is not None
        assert data["meta"]["retrieval_ms"] <= _RETRIEVAL_LATENCY_BUDGET_MS
        assert elapsed_ms <= _RETRIEVAL_LATENCY_BUDGET_MS
