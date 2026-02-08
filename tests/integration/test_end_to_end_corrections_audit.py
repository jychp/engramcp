"""End-to-end integration tests for correction cascades and audit completeness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client

from engramcp.config import AuditConfig
from engramcp.config import LLMConfig
from engramcp.graph import GraphStore
from engramcp.models.nodes import Concept
from engramcp.models.nodes import DerivedStatus
from engramcp.models.nodes import Fact
from engramcp.models.nodes import Pattern
from engramcp.models.nodes import Rule
from engramcp.models.relations import DerivedFrom
from engramcp.server import configure
from engramcp.server import mcp
from engramcp.server import shutdown


def _parse(result) -> dict:
    return json.loads(result.content[0].text)


@pytest.fixture
async def correction_graph_client(redis_container, neo4j_container, tmp_path: Path):
    audit_path = tmp_path / "correction_graph_audit.jsonl"
    await configure(
        redis_url=redis_container,
        enable_consolidation=True,
        neo4j_url=neo4j_container,
        llm_config=LLMConfig(provider="noop"),
        audit_config=AuditConfig(file_path=str(audit_path), enabled=True),
    )
    async with Client(mcp) as client:
        yield client, audit_path
    await shutdown()


@pytest.fixture
async def correction_wm_client(redis_container, tmp_path: Path):
    audit_path = tmp_path / "correction_wm_audit.jsonl"
    await configure(
        redis_url=redis_container,
        audit_config=AuditConfig(file_path=str(audit_path), enabled=True),
    )
    async with Client(mcp) as client:
        yield client, audit_path
    await shutdown()


class TestEndToEndCorrectionCascade:
    async def test_reclassify_cascades_to_derived_dependents(
        self,
        correction_graph_client,
        graph_store: GraphStore,
    ):
        client, audit_path = correction_graph_client

        fact = Fact(content="Pilot observed sustained crosswinds at destination")
        pattern = Pattern(content="Crosswinds frequently drive reroutes", derivation_run_id="r1")
        concept = Concept(content="Weather disruption concept", derivation_run_id="r2")
        rule = Rule(content="Crosswinds cause arrival delays", derivation_run_id="r3")

        for node in (fact, pattern, concept, rule):
            await graph_store.create_node(node)

        await graph_store.create_relationship(
            pattern.id,
            fact.id,
            DerivedFrom(derivation_run_id="r1", derivation_method="manual"),
        )
        await graph_store.create_relationship(
            concept.id,
            pattern.id,
            DerivedFrom(derivation_run_id="r2", derivation_method="manual"),
        )
        await graph_store.create_relationship(
            rule.id,
            concept.id,
            DerivedFrom(derivation_run_id="r3", derivation_method="manual"),
        )

        result = await client.call_tool(
            "correct_memory",
            {
                "target_id": pattern.id,
                "action": "reclassify",
                "payload": {"new_type": "Coincidence"},
            },
        )
        data = _parse(result)

        assert data["status"] == "applied"
        assert data["details"]["storage"] == "graph"
        affected = set(data["details"]["lifecycle"]["cascade"]["affected_nodes"])
        assert affected == {concept.id, rule.id}

        updated_pattern = await graph_store.get_node(pattern.id)
        updated_concept = await graph_store.get_node(concept.id)
        updated_rule = await graph_store.get_node(rule.id)
        assert updated_pattern is not None
        assert updated_concept is not None
        assert updated_rule is not None
        assert updated_pattern.status == DerivedStatus.dissolved
        assert updated_concept.status == DerivedStatus.contested
        assert updated_rule.status == DerivedStatus.contested

        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event_type"] == "CORRECT_MEMORY"
        assert event["payload"]["action"] == "reclassify"
        assert event["payload"]["target_id"] == pattern.id
        assert event["payload"]["storage"] == "graph"
        assert "lifecycle" in event["payload"]


class TestEndToEndCorrectionAuditCompleteness:
    async def test_correct_memory_wm_actions_emit_complete_audit_payloads(
        self,
        correction_wm_client,
    ):
        client, audit_path = correction_wm_client

        contest_id = _parse(
            await client.call_tool(
                "send_memory",
                {"content": "Unverified report", "confidence_hint": "B"},
            )
        )["memory_id"]
        annotate_id = _parse(
            await client.call_tool("send_memory", {"content": "Flight delayed"})
        )["memory_id"]
        split_id = _parse(
            await client.call_tool(
                "send_memory", {"content": "Entity with mixed identities"}
            )
        )["memory_id"]
        merge_target_id = _parse(
            await client.call_tool("send_memory", {"content": "Captain A profile"})
        )["memory_id"]
        merge_with_id = _parse(
            await client.call_tool("send_memory", {"content": "Captain A alias profile"})
        )["memory_id"]
        reclassify_id = _parse(
            await client.call_tool("send_memory", {"content": "Timeline note"})
        )["memory_id"]

        for payload in (
            {
                "target_id": contest_id,
                "action": "contest",
                "payload": {"reason": "conflicting statements"},
            },
            {
                "target_id": annotate_id,
                "action": "annotate",
                "payload": {"note": "Corroborated by tower bulletin"},
            },
            {
                "target_id": split_id,
                "action": "split_entity",
                "payload": {"split_into": ["identity_a", "identity_b"]},
            },
            {
                "target_id": merge_target_id,
                "action": "merge_entities",
                "payload": {"merge_with": merge_with_id},
            },
            {
                "target_id": reclassify_id,
                "action": "reclassify",
                "payload": {"new_type": "Event"},
            },
        ):
            result = await client.call_tool("correct_memory", payload)
            assert _parse(result)["status"] == "applied"

        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) == 5
        events = [json.loads(line) for line in lines]

        by_action = {event["payload"]["action"]: event["payload"] for event in events}
        assert set(by_action) == {
            "contest",
            "annotate",
            "split_entity",
            "merge_entities",
            "reclassify",
        }

        for payload in by_action.values():
            assert payload["status"] == "applied"
            assert payload["target_id"]

        assert by_action["contest"]["old_confidence"] == "B3"
        assert by_action["contest"]["new_confidence"] == "B4"
        assert "cascade" in by_action["contest"]

        assert by_action["annotate"]["note"] == "Corroborated by tower bulletin"
        assert by_action["annotate"]["annotation_count"] == 1

        assert by_action["split_entity"]["split_into"] == ["identity_a", "identity_b"]
        assert len(by_action["split_entity"]["created_memory_ids"]) == 2
        assert by_action["split_entity"]["storage"] == "working_memory"

        assert by_action["merge_entities"]["merged_into"] == merge_target_id
        assert by_action["merge_entities"]["merged_from"] == merge_with_id

        assert by_action["reclassify"]["old_type"] == "Fact"
        assert by_action["reclassify"]["new_type"] == "Event"
