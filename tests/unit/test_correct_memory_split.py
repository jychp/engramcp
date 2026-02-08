"""Unit tests for split flow and audit trail in correct_memory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client

from engramcp.config import AuditConfig
from engramcp.graph import GraphStore
from engramcp.models.nodes import Agent
from engramcp.models.nodes import AgentType
from engramcp.models.nodes import Concept
from engramcp.models.nodes import DerivedStatus
from engramcp.models.nodes import Pattern
from engramcp.models.nodes import Rule
from engramcp.models.relations import DerivedFrom
from engramcp.server import _get_wm
from engramcp.server import configure
from engramcp.server import mcp


def _parse(result) -> dict:
    return json.loads(result.content[0].text)


@pytest.fixture
async def split_client(redis_container, tmp_path: Path):
    audit_path = tmp_path / "split_audit.jsonl"
    await configure(
        redis_url=redis_container,
        audit_config=AuditConfig(file_path=str(audit_path), enabled=True),
    )
    async with Client(mcp) as client:
        yield client, audit_path


@pytest.fixture
async def graph_split_client(redis_container, neo4j_container, tmp_path: Path):
    audit_path = tmp_path / "graph_split_audit.jsonl"
    await configure(
        redis_url=redis_container,
        enable_consolidation=True,
        neo4j_url=neo4j_container,
        audit_config=AuditConfig(file_path=str(audit_path), enabled=True),
    )
    async with Client(mcp) as client:
        yield client, audit_path


class TestCorrectMemorySplitFlow:
    async def test_split_entity_creates_children_deletes_target_and_audits(
        self, split_client
    ):
        client, audit_path = split_client
        send_result = await client.call_tool(
            "send_memory", {"content": "Entity with mixed identities"}
        )
        target_id = _parse(send_result)["memory_id"]

        result = await client.call_tool(
            "correct_memory",
            {
                "target_id": target_id,
                "action": "split_entity",
                "payload": {"split_into": ["identity_a", "identity_b"]},
            },
        )
        data = _parse(result)

        assert data["status"] == "applied"
        assert data["action"] == "split_entity"
        assert data["details"]["split_into"] == ["identity_a", "identity_b"]
        created_ids = data["details"]["created_memory_ids"]
        assert len(created_ids) == 2
        assert all(mid.startswith("mem_") for mid in created_ids)

        wm = _get_wm()
        assert not await wm.exists(target_id)
        assert await wm.exists(created_ids[0])
        assert await wm.exists(created_ids[1])

        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event_type"] == "CORRECT_MEMORY"
        assert event["payload"]["action"] == "split_entity"
        assert event["payload"]["target_id"] == target_id
        assert event["payload"]["created_memory_ids"] == created_ids

    async def test_split_entity_rejects_invalid_payload(self, split_client):
        client, _ = split_client
        send_result = await client.call_tool("send_memory", {"content": "Entity"})
        target_id = _parse(send_result)["memory_id"]

        result = await client.call_tool(
            "correct_memory",
            {"target_id": target_id, "action": "split_entity", "payload": {}},
        )
        data = _parse(result)

        assert data["status"] == "rejected"
        assert data["error_code"] == "validation_error"

    async def test_split_entity_with_empty_list_keeps_target_memory(self, split_client):
        client, audit_path = split_client
        send_result = await client.call_tool("send_memory", {"content": "Entity"})
        target_id = _parse(send_result)["memory_id"]

        result = await client.call_tool(
            "correct_memory",
            {
                "target_id": target_id,
                "action": "split_entity",
                "payload": {"split_into": []},
            },
        )
        data = _parse(result)

        assert data["status"] == "rejected"
        assert data["error_code"] == "validation_error"

        wm = _get_wm()
        assert await wm.exists(target_id)
        if audit_path.exists():
            assert audit_path.read_text().strip() == ""


class TestCorrectMemoryActions:
    async def test_contest_downgrades_confidence_and_audits(self, split_client):
        client, audit_path = split_client
        send_result = await client.call_tool(
            "send_memory", {"content": "Claim under review", "confidence_hint": "B"}
        )
        target_id = _parse(send_result)["memory_id"]

        result = await client.call_tool(
            "correct_memory",
            {
                "target_id": target_id,
                "action": "contest",
                "payload": {"reason": "conflicting testimony"},
            },
        )
        data = _parse(result)

        assert data["status"] == "applied"
        assert data["action"] == "contest"
        assert data["details"]["old_confidence"] == "B3"
        assert data["details"]["new_confidence"] == "B4"
        assert data["details"]["cascade"]["triggered"] is False

        wm = _get_wm()
        updated = await wm.get(target_id)
        assert updated is not None
        assert updated.confidence == "B4"
        assert updated.properties["status"] == "contested"
        assert updated.properties["contest_reason"] == "conflicting testimony"

        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event_type"] == "CORRECT_MEMORY"
        assert event["payload"]["action"] == "contest"
        assert event["payload"]["target_id"] == target_id
        assert event["payload"]["old_confidence"] == "B3"
        assert event["payload"]["new_confidence"] == "B4"

    async def test_contest_caps_at_f6(self, split_client):
        client, _ = split_client
        send_result = await client.call_tool(
            "send_memory", {"content": "Weak claim", "confidence_hint": "F"}
        )
        target_id = _parse(send_result)["memory_id"]

        for _ in range(6):
            result = await client.call_tool(
                "correct_memory",
                {
                    "target_id": target_id,
                    "action": "contest",
                    "payload": {"reason": "still unverified"},
                },
            )
            data = _parse(result)
            assert data["status"] == "applied"

        wm = _get_wm()
        updated = await wm.get(target_id)
        assert updated is not None
        assert updated.confidence == "F6"

    async def test_annotate_adds_note_preserves_sources_and_audits(self, split_client):
        client, audit_path = split_client
        send_result = await client.call_tool(
            "send_memory",
            {
                "content": "Flight delayed",
                "source": {
                    "type": "flight_log",
                    "ref": "https://example.com/log.pdf",
                    "citation": "page 2",
                },
            },
        )
        target_id = _parse(send_result)["memory_id"]

        result = await client.call_tool(
            "correct_memory",
            {
                "target_id": target_id,
                "action": "annotate",
                "payload": {"note": "Weather bulletin confirms storm conditions"},
            },
        )
        data = _parse(result)

        assert data["status"] == "applied"
        assert data["action"] == "annotate"
        assert data["details"]["annotation_count"] == 1

        wm = _get_wm()
        updated = await wm.get(target_id)
        assert updated is not None
        assert len(updated.sources) == 1
        assert updated.sources[0]["type"] == "flight_log"
        assert updated.properties["annotations"][0]["note"] == (
            "Weather bulletin confirms storm conditions"
        )

        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event_type"] == "CORRECT_MEMORY"
        assert event["payload"]["action"] == "annotate"
        assert event["payload"]["note"] == "Weather bulletin confirms storm conditions"

    async def test_annotate_rejects_invalid_payload(self, split_client):
        client, _ = split_client
        send_result = await client.call_tool("send_memory", {"content": "Entity"})
        target_id = _parse(send_result)["memory_id"]

        result = await client.call_tool(
            "correct_memory",
            {"target_id": target_id, "action": "annotate", "payload": {}},
        )
        data = _parse(result)
        assert data["status"] == "rejected"
        assert data["error_code"] == "validation_error"

    async def test_merge_entities_merges_payload_and_deletes_secondary(self, split_client):
        client, audit_path = split_client
        send_a = await client.call_tool(
            "send_memory",
            {
                "content": "A is captain of flight 221",
                "source": {"type": "ops_log", "ref": "https://example.com/a"},
                "confidence_hint": "B",
            },
        )
        send_b = await client.call_tool(
            "send_memory",
            {
                "content": "A. Dupont commanded route SXM-STT",
                "source": {"type": "manifest", "ref": "https://example.com/b"},
                "confidence_hint": "C",
            },
        )
        target_id = _parse(send_a)["memory_id"]
        merge_with_id = _parse(send_b)["memory_id"]

        result = await client.call_tool(
            "correct_memory",
            {
                "target_id": target_id,
                "action": "merge_entities",
                "payload": {"merge_with": merge_with_id},
            },
        )
        data = _parse(result)

        assert data["status"] == "applied"
        assert data["action"] == "merge_entities"
        assert data["details"]["merged_into"] == target_id
        assert data["details"]["merged_from"] == merge_with_id
        assert data["details"]["storage"] == "working_memory"

        wm = _get_wm()
        merged = await wm.get(target_id)
        assert merged is not None
        assert "A is captain of flight 221" in merged.content
        assert "A. Dupont commanded route SXM-STT" in merged.content
        assert merged.confidence == "B3"
        assert len(merged.sources) == 2
        assert not await wm.exists(merge_with_id)

        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["payload"]["action"] == "merge_entities"
        assert event["payload"]["merged_into"] == target_id
        assert event["payload"]["merged_from"] == merge_with_id

    async def test_merge_entities_uses_graph_executor_when_nodes_exist(
        self,
        graph_split_client,
        graph_store: GraphStore,
    ):
        client, audit_path = graph_split_client
        survivor = Agent(name="John Smith", type=AgentType.person, aliases=["J. Smith"])
        absorbed = Agent(name="Jonathan Smith", type=AgentType.person)
        survivor_id = await graph_store.create_node(survivor)
        absorbed_id = await graph_store.create_node(absorbed)

        result = await client.call_tool(
            "correct_memory",
            {
                "target_id": survivor_id,
                "action": "merge_entities",
                "payload": {"merge_with": absorbed_id},
            },
        )
        data = _parse(result)
        assert data["status"] == "applied"
        assert data["details"]["storage"] == "graph"
        assert data["details"]["merged_into"] == survivor_id
        assert data["details"]["merged_from"] == absorbed_id
        assert isinstance(data["details"]["relations_transferred"], int)

        merged = await graph_store.get_node(survivor_id)
        removed = await graph_store.get_node(absorbed_id)
        assert merged is not None
        assert removed is None
        assert "Jonathan Smith" in merged.aliases

        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["payload"]["action"] == "merge_entities"
        assert event["payload"]["storage"] == "graph"

    async def test_merge_entities_rejects_self_merge(self, split_client):
        client, _ = split_client
        send_result = await client.call_tool("send_memory", {"content": "Entity"})
        target_id = _parse(send_result)["memory_id"]

        result = await client.call_tool(
            "correct_memory",
            {
                "target_id": target_id,
                "action": "merge_entities",
                "payload": {"merge_with": target_id},
            },
        )
        data = _parse(result)
        assert data["status"] == "rejected"
        assert data["error_code"] == "invalid_merge_target"

    async def test_reclassify_updates_type_and_audits(self, split_client):
        client, audit_path = split_client
        send_result = await client.call_tool("send_memory", {"content": "Timeline note"})
        target_id = _parse(send_result)["memory_id"]

        result = await client.call_tool(
            "correct_memory",
            {
                "target_id": target_id,
                "action": "reclassify",
                "payload": {"new_type": "Event"},
            },
        )
        data = _parse(result)

        assert data["status"] == "applied"
        assert data["action"] == "reclassify"
        assert data["details"]["old_type"] == "Fact"
        assert data["details"]["new_type"] == "Event"
        assert data["details"]["storage"] == "working_memory"

        wm = _get_wm()
        updated = await wm.get(target_id)
        assert updated is not None
        assert updated.type == "Event"
        assert updated.properties["reclassify_history"][0]["from"] == "Fact"
        assert updated.properties["reclassify_history"][0]["to"] == "Event"

        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["payload"]["action"] == "reclassify"
        assert event["payload"]["old_type"] == "Fact"
        assert event["payload"]["new_type"] == "Event"

    async def test_reclassify_derived_graph_node_updates_lifecycle(
        self,
        graph_split_client,
        graph_store: GraphStore,
    ):
        client, audit_path = graph_split_client
        pattern = Pattern(content="Recurring route detour", derivation_run_id="run-p")
        concept = Concept(content="Weather disruption concept", derivation_run_id="run-c")
        rule = Rule(content="Storms cause delays", derivation_run_id="run-r")
        for node in (pattern, concept, rule):
            await graph_store.create_node(node)
        await graph_store.create_relationship(
            concept.id,
            pattern.id,
            DerivedFrom(derivation_run_id="run-c", derivation_method="manual"),
        )
        await graph_store.create_relationship(
            rule.id,
            concept.id,
            DerivedFrom(derivation_run_id="run-r", derivation_method="manual"),
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
        assert data["details"]["old_type"] == "Pattern"
        assert data["details"]["new_type"] == "Coincidence"
        assert data["details"]["lifecycle"]["target_status"] == "dissolved"
        assert concept.id in data["details"]["lifecycle"]["cascade"]["affected_nodes"]
        assert rule.id in data["details"]["lifecycle"]["cascade"]["affected_nodes"]

        updated_pattern = await graph_store.get_node(pattern.id)
        updated_concept = await graph_store.get_node(concept.id)
        updated_rule = await graph_store.get_node(rule.id)
        assert updated_pattern is not None
        assert updated_concept is not None
        assert updated_rule is not None
        assert updated_pattern.status == DerivedStatus.dissolved
        assert updated_concept.status == DerivedStatus.dissolved
        assert updated_rule.status == DerivedStatus.dissolved

        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["payload"]["action"] == "reclassify"
        assert event["payload"]["storage"] == "graph"

    async def test_reclassify_graph_history_uses_structured_json_entries(
        self,
        graph_split_client,
        graph_store: GraphStore,
    ):
        client, _ = graph_split_client
        pattern = Pattern(content="Temporary derived signal", derivation_run_id="run-h")
        await graph_store.create_node(pattern)

        await client.call_tool(
            "correct_memory",
            {
                "target_id": pattern.id,
                "action": "reclassify",
                "payload": {"new_type": "Coincidence"},
            },
        )

        driver = graph_store._driver
        async with driver.session() as session:
            result = await session.run(
                "MATCH (n:Memory {id: $id}) RETURN n.reclassify_history AS history",
                id=pattern.id,
            )
            record = await result.single()
            assert record is not None
            history = record["history"]

        assert isinstance(history, list)
        assert history
        decoded = json.loads(history[0])
        assert decoded["from"] == "Pattern"
        assert decoded["to"] == "Coincidence"
        assert isinstance(decoded["at"], float)
