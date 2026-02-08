"""Unit tests for split flow and audit trail in correct_memory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client

from engramcp.config import AuditConfig
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
