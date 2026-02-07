"""Unit tests for the audit logger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engramcp.audit import AuditEvent
from engramcp.audit import AuditEventType
from engramcp.audit import AuditLogger
from engramcp.config import AuditConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type: AuditEventType = AuditEventType.SEND_MEMORY,
    timestamp: float = 1000.0,
    payload: dict | None = None,
) -> AuditEvent:
    return AuditEvent(
        timestamp=timestamp,
        event_type=event_type,
        payload=payload or {},
    )


def _config(tmp_path: Path, *, enabled: bool = True) -> AuditConfig:
    return AuditConfig(file_path=str(tmp_path / "test_audit.jsonl"), enabled=enabled)


# ---------------------------------------------------------------------------
# AuditEvent schema
# ---------------------------------------------------------------------------


class TestAuditEventSchema:
    def test_audit_event_has_correct_schema(self):
        evt = _make_event(
            event_type=AuditEventType.NODE_CREATED,
            timestamp=1234.5,
            payload={"node_id": "abc"},
        )
        assert evt.event_type == AuditEventType.NODE_CREATED
        assert evt.timestamp == 1234.5
        assert evt.payload == {"node_id": "abc"}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


class TestAuditLogWrite:
    @pytest.mark.asyncio
    async def test_log_event_writes_jsonl_line(self, tmp_path: Path):
        logger = AuditLogger(_config(tmp_path))
        await logger.log(_make_event())

        lines = Path(logger.config.file_path).read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "SEND_MEMORY"

    @pytest.mark.asyncio
    async def test_log_event_has_timestamp(self, tmp_path: Path):
        logger = AuditLogger(_config(tmp_path))
        await logger.log(_make_event(timestamp=9999.0))

        lines = Path(logger.config.file_path).read_text().strip().splitlines()
        data = json.loads(lines[0])
        assert data["timestamp"] == 9999.0

    @pytest.mark.asyncio
    async def test_multiple_events_append(self, tmp_path: Path):
        logger = AuditLogger(_config(tmp_path))
        await logger.log(_make_event(timestamp=1.0))
        await logger.log(_make_event(timestamp=2.0))
        await logger.log(_make_event(timestamp=3.0))

        lines = Path(logger.config.file_path).read_text().strip().splitlines()
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_audit_file_created_on_first_write(self, tmp_path: Path):
        cfg = _config(tmp_path)
        assert not Path(cfg.file_path).exists()

        logger = AuditLogger(cfg)
        await logger.log(_make_event())
        assert Path(cfg.file_path).exists()

    @pytest.mark.asyncio
    async def test_disabled_audit_does_not_write(self, tmp_path: Path):
        cfg = _config(tmp_path, enabled=False)
        logger = AuditLogger(cfg)
        await logger.log(_make_event())

        assert not Path(cfg.file_path).exists()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


class TestAuditLogRead:
    @pytest.mark.asyncio
    async def test_read_events_returns_all(self, tmp_path: Path):
        logger = AuditLogger(_config(tmp_path))
        await logger.log(_make_event(timestamp=1.0))
        await logger.log(_make_event(timestamp=2.0))

        events = await logger.read_events()
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_read_events_filter_by_type(self, tmp_path: Path):
        logger = AuditLogger(_config(tmp_path))
        await logger.log(
            _make_event(event_type=AuditEventType.SEND_MEMORY, timestamp=1.0)
        )
        await logger.log(
            _make_event(event_type=AuditEventType.GET_MEMORY, timestamp=2.0)
        )
        await logger.log(
            _make_event(event_type=AuditEventType.SEND_MEMORY, timestamp=3.0)
        )

        events = await logger.read_events(event_type=AuditEventType.SEND_MEMORY)
        assert len(events) == 2
        assert all(e.event_type == AuditEventType.SEND_MEMORY for e in events)

    @pytest.mark.asyncio
    async def test_read_events_filter_by_since(self, tmp_path: Path):
        logger = AuditLogger(_config(tmp_path))
        await logger.log(_make_event(timestamp=100.0))
        await logger.log(_make_event(timestamp=200.0))
        await logger.log(_make_event(timestamp=300.0))

        events = await logger.read_events(since=200.0)
        assert len(events) == 2
        assert events[0].timestamp == 200.0
        assert events[1].timestamp == 300.0

    @pytest.mark.asyncio
    async def test_read_events_empty_when_no_file(self, tmp_path: Path):
        logger = AuditLogger(_config(tmp_path))
        events = await logger.read_events()
        assert events == []
