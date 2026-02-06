"""Working memory unit tests.

Tests exercise ``WorkingMemory`` directly (no MCP client) with a real
Redis container.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from engramcp.memory import MemoryFragment
from engramcp.memory import WorkingMemory
from engramcp.models import agent_fingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fragment(content: str = "test fact", **kwargs) -> MemoryFragment:
    """Build a MemoryFragment with sensible defaults."""
    defaults: dict = {
        "content": content,
        "type": "Fact",
        "confidence": "B3",
        "timestamp": time.time(),
    }
    defaults.update(kwargs)
    return MemoryFragment(**defaults)


@pytest.fixture()
def wm(redis_client) -> WorkingMemory:
    """Return a WorkingMemory wired to the test Redis."""
    return WorkingMemory(redis_client, ttl=3600, max_size=1000)


# -----------------------------------------------------------------------
# TestWrite
# -----------------------------------------------------------------------


class TestWrite:
    """Writing fragments to working memory."""

    async def test_store_fragment(self, wm):
        fragment = _make_fragment("A met B on March 15")
        fid = await wm.store(fragment)
        assert fid == fragment.id
        assert await wm.exists(fid)

    async def test_fragment_has_timestamp(self, wm):
        before = time.time()
        fragment = _make_fragment()
        await wm.store(fragment)
        retrieved = await wm.get(fragment.id)
        assert retrieved is not None
        assert retrieved.timestamp >= before

    async def test_fragment_has_id(self, wm):
        fragment = _make_fragment()
        assert fragment.id  # auto-generated
        assert isinstance(fragment.id, str)
        assert len(fragment.id) > 0

    async def test_fragment_preserves_source_info(self, wm):
        fragment = _make_fragment(
            sources=[
                {
                    "id": "src_abc",
                    "type": "court_document",
                    "ref": "https://example.com/doc.pdf",
                    "citation": "page 3",
                    "reliability": "B",
                    "credibility": "2",
                }
            ]
        )
        await wm.store(fragment)
        retrieved = await wm.get(fragment.id)
        assert retrieved is not None
        assert len(retrieved.sources) == 1
        assert retrieved.sources[0]["type"] == "court_document"

    async def test_fragment_carries_agent_fingerprint(self, wm):
        fp = agent_fingerprint("analyst_1")
        fragment = _make_fragment(agent_id="analyst_1", agent_fingerprint=fp)
        await wm.store(fragment)
        retrieved = await wm.get(fragment.id)
        assert retrieved is not None
        assert retrieved.agent_fingerprint == fp


# -----------------------------------------------------------------------
# TestRead
# -----------------------------------------------------------------------


class TestRead:
    """Reading and searching working memory."""

    async def test_retrieve_recent_fragments(self, wm):
        for i in range(5):
            await wm.store(_make_fragment(f"fact number {i}"))
        recent = await wm.get_recent(limit=3)
        assert len(recent) == 3

    async def test_retrieve_by_keyword(self, wm):
        await wm.store(_make_fragment("Alice traveled to Paris"))
        await wm.store(_make_fragment("Bob stayed in London"))
        results = await wm.search("Paris")
        assert len(results) == 1
        assert "Paris" in results[0].content

    async def test_empty_on_no_match(self, wm):
        await wm.store(_make_fragment("Alice traveled to Paris"))
        results = await wm.search("nonexistent_query_xyz")
        assert results == []

    async def test_ordered_by_recency(self, wm):
        f1 = _make_fragment("old fact", timestamp=1000.0)
        f2 = _make_fragment("new fact", timestamp=2000.0)
        await wm.store(f1)
        await wm.store(f2)
        recent = await wm.get_recent(limit=10)
        assert recent[0].content == "new fact"
        assert recent[1].content == "old fact"


# -----------------------------------------------------------------------
# TestLifecycle
# -----------------------------------------------------------------------


class TestLifecycle:
    """TTL, size limits, and flush triggers."""

    async def test_ttl_expiration(self, redis_client):
        wm = WorkingMemory(redis_client, ttl=1, max_size=1000)
        fragment = _make_fragment()
        await wm.store(fragment)
        assert await wm.exists(fragment.id)
        # Wait for TTL to expire
        await asyncio.sleep(1.5)
        assert not await wm.exists(fragment.id)

    async def test_buffer_size_limit(self, redis_client):
        wm = WorkingMemory(redis_client, ttl=3600, max_size=5)
        for i in range(10):
            await wm.store(_make_fragment(f"fact {i}"))
        count = await wm.count()
        assert count <= 5

    async def test_flush_trigger_on_threshold(self, redis_client):
        flushed: list[list[MemoryFragment]] = []

        async def on_flush(fragments: list[MemoryFragment]) -> None:
            flushed.append(fragments)

        wm = WorkingMemory(
            redis_client,
            ttl=3600,
            max_size=1000,
            flush_threshold=3,
            on_flush=on_flush,
        )
        for i in range(3):
            await wm.store(_make_fragment(f"fact {i}"))
        assert len(flushed) == 1
        assert len(flushed[0]) == 3


# -----------------------------------------------------------------------
# TestAgentIdentity
# -----------------------------------------------------------------------


class TestAgentIdentity:
    """Agent fingerprinting on fragments."""

    async def test_agent_fingerprint_generated(self):
        fp = agent_fingerprint("analyst_1")
        assert fp is not None
        assert isinstance(fp, str)
        assert len(fp) == 16

    async def test_same_agent_same_fingerprint(self):
        fp1 = agent_fingerprint("analyst_1")
        fp2 = agent_fingerprint("analyst_1")
        assert fp1 == fp2

    async def test_fingerprint_stored_on_fragment(self, wm):
        fp = agent_fingerprint("analyst_1")
        fragment = _make_fragment(agent_id="analyst_1", agent_fingerprint=fp)
        await wm.store(fragment)
        retrieved = await wm.get(fragment.id)
        assert retrieved is not None
        assert retrieved.agent_fingerprint == fp
        assert retrieved.agent_id == "analyst_1"
