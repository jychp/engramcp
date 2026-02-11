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

    async def test_store_regenerates_id_on_collision(self, wm):
        first = _make_fragment("collision-one")
        duplicate = _make_fragment("collision-two", id=first.id)

        first_id = await wm.store(first)
        score_before = await wm._redis.zscore("engramcp:recency", first_id)
        second_id = await wm.store(duplicate)
        score_after = await wm._redis.zscore("engramcp:recency", first_id)

        assert first_id == first.id
        assert second_id != first_id
        assert await wm.exists(first_id)
        assert await wm.exists(second_id)
        assert score_before == score_after

    async def test_store_rolls_back_fragment_if_index_pipeline_fails(
        self, wm, monkeypatch
    ):
        fragment = _make_fragment("rollback case with keywords")
        key = f"engramcp:fragment:{fragment.id}"
        recency_key = "engramcp:recency"
        frag_kw_key = f"engramcp:frag_kw:{fragment.id}"
        keywords = {"rollback", "case", "with", "keywords"}

        class _FailingPipeline:
            def __init__(self, redis):
                self._redis = redis
                self._ops: list[tuple[str, tuple, dict]] = []

            def zadd(self, *_args, **_kwargs):
                self._ops.append(("zadd", _args, _kwargs))
                return self

            def set(self, *_args, **_kwargs):
                self._ops.append(("set", _args, _kwargs))
                return self

            def sadd(self, *_args, **_kwargs):
                self._ops.append(("sadd", _args, _kwargs))
                return self

            def expire(self, *_args, **_kwargs):
                self._ops.append(("expire", _args, _kwargs))
                return self

            async def execute(self):
                # Simulate an error after some writes were already applied.
                for idx, (name, args, kwargs) in enumerate(self._ops):
                    if idx >= 3:
                        break
                    await getattr(self._redis, name)(*args, **kwargs)
                raise RuntimeError("pipeline boom")

        monkeypatch.setattr(wm._redis, "pipeline", lambda: _FailingPipeline(wm._redis))

        with pytest.raises(RuntimeError, match="pipeline boom"):
            await wm.store(fragment)

        assert await wm._redis.get(key) is None
        assert await wm._redis.zscore(recency_key, fragment.id) is None
        assert await wm._redis.get(frag_kw_key) is None
        for word in keywords:
            assert not await wm._redis.sismember(f"engramcp:keyword:{word}", fragment.id)
        assert await wm.count() == 0


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
        await asyncio.sleep(0.05)
        assert len(flushed) == 1
        assert len(flushed[0]) == 3

    async def test_flush_trigger_is_non_blocking(self, redis_client):
        async def on_flush(_: list[MemoryFragment]) -> None:
            await asyncio.sleep(0.2)

        wm = WorkingMemory(
            redis_client,
            ttl=3600,
            max_size=1000,
            flush_threshold=1,
            on_flush=on_flush,
        )

        start = asyncio.get_running_loop().time()
        await wm.store(_make_fragment("non blocking trigger"))
        elapsed = asyncio.get_running_loop().time() - start

        assert elapsed < 0.1

    async def test_flush_retriggers_after_pending_threshold(self, redis_client):
        call_count = 0

        async def on_flush(_: list[MemoryFragment]) -> None:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)

        wm = WorkingMemory(
            redis_client,
            ttl=3600,
            max_size=1000,
            flush_threshold=2,
            on_flush=on_flush,
        )

        # First two fragments trigger a flush run.
        await wm.store(_make_fragment("fact 1"))
        await wm.store(_make_fragment("fact 2"))
        # During the first run, these stores should mark a pending flush.
        await wm.store(_make_fragment("fact 3"))
        await wm.store(_make_fragment("fact 4"))

        await asyncio.sleep(0.35)
        assert call_count >= 2

    async def test_flush_failure_does_not_leave_dangling_task(self, redis_client, monkeypatch):
        async def on_flush(_: list[MemoryFragment]) -> None:
            return None

        wm = WorkingMemory(
            redis_client,
            ttl=3600,
            max_size=1000,
            flush_threshold=1,
            on_flush=on_flush,
        )
        async def fail_get_recent(*, limit: int = 20) -> list[MemoryFragment]:
            del limit
            raise RuntimeError("boom")

        monkeypatch.setattr(wm, "get_recent", fail_get_recent)
        await wm.store(_make_fragment("flush boom"))
        await asyncio.sleep(0.05)
        assert wm._flush_task is None

    async def test_close_cancels_running_flush_task(self, redis_client):
        started = asyncio.Event()

        async def on_flush(_: list[MemoryFragment]) -> None:
            started.set()
            await asyncio.sleep(1.0)

        wm = WorkingMemory(
            redis_client,
            ttl=3600,
            max_size=1000,
            flush_threshold=1,
            on_flush=on_flush,
        )
        await wm.store(_make_fragment("slow flush"))
        await asyncio.wait_for(started.wait(), timeout=0.2)
        await wm.close()
        assert wm._flush_task is None

    async def test_evict_cleans_keyword_index_without_frag_kw(self, redis_client):
        wm = WorkingMemory(redis_client, ttl=3600, max_size=1)
        first = _make_fragment("Alpha marker")
        second = _make_fragment("Beta marker")

        await wm.store(first)
        # Simulate missing per-fragment keyword cache.
        await redis_client.delete(f"engramcp:frag_kw:{first.id}")

        await wm.store(second)  # Triggers eviction of the first fragment.

        alpha_members = await redis_client.smembers("engramcp:keyword:alpha")
        decoded = {
            member.decode() if isinstance(member, bytes) else member
            for member in alpha_members
        }
        assert first.id not in decoded


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
