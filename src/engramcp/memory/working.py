"""Redis-backed working memory buffer.

Fragments are stored as JSON strings keyed by ``engramcp:fragment:{id}``.
A sorted set ``engramcp:recency`` tracks insertion order (score = timestamp).
Sets ``engramcp:keyword:{word}`` enable simple keyword search.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Awaitable
from collections.abc import Callable

from pydantic import BaseModel
from pydantic import Field
from redis.asyncio import Redis  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Key prefixes
# ---------------------------------------------------------------------------

_PREFIX = "engramcp"
_FRAGMENT_KEY = f"{_PREFIX}:fragment"
_RECENCY_KEY = f"{_PREFIX}:recency"
_KEYWORD_KEY = f"{_PREFIX}:keyword"

# ---------------------------------------------------------------------------
# Confidence helpers (moved from server.py)
# ---------------------------------------------------------------------------

_RELIABILITY_ORDER = "ABCDEF"


def _confidence_passes(confidence: str | None, min_confidence: str) -> bool:
    """Check whether *confidence* meets or exceeds *min_confidence*.

    The NATO rating is ``<letter><number>`` (e.g. ``B2``).
    Letter A is best (index 0), F is worst (index 5).
    Number 1 is best, 6 is worst.

    A memory passes the filter when its letter is <= the filter letter
    **or** its number is <= the filter number.  The loosest filter is
    ``F6`` which lets everything through.
    """
    if min_confidence == "F6":
        return True
    if not confidence or len(confidence) < 2:
        return False

    mem_letter = confidence[0].upper()
    mem_number = confidence[1:]

    flt_letter = min_confidence[0].upper()
    flt_number = min_confidence[1:]

    try:
        letter_ok = _RELIABILITY_ORDER.index(mem_letter) <= _RELIABILITY_ORDER.index(
            flt_letter
        )
        number_ok = int(mem_number) <= int(flt_number)
    except (ValueError, IndexError):
        return False

    return letter_ok or number_ok


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens from *text*."""
    return set(_WORD_RE.findall(text.lower()))


# ---------------------------------------------------------------------------
# MemoryFragment
# ---------------------------------------------------------------------------


class MemoryFragment(BaseModel):
    """A single memory held in working memory."""

    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    content: str
    type: str = "Fact"
    dynamic_type: str | None = None
    confidence: str | None = None
    properties: dict = Field(default_factory=dict)
    participants: list = Field(default_factory=list)
    causal_chain: list = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    agent_id: str | None = None
    agent_fingerprint: str | None = None
    timestamp: float = Field(default_factory=lambda: __import__("time").time())


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------


class WorkingMemory:
    """Redis-backed working memory buffer with TTL, size limits, and search."""

    def __init__(
        self,
        redis: Redis,
        *,
        ttl: int = 3600,
        max_size: int = 1000,
        flush_threshold: int | None = None,
        on_flush: Callable[[list[MemoryFragment]], Awaitable[None]] | None = None,
    ) -> None:
        self._redis = redis
        self._ttl = ttl
        self._max_size = max_size
        self._flush_threshold = flush_threshold
        self._on_flush = on_flush

    # -- write --

    async def store(self, fragment: MemoryFragment) -> str:
        """Store a fragment and return its ID."""
        key = f"{_FRAGMENT_KEY}:{fragment.id}"
        data = fragment.model_dump_json()

        pipe = self._redis.pipeline()
        pipe.set(key, data, ex=self._ttl)
        pipe.zadd(_RECENCY_KEY, {fragment.id: fragment.timestamp})

        # Keyword index
        for word in _tokenize(fragment.content):
            pipe.sadd(f"{_KEYWORD_KEY}:{word}", fragment.id)
            pipe.expire(f"{_KEYWORD_KEY}:{word}", self._ttl)

        await pipe.execute()

        # Evict oldest if over max_size
        await self._evict_if_needed()

        # Flush trigger
        if self._flush_threshold and self._on_flush:
            current = await self.count()
            if current >= self._flush_threshold:
                fragments = await self.get_recent(limit=current)
                await self._on_flush(fragments)

        return fragment.id

    # -- read --

    async def get(self, fragment_id: str) -> MemoryFragment | None:
        """Retrieve a fragment by ID, or ``None`` if expired/missing."""
        data = await self._redis.get(f"{_FRAGMENT_KEY}:{fragment_id}")
        if data is None:
            return None
        return MemoryFragment.model_validate_json(data)

    async def exists(self, fragment_id: str) -> bool:
        """Check if a fragment exists (has not expired)."""
        return bool(await self._redis.exists(f"{_FRAGMENT_KEY}:{fragment_id}"))

    async def delete(self, fragment_id: str) -> None:
        """Delete a fragment and remove it from indexes."""
        fragment = await self.get(fragment_id)
        pipe = self._redis.pipeline()
        pipe.delete(f"{_FRAGMENT_KEY}:{fragment_id}")
        pipe.zrem(_RECENCY_KEY, fragment_id)
        if fragment:
            for word in _tokenize(fragment.content):
                pipe.srem(f"{_KEYWORD_KEY}:{word}", fragment_id)
        await pipe.execute()

    async def search(
        self,
        query: str,
        *,
        min_confidence: str = "F6",
        limit: int = 20,
    ) -> list[MemoryFragment]:
        """Search fragments by keyword with confidence filtering."""
        words = _tokenize(query)
        if not words:
            return []

        # Union of keyword sets
        keys = [f"{_KEYWORD_KEY}:{w}" for w in words]
        candidate_ids: set[bytes | str] = set()
        for k in keys:
            members = await self._redis.smembers(k)
            candidate_ids.update(members)

        if not candidate_ids:
            return []

        # Fetch and filter
        results: list[MemoryFragment] = []
        for cid in candidate_ids:
            fid = cid.decode() if isinstance(cid, bytes) else cid
            fragment = await self.get(fid)
            if fragment and _confidence_passes(fragment.confidence, min_confidence):
                results.append(fragment)

        # Sort by recency (newest first)
        results.sort(key=lambda f: f.timestamp, reverse=True)
        return results[:limit]

    async def get_recent(self, limit: int = 20) -> list[MemoryFragment]:
        """Return the most recent fragments, newest first."""
        # ZREVRANGE returns highest scores first
        ids = await self._redis.zrevrange(_RECENCY_KEY, 0, limit - 1)

        results: list[MemoryFragment] = []
        for raw_id in ids:
            fid = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            fragment = await self.get(fid)
            if fragment:
                results.append(fragment)
        return results

    async def count(self) -> int:
        """Return the number of live fragments."""
        return await self._redis.zcard(_RECENCY_KEY)

    async def clear(self) -> None:
        """Remove all fragments and indexes."""
        # Scan for all engramcp:* keys
        keys: list = []
        async for key in self._redis.scan_iter(match=f"{_PREFIX}:*"):
            keys.append(key)
        if keys:
            await self._redis.delete(*keys)

    # -- internal --

    async def _evict_if_needed(self) -> None:
        """Evict oldest fragments if count exceeds max_size."""
        current = await self.count()
        if current <= self._max_size:
            return

        excess = current - self._max_size
        # Get the oldest fragment IDs (lowest scores)
        oldest = await self._redis.zrange(_RECENCY_KEY, 0, excess - 1)
        for raw_id in oldest:
            fid = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            await self.delete(fid)
