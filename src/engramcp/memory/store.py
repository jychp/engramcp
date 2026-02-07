"""Redis-backed working memory store.

Fragments are stored as JSON strings keyed by ``engramcp:fragment:{id}``.
A sorted set ``engramcp:recency`` tracks insertion order (score = timestamp).
Sets ``engramcp:keyword:{word}`` enable simple keyword search.
A separate key ``engramcp:frag_kw:{id}`` stores each fragment's keywords
so that ``delete()`` can clean keyword indexes even after TTL expiry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable
from collections.abc import Callable

from redis.asyncio import Redis  # type: ignore[import-untyped]

from engramcp.memory.schemas import MemoryFragment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key prefixes
# ---------------------------------------------------------------------------

_PREFIX = "engramcp"
_FRAGMENT_KEY = f"{_PREFIX}:fragment"
_RECENCY_KEY = f"{_PREFIX}:recency"
_KEYWORD_KEY = f"{_PREFIX}:keyword"
_FRAG_KW_KEY = f"{_PREFIX}:frag_kw"

# ---------------------------------------------------------------------------
# Confidence helpers
# ---------------------------------------------------------------------------

_RELIABILITY_ORDER = "ABCDEF"


def _confidence_passes(confidence: str | None, min_confidence: str) -> bool:
    """Check whether *confidence* meets or exceeds *min_confidence*.

    The NATO rating is ``<letter><number>`` (e.g. ``B2``).
    Letter A is best (index 0), F is worst (index 5).
    Number 1 is best, 6 is worst.

    A memory passes the filter when **both** its letter is <= the filter
    letter **and** its number is <= the filter number.  The loosest filter
    is ``F6`` which lets everything through.
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

    return letter_ok and number_ok


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens from *text*."""
    return set(_WORD_RE.findall(text.lower()))


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------

_CLEAR_BATCH_SIZE = 100


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
        self._flush_lock = asyncio.Lock()

    # -- write --

    async def store(self, fragment: MemoryFragment) -> str:
        """Store a fragment and return its ID.

        Uses a single pipeline for the write, keyword indexing, and
        eviction check.  The flush callback is scheduled as a background
        task and wrapped in try/except for resilience.
        """
        key = f"{_FRAGMENT_KEY}:{fragment.id}"
        data = fragment.model_dump_json()
        keywords = list(_tokenize(fragment.content))

        pipe = self._redis.pipeline()
        pipe.set(key, data, ex=self._ttl)
        pipe.zadd(_RECENCY_KEY, {fragment.id: fragment.timestamp})

        # Store keywords in a separate key for cleanup after TTL expiry
        if keywords:
            kw_key = f"{_FRAG_KW_KEY}:{fragment.id}"
            pipe.set(kw_key, json.dumps(keywords), ex=self._ttl)

        # Keyword index
        for word in keywords:
            pipe.sadd(f"{_KEYWORD_KEY}:{word}", fragment.id)
            pipe.expire(f"{_KEYWORD_KEY}:{word}", self._ttl)

        await pipe.execute()

        # Evict oldest if over max_size
        await self._evict_if_needed()

        # Flush trigger (non-blocking, with error handling)
        if self._flush_threshold and self._on_flush:
            current = await self.count()
            if current >= self._flush_threshold:
                await self._trigger_flush(current)

        return fragment.id

    async def _trigger_flush(self, current: int) -> None:
        """Schedule flush callback as a background task with error handling."""
        if self._flush_lock.locked():
            return  # Another flush is already in progress

        async with self._flush_lock:
            fragments = await self.get_recent(limit=current)
            try:
                assert self._on_flush is not None
                await self._on_flush(fragments)
            except Exception:
                logger.exception("on_flush callback failed")

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
        """Delete a fragment and remove it from all indexes.

        Uses the ``engramcp:frag_kw:{id}`` key to find keywords even
        when the fragment itself has already expired.
        """
        # Try to get keywords from the dedicated key first, fall back to fragment
        kw_data = await self._redis.get(f"{_FRAG_KW_KEY}:{fragment_id}")
        if kw_data is not None:
            keywords: list[str] = json.loads(kw_data)
        else:
            fragment = await self.get(fragment_id)
            keywords = list(_tokenize(fragment.content)) if fragment else []

        pipe = self._redis.pipeline()
        pipe.delete(f"{_FRAGMENT_KEY}:{fragment_id}")
        pipe.delete(f"{_FRAG_KW_KEY}:{fragment_id}")
        pipe.zrem(_RECENCY_KEY, fragment_id)
        for word in keywords:
            pipe.srem(f"{_KEYWORD_KEY}:{word}", fragment_id)
        await pipe.execute()

    async def search(
        self,
        query: str,
        *,
        min_confidence: str = "F6",
    ) -> list[MemoryFragment]:
        """Search fragments by keyword with confidence filtering.

        Returns **all** matching fragments sorted by recency (newest first).
        The caller is responsible for applying a ``limit`` if needed.
        """
        words = _tokenize(query)
        if not words:
            return []

        # Union of keyword sets in a single round-trip
        keys = [f"{_KEYWORD_KEY}:{w}" for w in words]
        candidate_ids: set[bytes | str] = await self._redis.sunion(*keys)

        if not candidate_ids:
            return []

        # Batch-fetch all candidates via pipeline
        decoded_ids = [
            cid.decode() if isinstance(cid, bytes) else cid for cid in candidate_ids
        ]
        pipe = self._redis.pipeline()
        for fid in decoded_ids:
            pipe.get(f"{_FRAGMENT_KEY}:{fid}")
        raw_results = await pipe.execute()

        # Filter and collect stale IDs
        stale_ids: list[str] = []
        results: list[MemoryFragment] = []
        for fid, raw in zip(decoded_ids, raw_results):
            if raw is None:
                stale_ids.append(fid)
            else:
                fragment = MemoryFragment.model_validate_json(raw)
                if _confidence_passes(fragment.confidence, min_confidence):
                    results.append(fragment)

        # Prune stale entries from recency set and keyword sets
        if stale_ids:
            cleanup = self._redis.pipeline()
            for fid in stale_ids:
                cleanup.zrem(_RECENCY_KEY, fid)
                for k in keys:
                    cleanup.srem(k, fid)
            await cleanup.execute()

        # Sort by recency (newest first)
        results.sort(key=lambda f: f.timestamp, reverse=True)
        return results

    async def get_recent(self, limit: int = 20) -> list[MemoryFragment]:
        """Return the most recent fragments, newest first."""
        # ZREVRANGE returns highest scores first
        ids = await self._redis.zrevrange(_RECENCY_KEY, 0, limit - 1)

        if not ids:
            return []

        # Batch-fetch all fragments via pipeline
        decoded_ids = [
            raw_id.decode() if isinstance(raw_id, bytes) else raw_id for raw_id in ids
        ]
        pipe = self._redis.pipeline()
        for fid in decoded_ids:
            pipe.get(f"{_FRAGMENT_KEY}:{fid}")
        raw_results = await pipe.execute()

        stale_ids: list[str] = []
        results: list[MemoryFragment] = []
        for fid, raw in zip(decoded_ids, raw_results):
            if raw is not None:
                results.append(MemoryFragment.model_validate_json(raw))
            else:
                stale_ids.append(fid)

        if stale_ids:
            cleanup = self._redis.pipeline()
            for fid in stale_ids:
                cleanup.zrem(_RECENCY_KEY, fid)
            await cleanup.execute()

        return results

    async def count(self) -> int:
        """Return the number of live fragments."""
        return await self._redis.zcard(_RECENCY_KEY)

    async def clear(self) -> None:
        """Remove all fragments and indexes.

        Deletes in batches to avoid loading all keys into memory at once.
        """
        batch: list = []
        async for key in self._redis.scan_iter(match=f"{_PREFIX}:*"):
            batch.append(key)
            if len(batch) >= _CLEAR_BATCH_SIZE:
                await self._redis.delete(*batch)
                batch.clear()
        if batch:
            await self._redis.delete(*batch)

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
