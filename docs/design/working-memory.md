# Working Memory — Layer 1 Design

## Overview

Working memory is the hot buffer that receives all incoming memories via
`send_memory` before they are consolidated into the knowledge graph.  It is
backed by Redis, providing TTL-based expiration, keyword search, and
persistence across restarts.

## Architecture Decision: Redis

The original action plan allowed either an in-process dict with flush-to-disk
or Redis.  Redis was chosen because:

- **TTL-native** — Redis `SET ... EX` handles expiration without custom timers.
- **Persistence** — survives process restarts via RDB/AOF.
- **Shared access** — multiple server instances can share the same buffer.
- **Secondary indexes** — Redis Sets provide efficient keyword lookup.

`memory/persistence.py` (flush-to-disk) from the original plan is dropped.

## Redis Key Structure

```
engramcp:fragment:{id}     → JSON string (SET with EX ttl)
engramcp:recency           → Sorted Set (score=timestamp, member=id)
engramcp:keyword:{word}    → Set of fragment IDs (with EX ttl)
```

## Data Model

### MemoryFragment

```python
class MemoryFragment(BaseModel):
    id: str               # auto-generated "mem_{hex8}"
    content: str
    type: str = "Fact"
    dynamic_type: str | None
    confidence: str | None
    properties: dict
    participants: list
    causal_chain: list
    sources: list[dict]
    agent_id: str | None
    agent_fingerprint: str | None  # SHA-256[:16] of agent_id
    timestamp: float               # time.time()
```

## WorkingMemory API

| Method | Description |
|--------|-------------|
| `store(fragment)` | Write fragment to Redis with TTL, index keywords |
| `get(id)` | Retrieve by ID (or `None` if expired) |
| `exists(id)` | Check if fragment exists |
| `delete(id)` | Remove fragment and clean indexes |
| `search(query, min_confidence, limit)` | Keyword search with confidence filter |
| `get_recent(limit)` | Most recent fragments via sorted set |
| `count()` | Number of live fragments |
| `clear()` | Remove all `engramcp:*` keys |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ttl` | 3600 | Fragment TTL in seconds |
| `max_size` | 1000 | Max fragments before eviction |
| `flush_threshold` | None | Fragment count triggering flush callback |
| `on_flush` | None | Async callback receiving fragments for consolidation |

## Keyword Search

Content is tokenized into lowercase alphanumeric words.  Each word maps to a
Redis Set of fragment IDs (`engramcp:keyword:{word}`).  Search unions all
matching sets, fetches fragments, filters by confidence, and sorts by recency.

No RediSearch module is required — plain Redis data structures suffice for the
expected buffer size (<1000 fragments).

## Agent Fingerprint

`models/__init__.py` provides `agent_fingerprint(agent_id) -> str | None`:
SHA-256 of the agent ID truncated to 16 hex characters.  This deterministic
fingerprint is stored on each fragment for provenance tracking without
exposing the raw agent identifier.

## Size Limits and Eviction

When `store()` pushes the count above `max_size`, the oldest fragments (lowest
scores in the recency sorted set) are evicted via `delete()`.

## Flush Trigger

When `flush_threshold` is set and the count reaches that value, the
`on_flush(fragments)` callback is invoked with all current fragments.  This
hook will be wired to the consolidation engine in a later sprint.

## Test Infrastructure

- Redis 7 Alpine testcontainer (session-scoped, same pattern as Neo4j).
- `FLUSHDB` between tests via autouse fixture.
- 16 unit tests covering write, read, lifecycle, and agent identity.
- 2 integration tests verifying MCP roundtrip with real working memory.
