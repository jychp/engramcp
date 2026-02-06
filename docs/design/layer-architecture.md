# Layer Architecture — Design Document

> **Status**: Active
> **Scope**: Full system decomposition

---

## Overview

EngraMCP is decomposed into 8 testable layers, numbered 0 (innermost, data) to 7 (outermost, user-facing). Development follows a **top-down (outside-in) TDD** approach: we start from the MCP interface and work our way down, replacing mocks with real implementations layer by layer.

---

## Layer Map

```
Layer 7 — MCP Interface          server.py
Layer 6 — Retrieval Engine        engine/retrieval.py
Layer 5 — Concept Emergence       engine/concepts.py, engine/demand.py
Layer 4 — Consolidation Engine    engine/consolidation.py, engine/extraction.py
Layer 3 — Confidence Engine       engine/confidence.py
Layer 2 — Graph Store             graph/store.py, graph/schema.py
Layer 1 — Working Memory          memory/working.py, memory/persistence.py
Layer 0 — Graph Model             models/
```

---

## Layers (top-down)

### Layer 7 — MCP Interface

**Module**: `server.py`
**Depends on**: Layer 6 (Retrieval), Layer 1 (Working Memory), Layer 3 (Confidence)

The only entry point for agents. Exposes three tools via FastMCP v2:
- `send_memory` — write a memory into the system
- `get_memory` — retrieve relevant memories with causal reasoning
- `correct_memory` — correct, annotate, merge/split, reclassify

See [MCP Interface design doc](mcp-interface.md) for details.

### Layer 6 — Retrieval Engine

**Module**: `engine/retrieval.py`
**Depends on**: Layer 1 (Working Memory), Layer 2 (Graph Store), Layer 5 (Concepts)

Intelligent graph traversal with causal reasoning:
1. Working Memory check (hot path, recent fragments)
2. Graph traversal (causal chains, contradiction detection)
3. Multi-factor scoring (recency, causal proximity, confidence, access frequency)
4. Structured synthesis (DD2-shaped response)

Also tracks query patterns for concept emergence (feeds Layer 5).

### Layer 5 — Concept Emergence

**Modules**: `engine/concepts.py`, `engine/demand.py`
**Depends on**: Layer 4 (Consolidation), Layer 2 (Graph Store)

Domain concepts emerge from **retrieval demand**, not from extraction volume. The system only creates new types when users actually query for them.

- `demand.py` — query pattern tracker, demand signal emitter
- `concepts.py` — concept registry, stabilization, dynamic Neo4j labels

See [DD3 — Concept Emergence](../drafts/deep-dives.md#dd3) for the full mechanism.

### Layer 4 — Consolidation Engine

**Modules**: `engine/consolidation.py`, `engine/extraction.py`, `engine/abstraction.py`
**Depends on**: Layer 2 (Graph Store), Layer 3 (Confidence)

Async batch pipeline triggered when Working Memory buffer reaches a threshold:
1. **Extract** — LLM extracts entities, relations, claims from raw fragments
2. **Integrate** — merge into Knowledge Graph with entity resolution
3. **Abstract** — discover Patterns → Concepts → Rules

Entity resolution uses a three-level strategy. See [DD1 — Entity Resolution](../drafts/deep-dives.md#dd1).

### Layer 3 — Confidence Engine

**Module**: `engine/confidence.py`
**Depends on**: Layer 2 (Graph Store)

NATO two-dimensional rating system:
- **Letter** (A-F): source reliability, provided by the calling agent
- **Number** (1-6): information credibility, computed by the engine

Key behaviors:
- Upward propagation: derived nodes inherit worst letter, computed number with decay
- Downward propagation: corrections cascade through the derivation chain
- Corroboration requires genuinely independent sources (common ancestor detection)

### Layer 2 — Graph Store

**Modules**: `graph/store.py`, `graph/schema.py`, `graph/entity_resolution.py`, `graph/traceability.py`
**Depends on**: Layer 0 (Graph Model), Neo4j

Neo4j CRUD operations matching the ontology schema:
- Node creation/retrieval for all base and derived types
- Relationship management with confidence ratings
- Schema initialization (indexes, constraints)
- Entity resolution engine
- Source chain management and traceability

### Layer 1 — Working Memory

**Modules**: `memory/working.py`, `memory/persistence.py`
**Depends on**: none (standalone)

In-memory short-term buffer:
- Fast storage for raw `MemoryFragment` objects
- TTL-based expiration
- Buffer size limits
- Flush-to-disk for persistence across restarts
- Flush trigger on fragment count threshold (feeds Layer 4)

### Layer 0 — Graph Model

**Modules**: `models/`
**Depends on**: none (standalone)

Data definitions:
- `models/mcp.py` — Pydantic input/output schemas for MCP tools
- `models/nodes.py` — node type definitions (base + derived)
- `models/relations.py` — relationship type definitions
- `models/confidence.py` — NATO rating model
- `models/agent.py` — agent fingerprinting

---

## Data Flow

### Write Path

```
Agent → send_memory → [Layer 7]
    → store in Working Memory [Layer 1]
    → (threshold reached) → trigger Consolidation [Layer 4]
        → LLM extraction [Layer 4]
        → entity resolution [Layer 2]
        → graph integration [Layer 2]
        → confidence calculation [Layer 3]
        → abstraction (Pattern/Concept/Rule) [Layer 4]
```

### Read Path

```
Agent → get_memory → [Layer 7]
    → check Working Memory [Layer 1] (hot path)
    → graph traversal [Layer 6]
        → causal chain following [Layer 2]
        → contradiction detection [Layer 2]
        → dynamic label optimization [Layer 5]
    → multi-factor scoring [Layer 6]
    → structured synthesis [Layer 6]
    → query pattern logging [Layer 5]
```

### Correction Path

```
Agent → correct_memory → [Layer 7]
    → validate target exists [Layer 1 / Layer 2]
    → apply correction [Layer 2]
    → cascade confidence recalculation [Layer 3]
    → audit log [audit/logger.py]
```

---

## Mock Replacement Strategy

Upper layers are tested with mocks that get progressively replaced as lower layers are implemented:

| Layer | Initially mocked by | Replaced when |
|-------|-------------------|---------------|
| Working Memory (1) | In-memory dict in `server.py` | Layer 1 implemented |
| Graph Store (2) | Not needed until Layer 2 | Layer 2 implemented |
| Confidence Engine (3) | Static ratings | Layer 3 implemented |
| Consolidation LLM (4) | Deterministic JSON responses | Layer 4 implemented |
| Retrieval Engine (6) | Keyword matching in `server.py` | Layer 6 implemented |

---

## Design Principles

- **Outside-in TDD**: tests written before implementation at each layer
- **Frozen interfaces**: upper layer contracts don't change when lower layers are implemented
- **DDD boundaries**: each module directory (`models/`, `memory/`, `graph/`, `engine/`, `audit/`) is a bounded context
- **No circular dependencies**: layers only depend on layers below them (lower number)
- **Confidence on relations, not nodes**: the same fact can have different ratings from different sources
