# Graph Store — Layer 2 Design

## Overview

The graph store is the persistence layer for the EngraMCP knowledge graph.
It provides async CRUD operations on Neo4j, operating directly on the
Pydantic models defined in Layer 0 (`models/`).

## Architecture

```
GraphStore (graph/store.py)
    ├── create_node(node: MemoryNode) -> str
    ├── get_node(node_id) -> MemoryNode | None
    ├── update_node(node_id, **updates) -> MemoryNode | None
    ├── delete_node(node_id) -> bool
    ├── create_relationship(from_id, to_id, rel) -> bool
    ├── get_relationships(node_id, rel_type?, direction?) -> list[dict]
    └── Query methods (find_by_id, find_facts_by_agent, etc.)

init_schema (graph/schema.py)
    └── Creates indexes + constraints (idempotent, IF NOT EXISTS)
```

## Design Decisions

### D1. No graph-specific schemas

The graph store operates directly on `models/nodes.py` and
`models/relations.py` types — no separate `graph/schemas.py`. This follows
the same pattern as `memory/store.py` operating on `memory/schemas.py`.

### D2. Hybrid generic + typed API

- **Generic methods**: `create_node()` and `create_relationship()` dispatch
  via `node.node_labels` and `rel.rel_type` properties, so one method
  handles all 11 node types and 17 relationship types.
- **Typed query methods**: Each query has distinct Cypher optimized for its
  use case (temporal range, alias search, contradiction lookup, etc.).

### D3. Neo4j Community Edition constraints

Only the uniqueness constraint (`Memory.id IS UNIQUE`) is enforced at the
database level. NOT NULL constraints are enforced by Pydantic validation
at the application level, since Community Edition does not support
existence constraints.

### D4. Neo4j datetime handling

The Neo4j driver v5 returns `neo4j.time.DateTime` objects instead of
Python `datetime`. The `_convert_props()` helper converts these back to
Python types using `.to_native()` before Pydantic validation.

Python `datetime` objects are passed directly to the driver on write — the
driver handles conversion natively.

### D5. Agent alias search

No special index on the `aliases` array property (Neo4j doesn't support
indexing array properties). Uses the `ANY()` list predicate:

```cypher
WHERE a.name = $alias OR ANY(x IN a.aliases WHERE x = $alias)
```

Acceptable performance at current scale.

## Layer 0 Models

### Confidence (`models/confidence.py`)

- `Reliability` enum: A-F (source reliability)
- `Credibility` enum: 1-6 (information credibility)
- `NATORating`: frozen value object with `__str__`, `from_str()`, and
  `is_better_or_equal()` comparison

### Nodes (`models/nodes.py`)

11 node types organized in 4 categories:

| Category | Types | Labels |
|---|---|---|
| Knowledge | Fact, Event, Observation, Decision, Outcome | Memory[+Temporal] |
| Entity | Agent, Artifact | Memory |
| Source | Source | Memory |
| Derived | Pattern, Concept, Rule | Memory+Derived |

Each model has a `node_labels` property and a `LABEL_TO_MODEL` mapping
for deserialization.

### Relations (`models/relations.py`)

17 relationship types across 8 categories:

| Category | Types |
|---|---|
| Traceability | SourcedFrom, DerivedFrom, Cites |
| Causal | CausedBy, LeadsTo |
| Temporal | Preceded, Followed |
| Epistemic | Supports, Contradicts |
| Participation | ParticipatedIn, DecidedBy, ObservedBy |
| Reference | Mentions, Concerns |
| Abstraction | Generalizes, InstanceOf |
| Entity Resolution | PossiblySameAs |

## Schema (`graph/schema.py`)

The `init_schema()` function creates:
- 1 uniqueness constraint
- 12 node property indexes
- 6 relationship property indexes
- 1 full-text index on content fields

All use `IF NOT EXISTS` for idempotency.

## Testing

| Suite | Tests | Container |
|---|---|---|
| test_models.py (Layer 0) | 38 | None |
| test_graph_store.py — Schema | 3 | Neo4j |
| test_graph_store.py — Nodes | 15 | Neo4j |
| test_graph_store.py — Relationships | 9 | Neo4j |
| test_graph_store.py — Queries | 7 | Neo4j |
