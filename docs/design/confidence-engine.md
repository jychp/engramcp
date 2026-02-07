# Confidence Engine — Layer 3 Design

> Source traceability, independence detection, NATO confidence assessment, and propagation.

---

## Overview

The confidence engine implements **Layer 3** of the EngraMCP architecture. It provides:

1. **Source Traceability** — Follow citation chains (`CITES`) from sources to their roots
2. **Independence Detection** — Determine whether two sources share a common ancestor
3. **Credibility Assessment** — NATO Dimension 2 scoring based on corroboration and contradiction
4. **Upward Propagation** — Compute confidence for derived nodes (Pattern → Concept → Rule)
5. **Downward Cascade** — Propagate contests/retractions through the derivation hierarchy

---

## Architecture

```
ConfidenceEngine(graph_store, traceability, config)
        │                    │
        │         SourceTraceability(driver)
        │                    │
        └────────────────────┘
                    │
              GraphStore(driver)  ← existing, unchanged
                    │
                AsyncDriver ← existing Neo4j connection
```

- `SourceTraceability` takes `AsyncDriver` directly (needs variable-length Cypher paths)
- `ConfidenceEngine` takes `GraphStore` + `SourceTraceability` + optional `ConfidenceConfig`
- No changes to `GraphStore` required

---

## Module: `graph/traceability.py`

### Data Classes

```python
@dataclass(frozen=True)
class SourceChain:
    sources: list[Source]   # Ordered from immediate source to root
    root: Source            # The terminal source (no outgoing CITES)

@dataclass(frozen=True)
class IndependenceResult:
    independent: bool
    common_ancestor: str | None   # Source ID if not independent
    reason: str                   # Human-readable explanation
```

### Class: `SourceTraceability`

| Method | Description |
|---|---|
| `get_citation_chain(source_id, max_depth=10)` | Follow `CITES` chain to root using variable-length Cypher path |
| `find_root_source(source_id)` | Return the terminal source in the citation chain |
| `check_independence(source_a_id, source_b_id)` | Trace both chains, check for common ancestors. Conservative: unknown → not independent |
| `trace_fact_to_sources(fact_id)` | All `Source` nodes linked via `SOURCED_FROM` |
| `trace_fact_to_root_sources(fact_id)` | `Fact → SOURCED_FROM → Source → CITES* → root` |

### Cypher Patterns

Citation chain traversal uses variable-length paths:
```cypher
MATCH p = (s:Source {id: $id})-[:CITES*0..10]->(root:Source)
WHERE NOT (root)-[:CITES]->()
RETURN nodes(p) AS chain
ORDER BY length(p) DESC
LIMIT 1
```

Independence check uses common-ancestor detection:
```cypher
MATCH (a:Source {id: $a_id})-[:CITES*0..10]->(ancestor:Source),
      (b:Source {id: $b_id})-[:CITES*0..10]->(ancestor)
RETURN ancestor.id AS common_ancestor
LIMIT 1
```

---

## Module: `engine/confidence.py`

### Configuration

```python
@dataclass(frozen=True)
class ConfidenceConfig:
    default_reliability: Reliability = Reliability.F
    initial_credibility: Credibility = Credibility.THREE
    min_independent_sources: int = 2
    depth_decay_steps: int = 1
```

### Result Types

```python
@dataclass(frozen=True)
class CredibilityAssessment:
    credibility: Credibility
    supporting_count: int
    contradicting_count: int
    reason: str

@dataclass(frozen=True)
class PropagatedRating:
    reliability: Reliability
    credibility: Credibility
    source_count: int
    reason: str

@dataclass(frozen=True)
class CascadeResult:
    affected_nodes: list[str]    # IDs of nodes whose status changed
    reason: str
```

### Class: `ConfidenceEngine`

#### Dimension 1 — Reliability

| Method | Description |
|---|---|
| `reliability_from_hint(confidence_hint)` | Parse first letter of hint as `Reliability`; default to `config.default_reliability` |

The reliability letter comes from the source itself, set at ingestion time. The engine only needs to map hints to the enum.

#### Dimension 2 — Credibility

| Method | Description |
|---|---|
| `assess_credibility(fact_id)` | Compute credibility number based on corroboration/contradiction state |

Credibility rules (NATO-inspired):
- **Initial**: `config.initial_credibility` (default 3 = "Possibly true")
- **Corroborated by independent sources**: upgrade toward 1
- **Contradicted**: downgrade toward 5
- **Contradicted by stronger source + improbable**: 5
- **Insufficient data / unknown**: 6

#### Corroboration

| Method | Description |
|---|---|
| `check_corroboration(fact_id)` | Count independent sources via `SourceTraceability`; return `(count, source_ids)` |

A fact is corroborated when it has `>= min_independent_sources` independent sources (via `SOURCED_FROM`). Independence is checked pairwise using `SourceTraceability.check_independence()`.

#### Upward Propagation

| Method | Description |
|---|---|
| `propagate_upward(derived_node_id)` | Compute confidence for Pattern/Concept/Rule from source nodes |

Rules:
- **Reliability**: Worst (highest-index) reliability among all contributing source chains
- **Credibility**: Computed from convergence of evidence, then degraded by `depth_decay_steps` per abstraction layer
- Pattern (depth 1): worst source reliability + convergence credibility + 1 decay step
- Concept (depth 2): inherits from patterns + 1 additional decay step
- Rule (depth 3): inherits from concepts + 1 additional decay step

#### Downward Cascade

| Method | Description |
|---|---|
| `cascade_contest(contested_node_id)` | Propagate contest status through derivation hierarchy |

When a fact is contested:
1. Find all `Pattern` nodes derived from it (`DERIVED_FROM`)
2. Recompute each pattern's support — if remaining facts insufficient, set `status = dissolved`
3. Cascade to `Concept` and `Rule` nodes similarly
4. **Never delete** — only change status; audit trail preserved

---

## Key Design Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | `SourceTraceability` takes `AsyncDriver` directly | Needs `[:CITES*1..N]` variable-length paths not exposed by `GraphStore` |
| D2 | Propagated confidence is computed on demand, not stored | Avoids staleness; derived nodes lack `SOURCED_FROM` relations |
| D3 | Conservative defaults everywhere | Unknown independence → not independent; insufficient data → credibility 6 |
| D4 | Cascade recalculates, never deletes | Dissolved nodes keep audit trail; status changes to `dissolved` |
| D5 | Depth decay applied per abstraction layer | Each layer of derivation degrades credibility by `depth_decay_steps` |

---

## Test Plan

### `test_source_traceability.py` (13 tests, all Neo4j)

| Group | Tests |
|---|---|
| TestSourceCreation (3) | Source with all fields, minimal fields, reliability validation |
| TestSourceChains (3) | Cites relationship, chain depth traversal, find root source |
| TestIndependence (4) | Independent detected, common ancestor detected, transitive citation, conservative default |
| TestTraceability (3) | Fact traces to source, fact traces to citation, full chain to root |

### `test_confidence_engine.py` (24 tests, 2 pure + 22 Neo4j)

| Group | Tests |
|---|---|
| TestDimension1Reliability (3) | From hint, default when no hint, override via correct_memory |
| TestDimension2Credibility (5) | Initial uncorroborated, upgrade on corroboration, downgrade on contradiction, improbable+contradicted, unknown |
| TestCorroboration (3) | Independent boost, non-independent no boost, minimum 2 required |
| TestPropagationUpward (5) | Pattern letter weakest, pattern number from convergence, concept degraded, rule degraded, depth decay |
| TestPropagationDownward (5) | Contest cascades to pattern, dissolves weak pattern, cascades to concept, cascades to rule, recalculates not deletes |
| TestEdgeCases (3) | Single source, all contested, circular support |
