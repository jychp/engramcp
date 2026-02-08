# Consolidation Pipeline — Layer 4

> **Status:** Implemented
> **Layer:** 4
> **Depends on:** Extraction (Layer 4 partial), Entity Resolution (Layer 4 partial), Graph Store (Layer 0+2), Audit (Layer 3)

---

## Overview

The consolidation pipeline is the **orchestrator** that wires together extraction, entity resolution, graph integration, and audit logging into a single `ConsolidationPipeline.run(fragments)` method. It coordinates existing components — it does not contain complex logic itself.

```
MemoryFragment[] ──> ExtractionEngine.extract()
                          │
                          ▼
                    ExtractionResult
                    (entities, claims, relations)
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         Entity       Claim       Relation
        Resolution   Integration  Integration
              │           │           │
              ▼           ▼           ▼
         GraphStore  GraphStore  GraphStore
         (nodes)     (nodes)    (relationships)
              │           │           │
              └───────────┼───────────┘
                          ▼
                     AuditLogger
                          │
                          ▼
                ConsolidationRunResult
```

---

## Pipeline Flow

### 1. Generate run_id

Each pipeline execution gets a unique `uuid.uuid4().hex` identifier used for traceability across all graph mutations and audit events.

### 2. Extract

```python
extraction_result = await extraction_engine.extract(fragments)
```

Returns `ExtractionResult` with entities, claims, relations, and any errors from failed batches. Errors are propagated to the final result but do not abort the pipeline.

### 3. Resolve Entities

For each `ExtractedEntity` in the extraction result:

1. Query existing entities of the same type via `graph_store.find_by_label(entity.type)`
2. Convert graph nodes to `ExistingEntity` instances
3. Call `entity_resolver.resolve(entity, existing)` to get a `ResolutionCandidate`
4. Dispatch based on `ResolutionCandidate.action`:

| Action | Behavior |
|---|---|
| `create_new` | Create new node (Agent/Artifact), store `name -> node_id` mapping |
| `merge` | Execute merge via `MergeExecutor`, store `name -> survivor_id` mapping |
| `link` | Create new node + `POSSIBLY_SAME_AS` relationship, store mapping |
| `review` | Same as `link` (manual review later) |

Unknown entity types (not in `_ENTITY_TYPE_TO_MODEL`) are skipped with an error logged.

### 4. Create Claims

For each `ExtractedClaim`:

1. Map `claim_type` to node model via `_CLAIM_TYPE_TO_MODEL` (Fact, Event, Observation, Decision, Outcome)
2. Create the node in the graph
3. For each `involved_entity`, look up its `node_id` in the entity name map and create a `CONCERNS` relationship

Unknown claim types are skipped with an error logged. Event claims require `temporal_info` for the `occurred_at` field.

### 5. Create Sources + SOURCED_FROM

For each processed fragment:

1. Create a `Source` node from fragment metadata (type, agent_id, confidence/reliability)
2. Link claims that originated from that fragment via `SOURCED_FROM` relationships

### 6. Create Extracted Relations

For each `ExtractedRelation`:

1. Look up `from_entity` and `to_entity` names in the entity name map
2. Map `relation_type` to a Pydantic relation model via `_REL_TYPE_TO_MODEL`
3. Construct the relation model from extracted properties
4. Create the relationship via `GraphStore.create_relationship()`

Relations are skipped (with error logged) when:
- Either entity is not in the name map (not resolved)
- The relation type is unknown
- Property validation fails

### 7. Detect Contradictions

New claims are compared against existing graph claims with a lightweight
polarity heuristic. When opposite polarity is detected for the same normalized
claim base, the pipeline creates a `CONTRADICTS` relationship with
`resolution_status=unresolved`.

### 8. Run Abstraction

The pipeline performs a first abstraction pass:

1. Group repeated claims by normalized content
2. Create `Pattern` nodes for groups crossing `pattern_min_occurrences`
3. Create `Concept` when multiple patterns emerge in the run
4. Create `Rule` when causal signals (`CAUSED_BY` / `LEADS_TO`) exist

Each derived node is linked through `DERIVED_FROM`, `GENERALIZES`, and
`INSTANCE_OF` relationships.

### 9. Audit

Log events via `AuditLogger`:
- `CONSOLIDATION_RUN` — summary of the entire run
- `NODE_CREATED` — for each node created
- `RELATION_CREATED` — for each relationship created

### 10. Return Result

```python
@dataclass
class ConsolidationRunResult:
    run_id: str
    fragments_processed: int
    entities_created: int
    entities_merged: int
    entities_linked: int
    claims_created: int
    relations_created: int
    contradictions_detected: int
    patterns_created: int
    concepts_created: int
    rules_created: int
    errors: list[str]
```

---

## Helper Maps

### Entity Type -> Node Model

```python
_ENTITY_TYPE_TO_MODEL = {
    "Agent": Agent,
    "Artifact": Artifact,
}
```

### Claim Type -> Node Model

```python
_CLAIM_TYPE_TO_MODEL = {
    "Fact": Fact,
    "Event": Event,
    "Observation": Observation,
    "Decision": Decision,
    "Outcome": Outcome,
}
```

### Relation Type -> Relationship Model

```python
_REL_TYPE_TO_MODEL = {
    "CAUSED_BY": CausedBy,
    "LEADS_TO": LeadsTo,
    "PRECEDED": Preceded,
    "FOLLOWED": Followed,
    "SUPPORTS": Supports,
    "PARTICIPATED_IN": ParticipatedIn,
    "DECIDED_BY": DecidedBy,
    "OBSERVED_BY": ObservedBy,
    "MENTIONS": Mentions,
    "CONCERNS": Concerns,
}
```

---

## ExistingEntity Construction

Graph nodes are converted to `ExistingEntity` for entity resolution:

```python
def _node_to_existing_entity(node: MemoryNode) -> ExistingEntity:
    return ExistingEntity(
        node_id=node.id,
        name=getattr(node, "name", ""),
        type=<derived from node type>,
        aliases=getattr(node, "aliases", []),
        properties={...},
        fragment_ids=getattr(node, "source_fragment_ids", []),
    )
```

---

## Error Handling

The pipeline follows a **best-effort** strategy:
- Extraction errors are captured in `ExtractionResult.errors` and propagated
- Unknown entity/claim/relation types are skipped with error messages
- Failed relation construction (validation errors) are skipped with error messages
- Missing entity references for relations are skipped with error messages
- All errors accumulate in `ConsolidationRunResult.errors`

The pipeline never raises — it always returns a result with error details.

---

## Remaining Limits

- Abstraction is heuristic-based and intentionally conservative
- Idempotency / deduplication across repeated runs is still v2
- Performance optimization (loading all entities per type) remains v2
