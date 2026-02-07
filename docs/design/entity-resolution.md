# Entity Resolution — Design Document

> **Layer**: 4 (partial — consolidation)
> **Parent**: [Deep Dives DD1](../drafts/deep-dives.md#dd1)
> **Status**: In progress

---

## Overview

Entity resolution maps extracted entity names (from `ExtractedEntity`) to
existing graph nodes — or decides to create new ones. It is the bridge between
the extraction engine (Layer 4) and the graph store (Layer 0+2).

**Key design principle:** The resolver is pure logic. It receives an extracted
entity plus a list of existing entities (provided by the caller) and returns a
resolution decision. Graph mutations (merge execution) are handled by a
separate `MergeExecutor` class that accepts a `GraphStore`. This keeps
scoring/matching testable without containers.

---

## Three-Level Strategy

### Level 1 — Deterministic Normalization (no LLM)

Before comparison, names are normalized:

1. Unicode NFC normalization
2. Strip leading/trailing whitespace
3. Handle "Last, First" → "First Last" (single comma reorder)
4. Remove title tokens (Mr., Mrs., Dr., Esq., Jr., Sr., etc.)
5. Collapse multiple whitespace
6. Lowercase

After normalization, exact match = same entity. Also checks against existing
`aliases` on graph nodes — if "Jeff Epstein" is already an alias of "Jeffrey
Epstein", level 1 catches it.

### Level 2 — Fuzzy Matching (no LLM)

Three signals combined into a composite score:

| Signal | Description | Weight (default) |
|---|---|---|
| Name similarity | max(token Jaccard, 1 − edit distance) | 0.5 |
| Context overlap | Jaccard on fragment ID sets | 0.3 |
| Property compatibility | Compatible props → 0–1; blocking conflict → None | 0.2 |

**Composite score** = weighted sum. Blocking property conflict → 0.0.

| Score | Action |
|---|---|
| > 0.9 | `merge` — auto-merge with high confidence |
| 0.7 – 0.9 | `review` — merge candidate, flagged for review |
| 0.5 – 0.7 | `link` — create `POSSIBLY_SAME_AS` relation |
| < 0.5 | `create_new` — distinct entities |

### Level 3 — LLM-Assisted (optional)

For ambiguous cases (score in `[create_link_threshold, auto_merge_threshold)`
and `llm_assisted_enabled`), the LLM is asked:

> Are Entity A and Entity B the same? Reply SAME, DIFFERENT, or UNCERTAIN.

- SAME → `merge`
- DIFFERENT → `create_new`
- UNCERTAIN → `link` (POSSIBLY_SAME_AS)

Can be disabled via `EntityResolutionConfig.llm_assisted_enabled`.

---

## Anti-Pattern Guards

1. **Single-token names** (e.g. "Maxwell") — never auto-merge, only `link` at best.
2. **Cross-type** (e.g. Agent vs Artifact) — never merge candidates.
3. **No transitive auto-merge** — if A merges with B, and B had
   `POSSIBLY_SAME_AS` with C, the A↔C score is re-evaluated independently.

---

## Data Types

```python
@dataclass
class ExistingEntity:
    node_id: str
    name: str
    type: str
    aliases: list[str]
    properties: dict
    fragment_ids: list[str]

class ResolutionAction(str, Enum):
    merge = "merge"
    review = "review"
    link = "link"
    create_new = "create_new"

@dataclass
class ResolutionCandidate:
    entity_name: str
    existing_node_id: str | None
    existing_name: str | None
    score: float
    action: ResolutionAction
    method: str  # "level_1", "level_2", "level_3"

@dataclass
class MergeResult:
    survivor_id: str
    absorbed_id: str
    aliases_added: list[str]
    relations_transferred: int
```

---

## Components

### `normalize_name(name: str) -> str`

Pure function. Deterministic normalization (level 1).

### Scorer Functions (Level 2)

- `token_jaccard(a, b)` — Jaccard similarity on name tokens
- `normalized_edit_distance(a, b)` — Levenshtein (inline, no deps)
- `name_similarity(a, b)` — max(jaccard, 1 − edit_distance)
- `context_overlap(frag_ids_a, frag_ids_b)` — Jaccard on fragment sets
- `property_compatibility(props_a, props_b)` — None if blocking, else 0–1
- `composite_score(...)` — weighted combination

### `EntityResolver`

Orchestrates the three-level cascade. Takes `EntityResolutionConfig` and
optional `LLMAdapter`. The `resolve()` method returns a `ResolutionCandidate`.

### `build_disambiguation_prompt(...)`

Builds the level 3 LLM prompt for SAME / DIFFERENT / UNCERTAIN.

### `MergeExecutor`

Executes merge decisions against a `GraphStore`:

1. Get both nodes
2. Transfer all relationships from absorbed → survivor
3. Add absorbed name + aliases to survivor's aliases
4. Create `MERGED_FROM` relation (permanent traceability)
5. Remove any `POSSIBLY_SAME_AS` between them
6. Delete absorbed node
7. Return `MergeResult`

---

## MergedFrom Relation

```
(:Agent)-[:MERGED_FROM {merge_run_id, merged_at}]->(:Agent)
```

Permanent traceability — never deleted. Tracks which node was absorbed into
which survivor and when.

---

## Configuration

All parameters in `EntityResolutionConfig`:

| Parameter | Default | Description |
|---|---|---|
| `auto_merge_threshold` | 0.9 | Score above which auto-merge triggers |
| `flag_for_review_threshold` | 0.7 | Score above which review is flagged |
| `create_link_threshold` | 0.5 | Score above which POSSIBLY_SAME_AS is created |
| `llm_assisted_enabled` | True | Enable level 3 LLM disambiguation |
| `name_similarity_weight` | 0.5 | Weight for name similarity signal |
| `context_overlap_weight` | 0.3 | Weight for context overlap signal |
| `property_compatibility_weight` | 0.2 | Weight for property compatibility |
| `token_jaccard_threshold` | 0.6 | Minimum Jaccard for token matching |
| `edit_distance_max` | 0.3 | Maximum normalized edit distance |

---

## Not in Scope

- Split mechanics — deferred to `correct_memory` end-to-end wiring
- Audit logging of merges — Sprint 6d (orchestrator)
- Graph integration tests (real Neo4j) — Sprint 9
- Alias fulltext index — Community edition limitation; `find_agent_by_alias` suffices
