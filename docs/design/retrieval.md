# Retrieval Engine — Design Document

> **Layer**: 6 (`engine/retrieval.py`)
> **Status**: Core behavior implemented
> **Scope**: WM-first selection, bounded graph-context fallback, demand tracking hooks

---

## Overview

Layer 6 centralizes `get_memory` retrieval logic in a dedicated service (`RetrievalEngine`) instead of keeping synthesis logic inside `server.py`.

Current strategy:
1. Query `WorkingMemory` first (`search(query, min_confidence=...)`)
2. Apply response shaping (`compact`, `include_sources`, `limit`)
3. Track retrieval-demand pattern for Layer 5 concept emergence
4. Fallback to graph context retrieval when working memory has no match

This preserves MCP contract behavior while using depth-aware graph traversal through `max_depth`.

---

## Public API

`RetrievalEngine` exposes:

- `async retrieve(request: GetMemoryInput) -> GetMemoryResult`
- `query_demand_count(...) -> int` (test/introspection helper)
- `concept_candidate_count() -> int` (test/introspection helper)

`server.get_memory` validates input (`GetMemoryInput`) then delegates directly to `RetrievalEngine.retrieve`.

---

## Scoring Interface

Layer 6 introduces a scoring protocol:

- `RetrievalScorer`
  - `score_working_memory(fragment) -> float`
  - `score_graph_memory(memory) -> float`
- Default implementation: `RecencyConfidenceScorer`
  - Working-memory rank = recency + tiny confidence bonus
  - Graph fallback rank = confidence bonus only (no relation-confidence projection yet)

This keeps current ordering stable while exposing a swappable ranking interface for future retrieval quality work.

---

## Graph Fallback

Graph fallback is query-aware and context-aware:

- Uses `find_claim_context_by_content(query, limit, max_depth, include_sources, include_contradictions)` when available
- Enriches graph hits with bounded causal expansion (`max_depth`)
- Returns source trail and unresolved contradictions when requested
- Derives `dynamic_type` from labels beyond ontology base labels
- Falls back to legacy contracts (`find_claim_nodes_by_content` then `find_claim_nodes`) for compatibility

Compact mode omits heavy graph fields (`sources`, `causal_chain`, and top-level `contradictions`) consistently.

---

## Layer 5 Hook

Demand tracking is now executed in Layer 6 during retrieval:

- Normalized shape = requested properties + observed node types
- Recorded through `QueryDemandTracker`
- Threshold signal forwarded to `ConceptRegistry`

This keeps concept-emergence signals coupled to real retrieval demand, not ingestion volume.

---

## Test Coverage

- `tests/unit/test_retrieval.py`
  - `working_memory_first`
  - `falls_through_to_graph`
  - graph context fallback (`max_depth`, causal/source/contradiction shaping)
  - compact mode omission behavior
  - retrieval-shape hook tracking
- Existing MCP contract tests remain unchanged and validate end-to-end behavior through `server.get_memory`.
