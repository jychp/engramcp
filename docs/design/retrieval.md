# Retrieval Engine — Design Document

> **Layer**: 6 (`engine/retrieval.py`)
> **Status**: Foundation implemented
> **Scope**: WM-first selection, graph fallback stub, demand tracking hooks

---

## Overview

Layer 6 centralizes `get_memory` retrieval logic in a dedicated service (`RetrievalEngine`) instead of keeping synthesis logic inside `server.py`.

Current strategy:
1. Query `WorkingMemory` first (`search(query, min_confidence=...)`)
2. Apply response shaping (`compact`, `include_sources`, `limit`)
3. Track retrieval-demand pattern for Layer 5 concept emergence
4. Fallback to graph retrieval stub when working memory has no match

This preserves MCP contract behavior while opening a clean extension point for graph traversal and scoring evolution.

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
  - Graph fallback rank = confidence bonus only (no recency signal yet)

This keeps current ordering stable while exposing a swappable ranking interface for future retrieval quality work.

---

## Graph Fallback Stub

Current fallback is intentionally minimal:

- Uses `GraphRetriever.find_claim_nodes()`
- Converts only nodes carrying `id` and `content` into `MemoryEntry`
- Returns no contradictions or source-chain expansion yet

This is a placeholder for full traversal (depth-aware causal traversal, contradiction expansion, and confidence-aware synthesis).

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
  - retrieval-shape hook tracking
- Existing MCP contract tests remain unchanged and validate end-to-end behavior through `server.get_memory`.
