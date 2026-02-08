# MCP Interface — Design Document

> **Layer**: 7 (`server.py`)
> **Status**: Implemented
> **API Contract**: Frozen

---

## Overview

The MCP interface is the only entry point for agents interacting with EngraMCP. It exposes three tools via FastMCP v2: **write**, **read**, and **correct**.

The interface is wired to the Redis-backed `WorkingMemory` domain. `get_memory` delegates retrieval/synthesis to Layer 6 (`engine/retrieval.py`) with WM-first behavior and graph content fallback.

---

## Architecture Decisions

### Direct `@mcp.tool` decorators

We use plain `@mcp.tool` decorators on `server.py`, not a `ToolCollection` or router abstraction. With only 3 tools, an additional layer adds complexity without value.

### Pydantic return types

FastMCP v2 serializes Pydantic models automatically via `pydantic_core.to_jsonable_python()`. Tools return typed models (`SendMemoryResult`, `GetMemoryResult`, `CorrectMemoryResult`), not dicts. This gives us:
- Schema validation at the boundary
- Auto-generated JSON for MCP clients
- No manual `.model_dump()` calls

### Testing via `Client(mcp)`

`@mcp.tool` returns a `FunctionTool` object, not the original function. Calling tool functions directly is not possible. All tests use `fastmcp.Client` which exercises the full MCP protocol (serialization, validation, transport). This ensures contract compliance.

### Backend strategy

`server.py` delegates write/read/existence checks to `WorkingMemory` (`memory/store.py`) and exposes `_reset_working_memory()` for test cleanup.

---

## Tools

### `send_memory`

| Parameter | Required | Type | Description |
|---|---|---|---|
| `content` | yes | `str` | The affirmation / fact / observation |
| `source` | no | `dict` | Source reference (`type`, `ref`, `citation`) |
| `confidence_hint` | no | `str` | Source reliability letter (A-F) |
| `agent_id` | no | `str` | Calling agent identifier |

**Returns**: `SendMemoryResult(memory_id, status)`

**Behavior**:
- Validates input with `SendMemoryInput` (size limits, confidence format)
- Generates a `mem_<hex16>` ID
- Stores entry with default confidence `<hint_or_F>3` (hint letter + uncorroborated number)
- Attaches source as `SourceEntry` if provided

### `get_memory`

| Parameter | Required | Type | Default | Description |
|---|---|---|---|---|
| `query` | yes | `str` | — | Natural language query |
| `max_depth` | no | `int` | 3 | Max causal chain traversal depth |
| `min_confidence` | no | `str` | `"F6"` | Minimum NATO rating filter |
| `include_contradictions` | no | `bool` | `true` | Include contradictions |
| `include_sources` | no | `bool` | `true` | Include source chains |
| `limit` | no | `int` | 20 | Max memories returned |
| `compact` | no | `bool` | `false` | Omit sources, chains, participants |

**Returns**: `GetMemoryResult(memories[], contradictions[], meta{})`

**Response shape**: See [DD2 — get_memory Response Format](../drafts/deep-dives.md#dd2) for the full JSON contract.

**Confidence filtering**: A memory passes `min_confidence` when **both** its letter index **and** number are <= the filter's (AND semantics). `F6` lets everything through.

**Compact mode**: When `compact=true`, memories have empty `sources`, `causal_chain`, and `participants`; `contradictions` is empty.

### `correct_memory`

| Parameter | Required | Type | Description |
|---|---|---|---|
| `target_id` | yes | `str` | ID of the memory to correct |
| `action` | yes | `str` | One of: `contest`, `annotate`, `merge_entities`, `split_entity`, `reclassify` |
| `payload` | no | `dict` | Action-specific data |

**Returns**: `CorrectMemoryResult(target_id, action, status, details)`

**Behavior**:
- Validates input with `CorrectMemoryInput`
- Validates `action` against `CorrectionAction` enum
- Returns `status: "not_found"` if `target_id` doesn't exist
- Implements `contest`: validates `ContestPayload`, downgrades confidence one conservative step, marks memory as contested in properties, and writes a `CORRECT_MEMORY` audit event
- Implements `annotate`: validates `AnnotatePayload`, appends annotation metadata in properties without mutating source provenance, and writes a `CORRECT_MEMORY` audit event
- Implements `split_entity`: validates `SplitEntityPayload`, creates split child memories, deletes original target, and writes a `CORRECT_MEMORY` audit event
- Implements `merge_entities` (WM-first): validates `MergeEntitiesPayload`, merges target + secondary fragment content/provenance, deletes secondary fragment, and writes a `CORRECT_MEMORY` audit event
- Implements `reclassify` (WM-first): validates `ReclassifyPayload`, updates fragment type with reclassification history in properties, and writes a `CORRECT_MEMORY` audit event

### Error response contract

All tool responses include optional `error_code` and `message` fields. Rejected/errored calls set these fields for consistent MCP client handling.

---

## Models

All models are in `src/engramcp/models/schemas.py`.

### Enums

- `CorrectionAction`: `contest | annotate | merge_entities | split_entity | reclassify`
- `ContradictionNature`: `temporal_impossibility | factual_conflict | source_disagreement | logical_inconsistency`

### Input models

`SendMemoryInput`, `GetMemoryInput`, `CorrectMemoryInput`, `SourceInput`

### Output models

`SendMemoryResult`, `GetMemoryResult`, `CorrectMemoryResult`, `MemoryEntry`, `Participant`, `CausalLink`, `SourceEntry`, `Contradiction`, `MetaInfo`

---

## Test Coverage

| Suite | Count | Description |
|---|---|---|
| `tests/unit/test_mcp_interface.py` | 31 | Contract tests for all 3 tools |
| `tests/integration/test_mcp_protocol.py` | 3 | Tool registration, roundtrip, error handling |

All tests use `fastmcp.Client(mcp)` — no direct function calls.

---

## Future Changes

- Expand `correct_memory` `contest` from WM-local cascade hook to graph-wide confidence cascade via `engine/confidence.py`
- Replace WM-first `merge_entities` and `reclassify` with graph-backed lifecycle-aware mutations
- Tool signatures and response shapes are **frozen** — additive fields only without version bump
