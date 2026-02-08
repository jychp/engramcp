# Config Module + Audit Logger

> Design document for Sprint 6a — configuration dataclasses and async audit logging.

---

## Overview

This document covers two foundational modules that all subsequent consolidation
engine sub-sprints depend on:

1. **`config.py`** — Frozen dataclasses providing typed configuration with
   sensible defaults for every subsystem.
2. **`audit/`** — Append-only JSONL audit logger for full event replay.

---

## Config Module (`config.py`)

### Design Decisions

- **Frozen dataclasses** following the `ConfidenceConfig` pattern — immutable
  after construction, preventing accidental mutation.
- **No env-var loading or YAML parsing** — configuration is just data with
  defaults. Loading/parsing belongs to a future integration layer (YAGNI).
- **One dataclass per subsystem** — keeps concerns cleanly separated.
- **Explicit scenario-eval profile helper** — `scenario_eval_consolidation_config()`
  provides deterministic eval-only consolidation tuning (`4/2`) without changing
  production defaults (`10/3`).

### Dataclasses

| Class | Fields | Defaults |
|---|---|---|
| `LLMConfig` | provider, model, api_key, temperature, max_tokens, timeout_seconds | openai, gpt-4, None, 0.2, 4096, 30.0 |
| `ConsolidationConfig` | fragment_threshold, extraction_batch_size, pattern_min_occurrences | 10, 5, 3 |
| `EntityResolutionConfig` | auto_merge_threshold, flag_for_review_threshold, create_link_threshold, llm_assisted_enabled | 0.9, 0.7, 0.5, True |
| `AuditConfig` | file_path, enabled | engramcp_audit.jsonl, True |

---

## Audit Package (`audit/`)

### Architecture

```
AuditLogger
  ├── config: AuditConfig (frozen)
  ├── _lock: asyncio.Lock (serialization)
  ├── log(event) → asyncio.to_thread (append)
  └── read_events(event_type?, since?) → list[AuditEvent]
```

### Design Decisions

- **Async I/O via `asyncio.to_thread`** — avoids adding an `aiofiles`
  dependency while keeping the event loop non-blocking.
- **`asyncio.Lock`** — serializes concurrent writes to prevent interleaved
  JSON lines.
- **No-op when disabled** — `AuditLogger.log()` returns immediately if
  `config.enabled is False`, with no file creation.
- **JSONL format** — one JSON object per line for easy streaming, grepping,
  and replay. Each line is a serialized `AuditEvent`.

### AuditEventType Enum

| Value | Description |
|---|---|
| `SEND_MEMORY` | Agent submitted a new memory |
| `GET_MEMORY` | Agent queried memories |
| `CORRECT_MEMORY` | Agent requested a correction |
| `CONFIDENCE_CHANGE` | Confidence rating was recalculated |
| `CONSOLIDATION_RUN` | Batch consolidation pipeline executed |
| `NODE_CREATED` | New graph node created |
| `RELATION_CREATED` | New graph relationship created |

### AuditEvent Model

Frozen Pydantic model with three fields:

- `timestamp` (float) — Unix epoch, defaults to `time.time()`
- `event_type` (AuditEventType) — category of the event
- `payload` (dict[str, Any]) — arbitrary event-specific data

---

## Testing Strategy

All tests are **pure unit tests** — no containers, no I/O beyond `tmp_path`.

| Test Class | Count | Covers |
|---|---|---|
| `TestLLMConfig` | 1 | Default values |
| `TestConsolidationConfig` | 1 | Default values |
| `TestEntityResolutionConfig` | 1 | Default values |
| `TestAuditConfig` | 1 | Default values |
| `TestConfigImmutability` | 4 | Frozen enforcement for all 4 dataclasses |
| `TestAuditEventSchema` | 1 | AuditEvent field validation |
| `TestAuditLogWrite` | 5 | JSONL write, timestamp, append, file creation, disabled no-op |
| `TestAuditLogRead` | 4 | Read all, filter by type, filter by since, empty file |

**Total: 18 tests**

---

## Not in Scope

- Wiring audit logger into `server.py` (Sprint 6d)
- Migration of `ConfidenceConfig` to `config.py` (separate refactor)
- Env-var or YAML config loading
- Log rotation or size limits
