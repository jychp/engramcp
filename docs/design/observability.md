# Observability

> **Scope**: lightweight latency observability for request/consolidation/retrieval paths
> **Status**: Active design reference

---

## Purpose

Define the current latency observability baseline used by EngraMCP to:
- detect regressions in core paths,
- support deterministic unit/integration assertions,
- provide a migration path toward production telemetry (P4 follow-ups).

---

## Current Model

Implementation lives in:
- `src/engramcp/observability.py`

The recorder is:
- in-process (memory only),
- thread-safe (single-process lock),
- aggregation-based (no per-event persistence),
- log-emitting (one structured-ish log line per sample).

---

## Instrumented Operations

The following operation keys are currently emitted:
- `mcp.send_memory`
- `mcp.get_memory`
- `retrieval_engine.retrieve`
- `consolidation.run`

Each operation records:
- `count`
- `error_count`
- `total_ms`
- `avg_ms`
- `min_ms`
- `max_ms`
- `last_ms`

---

## API Surface

Public helpers:
- `record_latency(operation, duration_ms, ok=True)`
- `latency_metrics_snapshot()`
- `reset_latency_metrics()`

Usage notes:
- `latency_metrics_snapshot()` is intended for tests and local introspection.
- `reset_latency_metrics()` is a test helper and should be used to isolate assertions.

---

## Retrieval Latency Field

`get_memory` response metadata (`MetaInfo`) now includes:
- `retrieval_ms`

Semantics:
- wall-clock milliseconds spent in retrieval engine execution path,
- integer value (`int`),
- always present for successful retrieval responses,
- may be `0` for very fast paths.

---

## Perf Guardrail Test

CI-safe retrieval perf guardrail:
- `tests/integration/test_retrieval_performance.py`

What it validates:
- bounded retrieval latency on deep/branching graph topology,
- retrieval still returns graph hits under load shape,
- retrieval metadata includes bounded `retrieval_ms`.

Local fast command:

```bash
make test-retrieval-perf
```

---

## Limitations

Current scope is intentionally minimal:
- no persistence across process restarts,
- no cardinality controls for arbitrary operation names,
- no metrics endpoint/exporter,
- no distributed tracing/correlation IDs.

These are expected follow-ups for production hardening:
- Prometheus/OpenTelemetry export,
- long-horizon storage and dashboards,
- service-level objective alerting.
