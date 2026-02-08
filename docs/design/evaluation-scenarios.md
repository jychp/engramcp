# Evaluation Scenarios — Design Document

> **Scope**: Tiered end-to-end evaluations under `tests/scenarios/`
> **Status**: Active design reference

---

## Purpose

This document defines how end-to-end evaluation suites are structured, executed,
and interpreted for EngraMCP.

Canonical location for eval suites:
- `tests/scenarios/`

Canonical tiers:
- Tier 1: synthetic deterministic scenarios
- Tier 2: curated semi-real scenarios
- Tier 3: real raw-data validation scenarios

---

## Authority

`docs/design/*` is implementation authority.

`docs/drafts/evaluation-scenarios.md` is archival working material and must not
be treated as the current source of truth.

---

## Test Layout

Required structure:
- Scenario test modules: `tests/scenarios/test_*.py`
- Shared fixtures: `tests/scenarios/fixtures/`
- Shared helpers: `tests/scenarios/helpers/`

Marker conventions:
- `scenario`: auto-applied by path (`tests/scenarios/*`)
- `tier1`: scenario uses synthetic deterministic data
- `tier2`: scenario uses curated semi-real data
- `tier3`: scenario uses real raw data
- `real_llm`: scenario calls a real provider and is opt-in

---

## Execution Modes

Default scenario run (CI-safe):
- includes Tier 1/2/3 tests that do not require external providers
- excludes `real_llm`

Opt-in real-provider run:
- requires explicit user confirmation in collaborative workflows
- requires `ENGRAMCP_RUN_REAL_LLM_EVALS=1` and provider credentials

---

## Minimum Assertions Per Scenario

Each scenario should assert:
- retrieval relevance (`memories` content and count)
- contradiction behavior when applicable
- metadata sanity (`working_memory_hits`, `graph_hits`)
- deterministic expectations tied to fixture input

When assertions fail, tests should emit:
- scenario name
- query used
- compact response payload excerpt
- optional fixture fragment summary

---

## Reporting

Scenario jobs should produce:
- terminal summary (`-ra`, short traceback)
- JUnit XML report for CI artifact/debugging
- scenario metrics JSONL (`reports/scenario-metrics.jsonl`) for threshold calibration
- calibration summary JSON (`reports/eval-calibration.json`) for default/eval-profile recommendations

Recommended command shape:

```bash
uv run pytest tests/scenarios -m "scenario and not real_llm" -ra --tb=short \
  --maxfail=1 --durations=10 --junitxml=reports/pytest-scenarios.xml
```

Calibration command:

```bash
make calibrate-eval-thresholds
```

Current pass/fail metric classes:
- `retrieval_relevance`: `graph_hits >= 1`, returned memories >= 1, keyword hit >= 1
- `contradiction_coverage`: contradictions >= 1 for contradiction scenarios
- `corroboration`: unique supporting source IDs >= 2
- `derivation_traceability`: at least one `Rule` memory with `derivation_depth == 3` and `derivation_run_id` present

---

## Current Coverage Baseline

- Tier 1: deterministic smoke in `tests/scenarios/test_tier1_smoke_regression.py`
- Tier 1 (opt-in provider): real-LLM E2E smoke in
  `tests/scenarios/test_e2e_real_llm_eval.py`
- Tier 2: curated deterministic regression in
  `tests/scenarios/test_tier2_curated_regression.py`
- Tier 3: pending

---

## Next Additions

- Add reusable curated fixtures for causal chains and corroboration in Tier 2
- Add Tier 3 ingestion/evaluation harness over real raw dataset slices
- Add threshold-calibrated pass/fail metrics per tier
