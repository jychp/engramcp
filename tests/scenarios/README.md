# Scenario Evaluation Suites

All evaluation suites for Tier 1, Tier 2, and Tier 3 live under `tests/scenarios/`.

Structure convention:
- Test modules: `tests/scenarios/test_*.py`
- Fixtures: `tests/scenarios/fixtures/`
- Scenario helpers/utilities: `tests/scenarios/helpers/` (when needed)

Markers:
- `scenario` (auto-applied by path)
- `tier1`, `tier2`, `tier3` for evaluation tier
- `real_llm` only for provider-backed opt-in runs

Real-LLM evals:
- `tests/scenarios/test_e2e_real_llm_eval.py`
- Opt-in only via `ENGRAMCP_RUN_REAL_LLM_EVALS=1` and provider credentials
- Requires explicit user confirmation before execution in collaborative workflows

CI-safe scenario run:
- `make test-scenarios` (excludes `real_llm`, emits `reports/pytest-scenarios.xml`)
- `make test-scenarios-tier2` (runs only Tier 2 non-`real_llm`, emits `reports/pytest-scenarios-tier2.xml`)
- `make calibrate-eval-thresholds` (runs CI-safe scenarios, emits `reports/scenario-metrics.jsonl`, and writes calibration report to `reports/eval-calibration.json`)
