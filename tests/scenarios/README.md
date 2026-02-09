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
- Emits run metadata metrics (including `model_used`) to `reports/scenario-metrics-real-llm.jsonl`

CI-safe scenario run:
- `make test-scenarios` (excludes `real_llm`, emits `reports/pytest-scenarios.xml`)
- `make test-scenarios-tier2` (runs only Tier 2 non-`real_llm`, emits `reports/pytest-scenarios-tier2.xml`)
- `make calibrate-eval-thresholds` (runs CI-safe scenarios, emits `reports/scenario-metrics.jsonl`, and writes calibration report to `reports/eval-calibration.json`)
- `make verify-scenario-ground-truth` (runs CI-safe scenarios then verifies Tier 2 metric expectations + Tier 3 subset structure; emits `reports/ground-truth-verification.json`)
- `make verify-scenario-ground-truth-only` (runs only verifier against `reports/scenario-metrics-tier2.jsonl`; no scenario re-run)
- `make test-scenarios-real-llm` (alias of `make test-real-llm-evals` for explicit provider-backed runs)

Tier 2 scenario modules currently include:
- `tests/scenarios/test_tier2_curated_regression.py` (curated contradiction regression)
- `tests/scenarios/test_tier2_corroboration_derivation.py` (source corroboration + derivation traceability)
- `tests/scenarios/test_tier2_corporate_timeline.py` (semi-real temporal position-change tracking from archived S9 narrative)

Ground-truth verification fixtures/scripts:
- `tests/scenarios/fixtures/ground_truth_tier2.json` (required scenario metric classes)
- `tests/scenarios/fixtures/ground_truth_tier3_flight_logs_subset.json` (Tier 3 subset schema/alias baseline + explicit negative validation samples)
- `tests/scenarios/helpers/ground_truth_verify.py` (verification CLI)
