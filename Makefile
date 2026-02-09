.PHONY: test test-scenarios test-scenarios-tier2 test-real-llm-evals test-scenarios-real-llm calibrate-eval-thresholds verify-scenario-ground-truth verify-scenario-ground-truth-only

# Run the default pytest suite quickly.
test:
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest -q

# Run all CI-safe scenario tests (excludes real_llm) and emit scenario metrics.
test-scenarios:
	@mkdir -p reports
	@rm -f reports/scenario-metrics.jsonl reports/eval-calibration.json
	ENGRAMCP_SCENARIO_METRICS_PATH=reports/scenario-metrics.jsonl UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/scenarios -m "scenario and not real_llm" -ra --tb=short --maxfail=1 --durations=10 --junitxml=reports/pytest-scenarios.xml

# Run only Tier 2 CI-safe scenarios and emit Tier 2 metrics.
test-scenarios-tier2:
	@mkdir -p reports
	@rm -f reports/scenario-metrics-tier2.jsonl
	ENGRAMCP_SCENARIO_METRICS_PATH=reports/scenario-metrics-tier2.jsonl UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/scenarios -m "scenario and tier2 and not real_llm" -ra --tb=short --maxfail=1 --durations=10 --junitxml=reports/pytest-scenarios-tier2.xml

# Build calibration recommendations from CI-safe scenario metrics.
calibrate-eval-thresholds: test-scenarios
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run python scripts/calibrate_eval_thresholds.py --metrics reports/scenario-metrics.jsonl --output reports/eval-calibration.json

# Run Tier 2 scenarios then verify Tier 2/Tier 3 ground-truth fixtures.
verify-scenario-ground-truth: test-scenarios-tier2 verify-scenario-ground-truth-only

# Verify ground-truth fixtures against existing Tier 2 metrics only.
verify-scenario-ground-truth-only:
	@mkdir -p reports
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run python tests/scenarios/helpers/ground_truth_verify.py --metrics reports/scenario-metrics-tier2.jsonl --tier2-ground-truth tests/scenarios/fixtures/ground_truth_tier2.json --tier3-ground-truth tests/scenarios/fixtures/ground_truth_tier3_flight_logs_subset.json --output reports/ground-truth-verification.json

# Run opt-in provider-backed real-LLM scenario evals.
test-real-llm-evals:
	@if [ "$${ENGRAMCP_RUN_REAL_LLM_EVALS:-0}" != "1" ]; then \
		echo "Set ENGRAMCP_RUN_REAL_LLM_EVALS=1 to run real-LLM evals."; \
		exit 1; \
	fi
	@if [ -z "$${OPENAI_API_KEY:-}" ]; then \
		echo "OPENAI_API_KEY is required when running real-LLM evals."; \
		exit 1; \
	fi
	@mkdir -p reports
	ENGRAMCP_SCENARIO_METRICS_PATH=reports/scenario-metrics-real-llm.jsonl UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/scenarios/test_e2e_real_llm_eval.py -ra --tb=short --durations=10 --junitxml=reports/pytest-scenarios-real-llm.xml

# Alias for the explicit real-LLM scenario target.
test-scenarios-real-llm: test-real-llm-evals
