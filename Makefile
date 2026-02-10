.PHONY: test test-retrieval-perf test-bounded-resources test-scenarios test-scenarios-tier2 test-scenarios-tier3 test-real-llm-evals test-scenarios-real-llm calibrate-eval-thresholds verify-scenario-ground-truth verify-scenario-ground-truth-only

# Run the default pytest suite quickly.
test:
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest -q

# Run only the retrieval deep/branching performance guardrail test.
test-retrieval-perf:
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/integration/test_retrieval_performance.py -ra --tb=short --durations=10

# Run sustained-load bounded resource guardrail test (CPU/memory).
test-bounded-resources:
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/integration/test_bounded_resource_usage.py -ra --tb=short --durations=10

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

# Run only Tier 3 CI-safe scenarios and emit Tier 3 metrics.
test-scenarios-tier3:
	@mkdir -p reports
	@rm -f reports/scenario-metrics-tier3.jsonl
	ENGRAMCP_SCENARIO_METRICS_PATH=reports/scenario-metrics-tier3.jsonl UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/scenarios -m "scenario and tier3 and not real_llm" -ra --tb=short --maxfail=1 --durations=10 --junitxml=reports/pytest-scenarios-tier3.xml

# Build calibration recommendations from CI-safe scenario metrics.
calibrate-eval-thresholds: test-scenarios
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run python scripts/calibrate_eval_thresholds.py --metrics reports/scenario-metrics.jsonl --output reports/eval-calibration.json

# Run Tier 2 and Tier 3 scenarios then verify ground-truth fixtures.
verify-scenario-ground-truth: test-scenarios-tier2 test-scenarios-tier3
	@mkdir -p reports
	@cat reports/scenario-metrics-tier2.jsonl reports/scenario-metrics-tier3.jsonl > reports/scenario-metrics-tier23.jsonl
	$(MAKE) verify-scenario-ground-truth-only METRICS_PATH=reports/scenario-metrics-tier23.jsonl

# Verify ground-truth fixtures against an existing metrics file.
verify-scenario-ground-truth-only:
	@mkdir -p reports
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run python tests/scenarios/helpers/ground_truth_verify.py --metrics $${METRICS_PATH:-reports/scenario-metrics-tier2.jsonl} --tier2-ground-truth tests/scenarios/fixtures/ground_truth_tier2.json --tier3-ground-truth tests/scenarios/fixtures/ground_truth_tier3_flight_logs_subset.json --output reports/ground-truth-verification.json

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
	@rm -f reports/scenario-metrics-real-llm.jsonl
	ENGRAMCP_SCENARIO_METRICS_PATH=reports/scenario-metrics-real-llm.jsonl UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/scenarios/test_e2e_real_llm_eval.py -ra --tb=short --durations=10 --junitxml=reports/pytest-scenarios-real-llm.xml

# Alias for the explicit real-LLM scenario target.
test-scenarios-real-llm: test-real-llm-evals
