.PHONY: test test-scenarios test-scenarios-tier2 test-real-llm-evals test-scenarios-real-llm calibrate-eval-thresholds

test:
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest -q

test-scenarios:
	@mkdir -p reports
	@rm -f reports/scenario-metrics.jsonl reports/eval-calibration.json
	ENGRAMCP_SCENARIO_METRICS_PATH=reports/scenario-metrics.jsonl UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/scenarios -m "scenario and not real_llm" -ra --tb=short --maxfail=1 --durations=10 --junitxml=reports/pytest-scenarios.xml

test-scenarios-tier2:
	@mkdir -p reports
	@rm -f reports/scenario-metrics-tier2.jsonl
	ENGRAMCP_SCENARIO_METRICS_PATH=reports/scenario-metrics-tier2.jsonl UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/scenarios -m "scenario and tier2 and not real_llm" -ra --tb=short --maxfail=1 --durations=10 --junitxml=reports/pytest-scenarios-tier2.xml

calibrate-eval-thresholds: test-scenarios
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run python scripts/calibrate_eval_thresholds.py --metrics reports/scenario-metrics.jsonl --output reports/eval-calibration.json

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

test-scenarios-real-llm: test-real-llm-evals
