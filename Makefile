.PHONY: test test-scenarios test-scenarios-tier2 test-real-llm-evals

test:
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest -q

test-scenarios:
	@mkdir -p reports
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/scenarios -m "scenario and not real_llm" -ra --tb=short --maxfail=1 --durations=10 --junitxml=reports/pytest-scenarios.xml

test-scenarios-tier2:
	@mkdir -p reports
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/scenarios -m "scenario and tier2 and not real_llm" -ra --tb=short --maxfail=1 --durations=10 --junitxml=reports/pytest-scenarios-tier2.xml

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
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest tests/scenarios/test_e2e_real_llm_eval.py -ra --tb=short --durations=10 --junitxml=reports/pytest-scenarios-real-llm.xml
