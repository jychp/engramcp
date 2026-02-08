.PHONY: test test-real-llm-evals

test:
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest -q

test-real-llm-evals:
	@if [ "$${ENGRAMCP_RUN_REAL_LLM_EVALS:-0}" != "1" ]; then \
		echo "Set ENGRAMCP_RUN_REAL_LLM_EVALS=1 to run real-LLM evals."; \
		exit 1; \
	fi
	@if [ -z "$${OPENAI_API_KEY:-}" ]; then \
		echo "OPENAI_API_KEY is required when running real-LLM evals."; \
		exit 1; \
	fi
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/.uv-cache} uv run pytest -q tests/scenarios/test_e2e_real_llm_eval.py
