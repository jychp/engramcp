# Scenario Evaluation Suites

All evaluation suites for Tier 1, Tier 2, and Tier 3 live under `tests/scenarios/`.

Structure convention:
- Test modules: `tests/scenarios/test_*.py`
- Fixtures: `tests/scenarios/fixtures/`
- Scenario helpers/utilities: `tests/scenarios/helpers/` (when needed)

Real-LLM evals:
- `tests/scenarios/test_e2e_real_llm_eval.py`
- Opt-in only via `ENGRAMCP_RUN_REAL_LLM_EVALS=1` and provider credentials
- Requires explicit user confirmation before execution in collaborative workflows
