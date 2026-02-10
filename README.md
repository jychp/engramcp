# EngraMCP

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/jychp/engramcp/badge)](https://scorecard.dev/viewer/?uri=github.com/jychp/engramcp)
[![Checks](https://github.com/jychp/engramcp/actions/workflows/checks.yml/badge.svg)](https://github.com/jychp/engramcp/actions/workflows/checks.yml)
[![Build](https://github.com/jychp/engramcp/actions/workflows/ci.yml/badge.svg)](https://github.com/jychp/engramcp/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/engramcp)](https://pypi.org/project/engramcp/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

EngraMCP is a biomimetic memory engine for LLM agents. It models memory as a structured causal graph (instead of flat vector-only retrieval) to support traceability, contradiction handling, and progressive abstraction.

## Table of Contents

- [About](#about)
- [What's New](#whats-new)
- [Installation](#installation)
- [Configuration](#configuration)
- [Run Locally](#run-locally)
- [Documentation](#documentation)
- [Architecture](#architecture)
- [Development](#development)
- [Releases and Changelog](#releases-and-changelog)
- [Feedback and Contributions](#feedback-and-contributions)
- [Security](#security)
- [License](#license)
- [Contacts](#contacts)

## About

EngraMCP exposes three MCP tools:

- `send_memory` to ingest memory fragments
- `get_memory` to retrieve relevant working-memory fragments (keyword + confidence filter)
- `correct_memory` to apply contest/annotate/merge/split/reclassify corrections (with audit trail)

Core capabilities:

- Working memory to graph consolidation pipeline
- Layered architecture ready for graph-based retrieval and reasoning
- NATO-style confidence model (source reliability + claim credibility)
- Full source traceability across derived memories

## What's New

Current implemented foundation includes:

- MCP interface with the 3 core tools
- Redis-backed working memory
- Neo4j graph store and schema initialization
- Consolidation pipeline and extraction engine scaffolding
- Layer 5 concept-emergence bootstrap (query-demand tracking + concept candidate registry)
- Layer 6 retrieval service (`engine/retrieval.py`) with WM-first strategy, scoring interface, and bounded graph-context fallback (`max_depth`, causal chains, source trail, contradictions)
- `correct_memory` actions implemented with WM-first behavior for `contest`/`annotate`, graph-aware execution for `split_entity`/`merge_entities`, and graph-aware derived lifecycle updates for `reclassify` (WM fallback retained), with audit events
- Confidence engine with propagation logic
- Security and quality workflows in CI
- Uniform MCP validation and error response fields (`error_code`, `message`)

## Installation

Prerequisites:

- Python `3.13+`
- `uv`
- Redis `7+`
- Neo4j `5+` (or `docker compose`)

```bash
git clone git@github.com:jychp/engramcp.git
cd engramcp
uv sync
pre-commit install
```

## Configuration

Current configuration is code-based (dataclasses), not env-var based.

- LLM settings: `src/engramcp/config.py` (`LLMConfig`)
  - Providers: `openai` (requires `api_key`) and `noop` (deterministic local/testing adapter)
- Consolidation settings: `src/engramcp/config.py` (`ConsolidationConfig`)
  - Includes extraction retry/failure policy:
    - `extraction_max_retries`
    - `extraction_retry_backoff_seconds`
    - `retry_on_invalid_json`
    - `retry_on_schema_validation_error`
  - Scenario-only deterministic profile helper:
    - `scenario_eval_consolidation_config()` (`fragment_threshold=4`, `pattern_min_occurrences=2`)
- Entity resolution settings: `src/engramcp/config.py` (`EntityResolutionConfig`)
- Audit settings: `src/engramcp/config.py` (`AuditConfig`)

Working memory must be configured before using tools:

```python
import asyncio
from engramcp.server import configure

asyncio.run(configure(redis_url="redis://localhost:6379"))
```

Enable background consolidation (optional):

```python
from engramcp.config import LLMConfig

asyncio.run(
    configure(
        redis_url="redis://localhost:6379",
        enable_consolidation=True,
        neo4j_url="bolt://localhost:7687",
        llm_config=LLMConfig(provider="openai", api_key="YOUR_API_KEY"),
    )
)
```

## Run Locally

1. Start infrastructure services:

```bash
docker compose up -d redis neo4j
```

2. Verify services are healthy:

```bash
docker compose ps
```

3. Start using the MCP server in Python:

```python
import asyncio
from fastmcp import Client
from engramcp.server import configure, mcp


async def main():
    await configure(
        redis_url="redis://localhost:6379",
        enable_consolidation=True,
        neo4j_url="bolt://localhost:7687",
    )
    async with Client(mcp) as client:
        result = await client.call_tool("send_memory", {"content": "Hello memory"})
        print(result)


asyncio.run(main())
```

## Documentation

### Basic Usage

EngraMCP is used through MCP tool calls.

`send_memory` example:

```json
{
  "content": "A was on the flight to Virgin Islands on March 15",
  "source": {
    "type": "court_document",
    "ref": "https://example.com/doc.pdf",
    "citation": "page 47, line 12"
  },
  "confidence_hint": "B",
  "agent_id": "analyst_agent_1"
}
```

`get_memory` example:

```json
{
  "query": "Who traveled with Epstein to the Virgin Islands?",
  "max_depth": 3,
  "min_confidence": "D4",
  "compact": false
}
```

`correct_memory` example:

```json
{
  "target_id": "mem_a1b2c3",
  "action": "contest",
  "payload": {
    "reason": "Source unreliable"
  }
}
```

Full project documentation is available here:

- https://www.cubic.dev/wiki/jychp/engramcp
- Design docs in this repository: `docs/design/`
- Observability design note: `docs/design/observability.md`

## Architecture

EngraMCP follows a layered architecture from MCP interface to storage/consolidation engines.

- Overview: `docs/design/layer-architecture.md`
- MCP contract: `docs/design/mcp-interface.md`
- Working memory: `docs/design/working-memory.md`
- Graph store: `docs/design/graph-store.md`
- Observability: `docs/design/observability.md`
- Evaluation tiers and structure: `docs/design/evaluation-scenarios.md`

## Development

```bash
uv run pytest
uv run pre-commit run --all-files
make test-scenarios
make test-scenarios-tier2
make test-scenarios-tier3
make test-retrieval-perf
make test-bounded-resources
make test-scenarios-real-llm
make calibrate-eval-thresholds
make verify-scenario-ground-truth
make verify-scenario-ground-truth-only
```

`make calibrate-eval-thresholds` also writes:
- `reports/scenario-metrics.jsonl`
- `reports/eval-calibration.json`

`make verify-scenario-ground-truth` writes:
- `reports/ground-truth-verification.json`

`make verify-scenario-ground-truth-only` reuses existing metrics (no scenario re-run).
Optional: pass `METRICS_PATH=...` to point at a custom combined metrics JSONL.

Latency/perf notes:
- `get_memory` responses include `meta.retrieval_ms`.
- In-process latency aggregates are tracked for `send_memory`, `get_memory`, consolidation runs, and retrieval engine calls.
- CI includes a bounded retrieval perf integration test on a deep/branching graph topology.
- Sustained-load bounded-resource guardrail is available via `make test-bounded-resources` (CPU/wall-time per retrieval + memory-growth budget).

Optional real-LLM end-to-end evals (opt-in, requires API key and may incur cost):

```bash
cp .env.example .env
# edit .env and set ENGRAMCP_RUN_REAL_LLM_EVALS=1 + OPENAI_API_KEY
```

Execution policy:
- Real-LLM eval tests are never run implicitly in collaborative workflows.
- Always ask for explicit user confirmation before triggering them, even if `.env` is already configured.

```bash
make test-real-llm-evals
```

CI opt-in real-LLM job:
- `checks.yml` includes a dedicated provider-backed scenario job.
- It runs only on manual dispatch (`workflow_dispatch`) from `main` with `run_real_llm_evals=true`.
- It is bound to GitHub Environment `real-llm-evals` (configure `OPENAI_API_KEY` there, with required reviewers if desired).
- The job uses concurrency control (one in-flight run per branch/workflow) to avoid duplicate provider-cost runs.

Cost guardrail:
- Real-LLM scenario runs may incur provider charges; keep opt-in scope small and use `ENGRAMCP_EVAL_OPENAI_MODEL=gpt-4o-mini` unless higher capability is required.
- Real-LLM run metadata is emitted to `reports/scenario-metrics-real-llm.jsonl` (including `model_used`) for auditability.

Why tokenized graph retrieval matching:
- Retrieval now matches claim content on query tokens (not full-query substring only), reducing false negatives for natural-language queries like `meeting date` against stored claim text.

## Releases and Changelog

Versioning policy:

- EngraMCP follows **Semantic Versioning** (`MAJOR.MINOR.PATCH`)
- `PATCH`: backward-compatible bug fixes and internal improvements
- `MINOR`: backward-compatible features and enhancements
- `MAJOR`: breaking changes in public behavior or MCP contracts
- Releases are published from git tags (for example `1.2.3`)

- GitHub Releases: https://github.com/jychp/engramcp/releases
- Tags: https://github.com/jychp/engramcp/tags
- PyPI package history: https://pypi.org/project/engramcp/

## Feedback and Contributions

- Open issues and feature requests: https://github.com/jychp/engramcp/issues
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Reviewing guide: [REVIEWING.md](REVIEWING.md)
- Maintainers: [MAINTAINERS.md](MAINTAINERS.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Support: [SUPPORT.md](SUPPORT.md)

## Security

For vulnerability reporting, see [SECURITY.md](SECURITY.md).

## License

Apache 2.0. See [LICENSE](LICENSE).

## Contacts

- Maintainer: Jeremy Chapeau
- Email: jeremy@subimage.io
