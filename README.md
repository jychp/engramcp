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
- `correct_memory` to validate correction requests and apply the current correction stub

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
- Consolidation settings: `src/engramcp/config.py` (`ConsolidationConfig`)
- Entity resolution settings: `src/engramcp/config.py` (`EntityResolutionConfig`)
- Audit settings: `src/engramcp/config.py` (`AuditConfig`)

Working memory must be configured before using tools:

```python
import asyncio
from engramcp.server import configure

asyncio.run(configure(redis_url="redis://localhost:6379"))
```

## Run Locally

1. Start Neo4j (optional for current working-memory-only flows):

```bash
docker compose up -d neo4j
```

2. Start Redis (required):

```bash
docker run --rm -p 6379:6379 redis:7
```

3. Start using the MCP server in Python:

```python
import asyncio
from fastmcp import Client
from engramcp.server import configure, mcp


async def main():
    await configure(redis_url="redis://localhost:6379")
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

## Architecture

EngraMCP follows a layered architecture from MCP interface to storage/consolidation engines.

- Overview: `docs/design/layer-architecture.md`
- MCP contract: `docs/design/mcp-interface.md`
- Working memory: `docs/design/working-memory.md`
- Graph store: `docs/design/graph-store.md`

## Development

```bash
uv run pytest
uv run pre-commit run --all-files
```

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
