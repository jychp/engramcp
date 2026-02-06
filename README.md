# EngraMCP

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/jychp/engramcp/badge)](https://scorecard.dev/viewer/?uri=github.com/jychp/engramcp)
[![Checks](https://github.com/jychp/engramcp/actions/workflows/checks.yml/badge.svg)](https://github.com/jychp/engramcp/actions/workflows/checks.yml)
[![Build](https://github.com/jychp/engramcp/actions/workflows/ci.yml/badge.svg)](https://github.com/jychp/engramcp/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/engramcp)](https://pypi.org/project/engramcp/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A structured memory engine for LLMs that goes beyond flat vector-based RAG. EngraMCP models knowledge as a causal graph with an ontology, enabling reasoning, contradiction detection, and progressive abstraction — mimicking how human memory consolidates information over time.

## Features

- **Three MCP tools**: `send_memory` (write), `get_memory` (read), `correct_memory` (correct)
- **Biomimetic architecture**: working memory → async consolidation → knowledge graph
- **Causal reasoning**: explicit causal chains, temporal ordering, contradiction detection
- **NATO confidence model**: two-dimensional source reliability + information credibility
- **Full traceability**: every derived node traces back to its sources with exact citations
- **Domain-agnostic**: core ontology works on any subject matter, concepts emerge from usage

## Stack

| Component | Technology |
|---|---|
| Language | Python 3.13+ |
| MCP Server | [FastMCP](https://github.com/jlowin/fastmcp) v2 |
| Graph Database | Neo4j |
| Test Framework | pytest + testcontainers |

## Installation

```bash
# Clone the repository
git clone git@github.com:jychp/engramcp.git
cd engramcp

# Install dependencies
uv sync

# Install pre-commit hooks
pre-commit install
```

## Quick Start

EngraMCP exposes three MCP tools:

### `send_memory` — Write

Ingest a memory into the system.

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

### `get_memory` — Read

Retrieve relevant memories with causal reasoning.

```json
{
  "query": "Who traveled with Epstein to the Virgin Islands?",
  "max_depth": 3,
  "min_confidence": "D4",
  "compact": false
}
```

Returns structured `memories[]`, `contradictions[]`, and `meta{}`.

### `correct_memory` — Correct

Correct the knowledge graph: contest, annotate, merge/split entities, reclassify.

```json
{
  "target_id": "mem_a1b2c3",
  "action": "contest",
  "payload": { "reason": "Source unreliable" }
}
```

## Development

```bash
# Run tests
uv run pytest

# Run linting
pre-commit run --all-files
```

## License

Apache 2.0
