---
name: engramcp-audit
description: Codebase audit.
---

You are a senior software engineer performing a comprehensive codebase audit. Your goal is to produce an actionable report with prioritized findings.

## Deployment context

- EngraMCP is a **FastMCP v2 server** exposing 3 tools (send_memory, get_memory, correct_memory)
- It connects to **Redis** (working memory buffer) and **Neo4j** (knowledge graph)
- The server runs locally, connected to by MCP-compatible agents (Claude, etc.)
- Assume an untrusted agent can call any MCP tool with arbitrary inputs
- Secrets (Redis URL, Neo4j credentials) are in environment variables or config
- The project follows **DDD** (Domain-Driven Design) with bounded contexts: `memory/`, `graph/`, `engine/`, `models/`, `audit/`

## Audit scopes

Run **all scopes in parallel** using subagents. Each scope must produce a **top 5 findings max**, ranked by severity. No noise — only findings worth fixing.

### 1. Security

- Input validation on all MCP tool parameters (content injection, oversized payloads)
- Redis command injection via user-controlled keys or values
- Neo4j Cypher injection via string interpolation in queries
- Memory poisoning (agent submitting crafted data to manipulate retrieval)
- Secret leakage (connection strings in logs, error messages, tracebacks)
- Deserialization safety (Pydantic model_validate_json, dict payloads)

### 2. Reliability & error handling

- Silent failures (`except: pass`, missing error logging)
- Redis/Neo4j connection failures not handled gracefully
- `assert` used for runtime validation (disabled with `python -O`)
- Pydantic validation errors not surfaced properly to MCP callers
- Dead code (unused imports, unreachable branches, no-op methods)

### 3. Concurrency & race conditions

- Redis operations that should be atomic but aren't pipelined
- TOCTOU issues (check-then-act on Redis keys without transactions)
- Concurrent MCP tool calls modifying the same memory fragment
- Async resource cleanup (Redis/Neo4j clients properly closed)
- Fire-and-forget tasks without tracking or error handling

### 4. Data integrity

- Can a fragment be partially written (JSON + sorted set + keyword indexes out of sync)?
- What happens when Redis evicts keys via TTL mid-operation?
- Are keyword indexes cleaned up when fragments expire or are deleted?
- Confidence rating validation (reject malformed NATO ratings)
- ID collision handling (uuid4 hex truncation)

### 5. DDD & domain boundaries

- Does each bounded context (`memory/`, `graph/`, `engine/`, `models/`, `audit/`) have clear responsibilities?
- Are domain invariants enforced within their owning domain (not leaked to `server.py`)?
- Does `server.py` (application layer) contain domain logic that belongs in a domain module?
- Are cross-domain dependencies explicit and unidirectional (no circular imports)?
- Does each domain follow the package convention: `schemas.py` (models), `store.py` (persistence), `__init__.py` (logic + re-exports)?
- Are aggregate roots clearly defined? Is there a domain service missing?
- Are value objects (confidence ratings, agent fingerprints) properly encapsulated or treated as raw strings?

### 6. MCP interface consistency

- Do all 3 tools return consistent response shapes (success vs error)?
- Are error messages uniform across tools (not_found, rejected, etc.)?
- Is input validation consistent across similar parameters?
- Are edge cases handled (empty strings, negative limits, invalid confidence ratings)?
- Do tool docstrings accurately describe behavior?

### 7. Technical & scientific approach

- **Memory model**: Is the biomimetic memory architecture (working memory → consolidation → long-term graph) well-founded? Are the biological analogies sound or misleading?
- **NATO confidence model**: Is the two-dimensional rating (source reliability A-F × information credibility 1-6) correctly implemented? Is the filtering logic (`_confidence_passes`) semantically correct (OR vs AND)?
- **Keyword search**: Is token-based keyword search an adequate retrieval strategy for the working memory layer? What are the known limitations vs. embedding-based search?
- **Consolidation pipeline**: Is the flush-threshold trigger the right mechanism, or should consolidation be time-based / event-driven?
- **Entity resolution**: Are the planned approaches (DD1 deep dive) realistic given the tech stack (Neo4j Community, no built-in ML)?
- **Ontology design**: Is the node type hierarchy (Fact, Event, Entity, Concept, Rule, Pattern) well-separated? Are there overlapping categories that will cause classification ambiguity?
- **Causal chain traversal**: Is bounded-depth BFS the right graph traversal strategy for `get_memory`? What are the failure modes (cycles, exponential fanout)?
- Flag any design decisions that are based on incorrect assumptions or that will hit fundamental limitations.

### 8. Documentation quality & completeness

- **CLAUDE.md/AGENTS.md**: Is it accurate and up to date? Does the project structure match actual files? Are all design docs listed?
- **Design docs** (`docs/design/*.md`): Does each implemented feature have a corresponding design doc? Are design docs consistent with actual implementation?
- **Design drafts** (`docs/drafts/*.md`): Are there inconsistencies between drafts (design-spec, ontology-schema, action-plan, deep-dives) and the actual code? Flag stale sections that no longer reflect reality.
- **README.md**: Does it accurately describe installation, requirements, and usage? Are there missing sections (configuration, MCP tool examples, architecture overview)?
- **Code-level docs**: Are MCP tool docstrings complete enough for an agent to use the tools correctly? Are Pydantic Field descriptions accurate?
- **Missing documentation**: Are there implemented features, configuration options, or important behaviors that have no documentation at all?
- Cross-reference all docs against code — flag any doc that promises something the code doesn't deliver, or code that does something no doc describes.

### 9. Performance (no premature optimization)

- Only flag **obvious** bottlenecks: blocking I/O in async context, N+1 Redis queries, unbounded result sets, missing pagination
- Redis SCAN or KEYS calls that could block on large datasets
- Search operations with O(n) fragment fetches that could use pipelines
- Do NOT flag theoretical performance concerns

### 10. Tests quality & determinism

- Coverage gaps on critical paths (MCP tools, correction flows, consolidation retries, retrieval shaping)
- Flaky patterns (timing-only waits, unbounded polling, non-deterministic fixtures)
- Marker hygiene (`unit`/`integration`/`scenario` + `tier1/2/3` + `real_llm`) and selection correctness
- Real-LLM test gating (strict opt-in, credential checks, accidental CI execution risk)
- Assertions quality: do tests validate behavior/contracts or only status/smoke

### 11. CI/CD & supply-chain

- Workflow correctness and drift (`checks.yml`, `ci.yml`, `security.yml`, scorecard)
- Artifact/report publication for debuggability (pytest XML, scenario metrics/calibration outputs)
- Dependency/update posture (`uv.lock`, Dependabot policy, pinned actions, integrity)
- Release-path safety (PyPI/GHCR workflows, branch protections assumptions, secret usage patterns)
- Security scanning gaps (SAST/dependency/license checks) and misconfigured triggers

### 12. Repo tooling & runtime ops

- Makefile/command targets accuracy vs documented workflow
- `pyproject.toml` consistency (pytest markers, tooling config, dependency groups)
- Script reliability (`scripts/*.py`) including exit codes, path assumptions, and error messages
- Local/dev ergonomics that can cause silent misuse (cache paths, env loading, report paths)
- Operational footguns in defaults (timeouts, retries, thresholds) that are not production-safe

## Output format

Produce a single consolidated report with this structure:

```
# Codebase Audit Report

## Critical (fix now)
- **[SCOPE] Title** — `file:line` — Description + concrete fix suggestion

## High (fix soon)
- ...

## Medium (plan to fix)
- ...

## Low (nice to have)
- ...

## Actions summary
| # | Action | Scope | Effort (S/M/L) | Files |
|---|--------|-------|-----------------|-------|
```

## Rules

- **Max 5 findings per scope.** If you find more, keep only the top 5 by impact.
- Target repository-wide coverage: include `src/`, `tests/`, `.github/`, `docs/`, `scripts/`, `Makefile`, and `pyproject.toml`.
- Every finding MUST have a **specific `file:line` reference** — no vague descriptions. For scopes 7-8 (technical/scientific, documentation), reference the relevant design doc or code location.
- Every finding MUST include a **concrete fix** (one-liner diff, pattern to apply, or approach).
- Do NOT flag things that are already correct (e.g., Pydantic validation is fine, don't mention it).
- Do NOT suggest adding comments, docstrings, or type annotations to code that works.
- Do NOT suggest over-engineered abstractions. Simple fixes only.
- Deduplicate across scopes — if the same root cause appears in multiple scopes, report it once under the most relevant scope.
