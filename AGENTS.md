# EngraMCP - Claude Code Reference

> **CLAUDE.md/AGENTS.md is your memory.** After any significant code change, verify this file reflects the current state of the project. Update it if needed.
> Note: `CLAUDE.md/AGENTS.md` refer to the same file. `CLAUDE.md` is a symlink to `AGENTS.md`.


---

## Development Cycle

Every sprint/feature follows this workflow:

1. **Plan** — Define scope, write tests first (TDD, outside-in)
2. **Implement** — Multiple commits if needed, each passing lint
3. **Review** — Self-review of the implementation
4. **Document** — Create `docs/design/<feature>.md` if the feature warrants it
5. **Update references** — Update this file (CLAUDE.md/AGENTS.md) and README.md
6. **Wrap-up** — Review remaining tasks, capitalize in `TODO.md` (gitignored), propose GitHub issues for items that warrant tracking
7. **PR** — Create pull request, review, merge to `main`
8. **Release** — If needed, tag with semver (vX.Y.Z)

### Task Tracking

`TODO.md` (gitignored) is our internal task list for quick items. At the end of each sprint/feature:
1. List remaining work and open questions in `TODO.md`
2. Propose creating GitHub issues for items that are significant, cross-sprint, or need discussion
3. Keep `TODO.md` clean: remove completed items, keep it current

### Reference Projects

When exploring external projects for patterns or reference, clone them into `external/` (gitignored). Never commit external project code.

---

## Design Documents

| Document | Description |
|---|---|
| [Layer Architecture](docs/design/layer-architecture.md) | 8-layer decomposition, data flows, mock replacement strategy |
| [MCP Interface](docs/design/mcp-interface.md) | Layer 7 design: 3 tools, Pydantic models, testing strategy, frozen contract |
| [Working Memory](docs/design/working-memory.md) | Layer 1 design: Redis-backed buffer, MemoryFragment model, keyword search |
| [Graph Store](docs/design/graph-store.md) | Layer 0+2 design: Neo4j CRUD, node/relation models, schema init, query methods |
| [Confidence Engine](docs/design/confidence-engine.md) | Layer 3 design: source traceability, independence detection, NATO confidence, propagation |
| [Config & Audit](docs/design/config-audit.md) | Config dataclasses, async JSONL audit logger, AuditEventType enum |
| [Extraction](docs/design/extraction.md) | Layer 4 design: LLMAdapter protocol, ExtractionEngine, prompt builder, extraction schemas |
| [Entity Resolution](docs/design/entity-resolution.md) | Layer 4 design: three-level resolver, normalizer, scorer, merge executor, anti-patterns |
| [Consolidation Pipeline](docs/design/consolidation-pipeline.md) | Layer 4: async trigger wiring, extraction integration, idempotent claim/source consolidation, contradiction detection, and abstraction stages |
| [Retrieval Engine](docs/design/retrieval.md) | Layer 6 design: WM-first retrieval service, scoring protocol, bounded graph-context fallback (`max_depth`) across claim + derived nodes, demand tracking hooks |
| [Evaluation Scenarios](docs/design/evaluation-scenarios.md) | Tiered eval strategy (Tier 1/2/3), scenario layout under `tests/scenarios/`, markers, and CI/reporting conventions |

### Draft References (Archive / Migration Input)

| Draft | Description |
|---|---|
| [Design Spec](docs/drafts/design-spec.md) | Historical architecture draft (reference only) |
| [Ontology Schema](docs/drafts/ontology-schema.md) | Historical ontology draft (reference only) |
| [Action Plan](docs/drafts/action-plan.md) | Historical delivery plan (reference only) |
| [Deep Dives](docs/drafts/deep-dives.md) | Historical deep dives (reference only) |
| [Evaluation Scenarios Draft](docs/drafts/evaluation-scenarios.md) | Archived scenario ideation; canonical eval guidance now in `docs/design/evaluation-scenarios.md` |

## Community & Governance Documents

| Document | Description |
|---|---|
| [Contributing](CONTRIBUTING.md) | Contribution workflow, local setup, coding/review expectations |
| [Reviewing](REVIEWING.md) | PR review goals, reviewer/author checklists, security-sensitive review rules |
| [Maintainers](MAINTAINERS.md) | Maintainer roster, responsibilities, appointment policy |
| [Code of Conduct](CODE_OF_CONDUCT.md) | Community behavior standards, reporting and enforcement policy |
| [Support](SUPPORT.md) | Support channels and issue reporting expectations |
| [Security](SECURITY.md) | Private vulnerability reporting and coordinated disclosure policy |

---

## Architecture

Three-layer biomimetic memory engine exposed via MCP:

```
Agent → send_memory → [Working Memory] → (async consolidation) → [Neo4j Knowledge Graph]
Agent → get_memory  → [Retrieval Engine] → Working Memory + Graph traversal → Structured response
Agent → correct_memory → [WM-first contest/annotate + graph-aware split/merge/reclassify (WM fallback) + audit]
```

### Layers (top-down)

| Layer | Module | Description |
|---|---|---|
| 7 | `server.py` | MCP interface + consolidation/retrieval assembly/wiring + correction audit flows (WM-first `contest`/`annotate`; graph-aware `split_entity`/`merge_entities`/derived-lifecycle-aware `reclassify` with WM fallback where applicable; consolidation requires explicit LLM provider wiring via `LLMConfig`/adapter) |
| 6 | `engine/retrieval.py` | WM-first retrieval, bounded graph-context fallback (`max_depth`) over claim + derived nodes, scoring, synthesis, demand-hook emission |
| 5 | `engine/concepts.py`, `engine/demand.py` | Concept emergence from retrieval demand |
| 4 | `engine/consolidation.py`, `engine/extraction.py` | Async batch pipeline, LLM extraction, idempotent claim/source writes for repeated runs, contradiction detection, abstraction |
| 3 | `engine/confidence.py` | NATO rating, propagation, corroboration |
| 2 | `graph/store.py`, `graph/schema.py` | Neo4j CRUD, ontology, constraints |
| 1 | `memory/store.py` | Redis-backed buffer, TTL, keyword search |
| 0 | `models/` | Schema, indexes, constraints definitions |

### Stack

| Component | Technology |
|---|---|
| Language | Python 3.13+ |
| MCP Server | FastMCP v2 |
| Working Memory | Redis (testcontainers for tests) |
| Graph Database | Neo4j (testcontainers for tests) |
| Test Framework | pytest |
| LLM (consolidation) | Configurable (provider/model in config) |

---

## Project Structure

```
src/engramcp/
├── __init__.py
├── server.py               # FastMCP server, 3 tools
├── config.py               # LLM, consolidation, entity resolution, audit config
├── evaluation.py           # Shared scenario-eval pass/fail thresholds
├── models/                 # Shared data models
│   ├── __init__.py         # Agent fingerprinting + domain logic + re-exports
│   ├── schemas.py          # Pydantic input/output schemas for MCP tools
│   ├── nodes.py            # 11 node type models + LABEL_TO_MODEL mapping
│   ├── relations.py        # 18 relationship type models
│   └── confidence.py       # NATORating, Reliability, Credibility + helpers
├── memory/                 # Working memory
│   ├── __init__.py         # Domain API, re-exports
│   ├── schemas.py          # MemoryFragment model
│   └── store.py            # Redis-backed buffer, keyword search, non-blocking flush callback
├── graph/                  # Neo4j layer
│   ├── __init__.py         # Re-exports GraphStore, SourceTraceability, init_schema
│   ├── store.py            # CRUD operations + query methods
│   ├── schema.py           # Index/constraint init
│   ├── entity_resolution.py # Three-level entity resolution
│   └── traceability.py     # Source chain traversal, independence detection
├── engine/                 # Processing engines (Layers 3-6)
│   ├── __init__.py         # Re-exports ConfidenceEngine, ExtractionEngine, etc.
│   ├── confidence.py       # Confidence calculation & propagation
│   ├── concepts.py         # Concept candidate registry + lifecycle states
│   ├── demand.py           # Retrieval query pattern tracking + threshold signal emission
│   ├── schemas.py          # Extraction result models
│   ├── extraction.py       # LLMAdapter protocol + ExtractionEngine
│   ├── llm_adapters.py     # Concrete adapters (`openai`, `noop`) + adapter factory
│   ├── prompt_builder.py   # Dynamic extraction prompt
│   ├── consolidation.py    # Consolidation pipeline orchestrator + idempotent claim/source writes + contradiction/abstraction stages
│   └── retrieval.py        # Layer 6 retrieval service + scoring protocol + bounded graph-context fallback (claim + derived nodes)
└── audit/                  # Audit logging
    ├── __init__.py         # Re-exports AuditLogger, AuditEvent, AuditEventType
    ├── schemas.py          # AuditEventType enum + AuditEvent model
    └── store.py            # Async JSONL audit logger
```

---

## Code Patterns & Conventions

- **Language**: All code, comments, documentation, PRs, commits, and issues must be written in **English (US)**
- **Commits**: Never mention sprints in commit messages (internal concept). Use conventional commits (`chore:`, `feat:`, `fix:`, `docs:`, `refactor:`, `test:`)
- **TDD outside-in**: tests written before implementation
- **Package manager**: Always use `uv` to run Python commands (e.g. `uv run pytest`, `uv run python`)
- **Linting**: black, flake8, isort, mypy, pyupgrade (pre-commit)
- **Async**: pytest-asyncio with `asyncio_mode = "auto"`
- **Testing**: testcontainers for Neo4j and Redis (session-scoped fixtures)
- **Pytest markers**: tests are auto-labeled by path (`unit`, `integration`, `scenario`); scenario tier markers are explicit (`tier1`, `tier2`, `tier3`); `real_llm` is reserved for opt-in provider-backed evals
- **Test env loading**: pytest loads root `.env` (via `python-dotenv` in `tests/conftest.py`) for explicit opt-in test flags/credentials
- **Scenario eval location**: all Tier 1, Tier 2, and Tier 3 evaluation suites must live under `tests/scenarios/` (tests, fixtures, and helpers)
- **Real-LLM evals**: optional opt-in E2E evals live in `tests/scenarios/test_e2e_real_llm_eval.py` and require explicit environment opt-in + provider credentials
- **Real-LLM test execution policy**: always ask the user for explicit confirmation before running real-LLM evals (`ENGRAMCP_RUN_REAL_LLM_EVALS=1`), even when `.env` is present and fully configured
- **Scenario command targets**: use `make test-scenarios` for CI-safe evals (non-`real_llm`), `make test-scenarios-tier2` for curated Tier 2 iteration, `make calibrate-eval-thresholds` to generate calibration outputs from scenario metrics, and `make test-real-llm-evals` for explicit opt-in provider-backed runs
- **Scenario metrics/calibration artifacts**: CI-safe scenario runs emit JSONL metrics (`reports/scenario-metrics.jsonl`) and calibration reports (`reports/eval-calibration.json`) used to track pass/fail metric classes and threshold recommendations
- **Confidence**: NATO two-dimensional rating (letter = source reliability, number = credibility)
- **Confidence on relations, not nodes**: same fact can have different ratings from different sources
- **MCP errors**: tool responses may include `error_code` and `message` when rejected/errored
- **LLM provider wiring**: consolidation no longer uses implicit noop fallback; configure explicit `llm_config` provider (or inject `llm_adapter` in code/tests)
- **Scenario eval config profile**: use `scenario_eval_consolidation_config()` from `config.py` for deterministic scenario-only tuning (`fragment_threshold=4`, `pattern_min_occurrences=2`) instead of hardcoding values in tests
- **Extraction failure policy**: extraction output is schema-validated (`ExtractionResult`) and supports configurable retries for provider errors/invalid JSON/schema validation failures via `ConsolidationConfig` retry fields
- **Graph retrieval matching**: graph lookup by content (claims + derived nodes) uses tokenized query matching (ANY token contained) rather than full-query substring matching
- **Graph causal traversal query**: causal-chain retrieval filters relationship types at runtime (`type(rel) IN [...]`) to keep behavior stable while avoiding noisy Neo4j unknown-type notifications in sparse graphs/tests
- **DDD (Domain-Driven Design)**: each domain has bounded contexts with `models/`, `memory/`, `graph/`, `engine/`, `audit/` modules. Domain logic stays in its module; cross-cutting concerns use explicit interfaces.
- **Domain package structure**: each domain follows `schemas.py` (Pydantic models), `store.py` (DB access), `__init__.py` (business logic + re-exports). External code imports from the domain package (e.g. `from engramcp.memory import MemoryFragment`).
- **Import-cycle guard**: package re-exports in `graph/__init__.py` use lazy loading (`__getattr__`) to avoid eager cross-domain import cycles during bootstrap/tests.
- **Inline code markers**: use `# TODO:` for intentionally deferred work, and `# DEPRECATED: <reason>` when code is kept only for backward compatibility.

---

## Documentation Maintenance

### README.md
Public-facing file for users/contributors. Update when:
- New user-visible major feature
- Changes to requirements or installation
- Modifications to public config structure

### GitHub Issue Templates
Issue intake files in `.github/ISSUE_TEMPLATE/`. Update when:
- Issue taxonomy changes (bug/feature/question/improvement categories)
- Required triage metadata changes
- Security/support routing links change

### CLAUDE.md/AGENTS.md (this file)
Technical reference for Claude. Update when:
- After any significant code modification
- New pattern or convention established
- Architecture or flow changes
- New design document created (add to reference table)

**Post-modification checklist:**
1. Is the described architecture still accurate?
2. Are the listed enums/types up to date?
3. Is the project structure up to date?
4. Is there a new limitation to document?
5. Are design doc references complete?
6. **Tests:** If the modified code is testable (pure logic), are there unit tests?
