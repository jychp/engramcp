# EngraMCP - Claude Code Reference

> **CLAUDE.md is your memory.** After any significant code change, verify this file reflects the current state of the project. Update it if needed.

---

## Development Cycle

Every sprint/feature follows this workflow:

1. **Plan** — Define scope, write tests first (TDD, outside-in)
2. **Implement** — Multiple commits if needed, each passing lint
3. **Review** — Self-review of the implementation
4. **Document** — Create `docs/design/<feature>.md` if the feature warrants it
5. **Update references** — Update this file (CLAUDE.md) and README.md
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
| [Design Spec](docs/drafts/design-spec.md) | Architecture, MCP interface, ontology, confidence model |
| [Ontology Schema](docs/drafts/ontology-schema.md) | Neo4j node types, relationships, indexes, constraints |
| [Action Plan](docs/drafts/action-plan.md) | TDD sprint plan, layer decomposition, repo structure |
| [Deep Dives](docs/drafts/deep-dives.md) | DD1: Entity Resolution, DD2: get_memory format, DD3: Concept Emergence |
| [MCP Interface](docs/design/mcp-interface.md) | Layer 7 design: 3 tools, Pydantic models, testing strategy, frozen contract |

---

## Architecture

Three-layer biomimetic memory engine exposed via MCP:

```
Agent → send_memory → [Working Memory] → (async consolidation) → [Neo4j Knowledge Graph]
Agent → get_memory  → [Retrieval Engine] → Working Memory + Graph traversal → Structured response
Agent → correct_memory → [Graph mutations + cascade]
```

### Layers (top-down)

| Layer | Module | Description |
|---|---|---|
| 7 | `server.py` | MCP Interface (send_memory, get_memory, correct_memory) |
| 6 | `engine/retrieval.py` | Graph traversal, scoring, synthesis |
| 5 | `engine/concepts.py`, `engine/demand.py` | Concept emergence from retrieval demand |
| 4 | `engine/consolidation.py`, `engine/extraction.py` | Async batch pipeline, LLM extraction |
| 3 | `engine/confidence.py` | NATO rating, propagation, corroboration |
| 2 | `graph/store.py`, `graph/schema.py` | Neo4j CRUD, ontology, constraints |
| 1 | `memory/working.py` | In-memory buffer, TTL, flush-to-disk |
| 0 | `models/` | Schema, indexes, constraints definitions |

### Stack

| Component | Technology |
|---|---|
| Language | Python 3.13+ |
| MCP Server | FastMCP v2 |
| Graph Database | Neo4j (testcontainers for tests) |
| Test Framework | pytest |
| LLM (consolidation) | Configurable (provider/model in config) |

---

## Project Structure

```
src/engramcp/
├── __init__.py
├── server.py               # FastMCP server, 3 tools (✅ Sprint 1)
├── config.py               # LLM provider/model, thresholds, paths
├── models/                 # Data models
│   ├── __init__.py
│   ├── mcp.py              # Pydantic input/output schemas for MCP tools (✅ Sprint 1)
│   ├── nodes.py            # Node type definitions
│   ├── relations.py        # Relationship type definitions
│   ├── confidence.py       # NATO rating model
│   └── agent.py            # Agent fingerprinting
├── memory/                 # Working memory
│   ├── __init__.py
│   ├── working.py          # In-memory buffer
│   └── persistence.py      # Flush-to-disk / restore
├── graph/                  # Neo4j layer
│   ├── __init__.py
│   ├── store.py            # CRUD operations
│   ├── schema.py           # Index/constraint init
│   ├── entity_resolution.py
│   └── traceability.py     # Source chain management
├── engine/                 # Processing engines
│   ├── __init__.py
│   ├── confidence.py       # Confidence calculation & propagation
│   ├── consolidation.py    # Async batch pipeline
│   ├── extraction.py       # LLM extraction adapter
│   ├── abstraction.py      # Pattern/Concept/Rule emergence
│   ├── concepts.py         # Concept registry, stabilization
│   ├── demand.py           # Query pattern tracker
│   ├── prompt_builder.py   # Dynamic extraction prompt
│   └── retrieval.py        # Graph traversal & scoring
└── audit/
    ├── __init__.py
    └── logger.py           # JSONL audit log writer
```

---

## Code Patterns & Conventions

- **Language**: All code, comments, documentation, PRs, commits, and issues must be written in **English (US)**
- **Commits**: Never mention sprints in commit messages (internal concept). Use conventional commits (`chore:`, `feat:`, `fix:`, `docs:`, `refactor:`, `test:`)
- **TDD outside-in**: tests written before implementation
- **Linting**: black, flake8, isort, mypy, pyupgrade (pre-commit)
- **Async**: pytest-asyncio with `asyncio_mode = "auto"`
- **Testing**: testcontainers for Neo4j (session-scoped fixture)
- **Confidence**: NATO two-dimensional rating (letter = source reliability, number = credibility)
- **Confidence on relations, not nodes**: same fact can have different ratings from different sources
- **DDD (Domain-Driven Design)**: each domain has bounded contexts with `models/`, `memory/`, `graph/`, `engine/`, `audit/` modules. Domain logic stays in its module; cross-cutting concerns use explicit interfaces.

---

## Documentation Maintenance

### README.md
Public-facing file for users/contributors. Update when:
- New user-visible major feature
- Changes to requirements or installation
- Modifications to public config structure

### CLAUDE.md (this file)
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
