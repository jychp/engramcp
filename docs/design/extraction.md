# LLM Extraction Adapter — Layer 4 (partial)

> Design document for Sprint 6b: the extraction component of the consolidation
> engine. Takes raw `MemoryFragment` objects and uses an LLM to extract
> structured entities, relations, and claims.

---

## Overview

The extraction adapter sits between working memory (Layer 1) and the
consolidation pipeline (Layer 4 orchestrator, Sprint 6d). Its job:

1. Accept a batch of `MemoryFragment` objects
2. Build a structured prompt containing fragment data and the expected output
   schema
3. Call an LLM through the `LLMAdapter` protocol
4. Parse the response into validated `ExtractionResult` objects
5. Handle errors gracefully (partial success, invalid JSON, timeouts)

```
MemoryFragment[] ──► PromptBuilder ──► LLMAdapter.complete() ──► JSON parse ──► ExtractionResult
                                         ▲
                                    LLMConfig
```

---

## Components

### 1. LLMAdapter (Protocol)

A `@runtime_checkable` protocol with a single method:

```python
class LLMAdapter(Protocol):
    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str: ...
```

Concrete adapters (OpenAI, Anthropic, etc.) are **not in scope** for this
sprint. Tests use a `MockLLMAdapter` that returns pre-built JSON.

### 2. LLMError

Custom exception raised by adapters when a call fails (timeout, rate limit,
API error). The extraction engine catches this and records it in
`ExtractionResult.errors`.

### 3. ExtractionEngine

Main class orchestrating the extraction flow.

**Constructor:**
```python
ExtractionEngine(
    llm: LLMAdapter,
    llm_config: LLMConfig,
    consolidation_config: ConsolidationConfig,
)
```

**Public API:**
```python
async def extract(self, fragments: list[MemoryFragment]) -> ExtractionResult
```

**Internal flow:**

1. If `fragments` is empty, return an empty `ExtractionResult` immediately
   (no LLM call).
2. Split fragments into sub-batches of
   `consolidation_config.extraction_batch_size`.
3. Process each batch sequentially (avoids rate limits, simpler error
   handling).
4. For each batch: build prompt, call `llm.complete()`, parse output.
5. Merge all batch results into a single `ExtractionResult`.
6. Failed batches append an error message but do not abort the pipeline.

### 4. PromptBuilder

Single function:

```python
def build_extraction_prompt(fragments: list[MemoryFragment]) -> str
```

**Prompt structure:**

1. **System instructions** — role description and extraction task
2. **Fragment data** — delimited blocks with ID, content, type, confidence,
   participants, sources
3. **Extraction instructions** — what to extract (entities, relations,
   claims) with field specs
4. **Available types** — entity types (from ontology) and relation types
5. **Output format** — JSON schema auto-generated from
   `ExtractionResult.model_json_schema()`

Separate module because it will evolve independently when concept emergence
enriches prompts (Sprint 6e).

---

## Schemas

Five Pydantic models in `engine/schemas.py`:

### TemporalInfo

| Field | Type | Default | Description |
|---|---|---|---|
| `occurred_at` | `str \| None` | `None` | When it happened (ISO-8601 string) |
| `occurred_until` | `str \| None` | `None` | End of time range |
| `precision` | `str` | `"unknown"` | Granularity hint |

Uses **strings** (not `datetime`) because parsing to `datetime` +
`TemporalPrecision` enum happens in the consolidation pipeline (6d).

### ExtractedEntity

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Primary name |
| `type` | `str` | required | Ontology node type |
| `aliases` | `list[str]` | `[]` | Alternative names |
| `properties` | `dict` | `{}` | Arbitrary metadata |
| `disambiguating_context` | `str \| None` | `None` | Context to help entity resolution |
| `source_fragment_ids` | `list[str]` | `[]` | Which fragments mentioned this |

Entities reference by **name**, not ID. Entity resolution (Sprint 6c)
maps names to graph nodes.

### ExtractedRelation

| Field | Type | Default | Description |
|---|---|---|---|
| `from_entity` | `str` | required | Source entity name |
| `to_entity` | `str` | required | Target entity name |
| `relation_type` | `str` | required | Ontology relation type |
| `properties` | `dict` | `{}` | Arbitrary metadata |
| `source_fragment_ids` | `list[str]` | `[]` | Which fragments mentioned this |

### ExtractedClaim

| Field | Type | Default | Description |
|---|---|---|---|
| `content` | `str` | required | The claim text |
| `claim_type` | `str` | `"Fact"` | Ontology type |
| `confidence_hint` | `str \| None` | `None` | NATO hint (e.g. "B2") |
| `temporal_info` | `TemporalInfo \| None` | `None` | When it happened |
| `properties` | `dict` | `{}` | Arbitrary metadata |
| `involved_entities` | `list[str]` | `[]` | Entity names involved |
| `source_fragment_ids` | `list[str]` | `[]` | Which fragments support this |

### ExtractionResult

| Field | Type | Default | Description |
|---|---|---|---|
| `entities` | `list[ExtractedEntity]` | `[]` | Extracted entities |
| `relations` | `list[ExtractedRelation]` | `[]` | Extracted relations |
| `claims` | `list[ExtractedClaim]` | `[]` | Extracted claims |
| `fragment_ids_processed` | `list[str]` | `[]` | Which fragments were processed |
| `errors` | `list[str]` | `[]` | Error messages from failed batches |

The `errors` field enables **partial success**: failed batches don't abort
everything.

---

## Batching Strategy

Fragments are split into sub-batches based on
`ConsolidationConfig.extraction_batch_size` (default 5).

- **Sequential processing**: batches are processed one at a time to avoid
  rate limits and simplify error handling.
- **Merge**: results from all successful batches are concatenated.
- **Partial failure**: if batch N fails, results from batches 0..N-1 are
  kept, and the error is recorded.

---

## Error Handling

| Scenario | Behavior |
|---|---|
| `LLMError` raised | Empty result for that batch, error recorded |
| Invalid JSON returned | Empty result for that batch, error recorded |
| JSON with code fences | Code fences stripped before parsing |
| Empty fragment list | Return empty result immediately, no LLM call |
| Partial batch failure | Keep successful batch results, record errors |

---

## Not in Scope

- Concrete LLM adapters (OpenAI, Anthropic) — added when wiring real providers
- Entity resolution — Sprint 6c
- Graph integration (creating nodes from extraction results) — Sprint 6d
- Abstraction (Pattern/Concept/Rule emergence) — Sprint 6e
- Audit logging of extraction — Sprint 6d (orchestrator logs)
- Prompt enrichment with known concepts — Sprint 7
