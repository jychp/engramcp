"""Tests for the LLM extraction adapter (Layer 4 partial).

All pure unit tests — no containers. The LLM is mocked via MockLLMAdapter.
"""

from __future__ import annotations

import json

import pytest

from engramcp.config import ConsolidationConfig
from engramcp.config import LLMConfig
from engramcp.engine.extraction import ExtractionEngine
from engramcp.engine.extraction import LLMAdapter
from engramcp.engine.extraction import LLMError
from engramcp.engine.prompt_builder import build_extraction_prompt
from engramcp.engine.prompt_builder import ENTITY_TYPES
from engramcp.engine.prompt_builder import RELATION_TYPES
from engramcp.engine.schemas import ExtractedClaim
from engramcp.engine.schemas import ExtractedEntity
from engramcp.engine.schemas import ExtractedRelation
from engramcp.engine.schemas import ExtractionResult
from engramcp.engine.schemas import TemporalInfo
from engramcp.memory.schemas import MemoryFragment


# ---------------------------------------------------------------------------
# Mock LLM adapter
# ---------------------------------------------------------------------------


class MockLLMAdapter:
    """Test double for LLMAdapter.

    Returns a canned JSON response. Tracks calls for assertion.
    """

    def __init__(self, response: str | None = None) -> None:
        self.calls: list[dict] = []
        self._response = response or json.dumps(
            {
                "entities": [
                    {
                        "name": "Alice",
                        "type": "Agent",
                        "source_fragment_ids": ["mem_abc"],
                    }
                ],
                "relations": [
                    {
                        "from_entity": "Alice",
                        "to_entity": "Acme Corp",
                        "relation_type": "PARTICIPATED_IN",
                        "source_fragment_ids": ["mem_abc"],
                    }
                ],
                "claims": [
                    {
                        "content": "Alice joined Acme Corp in 2024",
                        "claim_type": "Fact",
                        "involved_entities": ["Alice", "Acme Corp"],
                        "source_fragment_ids": ["mem_abc"],
                    }
                ],
                "fragment_ids_processed": ["mem_abc"],
                "errors": [],
            }
        )

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self._response


class FailingLLMAdapter:
    """Mock that always raises LLMError."""

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        raise LLMError("API timeout")


class InvalidJSONLLMAdapter:
    """Mock that returns invalid JSON."""

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        return "this is not json at all {"


class CodeFenceLLMAdapter:
    """Mock that returns JSON wrapped in code fences."""

    def __init__(self, data: dict) -> None:
        self._data = data

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        return f"```json\n{json.dumps(self._data)}\n```"


class FailThenSucceedLLMAdapter:
    """Mock that fails once with LLMError, then returns valid JSON."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        del prompt, temperature, max_tokens, timeout_seconds
        self.calls += 1
        if self.calls == 1:
            raise LLMError("temporary upstream issue")
        return json.dumps(
            {
                "entities": [{"name": "RetrySuccess", "type": "Agent"}],
                "relations": [],
                "claims": [],
            }
        )


class InvalidJSONThenSucceedLLMAdapter:
    """Mock that returns invalid JSON first, then valid JSON."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        del prompt, temperature, max_tokens, timeout_seconds
        self.calls += 1
        if self.calls == 1:
            return '{"entities": [}'
        return json.dumps(
            {
                "entities": [{"name": "Recovered", "type": "Agent"}],
                "relations": [],
                "claims": [],
            }
        )


class InvalidSchemaThenSucceedLLMAdapter:
    """Mock that returns schema-invalid payload first, then valid JSON."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        del prompt, temperature, max_tokens, timeout_seconds
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "entities": [{"name": "Broken"}],  # missing required `type`
                    "relations": [],
                    "claims": [],
                }
            )
        return json.dumps(
            {
                "entities": [{"name": "SchemaRecovered", "type": "Agent"}],
                "relations": [],
                "claims": [],
            }
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fragment(
    content: str = "Alice joined Acme Corp",
    frag_id: str = "mem_abc",
    frag_type: str = "Fact",
    confidence: str | None = "B2",
) -> MemoryFragment:
    return MemoryFragment(
        id=frag_id,
        content=content,
        type=frag_type,
        confidence=confidence,
    )


def _make_fragments(n: int) -> list[MemoryFragment]:
    return [
        _make_fragment(
            content=f"Fragment content {i}",
            frag_id=f"mem_{i:04d}",
        )
        for i in range(n)
    ]


# ====================================================================
# TestExtractionSchemas
# ====================================================================


class TestExtractionSchemas:
    def test_extracted_entity_has_required_fields(self) -> None:
        entity = ExtractedEntity(name="Alice", type="Agent")
        assert entity.name == "Alice"
        assert entity.type == "Agent"
        assert entity.aliases == []
        assert entity.properties == {}
        assert entity.disambiguating_context is None
        assert entity.source_fragment_ids == []

    def test_extracted_relation_has_required_fields(self) -> None:
        rel = ExtractedRelation(
            from_entity="Alice",
            to_entity="Acme Corp",
            relation_type="PARTICIPATED_IN",
        )
        assert rel.from_entity == "Alice"
        assert rel.to_entity == "Acme Corp"
        assert rel.relation_type == "PARTICIPATED_IN"
        assert rel.properties == {}
        assert rel.source_fragment_ids == []

    def test_extracted_claim_has_required_fields(self) -> None:
        claim = ExtractedClaim(content="Alice joined Acme Corp in 2024")
        assert claim.content == "Alice joined Acme Corp in 2024"
        assert claim.claim_type == "Fact"
        assert claim.confidence_hint is None
        assert claim.temporal_info is None
        assert claim.properties == {}
        assert claim.involved_entities == []
        assert claim.source_fragment_ids == []

    def test_extraction_result_defaults_to_empty(self) -> None:
        result = ExtractionResult()
        assert result.entities == []
        assert result.relations == []
        assert result.claims == []
        assert result.fragment_ids_processed == []
        assert result.errors == []

    def test_temporal_info_defaults(self) -> None:
        info = TemporalInfo()
        assert info.occurred_at is None
        assert info.occurred_until is None
        assert info.precision == "unknown"


# ====================================================================
# TestBatchExtraction
# ====================================================================


class TestBatchExtraction:
    async def test_batch_extracts_entities_from_fragments(self) -> None:
        llm = MockLLMAdapter()
        engine = ExtractionEngine(llm=llm)
        fragments = [_make_fragment()]

        result = await engine.extract(fragments)

        assert len(result.entities) == 1
        assert result.entities[0].name == "Alice"
        assert result.entities[0].type == "Agent"

    async def test_extraction_output_is_valid_json(self) -> None:
        """The mock response is valid JSON parseable to ExtractionResult."""
        llm = MockLLMAdapter()
        engine = ExtractionEngine(llm=llm)
        fragments = [_make_fragment()]

        result = await engine.extract(fragments)

        # Should be a valid ExtractionResult with no errors
        assert isinstance(result, ExtractionResult)
        assert result.errors == []

    async def test_extracts_relations_from_fragments(self) -> None:
        llm = MockLLMAdapter()
        engine = ExtractionEngine(llm=llm)
        fragments = [_make_fragment()]

        result = await engine.extract(fragments)

        assert len(result.relations) == 1
        assert result.relations[0].from_entity == "Alice"
        assert result.relations[0].to_entity == "Acme Corp"
        assert result.relations[0].relation_type == "PARTICIPATED_IN"

    async def test_extracts_temporal_info(self) -> None:
        response = json.dumps(
            {
                "entities": [],
                "relations": [],
                "claims": [
                    {
                        "content": "Meeting on 2024-01-15",
                        "temporal_info": {
                            "occurred_at": "2024-01-15",
                            "precision": "day",
                        },
                        "source_fragment_ids": ["mem_abc"],
                    }
                ],
            }
        )
        llm = MockLLMAdapter(response=response)
        engine = ExtractionEngine(llm=llm)

        result = await engine.extract([_make_fragment()])

        assert len(result.claims) == 1
        assert result.claims[0].temporal_info is not None
        assert result.claims[0].temporal_info.occurred_at == "2024-01-15"
        assert result.claims[0].temporal_info.precision == "day"

    async def test_extracts_disambiguating_context(self) -> None:
        response = json.dumps(
            {
                "entities": [
                    {
                        "name": "Paris",
                        "type": "Agent",
                        "disambiguating_context": "Paris, the city in France",
                        "source_fragment_ids": ["mem_abc"],
                    }
                ],
                "relations": [],
                "claims": [],
            }
        )
        llm = MockLLMAdapter(response=response)
        engine = ExtractionEngine(llm=llm)

        result = await engine.extract([_make_fragment()])

        assert result.entities[0].disambiguating_context == "Paris, the city in France"


# ====================================================================
# TestBatching
# ====================================================================


class TestBatching:
    async def test_fragments_split_into_batches(self) -> None:
        """7 fragments / batch_size=5 -> 2 LLM calls."""
        llm = MockLLMAdapter()
        config = ConsolidationConfig(extraction_batch_size=5)
        engine = ExtractionEngine(llm=llm, consolidation_config=config)

        await engine.extract(_make_fragments(7))

        assert len(llm.calls) == 2

    async def test_single_batch_when_under_threshold(self) -> None:
        """3 fragments / batch_size=5 -> 1 LLM call."""
        llm = MockLLMAdapter()
        config = ConsolidationConfig(extraction_batch_size=5)
        engine = ExtractionEngine(llm=llm, consolidation_config=config)

        await engine.extract(_make_fragments(3))

        assert len(llm.calls) == 1

    async def test_results_merged_across_batches(self) -> None:
        """Results from multiple batches are combined."""
        llm = MockLLMAdapter()
        config = ConsolidationConfig(extraction_batch_size=3)
        engine = ExtractionEngine(llm=llm, consolidation_config=config)

        result = await engine.extract(_make_fragments(6))

        # 2 batches, each returns 1 entity -> 2 total
        assert len(result.entities) == 2
        assert len(result.relations) == 2
        assert len(result.claims) == 2

    async def test_fragment_ids_tracked(self) -> None:
        """All fragment IDs appear in fragment_ids_processed."""
        llm = MockLLMAdapter()
        config = ConsolidationConfig(extraction_batch_size=3)
        engine = ExtractionEngine(llm=llm, consolidation_config=config)
        fragments = _make_fragments(5)

        result = await engine.extract(fragments)

        expected_ids = {f.id for f in fragments}
        actual_ids = set(result.fragment_ids_processed)
        assert expected_ids.issubset(actual_ids)


# ====================================================================
# TestErrorHandling
# ====================================================================


class TestErrorHandling:
    async def test_handles_extraction_failure_gracefully(self) -> None:
        """LLMError -> empty result + error message."""
        llm = FailingLLMAdapter()
        engine = ExtractionEngine(llm=llm)

        result = await engine.extract([_make_fragment()])

        assert result.entities == []
        assert len(result.errors) == 1
        assert "LLM call failed" in result.errors[0]

    async def test_handles_invalid_json_gracefully(self) -> None:
        llm = InvalidJSONLLMAdapter()
        engine = ExtractionEngine(llm=llm)

        result = await engine.extract([_make_fragment()])

        assert result.entities == []
        assert len(result.errors) == 1
        assert "Invalid JSON" in result.errors[0]

    async def test_handles_partial_batch_failure(self) -> None:
        """Batch 1 OK + batch 2 fails -> batch 1 results kept."""
        call_count = 0

        class PartialFailAdapter:
            async def complete(
                self,
                prompt: str,
                *,
                temperature: float = 0.2,
                max_tokens: int = 4096,
                timeout_seconds: float = 30.0,
            ) -> str:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return json.dumps(
                        {
                            "entities": [
                                {"name": "Bob", "type": "Agent"},
                            ],
                            "relations": [],
                            "claims": [],
                        }
                    )
                raise LLMError("Rate limit exceeded")

        config = ConsolidationConfig(extraction_batch_size=2)
        engine = ExtractionEngine(llm=PartialFailAdapter(), consolidation_config=config)

        result = await engine.extract(_make_fragments(4))

        # Batch 1 succeeded
        assert len(result.entities) == 1
        assert result.entities[0].name == "Bob"
        # Batch 2 failed
        assert len(result.errors) == 1
        assert "Rate limit" in result.errors[0]

    async def test_handles_json_with_code_fences(self) -> None:
        """```json ... ``` stripped before parsing."""
        data = {
            "entities": [{"name": "Charlie", "type": "Agent"}],
            "relations": [],
            "claims": [],
        }
        llm = CodeFenceLLMAdapter(data)
        engine = ExtractionEngine(llm=llm)

        result = await engine.extract([_make_fragment()])

        assert len(result.entities) == 1
        assert result.entities[0].name == "Charlie"
        assert result.errors == []

    async def test_handles_empty_fragments_list(self) -> None:
        """No LLM call when fragment list is empty."""
        llm = MockLLMAdapter()
        engine = ExtractionEngine(llm=llm)

        result = await engine.extract([])

        assert result == ExtractionResult()
        assert len(llm.calls) == 0

    async def test_retries_llm_error_and_recovers(self) -> None:
        llm = FailThenSucceedLLMAdapter()
        config = ConsolidationConfig(extraction_max_retries=1)
        engine = ExtractionEngine(llm=llm, consolidation_config=config)

        result = await engine.extract([_make_fragment()])

        assert llm.calls == 2
        assert result.errors == []
        assert [entity.name for entity in result.entities] == ["RetrySuccess"]

    async def test_retries_invalid_json_and_recovers(self) -> None:
        llm = InvalidJSONThenSucceedLLMAdapter()
        config = ConsolidationConfig(extraction_max_retries=1)
        engine = ExtractionEngine(llm=llm, consolidation_config=config)

        result = await engine.extract([_make_fragment()])

        assert llm.calls == 2
        assert result.errors == []
        assert [entity.name for entity in result.entities] == ["Recovered"]

    async def test_does_not_retry_invalid_json_when_disabled(self) -> None:
        llm = InvalidJSONThenSucceedLLMAdapter()
        config = ConsolidationConfig(
            extraction_max_retries=1,
            retry_on_invalid_json=False,
        )
        engine = ExtractionEngine(llm=llm, consolidation_config=config)

        result = await engine.extract([_make_fragment()])

        assert llm.calls == 1
        assert len(result.errors) == 1
        assert "Invalid JSON" in result.errors[0]

    async def test_retries_schema_validation_error_and_recovers(self) -> None:
        llm = InvalidSchemaThenSucceedLLMAdapter()
        config = ConsolidationConfig(extraction_max_retries=1)
        engine = ExtractionEngine(llm=llm, consolidation_config=config)

        result = await engine.extract([_make_fragment()])

        assert llm.calls == 2
        assert result.errors == []
        assert [entity.name for entity in result.entities] == ["SchemaRecovered"]

    async def test_does_not_retry_schema_validation_when_disabled(self) -> None:
        llm = InvalidSchemaThenSucceedLLMAdapter()
        config = ConsolidationConfig(
            extraction_max_retries=1,
            retry_on_schema_validation_error=False,
        )
        engine = ExtractionEngine(llm=llm, consolidation_config=config)

        result = await engine.extract([_make_fragment()])

        assert llm.calls == 1
        assert len(result.errors) == 1
        assert "Schema validation failed" in result.errors[0]


# ====================================================================
# TestLLMConfigRespected
# ====================================================================


class TestLLMConfigRespected:
    async def test_llm_config_respected(self) -> None:
        """Default LLMConfig values are forwarded to the adapter."""
        llm = MockLLMAdapter()
        engine = ExtractionEngine(llm=llm)

        await engine.extract([_make_fragment()])

        call = llm.calls[0]
        assert call["temperature"] == 0.2
        assert call["max_tokens"] == 4096
        assert call["timeout_seconds"] == 30.0

    async def test_custom_llm_config_propagated(self) -> None:
        """Custom LLMConfig overrides are forwarded."""
        llm = MockLLMAdapter()
        config = LLMConfig(temperature=0.5, max_tokens=2048, timeout_seconds=60.0)
        engine = ExtractionEngine(llm=llm, llm_config=config)

        await engine.extract([_make_fragment()])

        call = llm.calls[0]
        assert call["temperature"] == 0.5
        assert call["max_tokens"] == 2048
        assert call["timeout_seconds"] == 60.0


# ====================================================================
# TestPromptBuilder
# ====================================================================


class TestPromptBuilder:
    def test_prompt_contains_fragment_content(self) -> None:
        fragment = _make_fragment(content="Alice joined Acme Corp")
        prompt = build_extraction_prompt([fragment])
        assert "Alice joined Acme Corp" in prompt

    def test_prompt_contains_fragment_ids(self) -> None:
        fragment = _make_fragment(frag_id="mem_test123")
        prompt = build_extraction_prompt([fragment])
        assert "mem_test123" in prompt

    def test_prompt_contains_output_schema(self) -> None:
        prompt = build_extraction_prompt([_make_fragment()])
        # Should contain the JSON schema
        assert "ExtractionResult" in prompt or "entities" in prompt
        assert "relations" in prompt
        assert "claims" in prompt

    def test_prompt_contains_entity_types(self) -> None:
        prompt = build_extraction_prompt([_make_fragment()])
        for entity_type in ENTITY_TYPES:
            assert entity_type in prompt

    def test_prompt_contains_relation_types(self) -> None:
        prompt = build_extraction_prompt([_make_fragment()])
        for rel_type in RELATION_TYPES:
            assert rel_type in prompt


# ====================================================================
# TestLLMAdapterProtocol
# ====================================================================


class TestLLMAdapterProtocol:
    def test_mock_satisfies_protocol(self) -> None:
        adapter = MockLLMAdapter()
        assert isinstance(adapter, LLMAdapter)

    def test_llm_error_is_exception(self) -> None:
        err = LLMError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"
