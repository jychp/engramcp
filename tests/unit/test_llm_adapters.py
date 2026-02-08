"""Unit tests for concrete LLM adapters and provider factory."""

from __future__ import annotations

import json

import pytest

from engramcp.config import LLMConfig
from engramcp.engine.llm_adapters import NoopLLMAdapter
from engramcp.engine.llm_adapters import OpenAICompatibleLLMAdapter
from engramcp.engine.llm_adapters import build_llm_adapter
from engramcp.engine.schemas import ExtractionResult
from engramcp.server import configure


class TestBuildLLMAdapter:
    def test_openai_provider_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key is required"):
            build_llm_adapter(LLMConfig(provider="openai", api_key=None))

    def test_noop_provider_is_supported(self) -> None:
        adapter = build_llm_adapter(LLMConfig(provider="noop"))
        assert isinstance(adapter, NoopLLMAdapter)

    def test_unsupported_provider_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported llm_config.provider"):
            build_llm_adapter(LLMConfig(provider="anthropic"))


class TestNoopLLMAdapter:
    async def test_noop_returns_valid_empty_extraction_json(self) -> None:
        adapter = NoopLLMAdapter()
        raw = await adapter.complete("irrelevant")
        data = json.loads(raw)
        result = ExtractionResult.model_validate(data)
        assert result.entities == []
        assert result.relations == []
        assert result.claims == []
        assert result.errors == []


class TestOpenAICompatibleAdapter:
    async def test_complete_uses_sync_path_via_thread(self, monkeypatch) -> None:
        adapter = OpenAICompatibleLLMAdapter(
            model="gpt-4",
            api_key="test",
        )

        def _fake_sync(prompt: str, *, temperature: float, max_tokens: int, timeout_seconds: float) -> str:
            assert prompt == "hello"
            assert temperature == 0.3
            assert max_tokens == 77
            assert timeout_seconds == 11.0
            return '{"ok": true}'

        monkeypatch.setattr(adapter, "_complete_sync", _fake_sync)

        result = await adapter.complete(
            "hello",
            temperature=0.3,
            max_tokens=77,
            timeout_seconds=11.0,
        )
        assert result == '{"ok": true}'


class TestServerLLMConfigGuard:
    async def test_configure_fails_fast_without_openai_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key is required"):
            await configure(
                redis_url="redis://localhost:6379",
                enable_consolidation=True,
                neo4j_url="bolt://localhost:7687",
            )
