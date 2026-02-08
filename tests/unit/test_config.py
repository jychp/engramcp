"""Unit tests for configuration dataclasses."""

from dataclasses import FrozenInstanceError

import pytest

from engramcp.config import AuditConfig
from engramcp.config import ConsolidationConfig
from engramcp.config import EntityResolutionConfig
from engramcp.config import LLMConfig


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4"
        assert cfg.api_key is None
        assert cfg.base_url == "https://api.openai.com/v1"
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 4096
        assert cfg.timeout_seconds == 30.0


# ---------------------------------------------------------------------------
# ConsolidationConfig
# ---------------------------------------------------------------------------


class TestConsolidationConfig:
    def test_defaults(self):
        cfg = ConsolidationConfig()
        assert cfg.fragment_threshold == 10
        assert cfg.extraction_batch_size == 5
        assert cfg.pattern_min_occurrences == 3


# ---------------------------------------------------------------------------
# EntityResolutionConfig
# ---------------------------------------------------------------------------


class TestEntityResolutionConfig:
    def test_defaults(self):
        cfg = EntityResolutionConfig()
        assert cfg.auto_merge_threshold == 0.9
        assert cfg.flag_for_review_threshold == 0.7
        assert cfg.create_link_threshold == 0.5
        assert cfg.llm_assisted_enabled is True


# ---------------------------------------------------------------------------
# AuditConfig
# ---------------------------------------------------------------------------


class TestAuditConfig:
    def test_defaults(self):
        cfg = AuditConfig()
        assert cfg.file_path == "engramcp_audit.jsonl"
        assert cfg.enabled is True


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestConfigImmutability:
    def test_llm_config_frozen(self):
        cfg = LLMConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.provider = "anthropic"

    def test_consolidation_config_frozen(self):
        cfg = ConsolidationConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.fragment_threshold = 99

    def test_entity_resolution_config_frozen(self):
        cfg = EntityResolutionConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.auto_merge_threshold = 0.5

    def test_audit_config_frozen(self):
        cfg = AuditConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.enabled = False
