"""Application configuration dataclasses.

Frozen dataclasses with sensible defaults for each subsystem.
No env-var loading or YAML parsing — just plain defaults that can
be overridden at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider settings used by the consolidation engine."""

    provider: str = "openai"
    model: str = "gpt-4"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class ConsolidationConfig:
    """Tuneable parameters for the consolidation pipeline."""

    fragment_threshold: int = 10
    extraction_batch_size: int = 5
    pattern_min_occurrences: int = 3


@dataclass(frozen=True)
class EntityResolutionConfig:
    """Thresholds for the three-level entity resolution strategy."""

    # Action thresholds
    auto_merge_threshold: float = 0.9
    flag_for_review_threshold: float = 0.7
    create_link_threshold: float = 0.5
    llm_assisted_enabled: bool = True
    # Scoring weights (must sum to 1.0)
    name_similarity_weight: float = 0.5
    context_overlap_weight: float = 0.3
    property_compatibility_weight: float = 0.2
    # Fuzzy matching parameters
    token_jaccard_threshold: float = 0.6
    edit_distance_max: float = 0.3


@dataclass(frozen=True)
class AuditConfig:
    """Settings for the JSONL audit logger."""

    file_path: str = "engramcp_audit.jsonl"
    enabled: bool = True
