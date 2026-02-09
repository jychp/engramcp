"""Shared scenario-evaluation thresholds and contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioThresholds:
    """Central pass/fail thresholds by evaluation class."""

    min_graph_hits: int = 1
    min_memories: int = 1
    min_contradictions: int = 1
    min_unique_sources_for_corroboration: int = 2
    min_independent_sources_for_confidence_upgrade: int = 2
    expected_dependent_independent_sources: int = 1
    expected_corroborated_credibility: str = "2"
    expected_dependent_credibility: str = "3"
    min_rule_entries_for_derivation: int = 1
    expected_rule_derivation_depth: int = 3
    min_changed_agents_in_timeline: int = 2
    min_timeline_statements: int = 4
    min_consistent_agent_hits: int = 1
    max_contradictions_for_temporal_evolution: int = 0
    min_extraction_precision_proxy: float = 0.9
    min_extraction_recall_proxy: float = 0.9
    max_false_merges: int = 0
    max_false_splits: int = 0
    min_citation_hits: int = 1
    max_contradictions_for_tier3_retrieval: int = 0


THRESHOLDS = ScenarioThresholds()
