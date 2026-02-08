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
    min_rule_entries_for_derivation: int = 1
    expected_rule_derivation_depth: int = 3


THRESHOLDS = ScenarioThresholds()
