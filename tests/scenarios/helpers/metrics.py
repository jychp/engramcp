"""Scenario-eval metric helpers and thresholds."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_METRICS_PATH = "reports/scenario-metrics.jsonl"


@dataclass(frozen=True)
class ScenarioMetric:
    """One emitted metric line for post-run calibration and reporting."""

    scenario: str
    tier: str
    metric_class: str
    values: dict[str, Any]


def _metrics_path() -> Path:
    path = os.getenv("ENGRAMCP_SCENARIO_METRICS_PATH", DEFAULT_METRICS_PATH)
    return Path(path)


def emit_scenario_metric(
    *,
    scenario: str,
    tier: str,
    metric_class: str,
    values: dict[str, Any],
) -> None:
    """Append a metric entry to JSONL file configured for scenario runs."""
    metric = ScenarioMetric(
        scenario=scenario,
        tier=tier,
        metric_class=metric_class,
        values=values,
    )
    path = _metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(metric), ensure_ascii=True, sort_keys=True))
        handle.write("\n")
