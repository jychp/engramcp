"""Aggregate scenario metrics and recommend calibration thresholds.

Usage:
    uv run python scripts/calibrate_eval_thresholds.py \
      --metrics reports/scenario-metrics.jsonl \
      --output reports/eval-calibration.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from engramcp.config import ConsolidationConfig
from engramcp.config import EntityResolutionConfig
from engramcp.config import scenario_eval_consolidation_config
from engramcp.evaluation import THRESHOLDS
from engramcp.engine.schemas import ExtractedEntity
from engramcp.graph.entity_resolution import EntityResolver
from engramcp.graph.entity_resolution import ExistingEntity
from engramcp.graph.entity_resolution import ResolutionAction


@dataclass(frozen=True)
class ResolutionCalibrationCase:
    expected: ResolutionAction
    entity: ExtractedEntity
    existing: list[ExistingEntity]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/scenario-metrics.jsonl")
    parser.add_argument("--output", default="reports/eval-calibration.json")
    return parser.parse_args()


def _load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _metric_pass(metric: dict) -> bool:
    cls = metric.get("metric_class")
    values = metric.get("values", {})
    if cls == "retrieval_relevance":
        return (
            values.get("graph_hits", 0) >= THRESHOLDS.min_graph_hits
            and values.get("returned_memories", 0) >= THRESHOLDS.min_memories
            and values.get("keyword_hits", 0) >= 1
        )
    if cls == "contradiction_coverage":
        return values.get("contradictions", 0) >= THRESHOLDS.min_contradictions
    if cls == "corroboration":
        return (
            values.get("unique_source_ids", 0)
            >= THRESHOLDS.min_unique_sources_for_corroboration
        )
    if cls == "derivation_traceability":
        return (
            values.get("rule_entries", 0) >= THRESHOLDS.min_rule_entries_for_derivation
            and values.get("rule_derivation_depth")
            == THRESHOLDS.expected_rule_derivation_depth
            and bool(values.get("has_derivation_run_id"))
        )
    return False


def _aggregate_metrics(rows: list[dict]) -> dict:
    expected_classes = {
        "retrieval_relevance",
        "contradiction_coverage",
        "corroboration",
        "derivation_traceability",
    }

    by_class: dict[str, list[dict]] = {}
    for row in rows:
        cls = str(row.get("metric_class", ""))
        by_class.setdefault(cls, []).append(row)

    classes: dict[str, dict] = {}
    missing = sorted(expected_classes - set(by_class))
    for cls, entries in sorted(by_class.items()):
        total = len(entries)
        passed = sum(1 for entry in entries if _metric_pass(entry))
        classes[cls] = {
            "total": total,
            "passed": passed,
            "pass_rate": (passed / total) if total else 0.0,
        }

    overall_pass = not missing and all(
        classes[c]["pass_rate"] >= 1.0 for c in expected_classes if c in classes
    )
    return {
        "expected_metric_classes": sorted(expected_classes),
        "missing_metric_classes": missing,
        "classes": classes,
        "overall_pass": overall_pass,
    }


def _calibration_cases() -> list[ResolutionCalibrationCase]:
    return [
        ResolutionCalibrationCase(
            expected=ResolutionAction.merge,
            entity=ExtractedEntity(name="Jeffrey Epstein", type="Agent"),
            existing=[
                ExistingEntity(
                    node_id="e1",
                    name="Jeffrey Epstein",
                    type="Agent",
                    aliases=["J. Epstein"],
                    properties={},
                    fragment_ids=["f1"],
                )
            ],
        ),
        ResolutionCalibrationCase(
            expected=ResolutionAction.link,
            entity=ExtractedEntity(name="Epstein", type="Agent"),
            existing=[
                ExistingEntity(
                    node_id="e1",
                    name="Jeffrey Epstein",
                    type="Agent",
                    aliases=[],
                    properties={},
                    fragment_ids=[],
                )
            ],
        ),
        ResolutionCalibrationCase(
            expected=ResolutionAction.create_new,
            entity=ExtractedEntity(name="Jefferson Epps", type="Agent"),
            existing=[
                ExistingEntity(
                    node_id="e1",
                    name="Jeffrey Epstein",
                    type="Agent",
                    aliases=[],
                    properties={"dob": "1960-01-01"},
                    fragment_ids=[],
                )
            ],
        ),
        ResolutionCalibrationCase(
            expected=ResolutionAction.create_new,
            entity=ExtractedEntity(name="Atlas Corp", type="Artifact"),
            existing=[
                ExistingEntity(
                    node_id="e1",
                    name="Atlas Corp",
                    type="Agent",
                    aliases=[],
                    properties={},
                    fragment_ids=[],
                )
            ],
        ),
        ResolutionCalibrationCase(
            expected=ResolutionAction.review,
            entity=ExtractedEntity(
                name="Jonathan Doe",
                type="Agent",
                source_fragment_ids=["f_shared"],
                disambiguating_context="Witness list",
            ),
            existing=[
                ExistingEntity(
                    node_id="e1",
                    name="John Doe",
                    type="Agent",
                    aliases=[],
                    properties={},
                    fragment_ids=["f_shared"],
                )
            ],
        ),
    ]


async def _score_thresholds(config: EntityResolutionConfig) -> dict:
    resolver = EntityResolver(config=config)
    cases = _calibration_cases()

    total = len(cases)
    exact = 0
    for case in cases:
        result = await resolver.resolve(case.entity, case.existing)
        if result.action == case.expected:
            exact += 1

    return {
        "total": total,
        "exact_matches": exact,
        "accuracy": (exact / total) if total else 0.0,
    }


async def _recommend_entity_resolution_thresholds() -> dict:
    baseline = EntityResolutionConfig()

    candidates: list[EntityResolutionConfig] = []
    for merge in (0.88, 0.9, 0.92):
        for review in (0.68, 0.7, 0.72):
            for link in (0.48, 0.5, 0.52):
                if not (link < review < merge):
                    continue
                candidates.append(
                    EntityResolutionConfig(
                        auto_merge_threshold=merge,
                        flag_for_review_threshold=review,
                        create_link_threshold=link,
                    )
                )

    scored: list[tuple[dict, EntityResolutionConfig]] = []
    for config in candidates:
        score = await _score_thresholds(config)
        scored.append((score, config))

    scored.sort(
        key=lambda item: (
            item[0]["accuracy"],
            -abs(item[1].auto_merge_threshold - baseline.auto_merge_threshold),
            -abs(
                item[1].flag_for_review_threshold - baseline.flag_for_review_threshold
            ),
            -abs(item[1].create_link_threshold - baseline.create_link_threshold),
        ),
        reverse=True,
    )

    best_score, best = scored[0]
    baseline_score = await _score_thresholds(baseline)
    return {
        "baseline": {
            "auto_merge_threshold": baseline.auto_merge_threshold,
            "flag_for_review_threshold": baseline.flag_for_review_threshold,
            "create_link_threshold": baseline.create_link_threshold,
            "accuracy": baseline_score["accuracy"],
        },
        "recommended": {
            "auto_merge_threshold": best.auto_merge_threshold,
            "flag_for_review_threshold": best.flag_for_review_threshold,
            "create_link_threshold": best.create_link_threshold,
            "accuracy": best_score["accuracy"],
        },
    }


def _recommend_consolidation_thresholds(metric_summary: dict) -> dict:
    baseline = ConsolidationConfig()
    eval_profile = scenario_eval_consolidation_config()
    derivation = metric_summary["classes"].get("derivation_traceability", {})

    # Scenario evidence currently supports keeping production defaults unchanged.
    # For scenario evals, lower pattern threshold improves deterministic
    # derivation observability in compact fixture sets.
    return {
        "baseline": {
            "fragment_threshold": baseline.fragment_threshold,
            "pattern_min_occurrences": baseline.pattern_min_occurrences,
        },
        "recommended_defaults": {
            "fragment_threshold": baseline.fragment_threshold,
            "pattern_min_occurrences": baseline.pattern_min_occurrences,
        },
        "recommended_eval_profile": {
            "fragment_threshold": eval_profile.fragment_threshold,
            "pattern_min_occurrences": eval_profile.pattern_min_occurrences,
        },
        "evidence": {
            "derivation_traceability_pass_rate": derivation.get("pass_rate", 0.0),
        },
    }


async def _main() -> int:
    args = _parse_args()
    metrics_path = Path(args.metrics)
    output_path = Path(args.output)

    rows = _load_metrics(metrics_path)
    metric_summary = _aggregate_metrics(rows)
    entity_resolution = await _recommend_entity_resolution_thresholds()
    consolidation = _recommend_consolidation_thresholds(metric_summary)

    report = {
        "input_metrics": str(metrics_path),
        "metrics": metric_summary,
        "entity_resolution": entity_resolution,
        "consolidation": consolidation,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")

    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if metric_summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
