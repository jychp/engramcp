"""Verify scenario ground-truth fixtures against emitted metrics and dataset subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engramcp.evaluation import THRESHOLDS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/scenario-metrics.jsonl")
    parser.add_argument(
        "--tier2-ground-truth",
        default="tests/scenarios/fixtures/ground_truth_tier2.json",
    )
    parser.add_argument(
        "--tier3-ground-truth",
        default="tests/scenarios/fixtures/ground_truth_tier3_flight_logs_subset.json",
    )
    parser.add_argument("--output", default="reports/ground-truth-verification.json")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                rows.append(json.loads(payload))
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
    if cls == "confidence_progression":
        return (
            values.get("independent_source_count", 0)
            >= THRESHOLDS.min_independent_sources_for_confidence_upgrade
            and values.get("independent_credibility")
            == THRESHOLDS.expected_corroborated_credibility
            and values.get("dependent_source_count")
            == THRESHOLDS.expected_dependent_independent_sources
            and values.get("dependent_credibility")
            == THRESHOLDS.expected_dependent_credibility
        )
    if cls == "derivation_traceability":
        return (
            values.get("rule_entries", 0) >= THRESHOLDS.min_rule_entries_for_derivation
            and values.get("rule_derivation_depth")
            == THRESHOLDS.expected_rule_derivation_depth
            and bool(values.get("has_derivation_run_id"))
        )
    if cls == "timeline_change_tracking":
        return (
            values.get("changed_agents_count", 0)
            >= THRESHOLDS.min_changed_agents_in_timeline
            and values.get("timeline_statement_count", 0)
            >= THRESHOLDS.min_timeline_statements
            and values.get("carol_consistency_hits", 0)
            >= THRESHOLDS.min_consistent_agent_hits
            and values.get("contradictions", 0)
            <= THRESHOLDS.max_contradictions_for_temporal_evolution
        )
    return False


def verify_tier2_ground_truth(*, metrics_rows: list[dict], tier2_ground_truth: dict) -> dict:
    metrics_by_key: dict[tuple[str, str], list[dict]] = {}
    for row in metrics_rows:
        key = (str(row.get("scenario", "")), str(row.get("metric_class", "")))
        metrics_by_key.setdefault(key, []).append(row)

    missing: list[dict] = []
    failing: list[dict] = []
    checks: list[dict] = []
    for scenario in tier2_ground_truth.get("scenarios", []):
        scenario_name = str(scenario["name"])
        for metric_class in scenario.get("required_metric_classes", []):
            key = (scenario_name, str(metric_class))
            candidates = metrics_by_key.get(key, [])
            if not candidates:
                missing.append({"scenario": scenario_name, "metric_class": metric_class})
                continue

            passed = all(_metric_pass(candidate) for candidate in candidates)
            checks.append(
                {
                    "scenario": scenario_name,
                    "metric_class": metric_class,
                    "samples": len(candidates),
                    "passed": passed,
                }
            )
            if not passed:
                failing.append({"scenario": scenario_name, "metric_class": metric_class})

    return {
        "checks": checks,
        "missing": missing,
        "failing": failing,
        "overall_pass": not missing and not failing,
    }


def verify_tier3_subset(*, tier3_ground_truth: dict) -> dict:
    required_fields = [str(field) for field in tier3_ground_truth.get("required_fields", [])]
    records = tier3_ground_truth.get("records", [])
    expectations = tier3_ground_truth.get("expectations", {})
    structural_errors, alias_variant_detected = _tier3_structural_errors(
        records=records,
        required_fields=required_fields,
        expectations=expectations,
    )
    negative_checks = _verify_tier3_negative_samples(
        tier3_ground_truth=tier3_ground_truth,
        required_fields=required_fields,
    )
    negative_failures = [check for check in negative_checks if not check["passed"]]
    overall_pass = not structural_errors and not negative_failures
    return {
        "dataset_name": tier3_ground_truth.get("dataset_name", "unknown"),
        "records_count": len(records),
        "required_fields": required_fields,
        "alias_variant_detected": alias_variant_detected,
        "structural_errors": structural_errors,
        "negative_checks": negative_checks,
        "overall_pass": overall_pass,
    }


def _tier3_structural_errors(
    *,
    records: list[dict],
    required_fields: list[str],
    expectations: dict,
) -> tuple[list[str], bool]:
    min_records = int(expectations.get("min_records", 1))
    requires_alias_variants = bool(expectations.get("requires_alias_variants", False))
    structural_errors: list[str] = []
    if len(records) < min_records:
        structural_errors.append(
            f"records_count {len(records)} is below min_records {min_records}"
        )

    for index, record in enumerate(records):
        for field in required_fields:
            if field not in record:
                structural_errors.append(f"record[{index}] missing required field '{field}'")

    alias_variant_detected = False
    if records:
        names = {
            str(name).strip().lower()
            for record in records
            for name in record.get("passengers", [])
            if str(name).strip()
        }
        alias_variant_detected = bool(
            ("epstein, jeffrey" in names and "jeffrey epstein" in names)
            or ("gm" in names and "ghislaine maxwell" in names)
            or ("jeff epstein" in names and "jeffrey epstein" in names)
        )
    if requires_alias_variants and not alias_variant_detected:
        structural_errors.append("alias variants were required but not detected in records")
    return structural_errors, alias_variant_detected


def _verify_tier3_negative_samples(
    *, tier3_ground_truth: dict, required_fields: list[str]
) -> list[dict]:
    results: list[dict] = []
    for sample in tier3_ground_truth.get("negative_samples", []):
        records = sample.get("records", [])
        expectations = sample.get("expectations", {})
        expected_error_substrings = [
            str(value) for value in sample.get("expected_error_substrings", [])
        ]
        errors, _ = _tier3_structural_errors(
            records=records,
            required_fields=required_fields,
            expectations=expectations,
        )
        joined_errors = "\n".join(errors)
        passed = all(fragment in joined_errors for fragment in expected_error_substrings)
        results.append(
            {
                "name": str(sample.get("name", "unnamed_negative_sample")),
                "errors": errors,
                "expected_error_substrings": expected_error_substrings,
                "passed": passed,
            }
        )
    return results


def run_verification(
    *,
    metrics_path: Path,
    tier2_ground_truth_path: Path,
    tier3_ground_truth_path: Path,
    output_path: Path,
) -> dict:
    metrics_rows = _load_metrics(metrics_path)
    tier2_ground_truth = _load_json(tier2_ground_truth_path)
    tier3_ground_truth = _load_json(tier3_ground_truth_path)

    tier2 = verify_tier2_ground_truth(
        metrics_rows=metrics_rows, tier2_ground_truth=tier2_ground_truth
    )
    tier3 = verify_tier3_subset(tier3_ground_truth=tier3_ground_truth)

    report = {
        "input_metrics": str(metrics_path),
        "tier2_ground_truth": str(tier2_ground_truth_path),
        "tier3_ground_truth": str(tier3_ground_truth_path),
        "tier2": tier2,
        "tier3": tier3,
        "overall_pass": bool(tier2["overall_pass"] and tier3["overall_pass"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    args = _parse_args()
    report = run_verification(
        metrics_path=Path(args.metrics),
        tier2_ground_truth_path=Path(args.tier2_ground_truth),
        tier3_ground_truth_path=Path(args.tier3_ground_truth),
        output_path=Path(args.output),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
