"""Tests for scenario metrics calibration script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_script(*, metrics_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    script = Path("scripts/calibrate_eval_thresholds.py")
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--metrics",
            str(metrics_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


class TestCalibrateEvalThresholdsScript:
    def test_returns_zero_and_writes_report_when_all_metric_classes_pass(self, tmp_path):
        metrics_path = tmp_path / "scenario-metrics.jsonl"
        output_path = tmp_path / "eval-calibration.json"

        rows = [
            {
                "scenario": "s1",
                "tier": "tier1",
                "metric_class": "retrieval_relevance",
                "values": {"graph_hits": 1, "returned_memories": 1, "keyword_hits": 1},
            },
            {
                "scenario": "s2",
                "tier": "tier2",
                "metric_class": "contradiction_coverage",
                "values": {"contradictions": 1},
            },
            {
                "scenario": "s3",
                "tier": "tier2",
                "metric_class": "corroboration",
                "values": {"unique_source_ids": 2},
            },
            {
                "scenario": "s4",
                "tier": "tier2",
                "metric_class": "confidence_progression",
                "values": {
                    "independent_source_count": 2,
                    "independent_credibility": "2",
                    "dependent_source_count": 1,
                    "dependent_credibility": "3",
                },
            },
            {
                "scenario": "s5",
                "tier": "tier2",
                "metric_class": "derivation_traceability",
                "values": {
                    "rule_entries": 1,
                    "rule_derivation_depth": 3,
                    "has_derivation_run_id": True,
                },
            },
            {
                "scenario": "s6",
                "tier": "tier2",
                "metric_class": "timeline_change_tracking",
                "values": {
                    "changed_agents_count": 2,
                    "timeline_statement_count": 4,
                    "carol_consistency_hits": 1,
                    "contradictions": 0,
                },
            },
            {
                "scenario": "s7",
                "tier": "tier3",
                "metric_class": "extraction_precision_recall_proxy",
                "values": {
                    "precision_proxy": 0.95,
                    "recall_proxy": 0.95,
                },
            },
            {
                "scenario": "s8",
                "tier": "tier3",
                "metric_class": "entity_merge_precision",
                "values": {
                    "false_merge_count": 0,
                    "false_split_count": 0,
                },
            },
            {
                "scenario": "s9",
                "tier": "tier3",
                "metric_class": "retrieval_usefulness",
                "values": {
                    "graph_hits": 2,
                    "returned_memories": 2,
                    "citation_hits": 2,
                    "contradictions": 0,
                },
            },
        ]
        _write_metrics(metrics_path, rows)

        result = _run_script(metrics_path=metrics_path, output_path=output_path)

        assert result.returncode == 0, result.stderr
        assert output_path.exists()

        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["metrics"]["overall_pass"] is True
        assert report["metrics"]["missing_metric_classes"] == []
        assert report["consolidation"]["recommended_eval_profile"] == {
            "fragment_threshold": 4,
            "pattern_min_occurrences": 2,
        }

    def test_returns_non_zero_when_metric_classes_are_missing(self, tmp_path):
        metrics_path = tmp_path / "scenario-metrics.jsonl"
        output_path = tmp_path / "eval-calibration.json"

        rows = [
            {
                "scenario": "s1",
                "tier": "tier1",
                "metric_class": "retrieval_relevance",
                "values": {"graph_hits": 1, "returned_memories": 1, "keyword_hits": 1},
            }
        ]
        _write_metrics(metrics_path, rows)

        result = _run_script(metrics_path=metrics_path, output_path=output_path)

        assert result.returncode == 1
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["metrics"]["overall_pass"] is False
        assert set(report["metrics"]["missing_metric_classes"]) == {
            "confidence_progression",
            "contradiction_coverage",
            "corroboration",
            "derivation_traceability",
            "entity_merge_precision",
            "extraction_precision_recall_proxy",
            "retrieval_usefulness",
            "timeline_change_tracking",
        }
