"""Tests for scenario ground-truth verification script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_script(
    *,
    metrics_path: Path,
    tier2_path: Path,
    tier3_path: Path,
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    script = Path("tests/scenarios/helpers/ground_truth_verify.py")
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--metrics",
            str(metrics_path),
            "--tier2-ground-truth",
            str(tier2_path),
            "--tier3-ground-truth",
            str(tier3_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def _write_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


class TestGroundTruthVerifyScript:
    def test_accepts_combined_metrics_file_for_tier2_and_tier3_runtime(self, tmp_path):
        metrics_path = tmp_path / "scenario-metrics-tier23.jsonl"
        tier2_path = tmp_path / "ground_truth_tier2.json"
        tier3_path = tmp_path / "ground_truth_tier3.json"
        output_path = tmp_path / "ground-truth-verification.json"

        _write_metrics(
            metrics_path,
            [
                {
                    "scenario": "tier2_curated_corporate_timeline",
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
                    "scenario": "tier3_flight_logs_subset_runtime",
                    "tier": "tier3",
                    "metric_class": "extraction_precision_recall_proxy",
                    "values": {
                        "precision_proxy": 0.95,
                        "recall_proxy": 0.95,
                    },
                },
                {
                    "scenario": "tier3_flight_logs_subset_runtime",
                    "tier": "tier3",
                    "metric_class": "entity_merge_precision",
                    "values": {
                        "false_merge_count": 0,
                        "false_split_count": 0,
                    },
                },
                {
                    "scenario": "tier3_flight_logs_subset_runtime",
                    "tier": "tier3",
                    "metric_class": "retrieval_usefulness",
                    "values": {
                        "graph_hits": 2,
                        "returned_memories": 2,
                        "citation_hits": 2,
                        "contradictions": 0,
                    },
                },
            ],
        )
        _write_json(
            tier2_path,
            {
                "tier": "tier2",
                "scenarios": [
                    {
                        "name": "tier2_curated_corporate_timeline",
                        "required_metric_classes": ["timeline_change_tracking"],
                    }
                ],
            },
        )
        _write_json(
            tier3_path,
            {
                "tier": "tier3",
                "dataset_name": "subset",
                "required_fields": ["flight_date", "origin", "destination", "passengers"],
                "records": [
                    {
                        "flight_date": "2002-01-03",
                        "origin": "TEB",
                        "destination": "STT",
                        "passengers": ["Jeffrey Epstein", "Ghislaine Maxwell"],
                    },
                    {
                        "flight_date": "2002-01-10",
                        "origin": "STT",
                        "destination": "TEB",
                        "passengers": ["Epstein, Jeffrey", "GM"],
                    },
                    {
                        "flight_date": "2002-01-15",
                        "origin": "TEB",
                        "destination": "PBI",
                        "passengers": ["Jeff Epstein"],
                    },
                ],
                "expectations": {"min_records": 3, "requires_alias_variants": True},
                "runtime_scenarios": [
                    {
                        "name": "tier3_flight_logs_subset_runtime",
                        "required_metric_classes": [
                            "extraction_precision_recall_proxy",
                            "entity_merge_precision",
                            "retrieval_usefulness",
                        ],
                    }
                ],
            },
        )

        result = _run_script(
            metrics_path=metrics_path,
            tier2_path=tier2_path,
            tier3_path=tier3_path,
            output_path=output_path,
        )

        assert result.returncode == 0, result.stderr
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["overall_pass"] is True
        assert report["input_metrics"].endswith("scenario-metrics-tier23.jsonl")
        assert report["tier3"]["runtime"]["overall_pass"] is True

    def test_returns_zero_for_valid_tier2_and_tier3_inputs(self, tmp_path):
        metrics_path = tmp_path / "scenario-metrics.jsonl"
        tier2_path = tmp_path / "ground_truth_tier2.json"
        tier3_path = tmp_path / "ground_truth_tier3.json"
        output_path = tmp_path / "ground-truth-verification.json"

        _write_metrics(
            metrics_path,
            [
                {
                    "scenario": "tier2_curated_corporate_timeline",
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
                    "scenario": "tier3_flight_logs_subset_runtime",
                    "tier": "tier3",
                    "metric_class": "extraction_precision_recall_proxy",
                    "values": {
                        "precision_proxy": 0.95,
                        "recall_proxy": 0.95,
                    },
                },
                {
                    "scenario": "tier3_flight_logs_subset_runtime",
                    "tier": "tier3",
                    "metric_class": "entity_merge_precision",
                    "values": {
                        "false_merge_count": 0,
                        "false_split_count": 0,
                    },
                },
                {
                    "scenario": "tier3_flight_logs_subset_runtime",
                    "tier": "tier3",
                    "metric_class": "retrieval_usefulness",
                    "values": {
                        "graph_hits": 2,
                        "returned_memories": 2,
                        "citation_hits": 2,
                        "contradictions": 0,
                    },
                },
            ],
        )
        _write_json(
            tier2_path,
            {
                "tier": "tier2",
                "scenarios": [
                    {
                        "name": "tier2_curated_corporate_timeline",
                        "required_metric_classes": ["timeline_change_tracking"],
                    }
                ],
            },
        )
        _write_json(
            tier3_path,
            {
                "tier": "tier3",
                "dataset_name": "subset",
                "required_fields": ["flight_date", "origin", "destination", "passengers"],
                "records": [
                    {
                        "flight_date": "2002-01-03",
                        "origin": "TEB",
                        "destination": "STT",
                        "passengers": ["Jeffrey Epstein", "Ghislaine Maxwell"],
                    },
                    {
                        "flight_date": "2002-01-10",
                        "origin": "STT",
                        "destination": "TEB",
                        "passengers": ["Epstein, Jeffrey", "GM"],
                    },
                    {
                        "flight_date": "2002-01-15",
                        "origin": "TEB",
                        "destination": "PBI",
                        "passengers": ["Jeff Epstein"],
                    },
                ],
                "expectations": {"min_records": 3, "requires_alias_variants": True},
                "runtime_scenarios": [
                    {
                        "name": "tier3_flight_logs_subset_runtime",
                        "required_metric_classes": [
                            "extraction_precision_recall_proxy",
                            "entity_merge_precision",
                            "retrieval_usefulness",
                        ],
                    }
                ],
                "negative_samples": [
                    {
                        "name": "missing_destination",
                        "records": [
                            {
                                "flight_date": "2002-02-01",
                                "origin": "TEB",
                                "passengers": ["Jane Doe"],
                            }
                        ],
                        "expectations": {
                            "min_records": 1,
                            "requires_alias_variants": False,
                        },
                        "expected_error_substrings": [
                            "missing required field 'destination'",
                        ],
                    }
                ],
            },
        )

        result = _run_script(
            metrics_path=metrics_path,
            tier2_path=tier2_path,
            tier3_path=tier3_path,
            output_path=output_path,
        )

        assert result.returncode == 0, result.stderr
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["overall_pass"] is True
        assert report["tier3"]["negative_checks"][0]["passed"] is True
        assert report["tier3"]["runtime"]["overall_pass"] is True

    def test_returns_non_zero_when_tier2_metric_is_missing(self, tmp_path):
        metrics_path = tmp_path / "scenario-metrics.jsonl"
        tier2_path = tmp_path / "ground_truth_tier2.json"
        tier3_path = tmp_path / "ground_truth_tier3.json"
        output_path = tmp_path / "ground-truth-verification.json"

        _write_metrics(metrics_path, [])
        _write_json(
            tier2_path,
            {
                "tier": "tier2",
                "scenarios": [
                    {
                        "name": "tier2_curated_travel_conflict",
                        "required_metric_classes": ["contradiction_coverage"],
                    }
                ],
            },
        )
        _write_json(
            tier3_path,
            {
                "tier": "tier3",
                "dataset_name": "subset",
                "required_fields": ["flight_date", "origin", "destination", "passengers"],
                "records": [
                    {
                        "flight_date": "2002-01-03",
                        "origin": "TEB",
                        "destination": "STT",
                        "passengers": ["Jeffrey Epstein", "Epstein, Jeffrey"],
                    }
                ],
                "expectations": {"min_records": 1, "requires_alias_variants": False},
            },
        )

        result = _run_script(
            metrics_path=metrics_path,
            tier2_path=tier2_path,
            tier3_path=tier3_path,
            output_path=output_path,
        )

        assert result.returncode == 1
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["overall_pass"] is False
        assert report["tier2"]["missing"] == [
            {
                "metric_class": "contradiction_coverage",
                "scenario": "tier2_curated_travel_conflict",
            }
        ]

    def test_returns_non_zero_when_tier3_negative_sample_does_not_fail_as_expected(
        self, tmp_path
    ):
        metrics_path = tmp_path / "scenario-metrics.jsonl"
        tier2_path = tmp_path / "ground_truth_tier2.json"
        tier3_path = tmp_path / "ground_truth_tier3.json"
        output_path = tmp_path / "ground-truth-verification.json"

        _write_metrics(
            metrics_path,
            [
                {
                    "scenario": "tier2_curated_corporate_timeline",
                    "tier": "tier2",
                    "metric_class": "timeline_change_tracking",
                    "values": {
                        "changed_agents_count": 2,
                        "timeline_statement_count": 4,
                        "carol_consistency_hits": 1,
                        "contradictions": 0,
                    },
                }
            ],
        )
        _write_json(
            tier2_path,
            {
                "tier": "tier2",
                "scenarios": [
                    {
                        "name": "tier2_curated_corporate_timeline",
                        "required_metric_classes": ["timeline_change_tracking"],
                    }
                ],
            },
        )
        _write_json(
            tier3_path,
            {
                "tier": "tier3",
                "dataset_name": "subset",
                "required_fields": ["flight_date", "origin", "destination", "passengers"],
                "records": [
                    {
                        "flight_date": "2002-01-03",
                        "origin": "TEB",
                        "destination": "STT",
                        "passengers": ["Jeffrey Epstein", "Epstein, Jeffrey"],
                    }
                ],
                "expectations": {"min_records": 1, "requires_alias_variants": False},
                "negative_samples": [
                    {
                        "name": "should_fail_but_wont",
                        "records": [
                            {
                                "flight_date": "2002-02-01",
                                "origin": "TEB",
                                "destination": "STT",
                                "passengers": ["Passenger A"],
                            }
                        ],
                        "expectations": {
                            "min_records": 1,
                            "requires_alias_variants": False,
                        },
                        "expected_error_substrings": [
                            "alias variants were required but not detected in records",
                        ],
                    }
                ],
            },
        )

        result = _run_script(
            metrics_path=metrics_path,
            tier2_path=tier2_path,
            tier3_path=tier3_path,
            output_path=output_path,
        )

        assert result.returncode == 1
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["overall_pass"] is False
        assert report["tier3"]["negative_checks"][0]["passed"] is False

    def test_returns_non_zero_when_tier3_runtime_metric_is_missing(self, tmp_path):
        metrics_path = tmp_path / "scenario-metrics.jsonl"
        tier2_path = tmp_path / "ground_truth_tier2.json"
        tier3_path = tmp_path / "ground_truth_tier3.json"
        output_path = tmp_path / "ground-truth-verification.json"

        _write_metrics(
            metrics_path,
            [
                {
                    "scenario": "tier2_curated_corporate_timeline",
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
                    "scenario": "tier3_flight_logs_subset_runtime",
                    "tier": "tier3",
                    "metric_class": "retrieval_usefulness",
                    "values": {
                        "graph_hits": 2,
                        "returned_memories": 2,
                        "citation_hits": 2,
                        "contradictions": 0,
                    },
                },
            ],
        )
        _write_json(
            tier2_path,
            {
                "tier": "tier2",
                "scenarios": [
                    {
                        "name": "tier2_curated_corporate_timeline",
                        "required_metric_classes": ["timeline_change_tracking"],
                    }
                ],
            },
        )
        _write_json(
            tier3_path,
            {
                "tier": "tier3",
                "dataset_name": "subset",
                "required_fields": ["flight_date", "origin", "destination", "passengers"],
                "records": [
                    {
                        "flight_date": "2002-01-03",
                        "origin": "TEB",
                        "destination": "STT",
                        "passengers": ["Jeffrey Epstein", "Ghislaine Maxwell"],
                    },
                    {
                        "flight_date": "2002-01-10",
                        "origin": "STT",
                        "destination": "TEB",
                        "passengers": ["Epstein, Jeffrey", "GM"],
                    },
                    {
                        "flight_date": "2002-01-15",
                        "origin": "TEB",
                        "destination": "PBI",
                        "passengers": ["Jeff Epstein"],
                    },
                ],
                "expectations": {"min_records": 3, "requires_alias_variants": True},
                "runtime_scenarios": [
                    {
                        "name": "tier3_flight_logs_subset_runtime",
                        "required_metric_classes": [
                            "extraction_precision_recall_proxy",
                            "entity_merge_precision",
                            "retrieval_usefulness",
                        ],
                    }
                ],
            },
        )

        result = _run_script(
            metrics_path=metrics_path,
            tier2_path=tier2_path,
            tier3_path=tier3_path,
            output_path=output_path,
        )

        assert result.returncode == 1
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["overall_pass"] is False
        assert report["tier3"]["runtime"]["missing"] == [
            {
                "metric_class": "extraction_precision_recall_proxy",
                "scenario": "tier3_flight_logs_subset_runtime",
            },
            {
                "metric_class": "entity_merge_precision",
                "scenario": "tier3_flight_logs_subset_runtime",
            },
        ]
