"""CPU-only synthetic tests for top-1 dynamics reporting helpers."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.top1_reporting import (
    aggregate_trajectory_flips,
    binary_ranking_metrics,
    counterfactual_attribution_table,
    future_token_rank_table,
    future_token_topk_table,
    grouped_flip_rate_table,
    order_sensitivity_table,
    predictiveness_table,
    prompt_clustered_bootstrap,
    quantile_bin_table,
    sample_matched_event_cohort,
    write_artifact_tables,
    write_csv,
    write_jsonl,
)


class Top1ReportingTest(unittest.TestCase):
    def test_binary_ranking_metrics_are_tie_aware_without_sklearn(self) -> None:
        perfect = binary_ranking_metrics([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        self.assertEqual(perfect["positive_count"], 2)
        self.assertAlmostEqual(perfect["prevalence"], 0.5)
        self.assertAlmostEqual(perfect["auroc"], 1.0)
        self.assertAlmostEqual(perfect["auprc"], 1.0)

        tied = binary_ranking_metrics([0, 1], [0.5, 0.5])
        self.assertAlmostEqual(tied["auroc"], 0.5)
        self.assertAlmostEqual(tied["auprc"], 0.5)

        reversed_margin = binary_ranking_metrics([0, 1], [0.9, 0.1], higher_is_positive=False)
        self.assertAlmostEqual(reversed_margin["auroc"], 1.0)

    def test_prompt_clustered_bootstrap_distinguishes_macro_and_micro(self) -> None:
        records = [
            {"prompt_index": "a", "flip": 0},
            {"prompt_index": "a", "flip": 0},
            {"prompt_index": "b", "flip": 1},
        ]
        macro = prompt_clustered_bootstrap(
            records, value_field="flip", iterations=300, seed=7, weighting="macro"
        )
        micro = prompt_clustered_bootstrap(
            records, value_field="flip", iterations=300, seed=7, weighting="micro"
        )
        self.assertAlmostEqual(macro["estimate"], 0.5)
        self.assertAlmostEqual(micro["estimate"], 1 / 3)
        self.assertEqual(macro["cluster_count"], 2)
        self.assertLessEqual(macro["ci95_low"], macro["estimate"])
        self.assertGreaterEqual(macro["ci95_high"], macro["estimate"])

    def test_quantile_bins_and_predictiveness_rows_are_cpu_only(self) -> None:
        records = [
            {"prompt_index": "a", "entropy": 0.1, "top1_flip": False},
            {"prompt_index": "a", "entropy": 0.2, "top1_flip": False},
            {"prompt_index": "b", "entropy": 0.8, "top1_flip": True},
            {"prompt_index": "b", "entropy": 0.9, "top1_flip": True},
        ]
        bins = quantile_bin_table(
            records,
            score_field="entropy",
            outcome_field="top1_flip",
            bin_count=2,
            bootstrap_iterations=None,
        )
        self.assertEqual(len(bins), 2)
        self.assertAlmostEqual(bins[0]["empirical_probability"], 0.0)
        self.assertAlmostEqual(bins[1]["empirical_probability"], 1.0)

        metrics = predictiveness_table(
            records,
            score_fields=("entropy",),
            bootstrap_iterations=None,
            include_quantile_rows=True,
            quantile_bins=2,
        )
        summary = next(row for row in metrics if row["row_type"] == "metric")
        self.assertAlmostEqual(summary["auroc"], 1.0)
        self.assertAlmostEqual(summary["auprc"], 1.0)
        self.assertEqual(sum(row["row_type"] == "quantile_bin" for row in metrics), 2)

    def test_matched_cohort_keeps_exact_dataset_phase_confidence_anchor_strata(self) -> None:
        records = [
            {"id": "e1", "top1_flip": True, "dataset": "gsm8k", "phase": "early", "previous_top1_probability": 0.92, "source_anchor_count": 2},
            {"id": "e2", "top1_flip": True, "dataset": "gsm8k", "phase": "early", "previous_top1_probability": 0.91, "source_anchor_count": 2},
            {"id": "c1", "top1_flip": False, "dataset": "gsm8k", "phase": "early", "previous_top1_probability": 0.93, "source_anchor_count": 2},
            {"id": "c2", "top1_flip": False, "dataset": "gsm8k", "phase": "early", "previous_top1_probability": 0.94, "source_anchor_count": 2},
            # This event has no same-anchor control and must be reported/dropped.
            {"id": "unmatched", "top1_flip": True, "dataset": "gsm8k", "phase": "early", "previous_top1_probability": 0.92, "source_anchor_count": 3},
        ]
        cohort = sample_matched_event_cohort(records, seed=3)
        self.assertEqual(cohort["selected_event_count"], 2)
        self.assertEqual(cohort["selected_control_count"], 2)
        self.assertEqual({row["cohort_role"] for row in cohort["records"]}, {"event", "control"})
        self.assertEqual(sum(row["dropped_event_count"] for row in cohort["diagnostics"]), 1)

    def test_trajectory_aggregation_reports_micro_macro_flipback_and_horizons(self) -> None:
        transitions = [
            {
                "prompt_index": "a", "position": 4, "source_step": 0, "target_step": 1,
                "previous_top1_token_id": 10, "next_top1_token_id": 11, "top1_flip": True,
                "dataset": "gsm8k",
            },
            {
                "prompt_index": "a", "position": 4, "source_step": 1, "target_step": 2,
                "previous_top1_token_id": 11, "next_top1_token_id": 10, "top1_flip": True,
                "dataset": "gsm8k",
            },
            {
                "prompt_index": "b", "position": 7, "source_step": 0, "target_step": 1,
                "previous_top1_token_id": 5, "next_top1_token_id": 5, "top1_flip": False,
                "dataset": "ifeval",
            },
        ]
        aggregate = aggregate_trajectory_flips(transitions, bootstrap_iterations=None)
        self.assertEqual(aggregate["micro"]["transition_count"], 3)
        self.assertEqual(aggregate["micro"]["flip_count"], 2)
        self.assertAlmostEqual(aggregate["transition_flip_rate"], 2 / 3)
        self.assertAlmostEqual(aggregate["macro"]["prompt_mean_transition_flip_rate"], 0.5)
        self.assertAlmostEqual(aggregate["position_ever_flip_rate"], 0.5)
        self.assertEqual(aggregate["flipback"]["flipback_count"], 1)
        self.assertEqual(aggregate["horizon_retention"]["1"]["eligible_count"], 3)
        self.assertEqual(aggregate["horizon_retention"]["1"]["retained_count"], 1)

        grouped = grouped_flip_rate_table(transitions, group_fields=("dataset",), bootstrap_iterations=None)
        self.assertEqual({row["dataset"] for row in grouped}, {"gsm8k", "ifeval"})

    def test_future_token_tables_use_membership_or_rank_and_keep_next_step_denominator(self) -> None:
        observations = [
            {
                "prompt_index": "a", "eventual_token_rank": 2,
                "current_top1_matches_eventual": False,
                "next_step_top1_in_current_top1": False,
                "next_step_top1_in_current_top2": True,
                "next_step_top1_in_current_top3": True,
                "next_step_top1_in_current_top5": True,
                "next_step_top1_in_current_top10": True,
            },
            {
                "prompt_index": "b", "eventual_token_rank": 8,
                "current_top1_matches_eventual": False,
                "next_step_top1_in_current_top1": False,
                "next_step_top1_in_current_top2": False,
                "next_step_top1_in_current_top3": False,
                "next_step_top1_in_current_top5": False,
                "next_step_top1_in_current_top10": True,
            },
            # No next state: eventual ranks remain eligible, next-step coverage does not.
            {"prompt_index": "b", "eventual_token_rank": 1, "current_top1_matches_eventual": True},
        ]
        coverage = future_token_topk_table(observations, bootstrap_iterations=None)
        eventual_k3 = next(
            row for row in coverage
            if row["coverage_kind"] == "eventual_committed_token" and row["k"] == 3
        )
        next_k3 = next(
            row for row in coverage
            if row["coverage_kind"] == "next_step_top1" and row["k"] == 3
        )
        self.assertEqual(eventual_k3["observation_count"], 3)
        self.assertAlmostEqual(eventual_k3["coverage_rate"], 2 / 3)
        self.assertEqual(next_k3["observation_count"], 2)
        self.assertAlmostEqual(next_k3["coverage_rate"], 0.5)

        ranks = future_token_rank_table(observations, bootstrap_iterations=None)
        rank_two = next(row for row in ranks if row["rank_category"] == "rank_2")
        self.assertEqual(rank_two["observation_count"], 2)
        self.assertAlmostEqual(rank_two["rate"], 0.5)

    def test_counterfactual_and_order_summaries_are_schema_tolerant(self) -> None:
        causal = counterfactual_attribution_table([
            {"classification": "singleton_sufficient", "reconstruction_matches": True},
            {"classification": "synergy_only", "reconstruction_matches": False},
            {"classification": "synergy_only", "reconstruction_matches": True},
        ])
        synergy = next(row for row in causal if row["classification"] == "synergy_only")
        self.assertEqual(synergy["event_count"], 2)
        self.assertAlmostEqual(synergy["classification_rate"], 2 / 3)

        replays = [
            {"event_id": "x", "mode": "fixed", "order": [1, 2], "steps": [{"prefix_size": 1, "tracked_top1": {9: 4}}]},
            {"event_id": "x", "mode": "fixed", "order": [2, 1], "steps": [{"prefix_size": 1, "tracked_top1": {9: 5}}]},
            {"event_id": "y", "mode": "adaptive", "order": [1, 2], "final_assignments": {1: 7, 2: 8}},
            {"event_id": "y", "mode": "adaptive", "order": [2, 1], "final_assignments": {1: 7, 2: 8}},
        ]
        order = order_sensitivity_table(replays)
        fixed = next(row for row in order if row["mode"] == "fixed")
        adaptive = next(row for row in order if row["mode"] == "adaptive")
        self.assertAlmostEqual(fixed["order_sensitivity_rate"], 1.0)
        self.assertAlmostEqual(adaptive["order_sensitivity_rate"], 0.0)

    def test_fixed_order_summary_ignores_anchor_keys_that_disappear_by_definition(self) -> None:
        # At prefix one, different orders naturally retain a different anchor
        # key.  Only the non-anchor position 9 is comparable; it stays fixed.
        # Treating the disappearing anchor keys as evidence would manufacture
        # a 100% fixed-order effect.
        replays = [
            {
                "state_key": "z",
                "mode": "fixed",
                "order": [{"position": 1}, {"position": 2}],
                "steps": [
                    {"prefix_size": 0, "committed_position": 1, "top1_token_id": 7, "original_token_is_top1": True, "tracked_top1": {"1": 7, "2": 8, "9": 4}},
                    {"prefix_size": 1, "committed_position": 2, "top1_token_id": 8, "original_token_is_top1": True, "tracked_top1": {"2": 8, "9": 4}},
                ],
            },
            {
                "state_key": "z",
                "mode": "fixed",
                "order": [{"position": 2}, {"position": 1}],
                "steps": [
                    {"prefix_size": 0, "committed_position": 2, "top1_token_id": 8, "original_token_is_top1": True, "tracked_top1": {"1": 7, "2": 8, "9": 4}},
                    {"prefix_size": 1, "committed_position": 1, "top1_token_id": 7, "original_token_is_top1": True, "tracked_top1": {"1": 7, "9": 4}},
                ],
            },
        ]
        fixed = next(
            row for row in order_sensitivity_table(replays, state_fields=("state_key",)) if row["mode"] == "fixed"
        )
        self.assertEqual(fixed["evaluated_state_count"], 1)
        self.assertEqual(fixed["order_sensitive_state_count"], 0)

    def test_jsonl_csv_and_table_writers_emit_reusable_artifacts(self) -> None:
        rows = [{"prompt_index": 1, "nested": {"token": "가"}, "value": 0.5}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jsonl_path = write_jsonl(root / "raw" / "rows.jsonl", rows)
            csv_path = write_csv(root / "tables" / "rows.csv", rows)
            artifacts = write_artifact_tables(root / "bundle", {"summary": rows}, formats=("csv", "jsonl"))
            self.assertTrue(jsonl_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertEqual(len(artifacts["summary"]), 2)
            self.assertEqual(json.loads(jsonl_path.read_text(encoding="utf-8"))["nested"]["token"], "가")
            with csv_path.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(json.loads(csv_rows[0]["nested"])["token"], "가")


if __name__ == "__main__":
    unittest.main()
