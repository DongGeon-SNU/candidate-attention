"""Synthetic CPU checks for scalar-only top-1 dynamics metrics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamics_metrics import (
    aggregate_trajectory_metrics,
    clustered_bootstrap_mean,
    distribution_summary,
    distribution_summary_from_probabilities,
    flipback_event_indices,
    fp32_probabilities,
    future_flip_label,
    horizon_top1_retention,
    one_step_flip,
    token_rank,
    top_k_contains,
    top_k_tokens,
    trajectory_flip_metrics,
    transition_metrics,
    transition_metrics_from_logits,
    wilson_interval,
)


class DynamicsMetricsTest(unittest.TestCase):
    def test_full_distribution_summary_uses_expected_fp32_statistics(self) -> None:
        # exp(logits) is [1, 2, 3, 4], so this is an exact, human-checkable
        # softmax distribution rather than a truncated top-k approximation.
        logits = [math.log(1), math.log(2), math.log(3), math.log(4)]
        summary = distribution_summary(logits)
        expected_probabilities = [0.1, 0.2, 0.3, 0.4]
        expected_entropy = -sum(value * math.log(value) for value in expected_probabilities)

        self.assertEqual(summary["top1_token_id"], 3)
        self.assertEqual(summary["top2_token_id"], 2)
        self.assertAlmostEqual(summary["top1_probability"], 0.4, places=7)
        self.assertAlmostEqual(summary["top2_probability"], 0.3, places=7)
        self.assertAlmostEqual(summary["top1_minus_top2"], 0.1, places=7)
        self.assertAlmostEqual(summary["top1_to_top2_ratio"], 4 / 3, places=7)
        self.assertAlmostEqual(summary["log_probability_ratio"], math.log(4 / 3), places=7)
        self.assertAlmostEqual(summary["logit_margin"], math.log(4 / 3), places=7)
        self.assertAlmostEqual(summary["entropy"], expected_entropy, places=7)
        self.assertAlmostEqual(summary["normalized_entropy"], expected_entropy / math.log(4), places=7)
        self.assertAlmostEqual(summary["effective_vocabulary_size"], math.exp(expected_entropy), places=7)
        self.assertAlmostEqual(summary["gini_impurity"], 0.7, places=7)
        self.assertAlmostEqual(summary["top3_probability_mass"], 0.9, places=7)
        self.assertAlmostEqual(summary["top5_probability_mass"], 1.0, places=7)
        self.assertAlmostEqual(summary["top10_probability_mass"], 1.0, places=7)
        self.assertAlmostEqual(sum(fp32_probabilities(logits)), 1.0, places=7)

    def test_rank_and_top_k_ties_break_by_token_id(self) -> None:
        probabilities = [0.5, 0.5, 0.0]
        self.assertEqual(token_rank(probabilities, 0), 1)
        self.assertEqual(token_rank(probabilities, 1), 2)
        self.assertEqual(top_k_tokens(probabilities, 2)[0], [0, 1])
        self.assertTrue(top_k_contains(probabilities, 1, 2))
        self.assertFalse(top_k_contains(probabilities, 2, 2))

    def test_transition_measures_tv_js_kl_overlap_and_rank_changes(self) -> None:
        previous = [0.6, 0.3, 0.1]
        current = [0.2, 0.7, 0.1]
        metrics = transition_metrics(previous, current, top_ks=(2,))

        self.assertTrue(metrics["top1_changed"])
        self.assertEqual(metrics["previous_top1_token_id"], 0)
        self.assertEqual(metrics["current_top1_token_id"], 1)
        self.assertAlmostEqual(metrics["tv_distance"], 0.4, places=7)
        self.assertGreater(metrics["js_divergence"], 0.0)
        self.assertGreater(metrics["kl_previous_to_current"], 0.0)
        self.assertGreater(metrics["kl_current_to_previous"], 0.0)
        self.assertEqual(metrics["top2_overlap_count"], 2)
        self.assertAlmostEqual(metrics["top2_overlap_fraction"], 1.0, places=7)
        self.assertEqual(metrics["previous_top1_current_rank"], 2)
        self.assertEqual(metrics["current_top1_previous_rank"], 2)
        self.assertAlmostEqual(metrics["previous_top1_probability_change"], -0.4, places=7)
        self.assertAlmostEqual(metrics["current_top1_probability_gain"], 0.4, places=7)
        self.assertAlmostEqual(metrics["old_to_new_log_odds_change"], math.log(7), places=7)

    def test_logits_transition_keeps_distribution_summaries_scalar_only(self) -> None:
        result = transition_metrics_from_logits(
            [math.log(0.6), math.log(0.3), math.log(0.1)],
            [math.log(0.2), math.log(0.7), math.log(0.1)],
        )
        self.assertTrue(result["top1_changed"])
        self.assertIn("entropy", result["previous_distribution"])
        self.assertIn("top_token_ids", result["current_distribution"])
        self.assertNotIn("probabilities", result["previous_distribution"])

    def test_flip_helpers_exclude_unmasked_transitions_and_identify_flipbacks(self) -> None:
        trajectory = [1, 1, 2, 1, 1, None, 3, 3]
        metrics = trajectory_flip_metrics(trajectory)

        self.assertTrue(one_step_flip(1, 2))
        self.assertFalse(one_step_flip(1, 2, both_masked=False))
        self.assertFalse(one_step_flip(1, None))
        self.assertEqual(metrics["eligible_transition_count"], 5)
        self.assertEqual(metrics["flip_count"], 2)
        self.assertAlmostEqual(metrics["transition_flip_rate"], 0.4, places=7)
        self.assertTrue(metrics["ever_flip"])
        self.assertEqual(flipback_event_indices(trajectory), [3])
        self.assertEqual(metrics["flipback_count"], 1)
        self.assertEqual(metrics["flipback_eligible_count"], 2)
        self.assertAlmostEqual(metrics["flipback_rate"], 0.5, places=7)
        self.assertAlmostEqual(metrics["mean_top1_run_length_states"], 1.75, places=7)
        self.assertEqual(metrics["horizon_retention"]["2"]["eligible_count"], 3)
        self.assertEqual(metrics["horizon_retention"]["2"]["retained_count"], 1)
        self.assertAlmostEqual(metrics["horizon_retention"]["2"]["retention_rate"], 1 / 3, places=7)
        self.assertEqual(future_flip_label(trajectory, 0, 2), True)
        self.assertIsNone(future_flip_label(trajectory, 3, 2))
        retention = horizon_top1_retention(trajectory, 1)
        self.assertEqual((retention["eligible_count"], retention["retained_count"]), (5, 3))

    def test_aggregate_trajectory_metrics_reports_micro_and_position_denominators(self) -> None:
        aggregate = aggregate_trajectory_metrics([[1, 2, 2], [5, 5, None], [9]])
        self.assertEqual(aggregate["position_count"], 3)
        self.assertEqual(aggregate["transition_count"], 3)
        self.assertEqual(aggregate["flip_count"], 1)
        self.assertAlmostEqual(aggregate["transition_flip_rate"], 1 / 3, places=7)
        self.assertEqual(aggregate["ever_flip_position_count"], 1)
        self.assertAlmostEqual(aggregate["position_ever_flip_rate"], 1 / 3, places=7)

    def test_wilson_and_cluster_bootstrap_are_prompt_clustered_and_deterministic(self) -> None:
        low, high = wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.2)
        self.assertLess(high, 0.3)

        values = {"prompt_a": [0.0, 0.0], "prompt_b": [1.0]}
        first = clustered_bootstrap_mean(values, iterations=400, seed=7, weighting="macro")
        second = clustered_bootstrap_mean(values, iterations=400, seed=7, weighting="macro")
        micro = clustered_bootstrap_mean(values, iterations=400, seed=7, weighting="micro")
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 0.5, places=7)
        self.assertAlmostEqual(micro["estimate"], 1 / 3, places=7)
        self.assertEqual(first["cluster_count"], 2)
        self.assertEqual(first["observation_count"], 3)
        self.assertLessEqual(first["ci95_low"], first["estimate"])
        self.assertGreaterEqual(first["ci95_high"], first["estimate"])

    def test_probability_summary_can_normalize_rounded_input(self) -> None:
        summary = distribution_summary_from_probabilities([2.0, 3.0])
        self.assertAlmostEqual(summary["top1_probability"], 0.6, places=7)
        self.assertAlmostEqual(summary["top2_probability"], 0.4, places=7)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("torch") is not None,
    "PyTorch is optional for the dynamics metrics module.",
)
class TorchDynamicsMetricsTest(unittest.TestCase):
    def test_tensor_logits_are_promoted_to_fp32_and_match_python_statistics(self) -> None:
        import torch

        logits = torch.tensor([math.log(1), math.log(2), math.log(3), math.log(4)], dtype=torch.float16)
        probabilities = fp32_probabilities(logits)
        self.assertEqual(probabilities.dtype, torch.float32)
        self.assertEqual(probabilities.device.type, "cpu")
        summary = distribution_summary(logits)
        self.assertEqual(summary["top1_token_id"], 3)
        self.assertAlmostEqual(summary["top1_probability"], 0.4, places=3)


if __name__ == "__main__":
    unittest.main()
