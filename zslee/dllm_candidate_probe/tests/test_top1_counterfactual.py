"""CPU-only checks for exact top-1 attribution and order replay semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - exercised in minimal non-Torch CI
    torch = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if torch is not None:
    from src.branching import Candidate
    from src.top1_counterfactual import (
        additive_log_space_prediction,
        classify_attribution,
        deterministic_anchor_orders,
        distribution_tv_js,
        exact_anchor_attribution,
        g_score,
        replay_anchor_orders,
    )


MASK = 7


def synergy_forward(input_ids: torch.Tensor) -> torch.Tensor:
    """A tiny contextual forward with a genuinely joint-only target flip."""

    logits = torch.full((1, 3, 8), -8.0, dtype=torch.float32)
    logits[:, :, 0] = 0.0
    has_left = int(input_ids[0, 0]) == 1
    has_right = int(input_ids[0, 1]) == 2
    # Target position two: base g(1, 0)=-2; each singleton is -1.2, while
    # their joint produces +2.  Singleton log-space addition predicts -0.4.
    logits[0, 2, 0] = 2.0
    logits[0, 2, 1] = 0.0
    if has_left and has_right:
        logits[0, 2, 0] = 0.0
        logits[0, 2, 1] = 2.0
    elif has_left or has_right:
        logits[0, 2, 1] = 0.8
    return logits


def order_forward(input_ids: torch.Tensor) -> torch.Tensor:
    """A context-sensitive forward whose adaptive assignment depends on order."""

    logits = torch.full((1, 3, 8), -8.0, dtype=torch.float32)
    logits[:, :, 0] = 0.0
    # Anchor 0 starts with token 1, but becomes token 4 after anchor 1 gets
    # its original assignment.  Anchor 1 has the analogous reaction.
    logits[0, 0, 4 if int(input_ids[0, 1]) == 2 else 1] = 3.0
    logits[0, 1, 3 if int(input_ids[0, 0]) == 1 else 2] = 3.0
    logits[0, 2, 6 if int(input_ids[0, 1]) == 2 and int(input_ids[0, 0]) != 1 else 5] = 3.0
    return logits


@unittest.skipUnless(torch is not None, "PyTorch is required for CPU tensor counterfactual tests.")
class Top1CounterfactualTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base = torch.tensor([[MASK, MASK, MASK]], dtype=torch.long)
        self.anchors = (Candidate(0, 1), Candidate(1, 2))

    def test_exact_attribution_measures_all_branches_and_joint_only_synergy(self) -> None:
        result = exact_anchor_attribution(
            self.base,
            self.anchors,
            mask_token_id=MASK,
            target_position=2,
            old_token_id=0,
            new_token_id=1,
            forward=synergy_forward,
            observed_next_input_ids=torch.tensor([[1, 2, MASK]], dtype=torch.long),
        )

        self.assertLess(result.base_g, 0.0)
        self.assertGreater(result.joint_g, 0.0)
        self.assertFalse(result.singleton_sufficient)
        self.assertFalse(result.additive_crosses)
        self.assertTrue(result.joint_crosses)
        self.assertEqual(result.classification, "synergy-only")
        self.assertEqual(result.exact_forward_count, 6)  # base + 2 singleton + joint + 2 LOO
        self.assertEqual(len(result.anchor_effects), 2)
        self.assertTrue(result.reconstruction_matches)
        self.assertTrue(result.reconstruction_input_ids_match)
        self.assertGreater(result.additive_prediction_tv, 0.0)
        self.assertGreater(result.additive_prediction_js, 0.0)
        self.assertAlmostEqual(
            result.joint_total_effect - result.singleton_effect_sum,
            result.additive_residual,
            places=6,
        )

    def test_reconstruction_mismatch_is_never_credited_to_anchors(self) -> None:
        result = exact_anchor_attribution(
            self.base,
            self.anchors,
            mask_token_id=MASK,
            target_position=2,
            old_token_id=0,
            new_token_id=1,
            forward=synergy_forward,
            # A non-anchor state mutation must preempt even a plausible g flip.
            observed_next_input_ids=torch.tensor([[1, 2, 6]], dtype=torch.long),
        )

        self.assertFalse(result.reconstruction_matches)
        self.assertEqual(result.classification, "unattributed")

    def test_g_and_log_space_addition_use_full_probability_distributions(self) -> None:
        base_logits = torch.tensor([[[2.0, 0.0, -2.0]]])
        one_logits = torch.tensor([[[2.0, 1.0, -2.0]]])
        two_logits = torch.tensor([[[2.0, 2.0, -2.0]]])
        self.assertAlmostEqual(g_score(base_logits, 0, 1, 0), -2.0, places=6)
        predicted = additive_log_space_prediction(
            torch.log_softmax(base_logits[0, 0], dim=-1),
            [torch.log_softmax(one_logits[0, 0], dim=-1)] * 2,
        )
        actual = torch.softmax(two_logits[0, 0], dim=-1)
        distances = distribution_tv_js(predicted, actual)
        self.assertAlmostEqual(float(predicted.sum()), 1.0, places=6)
        # Here the dummy model obeys exactly the singleton log-odds rule.
        self.assertAlmostEqual(distances["tv"], 0.0, places=6)
        self.assertAlmostEqual(distances["js"], 0.0, places=6)

    def test_classification_precedence_covers_required_categories(self) -> None:
        self.assertEqual(classify_attribution(-1.0, [0.1], 0.5), "singleton-sufficient")
        self.assertEqual(classify_attribution(-1.0, [-0.2, -0.2], 0.4), "multi-singleton-additive")
        self.assertEqual(classify_attribution(-1.0, [-0.6, -0.6], 0.4), "synergy-only")
        self.assertEqual(classify_attribution(-1.0, [0.1, -1.0], -0.2), "suppression/cancellation")
        self.assertEqual(
            classify_attribution(-1.0, [0.1], 0.5, reconstruction_matches=False), "unattributed"
        )

    def test_order_replay_checks_fixed_final_state_and_adaptive_order_sensitivity(self) -> None:
        audit = replay_anchor_orders(
            self.base,
            self.anchors,
            mask_token_id=MASK,
            forward=order_forward,
            threshold=0.5,
            tracked_positions=[2],
            seed=11,
        )

        self.assertEqual(len(audit.orders), 2)
        self.assertTrue(audit.fixed_final_states_all_equal)
        self.assertTrue(all(replay.final_state_matches_joint for replay in audit.fixed_replays))
        self.assertTrue(
            all(torch.equal(replay.final_input_ids, torch.tensor([[1, 2, MASK]])) for replay in audit.fixed_replays)
        )
        # The same position order has one original token and one changed token;
        # the opposite order produces a different final adaptive assignment.
        signatures = {
            tuple((candidate.position, candidate.token_id) for candidate in replay.final_assignments)
            for replay in audit.adaptive_replays
        }
        self.assertEqual(signatures, {((0, 1), (1, 3)), ((0, 4), (1, 2))})
        self.assertTrue(audit.adaptive_final_assignment_order_sensitive)
        self.assertTrue(audit.adaptive_decision_trajectory_order_sensitive)
        self.assertTrue(all(replay.original_assignment_match_rate == 0.5 for replay in audit.adaptive_replays))

    def test_order_selection_is_exhaustive_to_four_and_seeded_above_four(self) -> None:
        four = tuple(Candidate(position, position + 1) for position in range(4))
        self.assertEqual(len(deterministic_anchor_orders(four, max_permutations=3, seed=1)), 24)
        five = tuple(Candidate(position, position + 1) for position in range(5))
        first = deterministic_anchor_orders(five, max_permutations=32, seed=17)
        second = deterministic_anchor_orders(five, max_permutations=32, seed=17)
        self.assertEqual(len(first), 32)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 32)


if __name__ == "__main__":
    unittest.main()
