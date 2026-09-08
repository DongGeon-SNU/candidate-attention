"""CPU-only exact-definition checks for the VCCC oracle DP."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vccc_oracle import (  # noqa: E402
    classify_certificate,
    directional_dependency_graph,
    existential_dp,
    maximum_bottleneck_order,
    order_bottleneck,
    pairwise_safe,
    residual_rows,
)


def _row(margin: float, *, match: bool = True) -> dict[str, object]:
    return {
        "logit_margin": margin,
        "top1_matches_assignment": match,
        "is_logit_tie": abs(margin) < 1.0e-12,
    }


class VCCCOracleTest(unittest.TestCase):
    def test_exists_only_has_a_safe_order_but_not_universal_certificate(self) -> None:
        # Two values: candidate 1 cannot be safely queried after 0, while
        # order (1, 0) is safe.  Base queries are safe in both directions.
        margins = {
            (0, 0): _row(1.0),
            (1, 0): _row(1.0),
            (0, 2): _row(1.0),
            (1, 1): _row(-0.4, match=False),
        }
        result = classify_certificate(margins, 2, 0.0)
        self.assertFalse(result.all_pass)
        self.assertTrue(result.existential_pass)
        self.assertTrue(result.final_loo_pass is False)
        self.assertEqual(result.category, "EXISTS_ONLY")
        self.assertEqual(result.witness_order, (1, 0))
        self.assertEqual(result.maximum_bottleneck_order, (1, 0))

    def test_dp_agrees_with_permutation_maximum_for_four_candidates(self) -> None:
        import itertools

        margins: dict[tuple[int, int], dict[str, object]] = {}
        for target in range(4):
            for mask in range(16):
                if mask & (1 << target):
                    continue
                # Context-sensitive but fully finite toy margins.
                margins[(target, mask)] = _row(2.0 - 0.3 * mask.bit_count() - 0.1 * target)
        dp_score, dp_order = maximum_bottleneck_order(margins, 4)
        exhaustive = [
            (order_bottleneck(margins, order), order)
            for order in itertools.permutations(range(4))
        ]
        best = max(score for score, _order in exhaustive if score is not None)
        self.assertAlmostEqual(float(dp_score), float(best))
        self.assertAlmostEqual(float(order_bottleneck(margins, dp_order or ())), float(best))
        for gamma in (0.0, 0.25, 0.5, 1.0):
            observed, _witness = existential_dp(margins, 4, gamma)
            expected = any((score is not None and score >= gamma) for score, _order in exhaustive)
            self.assertEqual(observed, expected)

    def test_tie_needs_deterministic_top1_match(self) -> None:
        margins = {
            (0, 0): _row(0.0, match=False),
            (1, 0): _row(1.0),
            (0, 2): _row(0.0, match=False),
            (1, 1): _row(1.0),
        }
        result = classify_certificate(margins, 2, 0.0)
        self.assertFalse(result.existential_pass)
        self.assertGreater(result.tie_query_count, 0)

    def test_residual_and_pairwise_graph_have_stable_orientation(self) -> None:
        margins = {
            (0, 0): _row(1.0),
            (1, 0): _row(1.0),
            (0, 2): _row(1.0),
            # Revealing candidate 0 makes target 1 unsafe: 1 must precede 0.
            (1, 1): _row(-0.5, match=False),
        }
        graph = directional_dependency_graph(margins, 2, 0.0)
        self.assertEqual(graph["edges"], ((1, 0),))
        self.assertEqual(graph["topological_order"], (1, 0))
        self.assertTrue(pairwise_safe(margins, 2, 0.0) is False)
        rows = residual_rows(margins, 2)
        self.assertEqual(len(rows), 4)
        target_one = next(row for row in rows if row["target_index"] == 1 and row["revealed_mask"] == 1)
        self.assertAlmostEqual(float(target_one["residual"]), 0.0)


if __name__ == "__main__":
    unittest.main()
