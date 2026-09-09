"""CPU-only checks for the exact top-1 VCCC headroom oracle helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exact_top1_headroom import (  # noqa: E402
    certificate_cache_for_gamma,
    choose_largest_safe_mask,
    exact_set_certificate,
)


def _row(margin: float, *, match: bool = True, tie: bool = False) -> dict[str, object]:
    return {
        "logit_margin": margin,
        "top1_matches_assignment": match,
        "is_logit_tie": tie,
    }


def _fully_safe_margins(size: int) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (target, revealed): _row(1.0)
        for target in range(size)
        for revealed in range(1 << size)
        if not (revealed & (1 << target))
    }


class ExactTop1HeadroomTest(unittest.TestCase):
    def test_certificate_checks_every_subset_not_only_final_leave_one_out(self) -> None:
        # The only unsafe query is an *intermediate* context for target 2.
        # Its final LOO context (revealed=0b011) remains safe, so an
        # implementation that checked only final LOO would incorrectly pass.
        margins = _fully_safe_margins(3)
        margins[(2, 0b001)] = _row(-0.2, match=False)

        full = exact_set_certificate(margins, 0b111, 0.0)
        self.assertFalse(full.passes)
        self.assertEqual(full.query_count, 12)  # 3 targets * 2^(3 - 1) contexts
        self.assertEqual(full.violating_indices, (2,))
        self.assertEqual(full.first_failure_target_index, 2)
        self.assertEqual(full.first_failure_revealed_mask, 0b001)
        self.assertTrue(
            exact_set_certificate(margins, 0b011, 0.0).passes,
            "The failure must arise only when the third selected token is certified.",
        )

    def test_gamma_zero_tie_requires_deterministic_assignment_match(self) -> None:
        # Margin zero is inclusive at gamma=0, but a raw-logit tie whose
        # deterministic argmax selected another token is not a certificate.
        certificate = exact_set_certificate(
            {(0, 0): _row(0.0, match=False, tie=True)},
            0b1,
            0.0,
        )
        self.assertFalse(certificate.passes)
        self.assertEqual(certificate.certificate_margin, 0.0)
        self.assertEqual(certificate.first_failure_target_index, 0)
        self.assertFalse(certificate.first_failure_top1_matches)
        self.assertTrue(certificate.first_failure_is_tie)

    def test_preserving_policy_returns_none_when_required_base_batch_fails(self) -> None:
        # B={0,1} is invalid because target 1 already violates at the base
        # context.  A free oracle can still choose a different safe set, but
        # a preserving extension is forbidden from silently dropping B.
        margins = _fully_safe_margins(3)
        margins[(1, 0)] = _row(-0.1, match=False)
        certificates = certificate_cache_for_gamma(margins, 3, 0.0)

        preserving = choose_largest_safe_mask(
            certificates,
            probabilities=(0.9, 0.8, 0.7),
            positions=(10, 20, 30),
            required_mask=0b011,
        )
        free = choose_largest_safe_mask(
            certificates,
            probabilities=(0.9, 0.8, 0.7),
            positions=(10, 20, 30),
        )
        self.assertFalse(certificates[0b011].passes)
        self.assertIsNone(preserving)
        self.assertEqual(free, 0b101)

    def test_free_oracle_ties_use_probability_then_position_order(self) -> None:
        # The two candidate masks have equal size.  Higher summed log p wins
        # even though its position tuple is lexicographically later.
        certificates = certificate_cache_for_gamma(_fully_safe_margins(3), 3, 0.0)
        selected_by_probability = choose_largest_safe_mask(
            {0b011: certificates[0b011], 0b101: certificates[0b101]},
            probabilities=(0.5, 0.9, 0.8),
            positions=(10, 40, 20),
        )
        self.assertEqual(selected_by_probability, 0b011)

        # With the log-probability score tied, the lexicographically earlier
        # physical position tuple (10, 20) deterministically wins.
        selected_by_position = choose_largest_safe_mask(
            {0b011: certificates[0b011], 0b101: certificates[0b101]},
            probabilities=(0.8, 0.8, 0.8),
            positions=(10, 40, 20),
        )
        self.assertEqual(selected_by_position, 0b101)

        # Pool construction need not be positional (B comes first, then
        # confidence-ranked extras).  The fallback must sort physical
        # positions inside each candidate set before lexicographic comparison.
        selected_with_nonpositional_pool_order = choose_largest_safe_mask(
            {0b011: certificates[0b011], 0b110: certificates[0b110]},
            probabilities=(0.8, 0.8, 0.8),
            positions=(30, 100, 10),
        )
        self.assertEqual(selected_with_nonpositional_pool_order, 0b110)

    def test_empty_set_is_vacuously_safe(self) -> None:
        certificate = exact_set_certificate({}, 0, 1.0)
        self.assertTrue(certificate.passes)
        self.assertTrue(certificate.vacuous)
        self.assertEqual(certificate.query_count, 0)
        self.assertEqual(certificate.violating_indices, ())
        self.assertIsNone(certificate.certificate_margin)


if __name__ == "__main__":
    unittest.main()
