"""CPU-only schema/semantics checks for the exact headroom runner helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if "yaml" not in sys.modules and importlib.util.find_spec("yaml") is None:
    sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda value: {}))

from scripts.run_exact_top1_vccc_oracle_headroom import (  # noqa: E402
    POLICY_EXTENSION,
    POLICY_FAST,
    SelectedState,
    _pool_for_stratum,
    _validate_actual_batch,
    evaluate_policies_for_pool,
)


def _margins(size: int) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (target, mask): {
            "logit_margin": 1.0,
            "top1_matches_assignment": True,
            "is_logit_tie": False,
        }
        for target in range(size)
        for mask in range(1 << size)
        if not (mask & (1 << target))
    }


def _config() -> dict[str, object]:
    return {
        "certificate": {"tie_tolerance": 1.0e-7},
        "sampling": {"margin_thresholds": [0.0], "confidence_thresholds": [0.0, 0.9]},
        "decoding": {"threshold": 0.9},
    }


class ExactTop1HeadroomRunnerHelpersTest(unittest.TestCase):
    def _state(self) -> SelectedState:
        return SelectedState(
            {
                "state_key": "state",
                "prompt_id": "prompt",
                "dataset": "gsm8k",
                "step": 1,
                "token_sequence": [1, 2, 3],
                "position_summaries": {
                    "3": {"top1_token_id": 30},
                    "4": {"top1_token_id": 40},
                    "5": {"top1_token_id": 50},
                },
            },
            "primary",
            "unit",
            "test",
        )

    def test_pool_never_truncates_an_actual_batch_larger_than_m(self) -> None:
        assignments = {
            3: {"token_id": 30, "top1_probability": .95},
            4: {"token_id": 40, "top1_probability": .85},
            5: {"token_id": 50, "top1_probability": .75},
        }
        row, error = _pool_for_stratum(
            self._state(),
            actual_batch_positions=[3, 4, 5],
            assignments=assignments,
            requested_size=2,
        )
        self.assertEqual(error, "ineligible_base_batch_exceeds_pool_size")
        self.assertEqual(row["actual_fast_dllm_positions"], [3, 4, 5])
        self.assertIsNone(row["effective_pool_size"])

    def test_extension_keeps_base_failure_explicit_instead_of_zero_extra(self) -> None:
        margins = _margins(2)
        # Actual B={0}; its base query is unsafe.  A free set may still exist,
        # but preserving extension cannot report a zero-capacity success.
        margins[(0, 0)] = {
            "logit_margin": -.2,
            "top1_matches_assignment": False,
            "is_logit_tie": False,
        }
        rows, token_rows = evaluate_policies_for_pool(
            self._state(),
            requested_pool_size=2,
            positions=[3, 4],
            token_ids=[30, 40],
            probabilities=[.95, .8],
            actual_batch_positions=[3],
            margins=margins,
            config=_config(),
        )
        fast = next(row for row in rows if row["policy"] == POLICY_FAST)
        extension = next(row for row in rows if row["policy"] == POLICY_EXTENSION)
        self.assertTrue(fast["batch_failure"])
        self.assertEqual(extension["extension_status"], "base_batch_exact_certificate_failed")
        self.assertIsNone(extension["selected_size"])
        self.assertIsNone(extension["safe_added_capacity"])
        self.assertFalse(extension["has_at_least_one_safe_extra"])
        self.assertTrue(token_rows, "The actual base-token violation must retain a token-level witness.")

    def test_preserving_extension_uses_cached_subsets_and_adds_the_safe_token(self) -> None:
        rows, _tokens = evaluate_policies_for_pool(
            self._state(),
            requested_pool_size=2,
            positions=[3, 4],
            token_ids=[30, 40],
            probabilities=[.95, .8],
            actual_batch_positions=[3],
            margins=_margins(2),
            config=_config(),
        )
        extension = next(row for row in rows if row["policy"] == POLICY_EXTENSION)
        self.assertEqual(extension["selected_positions"], [3, 4])
        self.assertEqual(extension["safe_added_capacity"], 1)
        self.assertTrue(extension["certificate_pass"])

    def test_actual_batch_validation_requires_every_original_threshold_position(self) -> None:
        state = SelectedState(
            {
                **dict(self._state().record),
                "actual_committed_anchors": [
                    {
                        "position": 3,
                        "token_id": 30,
                        "confidence": .95,
                        "threshold_eligible": True,
                        "selected_by_fallback": False,
                        "is_highest_confidence": True,
                    }
                ],
            },
            "primary",
            "unit",
            "test",
        )
        assignments = {
            3: {"token_id": 30},
            4: {"token_id": 40},
        }
        natural = {
            3: {"natural_top1_token_id": 30, "natural_top1_probability": .95},
            4: {"natural_top1_token_id": 40, "natural_top1_probability": .91},
        }
        positions, error = _validate_actual_batch(
            state,
            mask_positions={3, 4},
            mask_positions_in_policy_order=[3, 4],
            fresh_assignments=assignments,
            natural_position_rows=natural,
            config=_config(),
        )
        self.assertIsNone(positions)
        self.assertEqual(error, "actual_batch_positions_do_not_match_original_threshold_plus_fallback_action")


if __name__ == "__main__":
    unittest.main()
