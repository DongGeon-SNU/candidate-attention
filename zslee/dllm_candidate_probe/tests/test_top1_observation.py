"""CPU-only checks for trajectory-level exact-observation bookkeeping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.top1_observation import position_trajectory_rows, prediction_label_rows


def _state(step: int, token: int) -> dict[str, object]:
    return {
        "prompt_id": "gsm8k:example-1",
        "prompt_index": 3,
        "dataset": "gsm8k",
        "gsm8k_exact_match": True,
        "position": 17,
        "step": step,
        "top1_token_id": token,
        "top1_probability": 0.7,
        "entropy": 1.0 + step,
        "logit_margin": 2.0 - step,
        "top5_mass_ge_0_8": True,
    }


class Top1ObservationBookkeepingTest(unittest.TestCase):
    def test_position_rows_keep_flipback_and_horizon_denominators(self) -> None:
        # a -> b -> a is an immediate flip-back.  No unobserved/committed
        # state is inserted, so both adjacent transitions are eligible.
        rows = position_trajectory_rows([_state(0, 5), _state(1, 6), _state(2, 5)])
        self.assertEqual(len(rows), 1)
        result = rows[0]
        self.assertEqual(result["eligible_transition_count"], 2)
        self.assertEqual(result["flip_count"], 2)
        self.assertTrue(result["ever_flip"])
        self.assertEqual(result["flipback_count"], 1)
        self.assertAlmostEqual(result["flipback_rate"], 0.5)
        self.assertEqual(result["top1_retention_1_eligible"], 2)
        self.assertEqual(result["top1_retention_2_eligible"], 1)
        self.assertAlmostEqual(result["top1_retention_2_step"], 1.0)
        self.assertTrue(result["gsm8k_exact_match"])

    def test_prediction_labels_never_turn_a_truncated_horizon_into_negative(self) -> None:
        records = prediction_label_rows([_state(0, 5), _state(1, 6), _state(2, 5)])
        by_step = {row["step"]: row for row in records}
        self.assertTrue(by_step[0]["next_step_top1_flip"])
        self.assertTrue(by_step[0]["future_2_step_top1_flip"])
        self.assertTrue(by_step[1]["next_step_top1_flip"])
        self.assertIsNone(by_step[1]["future_2_step_top1_flip"])
        self.assertIsNone(by_step[2]["next_step_top1_flip"])
        self.assertTrue(by_step[0]["any_flip_before_commit"])
        self.assertTrue(by_step[1]["any_flip_before_commit"])
        self.assertFalse(by_step[2]["any_flip_before_commit"])


if __name__ == "__main__":
    unittest.main()
