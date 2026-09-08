"""Non-GPU schema checks for the VCCC oracle runner."""

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

from scripts.run_vccc_oracle_audit import SelectedState, certificate_rows_for_state, state_candidates  # noqa: E402


def _config() -> dict[str, object]:
    return {
        "sampling": {
            "margin_thresholds": [0.0],
            "p1_minus_p2_bin_edges": [0.1, 1.0],
            "logit_margin_bin_edges": [0.0, 1.0],
            "local_topk_values": [1, 2, 3],
        },
        "certificate": {"tie_tolerance": 1.0e-7},
    }


class VCCCOracleRunnerHelpersTest(unittest.TestCase):
    def _record(self) -> dict[str, object]:
        return {
            "state_key": "state",
            "dataset": "gsm8k",
            "prompt_id": "gsm8k:0",
            "prompt_index": 0,
            "step": 2,
            "generated_mask_ratio": .5,
            "token_sequence": [1, 7, 7, 3],
            "position_summaries": {
                "1": {"top1_token_id": 10, "p1_minus_p2": .2},
                "2": {"top1_token_id": 11, "p1_minus_p2": .3},
            },
            "actual_committed_anchors": [{"position": 1, "token_id": 10}, {"position": 2, "token_id": 11}],
        }

    def test_state_candidates_rejects_natural_exact_mismatch(self) -> None:
        record = self._record()
        candidates = state_candidates(record)
        self.assertIsNotNone(candidates)
        self.assertEqual([(item.position, item.token_id) for item in candidates or ()], [(1, 10), (2, 11)])
        record["position_summaries"] = {"1": {"top1_token_id": 99}, "2": {"top1_token_id": 11}}
        self.assertIsNone(state_candidates(record))

    def test_certificate_output_keeps_u_e_and_final_loo_separate(self) -> None:
        selected = SelectedState(self._record(), "primary_policy", "unit", state_candidates(self._record()) or ())
        margins = {
            (0, 0): {"logit_margin": 1.0, "top1_matches_assignment": True, "is_logit_tie": False},
            (1, 0): {"logit_margin": 1.0, "top1_matches_assignment": True, "is_logit_tie": False},
            (0, 2): {"logit_margin": 1.0, "top1_matches_assignment": True, "is_logit_tie": False},
            (1, 1): {"logit_margin": -.1, "top1_matches_assignment": False, "is_logit_tie": False},
        }
        for payload in margins.values():
            payload.update({"probability_margin": float(payload["logit_margin"]), "top1_token_id": 10})
        topk = {
            0: {"top1_probability": .95, "p1_minus_p2": .2, "logit_margin": 1.0, "top_token_ids": [10, 1, 2]},
            1: {"top1_probability": .96, "p1_minus_p2": .3, "logit_margin": 1.0, "top_token_ids": [11, 1, 2]},
        }
        state_rows, witnesses, loo, interventions, residuals = certificate_rows_for_state(selected, margins, topk, _config())
        self.assertEqual(state_rows[0]["category"], "EXISTS_ONLY")
        self.assertFalse(state_rows[0]["final_loo_pass"])
        self.assertEqual(witnesses[0]["witness_order_positions"], [2, 1])
        self.assertEqual(len(loo), 2)
        self.assertEqual(len(interventions), 2)
        self.assertEqual(len(residuals), 4)


if __name__ == "__main__":
    unittest.main()
