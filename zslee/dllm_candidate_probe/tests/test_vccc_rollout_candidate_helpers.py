"""CPU-only protocol checks for branchpoint and candidate heterogeneity logic."""

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

from scripts.run_vccc_rollout_candidate_audit import (  # noqa: E402
    PriorPair,
    _control_provenance_error,
    first_branchpoint,
    heterogeneity_rows,
    pair_heterogeneity_rows,
)


def _record(step: int, *, masks: list[int], mass: float) -> dict[str, object]:
    return {
        "state_key": f"state-{step}",
        "step": step,
        "mask_positions": masks,
        "position_summaries": {
            "1": {
                "top5_mass": mass,
                "top1_probability": .5,
                "top_token_ids": [10, 11, 12, 13, 14],
                "top_probabilities": [mass / 5] * 5,
            }
        },
        "decoder_metadata": {"natural_policy": "confidence_ge_threshold_plus_argmax_fallback"},
        "actual_committed_anchors": [{"confidence": .95, "threshold_eligible": True, "selected_by_fallback": False}],
    }


class RolloutCandidateHelpersTest(unittest.TestCase):
    def test_first_branchpoint_is_earliest_strict_gate_and_freezes_top5(self) -> None:
        pair = PriorPair("selected", "prompt", 1, 2)
        point, status = first_branchpoint(
            pair,
            [_record(0, masks=[1, 2], mass=.9), _record(1, masks=[1], mass=.95), _record(2, masks=[1, 2], mass=.95)],
            strict_top5_mass_gate=.9,
        )
        self.assertEqual(status, "retained")
        self.assertIsNotNone(point)
        assert point is not None
        self.assertEqual(point.record["step"], 2)
        self.assertEqual(point.source_top5_token_ids, (10, 11, 12, 13, 14))
        self.assertAlmostEqual(sum(point.source_top5_probabilities), .95)

    def test_control_provenance_does_not_replace_original_policy(self) -> None:
        self.assertIsNone(_control_provenance_error(_record(0, masks=[1, 2], mass=.95), expected_policy="confidence_ge_threshold_plus_argmax_fallback", threshold=.9))
        broken = _record(0, masks=[1, 2], mass=.95)
        broken["decoder_metadata"] = {"natural_policy": "other"}
        self.assertEqual(_control_provenance_error(broken, expected_policy="confidence_ge_threshold_plus_argmax_fallback", threshold=.9), "missing_or_unexpected_control_policy_provenance")

    def test_candidate_heterogeneity_tracks_flip_and_replacement_separately(self) -> None:
        base = {"branchpoint_state_key": "s", "horizon": 2, "prompt_id": "p", "tstar_step": 1, "source_position": 3, "target_position": 5}
        rows = [
            {**base, "source_candidate_token_id": 10, "induced_flip": False, "changed_to": None},
            {**base, "source_candidate_token_id": 11, "induced_flip": True, "changed_to": 41},
            {**base, "source_candidate_token_id": 12, "induced_flip": True, "changed_to": 42},
        ]
        summary = heterogeneity_rows(rows)[0]
        self.assertTrue(summary["candidate_flip_outcome_heterogeneous"])
        self.assertTrue(summary["candidate_replacement_token_heterogeneous"])
        self.assertEqual(summary["distinct_flipped_replacement_token_count"], 2)
        pair_summary = pair_heterogeneity_rows(rows)[0]
        self.assertTrue(pair_summary["candidate_ever_flip_outcome_heterogeneous"])
        self.assertTrue(pair_summary["candidate_replacement_token_heterogeneous_across_horizons"])


if __name__ == "__main__":
    unittest.main()
