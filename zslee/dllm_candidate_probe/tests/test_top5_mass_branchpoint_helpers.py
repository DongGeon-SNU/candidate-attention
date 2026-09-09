"""CPU-only checks for strict top-5-mass branch-point eligibility."""

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

from scripts.run_top5_mass_branchpoint_audit import (  # noqa: E402
    OrderedPairRequest,
    find_first_branchpoint,
    validate_control_provenance,
)
from scripts.run_vccc_oracle_audit import SelectedState, state_candidates  # noqa: E402


def _record(step: int, *, masks: list[int], mass: float, p1: float = .5) -> dict[str, object]:
    return {
        "state_key": f"s{step}",
        "step": step,
        "mask_positions": masks,
        "position_summaries": {
            "1": {
                "top5_mass": mass,
                "top1_probability": p1,
                "top_token_ids": [9, 8, 7, 6, 5, 4],
                "top_probabilities": [mass / 5] * 5 + [.01],
            }
        },
        "decoder_metadata": {"natural_policy": "confidence_ge_threshold_plus_argmax_fallback"},
        "actual_committed_anchors": [
            {"confidence": .95, "threshold_eligible": True, "selected_by_fallback": False}
        ],
    }


class BranchpointEligibilityTest(unittest.TestCase):
    def _request(self) -> OrderedPairRequest:
        state_record = {
            "state_key": "selected",
            "prompt_id": "p",
            "setting": "primary",
            "threshold": .9,
            "position_summaries": {"1": {"top1_token_id": 9, "p1_minus_p2": .1}, "2": {"top1_token_id": 8, "p1_minus_p2": .1}},
            "actual_committed_anchors": [{"position": 1, "token_id": 9}, {"position": 2, "token_id": 8}],
        }
        selected = SelectedState(state_record, "primary_policy", "unit", state_candidates(state_record) or ())
        return OrderedPairRequest(selected, 1, 2, 0, 1)

    def test_first_strictly_eligible_step_is_retained_and_freezes_probabilities(self) -> None:
        branchpoint, status = find_first_branchpoint(
            self._request(),
            [_record(0, masks=[1, 2], mass=.9), _record(1, masks=[1], mass=.95), _record(2, masks=[1, 2], mass=.95)],
            strict_top5_mass_gate=.9,
        )
        self.assertEqual(status, "retained")
        self.assertIsNotNone(branchpoint)
        assert branchpoint is not None
        self.assertEqual(branchpoint.record["step"], 2)
        self.assertEqual(branchpoint.source_top5_token_ids, (9, 8, 7, 6, 5))
        self.assertAlmostEqual(sum(branchpoint.source_top5_probabilities), .95)

    def test_gate_failure_distinguishes_commit_from_low_mass(self) -> None:
        _point, status = find_first_branchpoint(
            self._request(), [_record(0, masks=[1, 2], mass=.9)], strict_top5_mass_gate=.9
        )
        self.assertEqual(status, "top5_mass_gate_not_met_before_commit")
        _point, status = find_first_branchpoint(
            self._request(), [_record(0, masks=[1], mass=.95)], strict_top5_mass_gate=.9
        )
        self.assertEqual(status, "no_step_with_both_positions_masked")

    def test_control_provenance_keeps_threshold_and_fallback_invariants(self) -> None:
        self.assertIsNone(validate_control_provenance(_record(0, masks=[1, 2], mass=.95), threshold=.9))
        bad = _record(0, masks=[1, 2], mass=.95)
        bad["actual_committed_anchors"] = [{"confidence": .5, "threshold_eligible": True, "selected_by_fallback": False}]
        self.assertEqual(validate_control_provenance(bad, threshold=.9), "threshold_anchor_below_threshold")


if __name__ == "__main__":
    unittest.main()
