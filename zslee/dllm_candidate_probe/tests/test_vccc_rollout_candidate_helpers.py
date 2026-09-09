"""CPU-only protocol checks for branchpoint and candidate heterogeneity logic."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
if "yaml" not in sys.modules and importlib.util.find_spec("yaml") is None:
    sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda value: {}))

from scripts.run_vccc_rollout_candidate_audit import (  # noqa: E402
    PriorPair,
    Rollout,
    _control_provenance_error,
    annotate_branch_outcomes,
    branch_summaries,
    compare_same_time,
    first_branchpoint,
    heterogeneity_rows,
    pair_heterogeneity_rows,
    prior_pairs,
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
            {**base, "source_candidate_token_id": 10, "same_time_control_available": True, "induced_flip": False, "changed_to": None},
            {**base, "source_candidate_token_id": 11, "same_time_control_available": True, "induced_flip": True, "changed_to": 41},
            {**base, "source_candidate_token_id": 12, "same_time_control_available": True, "induced_flip": True, "changed_to": 42},
        ]
        summary = heterogeneity_rows(rows)[0]
        self.assertTrue(summary["candidate_flip_outcome_heterogeneous"])
        self.assertTrue(summary["candidate_replacement_token_heterogeneous"])
        self.assertEqual(summary["distinct_flipped_replacement_token_count"], 2)
        pair_summary = pair_heterogeneity_rows(rows)[0]
        self.assertTrue(pair_summary["candidate_ever_flip_outcome_heterogeneous"])
        self.assertTrue(pair_summary["candidate_replacement_token_heterogeneous_across_horizons"])

    def test_extended_selection_preserves_old_pairs_and_is_outcome_blind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            old = [
                {"cohort": "primary_policy", "state_key": "old", "prompt_id": "p-old", "source_position": 1, "target_position": 2},
                {"cohort": "primary_policy", "state_key": "old", "prompt_id": "p-old", "source_position": 2, "target_position": 1},
            ]
            selected = [
                {"cohort": "primary_policy", "state_key": "old", "prompt_id": "p-old", "assignment_positions": [1, 2, 3]},
                {"cohort": "hard_order_sensitive", "state_key": "hard", "prompt_id": "p-hard", "assignment_positions": [4, 5]},
            ]
            (raw / "candidate_polarity_assignments.jsonl").write_text("\n".join(json.dumps(row) for row in old) + "\n", encoding="utf-8")
            (raw / "selected_states.jsonl").write_text("\n".join(json.dumps(row) for row in selected) + "\n", encoding="utf-8")
            pairs = prior_pairs(root, 4)
        keys = {(pair.selection_state_key, pair.source_position, pair.target_position) for pair in pairs}
        self.assertEqual(len(pairs), 4)
        self.assertIn(("old", 1, 2), keys)
        self.assertIn(("old", 2, 1), keys)
        self.assertTrue(all(pair.source_position != pair.target_position for pair in pairs))
        self.assertTrue(all(pair.selection_state_key != "hard" for pair in pairs))

    def test_post_control_horizons_are_retained_but_not_compared(self) -> None:
        base = {
            "branchpoint_state_key": "state",
            "prompt_id": "prompt",
            "source_position": 3,
            "target_position": 5,
            "source_candidate_token_id": 10,
            "source_candidate_probability": .9,
            "target_top2_token_id": 2,
            "target_top1_probability": .8,
            "target_top2_probability": .1,
            "target_logit_margin": 2.0,
            "target_probability_margin": .7,
            "branch_terminal_before_action": False,
        }
        control = Rollout(records=[{**base, "horizon": 0, "target_top1_token_id": 1}], final_target_token_id=1, terminal_horizon=0)
        treatment = Rollout(
            records=[
                {**base, "horizon": 0, "target_top1_token_id": 2},
                {**base, "horizon": 1, "target_top1_token_id": 3, "branch_terminal_before_action": True},
            ],
            final_target_token_id=3,
            terminal_horizon=1,
        )
        rows = compare_same_time(control, treatment)
        annotate_branch_outcomes(rows, treatment)
        self.assertTrue(rows[0]["same_time_control_available"])
        self.assertTrue(rows[0]["induced_flip"])
        self.assertFalse(rows[1]["same_time_control_available"])
        self.assertIsNone(rows[1]["induced_flip"])
        self.assertEqual(rows[1]["final_actual_committed_target_token_id"], 3)
        summary = branch_summaries(rows, {("state", 3, 5, 10): treatment})[0]
        self.assertEqual(summary["same_time_comparable_horizon_count"], 1)
        self.assertEqual(summary["post_control_horizon_count"], 1)


if __name__ == "__main__":
    unittest.main()
