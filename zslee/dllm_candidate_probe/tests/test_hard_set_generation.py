"""CPU-only invariants for hard-set graph generation and scalar cache keys."""

from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hard_set_generation import (
    AuditCandidate,
    PairMetric,
    anchor_conflicts,
    has_unique_positions,
    is_clique,
    pair_key,
    policy_audit_sets,
    set_signature,
    stress_mined_sets,
)
from src.set_audit import ExactScalarCache, pilot_pair_metrics, provenance_validation, scalar_cache_key, state_id, validate_pilot_artifacts


def candidate(position: int, token_id: int, kind: str = "unstable") -> AuditCandidate:
    return AuditCandidate("state", position, token_id, kind, repr(str(token_id)), 0.2, 0.4)


class HardSetGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.anchors = [candidate(1, 101, "anchor")]
        self.unstable = [candidate(position, 200 + position) for position in (2, 3, 4, 5, 6)]
        all_candidates = self.anchors + self.unstable
        self.metrics = {
            pair_key(a, b): PairMetric(a, b, 0.85, 0.8, 0.75, 0.1)
            for index, a in enumerate(all_candidates)
            for b in all_candidates[index + 1 :]
        }

    def test_clique_rejects_same_position_and_missing_edge(self) -> None:
        duplicate = [candidate(2, 1), candidate(2, 2), candidate(3, 3)]
        self.assertFalse(has_unique_positions(duplicate))
        self.assertFalse(is_clique(duplicate, self.metrics, graph="stability_only", pair_threshold=0.7, lift_delta=0.0))
        proposal = tuple(self.anchors + self.unstable[:2])
        self.assertTrue(is_clique(proposal, self.metrics, graph="stability_only", pair_threshold=0.7, lift_delta=0.0))
        del self.metrics[pair_key(proposal[0], proposal[1])]
        self.assertFalse(is_clique(proposal, self.metrics, graph="stability_only", pair_threshold=0.7, lift_delta=0.0))

    def test_strict_graph_detects_anchor_conflict(self) -> None:
        second_anchor = candidate(7, 107, "anchor")
        weak = PairMetric(self.anchors[0], second_anchor, 0.95, 0.1, 0.8)
        metrics = {pair_key(self.anchors[0], second_anchor): weak}
        conflicts = anchor_conflicts(
            [self.anchors[0], second_anchor], metrics, graph="strict_mutual_support", pair_threshold=0.7, lift_delta=0.4
        )
        self.assertEqual(len(conflicts), 1)

    def test_anchor_anchor_anchor_unstable_and_unstable_unstable_edges_are_required(self) -> None:
        second_anchor = candidate(7, 107, "anchor")
        proposed = (self.anchors[0], second_anchor, self.unstable[0], self.unstable[1])
        metrics = {
            pair_key(a, b): PairMetric(a, b, 0.9, 0.6, 0.6)
            for index, a in enumerate(proposed)
            for b in proposed[index + 1 :]
        }
        self.assertTrue(is_clique(proposed, metrics, graph="strict_mutual_support", pair_threshold=0.7, lift_delta=0.5))
        # Removing each edge class (A-A, A-U, U-U) invalidates the hard set.
        for left, right in ((proposed[0], proposed[1]), (proposed[0], proposed[2]), (proposed[2], proposed[3])):
            reduced = dict(metrics)
            del reduced[pair_key(left, right)]
            self.assertFalse(is_clique(proposed, reduced, graph="strict_mutual_support", pair_threshold=0.7, lift_delta=0.5))

    def test_policy_sets_are_reproducible_deduplicated_and_valid(self) -> None:
        first = policy_audit_sets(
            self.anchors, self.unstable, self.metrics, target_size=3, graph="stability_only",
            pair_threshold=0.7, lift_delta=0.0, seed=123, count=4, attempts_per_set=100,
        )
        second = policy_audit_sets(
            self.anchors, self.unstable, self.metrics, target_size=3, graph="stability_only",
            pair_threshold=0.7, lift_delta=0.0, seed=123, count=4, attempts_per_set=100,
        )
        self.assertEqual(first, second)
        self.assertEqual(len({set_signature(row) for row in first}), len(first))
        self.assertTrue(all(has_unique_positions(row) for row in first))
        self.assertTrue(all(is_clique(row, self.metrics, graph="stability_only", pair_threshold=0.7, lift_delta=0.0) for row in first))

    def test_stress_mining_preserves_hard_set_rule(self) -> None:
        result = stress_mined_sets(
            self.anchors, self.unstable, self.metrics, target_size=4, graph="strict_mutual_support",
            pair_threshold=0.7, lift_delta=0.7, seed=20260902, count=3,
        )
        self.assertTrue(result)
        self.assertTrue(all(is_clique(row, self.metrics, graph="strict_mutual_support", pair_threshold=0.7, lift_delta=0.7) for row in result))

    def test_scalar_cache_key_and_state_id_change_with_full_descriptor(self) -> None:
        target = self.unstable[0]
        first = scalar_cache_key(
            input_token_ids=[1, 9, 9], mask_positions=[1, 2], insertions=[self.anchors[0]], target=target,
            model_revision="main", dtype="torch.bfloat16",
        )
        second = scalar_cache_key(
            input_token_ids=[1, 9, 9], mask_positions=[1, 2], insertions=[self.anchors[0]], target=target,
            model_revision="revision-b", dtype="torch.bfloat16",
        )
        self.assertNotEqual(first, second)
        self.assertNotEqual(
            first,
            scalar_cache_key(
                input_token_ids=[1, 9, 9], mask_positions=[1, 2], insertions=[self.anchors[0]], target=target,
                model_revision="main", dtype="torch.bfloat16", cache_schema="hard-set-v2",
            ),
        )
        record = {"prompt_index": 0, "step": 1, "token_sequence": [1, 9], "mask_positions": [1]}
        changed = {**record, "token_sequence": [1, 8]}
        self.assertNotEqual(state_id(record), state_id(changed))
        with tempfile.TemporaryDirectory() as temporary:
            cache = ExactScalarCache(Path(temporary) / "scalar.jsonl")
            cache.put(first, 0.7, {"state_id": "state"})
            self.assertEqual(cache.get(first), 0.7)
            self.assertEqual(ExactScalarCache(cache.path).get(first), 0.7)

    def test_pilot_schema_and_provenance_are_checked_before_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "outputs" / "raw"
            raw.mkdir(parents=True)
            state = {
                "prompt_index": 0, "step": 0, "token_sequence": [1, 99], "mask_positions": [1],
                "position_summaries": {"1": {"top1_confidence": 0.5, "top5_cumulative_probability": 0.9, "top5_token_ids": [3], "top5_probabilities": [0.2]}},
            }
            pair = {
                "prompt_index": 0, "step": 0, "a": {"position": 1, "token_id": 3}, "b": {"position": 0, "token_id": 1},
                "pair_stability_q2": 0.7, "a_to_b_lift": 0.1, "b_to_a_lift": 0.2,
            }
            (raw / "pilot_states.jsonl").write_text(json.dumps(state) + "\n", encoding="utf-8")
            (raw / "pilot_pairs.jsonl").write_text(json.dumps(pair) + "\n", encoding="utf-8")
            (raw / "pilot_sets.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "outputs" / "pilot_result.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
            (root / "outputs" / "summary.md").write_text("fast-pin source-pin\n", encoding="utf-8")
            reuse = {
                "states": "outputs/raw/pilot_states.jsonl", "pairs": "outputs/raw/pilot_pairs.jsonl", "sets": "outputs/raw/pilot_sets.jsonl",
                "result": "outputs/pilot_result.json", "expected_fast_dllm_commit": "fast-pin", "expected_probe_source_commit": "source-pin",
            }
            states, pairs, result = validate_pilot_artifacts(root, reuse)
            self.assertEqual((len(states), len(pairs), result["status"]), (1, 1, "success"))
            provenance = provenance_validation(root, reuse)
            self.assertTrue(provenance["fast_dllm_pin_match"])
            self.assertTrue(provenance["pilot_source_commit_match"])

    def test_reused_pairs_are_scoped_to_full_state_not_local_node_ids(self) -> None:
        def state(prompt_index: int, token_sequence: list[int]) -> dict[str, object]:
            return {
                "prompt_index": prompt_index, "step": 0, "token_sequence": token_sequence, "mask_positions": [1, 2],
                "position_summaries": {
                    "1": {"top1_confidence": 0.4}, "2": {"top1_confidence": 0.4},
                },
            }
        first, second = state(0, [1, 99, 99]), state(1, [2, 99, 99])
        def pair(prompt_index: int, q2: float) -> dict[str, object]:
            return {
                "prompt_index": prompt_index, "step": 0, "a": {"position": 1, "token_id": 10}, "b": {"position": 2, "token_id": 11},
                "pair_stability_q2": q2, "a_to_b_lift": 0.1, "b_to_a_lift": 0.2,
            }
        indexed = pilot_pair_metrics([first, second], [pair(0, 0.55), pair(1, 0.95)])
        self.assertEqual(len(indexed), 2)
        self.assertEqual(next(iter(indexed[state_id(first)].values())).q2, 0.55)
        self.assertEqual(next(iter(indexed[state_id(second)].values())).q2, 0.95)


if __name__ == "__main__":
    unittest.main()
