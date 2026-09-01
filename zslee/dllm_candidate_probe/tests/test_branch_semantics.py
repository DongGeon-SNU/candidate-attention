"""No-model tests for the non-negotiable counterfactual branch semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.branching import Candidate, make_branch, make_leave_one_out_branches, validate_candidates
from src.set_search import valid_candidate_sets


class BranchSemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = 99
        self.base = [[1, 2, self.mask, 4, self.mask, self.mask, 7]]
        self.positions = [[0, 1, 2, 3, 4, 5, 6]]

    def test_singleton_changes_exactly_one_requested_mask(self) -> None:
        branch = make_branch(self.base, [Candidate(2, 11)], mask_token_id=self.mask)
        self.assertEqual(branch, [[1, 2, 11, 4, self.mask, self.mask, 7]])
        self.assertEqual(self.base[0][2], self.mask)

    def test_pair_changes_exactly_two_requested_masks(self) -> None:
        branch = make_branch(self.base, [Candidate(2, 11), Candidate(5, 12)], mask_token_id=self.mask)
        changed = [index for index, (before, after) in enumerate(zip(self.base[0], branch[0])) if before != after]
        self.assertEqual(changed, [2, 5])
        self.assertEqual(branch[0][2], 11)
        self.assertEqual(branch[0][5], 12)

    def test_same_position_candidates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most one"):
            make_branch(self.base, [Candidate(2, 11), Candidate(2, 12)], mask_token_id=self.mask)
        sets = list(valid_candidate_sets([Candidate(2, 11), Candidate(2, 12), Candidate(4, 13)], 2))
        self.assertEqual(sets, [(Candidate(2, 11), Candidate(4, 13)), (Candidate(2, 12), Candidate(4, 13))])

    def test_leave_one_out_removes_exactly_one_candidate(self) -> None:
        candidate_set = [Candidate(2, 11), Candidate(4, 12), Candidate(5, 13)]
        branches = make_leave_one_out_branches(self.base, candidate_set, mask_token_id=self.mask)
        self.assertEqual(len(branches), 3)
        for omitted, branch in branches.items():
            changed = [index for index, (before, after) in enumerate(zip(self.base[0], branch[0])) if before != after]
            expected = sorted(candidate.position for candidate in candidate_set if candidate != omitted)
            self.assertEqual(changed, expected)
            self.assertEqual(branch[0][omitted.position], self.mask)

    def test_non_mask_overwrite_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only replace"):
            make_branch(self.base, [Candidate(1, 11)], mask_token_id=self.mask)

    def test_position_ids_are_branch_invariant_by_construction(self) -> None:
        before = [row[:] for row in self.positions]
        make_branch(self.base, [Candidate(2, 11)], mask_token_id=self.mask)
        self.assertEqual(self.positions, before)

    def test_seed_independent_branch_construction_is_reproducible(self) -> None:
        first = make_branch(self.base, [Candidate(2, 11), Candidate(4, 12)], mask_token_id=self.mask)
        second = make_branch(self.base, [Candidate(2, 11), Candidate(4, 12)], mask_token_id=self.mask)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
