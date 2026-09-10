"""CPU-only checks for Experiment 3 exact-VCCC rollout helpers."""

from __future__ import annotations

import importlib.util
import importlib.machinery
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if "yaml" not in sys.modules and importlib.util.find_spec("yaml") is None:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda value: {}
    yaml_stub.__spec__ = importlib.machinery.ModuleSpec("yaml", loader=None)
    sys.modules.setdefault("yaml", yaml_stub)

from scripts.run_exact_top1_vccc_oracle_headroom import (
    FAST_POLICY,
    ForwardAccounting,
    MATCHED_FAST_FORWARD_CONVENTION,
    RolloutResult,
    agreement_rows,
    agreement_summary,
    _candidate_margin_payloads,
    commit_batch_summary,
    final_token_agreement,
    select_fast_dllm_action,
    select_prompt_seeds,
    worst_case_work_bounds,
)
from src.exact_top1_headroom import certificate_cache_for_gamma


def _config() -> dict[str, object]:
    return {
        "source": {
            "required_primary_threshold": 0.9,
            "required_decoder_policy": "confidence_ge_threshold_plus_argmax_fallback",
            "state_selection_salt": "headroom-primary-state",
            "prompt_selection_salt": "headroom-primary-prompt",
            "primary_prompt_cap": 10,
        },
        "decoding": {"generation_length": 2},
        "exact_vccc_oracle": {"tie_tolerance": 1.0e-7},
    }


def _state(
    prompt_id: str,
    state_key: str,
    step: int,
    *,
    tokens: list[int] | None = None,
) -> dict[str, object]:
    return {
        "setting": "primary",
        "threshold": 0.9,
        "prompt_id": prompt_id,
        "dataset": "gsm8k",
        "example_id": prompt_id,
        "state_key": state_key,
        "step": step,
        "token_sequence": tokens or [10, 11, 99, 99],
        "generation_length": 2,
        "measurement": "exact_no_cache",
        "decoder_metadata": {
            "natural_policy": "confidence_ge_threshold_plus_argmax_fallback",
            "exact_measurement_use_cache": False,
        },
        "position_summaries": {
            "2": {"top1_token_id": 7, "logit_margin": 1.0},
            "3": {"top1_token_id": 8, "logit_margin": 1.0},
        },
    }


class ExactTop1HeadroomRunnerHelpersTest(unittest.TestCase):
    def test_fast_threshold_rule_uses_first_argmax_only_when_no_position_is_eligible(self) -> None:
        # No position reaches .8, so the first .70 maximum is committed.
        self.assertEqual(select_fast_dllm_action((0.70, 0.70, 0.65), 0.8), (0,))
        # The fallback is already in the threshold set when one or more
        # positions reach threshold; it must not add a different low-p1 row.
        self.assertEqual(select_fast_dllm_action((0.70, 0.83, 0.81), 0.8), (1, 2))

    def test_final_output_agreement_reports_sequence_and_position_metrics(self) -> None:
        partial = final_token_agreement((1, 2, 3, 4), (1, 9, 3, 8))
        self.assertFalse(partial["final_sequence_exact_match"])
        self.assertEqual(partial["matching_token_count"], 2)
        self.assertEqual(partial["compared_token_count"], 4)
        self.assertEqual(partial["token_position_agreement"], 0.5)

        unequal_length = final_token_agreement((1, 2, 3), (1, 2))
        self.assertFalse(unequal_length["same_generation_length"])
        self.assertFalse(unequal_length["final_sequence_exact_match"])
        self.assertEqual(unequal_length["token_position_agreement"], 1.0)

    def test_primary_agreement_excludes_incomplete_rollouts(self) -> None:
        baseline = RolloutResult(
            prompt_id="p",
            dataset=None,
            example_id=None,
            policy=FAST_POLICY,
            candidate_k=None,
            forward_convention=MATCHED_FAST_FORWARD_CONVENTION,
            completed=True,
            terminal_status="completed",
            final_generation_token_ids=(1, 2),
            wall_seconds=1.0,
            accounting=ForwardAccounting(committed_tokens=2),
            step_rows=[],
            context_rows=[],
            query_rows=[],
        )
        incomplete_oracle = RolloutResult(
            prompt_id="p",
            dataset=None,
            example_id=None,
            policy="exact_top1_vccc_oracle",
            candidate_k=2,
            forward_convention="test",
            completed=False,
            terminal_status="incomplete",
            final_generation_token_ids=(1, 99),
            wall_seconds=1.0,
            accounting=ForwardAccounting(committed_tokens=1),
            step_rows=[],
            context_rows=[],
            query_rows=[],
        )
        rows = agreement_rows([baseline], [incomplete_oracle])
        self.assertEqual(rows[0]["agreement_status"], "excluded_incomplete_rollout_from_primary_agreement")
        self.assertIsNone(rows[0]["token_position_agreement"])
        self.assertEqual(rows[0]["raw_token_position_agreement"], 0.5)
        config = {"reporting": {"clustered_bootstrap_replicates": 3, "bootstrap_seed": 1}}
        summary = agreement_summary(rows, config)[0]
        self.assertEqual(summary["agreement_eligible_prompt_count"], 0)
        self.assertIsNone(summary["micro_token_position_agreement"])

    def test_vectorized_candidate_payloads_match_scalar_margin_summary(self) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover - CPU-only environment without torch
            self.skipTest("torch is unavailable")
        from scripts.run_vccc_oracle_audit import margin_summary

        # Include a non-top1 assigned value and an exact logit tie to exercise
        # the two scalar semantics that certificates depend on.
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.9, 0.3, 0.1],
                    [1.0, 1.0, -1.0, -2.0],
                    [0.5, 0.2, 0.4, 0.3],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.7, 0.4, 0.6, 0.2],
                    [0.1, 0.8, 0.3, 0.2],
                    [0.0, -0.5, 0.4, 0.6],
                ],
            ],
            dtype=torch.float32,
        )
        positions = (1, 2)
        token_ids = (1, 0)
        reduced = _candidate_margin_payloads(logits, positions, token_ids, tie_tolerance=1.0e-7)
        for batch_index in range(2):
            for target_index, (position, token_id) in enumerate(zip(positions, token_ids, strict=True)):
                expected = margin_summary(logits[batch_index], position, token_id, tie_tolerance=1.0e-7)
                actual = reduced[batch_index][target_index]
                self.assertEqual(actual["competitor_token_id"], expected["competitor_token_id"])
                self.assertEqual(actual["top1_token_id"], expected["top1_token_id"])
                self.assertEqual(actual["top1_matches_assignment"], expected["top1_matches_assignment"])
                self.assertEqual(actual["is_logit_tie"], expected["is_logit_tie"])
                self.assertEqual(actual["logit_margin"], expected["logit_margin"])
                for field in ("probability_margin", "assigned_probability", "competitor_probability"):
                    self.assertAlmostEqual(actual[field], expected[field], places=7)

        # The exact all-order decision must agree with the scalar helper at
        # both a zero threshold and a boundary derived from the same logits.
        four_logits = torch.cat((logits, logits), dim=0)
        vector_rows = _candidate_margin_payloads(four_logits, positions, token_ids, tie_tolerance=1.0e-7)
        scalar_margins = {}
        vector_margins = {}
        for mask in range(4):
            for target_index, (position, token_id) in enumerate(zip(positions, token_ids, strict=True)):
                if mask & (1 << target_index):
                    continue
                scalar_margins[(target_index, mask)] = margin_summary(
                    four_logits[mask], position, token_id, tie_tolerance=1.0e-7
                )
                vector_margins[(target_index, mask)] = vector_rows[mask][target_index]
        boundary = scalar_margins[(0, 0)]["logit_margin"]
        for gamma in (0.0, boundary):
            self.assertEqual(
                {
                    mask: certificate.passes
                    for mask, certificate in certificate_cache_for_gamma(
                        scalar_margins, 2, gamma, tolerance=1.0e-7
                    ).items()
                },
                {
                    mask: certificate.passes
                    for mask, certificate in certificate_cache_for_gamma(
                        vector_margins, 2, gamma, tolerance=1.0e-7
                    ).items()
                },
            )

    def test_prompt_selection_reuses_a_prompt_cohort_but_always_returns_t0_seed(self) -> None:
        trajectories = [
            _state("a", "a-state-0", 0),
            _state("a", "a-state-1", 1, tokens=[10, 11, 42, 99]),
            _state("b", "b-state-0", 0),
        ]
        seeds, screening, counts = select_prompt_seeds(trajectories, _config(), smoke=False)
        self.assertEqual({seed.prompt_id for seed in seeds}, {"a", "b"})
        self.assertEqual({seed.source_t0_state_key for seed in seeds}, {"a-state-0", "b-state-0"})
        self.assertEqual(counts["selected_t0_seed_count"], 2)
        self.assertTrue(any(row["status"] == "selected_t0_seed_pending_mask_validation" for row in screening))

    def test_prompt_selection_excludes_ambiguous_source_topk_ties_before_argmax_replay(self) -> None:
        tied = _state("a", "a-state-0", 0)
        tied["position_summaries"] = {
            "2": {"top1_token_id": 7, "logit_margin": 0.0},
            "3": {"top1_token_id": 8, "logit_margin": 1.0},
        }
        seeds, screening, counts = select_prompt_seeds([tied], _config(), smoke=False)
        self.assertEqual(seeds, [])
        self.assertEqual(counts["excluded_ambiguous_t0_source_top1_logit_tie"], 1)
        self.assertTrue(
            any(row["exclusion_reason"] == "ambiguous_t0_source_top1_logit_tie" for row in screening)
        )

    def test_commit_batch_summary_uses_actual_oracle_commit_sizes_not_requested_k(self) -> None:
        rows = [
            {"policy": FAST_POLICY, "candidate_k": None, "actual_commit_set_size": 3},
            {"policy": FAST_POLICY, "candidate_k": None, "actual_commit_set_size": 1},
            {
                "policy": "exact_top1_vccc_oracle",
                "candidate_k": 8,
                "actual_commit_set_size": 4,
                "effective_candidate_k": 8,
                "full_candidate_set_certificate_pass": False,
            },
            {
                "policy": "exact_top1_vccc_oracle",
                "candidate_k": 8,
                "actual_commit_set_size": 8,
                "effective_candidate_k": 8,
                "full_candidate_set_certificate_pass": True,
            },
        ]
        summary = {
            (row["policy"], row["candidate_k"]): row
            for row in commit_batch_summary(rows)
        }
        self.assertEqual(summary[("exact_top1_vccc_oracle", 8)]["actual_commit_set_size_mean"], 6.0)
        self.assertEqual(summary[("exact_top1_vccc_oracle", 8)]["full_topk_certificate_pass_rate"], 0.5)
        self.assertEqual(summary[(FAST_POLICY, None)]["actual_commit_set_size_median"], 2.0)

    def test_worst_case_work_bound_exposes_the_k8_exponential_cost(self) -> None:
        bounds = worst_case_work_bounds(16, (2, 4, 8), 100)
        self.assertEqual(bounds["exact_by_candidate_k"]["2"]["maximum_exact_forwards_per_prompt"], 62)
        self.assertEqual(bounds["exact_by_candidate_k"]["4"]["maximum_exact_forwards_per_prompt"], 222)
        self.assertEqual(bounds["exact_by_candidate_k"]["8"]["maximum_exact_forwards_per_prompt"], 2558)
        self.assertEqual(bounds["maximum_all_policy_model_forwards"], 287400)
        self.assertEqual(bounds["maximum_scalar_margin_queries"], 1047900)


if __name__ == "__main__":
    unittest.main()
