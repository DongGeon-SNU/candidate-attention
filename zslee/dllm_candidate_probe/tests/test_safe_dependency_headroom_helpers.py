"""CPU-only behavioral tests for v3 DAPD pair-token bookkeeping."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_safe_dependency_headroom.py"


def _load_runner():
    original_torch = sys.modules.get("torch")
    original_yaml = sys.modules.get("yaml")
    module_name = "safe_dependency_headroom_runner_helpers"
    original_module = sys.modules.get(module_name)
    if original_torch is None:
        class _Tensor:  # Matches no ordinary Python container in this CPU-only suite.
            pass

        sys.modules["torch"] = types.SimpleNamespace(Tensor=_Tensor)
    if original_yaml is None:
        sys.modules["yaml"] = types.SimpleNamespace()
    try:
        specification = importlib.util.spec_from_file_location(module_name, RUNNER)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
        return module
    finally:
        if original_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original_torch
        if original_yaml is None:
            sys.modules.pop("yaml", None)
        else:
            sys.modules["yaml"] = original_yaml
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


class SafeDependencyHeadroomHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()

    def test_rejection_snapshot_is_joined_to_the_same_pre_update_step(self) -> None:
        collector = self.runner._DapdTraceCollector("run", "prompt")
        collector.on_decision({
            "batch_index": 0,
            "candidate_position": 11,
            "candidate_rank": 1,
            "selected_positions_before": (13,),
            "accepted": False,
            "decisive_blocker_position": 13,
            "normalized_dependency_score": 0.2,
            "raw_attention_dependency_score": 0.4,
            "edge_threshold": 0.1,
            "combined_selection_score": 0.3,
        })
        collector.on_step({
            "decode_step": 4,
            "masked_position_count": 2,
            "mask_positions": (11, 13),
            "mask_top1_token_ids": (101, 303),
            "graph_selected_positions": (13,),
            "direct_selected_positions": (),
            "staged_added_positions": (),
            "selected_positions": (13,),
            "edge_threshold": 0.1,
            "algorithm": "dapd_direct",
        })
        event = collector.event_rows[0]
        self.assertEqual(event["top1_token_i_at_decision"], 101)
        self.assertEqual(event["top1_token_j_at_decision"], 303)
        self.assertEqual(event["position_i"], 11)
        self.assertEqual(event["position_j"], 13)

    def test_enrichment_keeps_all_three_pair_comparisons_distinct(self) -> None:
        rows = [{
            "prompt_id": "p0",
            "position_i": 11,
            "position_j": 13,
            "top1_token_i_at_decision": 101,
            "top1_token_j_at_decision": 303,
            "eligible_for_headroom": True,
        }]
        common = {"run_id": "run", "prompt_id": "p0", "seed": 7}
        enriched = self.runner._enrich_dapd_events(
            rows,
            common,
            prompt_length=10,
            dapd_ids=[0, 101, 0, 303],
            fast_ids=[0, 202, 0, 303],
        )[0]
        self.assertEqual(enriched["agreement_count"], 1)
        self.assertEqual(enriched["top1_to_dapd_terminal_agreement_count"], 2)
        self.assertEqual(enriched["top1_to_fast_terminal_agreement_count"], 1)
        self.assertTrue(enriched["top1_to_dapd_terminal_match_j"])

    def test_blocker_snapshot_is_retained_when_its_terminal_token_differs(self) -> None:
        rows = [{
            "prompt_id": "p0",
            "position_i": 11,
            "position_j": 13,
            "top1_token_i_at_decision": 101,
            "top1_token_j_at_decision": 999,
            "eligible_for_headroom": True,
        }]
        enriched = self.runner._enrich_dapd_events(
            rows,
            {"run_id": "run", "prompt_id": "p0", "seed": 7},
            prompt_length=10,
            dapd_ids=[0, 101, 0, 303],
            fast_ids=[0, 202, 0, 303],
        )[0]
        self.assertFalse(enriched["top1_to_dapd_terminal_match_j"])

    def test_summary_reports_each_comparison_family(self) -> None:
        events = [{
            "prompt_id": "p0",
            "eligible_for_headroom": True,
            "position_i_generation": 1,
            "position_j_generation": 3,
            "decode_step": 0,
            "candidate_rank": 0,
            "agreement_count": 1,
            "top1_to_dapd_terminal_agreement_count": 2,
            "top1_to_fast_terminal_agreement_count": 1,
        }]
        summary = self.runner._summarize_dapd(
            events,
            ["p0"],
            {"analysis": {"bootstrap_replicates": 4, "bootstrap_seed": 3}},
        )
        comparisons = summary["pair_token_comparisons"]
        self.assertEqual(
            comparisons["dapd_terminal_vs_fast_terminal"]["agreement_rates"]["1/2 match"],
            1.0,
        )
        self.assertEqual(
            comparisons["decision_top1_vs_dapd_terminal"]["agreement_rates"]["2/2 match"],
            1.0,
        )
        self.assertEqual(
            comparisons["decision_top1_vs_fast_terminal"]["agreement_rates"]["1/2 match"],
            1.0,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
