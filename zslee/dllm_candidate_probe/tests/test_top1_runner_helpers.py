"""CPU-only checks for top-1 audit runner safety helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# The production environment installs PyYAML.  The lightweight local runtime
# used for pure helper tests does not need to parse a config, so provide the
# smallest import shim rather than skipping resource/cohort safety coverage.
if importlib.util.find_spec("yaml") is None:
    sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda value: {}))

from scripts.run_top1_dynamics_audit import (  # noqa: E402
    ObservationBundle,
    _select_counterfactual_cohort,
    estimate_resources,
    state_identifier_from_common,
)


def _config() -> dict[str, object]:
    return {
        "decoding": {
            "generation_length": 16,
            "steps": 16,
            "requested_max_generation_length": 512,
            "requested_block_size": 64,
            "analysis_block_size": None,
        },
        "sampling": {
            "thresholds": {"sensitivity": [0.8, 0.95]},
            "max_counterfactual_events": 20,
            "event_target_per_cohort": 10,
            "matched_control_target": 10,
            "confidence_bins": [0.0, 0.95, 1.0000001],
            "bootstrap_seed": 7,
            "max_order_states_per_anchor_size": 12,
            "max_order_permutations": 32,
            "order_anchor_sizes": [2, 3, 4, 6, 8],
        },
        "resource_policy": {
            "max_peak_vram_mib": 81_920,
            "max_runtime_seconds": 14_400,
            "max_additional_disk_gib": 50,
            "counterfactual_runtime_fraction": 0.45,
        },
    }


class Top1RunnerHelpersTest(unittest.TestCase):
    def test_resource_bound_uses_configured_worst_case_not_stale_literals(self) -> None:
        bundle = ObservationBundle(
            trajectories=[],
            state_index=[],
            state_position_rows=[],
            transitions=[],
            eventual_rows=[],
            prompt_rows=[{"prompt_id": "smoke"}],
            sanity_rows=[],
            natural_seconds=1.0,
            exact_seconds=2.0,
            natural_forward_count=2,
            exact_forward_count=4,
            sanity_forward_count=0,
            peak_vram_mib=1_024.0,
        )
        estimate = estimate_resources(
            bundle,
            _config(),
            primary_example_count=564,
            sensitivity_example_upper_bound=150,
            analysis_calibration={"analysis_runtime_seconds_estimate": 100.0},
        )
        # Configured total-event cap × (base + 16 singleton + joint + 16 LOO).
        self.assertEqual(estimate["counterfactual_forward_upper_bound"], 20 * 34)
        # 12 states each for sizes 2,3,4,6,8; fixed + adaptive prefixes.
        self.assertEqual(estimate["order_forward_upper_bound"], 13_584)
        self.assertGreater(estimate["observational_runtime_seconds_estimate"], 100.0)

    def test_counterfactual_cap_preserves_multiple_matched_strata(self) -> None:
        rows: list[dict[str, object]] = []
        for dataset in ("gsm8k", "ifeval"):
            for role in ("event", "control"):
                for index in range(20):
                    rows.append(
                        {
                            "prompt_id": f"{dataset}-{role}-{index}",
                            "setting": "primary",
                            "threshold": 0.9,
                            "source_step": 0,
                            "target_step": 1,
                            "position": index,
                            "source_anchor_count": 2,
                            "source_top2_token_id": 9,
                            "top1_flip": role == "event",
                            "dataset": dataset,
                            "source_phase": "early",
                            "previous_top1_probability": 0.91,
                        }
                    )
        selected, metadata = _select_counterfactual_cohort(rows, _config())
        events = [row for row in selected if row["cohort_role"] == "event"]
        controls = [row for row in selected if row["cohort_role"] == "control"]
        self.assertEqual(len(events), 10)
        self.assertEqual(len(controls), 10)
        self.assertEqual({row["dataset"] for row in events}, {"gsm8k", "ifeval"})
        self.assertEqual(metadata["available_matched_pair_count"], 40)

    def test_state_key_includes_named_setting(self) -> None:
        base = {"prompt_id": "gsm8k:x", "threshold": 0.9, "setting": "smoke"}
        primary = {**base, "setting": "primary"}
        self.assertNotEqual(state_identifier_from_common(base, 2), state_identifier_from_common(primary, 2))


if __name__ == "__main__":
    unittest.main()
