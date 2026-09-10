"""CPU-only source-contract checks for the Safe Dependency Headroom audit.

These checks deliberately do not import the GPU runner or a model.  The audit
is only valid when the pinned decoder implementations expose observation-only
hooks; reproducing either decoder's selection loop in the runner would make a
trace a post-hoc approximation rather than evidence about the real decoder.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_safe_dependency_headroom.py"
SETUP = PROJECT_ROOT / "scripts" / "setup.sh"
DAPD_PATCH = PROJECT_ROOT / "patches" / "dapd_trace_hooks.patch"
FAST_PATCH = PROJECT_ROOT / "patches" / "fast_dllm_trace_hooks.patch"
CONFIGS = (
    PROJECT_ROOT / "configs" / "safe_dependency_headroom_smoke.yaml",
    PROJECT_ROOT / "configs" / "safe_dependency_headroom_full.yaml",
)
FULL_COHORT = PROJECT_ROOT / "prompts" / "safe_dependency_headroom_primary.jsonl"


def _mapping_scalar(config: str, section: str, key: str) -> str:
    """Return a simple scalar from a two-space-indented YAML mapping.

    Keeping this parser intentionally tiny makes the contract suite usable on
    an unprovisioned checkout where PyYAML, torch, and transformers are absent.
    The experiment configs use scalar values for these safety-critical fields.
    """

    match = re.search(
        rf"(?ms)^{re.escape(section)}:\s*$\n(?P<body>.*?)(?=^[^ \t]|\Z)",
        config,
    )
    if match is None:
        raise AssertionError(f"missing YAML mapping: {section}")
    scalar = re.search(
        rf"(?m)^  {re.escape(key)}:\s*(?P<value>[^#\n]+?)\s*$",
        match.group("body"),
    )
    if scalar is None:
        raise AssertionError(f"missing {section}.{key}")
    return scalar.group("value").strip().strip('"\'')


class SafeDependencyHeadroomContractTest(unittest.TestCase):
    def test_runner_is_valid_python_and_uses_observed_official_decoders(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        ast.parse(source, filename=str(RUNNER))

        # These were the v1 reimplementations.  A v2 trace must invoke the
        # upstream decoders and observe their real choices instead.
        self.assertNotRegex(source, r"(?m)^def (?:fast|dapd|trace)\(")
        self.assertIn("generate_dapd", source)
        self.assertIn("step_observer", source)
        self.assertIn("selection_observer", source)
        self.assertIn("_capture_fast_native_selection_without_step_observer", source)
        self.assertIn("native_selection_trace_equal", source)
        # Fast-dLLM's public generate entry point is decorated with
        # torch.no_grad(); checking the wrapper file would falsely resolve to
        # PyTorch rather than the pinned implementation.
        self.assertIn("inspect.unwrap(function)", source)

    def test_dapd_patch_exposes_first_blocker_and_step_observation(self) -> None:
        patch = DAPD_PATCH.read_text(encoding="utf-8")
        self.assertIn("dapd/core.py", patch)
        self.assertIn("dapd/generation.py", patch)
        self.assertIn("decision_observer", patch)
        self.assertIn("selection_observer", patch)
        self.assertIn("step_observer", patch)
        self.assertIn("raw_dependency", patch)
        # The event must come from the selector's actual first blocking edge,
        # not a graph replay in the audit runner.
        self.assertIn("decisive_blocker_position", patch)
        self.assertIn("raw_attention_dependency_score", patch)

    def test_fast_patch_exposes_an_observation_only_step_hook(self) -> None:
        patch = FAST_PATCH.read_text(encoding="utf-8")
        self.assertIn("v1/llada/generate.py", patch)
        self.assertIn("step_observer", patch)
        self.assertIn("selected_positions", patch)
        # A callback must be optional so calls without logging retain the
        # upstream API and decoder behavior.
        self.assertRegex(patch, r"step_observer\s*=\s*None")

    def test_observation_patches_are_well_formed_unified_diffs(self) -> None:
        for patch_path in (DAPD_PATCH, FAST_PATCH):
            with self.subTest(patch=patch_path.name):
                # --numstat parses hunks but does not modify the checkout.  It
                # catches truncated/corrupt hand-written patches before setup
                # reaches a costly H100 environment.
                result = subprocess.run(
                    [
                        "git",
                        "-c",
                        f"safe.directory={PROJECT_ROOT.resolve()}",
                        "apply",
                        "--numstat",
                        str(patch_path),
                    ],
                    cwd=PROJECT_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_setup_applies_and_verifies_the_pinned_hook_patches(self) -> None:
        setup = SETUP.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("dapd_trace_hooks.patch", setup)
        self.assertIn("fast_dllm_trace_hooks.patch", setup)
        # Verify both pristine applicability and the already-applied case,
        # so setup is deterministic and idempotent on a persistent H100 disk.
        self.assertIn("apply --check", setup)
        self.assertIn("apply --reverse --check", setup)
        self.assertIn("apply --reverse --check \"${patch_file}\" >/dev/null 2>&1", setup)
        self.assertIn("patched_files_sha256", setup)
        # Setup's byte comparison covers the initial patch. The runner must
        # additionally reject a vendor edit made after setup completed.
        self.assertIn("_verify_runtime_patch_provenance", runner)
        self.assertIn("patched file changed after setup verification", runner)
        self.assertIn("TORCH_VERSION", setup)
        self.assertIn("torch==${TORCH_VERSION}", setup)
        self.assertIn("dependency_versions.txt", runner)

    def test_runner_keeps_zero_event_raw_tables_parseable(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("RAW_CSV_SCHEMAS", source)
        self.assertIn('"dapd_events"', source)
        self.assertIn('"dapd_pairs"', source)
        self.assertIn("path.touch(exist_ok=False)", source)

    def test_full_config_has_a_versioned_runnable_prompt_cohort(self) -> None:
        self.assertTrue(FULL_COHORT.is_file())
        full_config = (PROJECT_ROOT / "configs" / "safe_dependency_headroom_full.yaml").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in FULL_COHORT.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(len({row["prompt_id"] for row in rows}), len(rows))
        self.assertTrue(all(isinstance(row.get("prompt"), str) and row["prompt"].strip() for row in rows))
        import hashlib
        self.assertIn(hashlib.sha256(FULL_COHORT.read_bytes()).hexdigest(), full_config)

    def test_configs_are_immutable_greedy_and_match_native_block_semantics(self) -> None:
        for path in CONFIGS:
            with self.subTest(config=path.name):
                config = path.read_text(encoding="utf-8")
                revision = _mapping_scalar(config, "model", "hf_revision")
                self.assertRegex(revision, r"^[0-9a-fA-F]{7,64}$")
                self.assertNotIn(revision.lower(), {"main", "master", "latest", "head"})

                self.assertEqual(_mapping_scalar(config, "decoding", "temperature"), "0.0")
                self.assertEqual(_mapping_scalar(config, "decoding", "top_p"), "null")
                self.assertEqual(
                    _mapping_scalar(config, "decoding", "generation_length"),
                    _mapping_scalar(config, "decoding", "block_length"),
                )
                self.assertEqual(
                    _mapping_scalar(config, "decoding", "steps"),
                    _mapping_scalar(config, "decoding", "generation_length"),
                )
                self.assertEqual(
                    _mapping_scalar(config, "decoding", "use_cache_policy"),
                    "native_source_default",
                )

    def test_configs_name_active_baselines_and_an_explicit_demask_blocker(self) -> None:
        for path in CONFIGS:
            with self.subTest(config=path.name):
                config = path.read_text(encoding="utf-8")
                inline_active = re.search(r"(?m)^\s+active:\s*\[(?P<items>[^\]]+)\]\s*$", config)
                if inline_active is not None:
                    active_items = {
                        item.strip().strip("'\"")
                        for item in inline_active.group("items").split(",")
                        if item.strip()
                    }
                else:
                    block_active = re.search(
                        r"(?ms)^baselines:\s*$\n\s+active:\s*$\n(?P<items>(?:\s+-\s+[^\n]+\n?)+)",
                        config,
                    )
                    self.assertIsNotNone(block_active)
                    active_items = set(
                        re.findall(r"(?m)^\s+-\s+([^\s#]+)", block_active.group("items"))
                    )
                self.assertEqual(active_items, {"fast_dllm", "dapd"})
                self.assertRegex(config, r"(?ms)^\s+demask:\s*$.*?^\s+status:\s+blocked[^\s]*\s*$")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
