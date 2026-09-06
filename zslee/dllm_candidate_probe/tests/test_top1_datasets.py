"""CPU-only tests for safe, deterministic public benchmark prompt loading."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.top1_datasets import canonical_dataset_name, load_benchmark, load_benchmarks


class Top1DatasetLoadingTest(unittest.TestCase):
    def _write_jsonl(self, root: Path, name: str, rows: list[dict[str, object]]) -> Path:
        path = root / name
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_gsm8k_local_file_extracts_final_answer_and_has_visible_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_jsonl(Path(temporary), "gsm8k.jsonl", [
                {"question": "What is 2 + 2?", "answer": "Reasoning. #### 4"},
                {"question": "What is 3 + 3?", "answer": "Reasoning. #### 6"},
            ])
            result = load_benchmark("GSM8K", local_path=path, limit=1, seed=7, allow_remote=False)
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.examples), 1)
        self.assertEqual(result.examples[0].reference["final_answer"], "4" if "2 + 2" in result.examples[0].prompt else "6")
        self.assertEqual(result.provenance["source_type"], "local_file")
        self.assertTrue(result.provenance["is_benchmark"])

    def test_humaneval_and_ifeval_local_schema_preserves_task_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            humaneval = self._write_jsonl(root, "humaneval.jsonl", [{
                "task_id": "HumanEval/0", "prompt": "def identity(x):\n", "canonical_solution": "    return x\n", "entry_point": "identity",
            }])
            ifeval = self._write_jsonl(root, "ifeval.jsonl", [{
                "key": 1000, "prompt": "Write two bullet points.",
                "instruction_id_list": ["detectable_format:number_bullet_lists"], "kwargs": [{"num_bullets": 2}],
            }])
            code = load_benchmark("human_eval", local_path=humaneval, allow_remote=False)
            instruction = load_benchmark("google/IFEval", local_path=ifeval, allow_remote=False)
        self.assertEqual(code.examples[0].example_id, "HumanEval/0")
        self.assertEqual(code.examples[0].metadata["entry_point"], "identity")
        self.assertEqual(instruction.examples[0].example_id, "1000")
        self.assertEqual(instruction.examples[0].metadata["instruction_id_list"], ["detectable_format:number_bullet_lists"])

    def test_missing_data_is_unavailable_unless_fallback_is_explicit(self) -> None:
        unavailable = load_benchmark("gsm8k", local_path="does-not-exist.jsonl", allow_remote=False)
        fallback = load_benchmark("gsm8k", local_path="does-not-exist.jsonl", allow_remote=False, allow_fallback=True)
        self.assertEqual(unavailable.status, "unavailable")
        self.assertFalse(unavailable.examples)
        self.assertEqual(fallback.status, "fallback")
        self.assertEqual(fallback.provenance["source_type"], "built_in_fallback")
        self.assertFalse(fallback.is_benchmark)

    def test_hash_selection_is_deterministic_and_batch_loader_keeps_failures_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_jsonl(Path(temporary), "gsm8k.jsonl", [
                {"question": f"Question {index}", "answer": f"#### {index}"} for index in range(10)
            ])
            first = load_benchmark("gsm8k", local_path=path, limit=4, seed=99, allow_remote=False)
            second = load_benchmark("gsm8k", local_path=path, limit=4, seed=99, allow_remote=False)
            results = load_benchmarks(
                ["gsm8k", "ifeval"], local_paths={"gsm8k": path}, limits={"gsm8k": 2}, allow_remote=False
            )
        self.assertEqual([item.example_id for item in first.examples], [item.example_id for item in second.examples])
        self.assertEqual(results["gsm8k"].status, "success")
        self.assertEqual(results["ifeval"].status, "unavailable")

    def test_aliases_are_strict(self) -> None:
        self.assertEqual(canonical_dataset_name("openai/openai_humaneval"), "humaneval")
        with self.assertRaises(ValueError):
            canonical_dataset_name("private/unknown-benchmark")


if __name__ == "__main__":
    unittest.main()
