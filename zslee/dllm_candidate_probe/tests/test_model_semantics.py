"""Torch-level checks for exact forward/cache semantics, run after setup."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.branching import Candidate, exact_forward, make_branch
from src.metrics import probabilities


class _Output:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FrozenToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.calls: list[dict[str, object]] = []

    def forward(self, *, input_ids, attention_mask, position_ids, use_cache):
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "position_ids": position_ids.detach().clone(),
                "use_cache": use_cache,
            }
        )
        logits = torch.nn.functional.one_hot(input_ids % 7, num_classes=7).float() * 4
        return _Output(logits)


class PositionlessToyModel(torch.nn.Module):
    """Matches Fast-dLLM LLaDA's public forward signature."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.seen = None

    def forward(self, *, input_ids, attention_mask, use_cache):
        self.seen = {"input_ids": input_ids.detach().clone(), "attention_mask": attention_mask.detach().clone(), "use_cache": use_cache}
        return _Output(torch.nn.functional.one_hot(input_ids % 7, num_classes=7).float())


class ModelSemanticsTest(unittest.TestCase):
    def test_exact_forward_is_no_cache_and_keeps_position_ids(self) -> None:
        model = FrozenToyModel()
        base = torch.tensor([[1, 99, 99, 4]])
        position_ids = torch.arange(4).unsqueeze(0)
        attention_mask = torch.ones_like(base)
        branch = make_branch(base, [Candidate(1, 3)], mask_token_id=99)
        logits = exact_forward(model, branch, attention_mask=attention_mask, position_ids=position_ids)
        self.assertEqual(tuple(logits.shape), (1, 4, 7))
        self.assertFalse(model.training)
        self.assertEqual(model.calls[0]["use_cache"], False)
        self.assertTrue(torch.equal(model.calls[0]["position_ids"], position_ids))
        self.assertTrue(torch.equal(model.calls[0]["attention_mask"], attention_mask))

    def test_probability_distribution_sums_to_one(self) -> None:
        logits = torch.tensor([[[0.0, 1.0, 2.0], [3.0, -1.0, 0.0]]])
        self.assertTrue(torch.allclose(probabilities(logits).sum(dim=-1), torch.ones((1, 2))))

    def test_positionless_llada_style_model_uses_fixed_length_internal_positions(self) -> None:
        model = PositionlessToyModel()
        input_ids = torch.tensor([[1, 99, 99, 4]])
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(4).unsqueeze(0)
        exact_forward(model, input_ids, attention_mask=attention_mask, position_ids=position_ids)
        self.assertEqual(model.seen["use_cache"], False)
        self.assertTrue(torch.equal(model.seen["input_ids"], input_ids))
        self.assertTrue(torch.equal(model.seen["attention_mask"], attention_mask))

    def test_same_seed_gives_same_counterfactual_logits(self) -> None:
        torch.manual_seed(123)
        first = torch.randn(1, 2, 3)
        torch.manual_seed(123)
        second = torch.randn(1, 2, 3)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
