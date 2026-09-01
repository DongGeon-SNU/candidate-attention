"""Scalar counterfactual metrics; vocabulary tensors never leave the GPU."""

from __future__ import annotations

from typing import Any, Iterable


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised on GPU runtime
        raise RuntimeError("Counterfactual metrics require PyTorch.") from error
    return torch


def probabilities(logits: Any) -> Any:
    """Return a numerically stable full-vocabulary distribution on device."""

    torch = _torch()
    return torch.softmax(logits.float(), dim=-1)


def token_probability(distribution: Any, position: int, token_id: int) -> Any:
    """Extract p(X_position=token_id) without materializing data on CPU."""

    return distribution[0, position, token_id]


def directed_lift(base_probability: Any, counterfactual_probability: Any, epsilon: float = 1e-12) -> Any:
    torch = _torch()
    return torch.log(counterfactual_probability + epsilon) - torch.log(base_probability + epsilon)


def symmetric_compatibility(a_to_b: Any, b_to_a: Any) -> Any:
    return 0.5 * (a_to_b + b_to_a)


def expected_independent_joint(base: Any, singleton_a: Any, singleton_b: Any, epsilon: float = 1e-12) -> Any:
    """Compute softmax(log p_a + log p_b - log p_base) on device."""

    torch = _torch()
    combined_logits = torch.log(singleton_a + epsilon) + torch.log(singleton_b + epsilon) - torch.log(base + epsilon)
    return torch.softmax(combined_logits, dim=-1)


def pair_residuals(
    base: Any,
    singleton_a: Any,
    singleton_b: Any,
    pair: Any,
    remaining_mask_positions: Iterable[int],
) -> dict[str, float | list[float]]:
    """TV residual summary over still-masked target positions only."""

    torch = _torch()
    values: list[float] = []
    for position in remaining_mask_positions:
        expected = expected_independent_joint(
            base[0, position], singleton_a[0, position], singleton_b[0, position]
        )
        observed = pair[0, position]
        values.append(float((0.5 * torch.abs(observed - expected).sum()).item()))
    return {
        "per_position_tv": values,
        "mean_tv": float(sum(values) / len(values)) if values else 0.0,
        "max_tv": float(max(values)) if values else 0.0,
    }


def pair_stability(probability_a_given_b: Any, probability_b_given_a: Any) -> float:
    return float(min(float(probability_a_given_b.item()), float(probability_b_given_a.item())))


def set_stability(leave_one_out_probabilities: Iterable[Any]) -> float:
    values = [float(value.item()) for value in leave_one_out_probabilities]
    if not values:
        raise ValueError("Set stability requires at least one leave-one-out probability.")
    return float(min(values))
