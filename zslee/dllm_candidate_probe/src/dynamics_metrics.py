"""Numerically stable, scalar-only metrics for top-1 decoding dynamics.

The functions here deliberately return Python scalars and small top-k lists.
They never serialize a full vocabulary distribution.  PyTorch is imported only
when a tensor is supplied, so the pure-Python trajectory/statistics helpers can
also be exercised in a lightweight CPU environment.

Definitions used by this module
-------------------------------
* Ranks are one-based.  Ties are broken by the smaller token ID, matching the
  usual ``argmax`` convention for a vector indexed by token ID.
* A natural transition is eligible only when the target position is still
  masked/observed in *both* adjacent states.  In trajectory helpers, ``None``
  marks an unobserved (normally already committed) state.
* A flip-back is an immediate reversal ``a -> b -> a`` over three consecutive
  observed states.  The result exposes its denominator explicitly rather than
  treating every position as eligible for a flip-back.
* TV and JS are the primary distribution-shift quantities.  KL values are
  included as diagnostics and use a finite clamp for numerical safety.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from statistics import NormalDist
from typing import Any, Literal, TypeAlias


EPSILON = 1e-12
DEFAULT_TOP_KS: tuple[int, ...] = (3, 5, 10)

ProbabilityVector: TypeAlias = Sequence[float] | Any
TokenTrajectory: TypeAlias = Sequence[int | None]


def _torch_or_none() -> Any | None:
    """Import torch lazily so that pure-Python consumers need no torch install."""

    try:
        import torch
    except ImportError:
        return None
    return torch


def _is_torch_tensor(value: Any) -> bool:
    torch = _torch_or_none()
    return bool(torch is not None and torch.is_tensor(value))


def _validate_top_ks(top_ks: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in top_ks}))
    if not values or any(value <= 0 for value in values):
        raise ValueError("top_ks must contain at least one positive integer.")
    return values


def _python_vector(values: ProbabilityVector, *, name: str) -> list[float]:
    """Convert a one-dimensional non-tensor vector to validated Python floats."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional numeric vector, not text.")
    try:
        result = [float(value) for value in values]
    except TypeError as error:
        raise TypeError(f"{name} must be a one-dimensional numeric vector.") from error
    if not result:
        raise ValueError(f"{name} must not be empty.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains a non-finite value.")
    return result


def _normalise_python_probabilities(values: ProbabilityVector) -> list[float]:
    result = _python_vector(values, name="probabilities")
    if any(value < 0 for value in result):
        raise ValueError("probabilities must be non-negative.")
    total = math.fsum(result)
    if total <= 0:
        raise ValueError("probabilities must have positive total mass.")
    return [value / total for value in result]


def _normalise_torch_probabilities(values: Any) -> Any:
    torch = _torch_or_none()
    if torch is None:  # pragma: no cover - guarded by _is_torch_tensor
        raise RuntimeError("PyTorch is required for tensor probabilities.")
    if values.ndim != 1:
        raise ValueError(f"probabilities must be one-dimensional; got shape {tuple(values.shape)}.")
    if values.numel() == 0:
        raise ValueError("probabilities must not be empty.")
    result = values.detach().float()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("probabilities contains a non-finite value.")
    if bool((result < 0).any().item()):
        raise ValueError("probabilities must be non-negative.")
    total = result.sum()
    if float(total.item()) <= 0:
        raise ValueError("probabilities must have positive total mass.")
    return result / total


def _normalise_probabilities(values: ProbabilityVector) -> ProbabilityVector:
    if _is_torch_tensor(values):
        return _normalise_torch_probabilities(values)
    return _normalise_python_probabilities(values)


def fp32_probabilities(logits: ProbabilityVector) -> ProbabilityVector:
    """Return a stable full-vocabulary softmax in FP32 (for tensors).

    A tensor result remains on its original device and is detached from the
    computational graph.  With a Python sequence, standard-library floating
    point arithmetic provides an equivalent stable softmax.
    """

    if _is_torch_tensor(logits):
        torch = _torch_or_none()
        assert torch is not None  # narrows the optional import for type checkers
        if logits.ndim != 1:
            raise ValueError(f"logits must be one-dimensional; got shape {tuple(logits.shape)}.")
        if logits.numel() == 0:
            raise ValueError("logits must not be empty.")
        stable_logits = logits.detach().float()
        if not bool(torch.isfinite(stable_logits).all().item()):
            raise ValueError("logits contains a non-finite value.")
        return torch.softmax(stable_logits, dim=-1)

    values = _python_vector(logits, name="logits")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(exponentials)
    return [value / denominator for value in exponentials]


def token_probability(probabilities: ProbabilityVector, token_id: int) -> float:
    """Return a scalar probability for ``token_id`` from a full distribution."""

    token_id = int(token_id)
    vector = _normalise_probabilities(probabilities)
    if token_id < 0 or token_id >= len(vector):
        raise IndexError(f"token_id {token_id} is outside the vocabulary of size {len(vector)}.")
    if _is_torch_tensor(vector):
        return float(vector[token_id].item())
    return float(vector[token_id])


def token_rank(probabilities: ProbabilityVector, token_id: int) -> int:
    """Return the one-based, deterministically tie-broken rank of ``token_id``."""

    token_id = int(token_id)
    vector = _normalise_probabilities(probabilities)
    vocabulary_size = len(vector)
    if token_id < 0 or token_id >= vocabulary_size:
        raise IndexError(f"token_id {token_id} is outside the vocabulary of size {vocabulary_size}.")
    if _is_torch_tensor(vector):
        torch = _torch_or_none()
        assert torch is not None
        target = vector[token_id]
        indices = torch.arange(vocabulary_size, device=vector.device)
        ahead = (vector > target) | ((vector == target) & (indices < token_id))
        return 1 + int(ahead.sum().item())
    target = vector[token_id]
    return 1 + sum(
        probability > target or (probability == target and index < token_id)
        for index, probability in enumerate(vector)
    )


def top_k_contains(probabilities: ProbabilityVector, token_id: int, k: int) -> bool:
    """Whether a token is in the deterministically ordered top-``k`` list."""

    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive.")
    return token_rank(probabilities, token_id) <= k


def top_k_tokens(probabilities: ProbabilityVector, k: int) -> tuple[list[int], list[float]]:
    """Return deterministic top-k token IDs and their probabilities as lists."""

    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive.")
    vector = _normalise_probabilities(probabilities)
    actual_k = min(k, len(vector))
    if _is_torch_tensor(vector):
        # torch.topk's tie order is not guaranteed across platforms.  Ties are
        # essentially absent for real model logits; normalize the rare exact
        # tie case below only when a deterministic order matters for reporting.
        values, indices = _torch_or_none().topk(vector, actual_k)  # type: ignore[union-attr]
        pairs = sorted(
            ((int(index), float(value)) for index, value in zip(indices.tolist(), values.tolist())),
            key=lambda item: (-item[1], item[0]),
        )
    else:
        pairs = sorted(enumerate(vector), key=lambda item: (-item[1], item[0]))[:actual_k]
    return [int(index) for index, _ in pairs], [float(value) for _, value in pairs]


def _logit_value(logits: ProbabilityVector, token_id: int) -> float:
    if _is_torch_tensor(logits):
        if logits.ndim != 1:
            raise ValueError(f"logits must be one-dimensional; got shape {tuple(logits.shape)}.")
        return float(logits.detach().float()[token_id].item())
    values = _python_vector(logits, name="logits")
    return float(values[token_id])


def distribution_summary_from_probabilities(
    probabilities: ProbabilityVector,
    *,
    logits: ProbabilityVector | None = None,
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Summarize a full distribution without returning the vocabulary vector.

    ``probabilities`` may be rounded or not sum exactly to one: it is
    normalized defensively.  If logits are supplied, their FP32 top-1/top-2
    difference is reported as ``logit_margin``.  Otherwise the equivalent
    log-probability ratio is used.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    requested_ks = _validate_top_ks(top_ks)
    vector = _normalise_probabilities(probabilities)
    vocabulary_size = len(vector)
    report_k = max(2, max(requested_ks))
    token_ids, top_probabilities = top_k_tokens(vector, report_k)
    top1_token_id = token_ids[0]
    top1_probability = top_probabilities[0]
    top2_token_id: int | None = token_ids[1] if vocabulary_size >= 2 else None
    top2_probability = top_probabilities[1] if vocabulary_size >= 2 else 0.0
    if vocabulary_size >= 2:
        probability_ratio: float | None = top1_probability / max(top2_probability, epsilon)
        log_probability_ratio: float | None = math.log(max(top1_probability, epsilon)) - math.log(
            max(top2_probability, epsilon)
        )
        if logits is None:
            logit_margin: float | None = log_probability_ratio
        else:
            if len(logits) != vocabulary_size:
                raise ValueError("logits and probabilities have different vocabulary sizes.")
            logit_margin = _logit_value(logits, top1_token_id) - _logit_value(logits, top2_token_id)
    else:
        probability_ratio = None
        log_probability_ratio = None
        logit_margin = None

    if _is_torch_tensor(vector):
        torch = _torch_or_none()
        assert torch is not None
        entropy = float((-(vector * torch.log(vector.clamp_min(epsilon))).sum()).item())
        gini_impurity = float((1.0 - torch.square(vector).sum()).item())
    else:
        entropy = -math.fsum(value * math.log(max(value, epsilon)) for value in vector)
        gini_impurity = 1.0 - math.fsum(value * value for value in vector)

    masses = {
        k: float(math.fsum(top_probabilities[: min(k, vocabulary_size)]))
        for k in requested_ks
    }
    summary: dict[str, Any] = {
        "vocabulary_size": vocabulary_size,
        "top1_token_id": top1_token_id,
        "top2_token_id": top2_token_id,
        "top1_probability": top1_probability,
        "top2_probability": top2_probability,
        "top1_minus_top2": top1_probability - top2_probability,
        "top1_to_top2_ratio": probability_ratio,
        "log_probability_ratio": log_probability_ratio,
        "logit_margin": logit_margin,
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(vocabulary_size) if vocabulary_size > 1 else 0.0,
        "effective_vocabulary_size": math.exp(entropy),
        "gini_impurity": gini_impurity,
        "top_token_ids": token_ids,
        "top_token_probabilities": top_probabilities,
    }
    for k, mass in masses.items():
        summary[f"top{k}_probability_mass"] = mass
        # A compact alias is useful in JSONL records and mirrors pilot naming.
        summary[f"top{k}_mass"] = mass
    return summary


def distribution_summary(
    logits: ProbabilityVector,
    *,
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Compute an FP32 full-distribution summary directly from logits."""

    return distribution_summary_from_probabilities(
        fp32_probabilities(logits), logits=logits, top_ks=top_ks, epsilon=epsilon
    )


def _distribution_pair(
    previous: ProbabilityVector, current: ProbabilityVector
) -> tuple[ProbabilityVector, ProbabilityVector]:
    """Normalize a pair while preserving the GPU tensor path when possible."""

    previous_is_tensor = _is_torch_tensor(previous)
    current_is_tensor = _is_torch_tensor(current)
    if previous_is_tensor and current_is_tensor and previous.device == current.device:
        left = _normalise_torch_probabilities(previous)
        right = _normalise_torch_probabilities(current)
        if len(left) != len(right):
            raise ValueError("previous and current distributions have different vocabulary sizes.")
        return left, right
    # A mixed backend is uncommon in the runner.  Converting to small Python
    # scalar operations preserves a correct, dependency-light fallback.
    left_values = previous.detach().float().cpu().tolist() if previous_is_tensor else previous
    right_values = current.detach().float().cpu().tolist() if current_is_tensor else current
    left = _normalise_python_probabilities(left_values)
    right = _normalise_python_probabilities(right_values)
    if len(left) != len(right):
        raise ValueError("previous and current distributions have different vocabulary sizes.")
    return left, right


def _scalar_distribution_distances(
    previous: ProbabilityVector, current: ProbabilityVector, epsilon: float
) -> tuple[float, float, float, float]:
    """Return TV, JS, KL(previous||current), and KL(current||previous)."""

    if _is_torch_tensor(previous) and _is_torch_tensor(current):
        torch = _torch_or_none()
        assert torch is not None
        midpoint = 0.5 * (previous + current)
        kl_previous_current = torch.sum(
            previous * (torch.log(previous.clamp_min(epsilon)) - torch.log(current.clamp_min(epsilon)))
        )
        kl_current_previous = torch.sum(
            current * (torch.log(current.clamp_min(epsilon)) - torch.log(previous.clamp_min(epsilon)))
        )
        kl_previous_midpoint = torch.sum(
            previous * (torch.log(previous.clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon)))
        )
        kl_current_midpoint = torch.sum(
            current * (torch.log(current.clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon)))
        )
        tv = 0.5 * torch.abs(previous - current).sum()
        return (
            float(tv.item()),
            float((0.5 * (kl_previous_midpoint + kl_current_midpoint)).item()),
            float(kl_previous_current.item()),
            float(kl_current_previous.item()),
        )

    previous_values = list(previous)
    current_values = list(current)
    tv = 0.5 * math.fsum(abs(left - right) for left, right in zip(previous_values, current_values))
    midpoint = [(left + right) / 2 for left, right in zip(previous_values, current_values)]
    kl_previous_current = math.fsum(
        left * (math.log(max(left, epsilon)) - math.log(max(right, epsilon)))
        for left, right in zip(previous_values, current_values)
    )
    kl_current_previous = math.fsum(
        right * (math.log(max(right, epsilon)) - math.log(max(left, epsilon)))
        for left, right in zip(previous_values, current_values)
    )
    js = 0.5 * (
        math.fsum(
            left * (math.log(max(left, epsilon)) - math.log(max(middle, epsilon)))
            for left, middle in zip(previous_values, midpoint)
        )
        + math.fsum(
            right * (math.log(max(right, epsilon)) - math.log(max(middle, epsilon)))
            for right, middle in zip(current_values, midpoint)
        )
    )
    return tv, js, kl_previous_current, kl_current_previous


def transition_metrics(
    previous_probabilities: ProbabilityVector,
    current_probabilities: ProbabilityVector,
    *,
    top_ks: Iterable[int] = (5, 10),
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Measure full-distribution and rank changes between two observed states.

    Inputs are probability distributions, not logits.  Use
    :func:`transition_metrics_from_logits` when starting from model logits.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    requested_ks = _validate_top_ks(top_ks)
    previous, current = _distribution_pair(previous_probabilities, current_probabilities)
    vocabulary_size = len(previous)
    previous_top_ids, previous_top_probabilities = top_k_tokens(previous, 1)
    current_top_ids, current_top_probabilities = top_k_tokens(current, 1)
    old_token = previous_top_ids[0]
    new_token = current_top_ids[0]
    old_previous_probability = previous_top_probabilities[0]
    new_current_probability = current_top_probabilities[0]
    old_current_probability = token_probability(current, old_token)
    new_previous_probability = token_probability(previous, new_token)
    tv, js, kl_previous_current, kl_current_previous = _scalar_distribution_distances(previous, current, epsilon)

    result: dict[str, Any] = {
        "vocabulary_size": vocabulary_size,
        "previous_top1_token_id": old_token,
        "current_top1_token_id": new_token,
        "top1_changed": old_token != new_token,
        "tv_distance": tv,
        "js_divergence": js,
        "kl_previous_to_current": kl_previous_current,
        "kl_current_to_previous": kl_current_previous,
        "previous_top1_previous_probability": old_previous_probability,
        "previous_top1_current_probability": old_current_probability,
        "previous_top1_current_rank": token_rank(current, old_token),
        "current_top1_previous_probability": new_previous_probability,
        "current_top1_previous_rank": token_rank(previous, new_token),
        "current_top1_current_probability": new_current_probability,
        "previous_top1_probability_change": old_current_probability - old_previous_probability,
        "current_top1_probability_gain": new_current_probability - new_previous_probability,
    }
    previous_log_odds = math.log(max(new_previous_probability, epsilon)) - math.log(max(old_previous_probability, epsilon))
    current_log_odds = math.log(max(new_current_probability, epsilon)) - math.log(max(old_current_probability, epsilon))
    result["old_to_new_log_odds_change"] = current_log_odds - previous_log_odds

    for k in requested_ks:
        actual_k = min(k, vocabulary_size)
        previous_ids, _ = top_k_tokens(previous, actual_k)
        current_ids, _ = top_k_tokens(current, actual_k)
        overlap_count = len(set(previous_ids).intersection(current_ids))
        union_count = len(set(previous_ids).union(current_ids))
        result[f"top{actual_k}_overlap_count"] = overlap_count
        result[f"top{actual_k}_overlap_fraction"] = overlap_count / actual_k
        result[f"top{actual_k}_jaccard"] = overlap_count / union_count if union_count else 1.0
    return result


def transition_metrics_from_logits(
    previous_logits: ProbabilityVector,
    current_logits: ProbabilityVector,
    *,
    top_ks: Iterable[int] = (5, 10),
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Compute transition metrics from logits using the same FP32 softmax path."""

    previous = fp32_probabilities(previous_logits)
    current = fp32_probabilities(current_logits)
    result = transition_metrics(previous, current, top_ks=top_ks, epsilon=epsilon)
    # These small nested summaries make a trajectory JSONL record self-
    # describing without writing full logits or distributions to disk.
    state_top_ks = tuple(sorted(set((*top_ks, 3, 5, 10))))
    result["previous_distribution"] = distribution_summary_from_probabilities(
        previous, logits=previous_logits, top_ks=state_top_ks, epsilon=epsilon
    )
    result["current_distribution"] = distribution_summary_from_probabilities(
        current, logits=current_logits, top_ks=state_top_ks, epsilon=epsilon
    )
    return result


def one_step_flip(previous_top1: int | None, current_top1: int | None, *, both_masked: bool = True) -> bool:
    """Return whether an eligible adjacent state transition changes top-1."""

    return bool(both_masked and previous_top1 is not None and current_top1 is not None and previous_top1 != current_top1)


def top1_runs(top1_tokens: TokenTrajectory) -> list[dict[str, int]]:
    """Return contiguous observed top-1 runs, breaking runs at ``None`` gaps."""

    runs: list[dict[str, int]] = []
    active_token: int | None = None
    active_start: int | None = None
    for index, token in enumerate(top1_tokens):
        if token is None:
            if active_token is not None:
                assert active_start is not None
                runs.append({
                    "token_id": active_token,
                    "start_state_index": active_start,
                    "end_state_index": index - 1,
                    "state_count": index - active_start,
                    "survival_steps": index - active_start - 1,
                })
            active_token = None
            active_start = None
            continue
        token = int(token)
        if active_token is None:
            active_token, active_start = token, index
        elif token != active_token:
            assert active_start is not None
            runs.append({
                "token_id": active_token,
                "start_state_index": active_start,
                "end_state_index": index - 1,
                "state_count": index - active_start,
                "survival_steps": index - active_start - 1,
            })
            active_token, active_start = token, index
    if active_token is not None:
        assert active_start is not None
        runs.append({
            "token_id": active_token,
            "start_state_index": active_start,
            "end_state_index": len(top1_tokens) - 1,
            "state_count": len(top1_tokens) - active_start,
            "survival_steps": len(top1_tokens) - active_start - 1,
        })
    return runs


def flipback_event_indices(top1_tokens: TokenTrajectory) -> list[int]:
    """Indices of immediate ``a -> b -> a`` top-1 flip-backs."""

    result: list[int] = []
    for index in range(2, len(top1_tokens)):
        first, second, third = top1_tokens[index - 2 : index + 1]
        if first is not None and second is not None and third is not None and first != second and first == third:
            result.append(index)
    return result


def horizon_top1_retention(top1_tokens: TokenTrajectory, horizon: int) -> dict[str, Any]:
    """Top-1 retention after ``horizon`` steps, excluding committed/gapped spans."""

    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    eligible = 0
    retained = 0
    for start in range(max(0, len(top1_tokens) - horizon)):
        window = top1_tokens[start : start + horizon + 1]
        if any(token is None for token in window):
            continue
        eligible += 1
        if window[0] == window[-1]:
            retained += 1
    return {
        "horizon": horizon,
        "eligible_count": eligible,
        "retained_count": retained,
        "retention_rate": retained / eligible if eligible else None,
    }


def future_flip_label(top1_tokens: TokenTrajectory, start_index: int, horizon: int) -> bool | None:
    """Whether a flip occurs within a fully observed future horizon.

    ``None`` means the target commits or becomes unobserved before the requested
    horizon is complete, so it must not become a negative example by accident.
    """

    start_index = int(start_index)
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if start_index < 0 or start_index + horizon >= len(top1_tokens):
        return None
    window = top1_tokens[start_index : start_index + horizon + 1]
    if any(token is None for token in window):
        return None
    return any(left != right for left, right in zip(window, window[1:]))


def trajectory_flip_metrics(
    top1_tokens: TokenTrajectory,
    *,
    horizons: Iterable[int] = (1, 2, 3, 5),
) -> dict[str, Any]:
    """Summarize one position's natural top-1 dynamics through its trajectory."""

    requested_horizons = _validate_top_ks(horizons)
    observed_state_count = sum(token is not None for token in top1_tokens)
    eligible_transitions = 0
    flip_count = 0
    for previous, current in zip(top1_tokens, top1_tokens[1:]):
        if previous is None or current is None:
            continue
        eligible_transitions += 1
        flip_count += int(previous != current)
    flipbacks = flipback_event_indices(top1_tokens)
    flipback_eligible = sum(
        first is not None and second is not None and third is not None and first != second
        for first, second, third in zip(top1_tokens, top1_tokens[1:], top1_tokens[2:])
    )
    runs = top1_runs(top1_tokens)
    run_lengths = [int(run["state_count"]) for run in runs]
    survival_steps = [int(run["survival_steps"]) for run in runs]
    result: dict[str, Any] = {
        "observed_state_count": observed_state_count,
        "eligible_transition_count": eligible_transitions,
        "flip_count": flip_count,
        "transition_flip_rate": flip_count / eligible_transitions if eligible_transitions else None,
        "ever_flip": bool(flip_count),
        "flipback_count": len(flipbacks),
        "flipback_event_indices": flipbacks,
        "flipback_eligible_count": flipback_eligible,
        "flipback_rate": len(flipbacks) / flipback_eligible if flipback_eligible else None,
        "top1_run_count": len(runs),
        "mean_top1_run_length_states": math.fsum(run_lengths) / len(run_lengths) if run_lengths else None,
        "mean_top1_survival_steps": math.fsum(survival_steps) / len(survival_steps) if survival_steps else None,
        "max_top1_run_length_states": max(run_lengths) if run_lengths else 0,
        "runs": runs,
    }
    result["horizon_retention"] = {
        str(horizon): horizon_top1_retention(top1_tokens, horizon) for horizon in requested_horizons
    }
    return result


def aggregate_trajectory_metrics(
    trajectories: Iterable[TokenTrajectory],
    *,
    horizons: Iterable[int] = (1, 2, 3, 5),
) -> dict[str, Any]:
    """Aggregate transition and position-level flip quantities across positions."""

    rows = [trajectory_flip_metrics(trajectory, horizons=horizons) for trajectory in trajectories]
    position_count = len(rows)
    transition_count = sum(int(row["eligible_transition_count"]) for row in rows)
    flip_count = sum(int(row["flip_count"]) for row in rows)
    ever_flip_count = sum(bool(row["ever_flip"]) for row in rows)
    flipback_count = sum(int(row["flipback_count"]) for row in rows)
    flipback_eligible = sum(int(row["flipback_eligible_count"]) for row in rows)
    return {
        "position_count": position_count,
        "transition_count": transition_count,
        "flip_count": flip_count,
        "transition_flip_rate": flip_count / transition_count if transition_count else None,
        "ever_flip_position_count": ever_flip_count,
        "position_ever_flip_rate": ever_flip_count / position_count if position_count else None,
        "flipback_count": flipback_count,
        "flipback_eligible_count": flipback_eligible,
        "flipback_rate": flipback_count / flipback_eligible if flipback_eligible else None,
        "position_rows": rows,
    }


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion."""

    successes, total = int(successes), int(total)
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Require 0 <= successes <= total.")
    if total == 0:
        return None, None
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    low, high = max(0.0, centre - margin), min(1.0, centre + margin)
    # Avoid exposing floating-point residue such as 2.8e-17 for an observed
    # zero, which makes JSON/CSV consumers misread a closed boundary.
    if successes == 0:
        low = 0.0
    if successes == total:
        high = 1.0
    return low, high


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of no values.")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one.")
    index = (len(sorted_values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = index - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def clustered_bootstrap_mean(
    cluster_values: Mapping[Hashable, Sequence[float]],
    *,
    iterations: int = 10_000,
    seed: int = 20260902,
    confidence: float = 0.95,
    weighting: Literal["macro", "micro"] = "macro",
) -> dict[str, Any]:
    """Cluster-resample a mean without treating within-prompt rows as IID.

    ``weighting='macro'`` (the default) averages one mean per prompt/cluster and
    is the appropriate primary estimate for this project.  ``'micro'`` weights
    a resampled cluster by its number of observations.  Values must be finite;
    empty clusters are ignored.
    """

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")
    if weighting not in {"macro", "micro"}:
        raise ValueError("weighting must be 'macro' or 'micro'.")
    cleaned: list[list[float]] = []
    for values in cluster_values.values():
        row = [float(value) for value in values]
        if not row:
            continue
        if not all(math.isfinite(value) for value in row):
            raise ValueError("cluster values must be finite.")
        cleaned.append(row)
    if not cleaned:
        return {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "cluster_count": 0,
            "observation_count": 0,
            "iterations": iterations,
            "confidence": confidence,
            "weighting": weighting,
        }

    cluster_means = [math.fsum(values) / len(values) for values in cleaned]
    cluster_sums = [math.fsum(values) for values in cleaned]
    cluster_counts = [len(values) for values in cleaned]
    if weighting == "macro":
        estimate = math.fsum(cluster_means) / len(cluster_means)
    else:
        estimate = math.fsum(cluster_sums) / math.fsum(cluster_counts)

    generator = random.Random(seed)
    cluster_count = len(cleaned)
    samples: list[float] = []
    for _ in range(iterations):
        selected = [generator.randrange(cluster_count) for _ in range(cluster_count)]
        if weighting == "macro":
            samples.append(math.fsum(cluster_means[index] for index in selected) / cluster_count)
        else:
            denominator = math.fsum(cluster_counts[index] for index in selected)
            samples.append(math.fsum(cluster_sums[index] for index in selected) / denominator)
    samples.sort()
    alpha = (1 - confidence) / 2
    return {
        "estimate": estimate,
        "ci95_low": _percentile(samples, alpha),
        "ci95_high": _percentile(samples, 1 - alpha),
        "cluster_count": cluster_count,
        "observation_count": sum(cluster_counts),
        "iterations": iterations,
        "confidence": confidence,
        "weighting": weighting,
    }
