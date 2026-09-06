"""Exact top-1 anchor attribution and sequential-order replay helpers.

This module deliberately knows nothing about a model object, decoding cache, or
dataset.  Callers supply ``forward(input_ids) -> logits``; production callers
must make that callable an independently evaluated ``use_cache=False`` forward
(normally by wrapping :func:`src.branching.exact_forward`).  Keeping the
boundary this small makes every branch observable in CPU tests and prevents a
KV cache from being accidentally threaded through a counterfactual branch.

The main entry point, :func:`exact_anchor_attribution`, evaluates a base state,
each singleton anchor, the complete actual anchor set, and leave-one-out
branches.  It reports the log-odds quantity used by the audit,

``g = log p(new_token) - log p(old_token)``,

as well as a full-vocabulary log-space additive prediction for the joint
distribution.  The order helpers replay the same concrete anchors either with
their parallel assignments fixed or with a fresh top-1 assignment at each
position.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .branching import Candidate, make_branch, make_leave_one_out_branches, validate_candidates


DEFAULT_EPSILON = 1.0e-12
DEFAULT_ORDER_SEED = 20260902

# ``forward`` intentionally has only one argument.  A model/cache-aware
# wrapper belongs at the runner boundary, where the correct attention and
# position tensors can be held fixed across every branch.
ExactForward = Callable[[Any], Any]
LogitPostprocessor = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Top1Decision:
    """Small scalar summary of a full-vocabulary distribution at one position."""

    position: int
    top1_token_id: int
    top1_probability: float
    top2_token_id: int | None
    top2_probability: float | None
    top1_top2_logit_margin: float | None
    entropy: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "top1_token_id": self.top1_token_id,
            "top1_probability": self.top1_probability,
            "top2_token_id": self.top2_token_id,
            "top2_probability": self.top2_probability,
            "top1_top2_logit_margin": self.top1_top2_logit_margin,
            "entropy": self.entropy,
        }


@dataclass(frozen=True)
class AnchorEffect:
    """One anchor's exact singleton and leave-one-out contributions to ``g``."""

    anchor: Candidate
    singleton_g: float
    singleton_effect: float
    leave_one_out_g: float
    leave_one_out_effect: float
    singleton_crosses: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.anchor.position,
            "token_id": self.anchor.token_id,
            "singleton_g": self.singleton_g,
            "singleton_effect": self.singleton_effect,
            "leave_one_out_g": self.leave_one_out_g,
            "leave_one_out_effect": self.leave_one_out_effect,
            "singleton_crosses": self.singleton_crosses,
        }


@dataclass(frozen=True)
class CounterfactualAttribution:
    """Scalar-only result of one exact anchor-set counterfactual audit.

    ``reconstruction_matches`` is ``None`` when no observed natural successor
    was supplied.  It is deliberately a separate diagnosis from the interaction
    classification: a mismatch takes precedence and produces ``"unattributed"``.
    """

    target_position: int
    old_token_id: int
    new_token_id: int
    anchors: tuple[Candidate, ...]
    base_g: float
    joint_g: float
    joint_total_effect: float
    singleton_effect_sum: float
    additive_predicted_g: float
    additive_residual: float
    additive_prediction_tv: float
    additive_prediction_js: float
    base_top1_token_id: int
    joint_top1_token_id: int
    singleton_sufficient: bool
    joint_crosses: bool
    additive_crosses: bool
    classification: str
    anchor_effects: tuple[AnchorEffect, ...]
    reconstruction_matches: bool | None
    reconstruction_input_ids_match: bool | None
    reconstruction_tv: float | None
    reconstruction_js: float | None
    exact_forward_count: int

    @property
    def per_anchor(self) -> tuple[AnchorEffect, ...]:
        """Compatibility-friendly alias for :attr:`anchor_effects`."""

        return self.anchor_effects

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_position": self.target_position,
            "old_token_id": self.old_token_id,
            "new_token_id": self.new_token_id,
            "anchors": [{"position": item.position, "token_id": item.token_id} for item in self.anchors],
            "anchor_count": len(self.anchors),
            "base_g": self.base_g,
            "joint_g": self.joint_g,
            "joint_total_effect": self.joint_total_effect,
            "singleton_effect_sum": self.singleton_effect_sum,
            "additive_predicted_g": self.additive_predicted_g,
            "additive_residual": self.additive_residual,
            "additive_prediction_tv": self.additive_prediction_tv,
            "additive_prediction_js": self.additive_prediction_js,
            "base_top1_token_id": self.base_top1_token_id,
            "joint_top1_token_id": self.joint_top1_token_id,
            "singleton_sufficient": self.singleton_sufficient,
            "joint_crosses": self.joint_crosses,
            "additive_crosses": self.additive_crosses,
            "classification": self.classification,
            "anchor_effects": [effect.as_dict() for effect in self.anchor_effects],
            # A second name is useful to row-oriented writers which used the
            # earlier wording in the experiment specification.
            "per_anchor": [effect.as_dict() for effect in self.anchor_effects],
            "reconstruction_matches": self.reconstruction_matches,
            "reconstruction_input_ids_match": self.reconstruction_input_ids_match,
            "reconstruction_tv": self.reconstruction_tv,
            "reconstruction_js": self.reconstruction_js,
            "exact_forward_count": self.exact_forward_count,
        }


@dataclass(frozen=True)
class ReplayStep:
    """The decision visible immediately before one sequential commit."""

    prefix_size: int
    original_anchor: Candidate
    selected_token_id: int
    top1_token_id: int
    top1_probability: float
    original_token_probability: float
    original_token_is_top1: bool
    original_token_threshold_eligible: bool | None
    top1_top2_logit_margin: float | None
    entropy: float
    tracked_top1: Mapping[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "prefix_size": self.prefix_size,
            "committed_position": self.original_anchor.position,
            "original_token_id": self.original_anchor.token_id,
            "selected_token_id": self.selected_token_id,
            "top1_token_id": self.top1_token_id,
            "top1_probability": self.top1_probability,
            "original_token_probability": self.original_token_probability,
            "original_token_is_top1": self.original_token_is_top1,
            "original_token_threshold_eligible": self.original_token_threshold_eligible,
            "top1_top2_logit_margin": self.top1_top2_logit_margin,
            "entropy": self.entropy,
            "tracked_top1": {str(position): token for position, token in self.tracked_top1.items()},
        }


@dataclass(frozen=True)
class FixedAssignmentReplay:
    """One fixed-token sequential replay for a particular anchor order."""

    order: tuple[Candidate, ...]
    steps: tuple[ReplayStep, ...]
    final_input_ids: Any
    final_state_matches_joint: bool
    exact_forward_count: int

    @property
    def final_fixed_state_equal(self) -> bool:
        """Alias used by explicit final-context sanity checks."""

        return self.final_state_matches_joint

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "fixed",
            "order": [{"position": item.position, "token_id": item.token_id} for item in self.order],
            "steps": [step.as_dict() for step in self.steps],
            "final_state_matches_joint": self.final_state_matches_joint,
            "final_fixed_state_equal": self.final_state_matches_joint,
            "exact_forward_count": self.exact_forward_count,
        }


@dataclass(frozen=True)
class AdaptiveSequentialReplay:
    """One sequential replay which chooses a fresh top-1 token at each position."""

    order: tuple[Candidate, ...]
    steps: tuple[ReplayStep, ...]
    final_input_ids: Any
    final_assignments: tuple[Candidate, ...]
    original_assignment_match_rate: float
    exact_forward_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "adaptive",
            "order": [{"position": item.position, "token_id": item.token_id} for item in self.order],
            "steps": [step.as_dict() for step in self.steps],
            "final_assignments": [
                {"position": item.position, "token_id": item.token_id} for item in self.final_assignments
            ],
            "original_assignment_match_rate": self.original_assignment_match_rate,
            "exact_forward_count": self.exact_forward_count,
        }


@dataclass(frozen=True)
class OrderReplayAudit:
    """All fixed/adaptive replays selected for one actual parallel anchor set."""

    orders: tuple[tuple[Candidate, ...], ...]
    fixed_replays: tuple[FixedAssignmentReplay, ...]
    adaptive_replays: tuple[AdaptiveSequentialReplay, ...]
    fixed_final_states_all_equal: bool
    adaptive_final_assignment_order_sensitive: bool
    adaptive_decision_trajectory_order_sensitive: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor_count": len(self.orders[0]) if self.orders else 0,
            "order_count": len(self.orders),
            "fixed_final_states_all_equal": self.fixed_final_states_all_equal,
            "adaptive_final_assignment_order_sensitive": self.adaptive_final_assignment_order_sensitive,
            "adaptive_decision_trajectory_order_sensitive": self.adaptive_decision_trajectory_order_sensitive,
            "fixed_replays": [replay.as_dict() for replay in self.fixed_replays],
            "adaptive_replays": [replay.as_dict() for replay in self.adaptive_replays],
        }


def _sequence_length(input_ids: Any) -> int:
    if hasattr(input_ids, "shape"):
        shape = tuple(int(value) for value in input_ids.shape)
        if len(shape) != 2 or shape[0] != 1:
            raise ValueError("input_ids must have shape [1, sequence].")
        return shape[1]
    if not input_ids or len(input_ids) != 1:
        raise ValueError("input_ids must be a nonempty batch-1 2-D sequence.")
    return len(input_ids[0])


def _token_at(input_ids: Any, position: int) -> int:
    return int(input_ids[0, position] if hasattr(input_ids, "shape") else input_ids[0][position])


def _clone_input_ids(input_ids: Any) -> Any:
    if hasattr(input_ids, "clone"):
        return input_ids.clone()
    return [list(row) for row in input_ids]


def _input_ids_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left, right))
    return left == right


def _normalise_anchors(base_input_ids: Any, anchors: Iterable[Candidate | tuple[int, int]]) -> tuple[Candidate, ...]:
    normalized = validate_candidates(anchors, _sequence_length(base_input_ids))
    if not normalized:
        raise ValueError("At least one actual anchor is required for a counterfactual audit.")
    # An actual parallel commit set has no intrinsic sequence order.  Canonical
    # ordering makes report rows and seeded permutation sampling reproducible.
    return tuple(sorted(normalized))


def _as_logits(value: Any) -> torch.Tensor:
    """Extract a [1, sequence, vocab] (or [sequence, vocab]) tensor."""

    logits = value.logits if hasattr(value, "logits") else value
    if not torch.is_tensor(logits):
        raise TypeError("Injected forward must return a logits tensor or an object exposing .logits.")
    if logits.ndim == 3:
        if int(logits.shape[0]) != 1:
            raise ValueError("Counterfactual forwards must return batch size 1 logits.")
    elif logits.ndim != 2:
        raise ValueError("Logits must have shape [1, sequence, vocab] or [sequence, vocab].")
    if int(logits.shape[-1]) < 1:
        raise ValueError("Logits vocabulary dimension must be nonempty.")
    return logits


def _forward_logits(forward: ExactForward, input_ids: Any) -> torch.Tensor:
    """Call the injected exact/no-cache boundary exactly once."""

    return _as_logits(forward(input_ids))


def _position_logits(logits: torch.Tensor, position: int, logit_postprocessor: LogitPostprocessor | None = None) -> torch.Tensor:
    logits = _as_logits(logits)
    position = int(position)
    sequence_length = int(logits.shape[-2])
    if position < 0 or position >= sequence_length:
        raise IndexError(f"Target position {position} is outside logits sequence length {sequence_length}.")
    row = logits[0, position] if logits.ndim == 3 else logits[position]
    row = row.detach().float()
    if logit_postprocessor is not None:
        row = logit_postprocessor(row)
        if not torch.is_tensor(row):
            raise TypeError("logit_postprocessor must return a torch tensor.")
        if row.ndim != 1:
            raise ValueError("logit_postprocessor must preserve a one-dimensional vocabulary row.")
        row = row.detach().float()
    if bool(torch.isnan(row).any().item()):
        raise ValueError("Logits contain NaN values.")
    return row


def log_probs_at_position(
    logits: torch.Tensor,
    position: int,
    *,
    logit_postprocessor: LogitPostprocessor | None = None,
) -> torch.Tensor:
    """Return a detached FP32 full-vocabulary log distribution at ``position``."""

    result = torch.log_softmax(_position_logits(logits, position, logit_postprocessor), dim=-1)
    if not bool(torch.isfinite(result).all().item()):
        # ``-inf`` input masking is allowed, but all-masked rows and +inf are
        # not valid probability distributions for the audit.
        if not bool(torch.isneginf(result).any().item()):
            raise ValueError("Logits do not produce a finite probability distribution.")
        probabilities = result.exp()
        if not bool(torch.isfinite(probabilities).all().item()) or float(probabilities.sum().item()) <= 0:
            raise ValueError("Logits do not produce a valid probability distribution.")
    return result


def probabilities_at_position(
    logits: torch.Tensor,
    position: int,
    *,
    logit_postprocessor: LogitPostprocessor | None = None,
) -> torch.Tensor:
    """Return the detached FP32 full-vocabulary softmax at ``position``."""

    return log_probs_at_position(logits, position, logit_postprocessor=logit_postprocessor).exp()


def g_score(
    logits: torch.Tensor,
    target_position: int,
    new_token_id: int,
    old_token_id: int,
    *,
    logit_postprocessor: LogitPostprocessor | None = None,
) -> float:
    """Compute ``g = log p(new_token) - log p(old_token)`` exactly in FP32."""

    log_probabilities = log_probs_at_position(logits, target_position, logit_postprocessor=logit_postprocessor)
    return _g_from_log_probabilities(log_probabilities, new_token_id, old_token_id)


def _g_from_log_probabilities(log_probabilities: torch.Tensor, new_token_id: int, old_token_id: int) -> float:
    """Calculate ``g`` from an already-computed one-dimensional log distribution."""

    vocabulary_size = int(log_probabilities.numel())
    new_token_id, old_token_id = int(new_token_id), int(old_token_id)
    if not 0 <= new_token_id < vocabulary_size or not 0 <= old_token_id < vocabulary_size:
        raise IndexError(f"Token IDs must be within vocabulary size {vocabulary_size}.")
    return float((log_probabilities[new_token_id] - log_probabilities[old_token_id]).item())


# The longer name makes call sites read naturally, while ``g_score`` is compact
# in tight audit code and follows the research notation.
log_probability_gap = g_score


def top1_decision(
    logits: torch.Tensor,
    position: int,
    *,
    logit_postprocessor: LogitPostprocessor | None = None,
) -> Top1Decision:
    """Summarize the current top-1/top-2 decision using the full distribution."""

    row = _position_logits(logits, position, logit_postprocessor)
    log_probabilities = torch.log_softmax(row, dim=-1)
    probabilities = log_probabilities.exp()
    top1 = int(torch.argmax(probabilities).item())
    top1_probability = float(probabilities[top1].item())
    if int(row.numel()) == 1:
        top2: int | None = None
        top2_probability: float | None = None
        margin: float | None = None
    else:
        # ``argmax`` gives the lowest token ID for an exact tie.  Explicitly
        # masking top1 applies the same rule to top2 without relying on
        # platform-specific ``topk`` tie ordering.
        remaining = row.clone()
        remaining[top1] = -torch.inf
        top2 = int(torch.argmax(remaining).item())
        top2_probability = float(probabilities[top2].item())
        margin = float((row[top1] - row[top2]).item())
    entropy = float((-torch.xlogy(probabilities, probabilities).sum()).item())
    return Top1Decision(
        position=int(position),
        top1_token_id=top1,
        top1_probability=top1_probability,
        top2_token_id=top2,
        top2_probability=top2_probability,
        top1_top2_logit_margin=margin,
        entropy=entropy,
    )


def _normalise_probability_vector(probabilities: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(probabilities) or probabilities.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional torch probability tensor.")
    result = probabilities.detach().float()
    if bool(torch.isnan(result).any().item()) or bool((result < 0).any().item()):
        raise ValueError(f"{name} must be finite and non-negative.")
    total = result.sum()
    if not bool(torch.isfinite(total).item()) or float(total.item()) <= 0:
        raise ValueError(f"{name} must have positive finite total mass.")
    return result / total


def distribution_tv_js(
    predicted_probabilities: torch.Tensor,
    actual_probabilities: torch.Tensor,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, float]:
    """Return full-distribution total variation and Jensen--Shannon distance.

    Inputs may carry rounding drift; both are normalized in FP32.  Only scalar
    metrics are returned so a runner need not retain vocabulary-sized tensors.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    predicted = _normalise_probability_vector(predicted_probabilities, name="predicted_probabilities")
    actual = _normalise_probability_vector(actual_probabilities, name="actual_probabilities")
    if predicted.shape != actual.shape:
        raise ValueError("predicted and actual distributions have different vocabulary sizes.")
    if predicted.device != actual.device:
        actual = actual.to(predicted.device)
    midpoint = 0.5 * (predicted + actual)
    tv = 0.5 * torch.abs(predicted - actual).sum()
    kl_predicted = torch.sum(
        predicted * (torch.log(predicted.clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon)))
    )
    kl_actual = torch.sum(actual * (torch.log(actual.clamp_min(epsilon)) - torch.log(midpoint.clamp_min(epsilon)))
    )
    return {"tv": float(tv.item()), "js": float((0.5 * (kl_predicted + kl_actual)).item())}


def additive_log_space_prediction(
    base_log_probabilities: torch.Tensor,
    singleton_log_probabilities: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Combine singleton distributions in the log-space additive approximation.

    For singleton distributions ``p_a`` and base ``p_0``, this computes

    ``p_hat(v) ∝ p_0(v) * product_a[p_a(v) / p_0(v)]``.

    Thus the predicted log-odds effect is the sum of singleton log-odds
    effects, while a final ``log_softmax`` supplies the necessary normalization.
    """

    base = base_log_probabilities.detach().float()
    if base.ndim != 1 or int(base.numel()) == 0:
        raise ValueError("base_log_probabilities must be a nonempty one-dimensional tensor.")
    singleton_rows = tuple(item.detach().float() for item in singleton_log_probabilities)
    if not singleton_rows:
        return base.exp()
    if any(item.shape != base.shape or item.ndim != 1 for item in singleton_rows):
        raise ValueError("Each singleton log distribution must match the base vocabulary shape.")
    combined = base.clone()
    for singleton in singleton_rows:
        # Identically filtered tokens have log probability -inf in both rows.
        # Treat their otherwise undefined ``-inf - -inf`` delta as zero so
        # special-token filtering can use this helper without creating NaNs.
        delta = singleton - base
        delta = torch.where(torch.isneginf(singleton) & torch.isneginf(base), torch.zeros_like(delta), delta)
        combined = combined + delta
    return torch.softmax(combined, dim=-1)


def additive_prediction_from_logits(
    base_logits: torch.Tensor,
    singleton_logits: Iterable[torch.Tensor],
    target_position: int,
    *,
    logit_postprocessor: LogitPostprocessor | None = None,
) -> torch.Tensor:
    """Convenience wrapper for :func:`additive_log_space_prediction` from logits."""

    base_log_probabilities = log_probs_at_position(
        base_logits, target_position, logit_postprocessor=logit_postprocessor
    )
    singleton_log_probabilities = (
        log_probs_at_position(item, target_position, logit_postprocessor=logit_postprocessor)
        for item in singleton_logits
    )
    return additive_log_space_prediction(base_log_probabilities, singleton_log_probabilities)


def crosses_from_negative_to_positive(base_g: float, candidate_g: float, *, tolerance: float = 0.0) -> bool:
    """Whether a counterfactual crosses the audit's strict ``g<0`` to ``g>0`` sign boundary."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    return bool(float(base_g) < -tolerance and float(candidate_g) > tolerance)


def classify_attribution(
    base_g: float,
    singleton_g_values: Iterable[float],
    joint_g: float,
    *,
    reconstruction_matches: bool | None = None,
    tolerance: float = 0.0,
) -> str:
    """Classify a flip using a mutually exclusive, explicit decision rule.

    ``unattributed`` is reserved first for a supplied natural successor that
    does not reconstruct from the recorded anchor set.  If it reconstructs,
    singletons get priority, then an additive multi-singleton crossing, then a
    genuine joint-only crossing.  A singleton/additive crossing that disappears
    in the actual joint state is ``suppression/cancellation``.
    """

    singleton_values = tuple(float(value) for value in singleton_g_values)
    singleton_crosses = any(
        crosses_from_negative_to_positive(base_g, value, tolerance=tolerance) for value in singleton_values
    )
    predicted_g = float(base_g) + sum(value - float(base_g) for value in singleton_values)
    additive_crosses = crosses_from_negative_to_positive(base_g, predicted_g, tolerance=tolerance)
    joint_crosses = crosses_from_negative_to_positive(base_g, joint_g, tolerance=tolerance)
    if reconstruction_matches is False:
        return "unattributed"
    if joint_crosses:
        if singleton_crosses:
            return "singleton-sufficient"
        if additive_crosses:
            return "multi-singleton-additive"
        return "synergy-only"
    if singleton_crosses or additive_crosses:
        return "suppression/cancellation"
    return "unattributed"


def _reconstruction_check(
    joint_probabilities: torch.Tensor,
    target_position: int,
    *,
    observed_next_input_ids: Any | None,
    expected_joint_input_ids: Any,
    observed_next_logits: Any | None,
    logit_postprocessor: LogitPostprocessor | None,
    epsilon: float,
    tolerance: float,
) -> tuple[bool | None, bool | None, float | None, float | None]:
    """Compare a reconstructed branch to an observed natural successor if supplied."""

    if tolerance < 0:
        raise ValueError("reconstruction_tolerance must be non-negative.")
    ids_match = (
        _input_ids_equal(observed_next_input_ids, expected_joint_input_ids)
        if observed_next_input_ids is not None
        else None
    )
    tv: float | None = None
    js: float | None = None
    logits_match: bool | None = None
    if observed_next_logits is not None:
        observed = probabilities_at_position(
            _as_logits(observed_next_logits), target_position, logit_postprocessor=logit_postprocessor
        )
        reconstructed = joint_probabilities
        distances = distribution_tv_js(reconstructed, observed, epsilon=epsilon)
        tv, js = distances["tv"], distances["js"]
        logits_match = tv <= tolerance
    if ids_match is None and logits_match is None:
        return None, None, tv, js
    if ids_match is None:
        return logits_match, None, tv, js
    if logits_match is None:
        return ids_match, ids_match, tv, js
    return bool(ids_match and logits_match), ids_match, tv, js


def exact_anchor_attribution(
    base_input_ids: Any,
    anchors: Iterable[Candidate | tuple[int, int]],
    *,
    mask_token_id: int,
    target_position: int,
    old_token_id: int,
    new_token_id: int,
    forward: ExactForward,
    observed_next_input_ids: Any | None = None,
    observed_next_logits: Any | None = None,
    logit_postprocessor: LogitPostprocessor | None = None,
    epsilon: float = DEFAULT_EPSILON,
    sign_tolerance: float = 0.0,
    reconstruction_tolerance: float = 1.0e-6,
) -> CounterfactualAttribution:
    """Run exact base/singleton/joint/leave-one-out anchor attribution.

    ``observed_next_*`` are optional natural-trajectory references.  When they
    are supplied and fail to match the branch made from ``anchors``, the result
    is explicitly labelled ``unattributed`` rather than mis-crediting anchors
    for a decoder state change which was not reconstructed.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    anchor_tuple = _normalise_anchors(base_input_ids, anchors)
    target_position = int(target_position)
    old_token_id, new_token_id = int(old_token_id), int(new_token_id)

    # Every input branch comes from ``make_branch`` (including the logical
    # empty branch), retaining its one-position-per-candidate invariant.
    base_branch = make_branch(base_input_ids, (), mask_token_id=mask_token_id)
    base_logits = _forward_logits(forward, base_branch)
    forward_count = 1
    base_log_probabilities = log_probs_at_position(
        base_logits, target_position, logit_postprocessor=logit_postprocessor
    )
    base_g = _g_from_log_probabilities(base_log_probabilities, new_token_id, old_token_id)
    base_top1_token_id = top1_decision(
        base_logits, target_position, logit_postprocessor=logit_postprocessor
    ).top1_token_id
    # Retain only a one-position vocabulary vector from this point onward;
    # retaining several full [sequence, vocabulary] logits tensors would make
    # a multi-anchor audit needlessly consume H100 memory.
    del base_logits

    singleton_log_probabilities: dict[Candidate, torch.Tensor] = {}
    singleton_g: dict[Candidate, float] = {}
    for anchor in anchor_tuple:
        branch = make_branch(base_input_ids, (anchor,), mask_token_id=mask_token_id)
        logits = _forward_logits(forward, branch)
        forward_count += 1
        singleton_log_probabilities[anchor] = log_probs_at_position(
            logits, target_position, logit_postprocessor=logit_postprocessor
        )
        singleton_g[anchor] = _g_from_log_probabilities(
            singleton_log_probabilities[anchor], new_token_id, old_token_id
        )
        del logits

    joint_input_ids = make_branch(base_input_ids, anchor_tuple, mask_token_id=mask_token_id)
    joint_logits = _forward_logits(forward, joint_input_ids)
    forward_count += 1
    joint_log_probabilities = log_probs_at_position(
        joint_logits, target_position, logit_postprocessor=logit_postprocessor
    )
    joint_g = _g_from_log_probabilities(joint_log_probabilities, new_token_id, old_token_id)
    joint_top1_token_id = top1_decision(
        joint_logits, target_position, logit_postprocessor=logit_postprocessor
    ).top1_token_id
    joint_probabilities = joint_log_probabilities.exp()

    if len(anchor_tuple) == 1:
        # Removing the only anchor is exactly the base branch already measured.
        leave_one_out_g: dict[Candidate, float] = {anchor_tuple[0]: base_g}
    else:
        leave_one_out_branches = make_leave_one_out_branches(
            base_input_ids, anchor_tuple, mask_token_id=mask_token_id
        )
        leave_one_out_g = {}
        for omitted in anchor_tuple:
            loo_logits = _forward_logits(forward, leave_one_out_branches[omitted])
            forward_count += 1
            loo_log_probabilities = log_probs_at_position(
                loo_logits, target_position, logit_postprocessor=logit_postprocessor
            )
            leave_one_out_g[omitted] = _g_from_log_probabilities(
                loo_log_probabilities, new_token_id, old_token_id
            )
            del loo_logits
    predicted_distribution = additive_log_space_prediction(
        base_log_probabilities, (singleton_log_probabilities[anchor] for anchor in anchor_tuple)
    )
    predicted_log_probabilities = torch.log(predicted_distribution.clamp_min(epsilon))
    additive_predicted_g = _g_from_log_probabilities(predicted_log_probabilities, new_token_id, old_token_id)
    additive_distances = distribution_tv_js(predicted_distribution, joint_probabilities, epsilon=epsilon)

    effects: list[AnchorEffect] = []
    for anchor in anchor_tuple:
        loo_g = leave_one_out_g[anchor]
        singleton_effect = singleton_g[anchor] - base_g
        effects.append(
            AnchorEffect(
                anchor=anchor,
                singleton_g=singleton_g[anchor],
                singleton_effect=singleton_effect,
                leave_one_out_g=loo_g,
                leave_one_out_effect=joint_g - loo_g,
                singleton_crosses=crosses_from_negative_to_positive(
                    base_g, singleton_g[anchor], tolerance=sign_tolerance
                ),
            )
        )
    singleton_effect_sum = sum(effect.singleton_effect for effect in effects)
    reconstruction_matches, reconstruction_ids_match, reconstruction_tv, reconstruction_js = _reconstruction_check(
        joint_probabilities,
        target_position,
        observed_next_input_ids=observed_next_input_ids,
        expected_joint_input_ids=joint_input_ids,
        observed_next_logits=observed_next_logits,
        logit_postprocessor=logit_postprocessor,
        epsilon=epsilon,
        tolerance=reconstruction_tolerance,
    )
    del joint_logits
    return CounterfactualAttribution(
        target_position=target_position,
        old_token_id=old_token_id,
        new_token_id=new_token_id,
        anchors=anchor_tuple,
        base_g=base_g,
        joint_g=joint_g,
        joint_total_effect=joint_g - base_g,
        singleton_effect_sum=singleton_effect_sum,
        additive_predicted_g=additive_predicted_g,
        additive_residual=(joint_g - base_g) - singleton_effect_sum,
        additive_prediction_tv=additive_distances["tv"],
        additive_prediction_js=additive_distances["js"],
        base_top1_token_id=base_top1_token_id,
        joint_top1_token_id=joint_top1_token_id,
        singleton_sufficient=any(effect.singleton_crosses for effect in effects),
        joint_crosses=crosses_from_negative_to_positive(base_g, joint_g, tolerance=sign_tolerance),
        additive_crosses=crosses_from_negative_to_positive(
            base_g, additive_predicted_g, tolerance=sign_tolerance
        ),
        classification=classify_attribution(
            base_g,
            (effect.singleton_g for effect in effects),
            joint_g,
            reconstruction_matches=reconstruction_matches,
            tolerance=sign_tolerance,
        ),
        anchor_effects=tuple(effects),
        reconstruction_matches=reconstruction_matches,
        reconstruction_input_ids_match=reconstruction_ids_match,
        reconstruction_tv=reconstruction_tv,
        reconstruction_js=reconstruction_js,
        exact_forward_count=forward_count,
    )


# More obvious name for runner call sites; keep both spellings to make the
# module easy to discover without duplicating the implementation.
audit_anchor_attribution = exact_anchor_attribution


def _unrank_permutation(items: Sequence[Candidate], rank: int) -> tuple[Candidate, ...]:
    """Convert a zero-based factoradic rank to one permutation without enumeration."""

    pool = list(items)
    result: list[Candidate] = []
    for width in range(len(pool), 0, -1):
        unit = factorial(width - 1)
        index, rank = divmod(rank, unit)
        result.append(pool.pop(index))
    return tuple(result)


def deterministic_anchor_orders(
    anchors: Iterable[Candidate | tuple[int, int]],
    *,
    max_permutations: int = 32,
    seed: int = DEFAULT_ORDER_SEED,
) -> tuple[tuple[Candidate, ...], ...]:
    """Return all orders for 2--4 anchors and seeded unique samples thereafter.

    Empty and singleton sets are also well-defined, which is useful in unit
    tests and callers that construct cohorts before filtering to multi-anchor
    states.  Anchor positions must be unique even though no input sequence is
    needed here.
    """

    normalized = tuple(
        candidate if isinstance(candidate, Candidate) else Candidate(*candidate) for candidate in anchors
    )
    if len({candidate.position for candidate in normalized}) != len(normalized):
        raise ValueError("An anchor order may contain at most one candidate per position.")
    ordered = tuple(sorted(normalized))
    max_permutations = int(max_permutations)
    if max_permutations < 1:
        raise ValueError("max_permutations must be positive.")
    if len(ordered) <= 4:
        return tuple(tuple(item) for item in permutations(ordered))
    total = factorial(len(ordered))
    count = min(max_permutations, total)
    rng = random.Random(int(seed))
    # Sampling integer permutation ranks avoids replacement/retry bias and
    # avoids materialising n! orders for anchor sets larger than four.
    ranks = rng.sample(range(total), count)
    return tuple(_unrank_permutation(ordered, rank) for rank in ranks)


# Shorter name for config-driven runners.
enumerate_replay_orders = deterministic_anchor_orders


def _resolve_order(
    anchors: tuple[Candidate, ...], order: Sequence[Candidate | tuple[int, int] | int] | None
) -> tuple[Candidate, ...]:
    if order is None:
        return anchors
    by_position = {anchor.position: anchor for anchor in anchors}
    resolved: list[Candidate] = []
    for item in order:
        if isinstance(item, int):
            try:
                candidate = by_position[item]
            except KeyError as error:
                raise ValueError(f"Order contains non-anchor position {item}.") from error
        else:
            candidate = item if isinstance(item, Candidate) else Candidate(*item)
            if candidate not in anchors:
                raise ValueError("Order must contain exactly the actual anchor assignments.")
        resolved.append(candidate)
    if len(resolved) != len(anchors) or len(set(resolved)) != len(anchors):
        raise ValueError("Order must contain every actual anchor exactly once.")
    return tuple(resolved)


def _tracked_positions(base_input_ids: Any, mask_token_id: int, positions: Iterable[int] | None) -> tuple[int, ...]:
    sequence_length = _sequence_length(base_input_ids)
    if positions is None:
        return tuple(
            position for position in range(sequence_length) if _token_at(base_input_ids, position) == int(mask_token_id)
        )
    result = tuple(int(position) for position in positions)
    if len(set(result)) != len(result):
        raise ValueError("tracked_positions must not repeat a position.")
    if any(position < 0 or position >= sequence_length for position in result):
        raise IndexError("tracked_positions contains a position outside the sequence.")
    return result


def _tracked_top1(
    logits: torch.Tensor,
    current_input_ids: Any,
    positions: Sequence[int],
    *,
    mask_token_id: int,
    logit_postprocessor: LogitPostprocessor | None,
) -> dict[int, int]:
    # Only still-masked positions are "remaining" decisions.  This also keeps
    # already committed fixed anchors from being misreported as changing targets.
    return {
        position: top1_decision(logits, position, logit_postprocessor=logit_postprocessor).top1_token_id
        for position in positions
        if _token_at(current_input_ids, position) == int(mask_token_id)
    }


def _replay_step(
    *,
    prefix_size: int,
    original_anchor: Candidate,
    selected_token_id: int,
    logits: torch.Tensor,
    current_input_ids: Any,
    threshold: float | None,
    tracked_positions: Sequence[int],
    mask_token_id: int,
    logit_postprocessor: LogitPostprocessor | None,
) -> ReplayStep:
    decision = top1_decision(logits, original_anchor.position, logit_postprocessor=logit_postprocessor)
    probabilities = probabilities_at_position(
        logits, original_anchor.position, logit_postprocessor=logit_postprocessor
    )
    original_probability = float(probabilities[original_anchor.token_id].item())
    return ReplayStep(
        prefix_size=prefix_size,
        original_anchor=original_anchor,
        selected_token_id=int(selected_token_id),
        top1_token_id=decision.top1_token_id,
        top1_probability=decision.top1_probability,
        original_token_probability=original_probability,
        original_token_is_top1=decision.top1_token_id == original_anchor.token_id,
        original_token_threshold_eligible=(original_probability >= threshold if threshold is not None else None),
        top1_top2_logit_margin=decision.top1_top2_logit_margin,
        entropy=decision.entropy,
        tracked_top1=_tracked_top1(
            logits,
            current_input_ids,
            tracked_positions,
            mask_token_id=mask_token_id,
            logit_postprocessor=logit_postprocessor,
        ),
    )


def fixed_assignment_replay(
    base_input_ids: Any,
    anchors: Iterable[Candidate | tuple[int, int]],
    *,
    mask_token_id: int,
    forward: ExactForward,
    order: Sequence[Candidate | tuple[int, int] | int] | None = None,
    threshold: float | None = None,
    tracked_positions: Iterable[int] | None = None,
    logit_postprocessor: LogitPostprocessor | None = None,
) -> FixedAssignmentReplay:
    """Replay fixed parallel assignments one position at a time in ``order``.

    The final context is compared to a direct one-shot ``make_branch`` result;
    equality is a sanity check only, never an order-sensitivity outcome.
    """

    anchor_tuple = _normalise_anchors(base_input_ids, anchors)
    order_tuple = _resolve_order(anchor_tuple, order)
    tracked = _tracked_positions(base_input_ids, mask_token_id, tracked_positions)
    current = make_branch(base_input_ids, (), mask_token_id=mask_token_id)
    steps: list[ReplayStep] = []
    forward_count = 0
    for prefix_size, anchor in enumerate(order_tuple):
        logits = _forward_logits(forward, current)
        forward_count += 1
        steps.append(
            _replay_step(
                prefix_size=prefix_size,
                original_anchor=anchor,
                selected_token_id=anchor.token_id,
                logits=logits,
                current_input_ids=current,
                threshold=threshold,
                tracked_positions=tracked,
                mask_token_id=mask_token_id,
                logit_postprocessor=logit_postprocessor,
            )
        )
        current = make_branch(current, (anchor,), mask_token_id=mask_token_id)
    joint = make_branch(base_input_ids, anchor_tuple, mask_token_id=mask_token_id)
    return FixedAssignmentReplay(
        order=order_tuple,
        steps=tuple(steps),
        final_input_ids=current,
        final_state_matches_joint=_input_ids_equal(current, joint),
        exact_forward_count=forward_count,
    )


def adaptive_sequential_replay(
    base_input_ids: Any,
    anchors: Iterable[Candidate | tuple[int, int]],
    *,
    mask_token_id: int,
    forward: ExactForward,
    order: Sequence[Candidate | tuple[int, int] | int] | None = None,
    threshold: float | None = None,
    tracked_positions: Iterable[int] | None = None,
    logit_postprocessor: LogitPostprocessor | None = None,
) -> AdaptiveSequentialReplay:
    """Replay positions in ``order`` but commit their current exact top-1 token."""

    anchor_tuple = _normalise_anchors(base_input_ids, anchors)
    order_tuple = _resolve_order(anchor_tuple, order)
    tracked = _tracked_positions(base_input_ids, mask_token_id, tracked_positions)
    current = make_branch(base_input_ids, (), mask_token_id=mask_token_id)
    steps: list[ReplayStep] = []
    assignments: list[Candidate] = []
    forward_count = 0
    for prefix_size, original_anchor in enumerate(order_tuple):
        logits = _forward_logits(forward, current)
        forward_count += 1
        selected_token_id = top1_decision(
            logits, original_anchor.position, logit_postprocessor=logit_postprocessor
        ).top1_token_id
        steps.append(
            _replay_step(
                prefix_size=prefix_size,
                original_anchor=original_anchor,
                selected_token_id=selected_token_id,
                logits=logits,
                current_input_ids=current,
                threshold=threshold,
                tracked_positions=tracked,
                mask_token_id=mask_token_id,
                logit_postprocessor=logit_postprocessor,
            )
        )
        selected = Candidate(original_anchor.position, selected_token_id)
        assignments.append(selected)
        current = make_branch(current, (selected,), mask_token_id=mask_token_id)
    match_count = sum(
        selected.token_id == next(anchor.token_id for anchor in anchor_tuple if anchor.position == selected.position)
        for selected in assignments
    )
    return AdaptiveSequentialReplay(
        order=order_tuple,
        steps=tuple(steps),
        final_input_ids=current,
        final_assignments=tuple(sorted(assignments)),
        original_assignment_match_rate=match_count / len(assignments),
        exact_forward_count=forward_count,
    )


def _assignment_signature(assignments: Sequence[Candidate]) -> tuple[tuple[int, int], ...]:
    return tuple((item.position, item.token_id) for item in sorted(assignments))


def _decision_trajectory_signature(replay: AdaptiveSequentialReplay) -> tuple[tuple[tuple[int, int], ...], ...]:
    # Do not include sequence order itself, or the anchors that are naturally
    # absent after they have been committed.  Otherwise two permutations would
    # look different merely because one prefix has removed A while another has
    # removed B.  The signature therefore compares only non-anchor positions
    # that are meaningful in every replay prefix.
    anchor_positions = {anchor.position for anchor in replay.order}
    return tuple(
        tuple(
            sorted(
                (int(position), int(token))
                for position, token in step.tracked_top1.items()
                if int(position) not in anchor_positions
            )
        )
        for step in replay.steps
    )


def replay_anchor_orders(
    base_input_ids: Any,
    anchors: Iterable[Candidate | tuple[int, int]],
    *,
    mask_token_id: int,
    forward: ExactForward,
    threshold: float | None = None,
    tracked_positions: Iterable[int] | None = None,
    max_permutations: int = 32,
    seed: int = DEFAULT_ORDER_SEED,
    logit_postprocessor: LogitPostprocessor | None = None,
) -> OrderReplayAudit:
    """Run deterministic fixed and adaptive replay cohorts for one anchor set."""

    anchor_tuple = _normalise_anchors(base_input_ids, anchors)
    orders = deterministic_anchor_orders(anchor_tuple, max_permutations=max_permutations, seed=seed)
    fixed = tuple(
        fixed_assignment_replay(
            base_input_ids,
            anchor_tuple,
            mask_token_id=mask_token_id,
            forward=forward,
            order=order,
            threshold=threshold,
            tracked_positions=tracked_positions,
            logit_postprocessor=logit_postprocessor,
        )
        for order in orders
    )
    adaptive = tuple(
        adaptive_sequential_replay(
            base_input_ids,
            anchor_tuple,
            mask_token_id=mask_token_id,
            forward=forward,
            order=order,
            threshold=threshold,
            tracked_positions=tracked_positions,
            logit_postprocessor=logit_postprocessor,
        )
        for order in orders
    )
    fixed_final_states_all_equal = bool(fixed) and all(replay.final_state_matches_joint for replay in fixed)
    final_assignment_signatures = {_assignment_signature(replay.final_assignments) for replay in adaptive}
    decision_trajectory_signatures = {_decision_trajectory_signature(replay) for replay in adaptive}
    return OrderReplayAudit(
        orders=orders,
        fixed_replays=fixed,
        adaptive_replays=adaptive,
        fixed_final_states_all_equal=fixed_final_states_all_equal,
        adaptive_final_assignment_order_sensitive=len(final_assignment_signatures) > 1,
        adaptive_decision_trajectory_order_sensitive=len(decision_trajectory_signatures) > 1,
    )


# Natural runner spelling; it mirrors the terminology in the experiment plan.
order_replay_audit = replay_anchor_orders


__all__ = [
    "AdaptiveSequentialReplay",
    "AnchorEffect",
    "CounterfactualAttribution",
    "DEFAULT_EPSILON",
    "DEFAULT_ORDER_SEED",
    "ExactForward",
    "FixedAssignmentReplay",
    "LogitPostprocessor",
    "OrderReplayAudit",
    "ReplayStep",
    "Top1Decision",
    "adaptive_sequential_replay",
    "additive_log_space_prediction",
    "additive_prediction_from_logits",
    "audit_anchor_attribution",
    "classify_attribution",
    "crosses_from_negative_to_positive",
    "deterministic_anchor_orders",
    "distribution_tv_js",
    "enumerate_replay_orders",
    "exact_anchor_attribution",
    "fixed_assignment_replay",
    "g_score",
    "log_probability_gap",
    "log_probs_at_position",
    "order_replay_audit",
    "probabilities_at_position",
    "replay_anchor_orders",
    "top1_decision",
]
