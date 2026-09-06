"""Natural top-1 trajectory collection for frozen Fast-dLLM / LLaDA.

The original candidate probe only retained low-parallel states.  This module
uses *the same* confidence-threshold transfer rule, but records every state and
every position that is masked at that state.  In particular, an ``AnchorCommit``
is an action actually applied to the decoding sequence, rather than an
inference from a confidence threshold after the fact.

There are deliberately no vocabulary tensors in any public record.  Full
vocabulary distributions stay on the device long enough to compute scalar
statistics (including entropy, TV, JS and KL), then are released.  Eventual
token rank/probability needs knowledge of a future commit; it is therefore
provided by an explicit post-trajectory replay rather than by retaining every
vocabulary distribution in memory.

The default logit behaviour exactly mirrors ``scripts.collect_states``:
``softmax(mask_logits.float())`` with no temperature, special-token filtering,
or other transform.  A caller that has verified an actual decoder transform can
provide it explicitly through ``logit_postprocessor``; this changes both the
recorded distribution and the natural transfer decisions, and is recorded in
the trajectory metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import log
from typing import Any, Callable, Iterable, Mapping, Sequence

from scripts.collect_states import mask_token_id, tokenized_prompt


DEFAULT_TOP_K = 10
DEFAULT_EPSILON = 1.0e-12

# A postprocessor receives the [masked_position, vocabulary] logits tensor and
# must return a tensor with the same shape.  It is intentionally not called by
# default so that natural collection remains byte-for-byte policy-equivalent to
# the existing collector's selection rule.
LogitPostprocessor = Callable[[Any], Any]


@dataclass(frozen=True)
class PositionDistributionSummary:
    """Full-distribution scalar summary for one masked position at one state."""

    position: int
    top_token_ids: tuple[int, ...]
    top_probabilities: tuple[float, ...]
    top_logits: tuple[float, ...]
    probability_sum: float
    entropy: float
    normalized_entropy: float
    effective_vocabulary_size: float
    gini_impurity: float
    top3_mass: float
    top5_mass: float
    top10_mass: float

    @property
    def top1_token_id(self) -> int:
        return self.top_token_ids[0]

    @property
    def top1_probability(self) -> float:
        return self.top_probabilities[0]

    @property
    def top2_token_id(self) -> int | None:
        return self.top_token_ids[1] if len(self.top_token_ids) > 1 else None

    @property
    def top2_probability(self) -> float:
        return self.top_probabilities[1] if len(self.top_probabilities) > 1 else 0.0

    @property
    def p1_minus_p2(self) -> float:
        return self.top1_probability - self.top2_probability

    @property
    def p1_over_p2(self) -> float:
        return self.top1_probability / max(self.top2_probability, DEFAULT_EPSILON)

    @property
    def log_probability_ratio(self) -> float:
        return log(max(self.top1_probability, DEFAULT_EPSILON)) - log(max(self.top2_probability, DEFAULT_EPSILON))

    @property
    def logit_margin(self) -> float:
        return self.top_logits[0] - (self.top_logits[1] if len(self.top_logits) > 1 else 0.0)

    def top_k_contains(self, token_id: int, k: int) -> bool:
        if k < 1:
            raise ValueError("k must be positive.")
        return int(token_id) in self.top_token_ids[:k]

    def public_record(self) -> dict[str, Any]:
        """Return only scalar/top-k data suitable for JSONL or Parquet output."""

        top5_count = min(5, len(self.top_token_ids))
        return {
            "position": self.position,
            "top1_token_id": self.top1_token_id,
            "top1_probability": self.top1_probability,
            "top2_token_id": self.top2_token_id,
            "top2_probability": self.top2_probability,
            "p1_minus_p2": self.p1_minus_p2,
            "p1_over_p2": self.p1_over_p2,
            "log_probability_ratio": self.log_probability_ratio,
            "logit_margin": self.logit_margin,
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "effective_vocabulary_size": self.effective_vocabulary_size,
            "gini_impurity": self.gini_impurity,
            "top3_mass": self.top3_mass,
            "top5_mass": self.top5_mass,
            "top10_mass": self.top10_mass,
            "probability_sum": self.probability_sum,
            "top_token_ids": list(self.top_token_ids),
            "top_probabilities": list(self.top_probabilities),
            "top_logits": list(self.top_logits),
            # Compatibility aliases make these records easy to compare to the
            # original pilot state summaries.
            "top5_token_ids": list(self.top_token_ids[:top5_count]),
            "top5_probabilities": list(self.top_probabilities[:top5_count]),
            "top5_cumulative_probability": self.top5_mass,
        }


@dataclass(frozen=True)
class AnchorCommit:
    """A token that the natural confidence-threshold policy actually commits."""

    position: int
    token_id: int
    confidence: float
    selection_reason: str
    threshold_eligible: bool
    selected_by_fallback: bool
    is_highest_confidence: bool

    def public_record(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "token_id": self.token_id,
            "confidence": self.confidence,
            "selection_reason": self.selection_reason,
            "threshold_eligible": self.threshold_eligible,
            "selected_by_fallback": self.selected_by_fallback,
            "is_highest_confidence": self.is_highest_confidence,
        }


@dataclass(frozen=True)
class PositionTransition:
    """One observable (masked in both states) position transition."""

    prompt_index: int
    source_step: int
    target_step: int
    position: int
    previous_top1_token_id: int
    next_top1_token_id: int
    previous_top1_probability: float
    next_top1_probability: float
    top1_flip: bool
    full_distribution_tv: float
    jensen_shannon: float
    kl_previous_to_next: float
    kl_next_to_previous: float
    top5_overlap_count: int
    top5_jaccard: float
    top10_overlap_count: int
    top10_jaccard: float
    previous_top1_next_probability: float
    previous_top1_next_rank: int
    next_top1_previous_probability: float
    next_top1_previous_rank: int
    previous_top1_probability_change: float
    next_top1_probability_change: float
    old_to_new_probability_mass_change: float
    old_new_log_odds_change: float
    source_anchor_count: int
    source_anchor_positions: tuple[int, ...]
    source_low_parallel: bool
    source_mask_ratio: float
    nearest_anchor_distance: int | None
    same_analysis_block_as_anchor: bool | None

    def public_record(self) -> dict[str, Any]:
        return {
            "prompt_index": self.prompt_index,
            "source_step": self.source_step,
            "target_step": self.target_step,
            "position": self.position,
            "previous_top1_token_id": self.previous_top1_token_id,
            "next_top1_token_id": self.next_top1_token_id,
            "previous_top1_probability": self.previous_top1_probability,
            "next_top1_probability": self.next_top1_probability,
            "top1_flip": self.top1_flip,
            "full_distribution_tv": self.full_distribution_tv,
            "jensen_shannon": self.jensen_shannon,
            "kl_previous_to_next": self.kl_previous_to_next,
            "kl_next_to_previous": self.kl_next_to_previous,
            "top5_overlap_count": self.top5_overlap_count,
            "top5_jaccard": self.top5_jaccard,
            "top10_overlap_count": self.top10_overlap_count,
            "top10_jaccard": self.top10_jaccard,
            "previous_top1_next_probability": self.previous_top1_next_probability,
            "previous_top1_next_rank": self.previous_top1_next_rank,
            "next_top1_previous_probability": self.next_top1_previous_probability,
            "next_top1_previous_rank": self.next_top1_previous_rank,
            "previous_top1_probability_change": self.previous_top1_probability_change,
            "next_top1_probability_change": self.next_top1_probability_change,
            "old_to_new_probability_mass_change": self.old_to_new_probability_mass_change,
            "old_new_log_odds_change": self.old_new_log_odds_change,
            "source_anchor_count": self.source_anchor_count,
            "source_anchor_positions": list(self.source_anchor_positions),
            "source_low_parallel": self.source_low_parallel,
            "source_mask_ratio": self.source_mask_ratio,
            "nearest_anchor_distance": self.nearest_anchor_distance,
            "same_analysis_block_as_anchor": self.same_analysis_block_as_anchor,
        }


@dataclass
class TrajectoryState:
    """A pre-transfer natural decoding state and the action taken from it."""

    prompt_index: int
    step: int
    input_ids: Any
    mask_positions: tuple[int, ...]
    position_summaries: dict[int, PositionDistributionSummary]
    actual_committed_anchors: tuple[AnchorCommit, ...]
    remaining_mask_count: int
    remaining_mask_count_after: int
    generated_mask_ratio: float
    sequence_mask_ratio: float
    low_parallel: bool

    def public_record(self) -> dict[str, Any]:
        return {
            "prompt_index": self.prompt_index,
            "step": self.step,
            "token_sequence": self.input_ids[0].detach().cpu().tolist(),
            "mask_positions": list(self.mask_positions),
            "position_summaries": {str(position): summary.public_record() for position, summary in self.position_summaries.items()},
            "actual_committed_anchors": [anchor.public_record() for anchor in self.actual_committed_anchors],
            "anchor_count": len(self.actual_committed_anchors),
            "remaining_mask_count": self.remaining_mask_count,
            "remaining_mask_count_after": self.remaining_mask_count_after,
            "generated_mask_ratio": self.generated_mask_ratio,
            "sequence_mask_ratio": self.sequence_mask_ratio,
            "low_parallel": self.low_parallel,
        }


@dataclass
class NaturalTrajectory:
    """All natural states and observable transitions for one prompt."""

    prompt_index: int
    prompt_token_count: int
    generation_length: int
    requested_steps: int
    threshold: float
    use_cache: bool
    mask_token_id: int
    states: list[TrajectoryState]
    transitions: list[PositionTransition]
    final_input_ids: Any
    completed: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def eventual_commits(self) -> dict[int, tuple[int, int]]:
        """Map generated position to ``(actual committed token, commit step)``.

        A position cannot be committed twice under the trajectory invariant.  An
        error here is preferable to silently treating an altered decoder state
        as a normal natural trajectory.
        """

        commits: dict[int, tuple[int, int]] = {}
        for state in self.states:
            for anchor in state.actual_committed_anchors:
                if anchor.position in commits:
                    raise RuntimeError(f"Position {anchor.position} was committed more than once.")
                commits[anchor.position] = (anchor.token_id, state.step)
        return commits

    def public_state_records(self) -> list[dict[str, Any]]:
        return [state.public_record() for state in self.states]

    def public_transition_records(self) -> list[dict[str, Any]]:
        return [transition.public_record() for transition in self.transitions]

    def manifest_record(self) -> dict[str, Any]:
        return {
            "prompt_index": self.prompt_index,
            "prompt_token_count": self.prompt_token_count,
            "generation_length": self.generation_length,
            "requested_steps": self.requested_steps,
            "threshold": self.threshold,
            "use_cache": self.use_cache,
            "mask_token_id": self.mask_token_id,
            "state_count": len(self.states),
            "transition_count": len(self.transitions),
            "committed_position_count": len(self.eventual_commits()),
            "completed": self.completed,
            **self.metadata,
        }


@dataclass(frozen=True)
class EventualTokenObservation:
    """A post-hoc eventual-token statistic for one observed masked position."""

    prompt_index: int
    step: int
    position: int
    eventual_token_id: int
    eventual_commit_step: int
    committed_this_step: bool
    steps_until_commit: int
    eventual_token_rank: int
    eventual_token_probability: float
    eventual_in_top1: bool
    eventual_in_top2: bool
    eventual_in_top3: bool
    eventual_in_top5: bool
    eventual_in_top10: bool
    current_top1_matches_eventual: bool
    next_step_top1_token_id: int | None
    next_step_top1_in_current_top1: bool | None
    next_step_top1_in_current_top2: bool | None
    next_step_top1_in_current_top3: bool | None
    next_step_top1_in_current_top5: bool | None
    next_step_top1_in_current_top10: bool | None
    eventual_first_top5_step: int | None = None

    def public_record(self) -> dict[str, Any]:
        return {
            "prompt_index": self.prompt_index,
            "step": self.step,
            "position": self.position,
            "eventual_token_id": self.eventual_token_id,
            "eventual_commit_step": self.eventual_commit_step,
            "committed_this_step": self.committed_this_step,
            "steps_until_commit": self.steps_until_commit,
            "eventual_token_rank": self.eventual_token_rank,
            "eventual_token_probability": self.eventual_token_probability,
            "eventual_in_top1": self.eventual_in_top1,
            "eventual_in_top2": self.eventual_in_top2,
            "eventual_in_top3": self.eventual_in_top3,
            "eventual_in_top5": self.eventual_in_top5,
            "eventual_in_top10": self.eventual_in_top10,
            "current_top1_matches_eventual": self.current_top1_matches_eventual,
            "next_step_top1_token_id": self.next_step_top1_token_id,
            "next_step_top1_in_current_top1": self.next_step_top1_in_current_top1,
            "next_step_top1_in_current_top2": self.next_step_top1_in_current_top2,
            "next_step_top1_in_current_top3": self.next_step_top1_in_current_top3,
            "next_step_top1_in_current_top5": self.next_step_top1_in_current_top5,
            "next_step_top1_in_current_top10": self.next_step_top1_in_current_top10,
            "eventual_first_top5_step": self.eventual_first_top5_step,
        }


@dataclass
class EventualReplayResult:
    """Compact result of replaying trajectory states for future-token metrics."""

    observations: list[EventualTokenObservation]
    forward_count: int
    use_cache: bool
    top1_mismatch_count: int
    max_probability_sum_error: float

    def public_records(self) -> list[dict[str, Any]]:
        return [observation.public_record() for observation in self.observations]

    def manifest_record(self) -> dict[str, Any]:
        return {
            "eventual_replay_forward_count": self.forward_count,
            "eventual_replay_use_cache": self.use_cache,
            "eventual_replay_top1_mismatch_count": self.top1_mismatch_count,
            "eventual_replay_max_probability_sum_error": self.max_probability_sum_error,
            "eventual_rank_tie_policy": "1 + count(probability > target_probability)",
        }


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised in GPU environment
        raise RuntimeError("Natural trajectory collection requires PyTorch.") from error
    return torch


def _model_logits(model: Any, input_ids: Any, *, use_cache: bool) -> Any:
    """Mirror the original collector's direct Fast-dLLM model invocation."""

    torch = _torch()
    model.eval()
    with torch.inference_mode():
        output = model(input_ids, use_cache=use_cache)
    if not hasattr(output, "logits"):
        raise TypeError("Model output must expose .logits.")
    return output.logits


def _apply_logit_postprocessor(mask_logits: Any, processor: LogitPostprocessor | None) -> Any:
    if processor is None:
        return mask_logits
    transformed = processor(mask_logits)
    if tuple(transformed.shape) != tuple(mask_logits.shape):
        raise ValueError("logit_postprocessor must preserve [masked_position, vocabulary] shape.")
    return transformed


def _mass(top_probabilities: Any, k: int) -> Any:
    return top_probabilities[:, : min(k, int(top_probabilities.shape[1]))].sum(dim=-1)


def summarize_masked_distributions(
    mask_logits: Any,
    mask_probabilities: Any,
    mask_positions: Sequence[int] | Any,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict[int, PositionDistributionSummary]:
    """Compute full-vocabulary scalar summaries without serializing a distribution.

    ``mask_logits`` and ``mask_probabilities`` must be aligned [N, V] tensors.
    This function vectorizes entropy/mass calculations across all N currently
    masked positions, then transfers only the scalar and top-k fields to CPU.
    """

    torch = _torch()
    if mask_logits.ndim != 2 or mask_probabilities.ndim != 2 or tuple(mask_logits.shape) != tuple(mask_probabilities.shape):
        raise ValueError("mask_logits and mask_probabilities must be equal-rank [N, V] tensors.")
    if int(mask_logits.shape[0]) != len(mask_positions):
        raise ValueError("mask_positions must align with mask_logits rows.")
    if int(mask_logits.shape[1]) < 1:
        raise ValueError("The vocabulary dimension must be nonempty.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")

    vocabulary_size = int(mask_logits.shape[-1])
    stored_k = min(int(top_k), vocabulary_size)
    probabilities = mask_probabilities.float()
    top_probabilities, top_token_ids = torch.topk(probabilities, k=stored_k, dim=-1)
    top_logits = mask_logits.float().gather(-1, top_token_ids)
    safe_probabilities = probabilities.clamp_min(DEFAULT_EPSILON)
    entropy = -(probabilities * safe_probabilities.log()).sum(dim=-1)
    normalized_entropy = entropy / log(vocabulary_size) if vocabulary_size > 1 else torch.zeros_like(entropy)
    effective_vocabulary_size = torch.exp(entropy)
    gini_impurity = 1.0 - probabilities.square().sum(dim=-1)
    probability_sum = probabilities.sum(dim=-1)

    # One bulk device-to-host conversion avoids an accidental synchronization
    # for every masked token.
    positions_cpu = [int(position) for position in (mask_positions.detach().cpu().tolist() if hasattr(mask_positions, "detach") else mask_positions)]
    ids_cpu = top_token_ids.detach().cpu().tolist()
    probs_cpu = top_probabilities.detach().cpu().tolist()
    logits_cpu = top_logits.detach().cpu().tolist()
    scalars = {
        "probability_sum": probability_sum.detach().cpu().tolist(),
        "entropy": entropy.detach().cpu().tolist(),
        "normalized_entropy": normalized_entropy.detach().cpu().tolist(),
        "effective_vocabulary_size": effective_vocabulary_size.detach().cpu().tolist(),
        "gini_impurity": gini_impurity.detach().cpu().tolist(),
        "top3_mass": _mass(top_probabilities, 3).detach().cpu().tolist(),
        "top5_mass": _mass(top_probabilities, 5).detach().cpu().tolist(),
        "top10_mass": _mass(top_probabilities, 10).detach().cpu().tolist(),
    }
    return {
        position: PositionDistributionSummary(
            position=position,
            top_token_ids=tuple(int(token) for token in ids_cpu[row]),
            top_probabilities=tuple(float(probability) for probability in probs_cpu[row]),
            top_logits=tuple(float(value) for value in logits_cpu[row]),
            probability_sum=float(scalars["probability_sum"][row]),
            entropy=float(scalars["entropy"][row]),
            normalized_entropy=float(scalars["normalized_entropy"][row]),
            effective_vocabulary_size=float(scalars["effective_vocabulary_size"][row]),
            gini_impurity=float(scalars["gini_impurity"][row]),
            top3_mass=float(scalars["top3_mass"][row]),
            top5_mass=float(scalars["top5_mass"][row]),
            top10_mass=float(scalars["top10_mass"][row]),
        )
        for row, position in enumerate(positions_cpu)
    }


def _rank_of_targets(distribution: Any, target_token_ids: Any) -> Any:
    """Return 1-based ranks; ties receive the best shared rank.

    This avoids sorting the full vocabulary.  Exact ties are not expected for
    normal BF16 logits; documenting this tie policy makes synthetic-logit tests
    and any rare tie interpretable.
    """

    target_probability = distribution.gather(1, target_token_ids.unsqueeze(1)).squeeze(1)
    return distribution.gt(target_probability.unsqueeze(1)).sum(dim=1).add(1)


def _top_overlap(left: PositionDistributionSummary, right: PositionDistributionSummary, k: int) -> tuple[int, float]:
    left_tokens = set(left.top_token_ids[:k])
    right_tokens = set(right.top_token_ids[:k])
    overlap = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    return overlap, float(overlap / union) if union else 1.0


def _state_transition_records(
    previous_state: TrajectoryState,
    previous_probabilities: Mapping[int, Any],
    current_state: TrajectoryState,
    current_probabilities: Any,
    current_row_by_position: Mapping[int, int],
    *,
    analysis_block_size: int | None,
) -> list[PositionTransition]:
    """Compute distribution shifts only for positions masked in both states."""

    torch = _torch()
    common_positions = [position for position in previous_state.mask_positions if position in current_row_by_position]
    if not common_positions:
        return []
    previous_matrix = torch.stack([previous_probabilities[position] for position in common_positions], dim=0).float()
    current_matrix = current_probabilities[
        torch.as_tensor([current_row_by_position[position] for position in common_positions], device=current_probabilities.device)
    ].float()
    midpoint = 0.5 * (previous_matrix + current_matrix)
    safe_previous = previous_matrix.clamp_min(DEFAULT_EPSILON)
    safe_current = current_matrix.clamp_min(DEFAULT_EPSILON)
    safe_midpoint = midpoint.clamp_min(DEFAULT_EPSILON)
    kl_previous_to_next = (previous_matrix * (safe_previous.log() - safe_current.log())).sum(dim=-1)
    kl_next_to_previous = (current_matrix * (safe_current.log() - safe_previous.log())).sum(dim=-1)
    jensen_shannon = 0.5 * (
        (previous_matrix * (safe_previous.log() - safe_midpoint.log())).sum(dim=-1)
        + (current_matrix * (safe_current.log() - safe_midpoint.log())).sum(dim=-1)
    )
    total_variation = 0.5 * (previous_matrix - current_matrix).abs().sum(dim=-1)

    previous_summaries = [previous_state.position_summaries[position] for position in common_positions]
    current_summaries = [current_state.position_summaries[position] for position in common_positions]
    previous_top1_ids = torch.as_tensor(
        [summary.top1_token_id for summary in previous_summaries], device=current_matrix.device, dtype=torch.long
    )
    current_top1_ids = torch.as_tensor(
        [summary.top1_token_id for summary in current_summaries], device=current_matrix.device, dtype=torch.long
    )
    previous_top1_next_probability = current_matrix.gather(1, previous_top1_ids.unsqueeze(1)).squeeze(1)
    current_top1_previous_probability = previous_matrix.gather(1, current_top1_ids.unsqueeze(1)).squeeze(1)
    previous_top1_next_rank = _rank_of_targets(current_matrix, previous_top1_ids)
    current_top1_previous_rank = _rank_of_targets(previous_matrix, current_top1_ids)
    previous_top1_previous_probability = previous_matrix.gather(1, previous_top1_ids.unsqueeze(1)).squeeze(1)
    current_top1_current_probability = current_matrix.gather(1, current_top1_ids.unsqueeze(1)).squeeze(1)
    old_change = previous_top1_next_probability - previous_top1_previous_probability
    new_change = current_top1_current_probability - current_top1_previous_probability
    old_new_log_odds_change = (
        (current_top1_current_probability.clamp_min(DEFAULT_EPSILON).log() - previous_top1_next_probability.clamp_min(DEFAULT_EPSILON).log())
        - (current_top1_previous_probability.clamp_min(DEFAULT_EPSILON).log() - previous_top1_previous_probability.clamp_min(DEFAULT_EPSILON).log())
    )

    # Scalar values are converted in bulk for the same reason as state metrics.
    scalar_rows = {
        "tv": total_variation.detach().cpu().tolist(),
        "js": jensen_shannon.detach().cpu().tolist(),
        "kl_previous_to_next": kl_previous_to_next.detach().cpu().tolist(),
        "kl_next_to_previous": kl_next_to_previous.detach().cpu().tolist(),
        "previous_top1_next_probability": previous_top1_next_probability.detach().cpu().tolist(),
        "previous_top1_next_rank": previous_top1_next_rank.detach().cpu().tolist(),
        "current_top1_previous_probability": current_top1_previous_probability.detach().cpu().tolist(),
        "current_top1_previous_rank": current_top1_previous_rank.detach().cpu().tolist(),
        "old_change": old_change.detach().cpu().tolist(),
        "new_change": new_change.detach().cpu().tolist(),
        "log_odds_change": old_new_log_odds_change.detach().cpu().tolist(),
    }
    anchor_positions = tuple(anchor.position for anchor in previous_state.actual_committed_anchors)
    records: list[PositionTransition] = []
    for row, (position, previous_summary, current_summary) in enumerate(
        zip(common_positions, previous_summaries, current_summaries, strict=True)
    ):
        top5_overlap_count, top5_jaccard = _top_overlap(previous_summary, current_summary, 5)
        top10_overlap_count, top10_jaccard = _top_overlap(previous_summary, current_summary, 10)
        nearest_distance = min((abs(position - anchor_position) for anchor_position in anchor_positions), default=None)
        same_block = (
            any(position // analysis_block_size == anchor_position // analysis_block_size for anchor_position in anchor_positions)
            if analysis_block_size is not None and analysis_block_size > 0 and anchor_positions
            else None
        )
        records.append(
            PositionTransition(
                prompt_index=current_state.prompt_index,
                source_step=previous_state.step,
                target_step=current_state.step,
                position=position,
                previous_top1_token_id=previous_summary.top1_token_id,
                next_top1_token_id=current_summary.top1_token_id,
                previous_top1_probability=previous_summary.top1_probability,
                next_top1_probability=current_summary.top1_probability,
                top1_flip=previous_summary.top1_token_id != current_summary.top1_token_id,
                full_distribution_tv=float(scalar_rows["tv"][row]),
                jensen_shannon=float(scalar_rows["js"][row]),
                kl_previous_to_next=float(scalar_rows["kl_previous_to_next"][row]),
                kl_next_to_previous=float(scalar_rows["kl_next_to_previous"][row]),
                top5_overlap_count=top5_overlap_count,
                top5_jaccard=top5_jaccard,
                top10_overlap_count=top10_overlap_count,
                top10_jaccard=top10_jaccard,
                previous_top1_next_probability=float(scalar_rows["previous_top1_next_probability"][row]),
                previous_top1_next_rank=int(scalar_rows["previous_top1_next_rank"][row]),
                next_top1_previous_probability=float(scalar_rows["current_top1_previous_probability"][row]),
                next_top1_previous_rank=int(scalar_rows["current_top1_previous_rank"][row]),
                previous_top1_probability_change=float(scalar_rows["old_change"][row]),
                next_top1_probability_change=float(scalar_rows["new_change"][row]),
                old_to_new_probability_mass_change=float(scalar_rows["new_change"][row] - scalar_rows["old_change"][row]),
                old_new_log_odds_change=float(scalar_rows["log_odds_change"][row]),
                source_anchor_count=len(anchor_positions),
                source_anchor_positions=anchor_positions,
                source_low_parallel=previous_state.low_parallel,
                source_mask_ratio=previous_state.generated_mask_ratio,
                nearest_anchor_distance=nearest_distance,
                same_analysis_block_as_anchor=same_block,
            )
        )
    return records


def _actual_anchors(
    positions: Any,
    summaries: Mapping[int, PositionDistributionSummary],
    transfer: Any,
    threshold_eligible: Any,
    fallback_was_needed: bool,
    highest_confidence_row: int,
) -> tuple[AnchorCommit, ...]:
    """Create action records aligned exactly with the tensor update below."""

    positions_list = [int(value) for value in positions.detach().cpu().tolist()]
    transfer_list = [bool(value) for value in transfer.detach().cpu().tolist()]
    eligible_list = [bool(value) for value in threshold_eligible.detach().cpu().tolist()]
    anchors: list[AnchorCommit] = []
    for row, position in enumerate(positions_list):
        if not transfer_list[row]:
            continue
        is_highest = row == highest_confidence_row
        is_fallback = fallback_was_needed and is_highest
        anchors.append(
            AnchorCommit(
                position=position,
                token_id=summaries[position].top1_token_id,
                confidence=summaries[position].top1_probability,
                selection_reason="fallback_highest_confidence" if is_fallback else "threshold",
                threshold_eligible=eligible_list[row],
                selected_by_fallback=is_fallback,
                is_highest_confidence=is_highest,
            )
        )
    return tuple(anchors)


def collect_natural_trajectory(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    prompt_index: int,
    generation_length: int,
    steps: int,
    threshold: float,
    use_cache: bool,
    low_parallel_max_transferred: int,
    low_parallel_remaining_fraction: float,
    top_k: int = DEFAULT_TOP_K,
    logit_postprocessor: LogitPostprocessor | None = None,
    logit_processing_description: str | None = None,
    analysis_block_size: int | None = None,
) -> NaturalTrajectory:
    """Run the natural Fast-dLLM confidence-threshold trajectory for one prompt.

    The transfer tensor and mutation below intentionally match
    ``scripts.collect_states.collect_states``:

    ``transfer = confidence >= threshold`` followed by transfer of the first
    ``torch.argmax(confidence)`` position.  Consequently fallback is only an
    additional commit when no position reaches threshold.  Every state is
    retained here, unlike the original pilot collector which only retains
    low-parallel states.
    """

    torch = _torch()
    if generation_length < 1:
        raise ValueError("generation_length must be at least one.")
    if steps < 1:
        raise ValueError("steps must be at least one.")
    if top_k < 10:
        raise ValueError("top_k must be at least 10 so top-10 coverage can be measured.")
    if analysis_block_size is not None and analysis_block_size < 1:
        raise ValueError("analysis_block_size must be positive when provided.")

    model.eval()
    mask_id = mask_token_id(model, tokenizer)
    device = next(model.parameters()).device
    prompt_ids = tokenized_prompt(tokenizer, prompt, device)
    prompt_token_count = int(prompt_ids.shape[1])
    x = torch.cat(
        [
            prompt_ids,
            torch.full((1, generation_length), mask_id, device=prompt_ids.device, dtype=prompt_ids.dtype),
        ],
        dim=1,
    )

    states: list[TrajectoryState] = []
    transitions: list[PositionTransition] = []
    previous_state: TrajectoryState | None = None
    previous_probabilities: dict[int, Any] | None = None
    for step in range(min(steps, generation_length)):
        mask_index = x.eq(mask_id)
        positions = torch.where(mask_index[0])[0]
        if positions.numel() == 0:
            break
        logits = _model_logits(model, x, use_cache=use_cache)
        raw_mask_logits = logits[0, positions]
        mask_logits = _apply_logit_postprocessor(raw_mask_logits, logit_postprocessor)
        probabilities = torch.softmax(mask_logits.float(), dim=-1)
        summaries = summarize_masked_distributions(mask_logits, probabilities, positions, top_k=top_k)

        # These use top-1 values obtained from precisely the same FP32 softmax
        # as the original collector.  Do not derive confidence from a top-k
        # renormalization; it would change the natural policy.
        proposed_ids = torch.as_tensor(
            [summaries[int(position)].top1_token_id for position in positions.detach().cpu().tolist()],
            device=x.device,
            dtype=x.dtype,
        )
        confidence = probabilities.max(dim=-1).values
        threshold_eligible = confidence.ge(threshold)
        transfer = threshold_eligible.clone()
        highest_confidence_row = int(torch.argmax(confidence).item())
        fallback_was_needed = not bool(threshold_eligible.any().item())
        transfer[highest_confidence_row] = True
        remaining_count = int(positions.numel())
        transfer_count = int(transfer.sum().item())
        low_parallel = (
            transfer_count <= low_parallel_max_transferred
            or transfer_count / max(remaining_count, 1) <= low_parallel_remaining_fraction
        )
        anchors = _actual_anchors(
            positions, summaries, transfer, threshold_eligible, fallback_was_needed, highest_confidence_row
        )
        state = TrajectoryState(
            prompt_index=prompt_index,
            step=step,
            input_ids=x.clone(),
            mask_positions=tuple(int(position) for position in positions.detach().cpu().tolist()),
            position_summaries=summaries,
            actual_committed_anchors=anchors,
            remaining_mask_count=remaining_count,
            remaining_mask_count_after=remaining_count - transfer_count,
            generated_mask_ratio=remaining_count / generation_length,
            sequence_mask_ratio=remaining_count / int(x.shape[1]),
            low_parallel=low_parallel,
        )
        current_row_by_position = {int(position): row for row, position in enumerate(positions.detach().cpu().tolist())}
        if previous_state is not None and previous_probabilities is not None:
            transitions.extend(
                _state_transition_records(
                    previous_state,
                    previous_probabilities,
                    state,
                    probabilities,
                    current_row_by_position,
                    analysis_block_size=analysis_block_size,
                )
            )
        states.append(state)

        # Keep only the immediately preceding state distribution.  Tensor views
        # deliberately keep one [remaining_masks, vocab] tensor alive, which is
        # enough for next-step shift metrics but cannot accumulate across rounds.
        previous_state = state
        previous_probabilities = {position: probabilities[row] for position, row in current_row_by_position.items()}
        transferred_positions = positions[transfer]
        x[0, transferred_positions] = proposed_ids[transfer]

    completed = not bool(x.eq(mask_id).any().item())
    return NaturalTrajectory(
        prompt_index=prompt_index,
        prompt_token_count=prompt_token_count,
        generation_length=generation_length,
        requested_steps=steps,
        threshold=float(threshold),
        use_cache=bool(use_cache),
        mask_token_id=mask_id,
        states=states,
        transitions=transitions,
        final_input_ids=x.clone(),
        completed=completed,
        metadata={
            "natural_policy": "confidence_ge_threshold_plus_argmax_fallback",
            "confidence_calculation": "softmax(processed_mask_logits.float())",
            "raw_logits_recorded": False,
            "processed_logits_recorded": False,
            "logit_processing": logit_processing_description
            or ("identity; matches scripts.collect_states" if logit_postprocessor is None else "caller_supplied_postprocessor"),
            "top_k_recorded": top_k,
            "distribution_statistics": "full vocabulary FP32 softmax; only scalar/top-k summaries persisted",
            "rank_tie_policy": "1 + count(probability > target_probability)",
            "analysis_block_size": analysis_block_size,
        },
    )


def verify_trajectory_alignment(trajectory: NaturalTrajectory) -> dict[str, Any]:
    """Check that every next natural state equals its predecessor plus anchors.

    The check is CPU-side and contains no forward calls.  It directly verifies
    the transition denominator invariant: a position is observed only if it is
    masked before and after the exact actions recorded at the source state.
    """

    torch = _torch()
    errors: list[str] = []
    checked_pairs = 0
    for previous, current in zip(trajectory.states, trajectory.states[1:], strict=False):
        if current.step != previous.step + 1:
            errors.append(f"state steps are not consecutive: {previous.step} -> {current.step}")
        expected = previous.input_ids.detach().clone()
        for anchor in previous.actual_committed_anchors:
            expected[0, anchor.position] = anchor.token_id
        if not torch.equal(expected, current.input_ids):
            errors.append(f"state {previous.step} does not lead to state {current.step} after recorded anchors")
        previous_masked = set(previous.mask_positions)
        current_masked = set(current.mask_positions)
        committed = {anchor.position for anchor in previous.actual_committed_anchors}
        if current_masked != previous_masked.difference(committed):
            errors.append(f"mask-set mismatch between state {previous.step} and {current.step}")
        checked_pairs += 1
    terminal_checked = False
    if trajectory.states:
        # The final natural action has no following saved pre-transfer state,
        # but it still defines eventual commits.  Verify it explicitly so a
        # malformed final update cannot contaminate eventual-token ranks.
        previous = trajectory.states[-1]
        expected_final = previous.input_ids.detach().clone()
        for anchor in previous.actual_committed_anchors:
            expected_final[0, anchor.position] = anchor.token_id
        if not torch.equal(expected_final, trajectory.final_input_ids):
            errors.append(f"final_input_ids does not match state {previous.step} plus recorded anchors")
        terminal_masked = set(
            int(position)
            for position in torch.where(trajectory.final_input_ids.eq(trajectory.mask_token_id))[1].detach().cpu().tolist()
        )
        expected_terminal_masked = set(previous.mask_positions).difference(
            anchor.position for anchor in previous.actual_committed_anchors
        )
        if terminal_masked != expected_terminal_masked:
            errors.append(f"terminal mask-set mismatch after state {previous.step}")
        terminal_checked = True
    return {
        "valid": not errors,
        "checked_state_pairs": checked_pairs,
        "checked_terminal_update": terminal_checked,
        "errors": errors,
    }


def _next_state_top1(state_by_step: Mapping[int, TrajectoryState], state: TrajectoryState, position: int) -> int | None:
    next_state = state_by_step.get(state.step + 1)
    if next_state is None:
        return None
    summary = next_state.position_summaries.get(position)
    return summary.top1_token_id if summary is not None else None


def replay_eventual_token_statistics(
    model: Any,
    trajectory: NaturalTrajectory,
    *,
    use_cache: bool = False,
    logit_postprocessor: LogitPostprocessor | None = None,
) -> EventualReplayResult:
    """Replay states to measure the eventual natural token without vocab dumps.

    Future committed-token information is introduced only after the baseline
    trajectory has finished.  It is never read by ``collect_natural_trajectory``
    and therefore cannot affect decoding.  The default ``use_cache=False`` is
    useful for a reproducible measurement; callers may choose ``True`` only
    when explicitly auditing cache-on natural-state equivalence.
    """

    torch = _torch()
    commits = trajectory.eventual_commits()
    state_by_step = {state.step: state for state in trajectory.states}
    observations: list[EventualTokenObservation] = []
    forward_count = 0
    top1_mismatch_count = 0
    max_probability_sum_error = 0.0
    for state in trajectory.states:
        rows = [position for position in state.mask_positions if position in commits]
        if not rows:
            continue
        logits = _model_logits(model, state.input_ids, use_cache=use_cache)
        forward_count += 1
        position_tensor = torch.as_tensor(rows, device=state.input_ids.device, dtype=torch.long)
        mask_logits = _apply_logit_postprocessor(logits[0, position_tensor], logit_postprocessor)
        probabilities = torch.softmax(mask_logits.float(), dim=-1)
        top_k = min(DEFAULT_TOP_K, int(probabilities.shape[-1]))
        current_top_probabilities, current_top_ids = torch.topk(probabilities, k=top_k, dim=-1)
        max_probability_sum_error = max(
            max_probability_sum_error,
            float((probabilities.sum(dim=-1) - 1.0).abs().max().detach().cpu().item()),
        )
        target_ids = torch.as_tensor([commits[position][0] for position in rows], device=probabilities.device, dtype=torch.long)
        target_probability = probabilities.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        target_rank = _rank_of_targets(probabilities, target_ids)
        current_top_ids_cpu = current_top_ids.detach().cpu().tolist()
        target_probability_cpu = target_probability.detach().cpu().tolist()
        target_rank_cpu = target_rank.detach().cpu().tolist()
        for row, position in enumerate(rows):
            original_top1 = state.position_summaries[position].top1_token_id
            replay_top1 = int(current_top_ids_cpu[row][0])
            if original_top1 != replay_top1:
                top1_mismatch_count += 1
            eventual_token, commit_step = commits[position]
            top_tokens = tuple(int(token) for token in current_top_ids_cpu[row])
            next_top1 = _next_state_top1(state_by_step, state, position)
            anchor_positions = {anchor.position for anchor in state.actual_committed_anchors}
            def next_in_top(k: int) -> bool | None:
                return None if next_top1 is None else next_top1 in top_tokens[: min(k, len(top_tokens))]
            observations.append(
                EventualTokenObservation(
                    prompt_index=trajectory.prompt_index,
                    step=state.step,
                    position=position,
                    eventual_token_id=eventual_token,
                    eventual_commit_step=commit_step,
                    committed_this_step=position in anchor_positions,
                    steps_until_commit=commit_step - state.step,
                    eventual_token_rank=int(target_rank_cpu[row]),
                    eventual_token_probability=float(target_probability_cpu[row]),
                    eventual_in_top1=eventual_token in top_tokens[:1],
                    eventual_in_top2=eventual_token in top_tokens[:2],
                    eventual_in_top3=eventual_token in top_tokens[:3],
                    eventual_in_top5=eventual_token in top_tokens[:5],
                    eventual_in_top10=eventual_token in top_tokens[:10],
                    current_top1_matches_eventual=replay_top1 == eventual_token,
                    next_step_top1_token_id=next_top1,
                    next_step_top1_in_current_top1=next_in_top(1),
                    next_step_top1_in_current_top2=next_in_top(2),
                    next_step_top1_in_current_top3=next_in_top(3),
                    next_step_top1_in_current_top5=next_in_top(5),
                    next_step_top1_in_current_top10=next_in_top(10),
                )
            )

    # The first top-5 entry is a property of a position's full earlier
    # trajectory, so populate it only after all observations have been made.
    first_top5: dict[int, int] = {}
    for observation in observations:
        if observation.eventual_in_top5:
            first_top5.setdefault(observation.position, observation.step)
    observations = [
        replace(observation, eventual_first_top5_step=first_top5.get(observation.position))
        for observation in observations
    ]
    return EventualReplayResult(
        observations=observations,
        forward_count=forward_count,
        use_cache=use_cache,
        top1_mismatch_count=top1_mismatch_count,
        max_probability_sum_error=max_probability_sum_error,
    )


def state_summary_lookup(state: TrajectoryState, position: int) -> PositionDistributionSummary:
    """Small explicit helper that raises a useful error for non-masked targets."""

    try:
        return state.position_summaries[position]
    except KeyError as error:
        raise KeyError(f"Position {position} is not masked at prompt={state.prompt_index}, step={state.step}.") from error


def transition_flip_counts(transitions: Iterable[PositionTransition]) -> dict[str, int]:
    """Return a denominator-safe micro count used by downstream aggregators."""

    materialized = list(transitions)
    return {
        "observable_masked_position_transitions": len(materialized),
        "top1_flips": sum(int(transition.top1_flip) for transition in materialized),
    }
