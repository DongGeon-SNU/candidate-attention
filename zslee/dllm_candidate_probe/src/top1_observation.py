"""Exact, scalar-only observation helpers for the top-1 dynamics audit.

The natural collector in :mod:`src.top1_trajectory` deliberately mirrors the
existing pilot.  This module is the complementary *measurement* path: it
replays each already-chosen state through an independent frozen forward with
``use_cache=False``.  It never changes a natural trajectory or uses future
tokens while decoding.  Full vocabulary tensors are retained only long enough
to calculate the requested scalar statistics and are then discarded.

The module is intentionally independent of the command-line runner so its
alignment and censoring rules can be unit-tested with a tiny dummy model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping

from src.branching import exact_forward
from src.dynamics_metrics import (
    flipback_event_indices,
    future_flip_label,
    horizon_top1_retention,
    top1_runs,
)
from src.top1_trajectory import (
    DEFAULT_EPSILON,
    EventualTokenObservation,
    LogitPostprocessor,
    NaturalTrajectory,
    PositionTransition,
    PositionDistributionSummary,
    TrajectoryState,
    _apply_logit_postprocessor,
    _state_transition_records,
    summarize_masked_distributions,
    verify_trajectory_alignment,
)


@dataclass
class ExactObservationResult:
    """Compact result of no-cache measurement of one natural trajectory."""

    exact_states: list[TrajectoryState]
    state_position_rows: list[dict[str, Any]]
    transitions: list[PositionTransition]
    eventual_observations: list[dict[str, Any]]
    forward_count: int
    natural_exact_top1_mismatch_count: int
    max_probability_sum_error: float
    max_margin_logprob_error: float
    completed: bool

    def manifest_record(self) -> dict[str, Any]:
        return {
            "exact_observation_forward_count": self.forward_count,
            "exact_observation_use_cache": False,
            "natural_exact_top1_mismatch_count": self.natural_exact_top1_mismatch_count,
            "max_probability_sum_error": self.max_probability_sum_error,
            "max_margin_logprob_error": self.max_margin_logprob_error,
            "trajectory_completed": self.completed,
            "eventual_token_censoring": (
                "Positions still masked after the collected trajectory are retained as censored "
                "observations and excluded from eventual-token rank denominators."
            ),
        }


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised on target GPU
        raise RuntimeError("Exact top-1 observation requires PyTorch.") from error
    return torch


def exact_logits(model: Any, input_ids: Any) -> Any:
    """Run the shared no-cache forward primitive with fixed-length inputs."""

    torch = _torch()
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    return exact_forward(
        model,
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )


def _rank_of_targets(probabilities: Any, target_ids: Any) -> Any:
    """Best shared 1-based rank, documented for exact ties.

    The production backbone almost never produces exact BF16 ties.  A shared
    rank is preferable to silently allowing a device-specific ``topk`` tie
    order to alter reported coverage.
    """

    target_probabilities = probabilities.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return probabilities.gt(target_probabilities.unsqueeze(1)).sum(dim=1).add(1)


def _summary_margin_error(summary: PositionDistributionSummary) -> float:
    """Check the softmax identity z1-z2 == log p1-log p2 in FP32."""

    if len(summary.top_token_ids) < 2:
        return 0.0
    return abs(summary.logit_margin - summary.log_probability_ratio)


def _state_rows(
    state: TrajectoryState,
    *,
    natural_state: TrajectoryState,
    trajectory: NaturalTrajectory,
) -> list[dict[str, Any]]:
    """Flatten one exact state to portable scalar rows without vocab tensors."""

    anchors = [anchor.public_record() for anchor in state.actual_committed_anchors]
    anchor_positions = [anchor["position"] for anchor in anchors]
    rows: list[dict[str, Any]] = []
    for position in state.mask_positions:
        summary = state.position_summaries[position]
        natural = natural_state.position_summaries[position]
        row = summary.public_record()
        row.update(
            {
                "prompt_index": state.prompt_index,
                "step": state.step,
                "position": position,
                "remaining_mask_count": state.remaining_mask_count,
                "remaining_mask_count_after": state.remaining_mask_count_after,
                "generated_mask_ratio": state.generated_mask_ratio,
                "sequence_mask_ratio": state.sequence_mask_ratio,
                "low_parallel": state.low_parallel,
                "anchor_count": len(anchors),
                "anchor_positions": anchor_positions,
                "anchors": anchors,
                "threshold": trajectory.threshold,
                "exact_use_cache": False,
                "natural_top1_token_id": natural.top1_token_id,
                "natural_top1_probability": natural.top1_probability,
                "natural_exact_top1_matches": natural.top1_token_id == summary.top1_token_id,
                "top5_mass_ge_0_8": summary.top5_mass >= 0.8,
            }
        )
        rows.append(row)
    return rows


def _eventual_rows_for_state(
    trajectory: NaturalTrajectory,
    state: TrajectoryState,
    distributions: Mapping[int, Any],
    commits: Mapping[int, tuple[int, int]],
) -> list[dict[str, Any]]:
    """Reduce one state's eventual-token fields while its logits are live.

    ``next_step_top1_*`` is filled after all exact state summaries exist.  This
    ordering is deliberate: it avoids retaining one full-vocabulary matrix for
    every state merely to calculate a scalar future-token table.
    """

    torch = _torch()
    rows: list[dict[str, Any]] = []
    current_anchor_positions = {anchor.position for anchor in state.actual_committed_anchors}
    for position in state.mask_positions:
        summary = state.position_summaries[position]
        common = {
            "prompt_index": trajectory.prompt_index,
            "step": state.step,
            "position": position,
            "current_top1_token_id": summary.top1_token_id,
            "current_top1_probability": summary.top1_probability,
            "current_top1_matches_eventual": None,
            "eventual_known": position in commits,
            "censored": position not in commits,
            "committed_this_step": position in current_anchor_positions,
            "top5_mass_ge_0_8": summary.top5_mass >= 0.8,
            "next_step_top1_token_id": None,
            "next_step_top1_in_current_top1": None,
            "next_step_top1_in_current_top2": None,
            "next_step_top1_in_current_top3": None,
            "next_step_top1_in_current_top5": None,
            "next_step_top1_in_current_top10": None,
        }
        if position not in commits:
            rows.append(common)
            continue
        eventual_token, commit_step = commits[position]
        probability = distributions[position]
        target = torch.as_tensor([eventual_token], device=probability.device, dtype=torch.long)
        target_probability = probability[eventual_token]
        target_rank = _rank_of_targets(probability.unsqueeze(0), target)[0]
        top_ids = summary.top_token_ids
        row = {
            **common,
            "eventual_token_id": eventual_token,
            "eventual_commit_step": commit_step,
            "steps_until_commit": commit_step - state.step,
            "eventual_token_rank": int(target_rank.detach().cpu().item()),
            "eventual_token_probability": float(target_probability.detach().cpu().item()),
            "eventual_in_top1": eventual_token in top_ids[:1],
            "eventual_in_top2": eventual_token in top_ids[:2],
            "eventual_in_top3": eventual_token in top_ids[:3],
            "eventual_in_top5": eventual_token in top_ids[:5],
            "eventual_in_top10": eventual_token in top_ids[:10],
            "current_top1_matches_eventual": summary.top1_token_id == eventual_token,
        }
        rows.append(row)
    return rows


def _fill_eventual_future_fields(rows: list[dict[str, Any]], exact_states: Iterable[TrajectoryState]) -> None:
    """Attach next-state top-1 coverage after scalar rows have been reduced."""

    summaries = {
        (state.step, position): summary
        for state in exact_states
        for position, summary in state.position_summaries.items()
    }
    first_top5: dict[int, int] = {}
    for row in rows:
        if row.get("eventual_in_top5"):
            first_top5.setdefault(int(row["position"]), int(row["step"]))
    for row in rows:
        position = int(row["position"])
        next_summary = summaries.get((int(row["step"]) + 1, position))
        if next_summary is not None:
            next_top1 = next_summary.top1_token_id
            tokens = tuple(int(token) for token in row.get("top_token_ids", ()))
            # Eventual rows are generated from a state summary, not the
            # flattened state record, so obtain its tokens from the same map.
            current_summary = summaries[(int(row["step"]), position)]
            tokens = current_summary.top_token_ids
            row["next_step_top1_token_id"] = next_top1
            for k in (1, 2, 3, 5, 10):
                row[f"next_step_top1_in_current_top{k}"] = next_top1 in tokens[:k]
        row["eventual_first_top5_step"] = first_top5.get(position) if row["eventual_known"] else None


def observe_trajectory_exactly(
    model: Any,
    trajectory: NaturalTrajectory,
    *,
    logit_postprocessor: LogitPostprocessor | None = None,
    analysis_block_size: int | None = None,
) -> ExactObservationResult:
    """Measure all saved natural states using independent ``use_cache=False`` forwards.

    ``trajectory`` is treated as immutable evidence: anchor choices and next
    state token sequences come exclusively from the natural run.  The exact
    replay only produces diagnostics, including cache-vs-no-cache agreement.
    """

    torch = _torch()
    alignment = verify_trajectory_alignment(trajectory)
    if not alignment["valid"]:
        raise RuntimeError("Natural trajectory failed alignment check: " + "; ".join(alignment["errors"]))

    exact_states: list[TrajectoryState] = []
    state_position_rows: list[dict[str, Any]] = []
    transitions: list[PositionTransition] = []
    eventual_rows: list[dict[str, Any]] = []
    commits = trajectory.eventual_commits()
    previous_state: TrajectoryState | None = None
    previous_probabilities: dict[int, Any] | None = None
    forward_count = 0
    mismatch_count = 0
    max_probability_sum_error = 0.0
    max_margin_logprob_error = 0.0

    for natural_state in trajectory.states:
        logits = exact_logits(model, natural_state.input_ids)
        forward_count += 1
        positions = torch.as_tensor(natural_state.mask_positions, device=logits.device, dtype=torch.long)
        processed_logits = _apply_logit_postprocessor(logits[0, positions], logit_postprocessor)
        probabilities = torch.softmax(processed_logits.float(), dim=-1)
        summaries = summarize_masked_distributions(
            processed_logits, probabilities, positions, top_k=max(10, trajectory.metadata.get("top_k_recorded", 10))
        )
        exact_state = replace(natural_state, position_summaries=summaries)
        exact_states.append(exact_state)
        state_position_rows.extend(_state_rows(exact_state, natural_state=natural_state, trajectory=trajectory))
        mismatch_count += sum(
            int(summaries[position].top1_token_id != natural_state.position_summaries[position].top1_token_id)
            for position in natural_state.mask_positions
        )
        if probabilities.numel():
            max_probability_sum_error = max(
                max_probability_sum_error,
                float((probabilities.sum(dim=-1) - 1.0).abs().max().detach().cpu().item()),
            )
        max_margin_logprob_error = max(
            max_margin_logprob_error,
            max((_summary_margin_error(summary) for summary in summaries.values()), default=0.0),
        )

        row_by_position = {position: row for row, position in enumerate(natural_state.mask_positions)}
        if previous_state is not None and previous_probabilities is not None:
            transitions.extend(
                _state_transition_records(
                    previous_state,
                    previous_probabilities,
                    exact_state,
                    probabilities,
                    row_by_position,
                    analysis_block_size=analysis_block_size,
                )
            )
        # Keep at most one vocabulary matrix alive for adjacent exact state
        # comparison.  The dictionary is discarded/replaced at the next state.
        previous_state = exact_state
        previous_probabilities = {position: probabilities[row] for position, row in row_by_position.items()}
        eventual_rows.extend(_eventual_rows_for_state(trajectory, exact_state, previous_probabilities, commits))

    _fill_eventual_future_fields(eventual_rows, exact_states)
    return ExactObservationResult(
        exact_states=exact_states,
        state_position_rows=state_position_rows,
        transitions=transitions,
        eventual_observations=eventual_rows,
        forward_count=forward_count,
        natural_exact_top1_mismatch_count=mismatch_count,
        max_probability_sum_error=max_probability_sum_error,
        max_margin_logprob_error=max_margin_logprob_error,
        completed=trajectory.completed,
    )


def position_trajectory_rows(state_position_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Derive position-level flip/run/horizon rows from exact scalar records.

    The observation window terminates at commit.  This is not treated as a
    stable negative future label: every horizon helper excludes a window that
    cannot be fully observed.
    """

    grouped: dict[tuple[Any, int], list[Mapping[str, Any]]] = {}
    for row in state_position_rows:
        key = (row.get("prompt_id", row.get("prompt_index")), int(row["position"]))
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for (prompt_key, position), members in grouped.items():
        ordered = sorted(members, key=lambda row: int(row["step"]))
        tokens = [int(row["top1_token_id"]) for row in ordered]
        one_step = sum(left != right for left, right in zip(tokens, tokens[1:]))
        runs = top1_runs(tokens)
        flipbacks = flipback_event_indices(tokens)
        eligible = max(0, len(tokens) - 1)
        source = ordered[0]
        row = {
            "prompt_id": prompt_key,
            "prompt_index": source.get("prompt_index"),
            "dataset": source.get("dataset"),
            # This is a prompt-level GSM8K label, not an eventual-token
            # label.  Retaining it lets the runner relate flip burden to
            # correctness without ever conflating the two concepts.
            "gsm8k_exact_match": source.get("gsm8k_exact_match"),
            "position": position,
            "observed_state_count": len(tokens),
            "eligible_transition_count": eligible,
            "flip_count": one_step,
            "ever_flip": one_step > 0,
            "flipback_count": len(flipbacks),
            "flipback_rate": len(flipbacks) / one_step if one_step else None,
            "mean_top1_survival_steps": (
                sum(run["survival_steps"] for run in runs) / len(runs) if runs else None
            ),
            "first_step": int(ordered[0]["step"]),
            "last_observed_step": int(ordered[-1]["step"]),
            "top5_mass_ge_0_8": bool(source.get("top5_mass_ge_0_8", False)),
        }
        for horizon in (1, 2, 3, 5):
            retention = horizon_top1_retention(tokens, horizon)
            row[f"top1_retention_{horizon}_step"] = retention["retention_rate"]
            row[f"top1_retention_{horizon}_eligible"] = retention["eligible_count"]
        result.append(row)
    return result


def prediction_label_rows(state_position_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Attach only fully observed future-flip labels to scalar state records."""

    grouped: dict[tuple[Any, int], list[Mapping[str, Any]]] = {}
    for row in state_position_rows:
        key = (row.get("prompt_id", row.get("prompt_index")), int(row["position"]))
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for _, members in grouped.items():
        ordered = sorted(members, key=lambda row: int(row["step"]))
        tokens = [int(row["top1_token_id"]) for row in ordered]
        any_future = [any(token != tokens[index] for token in tokens[index + 1 :]) for index in range(len(tokens))]
        for index, source in enumerate(ordered):
            row = dict(source)
            row["next_step_top1_flip"] = future_flip_label(tokens, index, 1)
            for horizon in (2, 3, 5):
                row[f"future_{horizon}_step_top1_flip"] = future_flip_label(tokens, index, horizon)
            # Any-flip-before-commit is known for every recorded state because
            # the suffix consists solely of pre-commit observations.  The final
            # state has a legitimate false label, not an unobserved future.
            row["any_flip_before_commit"] = any_future[index]
            result.append(row)
    return result


def repeated_exact_forward_check(
    model: Any,
    input_ids: Any,
    mask_positions: Iterable[int],
    *,
    logit_postprocessor: LogitPostprocessor | None = None,
) -> dict[str, Any]:
    """Run the same state twice and return reproducibility diagnostics."""

    torch = _torch()
    positions = list(int(position) for position in mask_positions)
    first = exact_logits(model, input_ids)[0, positions]
    second = exact_logits(model, input_ids)[0, positions]
    first = _apply_logit_postprocessor(first, logit_postprocessor).float()
    second = _apply_logit_postprocessor(second, logit_postprocessor).float()
    first_top1 = first.argmax(dim=-1)
    second_top1 = second.argmax(dim=-1)
    return {
        "checked_positions": len(positions),
        "top1_exact_match": bool(torch.equal(first_top1, second_top1)),
        "max_abs_logit_difference": float((first - second).abs().max().detach().cpu().item()) if positions else 0.0,
        "use_cache": False,
    }
