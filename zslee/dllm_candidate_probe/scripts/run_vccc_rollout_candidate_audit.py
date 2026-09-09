#!/usr/bin/env python3
"""Offline same-time-control candidate rollouts from VCCC polarity pairs.

This is deliberately distinct from the prior one-forward candidate-polarity
matrix.  It reuses the *same directed pairs* selected in a completed VCCC
candidate-polarity run, finds each pair's first eligible control branch point
``t*``, then rolls out one untouched control and five source-value treatments.

The normal decoder is never edited: every active branch uses the original
full-vocabulary ``p1 >= threshold`` reveal rule plus its deterministic
highest-confidence fallback.  Exact no-cache forwards are used for all policy
actions and for target measurement.  Once the target has been committed, a
separate shadow copy with only the target re-masked provides a measurement;
that shadow input never feeds into the actual branch state.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_vccc_oracle_audit import (  # noqa: E402
    ForwardAccounting,
    exact_logits_batched,
    read_jsonl,
    stable_key,
    write_csv,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PriorPair:
    """A directed primary pair selected by the preceding VCCC polarity run."""

    selection_state_key: str
    prompt_id: str
    source_position: int
    target_position: int


@dataclass(frozen=True)
class BranchPoint:
    """First eligible unchanged control state and frozen source candidates."""

    pair: PriorPair
    record: Mapping[str, Any]
    source_top5_token_ids: tuple[int, ...]
    source_top5_probabilities: tuple[float, ...]
    source_top5_mass: float
    source_p1: float


@dataclass
class Rollout:
    records: list[dict[str, Any]]
    final_target_token_id: int | None
    terminal_horizon: int | None


def _finite_top5(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 5:
        return None
    result = tuple(float(item) for item in value[:5])
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in result):
        return None
    return result


def _summary(record: Mapping[str, Any], position: int) -> Mapping[str, Any] | None:
    values = record.get("position_summaries")
    if not isinstance(values, Mapping):
        return None
    value = values.get(str(int(position)))
    return value if isinstance(value, Mapping) else None


def _pair_from_row(row: Mapping[str, Any], *, state_field: str) -> PriorPair | None:
    """Parse one directed pair while retaining malformed provenance upstream."""

    try:
        pair = PriorPair(
            selection_state_key=str(row[state_field]),
            prompt_id=str(row["prompt_id"]),
            source_position=int(row["source_position"]),
            target_position=int(row["target_position"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if pair.source_position == pair.target_position:
        raise RuntimeError(f"Invalid VCCC polarity pair with identical positions: {pair}")
    return pair


def prior_pairs(vccc_root: Path, target_count: int) -> list[PriorPair]:
    """Preserve old polarity directions, then expand deterministically to target.

    Expansion uses only primary-policy states already selected by the preceding
    VCCC audit; it changes sample size, not prompt/model/decoder provenance,
    and is independent of all rollout outcomes.
    """

    if target_count < 1:
        raise ValueError("target_count must be positive")

    path = vccc_root / "raw" / "candidate_polarity_assignments.jsonl"
    rows = read_jsonl(path)
    unique: dict[tuple[str, int, int], PriorPair] = {}
    for row in rows:
        if row.get("cohort") != "primary_policy":
            continue
        pair = _pair_from_row(row, state_field="state_key")
        if pair is None:
            continue
        unique.setdefault((pair.selection_state_key, pair.source_position, pair.target_position), pair)
    if not unique:
        raise RuntimeError(f"No primary directed candidate-polarity pairs found in {path}")
    preserved = sorted(
        unique.values(),
        key=lambda item: stable_key("rollout-preserved-polarity", item.selection_state_key, item.source_position, item.target_position),
    )
    if len(preserved) >= target_count:
        return preserved[:target_count]

    selected_path = vccc_root / "raw" / "selected_states.jsonl"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Cannot expand beyond {len(preserved)} previous polarity directions without {selected_path}"
        )
    expanded: dict[tuple[str, int, int], PriorPair] = {}
    for state in read_jsonl(selected_path):
        if state.get("cohort") != "primary_policy":
            continue
        try:
            state_key = str(state["state_key"])
            prompt_id = str(state["prompt_id"])
            positions = [int(value) for value in state["assignment_positions"]]
        except (KeyError, TypeError, ValueError):
            continue
        for source_position in positions:
            for target_position in positions:
                if source_position == target_position:
                    continue
                pair = PriorPair(state_key, prompt_id, source_position, target_position)
                expanded.setdefault((state_key, source_position, target_position), pair)
    additions = [pair for key, pair in expanded.items() if key not in unique]
    additions.sort(
        key=lambda item: stable_key("rollout-expanded-primary-policy", item.selection_state_key, item.source_position, item.target_position)
    )
    result = preserved + additions[: target_count - len(preserved)]
    if len(result) < target_count:
        raise RuntimeError(
            f"Prior VCCC run provides only {len(result)} unique primary-policy directed pairs; requested {target_count}"
        )
    return result


def _group_key(record: Mapping[str, Any]) -> tuple[str, str, float]:
    return (str(record.get("prompt_id")), str(record.get("setting")), float(record.get("threshold", math.nan)))


def _control_provenance_error(record: Mapping[str, Any], *, expected_policy: str, threshold: float) -> str | None:
    """Verify original-policy provenance without replacing it with an exact replay.

    The top-1 source artifact intentionally contains exact no-cache summaries,
    whereas normal reveal decisions were recorded from the original collector.
    Recomputing those decisions here would be a new control policy.  Instead,
    verify the collector declaration and the stored natural anchor metadata.
    """

    metadata = record.get("decoder_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("natural_policy") != expected_policy:
        return "missing_or_unexpected_control_policy_provenance"
    anchors = record.get("actual_committed_anchors")
    if not isinstance(anchors, list):
        return "missing_actual_committed_anchors"
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            return "invalid_anchor_record"
        try:
            confidence = float(anchor["confidence"])
        except (KeyError, TypeError, ValueError):
            return "missing_anchor_confidence"
        threshold_eligible = bool(anchor.get("threshold_eligible"))
        fallback = bool(anchor.get("selected_by_fallback"))
        if threshold_eligible and confidence < threshold:
            return "threshold_anchor_below_configured_threshold"
        if not threshold_eligible and not fallback:
            return "non_threshold_anchor_without_fallback"
    return None


def first_branchpoint(
    pair: PriorPair,
    records: Sequence[Mapping[str, Any]],
    *,
    strict_top5_mass_gate: float,
) -> tuple[BranchPoint | None, str]:
    """Find the first state satisfying the requested strict t* gate."""

    saw_both_masked = False
    for record in sorted(records, key=lambda item: int(item.get("step", -1))):
        masks = record.get("mask_positions")
        if not isinstance(masks, Sequence) or isinstance(masks, (str, bytes)):
            continue
        masked = {int(value) for value in masks}
        if pair.source_position not in masked or pair.target_position not in masked:
            continue
        saw_both_masked = True
        source = _summary(record, pair.source_position)
        if source is None:
            continue
        try:
            mass = float(source["top5_mass"])
            p1 = float(source["top1_probability"])
        except (KeyError, TypeError, ValueError):
            continue
        token_ids = source.get("top_token_ids")
        probabilities = _finite_top5(source.get("top_probabilities"))
        if (
            not isinstance(token_ids, Sequence)
            or isinstance(token_ids, (str, bytes))
            or len(token_ids) < 5
            or probabilities is None
            or not math.isfinite(mass)
            or not math.isfinite(p1)
        ):
            continue
        # State summaries are produced from full-vocabulary probabilities. A
        # mismatch here signals a corrupted/incompatible raw bundle rather than
        # permission to silently substitute a candidate list.
        if abs(sum(probabilities) - mass) > 2.0e-5:
            continue
        if mass <= strict_top5_mass_gate:
            continue
        return (
            BranchPoint(
                pair=pair,
                record=record,
                source_top5_token_ids=tuple(int(token) for token in token_ids[:5]),
                source_top5_probabilities=probabilities,
                source_top5_mass=mass,
                source_p1=p1,
            ),
            "retained",
        )
    return None, "no_step_with_both_positions_masked" if not saw_both_masked else "top5_mass_gate_not_met_before_source_commit"


def target_summary(logits: Any, position: int) -> dict[str, Any]:
    """Full-vocabulary top-1/top-2 statistics; no top-k renormalization."""

    import torch

    row = logits[int(position)].float()
    probabilities = torch.softmax(row, dim=-1)
    values, tokens = torch.topk(probabilities, k=2)
    top1_token = int(tokens[0].item())
    top2_token = int(tokens[1].item())
    top1_logit = float(row[top1_token].item())
    top2_logit = float(row[top2_token].item())
    return {
        "top1_token_id": top1_token,
        "top2_token_id": top2_token,
        "top1_probability": float(values[0].item()),
        "top2_probability": float(values[1].item()),
        "logit_margin": top1_logit - top2_logit,
        "probability_margin": float(values[0].item() - values[1].item()),
    }


def top5_summary(logits: Any, position: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Exact full-vocabulary top-5 values used to validate frozen t* input."""

    import torch

    probabilities = torch.softmax(logits[int(position)].float(), dim=-1)
    values, tokens = torch.topk(probabilities, k=5)
    return (
        tuple(int(token) for token in tokens.detach().cpu().tolist()),
        tuple(float(value) for value in values.detach().cpu().tolist()),
    )


def refresh_frozen_source_top5(
    model: Any,
    branchpoint: BranchPoint,
    *,
    mask_token_id: int,
    strict_top5_mass_gate: float,
    accounting: ForwardAccounting,
) -> tuple[BranchPoint | None, dict[str, Any]]:
    """Freeze source values from the exact replay that will define treatments.

    A source artifact contains the archived exact summary used to locate t*.
    The experiment itself must not silently mix that archived list with a
    separately loaded model's branch forwards.  We therefore replay the same
    fixed t* input once, use *that* full-vocabulary top-5 as the frozen branch
    candidates, and retain both versions in provenance.  The unchanged natural
    control trajectory still supplies the state sequence and branch location.
    """

    import torch

    source, target = branchpoint.pair.source_position, branchpoint.pair.target_position
    x = torch.tensor([branchpoint.record["token_sequence"]], device=next(model.parameters()).device, dtype=torch.long)
    if int(x[0, source].item()) != int(mask_token_id) or int(x[0, target].item()) != int(mask_token_id):
        raise RuntimeError(f"{branchpoint.record['state_key']}: t* source/target mask invariant failed during replay")
    logits = _forward(model, x, accounting)
    token_ids, probabilities = top5_summary(logits[0], source)
    mass = float(sum(probabilities))
    diagnostics = {
        "archived_source_top5_token_ids": list(branchpoint.source_top5_token_ids),
        "archived_source_top5_probabilities": list(branchpoint.source_top5_probabilities),
        "replayed_source_top5_token_ids": list(token_ids),
        "replayed_source_top5_probabilities": list(probabilities),
        "replayed_source_top5_mass": mass,
        "archived_replayed_top5_token_ids_match": token_ids == branchpoint.source_top5_token_ids,
        "archived_replayed_top5_probabilities_max_abs_difference": max(
            abs(left - right) for left, right in zip(probabilities, branchpoint.source_top5_probabilities, strict=True)
        ),
    }
    if mass <= strict_top5_mass_gate:
        diagnostics["status"] = "replayed_top5_mass_gate_not_met"
        return None, diagnostics
    diagnostics["status"] = "retained_after_exact_replay"
    return replace(
        branchpoint,
        source_top5_token_ids=token_ids,
        source_top5_probabilities=probabilities,
        source_top5_mass=mass,
        source_p1=float(probabilities[0]),
    ), diagnostics


def _normal_policy_step(
    input_ids: Any,
    logits: Any,
    *,
    mask_token_id: int,
    threshold: float,
) -> tuple[Any, list[int], list[int]]:
    """Apply the frozen collector's threshold-plus-argmax-fallback action."""

    import torch

    masked = torch.where(input_ids[0].eq(mask_token_id))[0]
    if not int(masked.numel()):
        return input_ids, [], []
    mask_logits = logits[0, masked]
    probabilities = torch.softmax(mask_logits.float(), dim=-1)
    confidence, token_ids = probabilities.max(dim=-1)
    transfer = confidence.ge(float(threshold))
    transfer[int(torch.argmax(confidence).item())] = True
    selected_rows = torch.where(transfer)[0]
    next_input = input_ids.clone()
    positions = [int(masked[row].item()) for row in selected_rows]
    values = [int(token_ids[row].item()) for row in selected_rows]
    for position, value in zip(positions, values, strict=True):
        next_input[0, position] = value
    return next_input, positions, values


def _forward(model: Any, input_ids: Any, accounting: ForwardAccounting) -> Any:
    logits = exact_logits_batched(model, input_ids)
    accounting.batch_calls += 1
    accounting.branch_forwards += 1
    accounting.polarity_branch_forwards += 1
    return logits


def _common(branchpoint: BranchPoint) -> dict[str, Any]:
    record = branchpoint.record
    pair = branchpoint.pair
    return {
        "selection_state_key": pair.selection_state_key,
        "branchpoint_state_key": record["state_key"],
        "dataset": record.get("dataset"),
        "prompt_id": record.get("prompt_id"),
        "prompt_index": record.get("prompt_index"),
        "tstar_step": record.get("step"),
        "source_position": pair.source_position,
        "target_position": pair.target_position,
        "source_top5_token_ids": list(branchpoint.source_top5_token_ids),
        "source_top5_probabilities": list(branchpoint.source_top5_probabilities),
        "source_top5_cumulative_mass": branchpoint.source_top5_mass,
        "source_p1": branchpoint.source_p1,
        "control_policy": "confidence_ge_threshold_plus_argmax_fallback",
        "exact_cache_policy": "use_cache=False",
    }


def rollout(
    model: Any,
    branchpoint: BranchPoint,
    *,
    mask_token_id: int,
    threshold: float,
    accounting: ForwardAccounting,
    branch_kind: str,
    source_candidate_token_id: int | None,
    source_candidate_probability: float | None,
) -> Rollout:
    """Run one actual branch through its own terminal state.

    Committed-target measurements use shadow copies.  A treatment is never
    truncated to the control length: forcing a source value can legitimately
    make its normal decoder rollout longer or shorter than control.
    """

    import torch

    pair = branchpoint.pair
    source, target = pair.source_position, pair.target_position
    x = torch.tensor([branchpoint.record["token_sequence"]], device=next(model.parameters()).device, dtype=torch.long)
    if int(x[0, source].item()) != int(mask_token_id) or int(x[0, target].item()) != int(mask_token_id):
        raise RuntimeError(f"{branchpoint.record['state_key']}: t* source/target mask invariant failed")
    if source_candidate_token_id is not None:
        x[0, source] = int(source_candidate_token_id)
    records: list[dict[str, Any]] = []
    final_target: int | None = None
    terminal_horizon: int | None = None
    common = _common(branchpoint)
    horizon = 0
    while True:
        masked = torch.where(x[0].eq(mask_token_id))[0]
        target_masked = bool(x[0, target].eq(mask_token_id).item())
        terminal = not int(masked.numel())
        actual_logits = None
        if terminal:
            if target_masked:
                raise RuntimeError("Terminal branch unexpectedly retains the target mask")
            shadow = x.clone()
            shadow[0, target] = mask_token_id
            shadow_logits = _forward(model, shadow, accounting)
            prediction = target_summary(shadow_logits[0], target)
            measurement_mode = "terminal_shadow_probe"
            action_positions: list[int] = []
            action_tokens: list[int] = []
        else:
            actual_logits = _forward(model, x, accounting)
            if branch_kind == "control" and horizon == 0:
                source_tokens, source_probabilities = top5_summary(actual_logits[0], source)
                if source_tokens != branchpoint.source_top5_token_ids:
                    raise RuntimeError(
                        f"{branchpoint.record['state_key']}: frozen source top-5 differs on the exact t* control replay"
                    )
                if any(abs(left - right) > 2.0e-5 for left, right in zip(source_probabilities, branchpoint.source_top5_probabilities, strict=True)):
                    raise RuntimeError(
                        f"{branchpoint.record['state_key']}: frozen source top-5 probabilities differ on the exact t* control replay"
                    )
            if target_masked:
                prediction = target_summary(actual_logits[0], target)
                measurement_mode = "ordinary_masked_output"
            else:
                shadow = x.clone()
                shadow[0, target] = mask_token_id
                shadow_logits = _forward(model, shadow, accounting)
                prediction = target_summary(shadow_logits[0], target)
                measurement_mode = "shadow_probe_committed_target"
            next_x, action_positions, action_tokens = _normal_policy_step(
                x, actual_logits, mask_token_id=mask_token_id, threshold=threshold
            )
            if target in action_positions:
                final_target = action_tokens[action_positions.index(target)]
            x = next_x
        if not target_masked:
            final_target = int(x[0, target].item())
        records.append(
            {
                **common,
                "branch_kind": branch_kind,
                "source_candidate_token_id": source_candidate_token_id,
                "source_candidate_probability": source_candidate_probability,
                "horizon": horizon,
                "target_actually_masked": target_masked,
                "target_measurement_mode": measurement_mode,
                "branch_terminal_before_action": terminal,
                "normal_policy_revealed_positions": action_positions,
                "normal_policy_revealed_token_ids": action_tokens,
                "target_top1_token_id": prediction["top1_token_id"],
                "target_top2_token_id": prediction["top2_token_id"],
                "target_top1_probability": prediction["top1_probability"],
                "target_top2_probability": prediction["top2_probability"],
                "target_logit_margin": prediction["logit_margin"],
                "target_probability_margin": prediction["probability_margin"],
            }
        )
        if terminal:
            terminal_horizon = horizon if terminal_horizon is None else terminal_horizon
            break
        horizon += 1
    if bool(x[0, target].eq(mask_token_id).item()):
        raise RuntimeError("Terminal branch unexpectedly retains the target mask")
    final_target = int(x[0, target].item())
    return Rollout(records=records, final_target_token_id=final_target, terminal_horizon=terminal_horizon)


def validate_control_top5(control: Rollout, branchpoint: BranchPoint) -> None:
    """Ensure t* was not accidentally reinterpreted between selection and run."""

    # ``rollout`` checks the first control forward against this exact replay's
    # frozen source list. This guard keeps the invariants explicit before
    # treatments are scheduled.
    if len(branchpoint.source_top5_token_ids) != 5 or len(branchpoint.source_top5_probabilities) != 5:
        raise RuntimeError("A retained branch point must contain exactly five frozen source candidates")
    if branchpoint.source_top5_token_ids[0] is None or not control.records:
        raise RuntimeError("Missing control/frozen top-1 provenance")


def compare_same_time(control: Rollout, treatment: Rollout) -> list[dict[str, Any]]:
    by_horizon = {int(row["horizon"]): row for row in control.records}
    output: list[dict[str, Any]] = []
    for row in treatment.records:
        control_row = by_horizon.get(int(row["horizon"]))
        if control_row is None:
            # The treatment has not ended yet, but its paired control has.
            # Keep the branch measurement for complete provenance; a
            # candidate-induced flip is undefined without a same-time control.
            output.append(
                {
                    **row,
                    "same_time_control_available": False,
                    "control_target_top1_token_id": None,
                    "control_target_top2_token_id": None,
                    "control_target_top1_probability": None,
                    "control_target_top2_probability": None,
                    "control_target_logit_margin": None,
                    "control_target_probability_margin": None,
                    "treatment_target_top1_token_id": row["target_top1_token_id"],
                    "treatment_target_top2_token_id": row["target_top2_token_id"],
                    "treatment_target_top1_probability": row["target_top1_probability"],
                    "treatment_target_top2_probability": row["target_top2_probability"],
                    "treatment_target_logit_margin": row["target_logit_margin"],
                    "treatment_target_probability_margin": row["target_probability_margin"],
                    "induced_flip": None,
                    "changed_to": None,
                }
            )
            continue
        changed = int(row["target_top1_token_id"]) != int(control_row["target_top1_token_id"])
        output.append(
            {
                **row,
                "same_time_control_available": True,
                "control_target_top1_token_id": control_row["target_top1_token_id"],
                "control_target_top2_token_id": control_row["target_top2_token_id"],
                "control_target_top1_probability": control_row["target_top1_probability"],
                "control_target_top2_probability": control_row["target_top2_probability"],
                "control_target_logit_margin": control_row["target_logit_margin"],
                "control_target_probability_margin": control_row["target_probability_margin"],
                "treatment_target_top1_token_id": row["target_top1_token_id"],
                "treatment_target_top2_token_id": row["target_top2_token_id"],
                "treatment_target_top1_probability": row["target_top1_probability"],
                "treatment_target_top2_probability": row["target_top2_probability"],
                "treatment_target_logit_margin": row["target_logit_margin"],
                "treatment_target_probability_margin": row["target_probability_margin"],
                "induced_flip": changed,
                "changed_to": row["target_top1_token_id"] if changed else None,
            }
        )
    return output


def _branch_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["branchpoint_state_key"]),
        int(row["source_position"]),
        int(row["target_position"]),
        int(row["source_candidate_token_id"]),
    )


def annotate_branch_outcomes(rows: Sequence[dict[str, Any]], final: Rollout) -> None:
    """Attach branch-level outcomes to every horizon row after rollout ends."""

    ordered = sorted(rows, key=lambda row: int(row["horizon"]))
    flips = [row for row in ordered if row.get("induced_flip") is True]
    first = int(flips[0]["horizon"]) if flips else None
    later_return = bool(
        first is not None
        and any(
            row.get("same_time_control_available") is True and row.get("induced_flip") is False
            for row in ordered
            if int(row["horizon"]) > first
        )
    )
    for row in rows:
        row["first_induced_flip_horizon"] = first
        row["returns_to_same_time_control_top1_after_first_flip"] = later_return
        row["final_actual_committed_target_token_id"] = final.final_target_token_id
        row["branch_terminal_horizon"] = final.terminal_horizon


def annotate_control_records(rows: Sequence[dict[str, Any]], final: Rollout) -> None:
    """Persist the control's final actual target token on every control horizon."""

    for row in rows:
        row["final_actual_committed_target_token_id"] = final.final_target_token_id
        row["branch_terminal_horizon"] = final.terminal_horizon


def branch_summaries(rows: Sequence[Mapping[str, Any]], finals: Mapping[tuple[str, int, int, int], Rollout]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_branch_key(row)].append(row)
    result: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        ordered = sorted(members, key=lambda row: int(row["horizon"]))
        flips = [row for row in ordered if row.get("induced_flip") is True]
        first = int(flips[0]["horizon"]) if flips else None
        later_return = bool(
            first is not None
            and any(
                row.get("same_time_control_available") is True and row.get("induced_flip") is False
                for row in ordered
                if int(row["horizon"]) > first
            )
        )
        final = finals[key]
        source = ordered[0]
        result.append(
            {
                "branchpoint_state_key": key[0],
                "prompt_id": source["prompt_id"],
                "source_position": key[1],
                "target_position": key[2],
                "source_candidate_token_id": key[3],
                "source_candidate_probability": source["source_candidate_probability"],
                "horizon_record_count": len(ordered),
                "same_time_comparable_horizon_count": sum(row.get("same_time_control_available") is True for row in ordered),
                "post_control_horizon_count": sum(row.get("same_time_control_available") is False for row in ordered),
                "first_induced_flip_horizon": first,
                "ever_induced_flip": bool(flips),
                "returns_to_same_time_control_top1_after_first_flip": later_return,
                "final_actual_committed_target_token_id": final.final_target_token_id,
                "branch_terminal_horizon": final.terminal_horizon,
            }
        )
    return result


def heterogeneity_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("same_time_control_available") is not True:
            continue
        grouped[(
            str(row["branchpoint_state_key"]),
            int(row["source_position"]),
            int(row["target_position"]),
            int(row["horizon"]),
        )].append(row)
    output: list[dict[str, Any]] = []
    for (state_key, source_position, target_position, horizon), members in sorted(grouped.items()):
        flip_values = {row["induced_flip"] for row in members}
        replacements = {int(row["changed_to"]) for row in members if row.get("changed_to") is not None}
        source = members[0]
        output.append(
            {
                "branchpoint_state_key": state_key,
                "prompt_id": source["prompt_id"],
                "tstar_step": source["tstar_step"],
                "horizon": horizon,
                "source_position": source_position,
                "target_position": target_position,
                "candidate_count": len(members),
                "candidate_flip_outcome_heterogeneous": len(flip_values) > 1,
                "distinct_flipped_replacement_token_count": len(replacements),
                "candidate_replacement_token_heterogeneous": len(replacements) > 1,
            }
        )
    return output


def pair_heterogeneity_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Candidate variation aggregated over all horizons for each (t*, i, j)."""

    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("same_time_control_available") is not True:
            continue
        grouped[(
            str(row["branchpoint_state_key"]),
            int(row["source_position"]),
            int(row["target_position"]),
        )].append(row)
    output: list[dict[str, Any]] = []
    for (state_key, source_position, target_position), members in sorted(grouped.items()):
        by_candidate: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in members:
            by_candidate[int(row["source_candidate_token_id"])].append(row)
        ever_flip = {candidate: any(row["induced_flip"] is True for row in candidate_rows) for candidate, candidate_rows in by_candidate.items()}
        replacements = {
            int(row["changed_to"])
            for row in members
            if row.get("changed_to") is not None
        }
        source = members[0]
        output.append(
            {
                "branchpoint_state_key": state_key,
                "prompt_id": source["prompt_id"],
                "tstar_step": source["tstar_step"],
                "source_position": source_position,
                "target_position": target_position,
                "candidate_count": len(by_candidate),
                "candidate_ever_flip_outcome_heterogeneous": len(set(ever_flip.values())) > 1,
                "ever_flipping_candidate_count": sum(ever_flip.values()),
                "distinct_replacement_token_count_across_horizons": len(replacements),
                "candidate_replacement_token_heterogeneous_across_horizons": len(replacements) > 1,
            }
        )
    return output


def rate(rows: Sequence[Mapping[str, Any]], *, active_only: bool) -> dict[str, Any]:
    chosen = [
        row
        for row in rows
        if row.get("same_time_control_available") is True
        and (not active_only or not bool(row["branch_terminal_before_action"]))
    ]
    numerator = sum(row["induced_flip"] is True for row in chosen)
    return {
        "scope": "same_time_control_active_horizons_only" if active_only else "all_genuine_same_time_control_horizons",
        "induced_flip_numerator": numerator,
        "induced_flip_denominator": len(chosen),
        "induced_flip_rate": numerator / len(chosen) if chosen else None,
    }


def make_output_root(base: Path, run_id: str | None) -> Path:
    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = base / identifier
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing rollout audit: {root}")
    for name in ("raw", "tables", "figures"):
        (root / name).mkdir(parents=True, exist_ok=False) if name == "raw" else (root / name).mkdir(parents=True, exist_ok=True)
    return root


def write_report(
    output_root: Path,
    *,
    selections: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    heterogeneity: Sequence[Mapping[str, Any]],
    pair_heterogeneity: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    retained = [row for row in selections if row.get("status") == "retained"]
    target_pair_count = int(metadata["target_directed_pair_count"])
    flip_rate_all = rate(outcomes, active_only=False)
    flip_rate_active = rate(outcomes, active_only=True)
    flip_hetero = sum(bool(row["candidate_ever_flip_outcome_heterogeneous"]) for row in pair_heterogeneity)
    token_hetero = sum(bool(row["candidate_replacement_token_heterogeneous_across_horizons"]) for row in pair_heterogeneity)
    direct = {
        "retained_directed_pairs": len(retained),
        "requested_directed_pairs": target_pair_count,
        "overall_induced_flip_rate": flip_rate_all,
        "active_rollout_induced_flip_rate": flip_rate_active,
        "tstar_source_target_tuples_with_candidate_flip_outcome_heterogeneity": flip_hetero,
        "tstar_source_target_tuple_count": len(pair_heterogeneity),
        "tstar_source_target_tuples_with_candidate_replacement_token_heterogeneity": token_hetero,
        "horizon_level_heterogeneity_tuple_count": len(heterogeneity),
    }
    lines = [
        "# VCCC offline candidate rollout audit",
        "",
        "## Direct answers",
        "",
        f"1. Retained directed source--target pairs: {len(retained)}/{target_pair_count} target (from {metadata['candidate_pool_pair_count']} deterministic candidate directions).",
        f"2. Overall genuine same-time-control induced flips: {flip_rate_all['induced_flip_numerator']}/{flip_rate_all['induced_flip_denominator']} ({(100 * flip_rate_all['induced_flip_rate']) if flip_rate_all['induced_flip_rate'] is not None else 0.0:.2f}%).",
        f"3. Same-time induced flips on pre-terminal-action horizons: {flip_rate_active['induced_flip_numerator']}/{flip_rate_active['induced_flip_denominator']} ({(100 * flip_rate_active['induced_flip_rate']) if flip_rate_active['induced_flip_rate'] is not None else 0.0:.2f}%).",
        f"4. Candidate ever-flip heterogeneity: {flip_hetero}/{len(pair_heterogeneity)} (t*, i, j) tuples.",
        f"5. Candidate replacement-token heterogeneity across horizons: {token_hetero}/{len(pair_heterogeneity)} (t*, i, j) tuples.",
        "",
        "## Protocol integrity",
        "",
        "For each retained directed pair, t* is the first saved original-control state with both positions masked and source M5 strictly greater than 0.9. The five source candidates (including current top-1) and their probabilities are frozen from that state. The control and all treatments use the original p1>=0.9 threshold-plus-argmax-fallback decoder action at every active step. Treatment source insertion occurs only at t*.",
        "",
        "When the target is committed, target predictions come from a shadow copy with only the target re-masked. The shadow pass is measurement-only and never changes the real rollout. Every treatment runs through its own terminal state, so its final actual target token is always recorded. `induced_flip` is defined only when the unchanged control has an identical horizon; treatment rows after control has ended retain their measurement but set `same_time_control_available=false` and `induced_flip=null`.",
        "",
        "No terminal carry-forward rows are synthesized. The all-horizons rate includes each genuine terminal-state measurement at most once; the pre-terminal-action rate excludes those terminal measurements.",
        "",
        "## Artifacts",
        "",
        "- `tables/branchpoint_pairs.csv`: retained/skipped pairs, t*, M5, p1, frozen top-5 tokens and probabilities.",
        "- `tables/target_horizon_outcomes.csv`: treatment predictions at every genuine branch horizon, with same-time control fields and `induced_flip` only where a control horizon exists.",
        "- `tables/branch_summaries.csv`: first comparable flip, later comparable return to control top-1, final actual target token, and comparable/post-control horizon counts per candidate branch.",
        "- `tables/candidate_heterogeneity.csv`: horizon-level candidate-conditioned variation.",
        "- `tables/candidate_pair_heterogeneity.csv`: the requested (t*, i, j)-level variation aggregated across horizons.",
        "",
        "## Runtime",
        "",
        f"- Exact forwards: {metadata['forward_accounting']['branch_forwards']}; model batch calls: {metadata['forward_accounting']['batch_calls']}",
        f"- Runtime seconds: {metadata['runtime_seconds']}; peak VRAM MiB: {metadata['peak_vram_mib']}",
    ]
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return direct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "vccc_rollout_candidate_audit.yaml")
    parser.add_argument("--probe-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-run", type=Path, required=True, help="Completed top1_dynamics_audit source run.")
    parser.add_argument("--vccc-run", type=Path, required=True, help="Completed preceding VCCC run whose polarity pairs are reused exactly.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run only the first retained directed pair after deterministic prior-pair ordering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wall_started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    probe_root = args.probe_root.resolve()
    source_root = args.source_run.resolve()
    vccc_root = args.vccc_run.resolve()
    source_manifest_path = source_root / "run_manifest.json"
    vccc_metadata_path = vccc_root / "run_metadata.json"
    if not source_manifest_path.exists() or not (source_root / "raw" / "trajectories.jsonl").exists():
        raise FileNotFoundError("Source run must provide run_manifest.json and raw/trajectories.jsonl")
    if not vccc_metadata_path.exists():
        raise FileNotFoundError(f"VCCC run lacks run_metadata.json: {vccc_root}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    vccc_metadata = json.loads(vccc_metadata_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") not in {"completed", "observational_completed_exact_audit_blocked_resource_estimate"}:
        raise RuntimeError(f"Unusable source top-1 run status: {source_manifest.get('status')!r}")
    if vccc_metadata.get("status") not in {"completed", "completed_smoke"}:
        raise RuntimeError(f"Unusable VCCC run status: {vccc_metadata.get('status')!r}")
    expected_fast = str(config["model"]["fast_dllm_commit"])
    if source_manifest.get("fast_dllm_requested_commit") is not None and str(source_manifest["fast_dllm_requested_commit"]) != expected_fast:
        raise RuntimeError("Source run Fast-dLLM commit does not match this experiment config")
    prior_source = vccc_metadata.get("source_run_root")
    if prior_source is not None and Path(str(prior_source)).resolve() != source_root:
        raise RuntimeError(f"VCCC run was derived from another source run: {prior_source}")
    output_base = Path(args.output_root) if args.output_root else vccc_root.parent
    expected_leaf = str(config["storage"]["output_root"]).replace("\\", "/").rstrip("/").split("/")[-1]
    if output_base.name != expected_leaf:
        raise RuntimeError(f"Output root must remain the previous experiment location *{expected_leaf}*, got {output_base}")
    output_root = make_output_root(output_base, args.run_id)
    shutil.copy2(args.config, output_root / "config.yaml")
    write_json(output_root / "source_manifest.json", source_manifest)
    write_json(output_root / "prior_vccc_metadata.json", vccc_metadata)
    trajectories = read_jsonl(source_root / "raw" / "trajectories.jsonl")
    by_state = {str(row["state_key"]): row for row in trajectories if "state_key" in row}
    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trajectories:
        if row.get("setting") == "primary" and float(row.get("threshold", -1.0)) == float(config["decoding"]["threshold"]):
            grouped[_group_key(row)].append(row)
    target_pair_count = 1 if args.smoke else int(config["sampling"]["target_directed_pair_count"])
    candidate_pool_pair_count = max(target_pair_count, target_pair_count * int(config["sampling"]["candidate_pool_multiplier"]))
    requested = prior_pairs(vccc_root, candidate_pool_pair_count)
    selections: list[dict[str, Any]] = []
    selection_by_pair: dict[PriorPair, dict[str, Any]] = {}
    retained: list[BranchPoint] = []
    skipped: Counter[str] = Counter()
    for pair in requested:
        selected_record = by_state.get(pair.selection_state_key)
        if selected_record is None:
            selections.append({"selection_state_key": pair.selection_state_key, "source_position": pair.source_position, "target_position": pair.target_position, "status": "selection_state_missing_from_source"})
            skipped["selection_state_missing_from_source"] += 1
            continue
        candidates = grouped.get(_group_key(selected_record), [])
        branchpoint, status = first_branchpoint(
            pair, candidates, strict_top5_mass_gate=float(config["branch_selection"]["source_top5_mass_strictly_greater_than"])
        )
        base = {
            "selection_state_key": pair.selection_state_key,
            "prompt_id": pair.prompt_id,
            "source_position": pair.source_position,
            "target_position": pair.target_position,
            "status": status,
        }
        if branchpoint is None:
            selections.append(base)
            skipped[status] += 1
            continue
        provenance_error = _control_provenance_error(
            branchpoint.record,
            expected_policy=str(config["branch_selection"]["required_control_policy"]),
            threshold=float(config["decoding"]["threshold"]),
        )
        if provenance_error is not None:
            selections.append({**base, "status": provenance_error})
            skipped[provenance_error] += 1
            continue
        retained.append(branchpoint)
        selection_row = {**base, "status": "eligible_pending_exact_replay", **_common(branchpoint)}
        selections.append(selection_row)
        selection_by_pair[pair] = selection_row
    if args.smoke:
        retained = retained[:1]
        retained_keys = {item.pair for item in retained}
        selections = [
            row
            for row in selections
            if row.get("status") != "eligible_pending_exact_replay"
            or any(
                pair.selection_state_key == row["selection_state_key"]
                and pair.source_position == row["source_position"]
                and pair.target_position == row["target_position"]
                for pair in retained_keys
            )
        ]
    if not retained:
        raise RuntimeError("No prior VCCC polarity pair passed the required t* eligibility condition.")

    from scripts.collect_states import mask_token_id
    from scripts.run_top1_dynamics_audit import load_model
    import torch

    model, tokenizer, _dtype = load_model(config, probe_root)
    mask_id = mask_token_id(model, tokenizer)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    accounting = ForwardAccounting(started=time.perf_counter())
    replay_refreshed: list[BranchPoint] = []
    for archived_branchpoint in retained:
        branchpoint, replay_diagnostics = refresh_frozen_source_top5(
            model,
            archived_branchpoint,
            mask_token_id=mask_id,
            strict_top5_mass_gate=float(config["branch_selection"]["source_top5_mass_strictly_greater_than"]),
            accounting=accounting,
        )
        selection_row = selection_by_pair[archived_branchpoint.pair]
        selection_row.update(replay_diagnostics)
        if branchpoint is None:
            selection_row["status"] = str(replay_diagnostics["status"])
            skipped[str(replay_diagnostics["status"])] += 1
            continue
        # `_common` deliberately replaces the archived source candidates with
        # the exact t* list that is frozen for every treatment branch.
        selection_row.update(_common(branchpoint))
        if len(replay_refreshed) >= target_pair_count:
            selection_row["status"] = "eligible_reserve_after_exact_replay"
            continue
        selection_row["status"] = "retained"
        replay_refreshed.append(branchpoint)
    retained = replay_refreshed
    if len(retained) < target_pair_count:
        raise RuntimeError(
            f"Only {len(retained)}/{target_pair_count} candidate directions passed the exact t* replay gate; expand the deterministic pool."
        )
    control_records: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    finals: dict[tuple[str, int, int, int], Rollout] = {}
    for pair_index, branchpoint in enumerate(retained, 1):
        control = rollout(
            model, branchpoint, mask_token_id=mask_id, threshold=float(config["decoding"]["threshold"]), accounting=accounting,
            branch_kind="control", source_candidate_token_id=None, source_candidate_probability=None,
        )
        validate_control_top5(control, branchpoint)
        annotate_control_records(control.records, control)
        control_records.extend(control.records)
        for token, probability in zip(branchpoint.source_top5_token_ids, branchpoint.source_top5_probabilities, strict=True):
            treatment = rollout(
                model, branchpoint, mask_token_id=mask_id, threshold=float(config["decoding"]["threshold"]), accounting=accounting,
                branch_kind="treatment", source_candidate_token_id=token, source_candidate_probability=probability,
            )
            compared = compare_same_time(control, treatment)
            annotate_branch_outcomes(compared, treatment)
            outcomes.extend(compared)
            finals[(str(branchpoint.record["state_key"]), int(branchpoint.pair.source_position), int(branchpoint.pair.target_position), int(token))] = treatment
        print(json.dumps({"stage": "rollout", "pair": pair_index, "total_pairs": len(retained), "state_key": branchpoint.record["state_key"], "forwards": accounting.branch_forwards}))
    torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - wall_started
    peak_vram_mib = float(torch.cuda.max_memory_allocated() / 2**20)
    summaries = branch_summaries(outcomes, finals)
    heterogeneity = heterogeneity_rows(outcomes)
    pair_heterogeneity = pair_heterogeneity_rows(outcomes)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_smoke" if args.smoke else "completed",
        "source_run_root": str(source_root),
        "prior_vccc_run_root": str(vccc_root),
        "model": config["model"]["name"],
        "exact_cache_policy": "use_cache=False",
        "normal_decoder_policy": str(config["branch_selection"]["required_control_policy"]),
        "strict_top5_mass_gate": float(config["branch_selection"]["source_top5_mass_strictly_greater_than"]),
        "source_candidate_freeze": "top-5 values/probabilities are refreshed once from the exact t* branch replay; archived values and match diagnostics are retained in branchpoint_pairs.csv",
        "pair_selection": "preserved prior primary-polarity directions plus deterministic, outcome-blind expansion from the same prior VCCC primary-policy selected states",
        "target_directed_pair_count": target_pair_count,
        "candidate_pool_pair_count": len(requested),
        "candidate_directions_considered_count": len(selections),
        "retained_directed_pair_count": len(retained),
        "skipped_pair_reasons": dict(sorted(skipped.items())),
        "same_time_control_definition": "compare treatment and unchanged control target argmax at identical horizon after t*; treatment-only post-control horizons retain measurements but induced_flip=null",
        "treatment_termination": "every forced-candidate branch is rolled out through its own terminal state; no terminal carry-forward rows are synthesized",
        "shadow_probe_definition": "if target is committed, copy current actual branch state, replace only target by mask, exact forward; never feed result into branch",
        "forward_accounting": accounting.__dict__,
        "runtime_seconds": runtime_seconds,
        "peak_vram_mib": peak_vram_mib,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_jsonl(output_root / "raw" / "branchpoint_pair_selection.jsonl", selections)
    write_jsonl(output_root / "raw" / "control_horizons.jsonl", control_records)
    write_jsonl(output_root / "raw" / "treatment_horizons.jsonl", outcomes)
    write_csv(output_root / "tables" / "branchpoint_pairs.csv", selections)
    write_csv(output_root / "tables" / "target_horizon_outcomes.csv", outcomes)
    write_csv(output_root / "tables" / "branch_summaries.csv", summaries)
    write_csv(output_root / "tables" / "candidate_heterogeneity.csv", heterogeneity)
    write_csv(output_root / "tables" / "candidate_pair_heterogeneity.csv", pair_heterogeneity)
    write_csv(output_root / "tables" / "exact_forward_counts.csv", [{
        "exact_branch_forwards": accounting.branch_forwards,
        "model_batch_calls": accounting.batch_calls,
        "runtime_seconds": runtime_seconds,
        "peak_vram_mib": peak_vram_mib,
    }])
    direct = write_report(
        output_root,
        selections=selections,
        outcomes=outcomes,
        heterogeneity=heterogeneity,
        pair_heterogeneity=pair_heterogeneity,
        metadata=metadata,
    )
    metadata["direct_answers"] = direct
    write_json(output_root / "run_metadata.json", metadata)
    write_json(output_root / "summary.json", {"status": metadata["status"], "metadata": metadata, "direct_answers": direct})
    print(json.dumps({"status": metadata["status"], "output_root": str(output_root), "forwards": accounting.branch_forwards, "runtime_seconds": runtime_seconds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
