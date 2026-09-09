#!/usr/bin/env python3
"""Exact all-order top-1 VCCC headroom audit.

This is a read-only continuation of a completed Fast-dLLM top-1 trajectory
audit.  It asks one deliberately narrow question: with the *current exact
top-1 token values fixed*, can an exhaustive all-order certificate extend the
batch that the unchanged confidence-plus-fallback Fast-dLLM policy committed?

The runner never changes a decoder trajectory, proposes alternative token
values, trains a predictor, or claims that the exponential verification work
is a deployable speedup.  Each branch is an independent full-vocabulary,
``use_cache=False`` forward over a fixed-length state.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_vccc_oracle_audit import (  # noqa: E402
    exact_logits_batched,
    margin_summary,
    read_jsonl,
    stable_key,
)
from src.exact_top1_headroom import (  # noqa: E402
    SetCertificate,
    certificate_cache_for_gamma,
    choose_largest_safe_mask,
    confidence_selected_mask,
    first_target_failure,
    mask_indices,
)
from src.top1_reporting import prompt_clustered_bootstrap, write_csv, write_jsonl  # noqa: E402


SCHEMA_VERSION = 1
POLICY_FAST = "fast_dllm_actual"
POLICY_EXTENSION = "fast_dllm_preserving_exact_extension"
POLICY_FREE = "free_exact_vccc_upper_bound"
POLICY_CONFIDENCE = "confidence_only_threshold"


@dataclass(frozen=True)
class SelectedState:
    """Outcome-blind source state retained for fresh exact verification."""

    record: Mapping[str, Any]
    cohort: str
    selection_reason: str
    prompt_partition: str


@dataclass
class ForwardAccounting:
    """Keep verification work separate from any idealized NFE proxy."""

    base_forwards: int = 0
    subset_context_forwards: int = 0
    model_batch_calls: int = 0
    started: float = 0.0

    @property
    def exact_forwards(self) -> int:
        return int(self.base_forwards + self.subset_context_forwards)


def _json_safe(value: Any) -> Any:
    """Convert audit metadata to strict portable JSON without numpy imports."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _source_common(state: SelectedState) -> dict[str, Any]:
    record = state.record
    return {
        "state_key": str(record.get("state_key")),
        "cohort": state.cohort,
        "selection_reason": state.selection_reason,
        "prompt_partition": state.prompt_partition,
        "dataset": record.get("dataset"),
        "prompt_id": record.get("prompt_id"),
        "prompt_index": record.get("prompt_index"),
        "source_step": record.get("step"),
        "generated_mask_ratio": record.get("generated_mask_ratio"),
        "sequence_length": len(record.get("token_sequence", ())),
        "source_threshold": record.get("threshold"),
    }


def prompt_partition(prompt_id: str, config: Mapping[str, Any]) -> str:
    """Fixed prompt-disjoint calibration/test split, retained in raw outputs."""

    split = config["matching"]
    calibration_percent = int(split["calibration_percent"])
    if not 1 <= calibration_percent <= 99:
        raise ValueError("matching.calibration_percent must be between 1 and 99")
    bucket = int(stable_key(str(split["prompt_split_salt"]), str(prompt_id))[:8], 16) % 100
    return "calibration" if bucket < calibration_percent else "test"


def _basic_source_error(record: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    """Validate fields needed before an outcome-blind state can be sampled."""

    required = ("state_key", "prompt_id", "token_sequence", "mask_positions", "actual_committed_anchors")
    if any(field not in record for field in required):
        return "missing_required_source_field"
    if not isinstance(record.get("token_sequence"), Sequence) or isinstance(record.get("token_sequence"), (str, bytes)):
        return "invalid_token_sequence"
    if not isinstance(record.get("mask_positions"), Sequence) or isinstance(record.get("mask_positions"), (str, bytes)):
        return "invalid_mask_positions"
    if not isinstance(record.get("actual_committed_anchors"), list):
        return "invalid_actual_batch"
    if record.get("measurement") != "exact_no_cache":
        return "source_state_is_not_declared_exact_no_cache_measurement"
    metadata = record.get("decoder_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("natural_policy") != config["decoder_policy"]["required_name"]:
        return "missing_or_unexpected_decoder_policy_provenance"
    if metadata.get("exact_measurement_use_cache") is not False:
        return "source_state_missing_exact_no_cache_measurement_provenance"
    return None


def natural_policy_rows_by_state(path: Path) -> dict[str, dict[int, Mapping[str, Any]]]:
    """Load the original-policy scalars needed to verify B without replaying it.

    `trajectories.jsonl` deliberately holds exact no-cache position summaries,
    while the natural batch action was made from the original collector's
    distributions.  The source audit saves both in `state_positions.jsonl`;
    use those natural fields to validate the historic action rather than
    silently replacing it with this audit's fresh exact forward.
    """

    if not path.exists():
        raise FileNotFoundError(
            "Source run lacks raw/state_positions.jsonl, which is required to verify the recorded normal Fast-dLLM batch."
        )
    grouped: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        try:
            state_key = str(row["state_key"])
            position = int(row["position"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed natural-policy source row in {path}: {row!r}") from error
        if position in grouped[state_key]:
            raise ValueError(f"Duplicate source position evidence for state={state_key!r}, position={position}")
        grouped[state_key][position] = row
    return dict(grouped)


def select_states(
    trajectories: Sequence[Mapping[str, Any]], config: Mapping[str, Any], *, smoke: bool
) -> tuple[list[SelectedState], list[dict[str, Any]], dict[str, int]]:
    """Pick prompt-diverse states before any fresh exact forward is observed.

    The primary sample uses at most one deterministically hashed state per
    prompt.  This prevents a long trajectory from dominating the primary
    cohort and makes the reported prompt-clustered CI especially transparent.
    No certificate, flip, margin, or base-batch-size outcome participates in
    selection.
    """

    sampling = config["sampling"]
    threshold = float(config["decoding"]["threshold"])
    candidate_by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    screening: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in trajectories:
        if record.get("setting") != "primary" or _finite_float(record.get("threshold")) != threshold:
            continue
        counts["source_primary_state_rows"] += 1
        error = _basic_source_error(record, config)
        base = {
            "state_key": record.get("state_key"),
            "prompt_id": record.get("prompt_id"),
            "dataset": record.get("dataset"),
            "source_step": record.get("step"),
            "cohort": "primary",
            "selection_stage": "outcome_blind_source_screen",
        }
        if error is not None:
            screening.append({**base, "status": "excluded_before_selection", "exclusion_reason": error})
            counts[f"excluded_{error}"] += 1
            continue
        candidate_by_prompt[str(record["prompt_id"])].append(record)

    prompt_representatives: list[Mapping[str, Any]] = []
    for prompt_id, records in candidate_by_prompt.items():
        representative = min(records, key=lambda row: stable_key("headroom-primary-state", prompt_id, row["state_key"]))
        prompt_representatives.append(representative)
        for record in records:
            if record is not representative:
                screening.append({
                    "state_key": record["state_key"],
                    "prompt_id": record["prompt_id"],
                    "dataset": record.get("dataset"),
                    "source_step": record.get("step"),
                    "cohort": "primary",
                    "selection_stage": "outcome_blind_source_screen",
                    "status": "not_selected_other_state_for_same_prompt",
                    "exclusion_reason": None,
                })

    prompt_representatives.sort(key=lambda row: stable_key("headroom-primary-prompt", row["prompt_id"], row["state_key"]))
    primary_cap = 1 if smoke else int(sampling["primary_prompt_cap"])
    hard_cap = 0 if smoke else int(sampling.get("hard_exploratory_prompt_cap", 0))
    primary_records = prompt_representatives[:primary_cap]
    hard_records = prompt_representatives[primary_cap : primary_cap + hard_cap]
    selected: list[SelectedState] = []
    selected_ids: set[str] = set()
    for cohort, reason, records in (
        ("primary", "deterministic_one_state_per_prompt", primary_records),
        ("hard_exploratory", "deterministic_remaining_prompt_sample", hard_records),
    ):
        for record in records:
            selected_ids.add(str(record["state_key"]))
            partition = prompt_partition(str(record["prompt_id"]), config)
            selected.append(SelectedState(record, cohort, reason, partition))
            screening.append({
                "state_key": record["state_key"],
                "prompt_id": record["prompt_id"],
                "dataset": record.get("dataset"),
                "source_step": record.get("step"),
                "cohort": cohort,
                "prompt_partition": partition,
                "selection_stage": "outcome_blind_source_screen",
                "status": "selected_pending_fresh_exact_replay",
                "exclusion_reason": None,
            })
    for record in prompt_representatives[primary_cap + hard_cap :]:
        if str(record["state_key"]) not in selected_ids:
            screening.append({
                "state_key": record["state_key"],
                "prompt_id": record["prompt_id"],
                "dataset": record.get("dataset"),
                "source_step": record.get("step"),
                "cohort": "not_selected",
                "selection_stage": "outcome_blind_source_screen",
                "status": "not_selected_prompt_cap",
                "exclusion_reason": None,
            })
    counts["source_primary_prompt_count"] = len(candidate_by_prompt)
    counts["selected_primary_state_count"] = len(primary_records)
    counts["selected_hard_state_count"] = len(hard_records)
    return selected, screening, dict(counts)


def _validate_actual_batch(
    state: SelectedState,
    *,
    mask_positions: set[int],
    mask_positions_in_policy_order: Sequence[int],
    fresh_assignments: Mapping[int, Mapping[str, Any]],
    natural_position_rows: Mapping[int, Mapping[str, Any]] | None,
    config: Mapping[str, Any],
) -> tuple[list[int] | None, str | None]:
    """Validate and preserve, rather than recompute, the natural policy batch."""

    record = state.record
    if natural_position_rows is None:
        return None, "missing_original_policy_position_evidence"
    anchors = record.get("actual_committed_anchors")
    if not isinstance(anchors, list) or not anchors:
        return None, "missing_or_empty_actual_fast_dllm_batch"
    threshold = float(config["decoding"]["threshold"])
    positions: list[int] = []
    fallback_count = 0
    threshold_count = 0
    anchor_by_position: dict[int, Mapping[str, Any]] = {}
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            return None, "invalid_actual_batch_anchor"
        try:
            position = int(anchor["position"])
            token_id = int(anchor["token_id"])
            confidence = float(anchor["confidence"])
        except (KeyError, TypeError, ValueError):
            return None, "invalid_actual_batch_anchor_fields"
        if position not in mask_positions:
            return None, "actual_batch_position_not_masked"
        if position in positions:
            return None, "duplicate_actual_batch_position"
        if not math.isfinite(confidence):
            return None, "invalid_actual_batch_confidence"
        eligible = bool(anchor.get("threshold_eligible"))
        fallback = bool(anchor.get("selected_by_fallback"))
        if eligible != (confidence >= threshold):
            return None, "stored_actual_batch_threshold_metadata_mismatch"
        if fallback:
            fallback_count += 1
        if eligible:
            threshold_count += 1
        if int(fresh_assignments[position]["token_id"]) != token_id:
            return None, "stored_actual_batch_token_differs_from_fresh_exact_top1"
        natural = natural_position_rows.get(position)
        try:
            natural_token = int(natural["natural_top1_token_id"]) if natural is not None else None
            natural_confidence = float(natural["natural_top1_probability"]) if natural is not None else None
        except (KeyError, TypeError, ValueError):
            natural_token = natural_confidence = None
        if natural_token != token_id or natural_confidence is None or not math.isfinite(natural_confidence):
            return None, "actual_batch_anchor_disagrees_with_original_policy_position_evidence"
        if abs(confidence - natural_confidence) > 2.0e-6:
            return None, "actual_batch_anchor_confidence_disagrees_with_original_policy_evidence"
        positions.append(position)
        anchor_by_position[position] = anchor
    if fallback_count > 1:
        return None, "multiple_actual_batch_fallback_tokens"
    if fallback_count and (threshold_count or len(positions) != 1):
        return None, "invalid_actual_batch_fallback_semantics"
    if not fallback_count and threshold_count != len(positions):
        return None, "nonthreshold_actual_batch_token_without_fallback"
    natural_values: list[tuple[int, int | None, float]] = []
    for position in mask_positions_in_policy_order:
        if position not in mask_positions:
            return None, "invalid_original_policy_mask_position_order"
        natural = natural_position_rows.get(int(position))
        try:
            natural_values.append((
                int(position),
                int(natural["natural_top1_token_id"]) if natural is not None else None,
                float(natural["natural_top1_probability"]) if natural is not None else math.nan,
            ))
        except (KeyError, TypeError, ValueError):
            return None, "missing_original_policy_position_evidence"
    if {position for position, _token, _confidence in natural_values} != mask_positions:
        return None, "incomplete_original_policy_position_evidence"
    if any(token is None or not math.isfinite(confidence) for _position, token, confidence in natural_values):
        return None, "invalid_original_policy_position_evidence"
    threshold_positions = [position for position, _token, confidence in natural_values if confidence >= threshold]
    if threshold_positions:
        expected_positions = sorted(threshold_positions)
        if fallback_count != 0 or threshold_count != len(positions):
            return None, "actual_batch_disagrees_with_original_threshold_action"
    else:
        # `torch.argmax` returns the first maximum in the collector's masked
        # position order.  Preserve that order explicitly for exact tie replay.
        best_position = natural_values[0][0]
        best_confidence = natural_values[0][2]
        for position, _token, confidence in natural_values[1:]:
            if confidence > best_confidence:
                best_position, best_confidence = position, confidence
        expected_positions = [best_position]
        fallback_anchor = anchor_by_position.get(best_position)
        if fallback_count != 1 or threshold_count != 0 or fallback_anchor is None or not bool(fallback_anchor.get("is_highest_confidence")):
            return None, "actual_batch_disagrees_with_original_fallback_action"
    if sorted(positions) != expected_positions:
        return None, "actual_batch_positions_do_not_match_original_threshold_plus_fallback_action"
    for position, token_id, _confidence in natural_values:
        if position in anchor_by_position and int(anchor_by_position[position]["token_id"]) != token_id:
            return None, "actual_batch_tokens_do_not_match_original_threshold_plus_fallback_action"

    summaries = record.get("position_summaries")
    if not isinstance(summaries, Mapping):
        return None, "missing_stored_position_summaries"
    for position in positions:
        summary = summaries.get(str(position))
        if not isinstance(summary, Mapping):
            return None, "missing_stored_base_token_summary"
        try:
            stored_token = int(summary["top1_token_id"])
        except (KeyError, TypeError, ValueError):
            return None, "invalid_stored_base_token_summary"
        if stored_token != int(fresh_assignments[position]["token_id"]):
            return None, "stored_base_token_differs_from_fresh_exact_top1"
    return sorted(positions), None


def _base_assignments(logits: Any, positions: Iterable[int], *, tie_tolerance: float) -> dict[int, dict[str, Any]]:
    """Extract only the fixed current exact top-1 assignment/scalars per mask."""

    import torch

    assignments: dict[int, dict[str, Any]] = {}
    for position in sorted(int(value) for value in positions):
        token_id = int(torch.argmax(logits[0, position].float()).item())
        summary = margin_summary(logits[0], position, token_id, tie_tolerance=tie_tolerance)
        assignments[position] = {
            "token_id": token_id,
            "top1_probability": float(summary["assigned_probability"]),
            "logit_margin": float(summary["logit_margin"]),
            "top2_token_id": int(summary["competitor_token_id"]),
            "top2_probability": float(summary["competitor_probability"]),
            "top1_matches_assignment": bool(summary["top1_matches_assignment"]),
            "is_logit_tie": bool(summary["is_logit_tie"]),
        }
    return assignments


def _pool_for_stratum(
    state: SelectedState,
    *,
    actual_batch_positions: Sequence[int],
    assignments: Mapping[int, Mapping[str, Any]],
    requested_size: int,
) -> tuple[dict[str, Any], str | None]:
    """Construct P=B plus confidence-ranked extras without truncating B."""

    base = list(sorted(int(value) for value in actual_batch_positions))
    common = _source_common(state)
    row: dict[str, Any] = {
        **common,
        "requested_pool_size": int(requested_size),
        "actual_fast_dllm_positions": base,
        "actual_fast_dllm_token_ids": [int(assignments[position]["token_id"]) for position in base],
        "actual_fast_dllm_size": len(base),
        "pool_positions": [],
        "pool_token_ids": [],
        "pool_top1_probabilities": [],
        "pool_extra_positions_ranked_by_fresh_confidence": [],
    }
    if len(base) > int(requested_size):
        row.update({
            "pool_status": "ineligible_base_batch_exceeds_pool_size",
            "effective_pool_size": None,
            "pool_size_shortfall": None,
        })
        return row, "ineligible_base_batch_exceeds_pool_size"
    extras = [position for position in assignments if position not in set(base)]
    extras.sort(key=lambda position: (-float(assignments[position]["top1_probability"]), int(position)))
    chosen = base + extras[: max(0, int(requested_size) - len(base))]
    summaries = state.record.get("position_summaries")
    if not isinstance(summaries, Mapping):
        row.update({"pool_status": "ineligible_missing_stored_position_summaries", "effective_pool_size": None})
        return row, "ineligible_missing_stored_position_summaries"
    for position in chosen:
        summary = summaries.get(str(position))
        try:
            stored_token = int(summary["top1_token_id"]) if isinstance(summary, Mapping) else None
        except (KeyError, TypeError, ValueError):
            stored_token = None
        if stored_token != int(assignments[position]["token_id"]):
            row.update({
                "pool_status": "ineligible_stored_pool_token_differs_from_fresh_exact_top1",
                "stored_mismatch_position": position,
                "stored_mismatch_token_id": stored_token,
                "fresh_top1_token_id": int(assignments[position]["token_id"]),
                "effective_pool_size": None,
            })
            return row, "ineligible_stored_pool_token_differs_from_fresh_exact_top1"
    row.update({
        "pool_status": "eligible",
        "effective_pool_size": len(chosen),
        "pool_size_shortfall": max(0, int(requested_size) - len(chosen)),
        "pool_positions": chosen,
        "pool_token_ids": [int(assignments[position]["token_id"]) for position in chosen],
        "pool_top1_probabilities": [float(assignments[position]["top1_probability"]) for position in chosen],
        "pool_extra_positions_ranked_by_fresh_confidence": extras,
    })
    return row, None


def _actual_batch_fresh_token_mismatches(
    record: Mapping[str, Any], assignments: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Make fresh-vs-stored B token exclusions directly inspectable."""

    result: list[dict[str, Any]] = []
    anchors = record.get("actual_committed_anchors")
    if not isinstance(anchors, list):
        return result
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            continue
        try:
            position = int(anchor["position"])
            stored_token = int(anchor["token_id"])
        except (KeyError, TypeError, ValueError):
            continue
        fresh = assignments.get(position)
        if fresh is not None and int(fresh["token_id"]) != stored_token:
            result.append({
                "position": position,
                "stored_actual_batch_token_id": stored_token,
                "fresh_exact_top1_token_id": int(fresh["token_id"]),
            })
    return result


def _context_position_list(mask: int, positions: Sequence[int]) -> list[int]:
    return [int(positions[index]) for index in mask_indices(int(mask), len(positions))]


def _evaluate_max_pool_contexts(
    model: Any,
    state: SelectedState,
    *,
    base_input_ids: Any,
    base_logits: Any,
    pool_positions: Sequence[int],
    pool_token_ids: Sequence[int],
    batch_size: int,
    tie_tolerance: float,
    accounting: ForwardAccounting,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every subset context once for the maximal nested pool.

    The base context has already been freshly replayed to establish `y`; it is
    inserted into the cache rather than repeated.  The full-reveal subset is
    nevertheless forwarded and logged, despite having zero still-masked pool
    targets, so every A subset P has a concrete exact evaluation record.
    """

    import torch

    pool_positions = tuple(int(value) for value in pool_positions)
    pool_token_ids = tuple(int(value) for value in pool_token_ids)
    size = len(pool_positions)
    if size < 1:
        raise ValueError("An exact headroom pool must contain the nonempty Fast-dLLM batch")
    if len(pool_token_ids) != size:
        raise ValueError("pool position/token lengths differ")
    common = _source_common(state)
    margins: dict[tuple[int, int], dict[str, Any]] = {}
    context_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    def consume(mask: int, logits: Any, row_index: int, *, forward_source: str) -> None:
        revealed_positions = _context_position_list(mask, pool_positions)
        revealed_token_ids = [pool_token_ids[index] for index in mask_indices(mask, size)]
        context_rows.append({
            **common,
            "max_pool_size": size,
            "revealed_mask": int(mask),
            "revealed_positions": revealed_positions,
            "revealed_token_ids": revealed_token_ids,
            "still_masked_pool_count": size - int(mask).bit_count(),
            "forward_source": forward_source,
            "full_reveal_has_no_pool_target_query": int(mask).bit_count() == size,
        })
        for target_index, (position, token_id) in enumerate(zip(pool_positions, pool_token_ids, strict=True)):
            if mask & (1 << target_index):
                continue
            payload = margin_summary(logits[row_index], position, token_id, tie_tolerance=tie_tolerance)
            payload.update({"target_index": target_index, "revealed_mask": int(mask)})
            margins[(target_index, int(mask))] = payload
            query_rows.append({
                **common,
                "max_pool_size": size,
                "target_pool_index": target_index,
                "target_position": position,
                "target_token_id": token_id,
                "revealed_mask": int(mask),
                "revealed_positions": revealed_positions,
                "logit_margin": payload["logit_margin"],
                "assigned_probability": payload["assigned_probability"],
                "competitor_token_id": payload["competitor_token_id"],
                "competitor_probability": payload["competitor_probability"],
                "top1_token_id": payload["top1_token_id"],
                "top1_matches_assignment": payload["top1_matches_assignment"],
                "is_logit_tie": payload["is_logit_tie"],
                "forward_source": forward_source,
            })

    consume(0, base_logits, 0, forward_source="fresh_base_exact_no_cache")
    branch_masks = list(range(1, 1 << size))
    for start in range(0, len(branch_masks), int(batch_size)):
        masks = branch_masks[start : start + int(batch_size)]
        branches = base_input_ids.expand(len(masks), -1).clone()
        for row_index, mask in enumerate(masks):
            for pool_index, position in enumerate(pool_positions):
                if mask & (1 << pool_index):
                    branches[row_index, position] = pool_token_ids[pool_index]
        logits = exact_logits_batched(model, branches)
        accounting.model_batch_calls += 1
        accounting.subset_context_forwards += len(masks)
        for row_index, mask in enumerate(masks):
            consume(mask, logits, row_index, forward_source="subset_exact_no_cache")
        del logits, branches
    del base_input_ids
    return margins, context_rows, query_rows


def _selected_payload(mask: int, positions: Sequence[int], token_ids: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    indices = mask_indices(int(mask), len(positions))
    return {
        "selected_mask": int(mask),
        "selected_pool_indices": list(indices),
        "selected_positions": [int(positions[index]) for index in indices],
        "selected_top1_token_ids": [int(token_ids[index]) for index in indices],
        "selected_top1_probabilities": [float(probabilities[index]) for index in indices],
        "selected_size": len(indices),
    }


def _witness_payload(
    certificate: SetCertificate,
    *,
    positions: Sequence[int],
    token_ids: Sequence[int],
) -> dict[str, Any]:
    """Turn a subset query witness into positions without trajectory claims."""

    if certificate.first_failure_target_index is None:
        return {
            "first_failure_target_position": None,
            "first_failure_target_token_id": None,
            "first_failure_revealed_subset_positions": [],
            "first_failure_revealed_subset_token_ids": [],
            "first_failure_canonical_completion_order_positions": [],
            "first_failure_is_all_order_query_witness": False,
        }
    target = int(certificate.first_failure_target_index)
    revealed = certificate.first_failure_revealed_mask or 0
    revealed_indices = mask_indices(revealed, len(positions))
    selected_indices = mask_indices(certificate.selected_mask, len(positions))
    remaining = [index for index in selected_indices if index != target and index not in revealed_indices]
    canonical = sorted(int(positions[index]) for index in revealed_indices) + [int(positions[target])] + sorted(
        int(positions[index]) for index in remaining
    )
    return {
        "first_failure_target_position": int(positions[target]),
        "first_failure_target_token_id": int(token_ids[target]),
        "first_failure_revealed_subset_positions": [int(positions[index]) for index in revealed_indices],
        "first_failure_revealed_subset_token_ids": [int(token_ids[index]) for index in revealed_indices],
        "first_failure_canonical_completion_order_positions": canonical,
        "first_failure_is_all_order_query_witness": True,
    }


def _certificate_payload(certificate: SetCertificate, *, positions: Sequence[int], token_ids: Sequence[int]) -> dict[str, Any]:
    return {
        "certificate_pass": bool(certificate.passes),
        "batch_failure": not bool(certificate.passes),
        "certificate_margin": certificate.certificate_margin,
        "certificate_query_count": certificate.query_count,
        "certificate_missing_query_count": certificate.missing_query_count,
        "certificate_vacuous_empty_set": certificate.vacuous,
        "token_violation_count": len(certificate.violating_indices),
        "token_violating_positions": [int(positions[index]) for index in certificate.violating_indices],
        "token_violating_token_ids": [int(token_ids[index]) for index in certificate.violating_indices],
        "minimum_margin_target_position": (
            None if certificate.minimum_margin_target_index is None else int(positions[certificate.minimum_margin_target_index])
        ),
        "minimum_margin_revealed_subset_positions": (
            []
            if certificate.minimum_margin_revealed_mask is None
            else _context_position_list(certificate.minimum_margin_revealed_mask, positions)
        ),
        "first_failure_margin": certificate.first_failure_margin,
        "first_failure_top1_matches_assignment": certificate.first_failure_top1_matches,
        "first_failure_is_logit_tie": certificate.first_failure_is_tie,
        **_witness_payload(certificate, positions=positions, token_ids=token_ids),
    }


def _policy_row(
    state: SelectedState,
    *,
    requested_pool_size: int,
    effective_pool_size: int,
    gamma: float,
    policy: str,
    policy_variant: str | None,
    selected_mask: int | None,
    certificate: SetCertificate | None,
    positions: Sequence[int],
    token_ids: Sequence[int],
    probabilities: Sequence[float],
    base_mask: int,
    extension_status: str | None = None,
) -> dict[str, Any]:
    """One complete state-policy result row, including explicit base failure."""

    base = {
        **_source_common(state),
        "requested_pool_size": int(requested_pool_size),
        "effective_pool_size": int(effective_pool_size),
        "gamma": float(gamma),
        "policy": policy,
        "policy_variant": policy_variant,
        "actual_fast_dllm_size": int(base_mask).bit_count(),
        "actual_fast_dllm_positions": _selected_payload(base_mask, positions, token_ids, probabilities)["selected_positions"],
        "actual_fast_dllm_mask": int(base_mask),
        "extension_status": extension_status,
    }
    if selected_mask is None or certificate is None:
        return {
            **base,
            "selected_mask": None,
            "selected_pool_indices": [],
            "selected_positions": [],
            "selected_top1_token_ids": [],
            "selected_top1_probabilities": [],
            "selected_size": None,
            "certificate_pass": None,
            "batch_failure": None,
            "certificate_margin": None,
            "certificate_query_count": None,
            "certificate_missing_query_count": None,
            "certificate_vacuous_empty_set": None,
            "token_violation_count": None,
            "token_violating_positions": [],
            "token_violating_token_ids": [],
            "mean_extra_vs_actual_fast_dllm": None,
            "safe_added_capacity": None,
            "has_at_least_one_safe_extra": False,
        }
    selected = _selected_payload(selected_mask, positions, token_ids, probabilities)
    extra = int(selected["selected_size"]) - int(base_mask).bit_count()
    return {
        **base,
        **selected,
        **_certificate_payload(certificate, positions=positions, token_ids=token_ids),
        "mean_extra_vs_actual_fast_dllm": extra,
        "safe_added_capacity": extra if policy == POLICY_EXTENSION and certificate.passes else None,
        "has_at_least_one_safe_extra": bool(policy == POLICY_EXTENSION and certificate.passes and extra > 0),
        "free_oracle_drops_actual_base_token": bool(policy == POLICY_FREE and (int(selected_mask) & int(base_mask)) != int(base_mask)),
    }


def evaluate_policies_for_pool(
    state: SelectedState,
    *,
    requested_pool_size: int,
    positions: Sequence[int],
    token_ids: Sequence[int],
    probabilities: Sequence[float],
    actual_batch_positions: Sequence[int],
    margins: Mapping[tuple[int, int], Mapping[str, Any] | float],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use the cached subset grid for B, preserving/free, and confidence policies."""

    position_to_index = {int(position): index for index, position in enumerate(positions)}
    base_mask = sum(1 << position_to_index[int(position)] for position in actual_batch_positions)
    if base_mask.bit_count() != len(actual_batch_positions):
        raise RuntimeError("Fast-dLLM batch is not contained in its eligible pool")
    policy_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    tolerance = float(config["certificate"]["tie_tolerance"])
    for gamma in (float(value) for value in config["sampling"]["margin_thresholds"]):
        certificates = certificate_cache_for_gamma(margins, len(positions), gamma, tolerance=tolerance)
        base_certificate = certificates[base_mask]
        row_specs: list[tuple[str, str | None, int | None, SetCertificate | None, str | None]] = [
            (POLICY_FAST, None, base_mask, base_certificate, None),
        ]
        if base_certificate.passes:
            extended_mask = choose_largest_safe_mask(certificates, probabilities, positions, required_mask=base_mask)
            if extended_mask is None:  # Impossible if B itself passes, but never silently coerce it.
                row_specs.append((POLICY_EXTENSION, None, None, None, "internal_no_preserving_safe_superset"))
            else:
                row_specs.append((POLICY_EXTENSION, None, extended_mask, certificates[extended_mask], "base_batch_exact_certificate_passed"))
        else:
            row_specs.append((POLICY_EXTENSION, None, None, None, "base_batch_exact_certificate_failed"))
        free_mask = choose_largest_safe_mask(certificates, probabilities, positions)
        if free_mask is None:
            raise RuntimeError("The empty set must be a vacuously safe free-oracle solution")
        row_specs.append((POLICY_FREE, "upper_bound_only", free_mask, certificates[free_mask], None))
        for tau in (float(value) for value in config["sampling"]["confidence_thresholds"]):
            selected_mask = confidence_selected_mask(probabilities, tau)
            row_specs.append((POLICY_CONFIDENCE, f"tau={tau:g}", selected_mask, certificates[selected_mask], None))

        for policy, variant, selected_mask, certificate, extension_status in row_specs:
            row = _policy_row(
                state,
                requested_pool_size=requested_pool_size,
                effective_pool_size=len(positions),
                gamma=gamma,
                policy=policy,
                policy_variant=variant,
                selected_mask=selected_mask,
                certificate=certificate,
                positions=positions,
                token_ids=token_ids,
                probabilities=probabilities,
                base_mask=base_mask,
                extension_status=extension_status,
            )
            policy_rows.append(row)
            if selected_mask is None or certificate is None:
                continue
            for index in mask_indices(selected_mask, len(positions)):
                target_failure = first_target_failure(
                    margins,
                    selected_mask,
                    index,
                    gamma,
                    tolerance=tolerance,
                )
                revealed_mask = None if target_failure is None else int(target_failure["revealed_mask"])
                revealed_indices = () if revealed_mask is None else mask_indices(revealed_mask, len(positions))
                remaining = [
                    candidate
                    for candidate in mask_indices(selected_mask, len(positions))
                    if candidate != index and candidate not in revealed_indices
                ]
                token_rows.append({
                    **_source_common(state),
                    "requested_pool_size": int(requested_pool_size),
                    "effective_pool_size": len(positions),
                    "gamma": gamma,
                    "policy": policy,
                    "policy_variant": variant,
                    "selected_position": int(positions[index]),
                    "selected_token_id": int(token_ids[index]),
                    "selected_top1_probability": float(probabilities[index]),
                    "token_violated": target_failure is not None,
                    "witness_revealed_subset_positions": [int(positions[value]) for value in revealed_indices],
                    "witness_revealed_subset_token_ids": [int(token_ids[value]) for value in revealed_indices],
                    "witness_target_position": int(positions[index]) if target_failure is not None else None,
                    "witness_target_token_id": int(token_ids[index]) if target_failure is not None else None,
                    "witness_logit_margin": None if target_failure is None else target_failure["logit_margin"],
                    "witness_top1_matches_assignment": None if target_failure is None else target_failure["top1_matches_assignment"],
                    "witness_is_logit_tie": None if target_failure is None else target_failure["is_logit_tie"],
                    "witness_canonical_completion_order_positions": (
                        []
                        if target_failure is None
                        else sorted(int(positions[value]) for value in revealed_indices)
                        + [int(positions[index])]
                        + sorted(int(positions[value]) for value in remaining)
                    ),
                    "witness_is_all_order_query_witness": target_failure is not None,
                })
    return policy_rows, token_rows


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Stable policy-result key used to join state and token evidence."""

    return (
        str(row.get("state_key")),
        str(row.get("cohort")),
        int(row.get("requested_pool_size")),
        float(row.get("gamma")),
        str(row.get("policy")),
        str(row.get("policy_variant") or ""),
    )


def _descriptive_quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * float(fraction)
    low = int(math.floor(location))
    high = int(math.ceil(location))
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def _metric_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Prompt-macro CIs with the raw micro statistic retained alongside."""

    valid: list[dict[str, Any]] = []
    for row in rows:
        value = _finite_float(row.get(field))
        prompt = row.get("prompt_id")
        if value is None or prompt is None:
            continue
        valid.append({"prompt_id": str(prompt), field: value})
    if not valid:
        return {
            "availability": "zero_count",
            "observation_count": 0,
            "micro_estimate": None,
            "prompt_macro_estimate": None,
            "prompt_clustered_ci95_low": None,
            "prompt_clustered_ci95_high": None,
            "prompt_cluster_count": 0,
            "bootstrap_iterations": int(iterations),
        }
    boot = prompt_clustered_bootstrap(
        valid,
        value_field=field,
        prompt_field="prompt_id",
        iterations=int(iterations),
        seed=int(seed),
        weighting="macro",
    )
    return {
        "availability": "available",
        "observation_count": len(valid),
        "micro_estimate": math.fsum(float(row[field]) for row in valid) / len(valid),
        "prompt_macro_estimate": boot["estimate"],
        "prompt_clustered_ci95_low": boot["ci95_low"],
        "prompt_clustered_ci95_high": boot["ci95_high"],
        "prompt_cluster_count": boot["cluster_count"],
        "bootstrap_iterations": boot["iterations"],
        "bootstrap_backend": boot.get("bootstrap_backend"),
    }


def _put_metric(
    destination: dict[str, Any],
    prefix: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    iterations: int,
    seed: int,
) -> None:
    for key, value in _metric_stats(rows, field=field, iterations=iterations, seed=seed).items():
        destination[f"{prefix}_{key}"] = value


def _summary_scope_rows(rows: Sequence[Mapping[str, Any]], *, scope: str) -> list[Mapping[str, Any]]:
    if scope == "all_eligible":
        return list(rows)
    if scope == "common_eligible_M4_M6_M8":
        return [row for row in rows if bool(row.get("common_eligible_M4_M6_M8"))]
    raise ValueError(f"Unknown summary scope {scope!r}")


def summarize_policies(
    policy_rows: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Report counts/certificate risk/capacity without pooling cohorts or M."""

    iterations = int(config["sampling"]["clustered_bootstrap_replicates"])
    seed = int(config["sampling"]["bootstrap_seed"])
    output: list[dict[str, Any]] = []
    # Requested M is the experimental stratum.  Effective pool size can be
    # smaller when fewer masked positions remain, so keep its distribution in
    # the row rather than accidentally treating that availability artifact as
    # a separate experiment.
    group_fields = ("cohort", "requested_pool_size", "gamma", "policy", "policy_variant")
    for scope in ("all_eligible", "common_eligible_M4_M6_M8"):
        scoped_states = _summary_scope_rows(policy_rows, scope=scope)
        scoped_tokens = _summary_scope_rows(token_rows, scope=scope)
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in scoped_states:
            groups[tuple(row.get(field) for field in group_fields)].append(row)
        for group_index, (key, rows) in enumerate(sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0]))):
            cohort, requested_size, gamma, policy, variant = key
            selected = [row for row in rows if row.get("selected_size") is not None]
            state_keys = {_row_key(row) for row in rows}
            group_tokens = [row for row in scoped_tokens if _row_key(row) in state_keys]
            counts = [float(row["selected_size"]) for row in selected]
            extras = [row for row in selected if row.get("mean_extra_vs_actual_fast_dllm") is not None]
            result: dict[str, Any] = {
                "scope": scope,
                "cohort": cohort,
                "requested_pool_size": requested_size,
                "effective_pool_size_min": min(int(row["effective_pool_size"]) for row in rows),
                "effective_pool_size_max": max(int(row["effective_pool_size"]) for row in rows),
                "effective_pool_size_distribution": dict(Counter(int(row["effective_pool_size"]) for row in rows)),
                "gamma": gamma,
                "policy": policy,
                "policy_variant": variant,
                "eligible_state_count": len(rows),
                "selected_set_available_state_count": len(selected),
                "selected_set_denominator_interpretation": (
                    "conditional_on_actual_B_exact_certificate_pass"
                    if policy == POLICY_EXTENSION
                    else "all_eligible_states_for_this_policy"
                ),
                "selected_token_count": len(group_tokens),
                "median_committed_tokens_per_state": _descriptive_quantile(counts, .5),
                "q1_committed_tokens_per_state": _descriptive_quantile(counts, .25),
                "q3_committed_tokens_per_state": _descriptive_quantile(counts, .75),
                "extension_base_batch_certificate_failed_state_count": sum(
                    row.get("extension_status") == "base_batch_exact_certificate_failed" for row in rows
                ),
                "free_oracle_drops_actual_base_token_state_count": sum(
                    bool(row.get("free_oracle_drops_actual_base_token")) for row in selected
                ),
            }
            offset = group_index * 17
            _put_metric(result, "batch_failure", selected, field="batch_failure", iterations=iterations, seed=seed + offset)
            _put_metric(result, "token_violation", group_tokens, field="token_violated", iterations=iterations, seed=seed + offset + 1)
            _put_metric(result, "committed_tokens", selected, field="selected_size", iterations=iterations, seed=seed + offset + 2)
            _put_metric(result, "extra_vs_actual_fast_dllm", extras, field="mean_extra_vs_actual_fast_dllm", iterations=iterations, seed=seed + offset + 3)
            if policy == POLICY_EXTENSION:
                base_pass = [row for row in rows if row.get("selected_size") is not None]
                _put_metric(result, "safe_added_capacity_conditional_on_B_pass", base_pass, field="safe_added_capacity", iterations=iterations, seed=seed + offset + 4)
                _put_metric(
                    result,
                    "actual_fast_dllm_tokens_on_same_B_pass_states",
                    base_pass,
                    field="actual_fast_dllm_size",
                    iterations=iterations,
                    seed=seed + offset + 14,
                )
                _put_metric(result, "state_has_at_least_one_safe_extra", rows, field="has_at_least_one_safe_extra", iterations=iterations, seed=seed + offset + 5)
                safe_values = [float(row["safe_added_capacity"]) for row in base_pass if row.get("safe_added_capacity") is not None]
                result["median_safe_added_capacity_conditional_on_B_pass"] = _descriptive_quantile(safe_values, .5)
            elif policy == POLICY_FREE:
                _put_metric(
                    result,
                    "free_oracle_drops_actual_base_token",
                    selected,
                    field="free_oracle_drops_actual_base_token",
                    iterations=iterations,
                    seed=seed + offset + 6,
                )
                result["median_safe_added_capacity_conditional_on_B_pass"] = None
            else:
                result["median_safe_added_capacity_conditional_on_B_pass"] = None
            mean_count = result.get("committed_tokens_prompt_macro_estimate")
            result["ideal_commit_tokens_per_decode_step"] = mean_count
            result["ideal_main_decode_nfe_per_committed_token"] = None if not mean_count else 1.0 / float(mean_count)
            output.append(result)
    return output


def confidence_sweep_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep confidence-only frontiers in a compact, plot-ready table."""

    result: list[dict[str, Any]] = []
    for row in summary_rows:
        if row.get("policy") != POLICY_CONFIDENCE:
            continue
        variant = str(row.get("policy_variant") or "")
        try:
            tau = float(variant.removeprefix("tau="))
        except ValueError:
            tau = None
        result.append({
            "scope": row.get("scope"),
            "cohort": row.get("cohort"),
            "requested_pool_size": row.get("requested_pool_size"),
            "effective_pool_size_min": row.get("effective_pool_size_min"),
            "effective_pool_size_max": row.get("effective_pool_size_max"),
            "gamma": row.get("gamma"),
            "confidence_threshold": tau,
            "mean_committed_tokens": row.get("committed_tokens_prompt_macro_estimate"),
            "mean_committed_tokens_ci95_low": row.get("committed_tokens_prompt_clustered_ci95_low"),
            "mean_committed_tokens_ci95_high": row.get("committed_tokens_prompt_clustered_ci95_high"),
            "batch_failure_rate": row.get("batch_failure_prompt_macro_estimate"),
            "batch_failure_ci95_low": row.get("batch_failure_prompt_clustered_ci95_low"),
            "batch_failure_ci95_high": row.get("batch_failure_prompt_clustered_ci95_high"),
            "token_violation_rate": row.get("token_violation_prompt_macro_estimate"),
            "token_violation_ci95_low": row.get("token_violation_prompt_clustered_ci95_low"),
            "token_violation_ci95_high": row.get("token_violation_prompt_clustered_ci95_high"),
            "selected_set_available_state_count": row.get("selected_set_available_state_count"),
        })
    return result


def _rows_for_match(
    rows: Sequence[Mapping[str, Any]],
    state_scope: set[tuple[str, str, int, float]],
    *,
    policy: str,
    variant: str | None = None,
    partition: str | None = None,
) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for row in rows:
        state_key = (
            str(row.get("state_key")),
            str(row.get("cohort")),
            int(row.get("requested_pool_size")),
            float(row.get("gamma")),
        )
        if state_key not in state_scope or row.get("policy") != policy:
            continue
        if variant is not None and str(row.get("policy_variant") or "") != variant:
            continue
        if partition is not None and row.get("prompt_partition") != partition:
            continue
        result.append(row)
    return result


def _match_metric_rows(
    policy_rows: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Mapping[str, Any]],
    *,
    state_scope: set[tuple[str, str, int, float]],
    policy: str,
    variant: str | None,
    partition: str,
    risk_unit: str,
) -> tuple[list[Mapping[str, Any]], str]:
    if risk_unit == "batch_failure":
        return _rows_for_match(policy_rows, state_scope, policy=policy, variant=variant, partition=partition), "batch_failure"
    if risk_unit == "token_violation":
        return _rows_for_match(token_rows, state_scope, policy=policy, variant=variant, partition=partition), "token_violated"
    raise ValueError(f"Unknown matching risk unit {risk_unit!r}")


def _paired_commit_delta_rows(
    oracle_rows: Sequence[Mapping[str, Any]], confidence_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Pair held-out capacity by identical state before prompt bootstrapping."""

    def key(row: Mapping[str, Any]) -> tuple[str, str, int, float]:
        return (
            str(row["state_key"]),
            str(row["cohort"]),
            int(row["requested_pool_size"]),
            float(row["gamma"]),
        )

    confidence_by_state = {key(row): row for row in confidence_rows if row.get("selected_size") is not None}
    output: list[dict[str, Any]] = []
    for oracle in oracle_rows:
        if oracle.get("selected_size") is None:
            continue
        confidence = confidence_by_state.get(key(oracle))
        if confidence is None:
            continue
        delta = int(oracle["selected_size"]) - int(confidence["selected_size"])
        output.append({
            "prompt_id": oracle["prompt_id"],
            "state_key": oracle["state_key"],
            "oracle_commit_count": int(oracle["selected_size"]),
            "confidence_commit_count": int(confidence["selected_size"]),
            "oracle_minus_confidence_commit_count": delta,
            "oracle_strictly_more_commits_than_confidence": delta > 0,
        })
    return output


def matched_risk_rows(
    policy_rows: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Choose tau on calibration prompts only and report held-out exact risk.

    This does not turn the oracle into a predictor.  It merely prevents the
    confidence threshold frontier from choosing a lucky post-hoc tau on the
    same prompts it is evaluated on.
    """

    matching = config["matching"]
    iterations = int(config["sampling"]["clustered_bootstrap_replicates"])
    seed = int(config["sampling"]["bootstrap_seed"])
    output: list[dict[str, Any]] = []
    dimensions = sorted({
        (str(row["cohort"]), int(row["requested_pool_size"]), float(row["gamma"]))
        for row in policy_rows
        if row.get("cohort") == "primary"
    })
    tau_variants = [f"tau={float(value):g}" for value in config["sampling"]["confidence_thresholds"]]
    for dimension_index, (cohort, pool_size, gamma) in enumerate(dimensions):
        universe = [
            row
            for row in policy_rows
            if str(row["cohort"]) == cohort and int(row["requested_pool_size"]) == pool_size and float(row["gamma"]) == gamma
        ]
        by_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in universe:
            by_policy[str(row["policy"])].append(row)
        comparisons = (
            (POLICY_EXTENSION, "preserving_extension_B_pass_only"),
            (POLICY_FREE, "free_oracle_upper_bound_all_states"),
        )
        for oracle_policy, scope_name in comparisons:
            oracle_states = [row for row in by_policy.get(oracle_policy, ()) if row.get("selected_size") is not None]
            state_scope = {
                (str(row["state_key"]), str(row["cohort"]), int(row["requested_pool_size"]), float(row["gamma"]))
                for row in oracle_states
            }
            if not state_scope:
                continue
            for risk_index, risk_unit in enumerate(matching["risk_units"]):
                for budget_index, risk_budget in enumerate(float(value) for value in matching["risk_budgets"]):
                    candidates: list[tuple[float, float, str, dict[str, Any]]] = []
                    for variant in tau_variants:
                        risk_rows, risk_field = _match_metric_rows(
                            policy_rows,
                            token_rows,
                            state_scope=state_scope,
                            policy=POLICY_CONFIDENCE,
                            variant=variant,
                            partition="calibration",
                            risk_unit=str(risk_unit),
                        )
                        risk = _metric_stats(
                            risk_rows,
                            field=risk_field,
                            iterations=iterations,
                            seed=seed + dimension_index * 101 + risk_index * 17 + budget_index,
                        )
                        capacity_rows = _rows_for_match(
                            policy_rows,
                            state_scope,
                            policy=POLICY_CONFIDENCE,
                            variant=variant,
                            partition="calibration",
                        )
                        capacity = _metric_stats(
                            capacity_rows,
                            field="selected_size",
                            iterations=iterations,
                            seed=seed + dimension_index * 101 + risk_index * 17 + budget_index + 1,
                        )
                        risk_estimate = risk.get("prompt_macro_estimate")
                        capacity_estimate = capacity.get("prompt_macro_estimate")
                        if risk_estimate is None or capacity_estimate is None or float(risk_estimate) > risk_budget:
                            continue
                        try:
                            tau = float(variant.removeprefix("tau="))
                        except ValueError:
                            continue
                        candidates.append((float(capacity_estimate), tau, variant, {"risk": risk, "capacity": capacity}))
                    base = {
                        "cohort": cohort,
                        "requested_pool_size": pool_size,
                        "gamma": gamma,
                        "oracle_policy": oracle_policy,
                        "oracle_scope": scope_name,
                        "risk_unit": risk_unit,
                        "risk_budget": risk_budget,
                        "calibration_prompt_disjoint": True,
                        "calibration_state_count": len(_rows_for_match(policy_rows, state_scope, policy=oracle_policy, partition="calibration")),
                        "test_state_count": len(_rows_for_match(policy_rows, state_scope, policy=oracle_policy, partition="test")),
                    }
                    if not candidates:
                        output.append({**base, "status": "no_confidence_threshold_met_calibration_risk_budget", "selected_confidence_threshold": None})
                        continue
                    # Max capacity, then lower tau (more transparent/less arbitrary) if tied.
                    capacity_value, tau, variant, calibration = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
                    confidence_test_risk_rows, confidence_risk_field = _match_metric_rows(
                        policy_rows,
                        token_rows,
                        state_scope=state_scope,
                        policy=POLICY_CONFIDENCE,
                        variant=variant,
                        partition="test",
                        risk_unit=str(risk_unit),
                    )
                    oracle_test_risk_rows, oracle_risk_field = _match_metric_rows(
                        policy_rows,
                        token_rows,
                        state_scope=state_scope,
                        policy=oracle_policy,
                        variant=None,
                        partition="test",
                        risk_unit=str(risk_unit),
                    )
                    confidence_test_capacity_rows = _rows_for_match(
                        policy_rows, state_scope, policy=POLICY_CONFIDENCE, variant=variant, partition="test"
                    )
                    oracle_test_capacity_rows = _rows_for_match(
                        policy_rows, state_scope, policy=oracle_policy, partition="test"
                    )
                    paired_capacity_rows = _paired_commit_delta_rows(
                        oracle_test_capacity_rows, confidence_test_capacity_rows
                    )
                    offset = dimension_index * 1000 + risk_index * 100 + budget_index * 10
                    output.append({
                        **base,
                        "status": "selected_on_calibration_evaluated_on_prompt_disjoint_test",
                        "selected_confidence_threshold": tau,
                        "selected_confidence_variant": variant,
                        "calibration_confidence_risk": calibration["risk"].get("prompt_macro_estimate"),
                        "calibration_confidence_mean_commit_count": capacity_value,
                        **{
                            f"confidence_test_risk_{key}": value
                            for key, value in _metric_stats(
                                confidence_test_risk_rows, field=confidence_risk_field, iterations=iterations, seed=seed + offset + 2
                            ).items()
                        },
                        **{
                            f"oracle_test_risk_{key}": value
                            for key, value in _metric_stats(
                                oracle_test_risk_rows, field=oracle_risk_field, iterations=iterations, seed=seed + offset + 3
                            ).items()
                        },
                        **{
                            f"confidence_test_commit_count_{key}": value
                            for key, value in _metric_stats(
                                confidence_test_capacity_rows, field="selected_size", iterations=iterations, seed=seed + offset + 4
                            ).items()
                        },
                        **{
                            f"oracle_test_commit_count_{key}": value
                            for key, value in _metric_stats(
                                oracle_test_capacity_rows, field="selected_size", iterations=iterations, seed=seed + offset + 5
                            ).items()
                        },
                        "heldout_paired_capacity_state_count": len(paired_capacity_rows),
                        **{
                            f"heldout_paired_oracle_minus_confidence_commit_count_{key}": value
                            for key, value in _metric_stats(
                                paired_capacity_rows,
                                field="oracle_minus_confidence_commit_count",
                                iterations=iterations,
                                seed=seed + offset + 6,
                            ).items()
                        },
                        **{
                            f"heldout_paired_oracle_strictly_more_commits_rate_{key}": value
                            for key, value in _metric_stats(
                                paired_capacity_rows,
                                field="oracle_strictly_more_commits_than_confidence",
                                iterations=iterations,
                                seed=seed + offset + 7,
                            ).items()
                        },
                    })
    return output


def _summary_lookup(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    cohort: str,
    pool_size: int,
    gamma: float,
    policy: str,
    variant: str | None = None,
    scope: str = "all_eligible",
) -> Mapping[str, Any] | None:
    for row in summary_rows:
        if (
            row.get("scope") == scope
            and row.get("cohort") == cohort
            and int(row.get("requested_pool_size", -1)) == int(pool_size)
            and float(row.get("gamma", math.nan)) == float(gamma)
            and row.get("policy") == policy
            and str(row.get("policy_variant") or "") == str(variant or "")
        ):
            return row
    return None


def _fmt_rate(value: Any) -> str:
    number = _finite_float(value)
    return "unavailable" if number is None else f"{100.0 * number:.2f}%"


def _fmt_value(value: Any, digits: int = 3) -> str:
    number = _finite_float(value)
    return "unavailable" if number is None else f"{number:.{digits}f}"


def headroom_decision(
    summary_rows: Sequence[Mapping[str, Any]], matched_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the predeclared GO/NO-GO rule, including held-out matching."""

    target_m = max(int(value) for value in config["sampling"]["pool_sizes"])
    row = _summary_lookup(
        summary_rows,
        cohort="primary",
        pool_size=target_m,
        gamma=0.0,
        policy=POLICY_EXTENSION,
    )
    core_risk_unit = str(config["matching"]["core_risk_unit"])
    core_risk_budget = float(config["matching"]["core_risk_budget"])
    criterion = (
        "GO only if (a) the primary M=max, gamma=0 preserving-extension result has strictly positive "
        "prompt-clustered 95% lower bounds for both conditional safe added capacity and unconditional "
        "state-level safe-extra coverage; and (b) its calibration-chosen, prompt-disjoint held-out "
        "oracle-minus-confidence commit-count delta has a strictly positive 95% lower bound while both "
        f"reported exact {core_risk_unit} rates are at most the predeclared {core_risk_budget:g} budget."
    )
    if row is None:
        return {
            "decision": "NO_GO_INSUFFICIENT_ELIGIBLE_PRIMARY_EVIDENCE",
            "criterion": criterion,
            "reason": "No eligible primary preserving-extension row exists at the largest requested pool size.",
        }
    capacity_low = _finite_float(row.get("safe_added_capacity_conditional_on_B_pass_prompt_clustered_ci95_low"))
    coverage_low = _finite_float(row.get("state_has_at_least_one_safe_extra_prompt_clustered_ci95_low"))
    matched = [
        row
        for row in matched_rows
        if row.get("status") == "selected_on_calibration_evaluated_on_prompt_disjoint_test"
        and row.get("cohort") == "primary"
        and int(row.get("requested_pool_size", -1)) == target_m
        and float(row.get("gamma", math.nan)) == 0.0
        and row.get("oracle_policy") == POLICY_EXTENSION
        and row.get("risk_unit") == core_risk_unit
        and float(row.get("risk_budget", math.nan)) == core_risk_budget
    ]
    if not matched:
        return {
            "decision": "NO_GO_NO_PROMPT_DISJOINT_MATCHED_CONFIDENCE_EVIDENCE",
            "criterion": criterion,
            "reason": "No calibration-selected confidence threshold produced an evaluable prompt-disjoint held-out comparison at the core risk budget.",
            "safe_added_capacity_ci95_low": capacity_low,
            "safe_extra_coverage_ci95_low": coverage_low,
        }
    matched_row = matched[0]
    delta_low = _finite_float(matched_row.get("heldout_paired_oracle_minus_confidence_commit_count_prompt_clustered_ci95_low"))
    confidence_risk = _finite_float(matched_row.get("confidence_test_risk_prompt_macro_estimate"))
    oracle_risk = _finite_float(matched_row.get("oracle_test_risk_prompt_macro_estimate"))
    if (
        capacity_low is not None
        and coverage_low is not None
        and delta_low is not None
        and confidence_risk is not None
        and oracle_risk is not None
        and capacity_low > 0.0
        and coverage_low > 0.0
        and delta_low > 0.0
        and confidence_risk <= core_risk_budget
        and oracle_risk <= core_risk_budget
    ):
        return {
            "decision": "GO_EXACT_ORACLE_EVIDENCE_OF_POSITIVE_SAFE_HEADROOM",
            "criterion": criterion,
            "reason": "Primary headroom and the paired held-out exact-oracle advantage over confidence-only selection both clear the predeclared direction-only rule. This supports only the exact-oracle direction, not deployment speed.",
            "safe_added_capacity_ci95_low": capacity_low,
            "safe_extra_coverage_ci95_low": coverage_low,
            "heldout_paired_oracle_minus_confidence_commit_count_ci95_low": delta_low,
            "heldout_confidence_risk": confidence_risk,
            "heldout_oracle_risk": oracle_risk,
        }
    return {
        "decision": "NO_GO_NO_ROBUST_POSITIVE_SAFE_HEADROOM",
        "criterion": criterion,
        "reason": "Primary safe-headroom evidence, the held-out paired confidence comparison, or exact risk matching did not clear the predeclared rule; do not mask this negative/inconclusive core result with upper-bound analyses.",
        "safe_added_capacity_ci95_low": capacity_low,
        "safe_extra_coverage_ci95_low": coverage_low,
        "heldout_paired_oracle_minus_confidence_commit_count_ci95_low": delta_low,
        "heldout_confidence_risk": confidence_risk,
        "heldout_oracle_risk": oracle_risk,
    }


def _plot_or_empty(axis: Any, message: str) -> None:
    axis.text(.5, .5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def write_figures(
    output_dir: Path,
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
    accounting: ForwardAccounting,
    config: Mapping[str, Any],
) -> list[str]:
    """Write compact capacity/risk and cost figures from exact scalar outputs."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(name: str) -> None:
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))

    # The requested capacity-vs-exact-risk frontier.  Both axes carry prompt
    # clustered CIs; the free policy is visibly labelled as an upper bound.
    target_m = max(int(value) for value in config["sampling"]["pool_sizes"])
    frontier = [
        row
        for row in summary_rows
        if row.get("scope") == "all_eligible"
        and row.get("cohort") == "primary"
        and int(row.get("requested_pool_size", -1)) == target_m
        and float(row.get("gamma", math.nan)) == 0.0
        and row.get("policy") in {POLICY_FAST, POLICY_EXTENSION, POLICY_FREE, POLICY_CONFIDENCE}
    ]
    plt.figure(figsize=(7.4, 5.2))
    axis = plt.gca()
    if frontier:
        styles = {
            POLICY_FAST: ("#2563eb", "o", "Fast-dLLM actual"),
            POLICY_EXTENSION: ("#16a34a", "s", "VCCC preserving extension"),
            POLICY_FREE: ("#9333ea", "^", "VCCC free (upper bound only)"),
            POLICY_CONFIDENCE: ("#ea580c", ".", "confidence-only tau sweep"),
        }
        seen: set[str] = set()
        for row in frontier:
            risk = _finite_float(row.get("batch_failure_prompt_macro_estimate"))
            capacity = _finite_float(row.get("committed_tokens_prompt_macro_estimate"))
            if risk is None or capacity is None:
                continue
            policy = str(row["policy"])
            color, marker, label = styles[policy]
            xlow = _finite_float(row.get("batch_failure_prompt_clustered_ci95_low"))
            xhigh = _finite_float(row.get("batch_failure_prompt_clustered_ci95_high"))
            ylow = _finite_float(row.get("committed_tokens_prompt_clustered_ci95_low"))
            yhigh = _finite_float(row.get("committed_tokens_prompt_clustered_ci95_high"))
            axis.errorbar(
                risk,
                capacity,
                xerr=[[max(0.0, risk - (xlow if xlow is not None else risk))], [max(0.0, (xhigh if xhigh is not None else risk) - risk)]],
                yerr=[[max(0.0, capacity - (ylow if ylow is not None else capacity))], [max(0.0, (yhigh if yhigh is not None else capacity) - capacity)]],
                fmt=marker,
                markersize=6,
                color=color,
                alpha=.8,
                label=label if policy not in seen else None,
            )
            seen.add(policy)
        axis.set_xlabel("exact all-order batch failure rate (prompt macro; 95% CI)")
        axis.set_ylabel("committed current-top-1 tokens / state (prompt macro; 95% CI)")
        axis.set_title(f"Exact safe-commit capacity vs reversal risk (primary, M={target_m}, gamma=0)")
        axis.legend(fontsize=8)
    else:
        _plot_or_empty(axis, "No eligible primary M=max gamma=0 frontier rows")
    save("safe_commit_capacity_vs_exact_reversal_risk.png")

    plt.figure(figsize=(7.4, 5.2))
    axis = plt.gca()
    if frontier:
        styles = {
            POLICY_FAST: ("#2563eb", "o", "Fast-dLLM actual"),
            POLICY_EXTENSION: ("#16a34a", "s", "VCCC preserving extension"),
            POLICY_FREE: ("#9333ea", "^", "VCCC free (upper bound only)"),
            POLICY_CONFIDENCE: ("#ea580c", ".", "confidence-only tau sweep"),
        }
        seen: set[str] = set()
        for row in frontier:
            risk = _finite_float(row.get("token_violation_prompt_macro_estimate"))
            capacity = _finite_float(row.get("committed_tokens_prompt_macro_estimate"))
            if risk is None or capacity is None:
                continue
            policy = str(row["policy"])
            color, marker, label = styles[policy]
            xlow = _finite_float(row.get("token_violation_prompt_clustered_ci95_low"))
            xhigh = _finite_float(row.get("token_violation_prompt_clustered_ci95_high"))
            ylow = _finite_float(row.get("committed_tokens_prompt_clustered_ci95_low"))
            yhigh = _finite_float(row.get("committed_tokens_prompt_clustered_ci95_high"))
            axis.errorbar(
                risk,
                capacity,
                xerr=[[max(0.0, risk - (xlow if xlow is not None else risk))], [max(0.0, (xhigh if xhigh is not None else risk) - risk)]],
                yerr=[[max(0.0, capacity - (ylow if ylow is not None else capacity))], [max(0.0, (yhigh if yhigh is not None else capacity) - capacity)]],
                fmt=marker,
                markersize=6,
                color=color,
                alpha=.8,
                label=label if policy not in seen else None,
            )
            seen.add(policy)
        axis.set_xlabel("exact all-order token violation rate (prompt macro; 95% CI)")
        axis.set_ylabel("committed current-top-1 tokens / state (prompt macro; 95% CI)")
        axis.set_title(f"Exact safe-commit capacity vs token violation risk (primary, M={target_m}, gamma=0)")
        axis.legend(fontsize=8)
    else:
        _plot_or_empty(axis, "No eligible primary M=max gamma=0 frontier rows")
    save("safe_commit_capacity_vs_exact_token_violation_risk.png")

    # Distribution makes it easy to distinguish a few large gains from broad
    # state-level headroom.
    distribution = [
        row
        for row in policy_rows
        if row.get("cohort") == "primary"
        and int(row.get("requested_pool_size", -1)) == target_m
        and float(row.get("gamma", math.nan)) == 0.0
        and row.get("policy") in {POLICY_FAST, POLICY_EXTENSION, POLICY_FREE}
        and row.get("selected_size") is not None
    ]
    plt.figure(figsize=(7.0, 4.6))
    axis = plt.gca()
    labels = []
    values = []
    for policy, label in (
        (POLICY_FAST, "Fast-dLLM B"),
        (POLICY_EXTENSION, "preserving B+T*"),
        (POLICY_FREE, "free exact upper bound"),
    ):
        selected = [float(row["selected_size"]) for row in distribution if row.get("policy") == policy]
        if selected:
            labels.append(label)
            values.append(selected)
    if values:
        axis.boxplot(values, tick_labels=labels, showmeans=True)
        axis.set_ylabel("committed tokens per selected state")
        axis.set_title(f"Exact set-size distribution (primary, M={target_m}, gamma=0)")
        axis.tick_params(axis="x", labelrotation=12)
    else:
        _plot_or_empty(axis, "No selected set-size rows")
    save("exact_oracle_set_size_distribution.png")

    extras = [
        float(row["safe_added_capacity"])
        for row in policy_rows
        if row.get("cohort") == "primary"
        and int(row.get("requested_pool_size", -1)) == target_m
        and float(row.get("gamma", math.nan)) == 0.0
        and row.get("policy") == POLICY_EXTENSION
        and row.get("safe_added_capacity") is not None
    ]
    plt.figure(figsize=(6.4, 4.4))
    axis = plt.gca()
    if extras:
        bins = list(range(0, max(int(value) for value in extras) + 2))
        axis.hist(extras, bins=bins, align="left", rwidth=.8, color="#16a34a")
        axis.set_xticks(bins[:-1])
        axis.set_xlabel("safe added positions |T*|, conditional on B passing")
        axis.set_ylabel("selected states")
        axis.set_title("Preserving-extension headroom distribution")
    else:
        _plot_or_empty(axis, "No base-passing preserving-extension states")
    save("preserving_extension_safe_extra_distribution.png")

    plt.figure(figsize=(6.6, 4.2))
    axis = plt.gca()
    exact_contexts = accounting.subset_context_forwards
    axis.bar(["fresh base\nforwards", "subset-context\nverification forwards"], [accounting.base_forwards, exact_contexts], color=["#2563eb", "#dc2626"])
    axis.set_ylabel("exact no-cache forwards")
    axis.set_title("Oracle verification cost (not a decoder speedup)")
    save("exact_oracle_verification_forward_cost.png")
    return paths


def write_report(
    output_root: Path,
    *,
    pool_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    matched_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Human-readable direct answer with upper bounds kept separate."""

    target_m = max(int(value) for value in config["sampling"]["pool_sizes"])
    primary_pools = [row for row in pool_rows if row.get("cohort") == "primary"]
    eligibility = {
        m: [row for row in primary_pools if int(row.get("requested_pool_size", -1)) == m]
        for m in (int(value) for value in config["sampling"]["pool_sizes"])
    }
    fast = _summary_lookup(summary_rows, cohort="primary", pool_size=target_m, gamma=0.0, policy=POLICY_FAST)
    extension = _summary_lookup(summary_rows, cohort="primary", pool_size=target_m, gamma=0.0, policy=POLICY_EXTENSION)
    free = _summary_lookup(summary_rows, cohort="primary", pool_size=target_m, gamma=0.0, policy=POLICY_FREE)
    decision = headroom_decision(summary_rows, matched_rows, config)
    matched = [
        row
        for row in matched_rows
        if int(row.get("requested_pool_size", -1)) == target_m
        and float(row.get("gamma", math.nan)) == 0.0
        and row.get("oracle_policy") == POLICY_EXTENSION
        and row.get("risk_unit") == config["matching"]["core_risk_unit"]
        and float(row.get("risk_budget", math.nan)) == float(config["matching"]["core_risk_budget"])
        and row.get("status") == "selected_on_calibration_evaluated_on_prompt_disjoint_test"
    ]
    selected_match = matched[0] if matched else None
    direct = {
        "decision": decision,
        "largest_pool_size": target_m,
        "primary_pool_eligibility": {
            str(m): {
                "eligible": sum(row.get("pool_status") == "eligible" for row in rows),
                "screened": len(rows),
            }
            for m, rows in eligibility.items()
        },
        "fast_dllm_actual": dict(fast) if fast is not None else None,
        "preserving_extension": dict(extension) if extension is not None else None,
        "free_upper_bound": dict(free) if free is not None else None,
        "prompt_disjoint_matched_confidence": dict(selected_match) if selected_match is not None else None,
    }
    lines = [
        "# Exact top-1 VCCC oracle headroom audit",
        "",
        "## Core GO/NO-GO decision",
        "",
        f"**{decision['decision']}** — {decision['reason']}",
        "",
        decision["criterion"],
        "",
        "This decision is about whether an exponentially expensive **exact oracle** finds positive safe-parallelism headroom over the unchanged Fast-dLLM batch. It is not a deployment or wall-clock speed claim.",
        "",
        "## Direct answers (primary cohort, gamma=0)",
        "",
        f"The primary result uses the predeclared outcome-blind one-state-per-prompt sample (cap={config['sampling']['primary_prompt_cap']}); it is not a claim about every source trajectory state.",
        "",
        "1. Primary eligible state counts by requested pool size: " + "; ".join(
            f"M={m}: {sum(row.get('pool_status') == 'eligible' for row in rows)}/{len(rows)}"
            for m, rows in eligibility.items()
        ) + ".",
        "2. Fast-dLLM actual batch exact all-order failure rate at M=" + str(target_m) + ": " + (
            _fmt_rate(fast.get("batch_failure_prompt_macro_estimate")) if fast else "unavailable"
        ) + ".",
        "3. Fast-dLLM-preserving extension: conditional mean safe additions |T*|=" + (
            _fmt_value(extension.get("safe_added_capacity_conditional_on_B_pass_prompt_macro_estimate")) if extension else "unavailable"
        ) + ", conditional median=" + (
            _fmt_value(extension.get("median_safe_added_capacity_conditional_on_B_pass")) if extension else "unavailable"
        ) + ", and unconditional states with at least one safe extra=" + (
            _fmt_rate(extension.get("state_has_at_least_one_safe_extra_prompt_macro_estimate")) if extension else "unavailable"
        ) + ". Base-batch certificate failures are explicit, not coded as zero additions.",
        "4. Free exact VCCC oracle (upper bound only): mean commit count=" + (
            _fmt_value(free.get("committed_tokens_prompt_macro_estimate")) if free else "unavailable"
        ) + "; it may drop Fast-dLLM tokens and is never pooled with preserving-extension coverage.",
        "5. Idealized main-decoding NFE proxy (assuming a future cheap predictor reproduced the selected batch): Fast-dLLM=" + (
            _fmt_value(fast.get("ideal_main_decode_nfe_per_committed_token")) if fast else "unavailable"
        ) + " main forwards/token, preserving extension=" + (
            _fmt_value(extension.get("ideal_main_decode_nfe_per_committed_token")) if extension else "unavailable"
        ) + ". Exact verification forwards remain separately counted below.",
    ]
    if selected_match is None:
        lines.extend([
            "6. Prompt-disjoint matched-risk confidence comparison: unavailable (no calibration threshold met the predeclared zero-risk budget with usable evidence).",
        ])
    else:
        lines.extend([
            "6. Prompt-disjoint matched-risk confidence comparison: tau="
            + _fmt_value(selected_match.get("selected_confidence_threshold"), 3)
            + ", held-out confidence batch risk="
            + _fmt_rate(selected_match.get("confidence_test_risk_prompt_macro_estimate"))
            + ", held-out confidence mean commits="
            + _fmt_value(selected_match.get("confidence_test_commit_count_prompt_macro_estimate"))
            + ", held-out exact preserving-oracle mean commits="
            + _fmt_value(selected_match.get("oracle_test_commit_count_prompt_macro_estimate"))
            + ", paired held-out oracle-minus-confidence mean commits="
            + _fmt_value(selected_match.get("heldout_paired_oracle_minus_confidence_commit_count_prompt_macro_estimate"))
            + " (95% lower="
            + _fmt_value(selected_match.get("heldout_paired_oracle_minus_confidence_commit_count_prompt_clustered_ci95_low"))
            + "). The threshold was selected on calibration prompts only.",
        ])
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Every selected position keeps its fresh exact current full-vocabulary top-1 token. For a selected set S, the certificate checks every target i in S under every revealed subset A of S minus i, using raw-logit margins, deterministic argmax, `use_cache=False`, and an inclusive margin threshold with an explicit tie rule. A logged subset/order is an all-order query witness, not an observed decoder trajectory.",
        "",
        "Within one state, the M=4/6/8 pools are nested (B first, then deterministic fresh-confidence extras). The runner evaluates every subset of the largest eligible pool once, including the all-revealed no-target context, and smaller-M certificates read their exact prefix contexts from that same cache; it does not reuse a model KV cache or a past-key value.",
        "",
        "The actual Fast-dLLM comparison batch B is read from the original threshold-plus-argmax-fallback policy. It is never recomputed from this audit's fresh exact forward. Any stored B token that differs from the fresh exact top-1 is excluded and logged. Confidence-only tau policies intentionally have no fallback, so they are a threshold frontier rather than a duplicate of B.",
        "",
        "## Exact verification accounting",
        "",
        f"- Fresh base exact forwards: {metadata['forward_accounting']['base_forwards']}",
        f"- Additional exact subset-context forwards: {metadata['forward_accounting']['subset_context_forwards']}",
        f"- Total exact no-cache forwards: {metadata['forward_accounting']['exact_forwards']}; model batch calls: {metadata['forward_accounting']['model_batch_calls']}",
        f"- Runtime seconds: {metadata['runtime_seconds']}; peak VRAM MiB: {metadata['peak_vram_mib']}",
        "",
        "## Artifacts",
        "",
        "- `raw/state_screening.jsonl`: outcome-blind selection and every fresh-replay exclusion.",
        "- `tables/selection_accounting.csv` and `tables/exclusions.csv`: every source-screening, prompt-cap, fresh-replay, and M-stratum availability decision.",
        "- `raw/subset_contexts.jsonl` and `raw/subset_margin_queries.jsonl`: one exact context record per subset and the full-vocabulary scalar margin queries.",
        "- `tables/state_pool_selection.csv`: B, P, M-stratum eligibility, and common-eligible flags.",
        "- `tables/policy_state_results.csv` and `tables/token_certificate_witnesses.csv`: policy certificates, selected tokens, violations, and query witnesses.",
        "- `tables/headroom_summary.csv`, `tables/confidence_sweep.csv`, `tables/matched_risk_test.csv`, and `tables/oracle_cost.csv`: prompt-clustered estimates, prompt-disjoint frontier matching, and cost/NFE interpretation.",
    ])
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return direct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "exact_top1_vccc_oracle_headroom.yaml",
    )
    parser.add_argument("--probe-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-run", type=Path, required=True, help="Completed top1_dynamics_audit run directory (read-only).")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one outcome-blind primary state; never use it for the GO/NO-GO decision.")
    return parser.parse_args()


def make_output_root(base: Path, run_id: str | None, *, run_prefix: str) -> Path:
    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = base / f"{run_prefix}{identifier}"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing exact headroom audit: {root}")
    for name in ("raw", "tables", "figures"):
        (root / name).mkdir(parents=True, exist_ok=False) if name == "raw" else (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _invalid_pool_row(state: SelectedState, requested_size: int, *, reason: str) -> dict[str, Any]:
    return {
        **_source_common(state),
        "requested_pool_size": int(requested_size),
        "pool_status": reason,
        "effective_pool_size": None,
        "actual_fast_dllm_size": None,
        "actual_fast_dllm_positions": [],
        "actual_fast_dllm_token_ids": [],
        "pool_positions": [],
        "pool_token_ids": [],
        "pool_top1_probabilities": [],
        "pool_extra_positions_ranked_by_fresh_confidence": [],
    }


def _check_source_manifest(source_manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    allowed = {"completed", "observational_completed_exact_audit_blocked_resource_estimate"}
    if source_manifest.get("status") not in allowed:
        raise RuntimeError(f"Source run is not a completed top-1 evidence bundle: {source_manifest.get('status')!r}")
    source_fast_commit = source_manifest.get("fast_dllm_requested_commit")
    expected_fast_commit = config["model"]["fast_dllm_commit"]
    if source_fast_commit is not None and str(source_fast_commit) != str(expected_fast_commit):
        raise RuntimeError(
            "Source run Fast-dLLM pin differs from the headroom audit config: "
            f"source={source_fast_commit!r}, audit={expected_fast_commit!r}"
        )
    source_snapshot = source_manifest.get("model_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise RuntimeError("Source run lacks model_snapshot provenance required for frozen-model reuse.")
    if str(source_snapshot.get("model_name")) != str(config["model"]["name"]):
        raise RuntimeError("Source run model name differs from the exact headroom audit config.")
    if str(source_snapshot.get("requested_hf_revision")) != str(config["model"]["hf_revision"]):
        raise RuntimeError("Source run requested HF revision differs from the exact headroom audit config.")
    semantics = source_manifest.get("decoder_semantics")
    if not isinstance(semantics, Mapping):
        raise RuntimeError("Source run lacks decoder_semantics provenance required for normal-policy reuse.")
    if semantics.get("fallback_rule") != config["decoder_policy"]["fallback_rule"]:
        raise RuntimeError("Source run fallback-rule provenance differs from the exact headroom audit config.")


def _add_relative_nfe(summary_rows: list[dict[str, Any]]) -> None:
    """Add an ideal policy-only NFE ratio; never include oracle forwards."""

    baselines: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in summary_rows:
        if row.get("policy") == POLICY_FAST:
            baselines[(
                row.get("scope"), row.get("cohort"), row.get("requested_pool_size"), row.get("gamma"),
            )] = row
    for row in summary_rows:
        baseline = baselines.get((
            row.get("scope"), row.get("cohort"), row.get("requested_pool_size"), row.get("gamma"),
        ))
        current = _finite_float(row.get("ideal_commit_tokens_per_decode_step"))
        baseline_count = _finite_float(baseline.get("ideal_commit_tokens_per_decode_step")) if baseline else None
        if row.get("policy") == POLICY_EXTENSION:
            # Extension is defined only where B passes.  Its proxy comparison
            # must use B on precisely those same states rather than silently
            # borrowing an all-state Fast-dLLM average.
            baseline_count = _finite_float(
                row.get("actual_fast_dllm_tokens_on_same_B_pass_states_prompt_macro_estimate")
            )
        row["ideal_main_decode_nfe_ratio_vs_fast_dllm"] = (
            None if current is None or baseline_count is None or current <= 0.0 else baseline_count / current
        )
        row["ideal_nfe_proxy_assumption"] = (
            "future cheap certificate predictor reproduces this policy's selected current-top1 set; "
            "exponential exact verification forwards excluded"
        )


def main() -> None:
    args = parse_args()
    wall_started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Headroom config must be a mapping")
    probe_root = args.probe_root.resolve()
    source_root = args.source_run.resolve()
    source_manifest_path = source_root / "run_manifest.json"
    trajectories_path = source_root / "raw" / "trajectories.jsonl"
    state_positions_path = source_root / "raw" / "state_positions.jsonl"
    if not source_manifest_path.exists() or not trajectories_path.exists() or not state_positions_path.exists():
        raise FileNotFoundError("Source run must contain run_manifest.json, raw/trajectories.jsonl, and raw/state_positions.jsonl")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _check_source_manifest(source_manifest, config)
    output_base = Path(args.output_root) if args.output_root else probe_root / str(config["storage"]["output_root"])
    output_root = make_output_root(output_base, args.run_id, run_prefix=str(config["storage"]["run_prefix"]))
    shutil.copy2(args.config, output_root / "config.yaml")
    write_json(output_root / "source_manifest.json", source_manifest)

    trajectories = read_jsonl(trajectories_path)
    original_policy_positions = natural_policy_rows_by_state(state_positions_path)
    selected, screening_rows, source_selection_counts = select_states(trajectories, config, smoke=bool(args.smoke))
    if not selected:
        write_jsonl(output_root / "raw" / "state_screening.jsonl", screening_rows)
        raise RuntimeError("No primary source states passed the outcome-blind source schema screen.")

    from scripts.collect_states import mask_token_id
    from scripts.run_top1_dynamics_audit import _model_snapshot_metadata, load_model
    import torch

    model, tokenizer, _device = load_model(config, probe_root)
    current_model_snapshot = _model_snapshot_metadata(model, tokenizer, config)
    source_model_snapshot = source_manifest["model_snapshot"]
    source_commit_hint = source_model_snapshot.get("resolved_hf_commit_hint")
    current_commit_hint = current_model_snapshot.get("resolved_hf_commit_hint")
    if source_commit_hint and current_commit_hint and str(source_commit_hint) != str(current_commit_hint):
        raise RuntimeError(
            "The currently loaded Hugging Face snapshot differs from the source exact-trajectory snapshot: "
            f"source={source_commit_hint!r}, current={current_commit_hint!r}"
        )
    model_snapshot_replay_verification = {
        "source": source_model_snapshot,
        "current": current_model_snapshot,
        "status": (
            "resolved_commit_match"
            if source_commit_hint and current_commit_hint and str(source_commit_hint) == str(current_commit_hint)
            else "requested_revision_match_resolved_commit_hint_unavailable"
        ),
    }
    mask_id = mask_token_id(model, tokenizer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    accounting = ForwardAccounting(started=time.perf_counter())
    requested_pool_sizes = tuple(int(value) for value in config["sampling"]["pool_sizes"])
    if tuple(sorted(set(requested_pool_sizes))) != requested_pool_sizes:
        raise ValueError("sampling.pool_sizes must be ascending and unique so nested forward reuse is valid")
    all_pool_rows: list[dict[str, Any]] = []
    all_base_rows: list[dict[str, Any]] = []
    all_context_rows: list[dict[str, Any]] = []
    all_query_rows: list[dict[str, Any]] = []
    all_policy_rows: list[dict[str, Any]] = []
    all_token_rows: list[dict[str, Any]] = []
    tie_tolerance = float(config["certificate"]["tie_tolerance"])

    for ordinal, state in enumerate(selected, 1):
        record = state.record
        common = _source_common(state)
        try:
            mask_positions_in_policy_order = [int(value) for value in record["mask_positions"]]
            mask_positions = set(mask_positions_in_policy_order)
        except (KeyError, TypeError, ValueError):
            mask_positions_in_policy_order = []
            mask_positions = set()
        if not mask_positions or len(mask_positions) != len(mask_positions_in_policy_order):
            reason = "fresh_replay_invalid_or_empty_mask_positions"
            screening_rows.append({**common, "selection_stage": "fresh_exact_replay", "status": "excluded", "exclusion_reason": reason})
            all_pool_rows.extend(_invalid_pool_row(state, m, reason=reason) for m in requested_pool_sizes)
            continue
        try:
            base_input_ids = torch.tensor([record["token_sequence"]], device=next(model.parameters()).device, dtype=torch.long)
        except (KeyError, TypeError, ValueError) as error:
            reason = "fresh_replay_cannot_construct_input_ids"
            screening_rows.append({**common, "selection_stage": "fresh_exact_replay", "status": "excluded", "exclusion_reason": reason, "detail": str(error)})
            all_pool_rows.extend(_invalid_pool_row(state, m, reason=reason) for m in requested_pool_sizes)
            continue
        out_of_bounds = [position for position in mask_positions if position < 0 or position >= int(base_input_ids.shape[1])]
        non_masked = [position for position in mask_positions if position not in out_of_bounds and int(base_input_ids[0, position].item()) != mask_id]
        if out_of_bounds or non_masked:
            reason = "fresh_replay_mask_position_out_of_bounds" if out_of_bounds else "fresh_replay_claimed_mask_is_not_mask_token"
            screening_rows.append({
                **common,
                "selection_stage": "fresh_exact_replay",
                "status": "excluded",
                "exclusion_reason": reason,
                "out_of_bounds_positions": out_of_bounds,
                "non_masked_positions": non_masked,
            })
            all_pool_rows.extend(_invalid_pool_row(state, m, reason=reason) for m in requested_pool_sizes)
            del base_input_ids
            continue
        base_logits = exact_logits_batched(model, base_input_ids)
        accounting.base_forwards += 1
        accounting.model_batch_calls += 1
        assignments = _base_assignments(base_logits, mask_positions, tie_tolerance=tie_tolerance)
        all_base_rows.extend({
            **common,
            "position": position,
            "fresh_exact_top1_token_id": payload["token_id"],
            "fresh_exact_top1_probability": payload["top1_probability"],
            "fresh_exact_logit_margin": payload["logit_margin"],
            "fresh_exact_top2_token_id": payload["top2_token_id"],
            "fresh_exact_top2_probability": payload["top2_probability"],
            "fresh_exact_top1_matches_assignment": payload["top1_matches_assignment"],
            "fresh_exact_is_logit_tie": payload["is_logit_tie"],
        } for position, payload in sorted(assignments.items()))
        actual_positions, error = _validate_actual_batch(
            state,
            mask_positions=mask_positions,
            mask_positions_in_policy_order=mask_positions_in_policy_order,
            fresh_assignments=assignments,
            natural_position_rows=original_policy_positions.get(str(record["state_key"])),
            config=config,
        )
        if error is not None or actual_positions is None:
            reason = error or "fresh_replay_unknown_actual_batch_error"
            screening_rows.append({
                **common,
                "selection_stage": "fresh_exact_replay",
                "status": "excluded",
                "exclusion_reason": reason,
                "actual_batch_fresh_token_mismatches": _actual_batch_fresh_token_mismatches(record, assignments),
            })
            all_pool_rows.extend(_invalid_pool_row(state, m, reason=reason) for m in requested_pool_sizes)
            del base_logits, base_input_ids
            continue
        screening_rows.append({
            **common,
            "selection_stage": "fresh_exact_replay",
            "status": "retained_after_fresh_exact_top1_validation",
            "exclusion_reason": None,
            "actual_fast_dllm_positions": actual_positions,
            "actual_fast_dllm_size": len(actual_positions),
        })
        eligible_pools: dict[int, dict[str, Any]] = {}
        for pool_size in requested_pool_sizes:
            pool_row, pool_error = _pool_for_stratum(
                state,
                actual_batch_positions=actual_positions,
                assignments=assignments,
                requested_size=pool_size,
            )
            all_pool_rows.append(pool_row)
            if pool_error is None:
                eligible_pools[pool_size] = pool_row
        if not eligible_pools:
            del base_logits, base_input_ids
            print(json.dumps({"stage": "fresh_replay", "state": ordinal, "total_states": len(selected), "state_key": record["state_key"], "status": "no_eligible_pool", "forwards": accounting.exact_forwards}))
            continue
        max_pool_size, max_pool = max(eligible_pools.items(), key=lambda item: (int(item[1]["effective_pool_size"]), item[0]))
        max_positions = [int(value) for value in max_pool["pool_positions"]]
        max_tokens = [int(value) for value in max_pool["pool_token_ids"]]
        for pool_size, pool in eligible_pools.items():
            positions = [int(value) for value in pool["pool_positions"]]
            if max_positions[: len(positions)] != positions:
                raise RuntimeError(f"{record['state_key']}: nested M strata do not share the required B-plus-confidence pool prefix")
        margins, context_rows, query_rows = _evaluate_max_pool_contexts(
            model,
            state,
            base_input_ids=base_input_ids,
            base_logits=base_logits,
            pool_positions=max_positions,
            pool_token_ids=max_tokens,
            batch_size=int(config["execution"]["subset_batch_size"]),
            tie_tolerance=tie_tolerance,
            accounting=accounting,
        )
        all_context_rows.extend(context_rows)
        all_query_rows.extend(query_rows)
        for pool_size, pool in eligible_pools.items():
            positions = [int(value) for value in pool["pool_positions"]]
            tokens = [int(value) for value in pool["pool_token_ids"]]
            probabilities = [float(value) for value in pool["pool_top1_probabilities"]]
            policy_rows, token_rows = evaluate_policies_for_pool(
                state,
                requested_pool_size=pool_size,
                positions=positions,
                token_ids=tokens,
                probabilities=probabilities,
                actual_batch_positions=actual_positions,
                margins=margins,
                config=config,
            )
            all_policy_rows.extend(policy_rows)
            all_token_rows.extend(token_rows)
        del base_logits
        print(json.dumps({
            "stage": "subset_grid",
            "state": ordinal,
            "total_states": len(selected),
            "state_key": record["state_key"],
            "max_requested_pool_size": max_pool_size,
            "max_effective_pool_size": len(max_positions),
            "forwards": accounting.exact_forwards,
        }))

    required_pool_set = set(requested_pool_sizes)
    eligible_by_state: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in all_pool_rows:
        if row.get("pool_status") == "eligible":
            eligible_by_state[(str(row["state_key"]), str(row["cohort"]))].add(int(row["requested_pool_size"]))
    common_eligible = {
        key for key, pools in eligible_by_state.items() if pools == required_pool_set
    }
    for rows in (all_pool_rows, all_policy_rows, all_token_rows, all_context_rows, all_query_rows, all_base_rows):
        for row in rows:
            row["common_eligible_M4_M6_M8"] = (str(row.get("state_key")), str(row.get("cohort"))) in common_eligible

    summary_rows = summarize_policies(all_policy_rows, all_token_rows, config)
    _add_relative_nfe(summary_rows)
    sweep_rows = confidence_sweep_rows(summary_rows)
    matched_rows = matched_risk_rows(all_policy_rows, all_token_rows, config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_mib: float | None = float(torch.cuda.max_memory_allocated() / 2**20)
    else:
        peak_vram_mib = None
    runtime_seconds = time.perf_counter() - wall_started
    accounting_payload = {
        "base_forwards": accounting.base_forwards,
        "subset_context_forwards": accounting.subset_context_forwards,
        "exact_forwards": accounting.exact_forwards,
        "model_batch_calls": accounting.model_batch_calls,
    }
    context_count_by_state = Counter(str(row["state_key"]) for row in all_context_rows)
    cost_rows = [{
        "exact_cache_policy": "use_cache_false",
        "base_forwards": accounting.base_forwards,
        "subset_context_forwards": accounting.subset_context_forwards,
        "total_exact_forwards": accounting.exact_forwards,
        "model_batch_calls": accounting.model_batch_calls,
        "evaluated_state_count": len(context_count_by_state),
        "mean_exact_contexts_per_evaluated_state": (
            math.fsum(context_count_by_state.values()) / len(context_count_by_state) if context_count_by_state else None
        ),
        "runtime_seconds": runtime_seconds,
        "peak_vram_mib": peak_vram_mib,
        "interpretation": "Counterfactual verification cost only; never an end-to-end decoding speedup.",
    }]
    exclusions = [
        row
        for row in screening_rows
        if row.get("status") not in {"selected_pending_fresh_exact_replay", "retained_after_fresh_exact_top1_validation"}
    ] + [
        row for row in all_pool_rows if row.get("pool_status") != "eligible"
    ]
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_smoke" if args.smoke else "completed",
        "audit_name": "exact_top1_vccc_oracle_headroom",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_root": str(source_root),
        "source_run_status": source_manifest.get("status"),
        "source_commit": source_manifest.get("source_git_commit"),
        "frozen_model": config["model"]["name"],
        "dtype": config["model"]["dtype"],
        "model_snapshot_replay_verification": model_snapshot_replay_verification,
        "actual_decoder_policy": config["decoder_policy"]["required_name"],
        "actual_decoder_policy_unchanged": True,
        "exact_cache_policy": "use_cache=False",
        "margin_definition": "assigned raw logit minus maximum full-vocabulary competitor raw logit",
        "tie_policy": "inclusive margin threshold plus deterministic torch.argmax must equal fixed top1 assignment",
        "source_selection": source_selection_counts,
        "selected_state_count": len(selected),
        "selected_primary_state_count": sum(state.cohort == "primary" for state in selected),
        "selected_hard_state_count": sum(state.cohort == "hard_exploratory" for state in selected),
        "fresh_replay_retained_state_count": sum(row.get("status") == "retained_after_fresh_exact_top1_validation" for row in screening_rows),
        "common_eligible_M4_M6_M8_state_count": len(common_eligible),
        "forward_accounting": accounting_payload,
        "runtime_seconds": runtime_seconds,
        "peak_vram_mib": peak_vram_mib,
        "ideal_nfe_interpretation": "A hypothetical cheap certificate predictor reproduces selected sets; exact verification forwards are excluded and no wall-clock speedup is claimed.",
        "free_oracle_interpretation": "Upper bound only; it can omit actual Fast-dLLM B positions and is not safe-extension coverage.",
        "prompt_split": {
            "calibration_percent": config["matching"]["calibration_percent"],
            "salt": config["matching"]["prompt_split_salt"],
            "selection": "confidence tau chosen only on calibration prompts then evaluated on disjoint test prompts",
        },
    }
    write_jsonl(output_root / "raw" / "state_screening.jsonl", screening_rows)
    write_jsonl(output_root / "raw" / "base_assignments.jsonl", all_base_rows)
    write_jsonl(output_root / "raw" / "subset_contexts.jsonl", all_context_rows)
    write_jsonl(output_root / "raw" / "subset_margin_queries.jsonl", all_query_rows)
    write_jsonl(output_root / "raw" / "policy_state_results.jsonl", all_policy_rows)
    write_jsonl(output_root / "raw" / "token_certificate_witnesses.jsonl", all_token_rows)
    tables: dict[str, Sequence[Mapping[str, Any]]] = {
        "selection_accounting": screening_rows,
        "state_pool_selection": all_pool_rows,
        "policy_state_results": all_policy_rows,
        "token_certificate_witnesses": all_token_rows,
        "headroom_summary": summary_rows,
        "confidence_sweep": sweep_rows,
        "matched_risk_test": matched_rows,
        "oracle_cost": cost_rows,
        "exclusions": exclusions,
    }
    for name, rows in tables.items():
        write_csv(output_root / "tables" / f"{name}.csv", rows)
    figures = write_figures(
        output_root / "figures",
        summary_rows=summary_rows,
        policy_rows=all_policy_rows,
        accounting=accounting,
        config=config,
    )
    metadata["figure_paths"] = figures
    direct = write_report(
        output_root,
        pool_rows=all_pool_rows,
        summary_rows=summary_rows,
        matched_rows=matched_rows,
        metadata=metadata,
        config=config,
    )
    metadata["direct_answer_summary"] = direct
    write_json(output_root / "run_metadata.json", metadata)
    write_json(output_root / "summary.json", {"status": metadata["status"], "metadata": metadata, "direct_answers": direct})
    print(json.dumps({
        "status": metadata["status"],
        "output_root": str(output_root),
        "forwards": accounting.exact_forwards,
        "runtime_seconds": runtime_seconds,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
