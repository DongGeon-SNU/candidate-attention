#!/usr/bin/env python3
"""Exact frozen-model oracle audit for Universal and Existential VCCC.

This runner *never* changes a decoder trajectory.  It reads exact state records
from a completed top-1-dynamics run, fixes each policy-selected assignment to
the current exact top-1 token, and measures every requested counterfactual
context with an independent full-vocabulary ``use_cache=False`` forward.

The output is intentionally a new timestamped directory.  Source raw evidence
and older top-1 reports are read-only inputs.
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

from src.branching import Candidate, exact_forward
from src.top1_reporting import (  # noqa: E402
    binary_ranking_metrics,
    prompt_clustered_bootstrap,
    write_csv,
    write_jsonl,
)
from src.vccc_oracle import (  # noqa: E402
    classify_certificate,
    directional_dependency_graph,
    fixed_confidence_bin,
    fixed_numeric_bin,
    order_bottleneck,
    pairwise_certificate_margin,
    pairwise_safe,
    quantile,
    residual_rows,
)


SCHEMA_VERSION = 1
PRIMARY_GAMMAS = (0.0, 0.25, 0.5, 1.0)
CONFIDENCE_BINS = ("p1<0.7", "0.7<=p1<0.9", "0.9<=p1<0.95", "0.95<=p1<0.99", "p1>=0.99")


@dataclass(frozen=True)
class SelectedState:
    record: Mapping[str, Any]
    cohort: str
    selection_reason: str
    candidates: tuple[Candidate, ...]


@dataclass
class ForwardAccounting:
    batch_calls: int = 0
    branch_forwards: int = 0
    subset_branch_forwards: int = 0
    polarity_branch_forwards: int = 0
    started: float = 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required raw artifact: {path}")
    result: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object in {path}:{number}")
        result.append(value)
    return result


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - target setup installs pyarrow
        raise RuntimeError("pyarrow is required to audit existing Parquet evidence") from error
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def stable_key(*parts: Any) -> str:
    packed = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def state_candidates(record: Mapping[str, Any]) -> tuple[Candidate, ...] | None:
    """Return policy anchors only when they equal exact current top-1 values."""

    summaries = record.get("position_summaries")
    anchors = record.get("actual_committed_anchors")
    if not isinstance(summaries, Mapping) or not isinstance(anchors, list) or len(anchors) < 2:
        return None
    candidates: list[Candidate] = []
    positions: set[int] = set()
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            return None
        position = int(anchor["position"])
        token_id = int(anchor["token_id"])
        summary = summaries.get(str(position))
        if not isinstance(summary, Mapping) or int(summary.get("top1_token_id", -1)) != token_id:
            # Version U/E require y_i to be *current exact* top-1.  Do not
            # silently treat a natural-vs-exact mismatch as a certificate.
            return None
        if position in positions:
            return None
        positions.add(position)
        candidates.append(Candidate(position, token_id))
    return tuple(sorted(candidates))


def _min_p1_gap(record: Mapping[str, Any], candidates: Sequence[Candidate]) -> float:
    summaries = record["position_summaries"]
    return min(float(summaries[str(item.position)].get("p1_minus_p2", math.inf)) for item in candidates)


def source_order_sensitive_ids(source_root: Path) -> set[str]:
    path = source_root / "raw" / "order_replay_state_summaries.jsonl"
    if not path.exists():
        return set()
    rows = read_jsonl(path)
    return {
        str(row["state_key"])
        for row in rows
        if bool(row.get("adaptive_final_assignment_order_sensitive"))
        or bool(row.get("adaptive_decision_trajectory_order_sensitive"))
    }


def choose_states(
    trajectories: Sequence[Mapping[str, Any]], config: Mapping[str, Any], source_root: Path
) -> tuple[list[SelectedState], list[dict[str, Any]]]:
    sample = config["sampling"]
    sizes = tuple(int(item) for item in sample["set_sizes"])
    primary_cap = int(sample["primary_states_per_size"])
    hard_cap = int(sample["hard_states_per_size"])
    existing_order_sensitive = source_order_sensitive_ids(source_root)
    candidates_by_size: dict[int, list[tuple[Mapping[str, Any], tuple[Candidate, ...]]]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    for record in trajectories:
        if record.get("setting") != "primary" or float(record.get("threshold", -1.0)) != float(sample["primary_threshold"]):
            continue
        anchors = record.get("actual_committed_anchors")
        if not isinstance(anchors, list):
            exclusions["missing_policy_anchors"] += 1
            continue
        if len(anchors) not in sizes:
            exclusions["unsupported_set_size"] += 1
            continue
        current = state_candidates(record)
        if current is None:
            exclusions["natural_exact_top1_mismatch_or_invalid"] += 1
            continue
        candidates_by_size[len(current)].append((record, current))

    selected: list[SelectedState] = []
    diagnostics: list[dict[str, Any]] = []
    for size in sizes:
        available = candidates_by_size.get(size, [])
        policy = sorted(available, key=lambda item: stable_key("policy", item[0]["state_key"]))[:primary_cap]
        chosen_ids = {str(record["state_key"]) for record, _items in policy}
        for record, current in policy:
            selected.append(SelectedState(record, "primary_policy", "deterministic_policy_state_sample", current))
        # Hard cohort is deliberately separate: low margin first, then states
        # previously found order-sensitive, with deterministic ties.
        remaining = [(record, current) for record, current in available if str(record["state_key"]) not in chosen_ids]
        hard = sorted(
            remaining,
            key=lambda item: (
                0 if str(item[0]["state_key"]) in existing_order_sensitive else 1,
                _min_p1_gap(item[0], item[1]),
                stable_key("hard", item[0]["state_key"]),
            ),
        )[:hard_cap]
        for record, current in hard:
            reason = "existing_order_sensitive" if str(record["state_key"]) in existing_order_sensitive else "smallest_policy_p1_minus_p2"
            selected.append(SelectedState(record, "hard_order_sensitive", reason, current))
        diagnostics.append(
            {
                "set_size": size,
                "available_eligible_states": len(available),
                "primary_selected_states": len(policy),
                "hard_selected_states": len(hard),
                "primary_unique_prompts": len({str(record["prompt_id"]) for record, _ in policy}),
                "hard_unique_prompts": len({str(record["prompt_id"]) for record, _ in hard}),
                "primary_target_states": primary_cap,
                "hard_target_states": hard_cap,
            }
        )
    diagnostics.extend({"kind": "exclusion", "reason": key, "count": value} for key, value in sorted(exclusions.items()))
    return selected, diagnostics


def exact_logits_batched(model: Any, input_ids: Any) -> Any:
    """No-cache exact forward with a fixed context and fixed position IDs."""

    import torch

    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
    return exact_forward(model, input_ids, attention_mask=attention_mask, position_ids=position_ids)


def margin_summary(logits: Any, position: int, token_id: int, *, tie_tolerance: float) -> dict[str, Any]:
    """Full-vocabulary top-1 and assigned-token margin, with no top-K renorm."""

    import torch

    row = logits[int(position)].float()
    assigned_logit = float(row[int(token_id)].item())
    competitor = row.clone()
    competitor[int(token_id)] = -torch.inf
    competitor_logit, competitor_token = torch.max(competitor, dim=0)
    logit_margin = assigned_logit - float(competitor_logit.item())
    probabilities = torch.softmax(row, dim=-1)
    assigned_probability = float(probabilities[int(token_id)].item())
    competitor_probability = float(probabilities[int(competitor_token.item())].item())
    deterministic_top1 = int(torch.argmax(row).item())
    return {
        "logit_margin": logit_margin,
        "probability_margin": assigned_probability - competitor_probability,
        "assigned_probability": assigned_probability,
        "competitor_probability": competitor_probability,
        "competitor_token_id": int(competitor_token.item()),
        "top1_token_id": deterministic_top1,
        "top1_matches_assignment": deterministic_top1 == int(token_id),
        "is_logit_tie": abs(logit_margin) <= float(tie_tolerance),
    }


def topk_summary(logits: Any, position: int, maximum_k: int) -> dict[str, Any]:
    import torch

    row = logits[int(position)].float()
    probabilities = torch.softmax(row, dim=-1)
    values, indices = torch.topk(probabilities, k=min(int(maximum_k), int(probabilities.numel())))
    return {
        "top_token_ids": [int(value) for value in indices.detach().cpu().tolist()],
        "top_probabilities": [float(value) for value in values.detach().cpu().tolist()],
        "top1_probability": float(values[0].item()),
        "p1_minus_p2": float((values[0] - values[1]).item()) if int(values.numel()) > 1 else None,
        "logit_margin": float((row[indices[0]] - row[indices[1]]).item()) if int(indices.numel()) > 1 else None,
    }


def evaluate_subset_state(
    model: Any,
    selected: SelectedState,
    *,
    mask_token_id: int,
    batch_size: int,
    tie_tolerance: float,
    accounting: ForwardAccounting,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, dict[str, Any]]]:
    """Measure every subset state once, then reduce all target margins on GPU."""

    import torch

    record = selected.record
    candidates = selected.candidates
    size = len(candidates)
    base = torch.tensor([record["token_sequence"]], device=next(model.parameters()).device, dtype=torch.long)
    for candidate in candidates:
        if int(base[0, candidate.position].item()) != int(mask_token_id):
            raise RuntimeError(f"{record['state_key']}: policy candidate position is not a mask")
    branch_masks = list(range((1 << size) - 1))  # full reveal has no queried target.
    margins: dict[tuple[int, int], dict[str, Any]] = {}
    base_topk: dict[int, dict[str, Any]] = {}
    for start in range(0, len(branch_masks), batch_size):
        masks = branch_masks[start : start + batch_size]
        branches = base.expand(len(masks), -1).clone()
        for row, mask in enumerate(masks):
            for index, candidate in enumerate(candidates):
                if mask & (1 << index):
                    branches[row, candidate.position] = candidate.token_id
        logits = exact_logits_batched(model, branches)
        accounting.batch_calls += 1
        accounting.branch_forwards += len(masks)
        accounting.subset_branch_forwards += len(masks)
        for row, mask in enumerate(masks):
            for target_index, target in enumerate(candidates):
                if mask & (1 << target_index):
                    continue
                payload = margin_summary(logits[row], target.position, target.token_id, tie_tolerance=tie_tolerance)
                payload.update({"target_index": target_index, "revealed_mask": mask})
                margins[(target_index, mask)] = payload
                if mask == 0:
                    base_topk[target_index] = topk_summary(logits[row], target.position, 16)
        del logits, branches
    del base
    return margins, base_topk


def _order_positions(order: Sequence[int] | None, candidates: Sequence[Candidate]) -> list[int] | None:
    return None if order is None else [int(candidates[index].position) for index in order]


def _row_common(selected: SelectedState) -> dict[str, Any]:
    record = selected.record
    return {
        "state_key": record["state_key"],
        "cohort": selected.cohort,
        "selection_reason": selected.selection_reason,
        "dataset": record.get("dataset"),
        "prompt_id": record.get("prompt_id"),
        "prompt_index": record.get("prompt_index"),
        "source_step": record.get("step"),
        "generated_mask_ratio": record.get("generated_mask_ratio"),
        "sequence_length": len(record["token_sequence"]),
        "set_size": len(selected.candidates),
        "assignment_positions": [item.position for item in selected.candidates],
        "assignment_token_ids": [item.token_id for item in selected.candidates],
    }


def certificate_rows_for_state(
    selected: SelectedState,
    margins: Mapping[tuple[int, int], Mapping[str, Any]],
    base_topk: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce A/B/D/E scalar evidence from an already exact subset grid."""

    sample = config["sampling"]
    tolerance = float(config["certificate"]["tie_tolerance"])
    common = _row_common(selected)
    states: list[dict[str, Any]] = []
    witnesses: list[dict[str, Any]] = []
    final_loo: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    size = len(selected.candidates)
    for gamma in sample["margin_thresholds"]:
        result = classify_certificate(margins, size, float(gamma), tolerance=tolerance)
        pair_safe_at_gamma = pairwise_safe(margins, size, float(gamma), tolerance=tolerance)
        pairwise_margin = pairwise_certificate_margin(margins, size)
        graph = directional_dependency_graph(margins, size, float(gamma), tolerance=tolerance)
        graph_order = graph["topological_order"]
        graph_bottleneck = order_bottleneck(margins, graph_order) if graph_order is not None else None
        states.append(
            {
                **common,
                "gamma": float(gamma),
                "category": result.category,
                "all_pass": result.all_pass,
                "existential_pass": result.existential_pass,
                "existential_plus_final_loo_pass": bool(result.existential_pass and result.final_loo_pass),
                "final_loo_pass": result.final_loo_pass,
                "all_order_logit_certificate": result.all_order_margin,
                "existential_logit_certificate": result.existential_margin,
                "final_loo_logit_certificate": result.final_loo_margin,
                "pairwise_lower_bound_certificate": pairwise_margin,
                "pair_safe": pair_safe_at_gamma,
                "pair_safe_exact_set_unsafe": bool(pair_safe_at_gamma and not result.all_pass),
                "tie_query_count": result.tie_query_count,
                "missing_query_count": result.missing_query_count,
                "graph_edge_count": graph["edge_count"],
                "graph_edge_density": graph["edge_density"],
                "graph_is_dag": graph["is_dag"],
                "graph_has_cycle": graph["has_cycle"],
                "graph_scc_count": graph["scc_count"],
                "graph_largest_scc_size": graph["largest_scc_size"],
                "graph_topological_order_is_exact_witness": (
                    None if graph_order is None else order_bottleneck(margins, graph_order) is not None
                    and bool(order_bottleneck(margins, graph_order) >= float(gamma) - tolerance)
                ),
                "graph_bottleneck_margin": graph_bottleneck,
                "graph_vs_exact_bottleneck_regret": (
                    None if graph_bottleneck is None or result.existential_margin is None
                    else result.existential_margin - graph_bottleneck
                ),
            }
        )
        witnesses.append(
            {
                **common,
                "gamma": float(gamma),
                "category": result.category,
                "existential_pass": result.existential_pass,
                "witness_order_indices": list(result.witness_order) if result.witness_order is not None else None,
                "witness_order_positions": _order_positions(result.witness_order, selected.candidates),
                "maximum_bottleneck_order_indices": list(result.maximum_bottleneck_order) if result.maximum_bottleneck_order is not None else None,
                "maximum_bottleneck_order_positions": _order_positions(result.maximum_bottleneck_order, selected.candidates),
                "maximum_bottleneck_margin": result.existential_margin,
                "graph_topological_order_indices": list(graph_order) if graph_order is not None else None,
                "graph_topological_order_positions": _order_positions(graph_order, selected.candidates),
            }
        )
    # Target-level LOO and directed singleton responses have one exact context
    # each.  These are the denominators for high-confidence/local-coverage B.
    for target_index, target in enumerate(selected.candidates):
        base = margins[(target_index, 0)]
        before = base_topk[target_index]
        confidence = before["top1_probability"]
        p1_gap = before["p1_minus_p2"]
        logit_gap = before["logit_margin"]
        full_without_target = ((1 << size) - 1) ^ (1 << target_index)
        loo = margins[(target_index, full_without_target)]
        final_loo.append(
            {
                **common,
                "target_index": target_index,
                "target_position": target.position,
                "target_token_id": target.token_id,
                "base_top1_probability": confidence,
                "base_p1_minus_p2": p1_gap,
                "base_logit_margin": logit_gap,
                "final_loo_logit_margin": loo["logit_margin"],
                "final_loo_probability_margin": loo.get("probability_margin"),
                "final_loo_reversal": not bool(loo["top1_matches_assignment"]),
                "final_loo_top1_token_id": loo["top1_token_id"],
                "final_loo_margin_shift": loo["logit_margin"] - base["logit_margin"],
                "confidence_bin": fixed_confidence_bin(confidence),
                "p1_minus_p2_bin": fixed_numeric_bin(p1_gap, sample["p1_minus_p2_bin_edges"], label="p1-p2"),
                "logit_margin_bin": fixed_numeric_bin(logit_gap, sample["logit_margin_bin_edges"], label="logit_margin"),
                # There is no distinct valid query that reveals the target
                # itself; in VCCC it remains masked.  This explicit alias
                # prevents interpreting 'full set' as a different intervention.
                "full_set_excluding_target_same_as_final_loo": True,
            }
        )
        for source_index, source in enumerate(selected.candidates):
            if source_index == target_index:
                continue
            singleton = margins[(target_index, 1 << source_index)]
            after_top1 = int(singleton["top1_token_id"])
            before_tokens = list(before["top_token_ids"])
            interventions.append(
                {
                    **common,
                    "target_index": target_index,
                    "target_position": target.position,
                    "target_token_id": target.token_id,
                    "source_index": source_index,
                    "source_position": source.position,
                    "source_token_id": source.token_id,
                    "base_top1_probability": confidence,
                    "base_p1_minus_p2": p1_gap,
                    "base_logit_margin": logit_gap,
                    "base_margin": base["logit_margin"],
                    "after_margin": singleton["logit_margin"],
                    "margin_shift": singleton["logit_margin"] - base["logit_margin"],
                    "counterfactual_reversal": not bool(singleton["top1_matches_assignment"]),
                    "after_top1_token_id": after_top1,
                    "after_top1_original_rank": (
                        before_tokens.index(after_top1) + 1 if after_top1 in before_tokens else len(before_tokens) + 1
                    ),
                    "confidence_bin": fixed_confidence_bin(confidence),
                    "p1_minus_p2_bin": fixed_numeric_bin(p1_gap, sample["p1_minus_p2_bin_edges"], label="p1-p2"),
                    "logit_margin_bin": fixed_numeric_bin(logit_gap, sample["logit_margin_bin_edges"], label="logit_margin"),
                    "polarity": "supportive" if singleton["logit_margin"] - base["logit_margin"] >= 0.0 else "destructive",
                    "policy_selected_target": True,
                    "high_confidence_target": confidence >= 0.9,
                    **{f"after_top1_in_pre_top{k}": after_top1 in before_tokens[:k] for k in sample["local_topk_values"]},
                }
            )
    for row in residual_rows(margins, size):
        target = selected.candidates[int(row["target_index"])]
        row.update({**common, "target_position": target.position, "target_token_id": target.token_id})
        residuals.append(row)
    return states, witnesses, final_loo, interventions, residuals


def _candidate_values(topk: Mapping[str, Any], *, mode: str, config: Mapping[str, Any]) -> list[int]:
    tokens = [int(token) for token in topk["top_token_ids"]]
    settings = config["candidate_polarity"]
    if mode == "fixed_k4":
        return tokens[:4]
    if mode == "fixed_k8":
        return tokens[:8]
    rho = float(settings["adaptive_mass"])
    cap = int(settings["adaptive_k_max"])
    values: list[int] = []
    mass = 0.0
    for token, probability in zip(tokens[:cap], topk["top_probabilities"]):
        values.append(token)
        mass += float(probability)
        if mass >= rho:
            break
    return values


def evaluate_polarity_pair(
    model: Any,
    selected: SelectedState,
    first_index: int,
    second_index: int,
    *,
    mask_token_id: int,
    batch_size: int,
    tie_tolerance: float,
    config: Mapping[str, Any],
    accounting: ForwardAccounting,
) -> list[dict[str, Any]]:
    """Measure KxK value-polarity matrices with K directional forwards, not K²."""

    import torch

    first, second = selected.candidates[first_index], selected.candidates[second_index]
    base = torch.tensor([selected.record["token_sequence"]], device=next(model.parameters()).device, dtype=torch.long)
    base_logits = exact_logits_batched(model, base)
    accounting.batch_calls += 1
    accounting.branch_forwards += 1
    accounting.polarity_branch_forwards += 1
    before = {
        first_index: topk_summary(base_logits[0], first.position, 16),
        second_index: topk_summary(base_logits[0], second.position, 16),
    }
    modes = ("fixed_k4", "fixed_k8", "adaptive_mass")
    output: list[dict[str, Any]] = []
    common = _row_common(selected)
    for source_index, target_index in ((first_index, second_index), (second_index, first_index)):
        source, target = selected.candidates[source_index], selected.candidates[target_index]
        source_values_by_mode = {mode: _candidate_values(before[source_index], mode=mode, config=config) for mode in modes}
        target_values_by_mode = {mode: _candidate_values(before[target_index], mode=mode, config=config) for mode in modes}
        union_source_values = sorted({value for values in source_values_by_mode.values() for value in values})
        source_logits: dict[int, Any] = {}
        for start in range(0, len(union_source_values), batch_size):
            tokens = union_source_values[start : start + batch_size]
            branches = base.expand(len(tokens), -1).clone()
            for row, token in enumerate(tokens):
                branches[row, source.position] = token
            logits = exact_logits_batched(model, branches)
            accounting.batch_calls += 1
            accounting.branch_forwards += len(tokens)
            accounting.polarity_branch_forwards += len(tokens)
            for row, token in enumerate(tokens):
                source_logits[token] = logits[row].detach()
            del logits, branches
        for mode in modes:
            source_values = source_values_by_mode[mode]
            target_values = target_values_by_mode[mode]
            for source_token in source_values:
                for target_token in target_values:
                    base_margin = margin_summary(base_logits[0], target.position, target_token, tie_tolerance=tie_tolerance)
                    after = margin_summary(source_logits[source_token], target.position, target_token, tie_tolerance=tie_tolerance)
                    delta = float(after["logit_margin"] - base_margin["logit_margin"])
                    output.append(
                        {
                            **common,
                            "candidate_set_mode": mode,
                            "source_position": source.position,
                            "target_position": target.position,
                            "source_candidate_token_id": source_token,
                            "target_candidate_token_id": target_token,
                            "base_logit_margin": base_margin["logit_margin"],
                            "intervened_logit_margin": after["logit_margin"],
                            "delta_logit_margin": delta,
                            "polarity": "supportive" if delta >= 0.0 else "destructive",
                            "target_candidate_is_top1_after": bool(after["top1_matches_assignment"]),
                            "after_top1_token_id": after["top1_token_id"],
                            "source_value_rank": before[source_index]["top_token_ids"].index(source_token) + 1,
                            "target_value_rank": before[target_index]["top_token_ids"].index(target_token) + 1,
                        }
                    )
        for tensor in source_logits.values():
            del tensor
    del base_logits, base
    return output


def select_polarity_pairs(selected: Sequence[SelectedState], target_count: int) -> list[tuple[SelectedState, int, int]]:
    candidates: list[tuple[SelectedState, int, int]] = []
    for state in selected:
        if state.cohort != "primary_policy":
            continue
        for left in range(len(state.candidates)):
            for right in range(left + 1, len(state.candidates)):
                candidates.append((state, left, right))
    return sorted(candidates, key=lambda item: stable_key("polarity", item[0].record["state_key"], item[1], item[2]))[:target_count]


def rate_row(records: Sequence[Mapping[str, Any]], *, outcome: str, prompt_field: str = "prompt_id") -> dict[str, Any]:
    values = [row for row in records if row.get(outcome) is not None and row.get(prompt_field) is not None]
    numerator = sum(bool(row[outcome]) for row in values)
    denominator = len(values)
    if not values:
        return {
            "numerator": 0,
            "denominator": 0,
            "micro_rate": None,
            "prompt_macro_rate": None,
            "clustered_ci95_low": None,
            "clustered_ci95_high": None,
            "prompt_cluster_count": 0,
            "availability": "zero_count",
        }
    boot = prompt_clustered_bootstrap(
        values,
        value_field=outcome,
        prompt_field=prompt_field,
        iterations=10_000,
        seed=20260908,
        weighting="macro",
    )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "micro_rate": numerator / denominator,
        "prompt_macro_rate": boot["estimate"],
        "clustered_ci95_low": boot["ci95_low"],
        "clustered_ci95_high": boot["ci95_high"],
        "prompt_cluster_count": boot["cluster_count"],
        "availability": "available",
    }


def certificate_summaries(state_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    dimensions = (
        ("overall", ()),
        ("dataset", ("dataset",)),
        ("timestep", ("source_step",)),
        ("mask_ratio", ("generated_mask_ratio",)),
        ("sequence_length", ("sequence_length",)),
        ("set_size", ("set_size",)),
    )
    for cohort in sorted({str(row["cohort"]) for row in state_rows}):
        for gamma in sorted({float(row["gamma"]) for row in state_rows if str(row["cohort"]) == cohort}):
            base = [row for row in state_rows if str(row["cohort"]) == cohort and float(row["gamma"]) == gamma]
            for name, fields in dimensions:
                groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
                for row in base:
                    if name == "mask_ratio":
                        value = float(row.get("generated_mask_ratio", 0.0))
                        key = ("[0,.25)" if value < .25 else "[.25,.5)" if value < .5 else "[.5,.75)" if value < .75 else "[.75,1]",)
                    elif name == "sequence_length":
                        length = int(row.get("sequence_length", 0))
                        key = ("<=128" if length <= 128 else "129-256" if length <= 256 else "257-512" if length <= 512 else "513+",)
                    else:
                        key = tuple(row.get(field) for field in fields) if fields else ("all",)
                    groups[key].append(row)
                for key, group in groups.items():
                    for category, outcome in (
                        ("ALL_PASS", "all_pass"),
                        ("EXISTS_ONLY", "is_exists_only"),
                        ("NO_SAFE_ORDER", "is_no_safe_order"),
                        ("EXISTS_ONLY_LOO_PASS", "is_exists_only_loo_pass"),
                        ("EXISTS_ONLY_LOO_FAIL", "is_exists_only_loo_fail"),
                        ("EXISTENTIAL_PLUS_LOO", "existential_plus_final_loo_pass"),
                    ):
                        expanded = []
                        for row in group:
                            payload = dict(row)
                            payload["is_exists_only"] = row["category"] == "EXISTS_ONLY"
                            payload["is_no_safe_order"] = row["category"] == "NO_SAFE_ORDER"
                            payload["is_exists_only_loo_pass"] = row["category"] == "EXISTS_ONLY" and bool(row["final_loo_pass"])
                            payload["is_exists_only_loo_fail"] = row["category"] == "EXISTS_ONLY" and not bool(row["final_loo_pass"])
                            expanded.append(payload)
                        denominator_group = [row for row in expanded if category not in {"EXISTS_ONLY_LOO_PASS", "EXISTS_ONLY_LOO_FAIL"} or row["category"] == "EXISTS_ONLY"]
                        summary = rate_row(denominator_group, outcome=outcome)
                        output.append({
                            "row_type": "summary",
                            "cohort": cohort,
                            "gamma": gamma,
                            "breakdown": name,
                            "breakdown_value": key[0] if len(key) == 1 else list(key),
                            "category": category,
                            **summary,
                        })
    return output


def high_confidence_table(interventions: Sequence[Mapping[str, Any]], final_loo: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    specifications = (
        ("confidence_bin", CONFIDENCE_BINS),
        ("p1_minus_p2_bin", sorted({str(row["p1_minus_p2_bin"]) for row in interventions if row.get("p1_minus_p2_bin") not in {None, "missing"}})),
        ("logit_margin_bin", sorted({str(row["logit_margin_bin"]) for row in interventions if row.get("logit_margin_bin") not in {None, "missing"}})),
    )
    for cohort in sorted({str(row["cohort"]) for row in list(interventions) + list(final_loo)}):
        for field, labels in specifications:
            for label in labels:
                singleton = [row for row in interventions if str(row["cohort"]) == cohort and row.get(field) == label]
                loo = [row for row in final_loo if str(row["cohort"]) == cohort and row.get(field) == label]
                for intervention_type, rows, outcome in (
                    ("singleton", singleton, "counterfactual_reversal"),
                    ("full_set_excluding_target", loo, "final_loo_reversal"),
                    ("final_loo", loo, "final_loo_reversal"),
                ):
                    summary = rate_row(rows, outcome=outcome)
                    shifts = [float(row.get("margin_shift", row.get("final_loo_margin_shift", math.nan))) for row in rows]
                    shifts = [value for value in shifts if math.isfinite(value)]
                    ranks = [int(row["after_top1_original_rank"]) for row in rows if row.get("after_top1_original_rank") is not None]
                    output.append({
                        "row_type": "bin_rate",
                        "cohort": cohort,
                        "bin_field": field,
                        "bin": label,
                        "intervention_type": intervention_type,
                        "full_set_alias_of_final_loo": intervention_type == "full_set_excluding_target",
                        "logging_missing_count": sum(row.get(field) is None or row.get(field) == "missing" for row in rows),
                        "mean_margin_shift": math.fsum(shifts) / len(shifts) if shifts else None,
                        "median_margin_shift": quantile(shifts, .5),
                        "mean_new_top1_original_rank": math.fsum(ranks) / len(ranks) if ranks else None,
                        **summary,
                    })
    return output


def risk_coverage_rows(interventions: Sequence[Mapping[str, Any]], state_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    token_margins = [float(row["base_logit_margin"]) for row in interventions if row.get("base_logit_margin") is not None]
    thresholds = sorted(set(PRIMARY_GAMMAS + tuple(value for value in (quantile(token_margins, q) for q in (.1, .25, .5, .75, .9)) if value is not None)))
    for threshold in thresholds:
        eligible = [row for row in interventions if float(row["base_logit_margin"]) >= threshold]
        output.append({
            "row_type": "token_risk_coverage",
            "margin_threshold": threshold,
            "coverage_numerator": len(eligible),
            "coverage_denominator": len(interventions),
            "coverage": len(eligible) / len(interventions) if interventions else None,
            "risk_numerator": sum(bool(row["counterfactual_reversal"]) for row in eligible),
            "risk_denominator": len(eligible),
            "risk": sum(bool(row["counterfactual_reversal"]) for row in eligible) / len(eligible) if eligible else None,
        })
    for gamma in PRIMARY_GAMMAS:
        rows = [row for row in state_rows if float(row["gamma"]) == gamma and row["cohort"] == "primary_policy"]
        output.append({
            "row_type": "set_risk_coverage",
            "margin_threshold": gamma,
            "coverage_numerator": sum(bool(row["all_pass"]) for row in rows),
            "coverage_denominator": len(rows),
            "coverage": sum(bool(row["all_pass"]) for row in rows) / len(rows) if rows else None,
            "risk_numerator": sum(not bool(row["all_pass"]) for row in rows if float(row["all_order_logit_certificate"] or -math.inf) >= gamma),
            "risk_denominator": sum(float(row["all_order_logit_certificate"] or -math.inf) >= gamma for row in rows),
            "risk": None,  # Filled below to preserve denominator explicitly.
        })
        last = output[-1]
        last["risk"] = last["risk_numerator"] / last["risk_denominator"] if last["risk_denominator"] else None
    return output


def local_topk_table(interventions: Sequence[Mapping[str, Any]], values: Sequence[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: dict[str, list[Mapping[str, Any]]] = {
        "all_interventions": list(interventions),
        "actual_reversals": [row for row in interventions if bool(row["counterfactual_reversal"])],
        "high_confidence_target": [row for row in interventions if bool(row["high_confidence_target"])],
        "policy_selected_target": [row for row in interventions if bool(row["policy_selected_target"])],
        "supportive": [row for row in interventions if row["polarity"] == "supportive"],
        "destructive": [row for row in interventions if row["polarity"] == "destructive"],
    }
    for name, rows in groups.items():
        for k in values:
            output.append({
                "stratum": name,
                "k": int(k),
                "covered_count": sum(bool(row.get(f"after_top1_in_pre_top{k}")) for row in rows),
                "observation_count": len(rows),
                "local_topk_coverage": (
                    sum(bool(row.get(f"after_top1_in_pre_top{k}")) for row in rows) / len(rows) if rows else None
                ),
            })
    return output


def polarity_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["state_key"], row["candidate_set_mode"], row["source_position"], row["target_position"])].append(row)
    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        deltas = [float(row["delta_logit_margin"]) for row in group]
        signs = {"positive" if value > 0 else "negative" if value < 0 else "zero" for value in deltas}
        output.append({
            "row_type": "pair_summary",
            "state_key": key[0],
            "candidate_set_mode": key[1],
            "source_position": key[2],
            "target_position": key[3],
            "assignment_count": len(group),
            "positive_shift_count": sum(value > 0 for value in deltas),
            "negative_shift_count": sum(value < 0 for value in deltas),
            "sign_entropy": -sum((count / len(deltas)) * math.log(count / len(deltas)) for count in Counter(signs).values()),
            "delta_range": max(deltas) - min(deltas),
            "contains_supportive_and_destructive": "positive" in signs and "negative" in signs,
            "reversal_assignment_count": sum(not bool(row["target_candidate_is_top1_after"]) for row in group),
            "assignment_dependent_reversal": len({bool(row["target_candidate_is_top1_after"]) for row in group}) > 1,
        })
    undirected: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        left, right = sorted((int(row["source_position"]), int(row["target_position"])))
        undirected[(row["state_key"], row["candidate_set_mode"], left, right)].append(row)
    for key, group in undirected.items():
        directed = {
            (int(row["source_position"]), int(row["target_position"]), int(row["source_candidate_token_id"]), int(row["target_candidate_token_id"])): float(row["delta_logit_margin"])
            for row in group
        }
        comparisons: list[float] = []
        for (source_pos, target_pos, source_token, target_token), forward_delta in directed.items():
            reverse = directed.get((target_pos, source_pos, target_token, source_token))
            if reverse is not None and source_pos == key[2]:
                comparisons.append(forward_delta - reverse)
        output.append({
            "row_type": "undirected_direction_summary",
            "state_key": key[0],
            "candidate_set_mode": key[1],
            "position_i": key[2],
            "position_j": key[3],
            "comparable_assignment_count": len(comparisons),
            "mean_directional_delta_difference": math.fsum(comparisons) / len(comparisons) if comparisons else None,
            "direction_reversal_assignment_count": sum(abs(value) > 1.0e-7 for value in comparisons),
            "direction_reversal_rate": sum(abs(value) > 1.0e-7 for value in comparisons) / len(comparisons) if comparisons else None,
        })
    return output


def causal_corrections(source_root: Path) -> list[dict[str, Any]]:
    rows = read_parquet_rows(source_root / "raw" / "counterfactual_events.parquet")
    events = [row for row in rows if row.get("cohort_role") == "event"]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in events:
        grouped[str(row.get("classification", "missing_classification"))].append(row)
    interpretations = {
        "singleton-sufficient": "at least one singleton branch crosses; sufficient, not uniquely necessary",
        "multi-singleton-additive": "no singleton alone crosses but their additive estimate crosses",
        "synergy-only": "joint branch crosses without singleton/additive crossing",
        "suppression/cancellation": "a singleton/additive crossing disappears in the joint branch",
        "unattributed": "natural reconstruction mismatch or no measured crossing; not credited as anchor causality",
    }
    output: list[dict[str, Any]] = []
    for classification, members in sorted(grouped.items()):
        output.append({
            "classification": classification,
            "event_count": len(members),
            "total_event_count": len(events),
            "event_rate": len(members) / len(events) if events else None,
            "joint_crosses_count": sum(bool(row.get("joint_crosses")) for row in members),
            "reconstruction_mismatch_count": sum(row.get("reconstruction_matches") is False for row in members),
            "interpretation": interpretations.get(classification, "unrecognised stored category; retained without coercion"),
            "source_status": "audit_of_existing_raw_counterfactual_events",
        })
    if not rows:
        output.append({"status": "source_counterfactual_events_missing", "event_count": 0, "total_event_count": 0})
    return output


def prompt_split(prompt_id: str) -> str:
    number = int(stable_key("split", prompt_id)[:8], 16) % 10
    return "test" if number < 2 else "calibration" if number < 4 else "train"


def certificate_calibration_table(state_rows: Sequence[Mapping[str, Any]], residuals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prompt-disjoint one-sided pairwise certificate calibration/evaluation."""

    output: list[dict[str, Any]] = []
    calibration_errors: dict[int, list[float]] = defaultdict(list)
    set_errors: dict[int, list[float]] = defaultdict(list)
    for row in residuals:
        if prompt_split(str(row["prompt_id"])) == "calibration":
            # Positive value is exactly the dangerous additive overestimate.
            calibration_errors[int(row["set_size"])].append(max(0.0, -float(row["residual"])))
    for row in state_rows:
        if row["cohort"] == "primary_policy" and float(row["gamma"]) == 0.0 and prompt_split(str(row["prompt_id"])) == "calibration":
            predicted = row.get("pairwise_lower_bound_certificate")
            exact = row.get("all_order_logit_certificate")
            if predicted is not None and exact is not None:
                set_errors[int(row["set_size"])].append(max(0.0, float(predicted) - float(exact)))
    for size in sorted({int(row["set_size"]) for row in state_rows}):
        edge_penalty = quantile(calibration_errors[size], .95)
        set_penalty = quantile(set_errors[size], .95)
        test = [
            row for row in state_rows
            if row["cohort"] == "primary_policy" and float(row["gamma"]) == 0.0
            and int(row["set_size"]) == size and prompt_split(str(row["prompt_id"])) == "test"
        ]
        for method, penalty in (("edge_residual_bound", edge_penalty), ("set_level_post_selection", set_penalty)):
            rows = []
            for row in test:
                predicted = row.get("pairwise_lower_bound_certificate")
                exact = row.get("all_order_logit_certificate")
                if predicted is None or exact is None or penalty is None:
                    continue
                rows.append({
                    "predicted_safe": float(predicted) - float(penalty) >= 0.0,
                    "exact_safe": bool(row["all_pass"]),
                    "unsafe": not bool(row["all_pass"]),
                    "score": -float(predicted),
                })
            certified = [row for row in rows if row["predicted_safe"]]
            unsafe = [row for row in rows if row["unsafe"]]
            ranking = binary_ranking_metrics([row["unsafe"] for row in rows], [row["score"] for row in rows])
            output.append({
                "set_size": size,
                "method": method,
                "calibration_prompt_disjoint": True,
                "calibration_row_count": len(calibration_errors[size]) if method == "edge_residual_bound" else len(set_errors[size]),
                "penalty_quantile": .95,
                "one_sided_penalty": penalty,
                "test_state_count": len(rows),
                "certified_count": len(certified),
                "false_certificate_count": sum(not row["exact_safe"] for row in certified),
                "false_certificate_rate": sum(not row["exact_safe"] for row in certified) / len(certified) if certified else None,
                "unsafe_batch_recall": sum(not row["predicted_safe"] for row in unsafe) / len(unsafe) if unsafe else None,
                "certified_batch_precision": sum(row["exact_safe"] for row in certified) / len(certified) if certified else None,
                "reversal_detection_auroc": ranking["auroc"],
                "reversal_detection_auprc": ranking["auprc"],
            })
    return output


def dependency_graph_table(state_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in state_rows:
        if row["cohort"] != "primary_policy":
            continue
        output.append({
            "state_key": row["state_key"],
            "prompt_id": row["prompt_id"],
            "dataset": row["dataset"],
            "gamma": row["gamma"],
            "set_size": row["set_size"],
            "node_count": row["set_size"],
            "directed_edge_count": row["graph_edge_count"],
            "edge_density": row["graph_edge_density"],
            "is_dag": row["graph_is_dag"],
            "has_cycle": row["graph_has_cycle"],
            "scc_count": row["graph_scc_count"],
            "largest_scc_size": row["graph_largest_scc_size"],
            "topological_order_is_exact_witness": row["graph_topological_order_is_exact_witness"],
            "exact_existential_pass": row["existential_pass"],
            "pairwise_acyclic_exact_no_witness": bool(row["graph_is_dag"] and not row["existential_pass"]),
            "pairwise_cycle_exact_witness": bool(row["graph_has_cycle"] and row["existential_pass"]),
            "greedy_exact_bottleneck_regret": row["graph_vs_exact_bottleneck_regret"],
        })
    return output


def _summary_value(rows: Sequence[Mapping[str, Any]], *, category: str, gamma: float = 0.0) -> Mapping[str, Any] | None:
    return next(
        (
            row for row in rows
            if row.get("cohort") == "primary_policy" and row.get("breakdown") == "overall"
            and float(row.get("gamma", -1.0)) == gamma and row.get("category") == category
        ),
        None,
    )


def _format_rate(row: Mapping[str, Any] | None) -> str:
    if not row or not row.get("denominator"):
        return "not estimable (0 denominator)"
    return f"{row['numerator']}/{row['denominator']} ({100 * float(row['micro_rate']):.2f}%)"


def write_report(
    output_root: Path,
    summaries: Sequence[Mapping[str, Any]],
    high_conf: Sequence[Mapping[str, Any]],
    polarity: Sequence[Mapping[str, Any]],
    local_topk: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    graph: Sequence[Mapping[str, Any]],
    corrections: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    all_pass = _summary_value(summaries, category="ALL_PASS")
    exists = _summary_value(summaries, category="EXISTS_ONLY")
    no_safe = _summary_value(summaries, category="NO_SAFE_ORDER")
    exists_loo_pass = _summary_value(summaries, category="EXISTS_ONLY_LOO_PASS")
    exists_loo_fail = _summary_value(summaries, category="EXISTS_ONLY_LOO_FAIL")
    high = [row for row in high_conf if row.get("bin_field") == "confidence_bin" and row.get("bin") in {"0.9<=p1<0.95", "0.95<=p1<0.99", "p1>=0.99"} and row.get("intervention_type") == "singleton"]
    polarity_pairs = [row for row in polarity if row.get("row_type") == "pair_summary" and row.get("candidate_set_mode") == "fixed_k4"]
    local = next((row for row in local_topk if row.get("stratum") == "actual_reversals" and int(row.get("k", -1)) == 4), None)
    graph_witness = [row for row in graph if row.get("topological_order_is_exact_witness") is not None]
    correction_total = sum(int(row.get("event_count", 0)) for row in corrections)
    exists_loo_total = int((exists_loo_pass or {}).get("denominator", 0))
    exists_loo_pass_count = int((exists_loo_pass or {}).get("numerator", 0))
    high_rows = [
        row for row in high
        if row.get("intervention_type") == "singleton" and row.get("availability") == "available"
    ]
    high_numerator = sum(int(row.get("numerator", 0)) for row in high_rows)
    high_denominator = sum(int(row.get("denominator", 0)) for row in high_rows)
    polarity_mixed = sum(bool(row.get("contains_supportive_and_destructive")) for row in polarity_pairs)
    calibration_rates = [float(row["false_certificate_rate"]) for row in calibration if row.get("false_certificate_rate") is not None]
    graph_agree = sum(bool(row.get("topological_order_is_exact_witness")) for row in graph_witness)
    synergy = next((row for row in corrections if row.get("classification") == "synergy-only"), None)
    all_rate = float((all_pass or {}).get("micro_rate") or 0.0)
    existential_loo = _summary_value(summaries, category="EXISTENTIAL_PLUS_LOO")
    existential_loo_rate = float((existential_loo or {}).get("micro_rate") or 0.0)
    gain = existential_loo_rate - all_rate
    if gain >= 0.05 and exists_loo_total and exists_loo_pass_count / exists_loo_total >= 0.5:
        primary_decision = "BRANCH: prioritise existential/partial-order certificate with mandatory final-LOO validation."
    elif all_rate > 0.0 and (not exists_loo_total or exists_loo_pass_count / exists_loo_total < 0.5):
        primary_decision = "GO: retain universal all-order current-top1 VCCC as the primary safe certificate."
    else:
        primary_decision = "NO-GO for a policy claim: collect a larger primary policy cohort before selecting U versus E+LOO."
    candidate_decision = (
        "keep candidate-value extension exploratory" if polarity_pairs and polarity_mixed / len(polarity_pairs) >= .2
        and local is not None and float(local.get("local_topk_coverage") or 0.0) >= .7
        else "defer candidate-value extension; polarity/local-topK evidence is not yet strong enough"
    )
    pairwise_decision = (
        "pairwise oracle is promising for a residual fallback" if calibration_rates and max(calibration_rates) <= .05
        else "do not replace exact oracle with pairwise-only certification; retain residual/exact fallback"
    )
    lines = [
        "# VCCC oracle audit report",
        "",
        "## Direct answers",
        "",
        f"1. Universal all-order current-top1 VCCC passes: {_format_rate(all_pass)}.",
        f"2. All-order failures with an existential safe witness: {_format_rate(exists)}.",
        f"3. Existential-only cases that also pass final LOO: {_format_rate(exists_loo_pass)}; final-LOO failures: {_format_rate(exists_loo_fail)}.",
        f"4. Of the existential-only cases, {exists_loo_pass_count}/{exists_loo_total} pass final LOO; the remainder is treated as early-locking risk, never merged with safe-parallelism coverage.",
        f"5. High-confidence (p1>=0.9, split into three fixed bins) context-only singleton reversals: {high_numerator}/{high_denominator} ({100 * high_numerator / high_denominator:.2f}% if high_denominator else 0.0). Zero-count bins are explicitly retained in `tables/high_confidence_reversal.csv`.",
        f"6. Fixed-K=4 position-pair matrices with both supportive and destructive values: {polarity_mixed}/{len(polarity_pairs)}.",
        f"7. Local top-4 coverage after actual reversals: {_format_rate({'numerator': local.get('covered_count'), 'denominator': local.get('observation_count'), 'micro_rate': local.get('local_topk_coverage')} if local else None)}.",
        f"8. Prompt-disjoint pairwise calibration/test rows: {len(calibration)}; observed false-certificate rates range from {min(calibration_rates) if calibration_rates else None} to {max(calibration_rates) if calibration_rates else None}.",
        f"9. Existing causal-attribution audit categories sum to {correction_total}; pure synergy-only is {synergy.get('event_count') if synergy else 0}/{synergy.get('total_event_count') if synergy else 0}. All previously omitted categories are enumerated in `tables/causal_audit_corrections.csv`.",
        f"10. Directional graph rows with a testable topological witness: {graph_agree}/{len(graph_witness)} exactly satisfy the graph-derived witness test; disagreement cases remain explicit in `tables/dependency_graph_stats.csv`.",
        f"11. All-order coverage is {all_rate:.4f}; existential+final-LOO coverage is {existential_loo_rate:.4f}; incremental certified coverage is {gain:.4f}.",
        f"12. {primary_decision} Candidate-value: {candidate_decision}. Pairwise: {pairwise_decision}",
        "",
        "## Interpretation boundary",
        "",
        "`natural_transition` from the source audit changes both the decoder trajectory and context. This oracle's primary quantity is `context_only_counterfactual`: the input context alone changes while sequence length and position IDs are fixed within each exact branch. The pinned collector exposes no separately controllable timestep/noise embedding, so a `timestep_only_control` is explicitly unsupported rather than inferred.",
        "",
        "Universal (U) fixes only the exact current top-1 assignment and checks all subsets/orders. Existential (E) uses subset DP. Candidate-value matrices are exploratory and are never pooled with U/E prevalence. Every certificate uses full-vocabulary raw-logit margins, deterministic argmax, `use_cache=False`, and an explicit tie rule: margin threshold is inclusive but deterministic top-1 must still be the assignment.",
        "",
        "## Runtime and provenance",
        "",
        f"- Source run: `{metadata['source_run_root']}`",
        f"- Exact branch forwards: {metadata['forward_accounting']['branch_forwards']}; model batch calls: {metadata['forward_accounting']['batch_calls']}",
        f"- Runtime seconds: {metadata['runtime_seconds']}; peak VRAM MiB: {metadata['peak_vram_mib']}",
        f"- Primary/hard selected states: {metadata['selected_primary_state_count']}/{metadata['selected_hard_state_count']}",
        "",
        "## Decision",
        "",
        "The decision in `summary.json` is rule-based and conservative: all-order is retained only if it has nontrivial certified coverage; existential/partial-order is preferred only if its final-LOO-passing gain is material; candidate-value extension requires both assignment-dependent polarity and adequate local top-K coverage; pairwise-only certification requires low prompt-disjoint false-certificate rates.",
    ]
    report = "\n".join(lines) + "\n"
    (output_root / "report.md").write_text(report, encoding="utf-8")
    return {
        "all_order": all_pass,
        "exists_only": exists,
        "no_safe_order": no_safe,
        "exists_only_loo_pass": exists_loo_pass,
        "exists_only_loo_fail": exists_loo_fail,
        "existential_plus_final_loo": existential_loo,
        "decision": {
            "primary": primary_decision,
            "candidate_value": candidate_decision,
            "pairwise": pairwise_decision,
        },
    }


def write_figures(
    figures: Path,
    summaries: Sequence[Mapping[str, Any]],
    high_conf: Sequence[Mapping[str, Any]],
    polarity_rows: Sequence[Mapping[str, Any]],
    local_rows: Sequence[Mapping[str, Any]],
    residuals: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    graph_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> list[str]:
    """Produce requested diagnostic figures; empty cohorts get labelled plots."""

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(name: str) -> None:
        path = figures / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))

    primary = [row for row in summaries if row.get("cohort") == "primary_policy" and row.get("breakdown") == "set_size" and float(row.get("gamma", -1)) == 0.0]
    sizes = sorted({int(row["breakdown_value"]) for row in primary if row.get("breakdown_value") is not None})
    plt.figure(figsize=(7, 4))
    bottom = [0.0] * len(sizes)
    for category, color in (("ALL_PASS", "#22c55e"), ("EXISTS_ONLY", "#f59e0b"), ("NO_SAFE_ORDER", "#ef4444")):
        values = [next((float(row["micro_rate"] or 0.0) for row in primary if int(row["breakdown_value"]) == size and row["category"] == category), 0.0) for size in sizes]
        plt.bar([str(size) for size in sizes], values, bottom=bottom, label=category, color=color)
        bottom = [left + right for left, right in zip(bottom, values)]
    plt.ylabel("state proportion")
    plt.xlabel("set size")
    plt.legend(fontsize=8)
    plt.title("Universal vs existential exact categories")
    save("all_vs_existential_by_set_size.png")

    exists = [row for row in summaries if row.get("cohort") == "primary_policy" and row.get("breakdown") == "overall" and float(row.get("gamma", -1)) == 0.0 and row.get("category") in {"EXISTS_ONLY_LOO_PASS", "EXISTS_ONLY_LOO_FAIL"}]
    plt.figure(figsize=(5, 4))
    labels = [str(row["category"]).replace("EXISTS_ONLY_", "") for row in exists] or ["no data"]
    values = [float(row["micro_rate"] or 0.0) for row in exists] or [0.0]
    plt.bar(labels, values, color=["#22c55e", "#ef4444"][:len(values)])
    plt.ylim(0, 1)
    plt.ylabel("proportion among EXISTS_ONLY")
    plt.title("Existential-only final LOO split")
    save("existential_only_final_loo.png")

    pareto = [row for row in summaries if row.get("cohort") == "primary_policy" and row.get("breakdown") == "set_size" and float(row.get("gamma", -1)) == 0.0]
    plt.figure(figsize=(7, 4))
    for category, style in (("ALL_PASS", "o-"), ("EXISTENTIAL_PLUS_LOO", "s-")):
        ys = [next((float(row["micro_rate"] or 0.0) for row in pareto if int(row["breakdown_value"]) == size and row["category"] == category), 0.0) for size in sizes]
        plt.plot(sizes, ys, style, label=category)
    plt.xlabel("set size")
    plt.ylabel("certified coverage")
    plt.legend()
    plt.title("All-order vs existential+LOO Pareto")
    save("certificate_pareto.png")

    risk = [row for row in high_conf if row.get("row_type") == "risk_coverage"]
    # risk rows are merged into high_conf in main before figure generation.
    plt.figure(figsize=(6, 4))
    token = [row for row in risk if row.get("row_type") == "token_risk_coverage"]
    if token:
        plt.plot([row["coverage"] for row in token], [row["risk"] for row in token], "o-")
    else:
        plt.text(.5, .5, "No eligible risk rows", ha="center")
    plt.xlabel("coverage")
    plt.ylabel("context-only reversal risk")
    plt.title("Margin risk–coverage")
    save("margin_risk_coverage.png")

    granular: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in polarity_rows:
        if row.get("row_type") != "pair_summary" and row.get("candidate_set_mode") == "fixed_k4":
            granular[(row["state_key"], row["candidate_set_mode"], row["source_position"], row["target_position"])].append(row)
    # Fixed deterministic hash order is the predeclared representative-case
    # rule: matrices are not selected on their apparent polarity strength.
    selected_matrices = sorted(granular.items(), key=lambda item: stable_key("matrix", *item[0]))[:10]
    if not selected_matrices:
        plt.figure(figsize=(5, 3))
        plt.text(.5, .5, "No eligible polarity pairs", ha="center", va="center")
        plt.axis("off")
        save("candidate_polarity_heatmap_index.png")
    for matrix_index, (key, rows) in enumerate(selected_matrices, 1):
        source_tokens = sorted({int(row["source_candidate_token_id"]) for row in rows}, key=lambda token: min(int(row["source_value_rank"]) for row in rows if int(row["source_candidate_token_id"]) == token))
        target_tokens = sorted({int(row["target_candidate_token_id"]) for row in rows}, key=lambda token: min(int(row["target_value_rank"]) for row in rows if int(row["target_candidate_token_id"]) == token))
        matrix = [[next(float(row["delta_logit_margin"]) for row in rows if int(row["source_candidate_token_id"]) == source and int(row["target_candidate_token_id"]) == target) for target in target_tokens] for source in source_tokens]
        maximum = max((abs(value) for vector in matrix for value in vector), default=1.0)
        plt.figure(figsize=(5, 4))
        image = plt.imshow(matrix, cmap="coolwarm", vmin=-maximum, vmax=maximum, aspect="auto")
        plt.colorbar(image, label="Δ target logit margin")
        plt.xticks(range(len(target_tokens)), [str(token) for token in target_tokens], rotation=45, ha="right", fontsize=7)
        plt.yticks(range(len(source_tokens)), [str(token) for token in source_tokens], fontsize=7)
        plt.xlabel(f"target position {key[3]} candidate token")
        plt.ylabel(f"source position {key[2]} candidate token")
        plt.title(f"Polarity matrix {matrix_index}: {str(key[0])[:8]}")
        save(f"candidate_polarity_matrix_{matrix_index:02d}.png")

    plt.figure(figsize=(6, 4))
    for name, rows in defaultdict(list, {str(row["stratum"]): [] for row in local_rows}).items():
        del rows
    for stratum in ("all_interventions", "actual_reversals", "destructive"):
        rows = [row for row in local_rows if row.get("stratum") == stratum]
        if rows:
            plt.plot([row["k"] for row in rows], [row["local_topk_coverage"] for row in rows], marker="o", label=stratum)
    plt.xlabel("K")
    plt.ylabel("local Top-K coverage")
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    plt.title("Exact local candidate coverage")
    save("local_topk_coverage.png")

    plt.figure(figsize=(6, 4))
    values = [float(row["residual"]) for row in residuals if row.get("residual") is not None]
    if values:
        plt.hist(values, bins=50, color="#6366f1")
        plt.axvline(0, color="black", linewidth=1)
    else:
        plt.text(.5, .5, "No residual rows", ha="center")
    plt.xlabel("exact minus additive residual")
    plt.title("Pairwise residual negative tail")
    save("pairwise_residual_negative_tail.png")

    plt.figure(figsize=(5, 5))
    predicted = [float(row["pairwise_lower_bound_certificate"]) for row in state_rows if row.get("cohort") == "primary_policy" and float(row.get("gamma", -1)) == 0.0 and row.get("pairwise_lower_bound_certificate") is not None]
    exact = [float(row["all_order_logit_certificate"]) for row in state_rows if row.get("cohort") == "primary_policy" and float(row.get("gamma", -1)) == 0.0 and row.get("pairwise_lower_bound_certificate") is not None]
    if predicted:
        plt.scatter(predicted, exact, s=12, alpha=.7)
        low, high = min(predicted + exact), max(predicted + exact)
        plt.plot([low, high], [low, high], "k--")
    else:
        plt.text(.5, .5, "No certificate rows", ha="center")
    plt.xlabel("pairwise lower-bound certificate")
    plt.ylabel("exact all-order certificate")
    plt.title("Pairwise vs exact certificate")
    save("pairwise_certificate_calibration.png")

    plt.figure(figsize=(6, 4))
    graph_primary = [row for row in graph_rows if float(row.get("gamma", -1)) == 0.0]
    plt.bar(["cycle", "DAG"], [sum(bool(row["has_cycle"]) for row in graph_primary), sum(bool(row["is_dag"]) for row in graph_primary)], color=["#ef4444", "#22c55e"])
    plt.ylabel("state count")
    plt.title("Directional dependency graph cycles")
    save("dependency_graph_cycles.png")

    plt.figure(figsize=(6, 4))
    accounting = metadata["forward_accounting"]
    labels = ["subset", "polarity"]
    values = [accounting["subset_branch_forwards"], accounting["polarity_branch_forwards"]]
    plt.bar(labels, values, color=["#2563eb", "#7c3aed"])
    plt.ylabel("exact branch forwards")
    plt.title("Runtime / exact-forward decomposition")
    save("runtime_forward_decomposition.png")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "vccc_oracle_audit.yaml")
    parser.add_argument("--probe-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-run", type=Path, required=True, help="Completed top1_dynamics_audit run directory (read-only).")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Use one policy and one hard state per set size; never interpret as final.")
    return parser.parse_args()


def make_output_root(base: Path, run_id: str | None) -> Path:
    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = base / identifier
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing VCCC oracle run: {root}")
    for name in ("raw", "tables", "figures"):
        (root / name).mkdir(parents=True, exist_ok=False) if name == "raw" else (root / name).mkdir(parents=True, exist_ok=True)
    return root


def main() -> None:
    args = parse_args()
    wall_started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    probe_root = args.probe_root.resolve()
    source_root = args.source_run.resolve()
    if not (source_root / "run_manifest.json").exists():
        raise FileNotFoundError(f"Source run lacks run_manifest.json: {source_root}")
    source_manifest = json.loads((source_root / "run_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") not in {"completed", "observational_completed_exact_audit_blocked_resource_estimate"}:
        raise RuntimeError(f"Source run is not a completed observational evidence bundle: {source_manifest.get('status')!r}")
    source_fast_commit = source_manifest.get("fast_dllm_requested_commit")
    expected_fast_commit = config["model"]["fast_dllm_commit"]
    if source_fast_commit is not None and str(source_fast_commit) != str(expected_fast_commit):
        raise RuntimeError(
            "Source run Fast-dLLM pin differs from this oracle config: "
            f"source={source_fast_commit!r}, oracle={expected_fast_commit!r}"
        )
    output_base = Path(args.output_root) if args.output_root else probe_root / str(config["storage"]["output_root"])
    output_root = make_output_root(output_base, args.run_id)
    shutil.copy2(args.config, output_root / "config.yaml")
    write_json(output_root / "source_manifest.json", source_manifest)
    trajectories = read_jsonl(source_root / "raw" / "trajectories.jsonl")
    selected, selection_diagnostics = choose_states(trajectories, config, source_root)
    if args.smoke:
        selected = [state for state in selected if sum(item.cohort == state.cohort and len(item.candidates) == len(state.candidates) for item in selected[: selected.index(state) + 1]) <= 1]
    if not selected:
        raise RuntimeError("No primary policy states met the exact-current-top1 and requested-set-size criteria.")
    from scripts.collect_states import mask_token_id
    from scripts.run_top1_dynamics_audit import load_model
    import torch

    model, tokenizer, _device = load_model(config, probe_root)
    mask_id = mask_token_id(model, tokenizer)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    accounting = ForwardAccounting(start=time.perf_counter())
    all_state_rows: list[dict[str, Any]] = []
    all_witness_rows: list[dict[str, Any]] = []
    all_final_loo: list[dict[str, Any]] = []
    all_interventions: list[dict[str, Any]] = []
    all_residuals: list[dict[str, Any]] = []
    for index, selected_state in enumerate(selected, 1):
        margins, base_topk = evaluate_subset_state(
            model,
            selected_state,
            mask_token_id=mask_id,
            batch_size=int(config["execution"]["subset_batch_size"]),
            tie_tolerance=float(config["certificate"]["tie_tolerance"]),
            accounting=accounting,
        )
        rows = certificate_rows_for_state(selected_state, margins, base_topk, config)
        all_state_rows.extend(rows[0])
        all_witness_rows.extend(rows[1])
        all_final_loo.extend(rows[2])
        all_interventions.extend(rows[3])
        all_residuals.extend(rows[4])
        print(json.dumps({"stage": "subset", "state": index, "total_states": len(selected), "state_key": selected_state.record["state_key"], "forwards": accounting.branch_forwards}))
    polarity_rows: list[dict[str, Any]] = []
    for selected_state, first, second in select_polarity_pairs(selected, int(config["candidate_polarity"]["representative_pair_count"])):
        polarity_rows.extend(evaluate_polarity_pair(
            model,
            selected_state,
            first,
            second,
            mask_token_id=mask_id,
            batch_size=int(config["execution"]["polarity_batch_size"]),
            tie_tolerance=float(config["certificate"]["tie_tolerance"]),
            config=config,
            accounting=accounting,
        ))
    torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - wall_started
    peak_vram_mib = float(torch.cuda.max_memory_allocated() / 2**20)
    summaries = certificate_summaries(all_state_rows)
    risk_rows = risk_coverage_rows(all_interventions, all_state_rows)
    high_conf = high_confidence_table(all_interventions, all_final_loo) + risk_rows
    local_rows = local_topk_table(all_interventions, config["sampling"]["local_topk_values"])
    polarity = polarity_rows + polarity_summary(polarity_rows)
    calibration = certificate_calibration_table(all_state_rows, all_residuals)
    graph_rows = dependency_graph_table(all_state_rows)
    corrections = causal_corrections(source_root)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_smoke" if args.smoke else "completed",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_root": str(source_root),
        "source_run_status": source_manifest.get("status"),
        "source_commit": source_manifest.get("source_git_commit"),
        "frozen_model": config["model"]["name"],
        "dtype": config["model"]["dtype"],
        "exact_cache_policy": "use_cache=False",
        "context_only_definition": "same fixed-length input and position IDs; only selected token context differs",
        "timestep_only_control": {"status": "not_implemented", "reason": "pinned collector/model surface has no independently controllable timestep/noise embedding"},
        "selected_state_count": len(selected),
        "selected_primary_state_count": sum(state.cohort == "primary_policy" for state in selected),
        "selected_hard_state_count": sum(state.cohort == "hard_order_sensitive" for state in selected),
        "selection_diagnostics": selection_diagnostics,
        "forward_accounting": accounting.__dict__,
        "runtime_seconds": runtime_seconds,
        "peak_vram_mib": peak_vram_mib,
        "margin_definition": "assigned logit minus maximum full-vocabulary competitor logit",
        "tie_policy": "inclusive margin threshold plus deterministic torch.argmax must equal assignment; ties are logged",
    }
    write_jsonl(output_root / "raw" / "selected_states.jsonl", [{**_row_common(state), "selection_reason": state.selection_reason} for state in selected])
    write_jsonl(output_root / "raw" / "singleton_interventions.jsonl", all_interventions)
    write_jsonl(output_root / "raw" / "subset_residuals.jsonl", all_residuals)
    write_jsonl(output_root / "raw" / "candidate_polarity_assignments.jsonl", polarity_rows)
    table_map = {
        "all_vs_existential": list(all_state_rows) + summaries,
        "witness_orders": all_witness_rows,
        "final_loo": all_final_loo,
        "high_confidence_reversal": high_conf,
        "candidate_polarity": polarity,
        "local_topk_coverage": local_rows,
        "pairwise_residuals": all_residuals,
        "pairwise_certificate_eval": calibration,
        "dependency_graph_stats": graph_rows,
        "causal_audit_corrections": corrections,
        "exact_forward_counts": [{
            "subset_branch_forwards": accounting.subset_branch_forwards,
            "polarity_branch_forwards": accounting.polarity_branch_forwards,
            "total_branch_forwards": accounting.branch_forwards,
            "model_batch_calls": accounting.batch_calls,
            "runtime_seconds": runtime_seconds,
            "peak_vram_mib": peak_vram_mib,
        }],
    }
    for name, rows in table_map.items():
        write_csv(output_root / "tables" / f"{name}.csv", rows)
    report_summary = write_report(output_root, summaries, high_conf, polarity, local_rows, calibration, graph_rows, corrections, metadata)
    metadata["direct_answer_summary"] = report_summary
    figures = write_figures(output_root / "figures", summaries, high_conf, polarity, local_rows, all_residuals, all_state_rows, graph_rows, metadata)
    metadata["figure_paths"] = figures
    write_json(output_root / "run_metadata.json", metadata)
    write_json(output_root / "summary.json", {"status": metadata["status"], "metadata": metadata, "direct_answers": report_summary})
    print(json.dumps({"status": metadata["status"], "output_root": str(output_root), "forwards": accounting.branch_forwards, "runtime_seconds": runtime_seconds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
