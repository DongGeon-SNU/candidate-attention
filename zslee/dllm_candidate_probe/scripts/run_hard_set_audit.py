#!/usr/bin/env python3
"""Audit pair-rule cliques with exact frozen-LLaDA leave-one-out branches.

The pilot artifacts supply base states and, where matching, existing unstable
pair metrics.  Anchors and every exact leave-one-out measurement are evaluated
as fresh full-sequence forwards with ``use_cache=False``.  A scalar-only cache
makes interruption/restart safe without persisting model internals.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_states import mask_token_id
from scripts.run_counterfactual_probe import git_commit, gpu_memory_snapshot, load_model, token_repr
from src.branching import Candidate, exact_forward, make_branch, make_leave_one_out_branches
from src.hard_set_generation import (
    AuditCandidate,
    PairMetric,
    anchor_conflicts,
    is_clique,
    pair_key,
    set_signature,
    stress_mined_sets,
    stress_score,
)
from src.set_audit import (
    ExactScalarCache,
    candidate_record,
    candidate_with_token,
    candidates_from_state,
    pilot_pair_metrics,
    pair_record,
    provenance_validation,
    scalar_cache_key,
    state_id,
    summarize_failures,
    validate_pilot_artifacts,
    wilson_interval,
    write_csv,
    write_json,
    write_jsonl,
)


def phase(step: int, total_steps: int) -> str:
    ratio = step / max(1, total_steps - 1)
    return "early" if ratio < 1 / 3 else "mid" if ratio < 2 / 3 else "late"


def confidence_bin(value: float) -> str:
    if value < 0.5:
        return "<0.5"
    if value < 0.7:
        return "[0.5,0.7)"
    if value < 0.9:
        return "[0.7,0.9)"
    return ">=0.9"


def conditions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    audit = config["hard_set"]
    result: list[dict[str, Any]] = []
    for graph in audit["graphs"]:
        for threshold in audit["pair_thresholds"]:
            deltas = [0.0] if graph == "stability_only" else audit["lift_deltas"]
            for delta in deltas:
                result.append({"graph": str(graph), "pair_threshold": float(threshold), "lift_delta": float(delta)})
    return result


def candidate_map(candidates: Iterable[AuditCandidate]) -> dict[str, AuditCandidate]:
    return {candidate.node_id: candidate for candidate in candidates}


def pair_metric_for_order(metric: PairMetric, a: AuditCandidate, b: AuditCandidate) -> PairMetric:
    """Return a metric whose directed lifts are oriented from supplied a/b."""

    if metric.a.node_id == a.node_id and metric.b.node_id == b.node_id:
        return PairMetric(a, b, metric.q2, metric.a_to_b_lift, metric.b_to_a_lift, metric.residual_mean_tv)
    if metric.a.node_id == b.node_id and metric.b.node_id == a.node_id:
        return PairMetric(a, b, metric.q2, metric.b_to_a_lift, metric.a_to_b_lift, metric.residual_mean_tv)
    raise ValueError("Pair metric candidates do not match requested orientation.")


def metric_from_singletons(
    a: AuditCandidate,
    b: AuditCandidate,
    singleton: Mapping[str, Mapping[str, float]],
    *,
    epsilon: float,
    residual_mean_tv: float | None = None,
) -> PairMetric:
    p_a_given_b = float(singleton[b.node_id][a.node_id])
    p_b_given_a = float(singleton[a.node_id][b.node_id])
    return PairMetric(
        a, b, min(p_a_given_b, p_b_given_a),
        math.log((p_b_given_a + epsilon) / (b.base_probability + epsilon)),
        math.log((p_a_given_b + epsilon) / (a.base_probability + epsilon)),
        residual_mean_tv,
    )


def audit_record_base(
    *, state: Mapping[str, Any], condition: Mapping[str, Any], cohort: str, candidate_set: tuple[AuditCandidate, ...], total_steps: int,
) -> dict[str, Any]:
    unstable = [item for item in candidate_set if item.kind == "unstable"]
    return {
        "state_id": state_id(state), "prompt_index": int(state["prompt_index"]), "step": int(state["step"]),
        "phase": phase(int(state["step"]), total_steps), "cohort": cohort,
        "graph": condition["graph"], "pair_threshold": condition["pair_threshold"], "lift_delta": condition["lift_delta"],
        "set_size": len(candidate_set), "anchor_included": any(item.kind == "anchor" for item in candidate_set),
        "anchor_count": sum(item.kind == "anchor" for item in candidate_set), "unstable_count": len(unstable),
        "mean_base_confidence": sum(item.base_confidence for item in candidate_set) / len(candidate_set),
        "base_confidence_bin": confidence_bin(sum(item.base_confidence for item in candidate_set) / len(candidate_set)),
        "candidates": [candidate_record(item) for item in candidate_set],
    }


class ExactEvaluator:
    """Runs each missing scalar descriptor once and saves it immediately."""

    def __init__(
        self, model: Any, dtype: Any, model_revision: str, cache_schema: str, cache: ExactScalarCache, mask_id: int,
    ) -> None:
        import torch

        self.model = model
        self.dtype = str(dtype)
        self.model_revision = model_revision
        self.cache_schema = cache_schema
        self.cache = cache
        self.mask_id = mask_id
        self.torch = torch
        self.forward_count = 0
        self.forward_seconds = 0.0

    def probabilities(
        self, state: Mapping[str, Any], insertions: Iterable[AuditCandidate], targets: Iterable[AuditCandidate]
    ) -> dict[str, float]:
        insertions = tuple(insertions)
        targets = tuple(targets)
        ids = [int(item) for item in state["token_sequence"]]
        masks = [int(item) for item in state["mask_positions"]]
        keys = {
            target.node_id: scalar_cache_key(
                input_token_ids=ids, mask_positions=masks, insertions=insertions, target=target,
                model_revision=self.model_revision, dtype=self.dtype, cache_schema=self.cache_schema,
            )
            for target in targets
        }
        output = {node: self.cache.get(key) for node, key in keys.items()}
        missing = [target for target in targets if output[target.node_id] is None]
        if missing:
            input_ids = self.torch.tensor([ids], dtype=self.torch.long, device="cuda")
            branch = make_branch(
                input_ids, [Candidate(item.position, item.token_id) for item in insertions], mask_token_id=self.mask_id
            )
            attention = self.torch.ones_like(branch)
            positions = self.torch.arange(branch.shape[1], device=branch.device).unsqueeze(0)
            started = time.perf_counter()
            logits = exact_forward(self.model, branch, attention_mask=attention, position_ids=positions)
            self.torch.cuda.synchronize()
            self.forward_seconds += time.perf_counter() - started
            self.forward_count += 1
            for target in missing:
                probability = float(self.torch.softmax(logits[0, target.position], dim=-1)[target.token_id].item())
                self.cache.put(
                    keys[target.node_id], probability,
                    {
                        "state_id": state_id(state), "target_position": target.position, "target_token_id": target.token_id,
                        "insertion_count": len(insertions), "model_revision": self.model_revision, "dtype": self.dtype,
                        "cache_schema": self.cache_schema,
                    },
                )
                output[target.node_id] = probability
            del logits
        return {node: float(value) for node, value in output.items() if value is not None}


def enrich_tokens(tokenizer: Any, candidates: Iterable[AuditCandidate]) -> list[AuditCandidate]:
    return [candidate_with_token(item, token_repr(tokenizer, item.token_id)) for item in candidates]


def singleton_metrics_for_state(
    state: Mapping[str, Any], candidates: list[AuditCandidate], evaluator: ExactEvaluator,
    existing: Mapping[tuple[str, str], PairMetric], epsilon: float,
) -> tuple[dict[tuple[str, str], PairMetric], list[dict[str, Any]]]:
    """Build every pair metric from exact scalar singletons, preserving residuals from pilot."""

    singleton: dict[str, Mapping[str, float]] = {}
    for inserted in candidates:
        singleton[inserted.node_id] = evaluator.probabilities(state, [inserted], [target for target in candidates if target != inserted])
    metrics: dict[tuple[str, str], PairMetric] = {}
    comparisons: list[dict[str, Any]] = []
    for a, b in combinations(candidates, 2):
        if a.position == b.position:
            continue
        key = pair_key(a, b)
        prior = existing.get(key)
        calculated = metric_from_singletons(a, b, singleton, epsilon=epsilon, residual_mean_tv=prior.residual_mean_tv if prior else None)
        if prior is not None:
            prior_oriented = pair_metric_for_order(prior, a, b)
            comparisons.append({
                "state_id": state_id(state), "pair": list(key),
                "q2_abs_difference": abs(calculated.q2 - prior_oriented.q2),
                "a_to_b_lift_abs_difference": abs(calculated.a_to_b_lift - prior_oriented.a_to_b_lift),
                "b_to_a_lift_abs_difference": abs(calculated.b_to_a_lift - prior_oriented.b_to_a_lift),
            })
            # Graph construction always uses the freshly measured values. The
            # pilot residual is optional stress-mining provenance only; it cannot
            # be reconstructed without persisting a full vocabulary tensor.
            metrics[key] = calculated
        else:
            metrics[key] = calculated
    return metrics, comparisons


def _collect_cohort_sets(
    *, anchors: list[AuditCandidate], unstable: list[AuditCandidate], metrics: Mapping[tuple[str, str], PairMetric],
    condition: Mapping[str, Any], target_size: int, count: int, seeds: list[int], attempts: int, cohort: str,
) -> list[tuple[tuple[AuditCandidate, ...], int]]:
    from src.hard_set_generation import policy_audit_sets

    collected: list[tuple[tuple[AuditCandidate, ...], int]] = []
    seen: set[tuple[str, ...]] = set()
    per_seed = max(1, math.ceil(count / max(1, len(seeds))))
    for seed in seeds:
        if cohort == "policy":
            proposals = policy_audit_sets(
                anchors, unstable, metrics, target_size=target_size, graph=str(condition["graph"]),
                pair_threshold=float(condition["pair_threshold"]), lift_delta=float(condition["lift_delta"]),
                seed=seed, count=per_seed, attempts_per_set=attempts,
            )
        elif cohort == "stress":
            proposals = stress_mined_sets(
                anchors, unstable, metrics, target_size=target_size, graph=str(condition["graph"]),
                pair_threshold=float(condition["pair_threshold"]), lift_delta=float(condition["lift_delta"]), seed=seed, count=per_seed,
            )
        else:
            raise ValueError(f"Unknown cohort {cohort}")
        for proposal in proposals:
            signature = set_signature(proposal)
            if signature in seen:
                continue
            seen.add(signature)
            collected.append((proposal, seed))
            if len(collected) >= count:
                return collected
    return collected


def plan_sets(
    states: list[Mapping[str, Any]], all_candidates: Mapping[str, tuple[list[AuditCandidate], list[AuditCandidate]]],
    all_metrics: Mapping[str, Mapping[tuple[str, str], PairMetric]], config: Mapping[str, Any], mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit = config["hard_set"]
    if mode == "smoke":
        target_sizes = [int(size) for size in audit["target_set_sizes"]]
        requested = int(audit["smoke_sets_per_size"])
        selected_states = states[: int(audit["smoke_state_limit"])]
        selected_conditions = conditions(config)
    else:
        target_sizes = [int(size) for size in audit["target_set_sizes"]]
        selected_states = states
        selected_conditions = conditions(config)
    plans: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    seeds = [int(seed) for seed in audit["policy_seeds"]]
    for condition in selected_conditions:
        for state in selected_states:
            identifier = state_id(state)
            anchors, unstable = all_candidates[identifier]
            metrics = all_metrics[identifier]
            conflicts = anchor_conflicts(
                anchors, metrics, graph=str(condition["graph"]), pair_threshold=float(condition["pair_threshold"]), lift_delta=float(condition["lift_delta"])
            )
            conflict_details = []
            for conflict in conflicts:
                if conflict is None:
                    conflict_details.append({"missing_pair_metric": True})
                else:
                    conflict_details.append(pair_record(conflict))
            conflict_rows.append({
                "state_id": identifier, "prompt_index": state["prompt_index"], "step": state["step"], **condition,
                "anchor_count": len(anchors), "anchor_pair_count": len(anchors) * (len(anchors) - 1) // 2,
                "failed_anchor_pair_count": len(conflicts), "anchor_conflict": bool(conflicts),
                "conflict_pairs": conflict_details,
            })
            if conflicts:
                continue
            # The count is divided across viable states; deduplication remains
            # per-state to avoid treating repeat state identities as iid draws.
            for size in target_sizes:
                for cohort in ("policy", "stress"):
                    if mode == "smoke":
                        requested = int(audit["smoke_sets_per_size"])
                    elif (
                        cohort == "policy" and condition["graph"] == "stability_only"
                        and float(condition["pair_threshold"]) == float(audit["primary_pair_threshold"])
                    ):
                        requested = int(audit["target_audited_sets_per_size"])
                    elif cohort == "stress" and condition["graph"] == "stability_only" and float(condition["pair_threshold"]) == float(audit["primary_pair_threshold"]):
                        requested = int(audit["primary_stress_sets_per_size"])
                    else:
                        requested = int(audit["non_primary_sets_per_size"])
                    if requested > int(audit["max_audited_sets_per_size"]):
                        raise ValueError("A requested hard-set count exceeds max_audited_sets_per_size.")
                    per_state = max(1, math.ceil(requested / max(1, len(selected_states))))
                    for candidate_set, seed in _collect_cohort_sets(
                        anchors=anchors, unstable=unstable, metrics=metrics, condition=condition, target_size=size,
                        count=per_state, seeds=seeds, attempts=int(audit["attempts_per_set"]), cohort=cohort,
                    ):
                        if not is_clique(
                            candidate_set, metrics, graph=str(condition["graph"]), pair_threshold=float(condition["pair_threshold"]), lift_delta=float(condition["lift_delta"])
                        ):
                            raise AssertionError("Generator emitted a non-clique set.")
                        plans.append({
                            "state_id": identifier, "condition": dict(condition), "cohort": cohort, "policy_seed": seed,
                            "candidate_set": candidate_set, "stress_score": stress_score(candidate_set, metrics, pair_threshold=float(condition["pair_threshold"])),
                        })
    return plans, conflict_rows


def estimate_seconds(
    plans: Iterable[Mapping[str, Any]], seconds_per_forward: float, singleton_forwards: int, estimated_peak_vram_mib: float,
) -> dict[str, Any]:
    plan_list = list(plans)
    loo_forwards = sum(len(item["candidate_set"]) for item in plan_list)
    return {
        "planned_sets": len(plan_list), "planned_leave_one_out_forwards": loo_forwards,
        "singleton_forwards_observed_or_required": singleton_forwards,
        "seconds_per_exact_forward": seconds_per_forward,
        "estimated_peak_vram_mib": estimated_peak_vram_mib,
        "estimated_seconds": (loo_forwards + singleton_forwards) * seconds_per_forward,
        "estimated_minutes": (loo_forwards + singleton_forwards) * seconds_per_forward / 60,
    }


def audit_sets(
    plans: Iterable[Mapping[str, Any]], state_by_id: Mapping[str, Mapping[str, Any]],
    metrics_by_state: Mapping[str, Mapping[tuple[str, str], PairMetric]], evaluator: ExactEvaluator, total_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    loo_records: list[dict[str, Any]] = []
    for plan in plans:
        state = state_by_id[plan["state_id"]]
        candidate_set = tuple(plan["candidate_set"])
        candidate_branches = make_leave_one_out_branches(
            evaluator.torch.tensor([[int(item) for item in state["token_sequence"]]], dtype=evaluator.torch.long, device="cuda"),
            [Candidate(item.position, item.token_id) for item in candidate_set], mask_token_id=evaluator.mask_id,
        )
        # Branch construction above asserts mask positions and one token per position.
        del candidate_branches
        loo: dict[str, float] = {}
        for omitted in candidate_set:
            inserted = tuple(item for item in candidate_set if item != omitted)
            loo[omitted.node_id] = evaluator.probabilities(state, inserted, [omitted])[omitted.node_id]
        metrics = metrics_by_state[plan["state_id"]]
        contained_metrics = [metrics[pair_key(a, b)] for a, b in combinations(candidate_set, 2)]
        pair_values = [metric.q2 for metric in contained_metrics]
        q_pair = min(pair_values)
        q_set = min(loo.values())
        worst = min(candidate_set, key=lambda item: loo[item.node_id])
        base = audit_record_base(
            state=state, condition=plan["condition"], cohort=str(plan["cohort"]), candidate_set=candidate_set, total_steps=total_steps,
        )
        base.update({
            "policy_seed": plan["policy_seed"], "stress_score": plan["stress_score"], "q_pair": q_pair, "q_set": q_set,
            "gap": q_pair - q_set, "pair_safe_set_unsafe": bool(q_set < float(plan["condition"]["pair_threshold"])),
            "worst_candidate": candidate_record(worst), "worst_candidate_loo_probability": loo[worst.node_id],
            "pair_metrics": [pair_record(metric) for metric in contained_metrics],
            "exact_policy": "full independent branch; use_cache=False",
        })
        records.append(base)
        loo_records.extend({
            "state_id": plan["state_id"], "cohort": plan["cohort"], **plan["condition"],
            "set_signature": list(set_signature(candidate_set)), "omitted": candidate_record(omitted),
            "loo_probability": loo[omitted.node_id], "use_cache": False,
        } for omitted in candidate_set)
    return records, loo_records


def make_figures(output_dir: Path, audit_rows: list[Mapping[str, Any]], conflict_rows: list[Mapping[str, Any]]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    created: list[str] = []
    if audit_rows:
        by_size: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in audit_rows:
            by_size[int(row["set_size"])].append(row)
        figure, axis = plt.subplots(figsize=(6, 4))
        sizes = sorted(by_size)
        rates = [sum(bool(row["pair_safe_set_unsafe"]) for row in by_size[size]) / len(by_size[size]) for size in sizes]
        axis.plot(sizes, rates, marker="o", color="#8a4f7d")
        axis.set_xlabel("set size")
        axis.set_ylabel("pair-safe / set-unsafe rate")
        axis.set_ylim(0, 1)
        figure.tight_layout()
        figure.savefig(output_dir / "failure_rate_vs_set_size.png", dpi=160)
        plt.close(figure)
        created.append("failure_rate_vs_set_size.png")

        figure, axis = plt.subplots(figsize=(6, 4))
        colors = ["#b33636" if row["pair_safe_set_unsafe"] else "#356aa0" for row in audit_rows]
        axis.scatter([row["q_pair"] for row in audit_rows], [row["q_set"] for row in audit_rows], s=18, alpha=0.75, c=colors)
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        axis.set_xlabel("Qpair(S)")
        axis.set_ylabel("Q(S)")
        figure.tight_layout()
        figure.savefig(output_dir / "qpair_vs_qset.png", dpi=160)
        plt.close(figure)
        created.append("qpair_vs_qset.png")

        values = [float(row["gap"]) for row in audit_rows]
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(values, bins=min(30, max(5, len(values))), color="#8a4f7d")
        axis.set_xlabel("Qpair − Q(S)")
        axis.set_ylabel("audited sets")
        figure.tight_layout()
        figure.savefig(output_dir / "failure_gap_distribution.png", dpi=160)
        plt.close(figure)
        created.append("failure_gap_distribution.png")
    if conflict_rows:
        by_threshold: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
        for row in conflict_rows:
            by_threshold[float(row["pair_threshold"])].append(row)
        figure, axis = plt.subplots(figsize=(6, 4))
        thresholds = sorted(by_threshold)
        rates = [sum(bool(row["anchor_conflict"]) for row in by_threshold[value]) / len(by_threshold[value]) for value in thresholds]
        axis.bar([str(value) for value in thresholds], rates, color="#a05a2c")
        axis.set_xlabel("pair threshold")
        axis.set_ylabel("anchor-conflict state rate")
        axis.set_ylim(0, 1)
        figure.tight_layout()
        figure.savefig(output_dir / "anchor_conflict_by_threshold.png", dpi=160)
        plt.close(figure)
        created.append("anchor_conflict_by_threshold.png")
    return created


def write_reports(
    probe_root: Path, mode: str, audit_rows: list[dict[str, Any]], conflict_rows: list[dict[str, Any]],
    loo_rows: list[dict[str, Any]], plans: Iterable[Mapping[str, Any]], state_by_id: Mapping[str, Mapping[str, Any]],
    metrics_by_state: Mapping[str, Mapping[tuple[str, str], PairMetric]], total_steps: int, estimate: Mapping[str, Any],
    provenance: Mapping[str, Any], run_result: Mapping[str, Any],
) -> None:
    root = probe_root / "outputs" / "hard_set"
    raw = root / "raw"
    tables = root / "tables"
    figures = root / "figures"
    metadata: list[dict[str, Any]] = []
    for plan in plans:
        row = audit_record_base(
            state=state_by_id[plan["state_id"]], condition=plan["condition"], cohort=str(plan["cohort"]),
            candidate_set=tuple(plan["candidate_set"]), total_steps=total_steps,
        )
        contained = [
            metrics_by_state[plan["state_id"]][pair_key(a, b)]
            for a, b in combinations(tuple(plan["candidate_set"]), 2)
        ]
        row["pair_metrics"] = [pair_record(metric) for metric in contained]
        row["q_pair"] = min(metric.q2 for metric in contained)
        row.update({"policy_seed": plan["policy_seed"], "stress_score": plan["stress_score"], "audit_completed": False})
        metadata.append(row)
    audited_by_signature = {
        (row["state_id"], row["cohort"], row["graph"], row["pair_threshold"], row["lift_delta"], tuple(sorted(f"{item['position']}:{item['token_id']}" for item in row["candidates"])))
        for row in audit_rows
    }
    for row in metadata:
        signature = (
            row["state_id"], row["cohort"], row["graph"], row["pair_threshold"], row["lift_delta"],
            tuple(sorted(f"{item['position']}:{item['token_id']}" for item in row["candidates"])),
        )
        row["audit_completed"] = signature in audited_by_signature
    write_jsonl(raw / "policy_set_metadata.jsonl", [row for row in metadata if row["cohort"] == "policy"])
    write_jsonl(raw / "stress_set_metadata.jsonl", [row for row in metadata if row["cohort"] == "stress"])
    write_jsonl(raw / f"{mode}_policy_sets.jsonl", [row for row in audit_rows if row["cohort"] == "policy"])
    write_jsonl(raw / f"{mode}_stress_sets.jsonl", [row for row in audit_rows if row["cohort"] == "stress"])
    write_jsonl(raw / f"{mode}_leave_one_out.jsonl", loo_rows)
    write_jsonl(raw / f"{mode}_anchor_conflicts.jsonl", conflict_rows)
    # Canonical latest-result names match the documented artifact layout.  If
    # a budget guard stopped before LOO, they retain the generated metadata so
    # the operator can inspect exactly what was estimated.
    write_jsonl(raw / "policy_audit_sets.jsonl", [row for row in audit_rows if row["cohort"] == "policy"] or [row for row in metadata if row["cohort"] == "policy"])
    write_jsonl(raw / "stress_mined_sets.jsonl", [row for row in audit_rows if row["cohort"] == "stress"] or [row for row in metadata if row["cohort"] == "stress"])
    write_jsonl(raw / "set_loo_results.jsonl", loo_rows)
    write_jsonl(raw / "anchor_conflicts.jsonl", conflict_rows)
    write_csv(tables / "failure_rate_by_size.csv", summarize_failures(audit_rows, ("cohort", "graph", "pair_threshold", "lift_delta", "set_size")))
    write_csv(tables / "failure_rate_by_threshold.csv", summarize_failures(audit_rows, ("cohort", "pair_threshold", "lift_delta")))
    write_csv(tables / "failure_rate_by_graph.csv", summarize_failures(audit_rows, ("cohort", "graph")))
    write_csv(tables / "failure_severity.csv", summarize_failures(
        audit_rows, ("cohort", "graph", "pair_threshold", "lift_delta", "set_size", "anchor_included", "anchor_count", "unstable_count", "phase", "base_confidence_bin")
    ))
    conflicts = []
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in conflict_rows:
        groups[(row["graph"], row["pair_threshold"], row["lift_delta"])].append(row)
    for key, rows in groups.items():
        eligible = [row for row in rows if int(row["anchor_count"]) >= 2]
        count = sum(bool(row["anchor_conflict"]) for row in eligible)
        low, high = wilson_interval(count, len(eligible))
        conflicts.append({
            "graph": key[0], "pair_threshold": key[1], "lift_delta": key[2], "state_count": len(rows),
            "states_with_two_or_more_anchors": len(eligible), "anchor_conflict_count": count,
            "anchor_conflict_rate": count / len(eligible) if eligible else None, "ci95_low": low, "ci95_high": high,
            "rule_of_three_upper_95": 3 / len(eligible) if eligible and count == 0 else None,
        })
    write_csv(tables / "anchor_conflict_rates.csv", conflicts)
    figures.mkdir(parents=True, exist_ok=True)
    figure_paths = make_figures(figures, audit_rows, conflict_rows)
    write_json(root / f"{mode}_resource_estimate.json", dict(estimate))
    resource_text = (
        "# Hard-set audit resource estimate\n\n"
        f"- Planned hard sets: {estimate['planned_sets']}\n"
        f"- Planned LOO exact forwards: {estimate['planned_leave_one_out_forwards']}\n"
        f"- Estimated exact-forward time: {estimate['seconds_per_exact_forward']:.4f} s\n"
        f"- Estimated total runtime: {estimate['estimated_minutes']:.2f} min\n"
        f"- Estimated peak VRAM: {estimate['estimated_peak_vram_mib']:.1f} MiB\n"
    )
    (root / f"{mode}_resource_estimate.md").write_text(resource_text, encoding="utf-8")
    (root / "resource_estimate.md").write_text(resource_text, encoding="utf-8")
    write_json(root / f"{mode}_run_result.json", dict(run_result))
    lines = [
        "# Hard-set audit summary", "",
        f"- Mode: `{mode}`; status: `{run_result['status']}`.",
        f"- Planned/audited sets: {estimate['planned_sets']}/{len(audit_rows)}; estimated runtime: {estimate['estimated_minutes']:.2f} min.",
        f"- Exact counterfactual policy: every singleton and LOO scalar is an independent `use_cache=False` forward; the cache stores only scalar target probabilities.",
        f"- Provenance: Fast-dLLM pin match={provenance['fast_dllm_pin_match']}; pilot source match={provenance['pilot_source_commit_match']}.",
        f"- Raw: `outputs/hard_set/raw/`; tables: `outputs/hard_set/tables/`; figures: {', '.join(figure_paths) or 'none'}.",
        "- Policy cohort is a sequential pair-rule policy estimate; stress cohort is deliberately mined and is not a population-rate estimate.",
        "- A zero observed failure is reported with Wilson intervals and a rule-of-three upper bound; it does not prove universal set safety.",
    ]
    primary = [
        row for row in audit_rows
        if row["cohort"] == "policy" and row["graph"] == "stability_only" and float(row["pair_threshold"]) == 0.7
    ]
    if primary:
        failures = sum(bool(row["pair_safe_set_unsafe"]) for row in primary)
        lines.extend([
            "", "## Direct answers", "",
            f"1. At the operational threshold 0.7, the policy-audit stability-only cohort had {failures}/{len(primary)} observed pair-safe/set-unsafe failures ({failures / len(primary):.3%}).",
        ])
        by_size: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in primary:
            by_size[int(row["set_size"])].append(row)
        size_text = "; ".join(
            f"|S|={size}: {sum(bool(row['pair_safe_set_unsafe']) for row in rows)}/{len(rows)} failures, mean gap={sum(float(row['gap']) for row in rows) / len(rows):.4f}"
            for size, rows in sorted(by_size.items())
        )
        lines.append(f"2. Size-stratified observed rates/gaps: {size_text}. This descriptive sample alone does not establish a monotone trend.")
        strict = [
            row for row in audit_rows
            if row["cohort"] == "policy" and row["graph"] == "strict_mutual_support"
            and float(row["pair_threshold"]) == 0.7 and float(row["lift_delta"]) == 0.0
        ]
        if strict:
            strict_failures = sum(bool(row["pair_safe_set_unsafe"]) for row in strict)
            lines.append(f"3. Strict mutual support at tau=0.7, delta=0.0: {strict_failures}/{len(strict)} failures ({strict_failures / len(strict):.3%}); its selection distribution differs from Graph 1, so this is not a matched causal comparison.")
        else:
            lines.append("3. Strict mutual-support comparison has no feasible audited sets at tau=0.7, delta=0.0.")
        eligible = [row for row in conflict_rows if int(row["anchor_count"]) >= 2 and float(row["pair_threshold"]) == 0.7]
        anchor_failures = sum(bool(row["anchor_conflict"]) for row in eligible)
        lines.append(f"4. At tau=0.7, {anchor_failures}/{len(eligible)} states with at least two anchors had an anchor-pair conflict (counted separately from the primary cohort).")
        lines.append("5. These results measure conditional self-consistency only. Even observed hard failures do not by themselves justify a practical referee; that needs sufficient incidence at operational thresholds and real set sizes. Conversely, zero failures do not establish pair-only sufficiency beyond the reported confidence bounds.")
    else:
        lines.extend([
            "", "## Direct answers", "",
            "1–5. No completed policy-audit sets at tau=0.7 are available in this run, so the operational failure rate, size trend, strict-graph comparison, and referee conclusion are not yet estimable. See the resource/status record rather than treating this as evidence of safety.",
        ])
    if provenance.get("warning"):
        lines.append(f"- Provenance limitation: {provenance['warning']}")
    if run_result["status"] == "blocked_over_30_minutes":
        lines.append("- No full LOO audit was run because the pre-run estimate exceeded the 30-minute shared-GPU policy.")
    text = "\n".join(lines) + "\n"
    (root / f"{mode}_summary.md").write_text(text, encoding="utf-8")
    (root / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "audit"), required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    states, pairs, pilot_result = validate_pilot_artifacts(args.probe_root, config["pilot_reuse"])
    provenance = provenance_validation(args.probe_root, config["pilot_reuse"])
    if not provenance["fast_dllm_pin_match"] or not provenance["pilot_source_commit_match"]:
        raise RuntimeError(
            "Pilot provenance does not match the configured Fast-dLLM/source pins; do not reuse these scalar artifacts. "
            "Restore the matching pilot summary or rerun the pilot under the reviewed configuration."
        )
    output_root = args.probe_root / "outputs" / "hard_set"
    if args.mode == "audit":
        smoke_path = output_root / "smoke_run_result.json"
        if not smoke_path.exists():
            raise RuntimeError("Hard-set audit requires a successful hard-set smoke run; execute bash scripts/run_hard_set_smoke.sh first.")
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke.get("status") != "success":
            raise RuntimeError(f"Hard-set audit is blocked by smoke status: {smoke.get('status')!r}.")
        if smoke.get("git_commit") != git_commit(args.probe_root):
            raise RuntimeError("Source changed after hard-set smoke; rerun bash scripts/run_hard_set_smoke.sh before auditing.")
    if args.mode == "smoke":
        states = states[: int(config["hard_set"]["smoke_state_limit"])]
    output_root.mkdir(parents=True, exist_ok=True)

    import torch

    gpu_before = gpu_memory_snapshot()
    model, tokenizer, dtype = load_model(config, args.probe_root)
    mask_id = mask_token_id(model, tokenizer)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    cache = ExactScalarCache(output_root / "raw" / "exact_scalar_cache.jsonl")
    model_revision = f"{config['model']['hf_revision']}|fast-dllm:{config['model']['fast_dllm_commit']}"
    evaluator = ExactEvaluator(
        model, dtype, model_revision, str(config["hard_set"]["exact_scalar_cache_schema"]), cache, mask_id,
    )
    existing_by_state = pilot_pair_metrics(states, pairs)
    candidates_by_state: dict[str, tuple[list[AuditCandidate], list[AuditCandidate]]] = {}
    metrics_by_state: dict[str, dict[tuple[str, str], PairMetric]] = {}
    comparisons: list[dict[str, Any]] = []
    for state in states:
        anchors, unstable = candidates_from_state(
            state, threshold=float(config["decoding"]["threshold"]), tau_mass=float(config["decoding"]["tau_mass"])
        )
        anchors = enrich_tokens(tokenizer, anchors)
        unstable = enrich_tokens(tokenizer, unstable)
        all_candidates = anchors + unstable
        candidates_by_state[state_id(state)] = (anchors, unstable)
        metrics, state_comparisons = singleton_metrics_for_state(
            state, all_candidates, evaluator, existing_by_state.get(state_id(state), {}),
            epsilon=float(config["hard_set"]["epsilon"])
        )
        metrics_by_state[state_id(state)] = metrics
        comparisons.extend(state_comparisons)
    q2_tolerance = float(config["hard_set"]["singleton_q2_match_tolerance"])
    lift_tolerance = float(config["hard_set"]["singleton_lift_match_tolerance"])
    bad_comparisons = [
        row for row in comparisons
        if row["q2_abs_difference"] > q2_tolerance
        or row["a_to_b_lift_abs_difference"] > lift_tolerance
        or row["b_to_a_lift_abs_difference"] > lift_tolerance
    ]
    write_jsonl(output_root / "raw" / f"{args.mode}_singleton_reuse_comparison.jsonl", comparisons)
    plans, conflict_rows = plan_sets(states, candidates_by_state, metrics_by_state, config, args.mode)
    seconds_per_forward = evaluator.forward_seconds / evaluator.forward_count if evaluator.forward_count else (
        float(pilot_result["elapsed_seconds"]) / max(1, int(pilot_result["exact_forward_count"]))
    )
    estimate = estimate_seconds(
        plans, seconds_per_forward, evaluator.forward_count,
        max(float(pilot_result.get("peak_vram_mib", 0.0)), float(torch.cuda.max_memory_allocated() / 2**20)),
    )
    state_by_id = {state_id(state): state for state in states}
    max_seconds = float(config["hard_set"]["max_runtime_seconds"])
    if bad_comparisons:
        status = "blocked_singleton_mismatch"
        audit_rows: list[dict[str, Any]] = []
        loo_rows: list[dict[str, Any]] = []
    elif args.mode == "audit" and estimate["estimated_seconds"] > max_seconds:
        status = "blocked_over_30_minutes"
        audit_rows = []
        loo_rows = []
    else:
        audit_rows, loo_rows = audit_sets(
            plans, state_by_id, metrics_by_state, evaluator, int(config["decoding"]["steps"])
        )
        status = "success" if audit_rows else "blocked_no_feasible_hard_sets"
    torch.cuda.synchronize()
    run_result = {
        "mode": args.mode, "status": status, "dtype": str(dtype), "elapsed_seconds": time.perf_counter() - started,
        "exact_forward_seconds": evaluator.forward_seconds,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20, "gpu_memory_before": gpu_before,
        "gpu_memory_after": gpu_memory_snapshot(), "exact_forward_count": evaluator.forward_count,
        "scalar_cache_entries": len(cache.entries), "states_examined": len(states), "planned_sets": len(plans),
        "audited_sets": len(audit_rows), "singleton_reuse_pairs_compared": len(comparisons),
        "singleton_reuse_pairs_outside_tolerance": len(bad_comparisons), "git_commit": git_commit(args.probe_root),
        "singleton_q2_match_tolerance": q2_tolerance, "singleton_lift_match_tolerance": lift_tolerance,
        "max_reuse_q2_abs_difference": max((row["q2_abs_difference"] for row in comparisons), default=0.0),
        "max_reuse_lift_abs_difference": max(
            (max(row["a_to_b_lift_abs_difference"], row["b_to_a_lift_abs_difference"]) for row in comparisons), default=0.0
        ),
        "cache_policy": "Each cache key covers full IDs, mask positions, insertions, model revision, dtype, cache schema, target, and use_cache=False.",
    }
    write_reports(
        args.probe_root, args.mode, audit_rows, conflict_rows, loo_rows, plans, state_by_id, metrics_by_state,
        int(config["decoding"]["steps"]), estimate, provenance, run_result,
    )
    print(json.dumps(run_result, ensure_ascii=False))


if __name__ == "__main__":
    main()
