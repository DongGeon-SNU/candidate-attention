#!/usr/bin/env python3
"""Audit candidate-value polarity only at predeclared high-top-5-mass branch points.

The natural trajectory stored by ``top1_dynamics_audit`` is the sole control
rollout.  For every representative *ordered* source--target pair, this runner
finds the first saved control step where both positions remain masked and the
source's exact full-vocabulary top-5 mass is strictly above the configured
gate.  It freezes that source top-5 list and its probabilities.  Every
treatment is then an independent fixed-length, ``use_cache=False`` branch of
that identical control snapshot, differing only by one forced source token.

No trajectory is continued after a treatment and no treatment is allowed to
alter control-state selection.  This deliberately preserves the previous
context-only candidate-polarity estimand while adding only the requested
eligibility condition for t*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
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
    SelectedState,
    choose_states,
    exact_logits_batched,
    margin_summary,
    read_jsonl,
    select_polarity_pairs,
    topk_summary,
    write_csv,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = 1
CONTROL_POLICY = "confidence_ge_threshold_plus_argmax_fallback"


@dataclass(frozen=True)
class OrderedPairRequest:
    """One directed pair inherited from the prior deterministic polarity sample."""

    selected_state: SelectedState
    source_position: int
    target_position: int
    source_selection_position: int
    target_selection_position: int


@dataclass(frozen=True)
class BranchPoint:
    """The first eligible unmodified control snapshot for one ordered pair."""

    request: OrderedPairRequest
    record: Mapping[str, Any]
    source_top5_token_ids: tuple[int, ...]
    source_top5_probabilities: tuple[float, ...]
    source_top5_mass: float
    source_p1: float


def _finite_probability_list(value: Any, *, count: int) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < count:
        return None
    result = tuple(float(item) for item in value[:count])
    return result if all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in result) else None


def _source_summary(record: Mapping[str, Any], position: int) -> Mapping[str, Any] | None:
    summaries = record.get("position_summaries")
    if not isinstance(summaries, Mapping):
        return None
    summary = summaries.get(str(int(position)))
    return summary if isinstance(summary, Mapping) else None


def find_first_branchpoint(
    request: OrderedPairRequest,
    records: Sequence[Mapping[str, Any]],
    *,
    strict_top5_mass_gate: float,
) -> tuple[BranchPoint | None, str]:
    """Return the earliest exact saved state that satisfies the t* definition.

    The strict comparison is intentional: the protocol states ``M5 > 0.9``,
    so a value exactly equal to the configured boundary remains ineligible.
    """

    source = int(request.source_position)
    target = int(request.target_position)
    saw_both_masked = False
    for record in sorted(records, key=lambda row: int(row.get("step", -1))):
        masks = record.get("mask_positions")
        if not isinstance(masks, Sequence) or isinstance(masks, (str, bytes)):
            continue
        mask_positions = {int(item) for item in masks}
        if source not in mask_positions or target not in mask_positions:
            continue
        saw_both_masked = True
        summary = _source_summary(record, source)
        if summary is None:
            continue
        try:
            mass = float(summary["top5_mass"])
            p1 = float(summary["top1_probability"])
        except (KeyError, TypeError, ValueError):
            continue
        tokens_raw = summary.get("top_token_ids")
        if not isinstance(tokens_raw, Sequence) or isinstance(tokens_raw, (str, bytes)) or len(tokens_raw) < 5:
            continue
        probabilities = _finite_probability_list(summary.get("top_probabilities"), count=5)
        if probabilities is None or not math.isfinite(mass) or not math.isfinite(p1):
            continue
        # The stored scalar is primary, while this check catches a corrupt or
        # schema-incompatible record before it can silently freeze a different
        # candidate distribution.
        if abs(sum(probabilities) - mass) > 2.0e-5:
            continue
        if mass <= float(strict_top5_mass_gate):
            continue
        return (
            BranchPoint(
                request=request,
                record=record,
                source_top5_token_ids=tuple(int(token) for token in tokens_raw[:5]),
                source_top5_probabilities=probabilities,
                source_top5_mass=mass,
                source_p1=p1,
            ),
            "retained",
        )
    return None, "no_step_with_both_positions_masked" if not saw_both_masked else "top5_mass_gate_not_met_before_commit"


def pair_requests(selected: Sequence[SelectedState], count: int) -> list[OrderedPairRequest]:
    """Reuse the prior stable-hash representative *unordered* pair selection."""

    output: list[OrderedPairRequest] = []
    for selected_state, left, right in select_polarity_pairs(selected, count):
        left_position = int(selected_state.candidates[left].position)
        right_position = int(selected_state.candidates[right].position)
        output.extend(
            (
                OrderedPairRequest(selected_state, left_position, right_position, left, right),
                OrderedPairRequest(selected_state, right_position, left_position, right, left),
            )
        )
    return output


def trajectory_group_key(record: Mapping[str, Any]) -> tuple[str, str, float]:
    return (
        str(record.get("prompt_id")),
        str(record.get("setting")),
        float(record.get("threshold", math.nan)),
    )


def request_group_key(request: OrderedPairRequest) -> tuple[str, str, float]:
    return trajectory_group_key(request.selected_state.record)


def validate_control_provenance(record: Mapping[str, Any], *, threshold: float) -> str | None:
    """Check that this is an untouched confidence-plus-fallback control state.

    The source log contains exact no-cache summaries but its actions were
    already chosen by the original collection pass.  We therefore validate
    the stored policy declaration and anchor provenance rather than incorrectly
    recomputing an action from the exact replay and calling it the control.
    """

    metadata = record.get("decoder_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("natural_policy") != CONTROL_POLICY:
        return "missing_or_unexpected_natural_policy_provenance"
    anchors = record.get("actual_committed_anchors")
    if not isinstance(anchors, list):
        return "missing_actual_committed_anchors"
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            return "invalid_anchor_provenance"
        try:
            confidence = float(anchor["confidence"])
        except (KeyError, TypeError, ValueError):
            return "missing_anchor_confidence"
        eligible = bool(anchor.get("threshold_eligible"))
        fallback = bool(anchor.get("selected_by_fallback"))
        if eligible and confidence < threshold:
            return "threshold_anchor_below_threshold"
        if not eligible and not fallback:
            return "non_threshold_anchor_without_fallback"
    return None


def _target_candidates(topk: Mapping[str, Any], mode: str, config: Mapping[str, Any]) -> list[int]:
    tokens = [int(token) for token in topk["top_token_ids"]]
    if mode == "fixed_k4":
        return tokens[:4]
    if mode == "fixed_k8":
        return tokens[:8]
    if mode != "adaptive_mass":
        raise ValueError(f"Unknown target candidate mode: {mode}")
    mass = 0.0
    result: list[int] = []
    for token, probability in zip(tokens[: int(config["sampling"]["adaptive_k_max"])], topk["top_probabilities"]):
        result.append(token)
        mass += float(probability)
        if mass >= float(config["sampling"]["adaptive_mass"]):
            break
    return result


def _common(branchpoint: BranchPoint) -> dict[str, Any]:
    record = branchpoint.record
    request = branchpoint.request
    selected = request.selected_state
    return {
        "selection_state_key": selected.record["state_key"],
        "branchpoint_state_key": record["state_key"],
        "cohort": selected.cohort,
        "selection_reason": selected.selection_reason,
        "dataset": record.get("dataset"),
        "prompt_id": record.get("prompt_id"),
        "prompt_index": record.get("prompt_index"),
        "selection_step": selected.record.get("step"),
        "tstar_step": record.get("step"),
        "source_position": request.source_position,
        "target_position": request.target_position,
        "source_selection_index": request.source_selection_position,
        "target_selection_index": request.target_selection_position,
        "source_top5_mass": branchpoint.source_top5_mass,
        "source_p1": branchpoint.source_p1,
        "source_top5_candidate_token_ids": list(branchpoint.source_top5_token_ids),
        "source_top5_candidate_probabilities": list(branchpoint.source_top5_probabilities),
        "source_top5_probability_sum": sum(branchpoint.source_top5_probabilities),
        "control_policy": CONTROL_POLICY,
        "control_trajectory_reused_unchanged": True,
        "treatment_definition": "same_tstar_context_force_one_frozen_source_top5_token_use_cache_false",
    }


def evaluate_branchpoint(
    model: Any,
    branchpoint: BranchPoint,
    *,
    mask_token_id: int,
    config: Mapping[str, Any],
    accounting: ForwardAccounting,
) -> list[dict[str, Any]]:
    """Evaluate all frozen source top-5 treatments from exactly one t* input."""

    import torch

    record = branchpoint.record
    source = branchpoint.request.source_position
    target = branchpoint.request.target_position
    base = torch.tensor([record["token_sequence"]], device=next(model.parameters()).device, dtype=torch.long)
    if int(base[0, source].item()) != int(mask_token_id) or int(base[0, target].item()) != int(mask_token_id):
        raise RuntimeError(f"{record['state_key']}: retained t* does not keep both source and target masked")
    base_logits = exact_logits_batched(model, base)
    accounting.batch_calls += 1
    accounting.branch_forwards += 1
    accounting.polarity_branch_forwards += 1
    source_exact = topk_summary(base_logits[0], source, 5)
    target_exact = topk_summary(base_logits[0], target, 16)
    frozen_tokens = list(branchpoint.source_top5_token_ids)
    if tuple(int(token) for token in source_exact["top_token_ids"][:5]) != tuple(frozen_tokens):
        raise RuntimeError(
            f"{record['state_key']}: exact t* source top-5 differs from the stored frozen control list; refusing mixed branches"
        )
    candidate_logits: dict[int, Any] = {}
    batch_size = int(config["execution"]["treatment_batch_size"])
    for start in range(0, len(frozen_tokens), batch_size):
        tokens = frozen_tokens[start : start + batch_size]
        branches = base.expand(len(tokens), -1).clone()
        for row, token in enumerate(tokens):
            branches[row, source] = token
        logits = exact_logits_batched(model, branches)
        accounting.batch_calls += 1
        accounting.branch_forwards += len(tokens)
        accounting.polarity_branch_forwards += len(tokens)
        for row, token in enumerate(tokens):
            candidate_logits[token] = logits[row].detach()
        del logits, branches

    common = _common(branchpoint)
    rows: list[dict[str, Any]] = []
    target_ranks = {token: rank + 1 for rank, token in enumerate(target_exact["top_token_ids"])}
    tolerance = float(config["certificate"]["tie_tolerance"])
    for mode in config["sampling"]["target_candidate_modes"]:
        target_values = _target_candidates(target_exact, str(mode), config)
        for source_rank, (source_token, source_probability) in enumerate(
            zip(branchpoint.source_top5_token_ids, branchpoint.source_top5_probabilities), 1
        ):
            after_logits = candidate_logits[source_token]
            for target_token in target_values:
                base_margin = margin_summary(base_logits[0], target, target_token, tie_tolerance=tolerance)
                after = margin_summary(after_logits, target, target_token, tie_tolerance=tolerance)
                delta = float(after["logit_margin"] - base_margin["logit_margin"])
                rows.append(
                    {
                        **common,
                        "candidate_set_mode": str(mode),
                        "source_candidate_token_id": source_token,
                        "source_candidate_rank": source_rank,
                        "source_candidate_probability": source_probability,
                        "target_candidate_token_id": target_token,
                        "target_candidate_rank": target_ranks.get(target_token),
                        "base_logit_margin": base_margin["logit_margin"],
                        "intervened_logit_margin": after["logit_margin"],
                        "delta_logit_margin": delta,
                        "polarity": "supportive" if delta >= 0.0 else "destructive",
                        "target_candidate_is_top1_after": bool(after["top1_matches_assignment"]),
                        "after_top1_token_id": after["top1_token_id"],
                        "base_target_top1_probability": target_exact["top1_probability"],
                        "exact_source_top5_mass_check": sum(source_exact["top_probabilities"][:5]),
                    }
                )
    del base_logits, base
    for value in candidate_logits.values():
        del value
    return rows


def polarity_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["branchpoint_state_key"]), str(row["candidate_set_mode"]))].append(row)
    result: list[dict[str, Any]] = []
    for (state_key, mode), members in sorted(grouped.items()):
        values = [float(row["delta_logit_margin"]) for row in members]
        supportive = sum(value >= 0.0 for value in values)
        destructive = len(values) - supportive
        source = members[0]
        result.append(
            {
                "branchpoint_state_key": state_key,
                "candidate_set_mode": mode,
                "cohort": source["cohort"],
                "prompt_id": source["prompt_id"],
                "tstar_step": source["tstar_step"],
                "source_position": source["source_position"],
                "target_position": source["target_position"],
                "source_top5_mass": source["source_top5_mass"],
                "source_p1": source["source_p1"],
                "matrix_cell_count": len(values),
                "supportive_cell_count": supportive,
                "destructive_cell_count": destructive,
                "contains_both_polarities": supportive > 0 and destructive > 0,
                "delta_min": min(values) if values else None,
                "delta_max": max(values) if values else None,
                "delta_range": max(values) - min(values) if values else None,
            }
        )
    return result


def write_report(
    output_root: Path,
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    retained = [row for row in pair_rows if row.get("status") == "retained"]
    skipped = [row for row in pair_rows if row.get("status") != "retained"]
    fixed_k4 = [row for row in summary_rows if row.get("candidate_set_mode") == "fixed_k4"]
    both = sum(bool(row["contains_both_polarities"]) for row in fixed_k4)
    direct = {
        "ordered_pair_requests": len(pair_rows),
        "retained_ordered_pairs": len(retained),
        "skipped_ordered_pairs": len(skipped),
        "fixed_k4_matrices_with_both_polarities": both,
        "fixed_k4_matrix_count": len(fixed_k4),
        "frozen_source_candidates_per_retained_pair": 5,
    }
    lines = [
        "# Top-5-mass branch-point candidate-polarity audit",
        "",
        "## Direct answers",
        "",
        f"1. Retained ordered source--target pairs: {len(retained)}/{len(pair_rows)}; skipped={len(skipped)}.",
        f"2. Every retained pair used the first control step where both positions were masked and source M5 was strictly greater than {metadata['strict_top5_mass_gate']}; M5=gate is excluded.",
        f"3. Each retained pair freezes exactly five source candidates and their probabilities at that same t*; treatment branches only force one of those tokens at the source.",
        f"4. Fixed-K=4 target matrices with both supportive and destructive cells: {both}/{len(fixed_k4)}.",
        "",
        "## Estimand and control integrity",
        "",
        "The control trajectory is read unchanged from the completed top-1 audit and retains its original `p1 >= 0.9` reveal rule plus argmax fallback. This runner does not resample, alter, or continue a treatment trajectory. A treatment is one independent full-vocabulary `use_cache=False` forward from the identical stored t* input, with only the source mask replaced. Thus this is the prior context-only candidate-polarity estimand with a stricter state-eligibility gate, not a post-intervention rollout-quality measurement.",
        "",
        "Each retained pair is recorded in `tables/branchpoint_pairs.csv`, including t*, M5, p1, and the five frozen source probabilities. Every matrix cell is recorded in `tables/candidate_polarity.csv`.",
        "",
        "## Runtime",
        "",
        f"- Exact branch forwards: {metadata['forward_accounting']['branch_forwards']}; model batch calls: {metadata['forward_accounting']['batch_calls']}",
        f"- Runtime seconds: {metadata['runtime_seconds']}; peak VRAM MiB: {metadata['peak_vram_mib']}",
    ]
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return direct


def write_figures(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - target setup installs matplotlib
        return []
    created: list[str] = []
    fixed = [row for row in rows if row.get("candidate_set_mode") == "fixed_k4"]
    if not fixed:
        return created
    representative = sorted(
        fixed,
        key=lambda row: (str(row["branchpoint_state_key"]), int(row["source_position"]), int(row["target_position"])),
    )[0]
    matrix_rows = [
        row for row in fixed
        if row["branchpoint_state_key"] == representative["branchpoint_state_key"]
        and row["source_position"] == representative["source_position"]
        and row["target_position"] == representative["target_position"]
    ]
    source_tokens = sorted({int(row["source_candidate_token_id"]) for row in matrix_rows})
    target_tokens = sorted({int(row["target_candidate_token_id"]) for row in matrix_rows})
    matrix = [
        [next(float(row["delta_logit_margin"]) for row in matrix_rows if int(row["source_candidate_token_id"]) == source and int(row["target_candidate_token_id"]) == target) for target in target_tokens]
        for source in source_tokens
    ]
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="coolwarm", aspect="auto")
    figure.colorbar(image, ax=axis, label="target logit-margin shift")
    axis.set_xticks(range(len(target_tokens)), [str(token) for token in target_tokens], rotation=45, ha="right")
    axis.set_yticks(range(len(source_tokens)), [str(token) for token in source_tokens])
    axis.set_xlabel("target candidate token")
    axis.set_ylabel("frozen source top-5 candidate token")
    axis.set_title(f"t*={representative['tstar_step']}, M5={float(representative['source_top5_mass']):.4f}")
    figure.tight_layout()
    path = output_root / "branchpoint_fixed_k4_polarity.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path.name)
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "top5_mass_branchpoint_audit.yaml")
    parser.add_argument("--probe-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Audit one representative unordered pair (two directions).")
    return parser.parse_args()


def make_output_root(base: Path, run_id: str | None) -> Path:
    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = base / identifier
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing branch-point audit: {root}")
    for name in ("raw", "tables", "figures"):
        (root / name).mkdir(parents=True, exist_ok=False) if name == "raw" else (root / name).mkdir(parents=True, exist_ok=True)
    return root


def main() -> None:
    args = parse_args()
    wall_started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    probe_root = args.probe_root.resolve()
    source_root = args.source_run.resolve()
    manifest_path = source_root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source run lacks run_manifest.json: {source_root}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") not in {"completed", "observational_completed_exact_audit_blocked_resource_estimate"}:
        raise RuntimeError(f"Source run is not a completed top-1 evidence bundle: {source_manifest.get('status')!r}")
    expected_fast = str(config["model"]["fast_dllm_commit"])
    actual_fast = source_manifest.get("fast_dllm_requested_commit")
    if actual_fast is not None and str(actual_fast) != expected_fast:
        raise RuntimeError(f"Source Fast-dLLM pin differs: source={actual_fast!r}, expected={expected_fast!r}")
    output_base = Path(args.output_root) if args.output_root else probe_root / str(config["storage"]["output_root"])
    output_root = make_output_root(output_base, args.run_id)
    shutil.copy2(args.config, output_root / "config.yaml")
    write_json(output_root / "source_manifest.json", source_manifest)
    trajectories = read_jsonl(source_root / "raw" / "trajectories.jsonl")
    selected, diagnostics = choose_states(trajectories, config, source_root)
    selected = [state for state in selected if state.cohort == "primary_policy"]
    unordered_count = 1 if args.smoke else int(config["sampling"]["representative_unordered_pair_count"])
    requests = pair_requests(selected, unordered_count)
    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for record in trajectories:
        if record.get("setting") == "primary" and float(record.get("threshold", -1.0)) == float(config["sampling"]["primary_threshold"]):
            grouped[trajectory_group_key(record)].append(record)

    pair_log: list[dict[str, Any]] = []
    retained: list[BranchPoint] = []
    skipped_reasons: Counter[str] = Counter()
    for request in requests:
        candidate_records = grouped.get(request_group_key(request), [])
        branchpoint, status = find_first_branchpoint(
            request,
            candidate_records,
            strict_top5_mass_gate=float(config["sampling"]["source_top5_mass_strictly_greater_than"]),
        )
        base_log = {
            "selection_state_key": request.selected_state.record["state_key"],
            "prompt_id": request.selected_state.record.get("prompt_id"),
            "selection_step": request.selected_state.record.get("step"),
            "source_position": request.source_position,
            "target_position": request.target_position,
            "status": status,
        }
        if branchpoint is None:
            skipped_reasons[status] += 1
            pair_log.append(base_log)
            continue
        control_error = validate_control_provenance(branchpoint.record, threshold=float(config["decoding"]["threshold"]))
        if control_error is not None:
            skipped_reasons[control_error] += 1
            pair_log.append({**base_log, "status": control_error})
            continue
        retained.append(branchpoint)
        pair_log.append({**base_log, "status": "retained", **_common(branchpoint)})
    if not retained:
        raise RuntimeError("No representative directed pair passed the strict t* top-5-mass eligibility gate.")

    from scripts.collect_states import mask_token_id
    from scripts.run_top1_dynamics_audit import load_model
    import torch

    model, tokenizer, _dtype = load_model(config, probe_root)
    mask_id = mask_token_id(model, tokenizer)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    accounting = ForwardAccounting(started=time.perf_counter())
    rows: list[dict[str, Any]] = []
    for index, branchpoint in enumerate(retained, 1):
        rows.extend(evaluate_branchpoint(model, branchpoint, mask_token_id=mask_id, config=config, accounting=accounting))
        print(json.dumps({"stage": "branchpoint_polarity", "pair": index, "total_pairs": len(retained), "state_key": branchpoint.record["state_key"], "forwards": accounting.branch_forwards}))
    torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - wall_started
    peak_vram_mib = float(torch.cuda.max_memory_allocated() / 2**20)
    summaries = polarity_summary(rows)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_smoke" if args.smoke else "completed",
        "source_run_root": str(source_root),
        "source_run_status": source_manifest.get("status"),
        "frozen_model": config["model"]["name"],
        "exact_cache_policy": "use_cache=False",
        "control_trajectory_policy": CONTROL_POLICY,
        "control_trajectory_reused_unchanged": True,
        "strict_top5_mass_gate": float(config["sampling"]["source_top5_mass_strictly_greater_than"]),
        "pair_selection": "same stable-hash primary-policy unordered selection as the prior candidate-polarity audit; evaluate both directions separately",
        "requested_unordered_pair_count": unordered_count,
        "requested_ordered_pair_count": len(requests),
        "retained_ordered_pair_count": len(retained),
        "skipped_pair_reasons": dict(sorted(skipped_reasons.items())),
        "selection_diagnostics": diagnostics,
        "forward_accounting": accounting.__dict__,
        "runtime_seconds": runtime_seconds,
        "peak_vram_mib": peak_vram_mib,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_jsonl(output_root / "raw" / "branchpoint_pair_selection.jsonl", pair_log)
    write_jsonl(output_root / "raw" / "candidate_polarity_interventions.jsonl", rows)
    write_csv(output_root / "tables" / "branchpoint_pairs.csv", pair_log)
    write_csv(output_root / "tables" / "candidate_polarity.csv", rows + summaries)
    write_csv(output_root / "tables" / "candidate_polarity_summary.csv", summaries)
    write_csv(output_root / "tables" / "exact_forward_counts.csv", [{
        "branch_forwards": accounting.branch_forwards,
        "model_batch_calls": accounting.batch_calls,
        "runtime_seconds": runtime_seconds,
        "peak_vram_mib": peak_vram_mib,
    }])
    direct = write_report(output_root, pair_rows=pair_log, summary_rows=summaries, metadata=metadata)
    metadata["direct_answers"] = direct
    metadata["figure_paths"] = write_figures(output_root / "figures", rows)
    write_json(output_root / "run_metadata.json", metadata)
    write_json(output_root / "summary.json", {"status": metadata["status"], "metadata": metadata, "direct_answers": direct})
    print(json.dumps({"status": metadata["status"], "output_root": str(output_root), "forwards": accounting.branch_forwards, "runtime_seconds": runtime_seconds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
