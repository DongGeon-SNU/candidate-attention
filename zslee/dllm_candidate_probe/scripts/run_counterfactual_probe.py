#!/usr/bin/env python3
"""Exact frozen-LLaDA counterfactual probe.

Fast-dLLM-style confidence-threshold decoding supplies diagnostic states. Every
base/singleton/pair/set/leave-one-out measurement below then starts from an
independent complete input sequence and calls the backbone with ``use_cache``
disabled.  No auxiliary module is trained or introduced.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_states import DecodingState, collect_states, mask_token_id
from src.branching import Candidate, exact_forward, make_branch, make_leave_one_out_branches
from src.metrics import (
    directed_lift,
    pair_residuals,
    pair_stability,
    probabilities,
    set_stability,
    symmetric_compatibility,
    token_probability,
)
from src.set_search import beam_sets


def candidate_key(a: Candidate, b: Candidate) -> tuple[Candidate, Candidate]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def token_repr(tokenizer: Any, token_id: int) -> str:
    try:
        return repr(tokenizer.decode([token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False))
    except TypeError:
        return repr(tokenizer.decode([token_id]))


def candidate_record(tokenizer: Any, candidate: Candidate) -> dict[str, Any]:
    return {"position": candidate.position, "token_id": candidate.token_id, "token": token_repr(tokenizer, candidate.token_id)}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_model(config: dict[str, Any], probe_root: Path) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; frozen LLaDA smoke cannot fall back to CPU or quantization.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(int(config["decoding"]["seed"]))
    random.seed(int(config["decoding"]["seed"]))
    dtype_setting = config["model"]["dtype"]
    if dtype_setting == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif dtype_setting == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_setting == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype setting: {dtype_setting}")
    model_name = config["model"]["name"]
    cache_dir = probe_root / "cache" / "huggingface"
    fast_llada_root = probe_root / "vendor" / "Fast-dLLM" / "v1" / "llada"
    if not (fast_llada_root / "model" / "modeling_llada.py").exists():
        raise RuntimeError("Pinned Fast-dLLM v1 LLaDA source is missing; run scripts/setup.sh first.")
    sys.path.insert(0, str(fast_llada_root))
    from model.modeling_llada import LLaDAModelLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=config["model"]["hf_revision"], trust_remote_code=True, cache_dir=cache_dir
    )
    model = LLaDAModelLM.from_pretrained(
        model_name,
        revision=config["model"]["hf_revision"],
        trust_remote_code=True,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    ).to("cuda").eval()
    return model, tokenizer, dtype


def candidates_for_state(state: DecodingState, config: dict[str, Any]) -> list[Candidate]:
    decoding = config["decoding"]
    eligible: list[tuple[float, int, dict[str, Any]]] = []
    for position, summary in state.position_summaries.items():
        if summary["top1_confidence"] < decoding["threshold"] and summary["top5_cumulative_probability"] >= decoding["tau_mass"]:
            eligible.append((summary["top1_confidence"], position, summary))
    eligible.sort(key=lambda item: (item[0], item[1]))
    candidates: list[Candidate] = []
    for _, position, summary in eligible[: decoding["max_candidate_positions"]]:
        for token_id in summary["top5_token_ids"][: decoding["candidates_per_position"]]:
            candidates.append(Candidate(position, int(token_id)))
    return candidates


def exact_state_probe(
    model: Any,
    tokenizer: Any,
    state: DecodingState,
    config: dict[str, Any],
    forward: Callable[[Any], Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure base, singleton, pair, set, and leave-one-out branches exactly."""

    import torch

    mask_id = mask_token_id(model, tokenizer)
    candidates = candidates_for_state(state, config)
    candidate_positions = {candidate.position for candidate in candidates}
    base_record = state.public_record()
    base_record["candidate_count"] = len(candidates)
    base_record["candidate_positions"] = sorted(candidate_positions)
    if len(candidate_positions) < 3:
        base_record["probe_status"] = "insufficient_distinct_candidate_positions"
        return base_record, [], []

    base_input_ids = state.input_ids
    attention_mask = torch.ones_like(base_input_ids)
    position_ids = torch.arange(base_input_ids.shape[1], device=base_input_ids.device).unsqueeze(0)
    base_probabilities = probabilities(forward(base_input_ids))

    singleton_probabilities: dict[Candidate, Any] = {}
    for candidate in candidates:
        branch = make_branch(base_input_ids, [candidate], mask_token_id=mask_id)
        singleton_probabilities[candidate] = probabilities(forward(branch))

    pair_rows: list[dict[str, Any]] = []
    pair_lookup: dict[tuple[Candidate, Candidate], dict[str, Any]] = {}
    epsilon = float(config["probe"]["epsilon"])
    for a, b in combinations(candidates, 2):
        if a.position == b.position:
            continue
        branch = make_branch(base_input_ids, [a, b], mask_token_id=mask_id)
        pair_probabilities = probabilities(forward(branch))
        a_to_b = directed_lift(
            token_probability(base_probabilities, b.position, b.token_id),
            token_probability(singleton_probabilities[a], b.position, b.token_id),
            epsilon,
        )
        b_to_a = directed_lift(
            token_probability(base_probabilities, a.position, a.token_id),
            token_probability(singleton_probabilities[b], a.position, a.token_id),
            epsilon,
        )
        compatibility = symmetric_compatibility(a_to_b, b_to_a)
        remaining = [position for position in state.mask_positions if position not in {a.position, b.position}]
        residual = pair_residuals(
            base_probabilities, singleton_probabilities[a], singleton_probabilities[b], pair_probabilities, remaining
        )
        q2 = pair_stability(
            token_probability(singleton_probabilities[b], a.position, a.token_id),
            token_probability(singleton_probabilities[a], b.position, b.token_id),
        )
        row = {
            "prompt_index": state.prompt_index,
            "step": state.step,
            "a": candidate_record(tokenizer, a),
            "b": candidate_record(tokenizer, b),
            "position_pair": [a.position, b.position],
            "mean_base_confidence": (
                state.position_summaries[a.position]["top1_confidence"]
                + state.position_summaries[b.position]["top1_confidence"]
            )
            / 2,
            "a_to_b_lift": float(a_to_b.item()),
            "b_to_a_lift": float(b_to_a.item()),
            "compatibility": float(compatibility.item()),
            "pair_stability_q2": q2,
            "residual_mean_tv": residual["mean_tv"],
            "residual_max_tv": residual["max_tv"],
            "remaining_position_tv": residual["per_position_tv"],
        }
        pair_rows.append(row)
        pair_lookup[candidate_key(a, b)] = row
        del pair_probabilities

    def pair_score(a: Candidate, b: Candidate) -> float:
        return float(pair_lookup[candidate_key(a, b)]["compatibility"])

    set_rows: list[dict[str, Any]] = []
    for size in config["probe"]["set_sizes"]:
        for scored in beam_sets(candidates, int(size), pair_score=pair_score, beam_width=int(config["probe"]["set_beam_width"]))[
            : int(config["probe"]["max_sets_per_state"])
        ]:
            # The full-set forward is deliberate even though Q(S) uses LOO branches.
            set_branch = make_branch(base_input_ids, scored.candidates, mask_token_id=mask_id)
            full_set_probabilities = probabilities(forward(set_branch))
            leave_one_out = make_leave_one_out_branches(base_input_ids, scored.candidates, mask_token_id=mask_id)
            loo_values = []
            for omitted, branch in leave_one_out.items():
                loo_probabilities = probabilities(forward(branch))
                loo_values.append(token_probability(loo_probabilities, omitted.position, omitted.token_id))
                del loo_probabilities
            q_set = set_stability(loo_values)
            contained_pairs = [pair_lookup[candidate_key(a, b)] for a, b in combinations(scored.candidates, 2)]
            base = {
                "prompt_index": state.prompt_index,
                "step": state.step,
                "size": int(size),
                "set_score": scored.score,
                "candidates": [candidate_record(tokenizer, item) for item in scored.candidates],
                "pair_compatibilities": [row["compatibility"] for row in contained_pairs],
                "pair_stabilities_q2": [row["pair_stability_q2"] for row in contained_pairs],
                "set_stability_q": q_set,
            }
            for threshold in sorted(set(config["probe"]["pair_stability_thresholds"]).intersection(config["probe"]["set_stability_thresholds"])):
                base[f"pair_safe_set_unsafe_at_{threshold}"] = bool(
                    min(base["pair_stabilities_q2"]) >= threshold and q_set < threshold
                )
            set_rows.append(base)
            del full_set_probabilities

    base_record["probe_status"] = "exact_branches_completed"
    base_record["branch_policy"] = {
        "base": "full exact forward, use_cache=False",
        "singleton": len(singleton_probabilities),
        "pairs": len(pair_rows),
        "sets": len(set_rows),
        "leave_one_out": "one exact no-cache forward per candidate in each selected set",
        "position_ids": "identical base-derived tensor passed to every branch",
    }
    return base_record, pair_rows, set_rows


def sign_flip_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, tuple[int, int]], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(row["prompt_index"], row["step"], tuple(sorted(row["position_pair"])))].append(row)
    rows = []
    for (prompt_index, step, positions), members in sorted(grouped.items()):
        values = [member["compatibility"] for member in members]
        rows.append(
            {
                "prompt_index": prompt_index,
                "step": step,
                "position_pair": list(positions),
                "candidate_combination_count": len(values),
                "compatibility_mean": sum(values) / len(values),
                "compatibility_std": (sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values)) ** 0.5,
                "compatibility_min": min(values),
                "compatibility_max": max(values),
                "mean_base_confidence": sum(member["mean_base_confidence"] for member in members) / len(members),
                "sign_flip": min(values) < 0 < max(values),
            }
        )
    return rows


def sign_flip_rate_by_confidence(sign_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.001)]
    rows = []
    for lower, upper in bins:
        members = [row for row in sign_rows if lower <= row["mean_base_confidence"] < upper]
        rows.append(
            {
                "base_confidence_bin": f"[{lower:.2f}, {upper:.2f})",
                "position_pair_count": len(members),
                "sign_flip_rate": sum(row["sign_flip"] for row in members) / len(members) if members else 0.0,
            }
        )
    return rows


def sign_flip_rate_by_phase(sign_rows: list[dict[str, Any]], configured_steps: int) -> list[dict[str, Any]]:
    phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sign_rows:
        fraction = row["step"] / max(configured_steps - 1, 1)
        phase = "early" if fraction < 1 / 3 else "middle" if fraction < 2 / 3 else "late"
        phases[phase].append(row)
    return [
        {
            "phase": phase,
            "position_pair_count": len(phases[phase]),
            "sign_flip_rate": sum(row["sign_flip"] for row in phases[phase]) / len(phases[phase]) if phases[phase] else 0.0,
        }
        for phase in ("early", "middle", "late")
    ]


def prompt_bootstrap_ci(pair_rows: list[dict[str, Any]], seed: int) -> dict[str, float] | None:
    """Prompt is the resampling unit, never individual candidate pairs."""

    prompt_means: dict[int, list[float]] = defaultdict(list)
    for row in pair_rows:
        prompt_means[row["prompt_index"]].append(row["compatibility"])
    values = [sum(group) / len(group) for group in prompt_means.values() if group]
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(1000))
    return {"mean": sum(values) / len(values), "ci95_low": samples[24], "ci95_high": samples[974], "prompt_count": len(values)}


def resource_estimate(forward_count: int, branch_seconds: float, peak_mib: float) -> dict[str, Any]:
    seconds_per_forward = branch_seconds / max(forward_count, 1)
    # Upper-bound configured branch count: base + 20 singleton + 150 cross-position
    # pairs + two 8-wide set beams with a full-set and up to four LOO branches.
    conservative_forwards_per_state = 1 + 20 + 150 + 8 * (1 + 3) + 8 * (1 + 4)
    pilot_seconds = seconds_per_forward * conservative_forwards_per_state * 20 * 1.25
    return {
        "measured_exact_forward_count": forward_count,
        "measured_exact_branch_seconds": branch_seconds,
        "estimated_seconds_per_exact_forward": seconds_per_forward,
        "conservative_forwards_per_state": conservative_forwards_per_state,
        "pilot_states": 20,
        "estimated_pilot_seconds": pilot_seconds,
        "estimated_pilot_minutes": pilot_seconds / 60,
        "peak_vram_mib": peak_mib,
        "pilot_eligible": pilot_seconds <= 30 * 60,
        "policy": "Pilot is blocked automatically when the estimate exceeds 30 minutes.",
    }


def write_resource_report(path: Path, data: dict[str, Any]) -> None:
    save_json(path.with_suffix(".json"), data)
    path.write_text(
        "# Resource estimate\n\n"
        f"- Peak VRAM allocated by this process: {data['peak_vram_mib']:.1f} MiB\n"
        f"- Exact forwards measured: {data['measured_exact_forward_count']}\n"
        f"- Mean exact-forward time: {data['estimated_seconds_per_exact_forward']:.3f} s\n"
        f"- Conservative 10-prompt/20-state pilot estimate: {data['estimated_pilot_minutes']:.1f} min\n"
        f"- Pilot eligible under 30-min policy: **{data['pilot_eligible']}**\n",
        encoding="utf-8",
    )


def gpu_memory_snapshot() -> list[str]:
    command = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=20)
    except FileNotFoundError:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()] if completed.returncode == 0 else []


def git_commit(probe_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(probe_root), "rev-parse", "HEAD"], text=True, capture_output=True, check=False, timeout=10
        )
    except FileNotFoundError:
        return "unavailable (git not installed in target)"
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable (source tree is not a git checkout)"


def make_figures(probe_root: Path, pair_rows: list[dict[str, Any]], set_rows: list[dict[str, Any]]) -> list[str]:
    """Create compact diagnostic plots from scalar summaries only."""

    if not pair_rows:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []
    figure_dir = probe_root / "outputs/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    compatibility = [row["compatibility"] for row in pair_rows]
    residuals = [row["residual_mean_tv"] for row in pair_rows]
    for values, filename, xlabel in (
        (compatibility, "compatibility_distribution.png", "symmetric compatibility C(a,b)"),
        (residuals, "joint_residual_distribution.png", "mean joint residual TV"),
    ):
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(values, bins=min(30, max(5, len(values))), color="#356aa0")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("pair count")
        figure.tight_layout()
        figure.savefig(figure_dir / filename, dpi=160)
        plt.close(figure)
        created.append(filename)
    first_positions = tuple(sorted(pair_rows[0]["position_pair"]))
    heat_rows = [row for row in pair_rows if tuple(sorted(row["position_pair"])) == first_positions]
    left = sorted({row["a"]["token"] for row in heat_rows})
    right = sorted({row["b"]["token"] for row in heat_rows})
    if left and right:
        matrix = np.full((len(left), len(right)), np.nan)
        for row in heat_rows:
            a_token, b_token = row["a"]["token"], row["b"]["token"]
            matrix[left.index(a_token), right.index(b_token)] = row["compatibility"]
        figure, axis = plt.subplots(figsize=(max(5, len(right)), max(4, len(left))))
        image = axis.imshow(matrix, cmap="coolwarm", aspect="auto")
        axis.set_xticks(range(len(right)), right, rotation=45, ha="right")
        axis.set_yticks(range(len(left)), left)
        axis.set_title(f"Candidate compatibility at positions {first_positions}")
        figure.colorbar(image, ax=axis, label="C(a,b)")
        figure.tight_layout()
        filename = "candidate_compatibility_heatmap.png"
        figure.savefig(figure_dir / filename, dpi=160)
        plt.close(figure)
        created.append(filename)
    threshold_keys = sorted({key for row in set_rows for key in row if key.startswith("pair_safe_set_unsafe")})
    if threshold_keys:
        rates = [sum(bool(row.get(key)) for row in set_rows) / len(set_rows) for key in threshold_keys]
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.bar(range(len(rates)), rates, color="#a05a2c")
        axis.set_xticks(range(len(rates)), [key.rsplit("_", 1)[-1] for key in threshold_keys])
        axis.set_xlabel("shared stability threshold")
        axis.set_ylabel("pair-safe / set-unsafe rate")
        axis.set_ylim(0, 1)
        figure.tight_layout()
        filename = "pair_safe_set_unsafe_rates.png"
        figure.savefig(figure_dir / filename, dpi=160)
        plt.close(figure)
        created.append(filename)
    return created


def write_summary(probe_root: Path, config: dict[str, Any], pair_rows: list[dict[str, Any]], set_rows: list[dict[str, Any]], mode: str) -> None:
    sign_rows = sign_flip_rows(pair_rows)
    confidence_rows = sign_flip_rate_by_confidence(sign_rows)
    phase_rows = sign_flip_rate_by_phase(sign_rows, int(config["decoding"]["steps"]))
    write_csv(probe_root / "outputs/tables/position_pair_compatibility.csv", sign_rows)
    write_csv(probe_root / "outputs/tables/sign_flip_by_confidence.csv", confidence_rows)
    write_csv(probe_root / "outputs/tables/sign_flip_by_phase.csv", phase_rows)
    write_csv(probe_root / "outputs/tables/pair_metrics.csv", [
        {key: value for key, value in row.items() if key not in {"a", "b", "remaining_position_tv"}} for row in pair_rows
    ])
    write_csv(probe_root / "outputs/tables/set_metrics.csv", set_rows)
    compatibility = [row["compatibility"] for row in pair_rows]
    residuals = [row["residual_mean_tv"] for row in pair_rows]
    sign_flip_rate = sum(row["sign_flip"] for row in sign_rows) / len(sign_rows) if sign_rows else 0.0
    threshold_rates = {
        key: sum(bool(row.get(key)) for row in set_rows) / len(set_rows) if set_rows else 0.0
        for key in sorted({key for row in set_rows for key in row if key.startswith("pair_safe_set_unsafe")})
    }
    set_rate_rows = []
    for size in sorted({row["size"] for row in set_rows}):
        members = [row for row in set_rows if row["size"] == size]
        for key in threshold_rates:
            set_rate_rows.append(
                {
                    "set_size": size,
                    "threshold": key.rsplit("_", 1)[-1],
                    "set_count": len(members),
                    "pair_safe_set_unsafe_rate": sum(bool(row.get(key)) for row in members) / len(members) if members else 0.0,
                }
            )
    write_csv(probe_root / "outputs/tables/pair_safe_set_unsafe_rates.csv", set_rate_rows)
    unsafe_cases = [
        row for row in set_rows if any(bool(value) for key, value in row.items() if key.startswith("pair_safe_set_unsafe"))
    ]
    safe_cases = [row for row in set_rows if row not in unsafe_cases]
    representative_cases = sorted(unsafe_cases, key=lambda row: row["set_stability_q"])[:3] + sorted(
        safe_cases, key=lambda row: -row["set_stability_q"]
    )[:2]
    save_json(probe_root / "outputs/raw/representative_cases.json", representative_cases)
    bootstrap = prompt_bootstrap_ci(pair_rows, int(config["decoding"]["seed"]))
    figures = make_figures(probe_root, pair_rows, set_rows)
    lines = [
        "# Frozen LLaDA counterfactual probe summary",
        "",
        f"- Mode: `{mode}`",
        f"- Backbone: `{config['model']['name']}`; frozen/eval; quantization disabled.",
        f"- Fast-dLLM source pin: `{config['model']['fast_dllm_commit']}`",
        f"- Probe source Git commit: `{git_commit(probe_root)}`",
        f"- Seed: `{config['decoding']['seed']}`; threshold: `{config['decoding']['threshold']}`; tau_mass: `{config['decoding']['tau_mass']}`.",
        f"- Pair branches: {len(pair_rows)}; selected sets: {len(set_rows)}.",
        f"- Compatibility: positive={sum(value > 0 for value in compatibility)}, negative={sum(value < 0 for value in compatibility)}.",
        f"- Same-position-pair candidate sign-flip rate: {sign_flip_rate:.3f}.",
        f"- Sign-flip by base-confidence and decoding phase: `outputs/tables/sign_flip_by_confidence.csv`, `outputs/tables/sign_flip_by_phase.csv`.",
        f"- Mean joint residual TV: {(sum(residuals) / len(residuals)) if residuals else 0.0:.6f}.",
        f"- Pair-safe / set-unsafe rates: `{json.dumps(threshold_rates)}`.",
        f"- Pair-safe / set-unsafe by size/threshold: `outputs/tables/pair_safe_set_unsafe_rates.csv`.",
        f"- Representative cases (up to five): `outputs/raw/representative_cases.json`.",
        f"- Prompt-level bootstrap compatibility CI: `{json.dumps(bootstrap) if bootstrap else 'not available (fewer than two prompts)'}`.",
        f"- Figures: {', '.join(f'`outputs/figures/{name}`' for name in figures) or 'not generated (no eligible pairs)' }.",
        f"- Re-run command: `bash scripts/run_{mode}.sh`.",
        f"- Result paths: `outputs/raw/{mode}_states.jsonl`, `outputs/raw/{mode}_pairs.jsonl`, `outputs/raw/{mode}_sets.jsonl`, and `outputs/tables/`.",
        "",
        "Raw records contain only token IDs, repr-form token strings, and scalar/top-k summaries; no attention maps, hidden states, KV caches, or vocabulary tensors are stored.",
    ]
    if mode == "smoke":
        lines.append("- Smoke resource decision: `outputs/resource_estimate.md` (the pilot command reads its JSON companion).")
    (probe_root / "outputs/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_eligible_pilot(probe_root: Path) -> None:
    estimate_path = probe_root / "outputs/resource_estimate.json"
    if not estimate_path.exists():
        raise RuntimeError("Pilot requires a successful smoke test and outputs/resource_estimate.json.")
    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    if not estimate.get("pilot_eligible", False):
        raise RuntimeError("Pilot is blocked: smoke-based conservative estimate exceeds the 30-minute policy.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.probe_root.joinpath("outputs/raw").mkdir(parents=True, exist_ok=True)
    args.probe_root.joinpath("outputs/tables").mkdir(parents=True, exist_ok=True)
    if args.mode == "pilot":
        require_eligible_pilot(args.probe_root)
    environment_report = args.probe_root / "outputs/environment_report.json"
    try:
        gpu_memory_before = json.loads(environment_report.read_text(encoding="utf-8"))["gpu"]["gpus"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        gpu_memory_before = gpu_memory_snapshot()

    import torch

    model, tokenizer, dtype = load_model(config, args.probe_root)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    forward_count = 0
    branch_seconds = 0.0

    def forward(input_ids: Any) -> Any:
        nonlocal forward_count, branch_seconds
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        began = time.perf_counter()
        logits = exact_forward(model, input_ids, attention_mask=attention_mask, position_ids=position_ids)
        torch.cuda.synchronize()
        branch_seconds += time.perf_counter() - began
        forward_count += 1
        return logits

    prompts = config["smoke_prompts"][:1] if args.mode == "smoke" else config["pilot_prompts"][:10]
    state_records: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    all_set_rows: list[dict[str, Any]] = []
    max_states = 1 if args.mode == "smoke" else 2
    for prompt_index, prompt in enumerate(prompts):
        states = collect_states(
            model, tokenizer, prompt, prompt_index=prompt_index,
            generation_length=int(config["decoding"]["generation_length"]),
            steps=int(config["decoding"]["steps"]),
            threshold=float(config["decoding"]["threshold"]),
            use_cache=bool(config["decoding"]["use_cache_for_state_collection"]),
            low_parallel_max_transferred=int(config["decoding"]["low_parallel_max_transferred"]),
            low_parallel_remaining_fraction=float(config["decoding"]["low_parallel_remaining_fraction"]),
        )[:max_states]
        for state in states:
            state_record, pair_rows, set_rows = exact_state_probe(model, tokenizer, state, config, forward)
            state_records.append(state_record)
            all_pair_rows.extend(pair_rows)
            all_set_rows.extend(set_rows)
            if args.mode == "smoke":
                break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    append_jsonl(args.probe_root / f"outputs/raw/{args.mode}_states.jsonl", state_records)
    append_jsonl(args.probe_root / f"outputs/raw/{args.mode}_pairs.jsonl", all_pair_rows)
    append_jsonl(args.probe_root / f"outputs/raw/{args.mode}_sets.jsonl", all_set_rows)
    write_summary(args.probe_root, config, all_pair_rows, all_set_rows, args.mode)
    run_result = {
        "mode": args.mode,
        "status": "success" if all_pair_rows and all_set_rows else "blocked_insufficient_eligible_candidates",
        "dtype": str(dtype),
        "elapsed_seconds": elapsed,
        "peak_vram_mib": peak_mib,
        "gpu_memory_before": gpu_memory_before,
        "gpu_memory_after": gpu_memory_snapshot(),
        "exact_forward_count": forward_count,
        "cache_policy": "State collection may use cache; every counterfactual forward used use_cache=False.",
        "states_examined": len(state_records),
        "pair_branches": len(all_pair_rows),
        "set_branches": len(all_set_rows),
    }
    save_json(args.probe_root / f"outputs/{args.mode}_result.json", run_result)
    if args.mode == "smoke":
        estimate = resource_estimate(forward_count, branch_seconds, peak_mib)
        write_resource_report(args.probe_root / "outputs/resource_estimate.md", estimate)
    print(json.dumps(run_result, ensure_ascii=False))


if __name__ == "__main__":
    main()
