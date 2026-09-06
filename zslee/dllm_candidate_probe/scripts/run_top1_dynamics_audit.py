#!/usr/bin/env python3
"""Run the frozen LLaDA top-1 dynamics audit on the target GPU server.

This command deliberately keeps natural decoding and measurement separate:

* natural trajectories use the existing threshold-plus-fallback policy;
* every reported distribution, rank, transition, and causal branch is replayed
  from a complete input with ``use_cache=False``;
* no learned predictor, head, or policy intervention is introduced.

It is safe to invoke only from a fresh per-run output directory.  The script
never deletes or overwrites an existing experiment run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_environment import markdown as environment_markdown
from scripts.check_environment import report as environment_report
from scripts.collect_states import collect_states
from src.top1_datasets import BenchmarkLoadResult, PromptExample, load_benchmark
from src.top1_observation import (
    ExactObservationResult,
    observe_trajectory_exactly,
    position_trajectory_rows,
    prediction_label_rows,
    repeated_exact_forward_check,
)
from src.top1_trajectory import collect_natural_trajectory, verify_trajectory_alignment


SCHEMA_VERSION = 1

# Kept in one place because the reporting stage and its smoke-calibrated time
# estimate must cover the exact same pre-specified score/outcome grid.
PREDICTIVENESS_SCORE_FIELDS: tuple[str, ...] = (
    "entropy",
    "normalized_entropy",
    "one_minus_p1",
    "p1_minus_p2",
    "logit_margin",
    "top5_mass",
    "tv_from_previous",
    "js_from_previous",
    "top1_probability_slope",
    "entropy_slope",
    "margin_slope",
    "entropy_two_state_slope",
    "margin_two_state_slope",
)
PREDICTIVENESS_OUTCOMES: tuple[str, ...] = (
    "next_step_top1_flip",
    "future_2_step_top1_flip",
    "future_3_step_top1_flip",
    "future_5_step_top1_flip",
    "any_flip_before_commit",
    "eventual_token_mismatch",
)
PREDICTIVENESS_SCORE_DIRECTIONS = {
    "p1_minus_p2": "lower",
    "logit_margin": "lower",
    "top5_mass": "lower",
    "top1_probability_slope": "lower",
    "margin_slope": "lower",
    "margin_two_state_slope": "lower",
}


@dataclass(frozen=True)
class RunPaths:
    root: Path
    raw: Path
    tables: Path
    figures: Path
    logs: Path


@dataclass
class ObservationBundle:
    trajectories: list[dict[str, Any]]
    state_index: list[dict[str, Any]]
    state_position_rows: list[dict[str, Any]]
    transitions: list[dict[str, Any]]
    eventual_rows: list[dict[str, Any]]
    prompt_rows: list[dict[str, Any]]
    sanity_rows: list[dict[str, Any]]
    natural_seconds: float
    exact_seconds: float
    natural_forward_count: int
    exact_forward_count: int
    sanity_forward_count: int
    peak_vram_mib: float | None


class BaselineCollectorDivergenceError(RuntimeError):
    """Stop the audit while retaining the exact failed compatibility record."""

    def __init__(self, sanity_record: Mapping[str, Any]) -> None:
        self.sanity_record = dict(sanity_record)
        reasons = self.sanity_record.get("mismatch_reasons", [])
        super().__init__(
            "Natural trajectory diverged from the historical collector; "
            "refusing to publish an audit with changed decoding actions: "
            + "; ".join(str(reason) for reason in reasons)
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def git_value(probe_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(probe_root), *args], text=True, capture_output=True, check=False, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def make_run_paths(base_output: Path, run_id: str | None) -> RunPaths:
    stem = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = base_output / stem
    if root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing top-1 audit run {root}. Choose --run-id or a new output root."
        )
    raw, tables, figures, logs = (root / name for name in ("raw", "tables", "figures", "logs"))
    for directory in (raw, tables, figures, logs):
        directory.mkdir(parents=True, exist_ok=False)
    return RunPaths(root=root, raw=raw, tables=tables, figures=figures, logs=logs)


def _safe_token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([int(token_id)])


def _decode_generation(tokenizer: Any, trajectory: Any) -> str:
    ids = trajectory.final_input_ids[0, trajectory.prompt_token_count :].detach().cpu().tolist()
    try:
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(ids)


def _normalise_answer(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # GSM8K answers conventionally end after ####.  A generated answer often
    # omits it, in which case use the final visible number rather than claiming
    # a bespoke semantic evaluator.
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return re.sub(r"\s+", " ", text).strip().lower()
    return matches[-1].replace(",", "").lstrip("+")


def gsm8k_exact_match(generated_text: str, reference: Mapping[str, Any] | None) -> bool | None:
    if not reference or "final_answer" not in reference:
        return None
    prediction, expected = _normalise_answer(generated_text), _normalise_answer(str(reference["final_answer"]))
    return prediction is not None and expected is not None and prediction == expected


def phase_label(step: int, state_count: int, boundaries: Sequence[float]) -> str:
    fraction = step / max(state_count - 1, 1)
    first, second = (float(boundaries[0]), float(boundaries[1]))
    return "early" if fraction < first else "middle" if fraction < second else "late"


def bucket_label(value: float | None, edges: Sequence[float]) -> str | None:
    if value is None or not math.isfinite(float(value)):
        return None
    numeric = float(value)
    for low, high in zip(edges, edges[1:]):
        if low <= numeric < high:
            return f"[{low:g}, {high:g})"
    if numeric == float(edges[-1]):
        return f"[{float(edges[-2]):g}, {float(edges[-1]):g}]"
    return None


def anchor_distance_bucket(value: int | None) -> str:
    """Compact, explicit strata for distance to the nearest committed anchor."""

    if value is None:
        return "unavailable_no_anchor"
    distance = int(value)
    if distance == 0:
        return "0"
    if distance == 1:
        return "1"
    if distance <= 3:
        return "2-3"
    if distance <= 7:
        return "4-7"
    if distance <= 15:
        return "8-15"
    return "16+"


def prompt_identifier(example: PromptExample) -> str:
    return f"{example.dataset}:{example.example_id}"


def _trajectory_common(
    example: PromptExample,
    trajectory: Any,
    generated_text: str,
    correctness: bool | None,
    *,
    setting: str,
) -> dict[str, Any]:
    return {
        "dataset": example.dataset,
        "example_id": example.example_id,
        "prompt_id": prompt_identifier(example),
        "prompt_index": trajectory.prompt_index,
        "prompt": example.prompt,
        "reference": dict(example.reference) if example.reference is not None else None,
        "task_metadata": dict(example.metadata),
        "setting": setting,
        "threshold": trajectory.threshold,
        "generated_text": generated_text,
        "gsm8k_exact_match": correctness,
        "trajectory_completed": trajectory.completed,
        "generation_length": trajectory.generation_length,
        "state_count": len(trajectory.states),
    }


def _attach_state_context(
    rows: Iterable[Mapping[str, Any]],
    common: Mapping[str, Any],
    exact_states: Sequence[Any],
    boundaries: Sequence[float],
    confidence_edges: Sequence[float],
) -> list[dict[str, Any]]:
    total_states = len(exact_states)
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = {**common, **dict(raw)}
        row["state_key"] = state_identifier_from_common(common, int(row["step"]))
        row["phase"] = phase_label(int(row["step"]), total_states, boundaries)
        row["confidence_bucket"] = bucket_label(row.get("top1_probability"), confidence_edges)
        result.append(row)
    return result


def state_identifier_from_common(common: Mapping[str, Any], step: int) -> str:
    # A prompt can appear in smoke, primary, and threshold-sensitivity runs.
    # Include the named setting so raw artifacts remain a true key-value log
    # rather than silently colliding at a shared 0.9 threshold.
    material = f"{common['prompt_id']}|{common['setting']}|{float(common['threshold']):.8g}|{int(step)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _decorate_transitions(
    transitions: Iterable[Any],
    common: Mapping[str, Any],
    exact_states: Sequence[Any],
    boundaries: Sequence[float],
    confidence_edges: Sequence[float],
) -> list[dict[str, Any]]:
    state_by_step = {state.step: state for state in exact_states}
    result: list[dict[str, Any]] = []
    for transition in transitions:
        source = state_by_step[transition.source_step]
        target = state_by_step[transition.target_step]
        summary = source.position_summaries[transition.position]
        next_summary = target.position_summaries[transition.position]
        row = {**common, **transition.public_record()}
        row.update(
            {
                "state_key": state_identifier_from_common(common, transition.source_step),
                "source_token_sequence": source.input_ids[0].detach().cpu().tolist(),
                "next_token_sequence": state_by_step[transition.target_step].input_ids[0].detach().cpu().tolist(),
                "source_mask_positions": list(source.mask_positions),
                "source_anchors": [anchor.public_record() for anchor in source.actual_committed_anchors],
                "source_top2_token_id": summary.top2_token_id,
                "source_entropy": summary.entropy,
                "source_normalized_entropy": summary.normalized_entropy,
                "source_p1_minus_p2": summary.p1_minus_p2,
                "source_logit_margin": summary.logit_margin,
                "source_top5_mass": summary.top5_mass,
                "next_entropy": next_summary.entropy,
                "entropy_change": next_summary.entropy - summary.entropy,
                "next_logit_margin": next_summary.logit_margin,
                "logit_margin_change": next_summary.logit_margin - summary.logit_margin,
                "source_phase": phase_label(transition.source_step, len(exact_states), boundaries),
                "source_confidence_bucket": bucket_label(summary.top1_probability, confidence_edges),
                # Unprefixed aliases deliberately feed generic reporting
                # helpers; the source_* names retain causal-audit clarity.
                "phase": phase_label(transition.source_step, len(exact_states), boundaries),
                "confidence_bucket": bucket_label(summary.top1_probability, confidence_edges),
                "anchor_set_size": transition.source_anchor_count,
                "low_parallel": transition.source_low_parallel,
                "same_block_as_anchor": transition.same_analysis_block_as_anchor,
                "mask_ratio_bucket": (
                    "[0,.25)" if transition.source_mask_ratio < .25 else "[.25,.5)"
                    if transition.source_mask_ratio < .5 else "[.5,.75)"
                    if transition.source_mask_ratio < .75 else "[.75,1]"
                ),
                "entropy": summary.entropy,
                "normalized_entropy": summary.normalized_entropy,
                "logit_margin": summary.logit_margin,
                "p1_minus_p2": summary.p1_minus_p2,
                "one_minus_p1": 1.0 - summary.top1_probability,
                "top5_mass": summary.top5_mass,
                "top5_mass_ge_0_8": summary.top5_mass >= 0.8,
                "nearest_anchor_distance_bucket": anchor_distance_bucket(transition.nearest_anchor_distance),
            }
        )
        result.append(row)
    return result


def _decorate_eventual_rows(
    rows: Iterable[Mapping[str, Any]],
    common: Mapping[str, Any],
    exact_states: Sequence[Any],
    boundaries: Sequence[float],
    confidence_edges: Sequence[float],
) -> list[dict[str, Any]]:
    state_by_step = {state.step: state for state in exact_states}
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = {**common, **dict(raw)}
        state = state_by_step[int(row["step"])]
        summary = state.position_summaries[int(row["position"])]
        row.update(
            {
                "state_key": state_identifier_from_common(common, int(row["step"])),
                "phase": phase_label(int(row["step"]), len(exact_states), boundaries),
                "confidence_bucket": bucket_label(summary.top1_probability, confidence_edges),
                "entropy": summary.entropy,
                "logit_margin": summary.logit_margin,
                "top1_probability": summary.top1_probability,
            }
        )
        result.append(row)
    return result


def _state_records(
    common: Mapping[str, Any], exact_states: Sequence[Any], *, decoder_metadata: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    for state in exact_states:
        state_key = state_identifier_from_common(common, state.step)
        record = {**common, **state.public_record()}
        record.update(
            {
                "state_key": state_key,
                "measurement": "exact_no_cache",
                "decoder_metadata": dict(decoder_metadata),
            }
        )
        records.append(record)
        index.append(
            {
                **{key: common[key] for key in ("dataset", "example_id", "prompt_id", "prompt_index", "setting", "threshold")},
                "state_key": state_key,
                "step": state.step,
                "token_sequence": record["token_sequence"],
                "mask_positions": list(state.mask_positions),
                "actual_committed_anchors": record["actual_committed_anchors"],
                "anchor_count": len(state.actual_committed_anchors),
                "low_parallel": state.low_parallel,
                "generated_mask_ratio": state.generated_mask_ratio,
            }
        )
    return records, index


def _load_examples(
    config: Mapping[str, Any],
    probe_root: Path,
    *,
    setting: str,
    allow_remote: bool,
    allow_fallback: bool,
) -> tuple[list[PromptExample], list[dict[str, Any]]]:
    selected: list[PromptExample] = []
    manifests: list[dict[str, Any]] = []
    smoke_per_dataset = int(config["sampling"]["smoke_prompts_per_dataset"])
    seed = int(config["decoding"]["seed"])
    cache_dir = probe_root / "cache" / "huggingface" / "datasets"
    for dataset in config["datasets"]:
        name = str(dataset["name"])
        if setting == "smoke":
            limit: int | None = smoke_per_dataset
        elif setting == "primary":
            raw_limit = dataset.get("primary_prompt_limit")
            limit = None if raw_limit in (None, "all") else int(raw_limit)
        else:
            limit = int(dataset["sensitivity_prompt_limit"])
        loaded: BenchmarkLoadResult = load_benchmark(
            name,
            limit=limit,
            seed=seed,
            cache_dir=cache_dir,
            allow_remote=allow_remote,
            allow_fallback=allow_fallback,
        )
        manifest = loaded.manifest_record()
        manifest["requested_setting"] = setting
        manifest["requested_limit"] = limit
        manifests.append(manifest)
        selected.extend(loaded.examples)
    return selected, manifests


def load_model(config: Mapping[str, Any], probe_root: Path) -> tuple[Any, Any, Any]:
    """Reuse the pilot's no-quantization Fast-dLLM loader verbatim."""

    from scripts.run_counterfactual_probe import load_model as pilot_load_model

    return pilot_load_model(dict(config), probe_root)


def collect_observations(
    model: Any,
    tokenizer: Any,
    examples: Sequence[PromptExample],
    config: Mapping[str, Any],
    *,
    threshold: float,
    setting: str,
) -> ObservationBundle:
    """Collect natural trajectories and exact no-cache scalar observations."""

    import torch

    decoding = config["decoding"]
    sampling = config["sampling"]
    boundaries = sampling["phase_boundaries"]
    confidence_edges = sampling["confidence_bins"]
    decoder_metadata = {
        "natural_policy": "confidence_ge_threshold_plus_argmax_fallback",
        "natural_collection_use_cache": bool(decoding["use_cache_for_state_collection"]),
        "exact_measurement_use_cache": False,
        "logit_processing": decoding["logit_processing"],
        "special_token_filtering": decoding["special_token_filtering"],
        "configured_temperature": decoding["temperature"],
        "temperature_applied_by_existing_collector": False,
        "requested_block_size": decoding.get("requested_block_size"),
        "verified_analysis_block_size": decoding.get("analysis_block_size"),
    }
    trajectories: list[dict[str, Any]] = []
    state_index: list[dict[str, Any]] = []
    state_positions: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    eventual_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    natural_seconds = exact_seconds = 0.0
    natural_forward_count = exact_forward_count = 0
    sanity_forward_count = 0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for prompt_index, example in enumerate(examples):
        natural_started = time.perf_counter()
        trajectory = collect_natural_trajectory(
            model,
            tokenizer,
            example.prompt,
            prompt_index=prompt_index,
            generation_length=int(decoding["generation_length"]),
            steps=int(decoding["steps"]),
            threshold=float(threshold),
            use_cache=bool(decoding["use_cache_for_state_collection"]),
            low_parallel_max_transferred=int(decoding["low_parallel_max_transferred"]),
            low_parallel_remaining_fraction=float(decoding["low_parallel_remaining_fraction"]),
            top_k=int(decoding["top_k"]),
            analysis_block_size=decoding.get("analysis_block_size"),
            logit_processing_description=str(decoding["logit_processing"]),
        )
        torch.cuda.synchronize()
        natural_seconds += time.perf_counter() - natural_started
        natural_forward_count += len(trajectory.states)
        alignment = verify_trajectory_alignment(trajectory)
        if not alignment["valid"]:
            raise RuntimeError("Natural trajectory alignment failure: " + "; ".join(alignment["errors"]))

        # Once per setting, replay the unchanged historical collector and
        # compare every low-parallel state it exposes.  This is intentionally
        # a measurement-only sanity forward: the natural trajectory above is
        # the sole source of actions and all downstream evidence.
        if prompt_index == 0:
            baseline_states = collect_states(
                model,
                tokenizer,
                example.prompt,
                prompt_index=prompt_index,
                generation_length=int(decoding["generation_length"]),
                steps=int(decoding["steps"]),
                threshold=float(threshold),
                use_cache=bool(decoding["use_cache_for_state_collection"]),
                low_parallel_max_transferred=int(decoding["low_parallel_max_transferred"]),
                low_parallel_remaining_fraction=float(decoding["low_parallel_remaining_fraction"]),
            )
            natural_by_step = {state.step: state for state in trajectory.states}
            expected_low_parallel_steps = {
                state.step for state in trajectory.states if state.low_parallel
            }
            baseline_low_parallel_steps = {state.step for state in baseline_states}
            baseline_equivalent = True
            mismatch_reasons: list[str] = []
            if baseline_low_parallel_steps != expected_low_parallel_steps:
                baseline_equivalent = False
                mismatch_reasons.append(
                    "low-parallel state set differs: historical="
                    f"{sorted(baseline_low_parallel_steps)} logged={sorted(expected_low_parallel_steps)}"
                )
            for baseline in baseline_states:
                natural = natural_by_step.get(baseline.step)
                if natural is None or not torch.equal(baseline.input_ids, natural.input_ids):
                    baseline_equivalent = False
                    mismatch_reasons.append(f"state {baseline.step} input differs")
                    continue
                observed_positions = [anchor.position for anchor in natural.actual_committed_anchors]
                if baseline.transferred_positions != observed_positions:
                    baseline_equivalent = False
                    mismatch_reasons.append(f"state {baseline.step} transfer positions differ")
            baseline_sanity_record = {
                "dataset": example.dataset,
                "prompt_id": prompt_identifier(example),
                "prompt_index": prompt_index,
                "setting": setting,
                "threshold": threshold,
                "sanity_check": "baseline_collector_output_unchanged_by_logging",
                "valid": baseline_equivalent,
                "baseline_low_parallel_state_count": len(baseline_states),
                "logged_low_parallel_state_count": len(expected_low_parallel_steps),
                "mismatch_reasons": mismatch_reasons,
            }
            sanity_rows.append(baseline_sanity_record)
            if not baseline_equivalent:
                raise BaselineCollectorDivergenceError(baseline_sanity_record)
            # The historical collector itself cannot expose its total forward
            # count; an aligned natural trajectory has exactly one model call
            # per recorded step, giving a transparent deterministic count.
            sanity_forward_count += len(trajectory.states)

        exact_started = time.perf_counter()
        exact: ExactObservationResult = observe_trajectory_exactly(
            model,
            trajectory,
            analysis_block_size=decoding.get("analysis_block_size"),
        )
        torch.cuda.synchronize()
        exact_seconds += time.perf_counter() - exact_started
        exact_forward_count += exact.forward_count
        generated_text = _decode_generation(tokenizer, trajectory)
        correctness = gsm8k_exact_match(generated_text, example.reference) if example.dataset == "gsm8k" else None
        common = _trajectory_common(example, trajectory, generated_text, correctness, setting=setting)
        records, index = _state_records(common, exact.exact_states, decoder_metadata=decoder_metadata)
        trajectories.extend(records)
        state_index.extend(index)
        positions = _attach_state_context(
            exact.state_position_rows, common, exact.exact_states, boundaries, confidence_edges
        )
        state_positions.extend(positions)
        transitions.extend(
            _decorate_transitions(exact.transitions, common, exact.exact_states, boundaries, confidence_edges)
        )
        eventual_rows.extend(
            _decorate_eventual_rows(exact.eventual_observations, common, exact.exact_states, boundaries, confidence_edges)
        )
        prompt_rows.append(
            {
                **common,
                "natural_alignment_valid": alignment["valid"],
                "natural_alignment_checked_pairs": alignment["checked_state_pairs"],
                **exact.manifest_record(),
            }
        )
        if trajectory.states:
            reproducibility = repeated_exact_forward_check(
                model,
                trajectory.states[0].input_ids,
                trajectory.states[0].mask_positions,
            )
            sanity_rows.append({**common, "sanity_check": "repeated_exact_forward", **reproducibility})
            sanity_forward_count += 2
        sanity_rows.append(
            {
                **common,
                "sanity_check": "trajectory_alignment_and_probability_identities",
                "valid": alignment["valid"],
                "probability_sum_error": exact.max_probability_sum_error,
                "margin_log_probability_ratio_error": exact.max_margin_logprob_error,
                "natural_exact_top1_mismatch_count": exact.natural_exact_top1_mismatch_count,
            }
        )

    peak_mib = float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
    return ObservationBundle(
        trajectories=trajectories,
        state_index=state_index,
        state_position_rows=state_positions,
        transitions=transitions,
        eventual_rows=eventual_rows,
        prompt_rows=prompt_rows,
        sanity_rows=sanity_rows,
        natural_seconds=natural_seconds,
        exact_seconds=exact_seconds,
        natural_forward_count=natural_forward_count,
        exact_forward_count=exact_forward_count,
        sanity_forward_count=sanity_forward_count,
        peak_vram_mib=peak_mib,
    )


def _estimated_position_rows_per_prompt(bundle: ObservationBundle) -> float:
    prompt_count = max(len(bundle.prompt_rows), 1)
    return len(bundle.state_position_rows) / prompt_count


def estimate_resources(
    bundle: ObservationBundle,
    config: Mapping[str, Any],
    *,
    primary_example_count: int,
    sensitivity_example_upper_bound: int,
    analysis_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Conservative, smoke-calibrated resource estimate before promotion."""

    policy = config["resource_policy"]
    generation_steps = min(int(config["decoding"]["generation_length"]), int(config["decoding"]["steps"]))
    natural_per_forward = bundle.natural_seconds / max(bundle.natural_forward_count, 1)
    exact_per_forward = bundle.exact_seconds / max(bundle.exact_forward_count, 1)
    primary_forward_bound = primary_example_count * generation_steps
    sensitivity_forward_bound = (
        sensitivity_example_upper_bound
        * len(config["sampling"]["thresholds"]["sensitivity"])
        * generation_steps
    )
    observation_forwards = primary_forward_bound + sensitivity_forward_bound
    # Include the smoke collection already spent before this gate, then project
    # the planned natural + exact observation forwards.  It is intentionally a
    # safe upper bound: a short/incomplete trajectory only makes it cheaper.
    smoke_seconds = bundle.natural_seconds + bundle.exact_seconds
    observation_seconds = smoke_seconds + observation_forwards * (natural_per_forward + exact_per_forward)
    # For n actual anchors, attribution makes base + n singleton + joint + n
    # leave-one-out forwards (the n=1 implementation reuses base, so this is
    # still conservative).  A state can commit at most generation_length
    # anchors.  Do not use a typical four-anchor assumption for a hard gate.
    event_cap = int(config["sampling"]["max_counterfactual_events"])
    maximum_anchor_count = int(config["decoding"]["generation_length"])
    counterfactual_forward_bound = event_cap * (2 * maximum_anchor_count + 2)
    # Fixed and adaptive order replays each make one forward for every prefix.
    # For 2--4 anchors all permutations are evaluated; for larger sets the
    # configured deterministic cap applies.  This derives the budget from the
    # actual configuration rather than relying on a stale literal.
    order_state_cap = int(config["sampling"]["max_order_states_per_anchor_size"])
    order_permutation_cap = int(config["sampling"]["max_order_permutations"])
    order_forward_bound = sum(
        order_state_cap
        * 2
        * anchor_count
        * (math.factorial(anchor_count) if anchor_count <= 4 else min(math.factorial(anchor_count), order_permutation_cap))
        for anchor_count in (int(value) for value in config["sampling"]["order_anchor_sizes"])
    )
    analysis_seconds = float(analysis_calibration["analysis_runtime_seconds_estimate"])
    observational_seconds = observation_seconds + analysis_seconds
    exact_audit_seconds = (counterfactual_forward_bound + order_forward_bound) * exact_per_forward
    full_seconds = observational_seconds + exact_audit_seconds
    sample_bytes = len(json.dumps(bundle.trajectories, default=_json_default, ensure_ascii=False).encode("utf-8"))
    bytes_per_prompt = sample_bytes / max(len(bundle.prompt_rows), 1)
    disk_bytes = bytes_per_prompt * (
        primary_example_count
        + sensitivity_example_upper_bound * len(config["sampling"]["thresholds"]["sensitivity"])
    )
    analysis_fixed = float(analysis_calibration.get("analysis_fixed_overhead_seconds", 0.0))

    def projected_analysis_seconds(prompt_count: int) -> float:
        """Scale smoke-calibrated ranking work while retaining fixed overhead."""

        if primary_example_count <= 0:
            return analysis_seconds
        primary_work = primary_example_count * math.log2(primary_example_count + 1)
        candidate_work = prompt_count * math.log2(prompt_count + 1)
        variable = max(0.0, analysis_seconds - analysis_fixed)
        return analysis_fixed + variable * candidate_work / max(primary_work, 1.0)

    def projected_observation_seconds(prompt_count: int, sensitivity_count: int) -> float:
        forward_count = (
            prompt_count
            + sensitivity_count * len(config["sampling"]["thresholds"]["sensitivity"])
        ) * generation_steps
        return smoke_seconds + forward_count * (natural_per_forward + exact_per_forward)

    reduced_primary_count = min(100, primary_example_count)
    half_event_cap = max(1, event_cap // 2)
    half_order_state_cap = max(1, order_state_cap // 2)
    half_order_forward_bound = sum(
        half_order_state_cap
        * 2
        * anchor_count
        * (math.factorial(anchor_count) if anchor_count <= 4 else min(math.factorial(anchor_count), order_permutation_cap))
        for anchor_count in (int(value) for value in config["sampling"]["order_anchor_sizes"])
    )
    primary_only_observation = projected_observation_seconds(primary_example_count, 0)
    primary_only_analysis = projected_analysis_seconds(primary_example_count)
    reduced_causal_exact = (
        half_event_cap * (2 * maximum_anchor_count + 2) + half_order_forward_bound
    ) * exact_per_forward
    smaller_primary_observation = projected_observation_seconds(reduced_primary_count, 0)
    smaller_primary_analysis = projected_analysis_seconds(reduced_primary_count)
    reduction_options = [
        {
            "name": "primary_observational_only",
            "primary_prompt_count": primary_example_count,
            "sensitivity_prompt_upper_bound": 0,
            "counterfactual_event_cap": 0,
            "order_state_cap_per_anchor_size": 0,
            "estimated_runtime_seconds": primary_only_observation + primary_only_analysis,
        },
        {
            "name": "primary_with_half_causal_and_order_caps",
            "primary_prompt_count": primary_example_count,
            "sensitivity_prompt_upper_bound": 0,
            "counterfactual_event_cap": half_event_cap,
            "order_state_cap_per_anchor_size": half_order_state_cap,
            "estimated_runtime_seconds": primary_only_observation + primary_only_analysis + reduced_causal_exact,
        },
        {
            "name": "hundred_prompt_primary_observational_only",
            "primary_prompt_count": reduced_primary_count,
            "sensitivity_prompt_upper_bound": 0,
            "counterfactual_event_cap": 0,
            "order_state_cap_per_anchor_size": 0,
            "estimated_runtime_seconds": smaller_primary_observation + smaller_primary_analysis,
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "smoke_prompt_count": len(bundle.prompt_rows),
        "smoke_natural_forward_count": bundle.natural_forward_count,
        "smoke_exact_forward_count": bundle.exact_forward_count,
        "seconds_per_natural_forward": natural_per_forward,
        "seconds_per_exact_forward": exact_per_forward,
        "peak_vram_mib": bundle.peak_vram_mib,
        "configured_generation_length": config["decoding"]["generation_length"],
        "requested_max_generation_length": config["decoding"].get("requested_max_generation_length"),
        "requested_block_size": config["decoding"].get("requested_block_size"),
        "verified_analysis_block_size": config["decoding"].get("analysis_block_size"),
        "primary_example_count": primary_example_count,
        "sensitivity_example_upper_bound": sensitivity_example_upper_bound,
        "smoke_runtime_seconds_observed": smoke_seconds,
        "observation_forward_upper_bound": observation_forwards * 2,
        "observation_runtime_seconds_estimate": observation_seconds,
        "analysis_calibration": dict(analysis_calibration),
        "analysis_runtime_seconds_estimate": analysis_seconds,
        "observational_runtime_seconds_estimate": observational_seconds,
        "counterfactual_forward_upper_bound": counterfactual_forward_bound,
        "counterfactual_maximum_anchor_count_assumed": maximum_anchor_count,
        "order_forward_upper_bound": order_forward_bound,
        "exact_audit_runtime_seconds_estimate": exact_audit_seconds,
        "exact_audit_runtime_budget_seconds": float(policy["max_runtime_seconds"]) * float(
            policy["counterfactual_runtime_fraction"]
        ),
        "exact_audit_auto_run_eligible": bool(
            exact_audit_seconds
            < float(policy["max_runtime_seconds"]) * float(policy["counterfactual_runtime_fraction"])
        ),
        "full_runtime_seconds_estimate": full_seconds,
        "additional_disk_bytes_estimate": disk_bytes,
        "additional_disk_gib_estimate": disk_bytes / 2**30,
        "resource_reduction_options": reduction_options,
        "max_peak_vram_mib": policy["max_peak_vram_mib"],
        "max_runtime_seconds": policy["max_runtime_seconds"],
        "max_additional_disk_gib": policy["max_additional_disk_gib"],
        "observational_auto_run_eligible": bool(
            bundle.peak_vram_mib is not None
            and bundle.peak_vram_mib < float(policy["max_peak_vram_mib"])
            and observational_seconds < float(policy["max_runtime_seconds"])
            and disk_bytes < float(policy["max_additional_disk_gib"]) * 2**30
        ),
    }


def _runtime_environment(probe_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    data = environment_report(probe_root)
    data.update(
        {
            "audit_schema_version": SCHEMA_VERSION,
            "source_git_commit": git_value(probe_root, "rev-parse", "HEAD"),
            "source_git_status": git_value(probe_root, "status", "--short"),
            "fast_dllm_requested_commit": config["model"]["fast_dllm_commit"],
            "fast_dllm_checked_out_commit": git_value(
                probe_root / "vendor" / "Fast-dLLM", "rev-parse", "HEAD"
            ) if (probe_root / "vendor" / "Fast-dLLM").exists() else None,
            "platform": platform.platform(),
            "python_executable": sys.executable,
        }
    )
    return data


def _model_snapshot_metadata(model: Any, tokenizer: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Record resolved Hub snapshot hints without assuming a mutable ref is immutable."""

    model_config = getattr(model, "config", None)
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(tokenizer_kwargs, Mapping):
        tokenizer_kwargs = {}
    resolved_candidates = (
        getattr(model_config, "_commit_hash", None),
        getattr(model_config, "commit_hash", None),
        tokenizer_kwargs.get("_commit_hash"),
        tokenizer_kwargs.get("commit_hash"),
    )
    resolved = next((str(value) for value in resolved_candidates if value), None)
    return {
        "model_name": config["model"]["name"],
        "requested_hf_revision": config["model"]["hf_revision"],
        "resolved_hf_commit_hint": resolved,
        "revision_is_immutable": bool(
            resolved and str(config["model"]["hf_revision"]) == resolved
        ),
        "note": (
            "The configured Hub revision is mutable unless it is an immutable commit hash; "
            "the resolved commit hint is recorded when Transformers exposes it."
        ),
    }


def verify_existing_collector_semantics(probe_root: Path) -> dict[str, Any]:
    """Fail closed if the declared raw-logit policy no longer matches source."""

    source_path = probe_root / "scripts" / "collect_states.py"
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as error:
        return {
            "valid": False,
            "source_path": str(source_path),
            "reason": f"could not read historical collector: {type(error).__name__}",
        }
    required_fragments = {
        "fp32_raw_softmax": "torch.softmax(mask_logits.float(), dim=-1)",
        "threshold_rule": "transfer = confidence.ge(threshold)",
        "fallback_rule": "transfer[torch.argmax(confidence)] = True",
        "cache_parameter": "model(x, use_cache=use_cache).logits",
    }
    matches = {name: fragment in source for name, fragment in required_fragments.items()}
    return {
        "valid": all(matches.values()),
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "required_source_fragments": matches,
        "declared_special_token_filtering": "none_in_existing_collector",
        "declared_temperature_application": "none_in_existing_collector",
    }


def _write_observation_artifacts(paths: RunPaths, bundle: ObservationBundle) -> None:
    """Write raw scalar evidence; Parquet/tables are added by analysis stage."""

    from src.top1_reporting import write_jsonl, write_parquet

    write_jsonl(paths.raw / "trajectories.jsonl", bundle.trajectories)
    write_jsonl(paths.raw / "state_index.jsonl", bundle.state_index)
    write_jsonl(paths.raw / "state_positions.jsonl", bundle.state_position_rows)
    write_jsonl(paths.raw / "eventual_token_observations.jsonl", bundle.eventual_rows)
    write_jsonl(paths.raw / "prompt_results.jsonl", bundle.prompt_rows)
    write_jsonl(paths.raw / "sanity_checks.jsonl", bundle.sanity_rows)
    write_parquet(paths.raw / "transitions.parquet", bundle.transitions)


def _persist_baseline_divergence(
    paths: RunPaths,
    manifest: dict[str, Any],
    error: BaselineCollectorDivergenceError,
) -> None:
    """Leave a machine-readable failure record instead of an orphaned run."""

    write_json(paths.raw / "baseline_collector_divergence.json", error.sanity_record)
    manifest.update(
        {
            "status": "blocked_baseline_collector_divergence",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_collector_divergence": error.sanity_record,
        }
    )
    write_json(paths.root / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_root": str(paths.root),
                "reasons": error.sanity_record.get("mismatch_reasons", []),
            },
            ensure_ascii=False,
        )
    )


def merge_observation_bundles(*bundles: ObservationBundle) -> ObservationBundle:
    """Concatenate independently collected settings without losing provenance."""

    if not bundles:
        raise ValueError("At least one observation bundle is required.")
    return ObservationBundle(
        trajectories=[row for bundle in bundles for row in bundle.trajectories],
        state_index=[row for bundle in bundles for row in bundle.state_index],
        state_position_rows=[row for bundle in bundles for row in bundle.state_position_rows],
        transitions=[row for bundle in bundles for row in bundle.transitions],
        eventual_rows=[row for bundle in bundles for row in bundle.eventual_rows],
        prompt_rows=[row for bundle in bundles for row in bundle.prompt_rows],
        sanity_rows=[row for bundle in bundles for row in bundle.sanity_rows],
        natural_seconds=sum(bundle.natural_seconds for bundle in bundles),
        exact_seconds=sum(bundle.exact_seconds for bundle in bundles),
        natural_forward_count=sum(bundle.natural_forward_count for bundle in bundles),
        exact_forward_count=sum(bundle.exact_forward_count for bundle in bundles),
        sanity_forward_count=sum(bundle.sanity_forward_count for bundle in bundles),
        peak_vram_mib=max(
            (bundle.peak_vram_mib for bundle in bundles if bundle.peak_vram_mib is not None), default=None
        ),
    )


def _primary_rows(rows: Iterable[Mapping[str, Any]], primary_threshold: float) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row.get("setting") == "primary" and float(row.get("threshold", -1.0)) == float(primary_threshold)
    ]


def _event_identifier(record: Mapping[str, Any]) -> str:
    material = "|".join(
        str(record.get(key, ""))
        for key in ("prompt_id", "setting", "threshold", "source_step", "target_step", "position")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _select_counterfactual_cohort(
    transitions: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match and proportionally cap flip/control pairs before exact forwards."""

    from src.top1_reporting import sample_matched_event_cohort

    total_cap = int(config["sampling"]["max_counterfactual_events"])
    event_target = int(config["sampling"].get("event_target_per_cohort", total_cap // 2))
    control_target = int(config["sampling"].get("matched_control_target", total_cap // 2))
    # Controls-per-event is one by construction.  Capping independently after
    # a global lexical sort would destroy the exact matching strata, so cap
    # matched pairs together below.
    pair_cap = max(0, min(event_target, control_target, total_cap // 2))
    candidate_rows = [
        dict(row)
        for row in transitions
        if int(row.get("source_anchor_count", 0)) > 0 and row.get("source_top2_token_id") is not None
    ]
    cohort = sample_matched_event_cohort(
        candidate_rows,
        event_field="top1_flip",
        phase_field="source_phase",
        confidence_field="previous_top1_probability",
        anchor_count_field="source_anchor_count",
        confidence_bins=tuple(float(item) for item in config["sampling"]["confidence_bins"]),
        seed=int(config["sampling"]["bootstrap_seed"]),
    )
    def stratum_key(row: Mapping[str, Any]) -> str:
        return json.dumps(row.get("match_stratum", []), ensure_ascii=False, sort_keys=True, default=_json_default)

    def deterministic_order(rows: Iterable[Mapping[str, Any]], namespace: str) -> list[dict[str, Any]]:
        seed = int(config["sampling"]["bootstrap_seed"])
        return sorted(
            (dict(row) for row in rows),
            key=lambda row: hashlib.sha256(
                f"{seed}|{namespace}|{_event_identifier(row)}".encode("utf-8")
            ).hexdigest(),
        )

    events_by_stratum: dict[str, list[dict[str, Any]]] = {}
    controls_by_stratum: dict[str, list[dict[str, Any]]] = {}
    for record in cohort["event_records"]:
        events_by_stratum.setdefault(stratum_key(record), []).append(dict(record))
    for record in cohort["control_records"]:
        controls_by_stratum.setdefault(stratum_key(record), []).append(dict(record))
    available_pairs = {
        key: min(len(events_by_stratum.get(key, [])), len(controls_by_stratum.get(key, [])))
        for key in sorted(set(events_by_stratum) | set(controls_by_stratum))
    }
    available_pair_count = sum(available_pairs.values())
    if pair_cap >= available_pair_count:
        allocation = dict(available_pairs)
    elif pair_cap <= 0:
        allocation = {key: 0 for key in available_pairs}
    else:
        # Largest-remainder proportional allocation keeps every included pair
        # in a valid stratum while preventing lexicographically early datasets
        # or phases from taking the entire cap.
        raw = {key: pair_cap * count / available_pair_count for key, count in available_pairs.items()}
        allocation = {key: min(count, int(math.floor(raw[key]))) for key, count in available_pairs.items()}
        remaining = pair_cap - sum(allocation.values())
        for key in sorted(
            available_pairs,
            key=lambda item: (-(raw[item] - math.floor(raw[item])), item),
        ):
            if remaining <= 0:
                break
            if allocation[key] < available_pairs[key]:
                allocation[key] += 1
                remaining -= 1
    events: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    post_cap_strata: list[dict[str, Any]] = []
    for key in sorted(available_pairs):
        take = allocation[key]
        selected_events = deterministic_order(events_by_stratum.get(key, []), "event")[:take]
        selected_controls = deterministic_order(controls_by_stratum.get(key, []), "control")[:take]
        events.extend(selected_events)
        controls.extend(selected_controls)
        post_cap_strata.append(
            {
                "match_stratum": json.loads(key),
                "available_matched_pair_count": available_pairs[key],
                "selected_matched_pair_count": take,
            }
        )
    selected = [*events, *controls]
    for row in selected:
        row["event_id"] = _event_identifier(row)
    diagnostics = {
        **{key: value for key, value in cohort.items() if key not in {"event_records", "control_records", "records"}},
        "max_counterfactual_event_total": total_cap,
        "event_target_per_cohort": event_target,
        "matched_control_target": control_target,
        "matched_pair_cap": pair_cap,
        "available_matched_pair_count": available_pair_count,
        "post_cap_strata": post_cap_strata,
        "selected_event_count_after_cap": len(events),
        "selected_control_count_after_cap": len(controls),
    }
    return selected, diagnostics


def run_counterfactual_audit(
    model: Any,
    tokenizer: Any,
    transitions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run exact base/singleton/joint/LOO audits on matched event cohorts.

    For a non-flip control, the comparator is its *source-state top-2*, not an
    invented future commit.  This makes the control a decision-margin audit and
    labels it clearly so it is never pooled with causal flip classifications.
    """

    import torch

    from scripts.collect_states import mask_token_id
    from src.branching import Candidate
    from src.top1_counterfactual import exact_anchor_attribution
    from src.top1_observation import exact_logits

    selected, cohort_diagnostics = _select_counterfactual_cohort(transitions, config)
    mask_id = mask_token_id(model, tokenizer)
    rows: list[dict[str, Any]] = []
    total_forwards = 0
    skipped: list[dict[str, Any]] = []

    def forward(input_ids: Any) -> Any:
        return exact_logits(model, input_ids)

    for event in selected:
        anchors = tuple(
            Candidate(int(anchor["position"]), int(anchor["token_id"]))
            for anchor in event.get("source_anchors", [])
        )
        if not anchors:
            skipped.append({"event_id": event["event_id"], "reason": "no_recorded_actual_anchors"})
            continue
        base_input_ids = torch.tensor([event["source_token_sequence"]], device=next(model.parameters()).device, dtype=torch.long)
        observed_next = torch.tensor([event["next_token_sequence"]], device=base_input_ids.device, dtype=torch.long)
        is_flip = bool(event["top1_flip"])
        old_token = int(event["previous_top1_token_id"])
        new_token = int(event["next_top1_token_id"] if is_flip else event["source_top2_token_id"])
        attribution = exact_anchor_attribution(
            base_input_ids,
            anchors,
            mask_token_id=mask_id,
            target_position=int(event["position"]),
            old_token_id=old_token,
            new_token_id=new_token,
            forward=forward,
            observed_next_input_ids=observed_next,
            epsilon=float(config["decoding"].get("epsilon", 1e-12)),
        )
        payload = attribution.as_dict()
        payload.update(
            {
                "event_id": event["event_id"],
                "cohort_role": event["cohort_role"],
                "is_flip_event": is_flip,
                "control_comparator": None if is_flip else "source_state_top2_vs_current_top1",
                "dataset": event.get("dataset"),
                "prompt_id": event.get("prompt_id"),
                "prompt_index": event.get("prompt_index"),
                "setting": event.get("setting"),
                "threshold": event.get("threshold"),
                "source_step": event.get("source_step"),
                "target_step": event.get("target_step"),
                "position": event.get("position"),
                "phase": event.get("source_phase"),
                "anchor_set_size": event.get("source_anchor_count"),
                "source_top1_probability": event.get("previous_top1_probability"),
                "source_entropy": event.get("source_entropy"),
                "source_logit_margin": event.get("source_logit_margin"),
                "source_top5_mass": event.get("source_top5_mass"),
                "eventual_context": {
                    "previous_top1_token_id": event.get("previous_top1_token_id"),
                    "next_top1_token_id": event.get("next_top1_token_id"),
                },
            }
        )
        rows.append(payload)
        total_forwards += attribution.exact_forward_count
        del base_input_ids, observed_next
    return rows, {
        "cohort": cohort_diagnostics,
        "selected_record_count": len(selected),
        "audited_record_count": len(rows),
        "skipped_records": skipped,
        "exact_forward_count": total_forwards,
        "exact_forward_cache_policy": "use_cache=False",
    }


def _seed_for_state(base_seed: int, state_key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{state_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _tracked_flip_count(steps: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    previous: Mapping[str, Any] | None = None
    for step in steps:
        tracked = step.get("tracked_top1", {})
        if previous is not None:
            count += sum(previous[key] != tracked[key] for key in set(previous).intersection(tracked))
        previous = tracked
    return count


def run_order_replays(
    model: Any,
    tokenizer: Any,
    state_index: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Audit fixed and adaptive order sensitivity on actual multi-anchor states."""

    import torch

    from scripts.collect_states import mask_token_id
    from src.branching import Candidate
    from src.top1_counterfactual import replay_anchor_orders
    from src.top1_observation import exact_logits

    primary_threshold = float(config["sampling"]["thresholds"]["primary"])
    desired_sizes = set(int(item) for item in config["sampling"]["order_anchor_sizes"])
    state_groups: dict[int, list[Mapping[str, Any]]] = {}
    for state in state_index:
        if state.get("setting") != "primary" or float(state.get("threshold", -1.0)) != primary_threshold:
            continue
        size = int(state.get("anchor_count", 0))
        if size in desired_sizes:
            state_groups.setdefault(size, []).append(state)
    selected_states: list[Mapping[str, Any]] = []
    cap = int(config["sampling"]["max_order_states_per_anchor_size"])
    for size in sorted(desired_sizes):
        candidates = sorted(
            state_groups.get(size, []),
            key=lambda row: hashlib.sha256(str(row["state_key"]).encode("utf-8")).hexdigest(),
        )
        selected_states.extend(candidates[:cap])

    mask_id = mask_token_id(model, tokenizer)
    order_rows: list[dict[str, Any]] = []
    state_summaries: list[dict[str, Any]] = []
    total_forwards = 0

    def forward(input_ids: Any) -> Any:
        return exact_logits(model, input_ids)

    for state in selected_states:
        anchors = tuple(
            Candidate(int(anchor["position"]), int(anchor["token_id"]))
            for anchor in state["actual_committed_anchors"]
        )
        anchor_positions = {anchor.position for anchor in anchors}
        # Anchor keys disappear at different prefixes in different orders by
        # definition.  Track the remaining non-anchor masked positions for
        # meaningful cross-order intermediate-decision comparisons.
        tracked_non_anchor_positions = tuple(
            int(position) for position in state["mask_positions"] if int(position) not in anchor_positions
        )
        base = torch.tensor([state["token_sequence"]], device=next(model.parameters()).device, dtype=torch.long)
        audit = replay_anchor_orders(
            base,
            anchors,
            mask_token_id=mask_id,
            forward=forward,
            threshold=float(state["threshold"]),
            tracked_positions=tracked_non_anchor_positions,
            max_permutations=int(config["sampling"]["max_order_permutations"]),
            seed=_seed_for_state(int(config["decoding"]["seed"]), str(state["state_key"])),
        )
        summary = audit.as_dict()
        state_summaries.append(
            {
                "state_key": state["state_key"],
                "dataset": state.get("dataset"),
                "prompt_id": state.get("prompt_id"),
                "prompt_index": state.get("prompt_index"),
                "source_step": state.get("step"),
                "anchor_set_size": len(anchors),
                "low_parallel": state.get("low_parallel"),
                "generated_mask_ratio": state.get("generated_mask_ratio"),
                "tracked_position_scope": "non_anchor_masked_positions",
                "tracked_position_count": len(tracked_non_anchor_positions),
                **{key: value for key, value in summary.items() if key not in {"fixed_replays", "adaptive_replays"}},
            }
        )
        for replay in audit.fixed_replays:
            payload = replay.as_dict()
            payload.update(
                {
                    "state_key": state["state_key"],
                    "dataset": state.get("dataset"),
                    "prompt_id": state.get("prompt_id"),
                    "prompt_index": state.get("prompt_index"),
                    "source_step": state.get("step"),
                    "anchor_set_size": len(anchors),
                    "low_parallel": state.get("low_parallel"),
                    "tracked_position_scope": "non_anchor_masked_positions",
                    "tracked_position_count": len(tracked_non_anchor_positions),
                    "intermediate_tracked_top1_flip_count": _tracked_flip_count(payload["steps"]),
                }
            )
            order_rows.append(payload)
            total_forwards += replay.exact_forward_count
        for replay in audit.adaptive_replays:
            payload = replay.as_dict()
            payload.update(
                {
                    "state_key": state["state_key"],
                    "dataset": state.get("dataset"),
                    "prompt_id": state.get("prompt_id"),
                    "prompt_index": state.get("prompt_index"),
                    "source_step": state.get("step"),
                    "anchor_set_size": len(anchors),
                    "low_parallel": state.get("low_parallel"),
                    "tracked_position_scope": "non_anchor_masked_positions",
                    "tracked_position_count": len(tracked_non_anchor_positions),
                    "intermediate_tracked_top1_flip_count": _tracked_flip_count(payload["steps"]),
                }
            )
            order_rows.append(payload)
            total_forwards += replay.exact_forward_count
        del base
    return order_rows, state_summaries, {
        "eligible_state_counts_by_anchor_size": {str(size): len(state_groups.get(size, [])) for size in sorted(desired_sizes)},
        "selected_state_count": len(selected_states),
        "exact_forward_count": total_forwards,
        "exact_forward_cache_policy": "use_cache=False",
    }


def build_prediction_records(
    state_position_rows: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    eventual_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join scalar decision features to future labels without decoding leakage."""

    rows = prediction_label_rows(state_position_rows)
    by_position: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_position.setdefault((str(row["prompt_id"]), int(row["position"])), []).append(row)
    transition_by_target = {
        (str(row["prompt_id"]), int(row["position"]), int(row["target_step"])): row
        for row in transitions
    }
    eventual_by_state = {
        (str(row["prompt_id"]), int(row["position"]), int(row["step"])): row
        for row in eventual_rows
    }
    result: list[dict[str, Any]] = []
    for key, members in by_position.items():
        ordered = sorted(members, key=lambda row: int(row["step"]))
        for index, row in enumerate(ordered):
            payload = dict(row)
            prior = ordered[index - 1] if index >= 1 else None
            prior2 = ordered[index - 2] if index >= 2 else None
            payload["one_minus_p1"] = 1.0 - float(payload["top1_probability"])
            payload["top1_probability_slope"] = (
                float(payload["top1_probability"]) - float(prior["top1_probability"]) if prior else None
            )
            payload["entropy_slope"] = float(payload["entropy"]) - float(prior["entropy"]) if prior else None
            payload["margin_slope"] = float(payload["logit_margin"]) - float(prior["logit_margin"]) if prior else None
            payload["entropy_two_state_slope"] = (
                float(payload["entropy"]) - float(prior2["entropy"]) if prior2 else None
            )
            payload["margin_two_state_slope"] = (
                float(payload["logit_margin"]) - float(prior2["logit_margin"]) if prior2 else None
            )
            incoming = transition_by_target.get((key[0], key[1], int(payload["step"])))
            payload["tv_from_previous"] = incoming.get("full_distribution_tv") if incoming else None
            payload["js_from_previous"] = incoming.get("jensen_shannon") if incoming else None
            eventual = eventual_by_state.get((key[0], key[1], int(payload["step"])))
            payload["eventual_token_mismatch"] = (
                None
                if eventual is None or eventual.get("current_top1_matches_eventual") is None
                else not bool(eventual["current_top1_matches_eventual"])
            )
            result.append(payload)
    return result


def calibrate_analysis_runtime(
    smoke_bundle: ObservationBundle,
    config: Mapping[str, Any],
    *,
    primary_example_count: int,
) -> dict[str, Any]:
    """Time a tiny clustered-bootstrap workload and conservatively extrapolate.

    Forward-only projections are not sufficient for this protocol: 10,000
    prompt-clustered ranking resamples across the pre-specified score grid can
    dominate wall time.  The calibration uses only smoke scalar records and
    never touches the model or changes any collected trajectory.
    """

    from src.top1_reporting import predictiveness_table, quantile_bin_table

    policy = config["resource_policy"]
    calibration_replicates = int(policy["analysis_calibration_bootstrap_replicates"])
    target_replicates = int(config["sampling"]["clustered_bootstrap_replicates"])
    safety_factor = float(policy["analysis_runtime_safety_factor"])
    fixed_overhead = float(policy["analysis_fixed_overhead_seconds"])
    prediction_rows = build_prediction_records(
        smoke_bundle.state_position_rows, smoke_bundle.transitions, smoke_bundle.eventual_rows
    )
    if not prediction_rows:
        # There are no observable future labels even in the smoke run.  The
        # primary may still have them, so do not pretend a zero-row calculation
        # supplied a throughput estimate.  Fail the automatic gate closed;
        # users can inspect this artifact and choose a new smoke cohort rather
        # than silently running an unbounded 10,000-draw analysis.
        return {
            "status": "no_smoke_prediction_rows",
            "calibration_bootstrap_replicates": calibration_replicates,
            "target_bootstrap_replicates": target_replicates,
            "smoke_prediction_record_count": 0,
            "projected_primary_prediction_record_count": 0,
            "rank_bootstrap_seconds_per_smoke_draw": None,
            "quantile_bootstrap_seconds_per_smoke_draw": None,
            "rank_job_equivalent_count": 0,
            "quantile_job_equivalent_count": 0,
            "analysis_runtime_safety_factor": safety_factor,
            "analysis_fixed_overhead_seconds": fixed_overhead,
            "analysis_runtime_seconds_estimate": float(policy["max_runtime_seconds"]) + 1.0,
            "note": "No smoke future-label rows were observable; automatic promotion is blocked because bootstrap throughput could not be measured.",
        }
    started = time.perf_counter()
    predictiveness_table(
        prediction_rows,
        score_fields=("entropy",),
        outcome_field="next_step_top1_flip",
        prompt_field="prompt_id",
        score_directions=PREDICTIVENESS_SCORE_DIRECTIONS,
        quantile_bins=int(config["sampling"]["metric_quantile_bins"]),
        bootstrap_iterations=calibration_replicates,
        bootstrap_seed=int(config["sampling"]["bootstrap_seed"]),
        include_quantile_rows=False,
    )
    rank_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    quantile_bin_table(
        prediction_rows,
        score_field="entropy",
        outcome_field="next_step_top1_flip",
        prompt_field="prompt_id",
        bin_count=int(config["sampling"]["metric_quantile_bins"]),
        bootstrap_iterations=calibration_replicates,
        bootstrap_seed=int(config["sampling"]["bootstrap_seed"]) + 1,
    )
    quantile_elapsed = time.perf_counter() - started

    smoke_prompt_count = max(len(smoke_bundle.prompt_rows), 1)
    projected_rows = max(
        1,
        int(math.ceil(len(prediction_rows) / smoke_prompt_count * max(primary_example_count, 1))),
    )
    # Ranking bootstrap work is approximately O(n log n) per draw.  Scaling
    # by this deliberately upper-bounds simple per-row extrapolation.
    observed_work = len(prediction_rows) * math.log2(len(prediction_rows) + 1)
    projected_work = projected_rows * math.log2(projected_rows + 1)
    row_work_scale = projected_work / max(observed_work, 1.0)
    rank_jobs = len(PREDICTIVENESS_SCORE_FIELDS) * (
        len(PREDICTIVENESS_OUTCOMES) + 2
    )
    # Quantile curves are specified for the next-step outcome; their bin-level
    # prompt bootstrap is separately timed because it has different cost.
    quantile_jobs = len(PREDICTIVENESS_SCORE_FIELDS)
    rank_estimate = rank_elapsed / calibration_replicates * target_replicates * row_work_scale * rank_jobs
    quantile_estimate = (
        quantile_elapsed / calibration_replicates * target_replicates * row_work_scale * quantile_jobs
    )
    return {
        "status": "smoke_calibrated",
        "calibration_bootstrap_replicates": calibration_replicates,
        "target_bootstrap_replicates": target_replicates,
        "smoke_prediction_record_count": len(prediction_rows),
        "projected_primary_prediction_record_count": projected_rows,
        "row_work_scale": row_work_scale,
        "rank_bootstrap_seconds_per_smoke_draw": rank_elapsed / calibration_replicates,
        "quantile_bootstrap_seconds_per_smoke_draw": quantile_elapsed / calibration_replicates,
        "rank_job_equivalent_count": rank_jobs,
        "quantile_job_equivalent_count": quantile_jobs,
        "analysis_runtime_safety_factor": safety_factor,
        "analysis_fixed_overhead_seconds": fixed_overhead,
        "analysis_runtime_seconds_estimate": safety_factor * (rank_estimate + quantile_estimate) + fixed_overhead,
        "scope": {
            "global_score_outcome_grid": len(PREDICTIVENESS_SCORE_FIELDS) * len(PREDICTIVENESS_OUTCOMES),
            "phase_and_confidence_stratified_outcome": "next_step_top1_flip",
            "quantile_curve_outcome": "next_step_top1_flip",
        },
    }


def stratified_next_step_predictiveness(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    stratum_field: str,
    config: Mapping[str, Any],
    seed_offset: int,
) -> list[dict[str, Any]]:
    """Report the pre-specified next-step score audit within named strata.

    All six future outcomes are reported globally.  Phase/confidence slices
    focus on the primary next-step target so the required 10,000-draw
    clustered CI remains a bounded, smoke-calibrated analysis rather than an
    unaccounted combinatorial expansion.
    """

    from src.top1_reporting import predictiveness_table

    groups: dict[str, list[Mapping[str, Any]]] = {}
    display_values: dict[str, Any] = {}
    for row in prediction_rows:
        value = row.get(stratum_field)
        if value is None:
            continue
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
        groups.setdefault(key, []).append(row)
        display_values[key] = value
    output: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(groups)):
        rows = predictiveness_table(
            groups[key],
            score_fields=PREDICTIVENESS_SCORE_FIELDS,
            outcome_field="next_step_top1_flip",
            prompt_field="prompt_id",
            score_directions=PREDICTIVENESS_SCORE_DIRECTIONS,
            quantile_bins=int(config["sampling"]["metric_quantile_bins"]),
            bootstrap_iterations=int(config["sampling"]["clustered_bootstrap_replicates"]),
            bootstrap_seed=int(config["sampling"]["bootstrap_seed"]) + seed_offset + index * 41,
            include_quantile_rows=False,
        )
        for item in rows:
            output.append(
                {
                    **item,
                    "analysis_stratification": stratum_field,
                    "analysis_stratum": display_values[key],
                }
            )
    return output


def _prompt_flip_burden(position_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in position_rows:
        grouped.setdefault(str(row["prompt_id"]), []).append(row)
    result: list[dict[str, Any]] = []
    for prompt_id, rows in grouped.items():
        source = rows[0]
        flips = sum(int(row["flip_count"]) for row in rows)
        transitions = sum(int(row["eligible_transition_count"]) for row in rows)
        result.append(
            {
                "prompt_id": prompt_id,
                "prompt_index": source.get("prompt_index"),
                "dataset": source.get("dataset"),
                "gsm8k_exact_match": source.get("gsm8k_exact_match"),
                "position_count": len(rows),
                "flip_count": flips,
                "transition_count": transitions,
                "flip_burden_per_transition": flips / transitions if transitions else None,
                "ever_flip_position_rate": sum(bool(row["ever_flip"]) for row in rows) / len(rows) if rows else None,
            }
        )
    return result


def eventual_mismatch_duration_rows(
    eventual_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize how long a non-eventual current top-1 persists pre-commit."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in eventual_rows:
        if row.get("current_top1_matches_eventual") is None:
            continue  # censored / no eventual commit; never turn into a negative.
        grouped.setdefault((str(row["prompt_id"]), int(row["position"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (_prompt_id, position), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        mismatch_flags = [not bool(row["current_top1_matches_eventual"]) for row in ordered]
        streaks: list[int] = []
        active = 0
        for mismatch in mismatch_flags:
            if mismatch:
                active += 1
            elif active:
                streaks.append(active)
                active = 0
        if active:
            streaks.append(active)
        source = ordered[0]
        result.append(
            {
                "dataset": source.get("dataset"),
                "prompt_id": source.get("prompt_id"),
                "prompt_index": source.get("prompt_index"),
                "position": position,
                "eventual_known_state_count": len(ordered),
                "mismatch_state_count": sum(mismatch_flags),
                "mismatch_state_rate": sum(mismatch_flags) / len(ordered) if ordered else None,
                "ever_mismatched_before_commit": any(mismatch_flags),
                "mismatch_run_count": len(streaks),
                "longest_mismatch_run_states": max(streaks, default=0),
                "mean_mismatch_run_states": sum(streaks) / len(streaks) if streaks else 0.0,
                "initial_mismatch_run_states": streaks[0] if mismatch_flags and mismatch_flags[0] else 0,
                "first_observed_step": int(ordered[0]["step"]),
                "eventual_commit_step": source.get("eventual_commit_step"),
            }
        )
    return result


def make_analysis_tables(
    bundle: ObservationBundle,
    counterfactual_rows: Sequence[Mapping[str, Any]],
    order_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    """Compute every report table from the primary exact observational cohort."""

    from src.top1_reporting import (
        aggregate_trajectory_flips,
        counterfactual_attribution_table,
        future_token_rank_table,
        future_token_topk_table,
        grouped_flip_rate_table,
        order_sensitivity_table,
        predictiveness_table,
        quantile_bin_table,
    )

    primary_threshold = float(config["sampling"]["thresholds"]["primary"])
    transitions = _primary_rows(bundle.transitions, primary_threshold)
    states = _primary_rows(bundle.state_position_rows, primary_threshold)
    eventual = _primary_rows(bundle.eventual_rows, primary_threshold)
    bootstrap_iterations = int(config["sampling"]["clustered_bootstrap_replicates"])
    bootstrap_seed = int(config["sampling"]["bootstrap_seed"])
    aggregate = aggregate_trajectory_flips(
        transitions,
        prompt_field="prompt_id",
        position_universe=states,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    position_rows = position_trajectory_rows(states)
    # Reporting's transition reconstruction is primary for denominators and
    # CIs; this parallel state-row view keeps positions without a transition
    # visible in prompt-burden/representative-case operations.
    prompt_burden = _prompt_flip_burden(position_rows)
    mismatch_duration = eventual_mismatch_duration_rows(eventual)
    for row in position_rows:
        row["position_ever_flip"] = row["ever_flip"]
    flip_rates = grouped_flip_rate_table(
        transitions,
        group_fields=("dataset",),
        prompt_field="prompt_id",
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    flip_rates.insert(
        0,
        {
            "dataset": "all",
            "transition_count": aggregate["micro"]["transition_count"],
            "flip_count": aggregate["micro"]["flip_count"],
            "transition_flip_rate": aggregate["micro"]["transition_flip_rate"],
            "prompt_count": aggregate["macro"]["prompt_count"],
            "prompt_macro_flip_rate": aggregate["macro"]["prompt_mean_transition_flip_rate"],
            "prompt_macro_ci95_low": aggregate["macro"]["clustered_ci95_low"],
            "prompt_macro_ci95_high": aggregate["macro"]["clustered_ci95_high"],
            "micro_clustered_ci95_low": aggregate["micro"]["clustered_ci95_low"],
            "micro_clustered_ci95_high": aggregate["micro"]["clustered_ci95_high"],
            "bootstrap_iterations": bootstrap_iterations,
        },
    )
    flip_rates_by_phase = grouped_flip_rate_table(
        transitions,
        group_fields=("dataset", "phase"),
        prompt_field="prompt_id",
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed + 101,
    )
    flip_rates_by_stratum: list[dict[str, Any]] = []
    for offset, field in enumerate(
        (
            "mask_ratio_bucket",
            "confidence_bucket",
            "anchor_set_size",
            "nearest_anchor_distance_bucket",
            "low_parallel",
            "same_block_as_anchor",
            "top5_mass_ge_0_8",
            "gsm8k_exact_match",
        )
    ):
        rows = grouped_flip_rate_table(
            transitions,
            group_fields=(field,),
            prompt_field="prompt_id",
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed + 120 + offset * 13,
        )
        for row in rows:
            row["stratification"] = field
        flip_rates_by_stratum.extend(rows)
    entropy_bins = quantile_bin_table(
        transitions,
        score_field="entropy",
        outcome_field="top1_flip",
        prompt_field="prompt_id",
        bin_count=int(config["sampling"]["metric_quantile_bins"]),
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed + 202,
    )
    margin_bins = quantile_bin_table(
        transitions,
        score_field="logit_margin",
        outcome_field="top1_flip",
        prompt_field="prompt_id",
        bin_count=int(config["sampling"]["metric_quantile_bins"]),
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed + 303,
    )
    for row in entropy_bins:
        row["score"] = "entropy"
    for row in margin_bins:
        row["score"] = "logit_margin"
    topk = future_token_topk_table(
        eventual,
        group_fields=("dataset",),
        prompt_field="prompt_id",
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed + 404,
    )
    topk.extend(
        future_token_topk_table(
            eventual,
            prompt_field="prompt_id",
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed + 405,
        )
    )
    future_ranks = future_token_rank_table(
        eventual,
        group_fields=("dataset",),
        prompt_field="prompt_id",
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed + 505,
    )
    future_ranks.extend(
        future_token_rank_table(
            eventual,
            prompt_field="prompt_id",
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed + 506,
        )
    )
    prediction_rows = build_prediction_records(states, transitions, eventual)
    predictiveness: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(PREDICTIVENESS_OUTCOMES):
        rows = predictiveness_table(
            prediction_rows,
            score_fields=PREDICTIVENESS_SCORE_FIELDS,
            outcome_field=outcome,
            prompt_field="prompt_id",
            score_directions=PREDICTIVENESS_SCORE_DIRECTIONS,
            quantile_bins=int(config["sampling"]["metric_quantile_bins"]),
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed + 600 + outcome_index * 31,
            include_quantile_rows=outcome == "next_step_top1_flip",
        )
        predictiveness.extend(rows)
    predictiveness_by_phase = stratified_next_step_predictiveness(
        prediction_rows,
        stratum_field="phase",
        config=config,
        seed_offset=900,
    )
    predictiveness_by_confidence = stratified_next_step_predictiveness(
        prediction_rows,
        stratum_field="confidence_bucket",
        config=config,
        seed_offset=1_500,
    )
    causal_primary = [row for row in counterfactual_rows if row.get("cohort_role") == "event"]
    causal_controls = [row for row in counterfactual_rows if row.get("cohort_role") == "control"]
    causal_summary = counterfactual_attribution_table(causal_primary, group_fields=("dataset",))
    causal_summary.extend(counterfactual_attribution_table(causal_primary))
    causal_control_summary = counterfactual_attribution_table(causal_controls, group_fields=("dataset",))
    causal_control_summary.extend(counterfactual_attribution_table(causal_controls))
    for row in causal_control_summary:
        row["interpretation"] = "matched non-flip decision-margin control; not pooled with causal flip classes"
    order_summary = order_sensitivity_table(
        order_rows,
        state_fields=("state_key",),
        group_fields=("anchor_set_size",),
    )
    order_summary.extend(order_sensitivity_table(order_rows, state_fields=("state_key",)))
    dataset_summary: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in transitions}):
        members = [row for row in transitions if str(row.get("dataset")) == dataset]
        positions = [row for row in states if str(row.get("dataset")) == dataset]
        data_aggregate = aggregate_trajectory_flips(
            members,
            prompt_field="prompt_id",
            position_universe=positions,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed + len(dataset_summary) * 29,
        )
        dataset_summary.append(
            {
                "dataset": dataset,
                "transition_count": data_aggregate["micro"]["transition_count"],
                "transition_flip_rate": data_aggregate["micro"]["transition_flip_rate"],
                "position_count": data_aggregate["position"]["position_count"],
                "position_ever_flip_rate": data_aggregate["position"]["position_ever_flip_rate"],
                "prompt_count": data_aggregate["macro"]["prompt_count"],
                "prompt_macro_flip_rate": data_aggregate["macro"]["prompt_mean_transition_flip_rate"],
                "prompt_macro_ci95_low": data_aggregate["macro"]["clustered_ci95_low"],
                "prompt_macro_ci95_high": data_aggregate["macro"]["clustered_ci95_high"],
            }
        )
    tables = {
        "flip_rates": flip_rates,
        "flip_rates_by_phase": flip_rates_by_phase,
        "flip_rates_by_stratum": flip_rates_by_stratum,
        "flip_rates_by_entropy": [*entropy_bins, *margin_bins],
        "future_token_rank": future_ranks,
        "topk_coverage": topk,
        "metric_predictiveness": predictiveness,
        "metric_predictiveness_by_phase": predictiveness_by_phase,
        "metric_predictiveness_by_confidence": predictiveness_by_confidence,
        "counterfactual_attribution": causal_summary,
        "counterfactual_controls": causal_control_summary,
        "order_sensitivity": order_summary,
        "dataset_summary": dataset_summary,
        "position_flip_dynamics": position_rows,
        "prompt_flip_burden": prompt_burden,
        "eventual_token_mismatch_duration": mismatch_duration,
        "horizon_retention": [
            {"horizon_steps": int(horizon), **dict(values)}
            for horizon, values in sorted(aggregate["horizon_retention"].items(), key=lambda item: int(item[0]))
        ],
    }
    metadata = {
        "primary_transition_count": len(transitions),
        "primary_state_position_count": len(states),
        "primary_eventual_observation_count": len(eventual),
        "aggregate": aggregate,
        "prediction_record_count": len(prediction_rows),
        "counterfactual_flip_event_count": len(causal_primary),
        "counterfactual_control_count": len(causal_controls),
    }
    return tables, metadata, prediction_rows


def _finite_values(rows: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def write_additional_figures(
    figure_dir: Path,
    *,
    transitions: Sequence[Mapping[str, Any]],
    position_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    predictiveness_rows: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
    order_summary_rows: Sequence[Mapping[str, Any]],
    prompt_burden_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Generate the remaining required figures from compact scalar tables."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - target dependency
        return {"written": [], "skipped": [f"matplotlib unavailable: {error}"]}
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped: list[str] = []

    def save(figure: Any, name: str) -> None:
        figure.tight_layout()
        path = figure_dir / name
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        written.append(str(path))

    if transitions:
        # Phase/mask ratio view: bins are a descriptive stratification, not a
        # claim about an unverified Fast-dLLM block implementation.
        bins: dict[str, list[bool]] = {"[0,.25)": [], "[.25,.5)": [], "[.5,.75)": [], "[.75,1]": []}
        for row in transitions:
            ratio = float(row.get("source_mask_ratio", 0.0))
            label = "[0,.25)" if ratio < .25 else "[.25,.5)" if ratio < .5 else "[.5,.75)" if ratio < .75 else "[.75,1]"
            bins[label].append(bool(row.get("top1_flip", False)))
        labels = list(bins)
        values = [sum(bins[label]) / len(bins[label]) if bins[label] else 0.0 for label in labels]
        figure, axis = plt.subplots(figsize=(5.5, 3.4))
        axis.plot(labels, values, marker="o", color="#0f766e")
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("source generated-mask ratio")
        axis.set_ylabel("top-1 flip rate")
        axis.set_title("Flip rate by mask ratio")
        save(figure, "flip_rate_by_mask_ratio.png")

        for metric, name, title in (
            ("entropy", "entropy_flip_distribution.png", "Entropy: flip vs non-flip"),
            ("previous_top1_probability", "confidence_margin_flip_distribution.png", "Confidence/margin: flip vs non-flip"),
        ):
            flip = [float(row[metric]) for row in transitions if bool(row.get("top1_flip")) and row.get(metric) is not None]
            stable = [float(row[metric]) for row in transitions if not bool(row.get("top1_flip")) and row.get(metric) is not None]
            if metric == "previous_top1_probability":
                # Pair confidence and margin in one two-panel figure.
                figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
                if stable:
                    axes[0].hist(stable, bins="auto", alpha=.55, label="non-flip", color="#94a3b8")
                if flip:
                    axes[0].hist(flip, bins="auto", alpha=.55, label="flip", color="#ef4444")
                axes[0].set_title("top-1 confidence")
                margin_flip = [float(row["logit_margin"]) for row in transitions if bool(row.get("top1_flip"))]
                margin_stable = [float(row["logit_margin"]) for row in transitions if not bool(row.get("top1_flip"))]
                if margin_stable:
                    axes[1].hist(margin_stable, bins="auto", alpha=.55, label="non-flip", color="#94a3b8")
                if margin_flip:
                    axes[1].hist(margin_flip, bins="auto", alpha=.55, label="flip", color="#ef4444")
                axes[1].set_title("top-1/top-2 logit margin")
                if stable or flip:
                    axes[0].legend()
                figure.suptitle(title)
                save(figure, name)
            elif flip or stable:
                figure, axis = plt.subplots(figsize=(5.0, 3.4))
                if stable:
                    axis.hist(stable, bins="auto", alpha=.55, label="non-flip", color="#94a3b8")
                if flip:
                    axis.hist(flip, bins="auto", alpha=.55, label="flip", color="#ef4444")
                axis.set_title(title)
                axis.set_xlabel(metric)
                axis.legend()
                save(figure, name)

        old = _finite_values(transitions, "previous_top1_probability")
        new = _finite_values(transitions, "next_top1_probability")
        if old and new and len(old) == len(new):
            figure, axis = plt.subplots(figsize=(4.6, 4.0))
            colors = ["#ef4444" if bool(row.get("top1_flip")) else "#64748b" for row in transitions]
            axis.scatter(old, new, s=10, alpha=.5, c=colors)
            axis.plot([0, 1], [0, 1], color="#111827", linewidth=1, linestyle="--")
            axis.set_xlabel("old top-1 probability")
            axis.set_ylabel("new top-1 probability")
            axis.set_title("Old/new top-1 probability trajectory")
            save(figure, "old_new_top1_probability.png")

    # Entropy/margin quantile curves.  The score table carries all outcomes;
    # select the requested next-step target only.
    quantiles = [
        row for row in predictiveness_rows
        if row.get("row_type") == "quantile_bin" and row.get("outcome") == "next_step_top1_flip"
        and row.get("score") in {"entropy", "logit_margin"}
    ]
    if quantiles:
        figure, axis = plt.subplots(figsize=(5.2, 3.4))
        for score, color in (("entropy", "#7c3aed"), ("logit_margin", "#0284c7")):
            series = sorted((row for row in quantiles if row.get("score") == score), key=lambda row: int(row.get("bin_index", 0)))
            if series:
                axis.plot([row["bin_index"] for row in series], [row["empirical_probability"] for row in series], marker="o", label=score, color=color)
        axis.set_xlabel("quantile bin")
        axis.set_ylabel("empirical next-step flip probability")
        axis.set_title("Future flip probability by entropy/margin")
        axis.legend()
        save(figure, "future_flip_probability_entropy_margin.png")

    if position_rows:
        counts = [int(row["flip_count"]) for row in position_rows]
        flipbacks = [int(row["flipback_count"]) for row in position_rows]
        figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))
        axes[0].hist(counts, bins=range(0, max(counts, default=0) + 2), align="left", color="#4f46e5")
        axes[0].set_title("Flips per position")
        axes[0].set_xlabel("flip count")
        axes[1].bar(["none", "any flip-back"], [sum(value == 0 for value in flipbacks), sum(value > 0 for value in flipbacks)], color=["#94a3b8", "#f97316"])
        axes[1].set_title("Flip-back positions")
        save(figure, "position_flip_count_and_flipback.png")

    if counterfactual_rows:
        causal = [row for row in counterfactual_rows if row.get("cohort_role") == "event"]
        if causal:
            figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
            x = [float(row["singleton_effect_sum"]) for row in causal]
            y = [float(row["joint_total_effect"]) for row in causal]
            axes[0].scatter(x, y, alpha=.6, s=18, color="#d97706")
            lower, upper = min(x + y), max(x + y)
            axes[0].plot([lower, upper], [lower, upper], linestyle="--", color="#111827")
            axes[0].set_xlabel("sum singleton effects")
            axes[0].set_ylabel("actual joint effect")
            axes[0].set_title("Singleton sum vs joint effect")
            classes = sorted({str(row["classification"]) for row in causal})
            class_counts = [sum(row["classification"] == label for row in causal) / len(causal) for label in classes]
            axes[1].bar(classes, class_counts, color="#e11d48")
            axes[1].tick_params(axis="x", rotation=30)
            axes[1].set_ylim(0.0, 1.0)
            axes[1].set_title("Synergy-only and other attribution rates")
            save(figure, "singleton_joint_and_synergy.png")

    if order_summary_rows:
        # Use the cross-order table, never the existence of a change within
        # one replay.  The latter answers a different question and would make
        # this figure falsely claim order sensitivity.
        adaptive_by_size = [
            row
            for row in order_summary_rows
            if row.get("mode") == "adaptive"
            and row.get("anchor_set_size") is not None
            and row.get("order_sensitivity_rate") is not None
        ]
        if adaptive_by_size:
            adaptive_by_size.sort(key=lambda row: int(row["anchor_set_size"]))
            sizes = [int(row["anchor_set_size"]) for row in adaptive_by_size]
            rates = [float(row["order_sensitivity_rate"]) for row in adaptive_by_size]
            figure, axis = plt.subplots(figsize=(5.2, 3.4))
            axis.bar([str(size) for size in sizes], rates, color="#db2777")
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("parallel anchor-set size")
            axis.set_ylabel("adaptive order-sensitive state rate")
            axis.set_title("Cross-order sensitivity by anchor-set size")
            save(figure, "order_sensitivity_by_anchor_size.png")

    correctness = [row for row in prompt_burden_rows if row.get("dataset") == "gsm8k" and row.get("gsm8k_exact_match") is not None]
    if correctness:
        figure, axis = plt.subplots(figsize=(4.8, 3.4))
        correct = [float(row["flip_burden_per_transition"] or 0.0) for row in correctness if row["gsm8k_exact_match"]]
        incorrect = [float(row["flip_burden_per_transition"] or 0.0) for row in correctness if not row["gsm8k_exact_match"]]
        axis.boxplot([correct or [0.0], incorrect or [0.0]], labels=["correct", "incorrect"])
        axis.set_ylabel("flip burden per transition")
        axis.set_title("GSM8K correctness vs flip burden")
        save(figure, "correctness_vs_flip_burden.png")
    elif prompt_burden_rows:
        skipped.append("GSM8K correctness figure skipped: no completed scored GSM8K prompts.")
    return {"written": written, "skipped": skipped}


def make_representative_cases(
    tokenizer: Any,
    transitions: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    eventual_rows: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
    order_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Make compact human-readable cases without dumping vocabulary tensors."""

    by_state_position = {
        (str(row["prompt_id"]), int(row["step"]), int(row["position"])): row for row in states
    }
    by_eventual = {
        (str(row["prompt_id"]), int(row["step"]), int(row["position"])): row for row in eventual_rows
    }
    by_event = {
        str(row["event_id"]): row
        for row in counterfactual_rows
        if row.get("event_id") is not None
    }
    order_by_state: dict[str, list[Mapping[str, Any]]] = {}
    for row in order_rows:
        order_by_state.setdefault(str(row.get("state_key")), []).append(row)
    flips = sorted(
        (row for row in transitions if bool(row.get("top1_flip"))),
        key=lambda row: float(row.get("full_distribution_tv", 0.0)), reverse=True,
    )[:10]
    cases: list[dict[str, Any]] = []
    for row in flips:
        state = by_state_position.get((str(row["prompt_id"]), int(row["source_step"]), int(row["position"])))
        next_state = by_state_position.get((str(row["prompt_id"]), int(row["target_step"]), int(row["position"])))
        eventual = by_eventual.get((str(row["prompt_id"]), int(row["source_step"]), int(row["position"])))
        tokens = row.get("source_token_sequence", [])
        position = int(row["position"])
        lower, upper = max(0, position - 6), min(len(tokens), position + 7)
        context = [
            {"position": index, "token_id": int(tokens[index]), "token": _safe_token_text(tokenizer, int(tokens[index]))}
            for index in range(lower, upper)
        ]
        def readable_top10(source: Mapping[str, Any] | None) -> list[dict[str, Any]]:
            if source is None:
                return []
            return [
                {
                    "token_id": int(token),
                    "token": _safe_token_text(tokenizer, int(token)),
                    "probability": float(probability),
                }
                for token, probability in zip(
                    source.get("top_token_ids", []), source.get("top_probabilities", []), strict=True
                )
            ]
        source_top10 = readable_top10(state)
        next_top10 = readable_top10(next_state)
        event_id = _event_identifier(row)
        cases.append(
            {
                "event_id": event_id,
                "dataset": row.get("dataset"),
                "prompt_id": row.get("prompt_id"),
                "prompt": row.get("prompt"),
                "generated_text": row.get("generated_text"),
                "token_position": position,
                "round": row.get("source_step"),
                "nearby_context": context,
                "old_top1": {"token_id": row["previous_top1_token_id"], "token": _safe_token_text(tokenizer, int(row["previous_top1_token_id"]))},
                "new_top1": {"token_id": row["next_top1_token_id"], "token": _safe_token_text(tokenizer, int(row["next_top1_token_id"]))},
                "eventual_token": (
                    {"token_id": eventual["eventual_token_id"], "token": _safe_token_text(tokenizer, int(eventual["eventual_token_id"]))}
                    if eventual and eventual.get("eventual_known") else None
                ),
                "source_top10": source_top10,
                "next_state_top10": next_top10,
                "metrics": {
                    "entropy": row.get("source_entropy"), "logit_margin": row.get("source_logit_margin"),
                    "tv": row.get("full_distribution_tv"), "js": row.get("jensen_shannon"),
                },
                "anchors": [
                    {**anchor, "token": _safe_token_text(tokenizer, int(anchor["token_id"]))}
                    for anchor in row.get("source_anchors", [])
                ],
                "counterfactual": by_event.get(event_id),
                "order_decision_trajectories": order_by_state.get(str(row.get("state_key")), []),
            }
        )
    return cases


def _format_rate(numerator: int | None, denominator: int | None, rate: float | None) -> str:
    if numerator is None or denominator is None or rate is None:
        return "not available"
    return f"{numerator}/{denominator} ({rate * 100:.2f}%)"


def _lookup_coverage(rows: Sequence[Mapping[str, Any]], kind: str, k: int) -> Mapping[str, Any] | None:
    candidates = [
        row for row in rows
        if row.get("coverage_kind") == kind and int(row.get("k", -1)) == k and row.get("dataset") is None
    ]
    return candidates[0] if candidates else None


def make_handoff(
    *,
    paths: RunPaths,
    analysis_metadata: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    counterfactual_metadata: Mapping[str, Any],
    order_metadata: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
) -> str:
    """Answer the protocol's eleven questions with explicit denominators."""

    aggregate = analysis_metadata["aggregate"]
    micro, macro, position, flipback = aggregate["micro"], aggregate["macro"], aggregate["position"], aggregate["flipback"]
    topk = tables["topk_coverage"]
    ranks = tables["future_token_rank"]
    predictiveness = [
        row for row in tables["metric_predictiveness"]
        if row.get("row_type") == "metric" and row.get("outcome") == "next_step_top1_flip" and row.get("auroc") is not None
    ]
    best_score = max(predictiveness, key=lambda row: float(row["auroc"]), default=None)
    causal = [row for row in tables["counterfactual_attribution"] if row.get("dataset") is None]
    order = [row for row in tables["order_sensitivity"] if row.get("anchor_set_size") is None and row.get("mode") == "adaptive"]
    high_conf_transitions = [
        row for row in runtime_metadata.get("primary_transitions", []) if float(row.get("previous_top1_probability", 0.0)) >= .9
    ]
    high_conf_flips = sum(bool(row.get("top1_flip")) for row in high_conf_transitions)
    rank_lines = []
    for label in ("rank_2", "within_top3_cumulative", "within_top5_cumulative", "outside_top5"):
        match = next((row for row in ranks if row.get("rank_category") == label and row.get("dataset") is None), None)
        rank_lines.append(f"{label}={_format_rate(match.get('member_count') if match else None, match.get('observation_count') if match else None, match.get('rate') if match else None)}")
    coverage_lines = []
    for k in (1, 3, 5, 10):
        match = _lookup_coverage(topk, "eventual_committed_token", k)
        coverage_lines.append(f"top-{k}={_format_rate(match.get('covered_count') if match else None, match.get('observation_count') if match else None, match.get('coverage_rate') if match else None)}")
    causal_lines = []
    for label in ("singleton-sufficient", "synergy-only"):
        match = next((row for row in causal if row.get("classification") == label), None)
        causal_lines.append(f"{label}={_format_rate(match.get('event_count') if match else None, match.get('total_event_count') if match else None, match.get('classification_rate') if match else None)}")
    order_line = "not available"
    if order:
        row = order[0]
        order_line = _format_rate(row.get("order_sensitive_state_count"), row.get("evaluated_state_count"), row.get("order_sensitivity_rate"))
    best_score_line = (
        f"{best_score['score']} (AUROC={best_score['auroc']}, AUPRC={best_score['auprc']}, prevalence={best_score['prevalence']})"
        if best_score else "not available"
    )
    lines = [
        "# Top-1 dynamics audit handoff",
        "",
        "## Required numerical answers",
        "",
        f"1. Transition-level top-1 flip rate: {_format_rate(micro['flip_count'], micro['transition_count'], micro['transition_flip_rate'])}; prompt-macro={macro['prompt_mean_transition_flip_rate']!s}, 95% clustered CI [{macro['clustered_ci95_low']!s}, {macro['clustered_ci95_high']!s}].",
        f"2. Position ever-flip rate before commit: {_format_rate(position['ever_flip_position_count'], position['position_count'], position['position_ever_flip_rate'])}.",
        f"3. Repetition/flip-back: flip-back={_format_rate(flipback['flipback_count'], flipback['flipback_eligible_count'], flipback['flipback_rate'])}; mean flips/position={position['mean_flips_per_position']!s}.",
        f"4. Eventual committed-token coverage: {', '.join(coverage_lines)}.",
        f"5. When current top-1 differs from the eventual token: {', '.join(rank_lines)}.",
        f"6. Best single next-step flip discriminator by AUROC: {best_score_line}.",
        f"7. High-confidence (p1>=0.9) top-1 flips: {_format_rate(high_conf_flips, len(high_conf_transitions), high_conf_flips / len(high_conf_transitions) if high_conf_transitions else None)}.",
        f"8. Exact causal flip attribution: {', '.join(causal_lines)}. Counterfactual cohort exact-forward count={counterfactual_metadata.get('exact_forward_count')!s}.",
        f"9. Adaptive order-sensitive states: {order_line}; order audit exact-forward count={order_metadata.get('exact_forward_count')!s}.",
        "10. Candidate-set interpretation must use the coverage, screenability, and causal/order tables together; a flip alone is not a sufficient top-K conclusion. See `tables/topk_coverage.csv`, `tables/metric_predictiveness.csv`, `tables/counterfactual_attribution.csv`, and `tables/order_sensitivity.csv`.",
        "11. Limits: all CIs are prompt-clustered, current results condition on available public benchmark prompts and completed/censored trajectories, and low event counts or unavailable strata are reported as unavailable rather than generalized.",
        "",
        "## Run accounting",
        "",
        f"- Total frozen forwards: {runtime_metadata.get('total_forward_count')!s} (natural={runtime_metadata.get('natural_forward_count')!s}, exact observation={runtime_metadata.get('exact_observation_forward_count')!s}, sanity={runtime_metadata.get('sanity_forward_count')!s}, counterfactual={counterfactual_metadata.get('exact_forward_count')!s}, order={order_metadata.get('exact_forward_count')!s}).",
        f"- Runtime seconds: {runtime_metadata.get('runtime_seconds')!s}; peak VRAM MiB: {runtime_metadata.get('peak_vram_mib')!s}.",
        f"- Output root: `{paths.root}`.",
        "- Exact measurements/counterfactuals used `use_cache=False`; natural state collection records the existing collector policy and does not retain or reuse past key values.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "top1_dynamics_audit.yaml")
    parser.add_argument("--probe-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--mode", choices=("smoke", "auto", "observational", "all"), default="auto",
        help="smoke only; auto resource-gated full run; observational skips causal/order stages; all requests all stages subject to the same gate.",
    )
    parser.add_argument("--allow-remote-datasets", action="store_true")
    parser.add_argument("--allow-fallback-smoke-data", action="store_true")
    parser.add_argument("--skip-resource-gate", action="store_true", help="Record an override; do not use for unattended runs.")
    return parser.parse_args()


def main() -> None:
    wall_started = time.perf_counter()
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    probe_root = args.probe_root.resolve()
    configured_output = Path(args.output_root) if args.output_root else probe_root / config["storage"]["output_root"]
    paths = make_run_paths(configured_output, args.run_id)
    environment = _runtime_environment(probe_root, config)
    write_json(paths.root / "config.json", config)
    write_json(paths.root / "environment.json", environment)
    write_text(paths.root / "environment.md", environment_markdown(environment))

    # Every nontrivial execution starts with a 2--3 prompt smoke collection in
    # its own provenance-labelled setting, even when the requested final mode
    # is observational-only.  This is how the <80 GiB/<4 h/<50 GiB gate stays
    # enforceable rather than being a post-hoc report.
    initial_setting = "smoke"
    examples, dataset_manifest = _load_examples(
        config,
        probe_root,
        setting=initial_setting,
        allow_remote=args.allow_remote_datasets,
        allow_fallback=args.allow_fallback_smoke_data and initial_setting == "smoke",
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "mode": args.mode,
        "source_git_commit": environment.get("source_git_commit"),
        "fast_dllm_requested_commit": config["model"]["fast_dllm_commit"],
        "decoder_semantics": {
            "raw_vs_processed_logits": config["decoding"]["logit_processing"],
            "special_token_filtering": config["decoding"]["special_token_filtering"],
            "temperature": config["decoding"]["temperature"],
            "confidence_rule": config["decoding"]["confidence_rule"],
            "fallback_rule": config["decoding"]["fallback_rule"],
        },
        "datasets": dataset_manifest,
        "status": "started",
    }
    write_json(paths.root / "run_manifest.json", manifest)
    if not examples:
        manifest["status"] = "blocked_no_available_benchmark_prompts"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(paths.root / "run_manifest.json", manifest)
        raise RuntimeError("No public/cached benchmark prompts were available; see run_manifest.json.")

    requested_fast_dllm_commit = str(config["model"]["fast_dllm_commit"])
    checked_out_fast_dllm_commit = environment.get("fast_dllm_checked_out_commit")
    if checked_out_fast_dllm_commit != requested_fast_dllm_commit:
        manifest.update(
            {
                "status": "blocked_fast_dllm_pin_mismatch",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "fast_dllm_pin_check": {
                    "requested_commit": requested_fast_dllm_commit,
                    "checked_out_commit": checked_out_fast_dllm_commit,
                    "action": "Run scripts/setup.sh or restore vendor/Fast-dLLM to the configured detached commit.",
                },
            }
        )
        write_json(paths.root / "run_manifest.json", manifest)
        print(
            json.dumps(
                {"status": manifest["status"], "output_root": str(paths.root), "pin": manifest["fast_dllm_pin_check"]},
                ensure_ascii=False,
            )
        )
        return

    collector_semantics = verify_existing_collector_semantics(probe_root)
    manifest["decoder_semantics"]["historical_collector_verification"] = collector_semantics
    if not collector_semantics["valid"]:
        manifest.update(
            {
                "status": "blocked_decoder_semantics_unverified",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(paths.root / "run_manifest.json", manifest)
        print(
            json.dumps(
                {"status": manifest["status"], "output_root": str(paths.root), "verification": collector_semantics},
                ensure_ascii=False,
            )
        )
        return

    model, tokenizer, dtype = load_model(config, probe_root)
    model_snapshot = _model_snapshot_metadata(model, tokenizer, config)
    environment["model_snapshot"] = model_snapshot
    write_json(paths.root / "environment.json", environment)
    write_text(paths.root / "environment.md", environment_markdown(environment))
    thresholds = config["sampling"]["thresholds"]
    threshold = float(thresholds["primary"])
    try:
        smoke_bundle = collect_observations(
            model, tokenizer, examples, config, threshold=threshold, setting=initial_setting
        )
    except BaselineCollectorDivergenceError as error:
        _persist_baseline_divergence(paths, manifest, error)
        return
    _write_observation_artifacts(paths, smoke_bundle)

    primary_examples, primary_manifest = _load_examples(
        config, probe_root, setting="primary", allow_remote=args.allow_remote_datasets, allow_fallback=False
    )
    planned_sensitivity_upper_bound = (
        sum(int(dataset["sensitivity_prompt_limit"]) for dataset in config["datasets"])
        if args.mode in {"auto", "all"}
        else 0
    )
    analysis_calibration = calibrate_analysis_runtime(
        smoke_bundle,
        config,
        primary_example_count=len(primary_examples),
    )
    estimate = estimate_resources(
        smoke_bundle,
        config,
        primary_example_count=len(primary_examples),
        sensitivity_example_upper_bound=planned_sensitivity_upper_bound,
        analysis_calibration=analysis_calibration,
    )
    write_json(paths.root / "resource_estimate.json", estimate)
    write_text(paths.root / "resource_estimate.md", "# Top-1 dynamics resource estimate\n\n```json\n" + json.dumps(estimate, indent=2) + "\n```\n")
    manifest.update(
        {
            "dtype": str(dtype),
            "model_snapshot": model_snapshot,
            "smoke_or_initial_observation": {
                "prompt_count": len(smoke_bundle.prompt_rows),
                "natural_forward_count": smoke_bundle.natural_forward_count,
                "exact_forward_count": smoke_bundle.exact_forward_count,
                "peak_vram_mib": smoke_bundle.peak_vram_mib,
            },
            "resource_estimate": estimate,
        }
    )

    if args.mode == "smoke":
        manifest["status"] = "smoke_completed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(paths.root / "run_manifest.json", manifest)
        print(json.dumps({"status": manifest["status"], "output_root": str(paths.root)}, ensure_ascii=False))
        return

    if not estimate["observational_auto_run_eligible"] and not args.skip_resource_gate:
        manifest["status"] = "blocked_resource_gate"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(paths.root / "run_manifest.json", manifest)
        print(json.dumps({"status": manifest["status"], "output_root": str(paths.root), "estimate": estimate}, ensure_ascii=False))
        return

    if not primary_examples:
        manifest["status"] = "blocked_primary_benchmark_unavailable"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(paths.root / "run_manifest.json", manifest)
        raise RuntimeError("No primary benchmark prompts were available; smoke artifacts remain available for diagnosis.")

    try:
        primary_bundle = collect_observations(
            model, tokenizer, primary_examples, config, threshold=threshold, setting="primary"
        )
        all_bundles: list[ObservationBundle] = [smoke_bundle, primary_bundle]
        all_dataset_manifests = [*dataset_manifest, *primary_manifest]
        if args.mode in {"auto", "all"}:
            for sensitivity_threshold in config["sampling"]["thresholds"]["sensitivity"]:
                sensitivity_examples, sensitivity_manifest = _load_examples(
                    config,
                    probe_root,
                    setting="sensitivity",
                    allow_remote=args.allow_remote_datasets,
                    allow_fallback=False,
                )
                all_dataset_manifests.extend(sensitivity_manifest)
                if sensitivity_examples:
                    setting = f"sensitivity_threshold_{str(sensitivity_threshold).replace('.', 'p')}"
                    all_bundles.append(
                        collect_observations(
                            model,
                            tokenizer,
                            sensitivity_examples,
                            config,
                            threshold=float(sensitivity_threshold),
                            setting=setting,
                        )
                    )
    except BaselineCollectorDivergenceError as error:
        _persist_baseline_divergence(paths, manifest, error)
        return
    full_bundle = merge_observation_bundles(*all_bundles)
    _write_observation_artifacts(paths, full_bundle)

    # Counterfactual/order work runs only after natural/exact observation has
    # completed and uses its fixed baseline state records.  The separate full
    # estimate prevents an unexpectedly expensive causal audit from quietly
    # exceeding the four-hour unattended-run policy.
    full_audit_allowed = (
        (
            estimate["full_runtime_seconds_estimate"] < float(config["resource_policy"]["max_runtime_seconds"])
            and estimate["exact_audit_auto_run_eligible"]
        )
        or args.skip_resource_gate
    )
    counterfactual_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    counterfactual_metadata: dict[str, Any] = {"status": "not_run"}
    order_metadata: dict[str, Any] = {"status": "not_run"}
    order_state_summaries: list[dict[str, Any]] = []
    if args.mode in {"auto", "all"} and full_audit_allowed:
        primary_transitions = _primary_rows(full_bundle.transitions, threshold)
        counterfactual_rows, counterfactual_metadata = run_counterfactual_audit(
            model, tokenizer, primary_transitions, config
        )
        order_rows, order_state_summaries, order_metadata = run_order_replays(
            model, tokenizer, full_bundle.state_index, config
        )
    elif args.mode in {"auto", "all"}:
        counterfactual_metadata = {
            "status": "blocked_resource_estimate",
            "full_runtime_seconds_estimate": estimate["full_runtime_seconds_estimate"],
            "exact_audit_runtime_seconds_estimate": estimate["exact_audit_runtime_seconds_estimate"],
            "exact_audit_runtime_budget_seconds": estimate["exact_audit_runtime_budget_seconds"],
            "max_runtime_seconds": config["resource_policy"]["max_runtime_seconds"],
        }
        order_metadata = dict(counterfactual_metadata)
    else:
        counterfactual_metadata = {"status": "not_requested_observational_mode", "exact_forward_count": 0}
        order_metadata = {"status": "not_requested_observational_mode", "exact_forward_count": 0}

    from src.top1_reporting import write_artifact_tables, write_basic_figures, write_jsonl, write_parquet

    primary_transitions = _primary_rows(full_bundle.transitions, threshold)
    write_parquet(paths.raw / "flip_events.parquet", [row for row in primary_transitions if bool(row.get("top1_flip"))])
    write_parquet(paths.raw / "counterfactual_events.parquet", counterfactual_rows)
    write_parquet(paths.raw / "order_replays.parquet", order_rows)
    write_jsonl(paths.raw / "order_replay_state_summaries.jsonl", order_state_summaries)
    write_json(paths.raw / "counterfactual_cohort.json", counterfactual_metadata)

    tables, analysis_metadata, prediction_rows = make_analysis_tables(
        full_bundle, counterfactual_rows, order_rows, config
    )
    write_artifact_tables(paths.tables, tables, formats=("csv",))
    basic_figures = write_basic_figures(
        paths.figures,
        transitions=primary_transitions,
        future_token_rows=tables["topk_coverage"],
        quantile_rows=tables["metric_predictiveness"],
        counterfactual_rows=tables["counterfactual_attribution"],
        order_rows=tables["order_sensitivity"],
        prompt_field="prompt_id",
    )
    primary_states = _primary_rows(full_bundle.state_position_rows, threshold)
    additional_figures = write_additional_figures(
        paths.figures,
        transitions=primary_transitions,
        position_rows=tables["position_flip_dynamics"],
        prediction_rows=prediction_rows,
        predictiveness_rows=tables["metric_predictiveness"],
        counterfactual_rows=counterfactual_rows,
        order_summary_rows=tables["order_sensitivity"],
        prompt_burden_rows=tables["prompt_flip_burden"],
    )
    representative = make_representative_cases(
        tokenizer,
        primary_transitions,
        primary_states,
        _primary_rows(full_bundle.eventual_rows, threshold),
        counterfactual_rows,
        order_rows,
    )
    write_json(paths.raw / "representative_cases.json", representative)
    runtime_seconds = time.perf_counter() - wall_started
    runtime_metadata = {
        "runtime_seconds": runtime_seconds,
        "peak_vram_mib": full_bundle.peak_vram_mib,
        "natural_forward_count": full_bundle.natural_forward_count,
        "exact_observation_forward_count": full_bundle.exact_forward_count,
        "sanity_forward_count": full_bundle.sanity_forward_count,
        "total_forward_count": (
            full_bundle.natural_forward_count + full_bundle.exact_forward_count + full_bundle.sanity_forward_count
            + int(counterfactual_metadata.get("exact_forward_count", 0))
            + int(order_metadata.get("exact_forward_count", 0))
        ),
        "primary_transitions": primary_transitions,
    }
    handoff = make_handoff(
        paths=paths,
        analysis_metadata=analysis_metadata,
        tables=tables,
        counterfactual_metadata=counterfactual_metadata,
        order_metadata=order_metadata,
        runtime_metadata=runtime_metadata,
    )
    write_text(paths.root / "handoff.md", handoff)
    write_text(
        paths.root / "summary.md",
        "# Frozen LLaDA top-1 dynamics audit\n\n"
        + f"- Status: completed (`{args.mode}`).\n"
        + f"- Primary exact transition count: {analysis_metadata['primary_transition_count']}.\n"
        + f"- Total frozen forwards: {runtime_metadata['total_forward_count']}.\n"
        + f"- Peak VRAM MiB: {runtime_metadata['peak_vram_mib']}.\n"
        + f"- See [handoff](handoff.md) and `tables/` for denominators/CIs.\n",
    )
    # Keep manifest compact: aggregate contains per-position/prompt rows that
    # already live in dedicated CSV artifacts.
    aggregate_compact = {
        key: value for key, value in analysis_metadata["aggregate"].items() if key not in {"position_rows", "prompt_rows"}
    }
    manifest.update(
        {
            "datasets": all_dataset_manifests,
            "status": "completed" if full_audit_allowed or args.mode == "observational" else "observational_completed_exact_audit_blocked_resource_estimate",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {key: value for key, value in runtime_metadata.items() if key != "primary_transitions"},
            "counterfactual": counterfactual_metadata,
            "order_replay": order_metadata,
            "analysis": {**{key: value for key, value in analysis_metadata.items() if key != "aggregate"}, "aggregate": aggregate_compact},
            "figures": {"basic": basic_figures, "additional": additional_figures},
            "representative_case_count": len(representative),
        }
    )
    write_json(paths.root / "run_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output_root": str(paths.root), "forwards": runtime_metadata["total_forward_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
