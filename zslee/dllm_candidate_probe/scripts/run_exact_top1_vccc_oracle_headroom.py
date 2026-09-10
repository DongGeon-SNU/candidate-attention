#!/usr/bin/env python3
"""Experiment 3: exact all-order VCCC oracle checking.

This offline experiment starts every policy from an identical archived t=0
prompt seed. At each exact-oracle state, it ranks only still-masked generation
positions by the fresh FP32 probability gap p1-p2, takes the top-K candidate
pool (K=2, 4, 8), and exhaustively checks every subset context with a
full-vocabulary no-cache forward. The action is the maximum-cardinality
all-order-safe subset of that pool, not an unconditional top-K commit.

The comparator is the normal Fast-dLLM threshold-plus-fallback action at
threshold 0.8. Its native collector-style ``use_cache=True`` rollout is timed
as the operational Fast-dLLM reference. A second, no-cache matched rollout is
used only for a controlled final-output comparison with the oracle. The
oracle's exponential subset forwards are included in its wall-clock timing
and separately counted; this is an offline oracle, not a deployable speed
claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_vccc_oracle_audit import (
    exact_logits_batched,
    margin_summary,
    read_jsonl,
    stable_key,
)
from src.exact_top1_headroom import (
    certificate_cache_for_gamma,
    choose_largest_safe_mask,
    mask_indices,
    select_top_probability_margin_positions,
)
from src.top1_reporting import prompt_clustered_bootstrap, write_csv, write_jsonl


SCHEMA_VERSION = 4
# The matched control is the output-agreement reference. The native control
# mirrors the historical collector's direct use_cache=True invocation for the
# separately reported operational Fast-dLLM throughput.
FAST_POLICY = "fast_dllm_threshold_0p8_exact_matched"
NATIVE_FAST_POLICY = "fast_dllm_threshold_0p8_native"
EXACT_POLICY = "exact_top1_vccc_oracle"
MATCHED_FAST_FORWARD_CONVENTION = "fixed_position_full_vocabulary_use_cache_false"
NATIVE_FAST_FORWARD_CONVENTION = "historical_fast_dllm_direct_model_use_cache_true_without_past_key_reuse"
EXACT_FORWARD_CONVENTION = "every_candidate_subset_uses_fixed_position_full_vocabulary_use_cache_false"


@dataclass(frozen=True)
class PromptSeed:
    """One deterministic t=0 replay seed from the completed source audit."""

    prompt_id: str
    dataset: str | None
    example_id: str | None
    source_representative_state_key: str
    source_t0_state_key: str
    token_sequence: tuple[int, ...]
    generation_length: int
    source_t0_top1_token_ids: tuple[tuple[int, int], ...]

    @property
    def generation_start(self) -> int:
        return len(self.token_sequence) - int(self.generation_length)

    @property
    def generation_positions(self) -> tuple[int, ...]:
        return tuple(range(self.generation_start, len(self.token_sequence)))


@dataclass
class ForwardAccounting:
    """Per-rollout logical forwards and physical model batch calls."""

    action_forwards: int = 0
    subset_context_forwards: int = 0
    model_batch_calls: int = 0
    failed_model_batch_calls: int = 0
    subset_oom_retries: int = 0
    smallest_successful_subset_batch_size: int | None = None
    policy_action_steps: int = 0
    committed_tokens: int = 0

    @property
    def model_forward_evaluations(self) -> int:
        return int(self.action_forwards + self.subset_context_forwards)


@dataclass
class RolloutResult:
    """Terminal policy result plus serializable step-level evidence."""

    prompt_id: str
    dataset: str | None
    example_id: str | None
    policy: str
    candidate_k: int | None
    forward_convention: str
    completed: bool
    terminal_status: str
    final_generation_token_ids: tuple[int, ...]
    wall_seconds: float
    accounting: ForwardAccounting
    step_rows: list[dict[str, Any]]
    context_rows: list[dict[str, Any]]
    query_rows: list[dict[str, Any]]

    def prompt_row(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "dataset": self.dataset,
            "example_id": self.example_id,
            "policy": self.policy,
            "candidate_k": self.candidate_k,
            "forward_convention": self.forward_convention,
            "completed": self.completed,
            "terminal_status": self.terminal_status,
            "final_generation_token_ids": list(self.final_generation_token_ids),
            "final_generation_sha256": hashlib.sha256(
                ",".join(str(value) for value in self.final_generation_token_ids).encode("utf-8")
            ).hexdigest(),
            "generation_length": len(self.final_generation_token_ids),
            "wall_seconds": self.wall_seconds,
            "verification_aware_throughput_tokens_per_second": (
                self.accounting.committed_tokens / self.wall_seconds if self.wall_seconds > 0.0 else None
            ),
            **asdict(self.accounting),
            "model_forward_evaluations": self.accounting.model_forward_evaluations,
            "exact_forward_evaluations": (
                self.accounting.model_forward_evaluations
                if "use_cache_false" in self.forward_convention
                else None
            ),
            "mean_commits_per_action_step": (
                self.accounting.committed_tokens / self.accounting.policy_action_steps
                if self.accounting.policy_action_steps
                else None
            ),
        }


def _json_safe(value: Any) -> Any:
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


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically replace a small progress record on the persistent volume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Append durable JSONL evidence without retaining every prompt in RAM."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sync_cuda(torch: Any) -> None:
    if bool(torch.cuda.is_available()):
        torch.cuda.synchronize()


def _source_record_error(record: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    """Check source provenance before prompt selection, without outcomes."""

    required = ("state_key", "prompt_id", "step", "token_sequence", "generation_length", "measurement")
    if any(field not in record for field in required):
        return "missing_required_source_field"
    if record.get("measurement") != "exact_no_cache":
        return "source_t0_is_not_exact_no_cache_measurement"
    if not isinstance(record.get("token_sequence"), Sequence) or isinstance(record.get("token_sequence"), (str, bytes)):
        return "invalid_source_token_sequence"
    if _finite_float(record.get("generation_length")) is None or int(record["generation_length"]) < 1:
        return "invalid_source_generation_length"
    metadata = record.get("decoder_metadata")
    expected_policy = str(config["source"]["required_decoder_policy"])
    if not isinstance(metadata, Mapping) or metadata.get("natural_policy") != expected_policy:
        return "missing_or_unexpected_source_decoder_policy"
    if metadata.get("exact_measurement_use_cache") is not False:
        return "source_missing_exact_no_cache_provenance"
    return None


def _source_t0_top1_ids(
    record: Mapping[str, Any], *, generation_start: int, generation_length: int
) -> tuple[tuple[int, int], ...] | None:
    """Recover archived exact t=0 top-1 values for fail-closed replay checks."""

    summaries = record.get("position_summaries")
    if not isinstance(summaries, Mapping):
        return None
    result: list[tuple[int, int]] = []
    for position in range(int(generation_start), int(generation_start) + int(generation_length)):
        summary = summaries.get(str(position))
        if not isinstance(summary, Mapping):
            return None
        try:
            token_id = int(summary["top1_token_id"])
        except (KeyError, TypeError, ValueError):
            return None
        result.append((position, token_id))
    return tuple(result)


def _source_t0_ambiguous_tie_positions(
    record: Mapping[str, Any], *, generation_start: int, generation_length: int, tie_tolerance: float
) -> list[int] | None:
    """Find source top-k ties whose stored top-1 is not an argmax guarantee."""

    summaries = record.get("position_summaries")
    if not isinstance(summaries, Mapping):
        return None
    tied: list[int] = []
    for position in range(int(generation_start), int(generation_start) + int(generation_length)):
        summary = summaries.get(str(position))
        if not isinstance(summary, Mapping):
            return None
        margin = _finite_float(summary.get("logit_margin"))
        if margin is None:
            return None
        if abs(margin) <= float(tie_tolerance):
            tied.append(position)
    return tied


def select_prompt_seeds(
    trajectories: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    smoke: bool,
) -> tuple[list[PromptSeed], list[dict[str, Any]], dict[str, int]]:
    """Reuse the prior deterministic prompt set, then take each t=0 seed."""

    threshold = float(config["source"]["required_primary_threshold"])
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    screening: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in trajectories:
        if record.get("setting") != "primary" or _finite_float(record.get("threshold")) != threshold:
            continue
        counts["source_primary_state_rows"] += 1
        error = _source_record_error(record, config)
        common = {
            "prompt_id": record.get("prompt_id"),
            "dataset": record.get("dataset"),
            "example_id": record.get("example_id"),
            "state_key": record.get("state_key"),
            "source_step": record.get("step"),
            "selection_stage": "outcome_blind_prompt_seed_screen",
        }
        if error is not None:
            screening.append({**common, "status": "excluded_before_selection", "exclusion_reason": error})
            counts[f"excluded_{error}"] += 1
            continue
        grouped[str(record["prompt_id"])].append(record)

    representatives: list[Mapping[str, Any]] = []
    state_selection_salt = str(config["source"]["state_selection_salt"])
    prompt_selection_salt = str(config["source"]["prompt_selection_salt"])
    for prompt_id, rows in grouped.items():
        representative = min(rows, key=lambda row: stable_key(state_selection_salt, prompt_id, row["state_key"]))
        representatives.append(representative)
    representatives.sort(
        key=lambda row: stable_key(prompt_selection_salt, row["prompt_id"], row["state_key"])
    )
    cap = 1 if smoke else int(config["source"]["primary_prompt_cap"])
    chosen = representatives[:cap]
    counts["source_primary_prompt_count"] = len(grouped)
    counts["selected_prompt_count_before_t0_validation"] = len(chosen)

    seeds: list[PromptSeed] = []
    chosen_ids = {str(row["prompt_id"]) for row in chosen}
    for representative in chosen:
        prompt_id = str(representative["prompt_id"])
        t0_rows = [row for row in grouped[prompt_id] if int(row.get("step", -1)) == 0]
        common = {
            "prompt_id": prompt_id,
            "dataset": representative.get("dataset"),
            "example_id": representative.get("example_id"),
            "state_key": representative.get("state_key"),
            "source_step": representative.get("step"),
            "selection_stage": "t0_seed_resolution",
        }
        if len(t0_rows) != 1:
            reason = "missing_t0_source_state" if not t0_rows else "duplicate_t0_source_states"
            screening.append({**common, "status": "excluded_after_selection", "exclusion_reason": reason})
            counts[f"excluded_{reason}"] += 1
            continue
        t0 = t0_rows[0]
        try:
            token_sequence = tuple(int(value) for value in t0["token_sequence"])
            generation_length = int(t0["generation_length"])
        except (KeyError, TypeError, ValueError):
            screening.append({
                **common,
                "status": "excluded_after_selection",
                "exclusion_reason": "invalid_t0_token_sequence_or_generation_length",
            })
            counts["excluded_invalid_t0_token_sequence_or_generation_length"] += 1
            continue
        if len(token_sequence) <= generation_length:
            screening.append({
                **common,
                "status": "excluded_after_selection",
                "exclusion_reason": "t0_generation_length_not_shorter_than_sequence",
            })
            counts["excluded_t0_generation_length_not_shorter_than_sequence"] += 1
            continue
        configured_length = int(config["decoding"]["generation_length"])
        if generation_length != configured_length:
            screening.append({
                **common,
                "status": "excluded_after_selection",
                "exclusion_reason": "t0_generation_length_differs_from_config",
                "source_generation_length": generation_length,
                "configured_generation_length": configured_length,
            })
            counts["excluded_t0_generation_length_differs_from_config"] += 1
            continue
        t0_top1_ids = _source_t0_top1_ids(
            t0,
            generation_start=len(token_sequence) - generation_length,
            generation_length=generation_length,
        )
        if t0_top1_ids is None:
            screening.append({
                **common,
                "status": "excluded_after_selection",
                "exclusion_reason": "missing_t0_exact_position_summaries",
            })
            counts["excluded_missing_t0_exact_position_summaries"] += 1
            continue
        source_ties = _source_t0_ambiguous_tie_positions(
            t0,
            generation_start=len(token_sequence) - generation_length,
            generation_length=generation_length,
            tie_tolerance=float(config["exact_vccc_oracle"]["tie_tolerance"]),
        )
        if source_ties is None:
            screening.append({
                **common,
                "status": "excluded_after_selection",
                "exclusion_reason": "missing_t0_exact_logit_margin",
            })
            counts["excluded_missing_t0_exact_logit_margin"] += 1
            continue
        if source_ties:
            screening.append({
                **common,
                "status": "excluded_after_selection",
                "exclusion_reason": "ambiguous_t0_source_top1_logit_tie",
                "ambiguous_t0_positions": source_ties,
            })
            counts["excluded_ambiguous_t0_source_top1_logit_tie"] += 1
            continue
        seed = PromptSeed(
            prompt_id=prompt_id,
            dataset=None if t0.get("dataset") is None else str(t0.get("dataset")),
            example_id=None if t0.get("example_id") is None else str(t0.get("example_id")),
            source_representative_state_key=str(representative["state_key"]),
            source_t0_state_key=str(t0["state_key"]),
            token_sequence=token_sequence,
            generation_length=generation_length,
            source_t0_top1_token_ids=t0_top1_ids,
        )
        seeds.append(seed)
        screening.append({
            **common,
            "source_t0_state_key": seed.source_t0_state_key,
            "status": "selected_t0_seed_pending_mask_validation",
            "exclusion_reason": None,
        })
    for representative in representatives[cap:]:
        if str(representative["prompt_id"]) not in chosen_ids:
            screening.append({
                "prompt_id": representative.get("prompt_id"),
                "dataset": representative.get("dataset"),
                "example_id": representative.get("example_id"),
                "state_key": representative.get("state_key"),
                "source_step": representative.get("step"),
                "selection_stage": "outcome_blind_prompt_seed_screen",
                "status": "not_selected_prompt_cap",
                "exclusion_reason": None,
            })
    counts["selected_t0_seed_count"] = len(seeds)
    return seeds, screening, dict(counts)


def _check_source_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    status = str(manifest.get("status", ""))
    if not status.startswith("completed"):
        raise RuntimeError(f"Source top-1 run must be completed; observed status={status!r}")
    expected_commit = str(config["model"]["fast_dllm_commit"])
    source_commit = manifest.get("fast_dllm_requested_commit")
    if source_commit is not None and str(source_commit) != expected_commit:
        raise RuntimeError(
            f"Source Fast-dLLM commit mismatch: expected {expected_commit!r}, observed {source_commit!r}"
        )
    source_snapshot = manifest.get("model_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise RuntimeError("Source run lacks model_snapshot provenance required for a frozen replay.")
    for field, expected in (
        ("model_name", config["model"]["name"]),
        ("requested_hf_revision", config["model"]["hf_revision"]),
    ):
        observed = source_snapshot.get(field)
        if observed is not None and str(observed) != str(expected):
            raise RuntimeError(
                f"Source model provenance mismatch for {field}: expected {expected!r}, observed {observed!r}"
            )


def _validate_t0_masks(seed: PromptSeed, *, mask_token_id: int) -> str | None:
    invalid = [
        position
        for position in seed.generation_positions
        if int(seed.token_sequence[position]) != int(mask_token_id)
    ]
    if invalid:
        return f"t0_generation_positions_not_masked:{invalid}"
    return None


def _active_generation_positions(input_ids: Any, seed: PromptSeed, *, mask_token_id: int) -> list[int]:
    return [
        position
        for position in seed.generation_positions
        if int(input_ids[0, position].item()) == int(mask_token_id)
    ]


def _base_assignments(
    logits: Any,
    positions: Iterable[int],
    *,
    tie_tolerance: float,
) -> dict[int, dict[str, Any]]:
    """Full-vocabulary current top-1 values and p1-p2 margins for a state."""

    import torch

    result: dict[int, dict[str, Any]] = {}
    for position in sorted(int(value) for value in positions):
        token_id = int(torch.argmax(logits[0, position].float()).item())
        summary = margin_summary(logits[0], position, token_id, tie_tolerance=tie_tolerance)
        top1_probability = float(summary["assigned_probability"])
        top2_probability = float(summary["competitor_probability"])
        result[position] = {
            "token_id": token_id,
            "top1_probability": top1_probability,
            "top2_token_id": int(summary["competitor_token_id"]),
            "top2_probability": top2_probability,
            "probability_margin": top1_probability - top2_probability,
            "logit_margin": float(summary["logit_margin"]),
            "top1_matches_assignment": bool(summary["top1_matches_assignment"]),
            "is_logit_tie": bool(summary["is_logit_tie"]),
        }
    return result


def _native_fast_logits(model: Any, input_ids: Any) -> Any:
    """Match the historical collector's direct native Fast-dLLM invocation."""

    import torch

    with torch.inference_mode():
        output = model(input_ids, use_cache=True)
    logits = getattr(output, "logits", None)
    if logits is None:
        raise RuntimeError("Native Fast-dLLM model call did not return logits.")
    return logits


def _validate_source_t0_replay(seed: PromptSeed, assignments: Mapping[int, Mapping[str, Any]]) -> None:
    """Fail closed if a fresh exact replay differs from archived t=0 top-1s."""

    expected = dict(seed.source_t0_top1_token_ids)
    missing = sorted(set(expected) - set(assignments))
    mismatches = [
        {
            "position": position,
            "source_top1_token_id": expected[position],
            "fresh_top1_token_id": int(assignments[position]["token_id"]),
        }
        for position in sorted(expected)
        if position in assignments and int(assignments[position]["token_id"]) != int(expected[position])
    ]
    if missing or mismatches:
        raise RuntimeError(
            f"{seed.prompt_id}: archived exact t=0 replay mismatch; "
            f"missing_positions={missing}, mismatches={mismatches[:8]}"
        )


def _candidate_margin_payloads(
    logits: Any,
    candidate_positions: Sequence[int],
    candidate_token_ids: Sequence[int],
    *,
    tie_tolerance: float,
) -> list[list[dict[str, Any]]]:
    """Reduce all subset/target scalar margins on GPU, then transfer once.

    The competing token is computed after setting the assigned frozen token to
    ``-inf``, exactly as the scalar helper does.  It keeps the scalar helper's
    FP32 ``torch.softmax`` convention, then forms margins and the tie test on
    the host from the transferred FP32 values, matching its Python-float
    arithmetic without one CUDA synchronization per scalar query.
    """

    import torch

    positions = tuple(int(value) for value in candidate_positions)
    token_ids = tuple(int(value) for value in candidate_token_ids)
    if not positions or len(positions) != len(token_ids):
        raise ValueError("Candidate positions and token IDs must be equal-length and nonempty.")
    if logits.ndim != 3:
        raise ValueError(f"Expected [batch, sequence, vocab] logits, received shape={tuple(logits.shape)}")
    position_index = torch.tensor(positions, device=logits.device, dtype=torch.long)
    assigned_tokens = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    rows = logits[:, position_index, :].float()
    batch_size, candidate_count, vocabulary_size = rows.shape
    if vocabulary_size < 2:
        raise ValueError("Full-vocabulary VCCC margins require at least two tokens.")
    gather_index = assigned_tokens.view(1, candidate_count, 1).expand(batch_size, -1, -1)
    assigned_logits = rows.gather(dim=-1, index=gather_index).squeeze(-1)
    competitor_rows = rows.clone()
    competitor_rows.scatter_(dim=-1, index=gather_index, value=float("-inf"))
    competitor_logits, competitor_tokens = torch.max(competitor_rows, dim=-1)
    top1_tokens = torch.argmax(rows, dim=-1)
    # Reshape is a view for the contiguous advanced-indexed tensor.  Calling
    # the same FP32 softmax operator as margin_summary avoids substituting a
    # logsumexp/exp identity with slightly different roundoff behavior.
    probabilities = torch.softmax(rows.reshape(-1, vocabulary_size), dim=-1).reshape_as(rows)
    assigned_probabilities = probabilities.gather(dim=-1, index=gather_index).squeeze(-1)
    competitor_probabilities = probabilities.gather(
        dim=-1, index=competitor_tokens.unsqueeze(-1)
    ).squeeze(-1)
    packed = torch.stack(
        (
            assigned_logits,
            competitor_logits,
            assigned_probabilities,
            competitor_probabilities,
            competitor_tokens.to(dtype=rows.dtype),
            top1_tokens.to(dtype=rows.dtype),
        ),
        dim=-1,
    )
    # One device-to-host transfer per model batch replaces K scalar transfers
    # per subset context. Token IDs are safely exact in fp32 at this vocabulary
    # size; convert them back to ints in the portable record below.
    values = packed.detach().cpu().tolist()
    return [
        [
            {
                "logit_margin": float(cell[0]) - float(cell[1]),
                "probability_margin": float(cell[2]) - float(cell[3]),
                "assigned_probability": float(cell[2]),
                "competitor_probability": float(cell[3]),
                "competitor_token_id": int(cell[4]),
                "top1_token_id": int(cell[5]),
                "top1_matches_assignment": int(cell[5]) == int(token_ids[target_index]),
                "is_logit_tie": abs(float(cell[0]) - float(cell[1])) <= float(tie_tolerance),
            }
            for target_index, cell in enumerate(batch)
        ]
        for batch in values
    ]


def validate_vectorized_margin_reducer(
    logits: Any,
    candidate_positions: Sequence[int],
    candidate_token_ids: Sequence[int],
    *,
    tie_tolerance: float,
) -> dict[str, Any]:
    """One small runtime equivalence check against the scalar certificate path."""

    reduced = _candidate_margin_payloads(
        logits, candidate_positions, candidate_token_ids, tie_tolerance=tie_tolerance
    )
    mismatches: list[dict[str, Any]] = []
    for row_index in range(int(logits.shape[0])):
        for target_index, (position, token_id) in enumerate(
            zip(candidate_positions, candidate_token_ids, strict=True)
        ):
            expected = margin_summary(
                logits[row_index], int(position), int(token_id), tie_tolerance=tie_tolerance
            )
            actual = reduced[row_index][target_index]
            exact_fields = (
                "logit_margin",
                "competitor_token_id",
                "top1_token_id",
                "top1_matches_assignment",
                "is_logit_tie",
            )
            if any(actual[field] != expected[field] for field in exact_fields) or any(
                not math.isclose(
                    float(actual[field]), float(expected[field]), rel_tol=1.0e-6, abs_tol=1.0e-7
                )
                for field in ("probability_margin", "assigned_probability", "competitor_probability")
            ):
                mismatches.append({
                    "row_index": row_index,
                    "target_position": int(position),
                    "target_token_id": int(token_id),
                    "expected": expected,
                    "actual": actual,
                })
    if mismatches:
        raise RuntimeError(
            "Vectorized margin reducer diverged from scalar certificate semantics: "
            + json.dumps(_json_safe(mismatches[:3]), ensure_ascii=False, sort_keys=True)
        )
    return {
        "status": "passed",
        "validated_batch_count": int(logits.shape[0]),
        "validated_candidate_count": len(candidate_positions),
        "additional_model_forwards": 0,
    }


def _is_cuda_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def select_fast_dllm_action(probabilities: Sequence[float], threshold: float) -> tuple[int, ...]:
    """Normal threshold action with the collector's first-argmax fallback."""

    if not probabilities:
        raise ValueError("Fast-dLLM action requires at least one masked generation position")
    selected = [index for index, probability in enumerate(probabilities) if float(probability) >= float(threshold)]
    best = max(range(len(probabilities)), key=lambda index: (float(probabilities[index]), -index))
    if best not in selected:
        selected.append(best)
    return tuple(sorted(selected))


def final_token_agreement(
    baseline_tokens: Sequence[int], oracle_tokens: Sequence[int]
) -> dict[str, Any]:
    """Compare terminal generated positions, not divergent intermediate states."""

    common_length = min(len(baseline_tokens), len(oracle_tokens))
    matching = sum(
        int(baseline_tokens[index]) == int(oracle_tokens[index]) for index in range(common_length)
    )
    same_length = len(baseline_tokens) == len(oracle_tokens)
    return {
        "baseline_generation_length": len(baseline_tokens),
        "oracle_generation_length": len(oracle_tokens),
        "same_generation_length": same_length,
        "compared_token_count": common_length,
        "matching_token_count": matching,
        "token_position_agreement": matching / common_length if common_length else None,
        "final_sequence_exact_match": bool(
            same_length
            and all(int(left) == int(right) for left, right in zip(baseline_tokens, oracle_tokens, strict=True))
        ),
    }


def _context_positions(mask: int, positions: Sequence[int]) -> list[int]:
    return [int(positions[index]) for index in mask_indices(int(mask), len(positions))]


def _evaluate_candidate_contexts(
    model: Any,
    input_ids: Any,
    base_logits: Any,
    *,
    common: Mapping[str, Any],
    candidate_positions: Sequence[int],
    candidate_token_ids: Sequence[int],
    subset_batch_size: int,
    tie_tolerance: float,
    accounting: ForwardAccounting,
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate every subset of one current top-K pool with no-cache forwards."""

    import torch

    positions = tuple(int(value) for value in candidate_positions)
    token_ids = tuple(int(value) for value in candidate_token_ids)
    if not positions or len(positions) != len(token_ids):
        raise ValueError("Candidate positions and tokens must be equal-length and nonempty.")
    size = len(positions)
    margins: dict[tuple[int, int], dict[str, Any]] = {}
    contexts: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []

    def consume(
        mask: int,
        payloads: Sequence[Sequence[Mapping[str, Any]]],
        row_index: int,
        source: str,
        successful_batch_size: int,
    ) -> None:
        revealed = _context_positions(mask, positions)
        contexts.append({
            **common,
            "effective_candidate_k": size,
            "revealed_mask": int(mask),
            "revealed_positions": revealed,
            "revealed_count": len(revealed),
            "still_masked_candidate_count": size - int(mask).bit_count(),
            "forward_source": source,
            "successful_subset_batch_size": int(successful_batch_size),
            "full_reveal_has_no_candidate_target_query": int(mask).bit_count() == size,
        })
        for target_index, (position, token_id) in enumerate(zip(positions, token_ids, strict=True)):
            if int(mask) & (1 << target_index):
                continue
            payload = dict(payloads[row_index][target_index])
            payload.update({"target_index": target_index, "revealed_mask": int(mask)})
            margins[(target_index, int(mask))] = payload
            queries.append({
                **common,
                "effective_candidate_k": size,
                "target_candidate_index": target_index,
                "target_position": position,
                "target_token_id": token_id,
                "revealed_mask": int(mask),
                "logit_margin": payload["logit_margin"],
                "probability_margin": payload["probability_margin"],
                "assigned_probability": payload["assigned_probability"],
                "competitor_token_id": payload["competitor_token_id"],
                "competitor_probability": payload["competitor_probability"],
                "top1_token_id": payload["top1_token_id"],
                "top1_matches_assignment": payload["top1_matches_assignment"],
                "is_logit_tie": payload["is_logit_tie"],
                "forward_source": source,
            })

    base_payloads = _candidate_margin_payloads(
        base_logits, positions, token_ids, tie_tolerance=tie_tolerance
    )
    consume(0, base_payloads, 0, "fresh_base_exact_no_cache", 1)
    branch_masks = list(range(1, 1 << size))
    current_batch_size = max(1, int(subset_batch_size))
    start = 0
    while start < len(branch_masks):
        masks = branch_masks[start : start + current_batch_size]
        branches = None
        logits = None
        payloads = None
        try:
            branches = input_ids.expand(len(masks), -1).clone()
            for row_index, mask in enumerate(masks):
                for candidate_index, position in enumerate(positions):
                    if int(mask) & (1 << candidate_index):
                        branches[row_index, position] = token_ids[candidate_index]
            logits = exact_logits_batched(model, branches)
            # The reducer itself materializes [batch, K, vocab] FP32 views;
            # include it in the same adaptive OOM boundary as the forward.
            payloads = _candidate_margin_payloads(
                logits, positions, token_ids, tie_tolerance=tie_tolerance
            )
        except RuntimeError as error:
            accounting.failed_model_batch_calls += 1
            if not _is_cuda_oom(error) or current_batch_size == 1:
                raise
            accounting.subset_oom_retries += 1
            del logits, payloads, branches
            if bool(torch.cuda.is_available()):
                torch.cuda.empty_cache()
            current_batch_size = max(1, current_batch_size // 2)
            continue
        accounting.model_batch_calls += 1
        accounting.subset_context_forwards += len(masks)
        if accounting.smallest_successful_subset_batch_size is None:
            accounting.smallest_successful_subset_batch_size = len(masks)
        else:
            accounting.smallest_successful_subset_batch_size = min(
                accounting.smallest_successful_subset_batch_size, len(masks)
            )
        for row_index, mask in enumerate(masks):
            consume(mask, payloads, row_index, "subset_exact_no_cache", len(masks))
        start += len(masks)
        del logits, branches, payloads
    return margins, contexts, queries


def _common_step_fields(
    seed: PromptSeed,
    *,
    policy: str,
    candidate_k: int | None,
    forward_convention: str,
    step: int,
) -> dict[str, Any]:
    return {
        "prompt_id": seed.prompt_id,
        "dataset": seed.dataset,
        "example_id": seed.example_id,
        "source_representative_state_key": seed.source_representative_state_key,
        "source_t0_state_key": seed.source_t0_state_key,
        "policy": policy,
        "candidate_k": candidate_k,
        "forward_convention": forward_convention,
        "step": int(step),
    }


def _fast_rollout(
    model: Any,
    seed: PromptSeed,
    *,
    mask_token_id: int,
    threshold: float,
    tie_tolerance: float,
    policy: str,
    forward_convention: str,
    native_use_cache: bool,
    validate_source_t0: bool,
) -> RolloutResult:
    """Fresh threshold=.8 rollout from the shared t=0 seed."""

    import torch

    device = next(model.parameters()).device
    x = torch.tensor([seed.token_sequence], device=device, dtype=torch.long)
    accounting = ForwardAccounting()
    rows: list[dict[str, Any]] = []
    _sync_cuda(torch)
    started = time.perf_counter()
    for step in range(int(seed.generation_length)):
        active = _active_generation_positions(x, seed, mask_token_id=mask_token_id)
        if not active:
            break
        logits = _native_fast_logits(model, x) if native_use_cache else exact_logits_batched(model, x)
        accounting.action_forwards += 1
        accounting.model_batch_calls += 1
        assignments = _base_assignments(logits, active, tie_tolerance=tie_tolerance)
        if step == 0 and validate_source_t0:
            _validate_source_t0_replay(seed, assignments)
        probabilities = [float(assignments[position]["top1_probability"]) for position in active]
        selected_indices = select_fast_dllm_action(probabilities, threshold)
        threshold_indices = tuple(
            index for index, probability in enumerate(probabilities) if probability >= float(threshold)
        )
        fallback_used = not threshold_indices
        selected_positions = [int(active[index]) for index in selected_indices]
        selected_tokens = [int(assignments[position]["token_id"]) for position in selected_positions]
        for position, token_id in zip(selected_positions, selected_tokens, strict=True):
            x[0, position] = token_id
        common = _common_step_fields(
            seed,
            policy=policy,
            candidate_k=None,
            forward_convention=forward_convention,
            step=step,
        )
        rows.append({
            **common,
            "remaining_generation_masks_before": len(active),
            "remaining_generation_masks_after": len(active) - len(selected_positions),
            "baseline_threshold": float(threshold),
            "candidate_requested_k": None,
            "effective_candidate_k": None,
            "candidate_positions": [],
            "candidate_top1_token_ids": [],
            "candidate_top1_probabilities": [],
            "candidate_top2_probabilities": [],
            "candidate_probability_margins": [],
            "full_candidate_set_certificate_pass": None,
            "selected_set_certificate_pass": None,
            "selected_set_certificate_min_logit_margin": None,
            "selected_set_certificate_query_count": None,
            "selected_positions": selected_positions,
            "selected_top1_token_ids": selected_tokens,
            "actual_commit_set_size": len(selected_positions),
            "threshold_eligible_positions": [int(active[index]) for index in threshold_indices],
            "fallback_used": fallback_used,
        })
        accounting.policy_action_steps += 1
        accounting.committed_tokens += len(selected_positions)
        del logits
    _sync_cuda(torch)
    elapsed = time.perf_counter() - started
    completed = not _active_generation_positions(x, seed, mask_token_id=mask_token_id)
    return RolloutResult(
        prompt_id=seed.prompt_id,
        dataset=seed.dataset,
        example_id=seed.example_id,
        policy=policy,
        candidate_k=None,
        forward_convention=forward_convention,
        completed=completed,
        terminal_status="completed" if completed else "max_steps_reached_with_generation_masks",
        final_generation_token_ids=tuple(
            int(value) for value in x[0, seed.generation_start :].detach().cpu().tolist()
        ),
        wall_seconds=elapsed,
        accounting=accounting,
        step_rows=rows,
        context_rows=[],
        query_rows=[],
    )


def _exact_oracle_rollout(
    model: Any,
    seed: PromptSeed,
    *,
    mask_token_id: int,
    candidate_k: int,
    subset_batch_size: int,
    gamma: float,
    tie_tolerance: float,
) -> RolloutResult:
    """Fresh exact-VCCC rollout that commits C_K at every action step."""

    import torch

    device = next(model.parameters()).device
    x = torch.tensor([seed.token_sequence], device=device, dtype=torch.long)
    accounting = ForwardAccounting()
    rows: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    _sync_cuda(torch)
    started = time.perf_counter()
    for step in range(int(seed.generation_length)):
        active = _active_generation_positions(x, seed, mask_token_id=mask_token_id)
        if not active:
            break
        base_logits = exact_logits_batched(model, x)
        accounting.action_forwards += 1
        accounting.model_batch_calls += 1
        assignments = _base_assignments(base_logits, active, tie_tolerance=tie_tolerance)
        ranked_active_positions = select_top_probability_margin_positions(assignments, len(assignments))
        candidate_positions = ranked_active_positions[: int(candidate_k)]
        candidate_token_ids = tuple(int(assignments[position]["token_id"]) for position in candidate_positions)
        candidate_probabilities = tuple(
            float(assignments[position]["top1_probability"]) for position in candidate_positions
        )
        candidate_top2_probabilities = tuple(
            float(assignments[position]["top2_probability"]) for position in candidate_positions
        )
        candidate_probability_margins = tuple(
            float(assignments[position]["probability_margin"]) for position in candidate_positions
        )
        common = _common_step_fields(
            seed,
            policy=EXACT_POLICY,
            candidate_k=int(candidate_k),
            forward_convention=EXACT_FORWARD_CONVENTION,
            step=step,
        )
        margins, state_contexts, state_queries = _evaluate_candidate_contexts(
            model,
            x,
            base_logits,
            common=common,
            candidate_positions=candidate_positions,
            candidate_token_ids=candidate_token_ids,
            subset_batch_size=int(subset_batch_size),
            tie_tolerance=float(tie_tolerance),
            accounting=accounting,
        )
        contexts.extend(state_contexts)
        queries.extend(state_queries)
        certificates = certificate_cache_for_gamma(
            margins,
            len(candidate_positions),
            float(gamma),
            tolerance=float(tie_tolerance),
        )
        full_mask = (1 << len(candidate_positions)) - 1
        full_certificate = certificates[full_mask]
        selected_mask = choose_largest_safe_mask(
            certificates,
            candidate_probabilities,
            candidate_positions,
            tie_scores=candidate_probability_margins,
        )
        if selected_mask is None or int(selected_mask) == 0:
            raise RuntimeError(
                f"{seed.prompt_id}: exact VCCC found no nonempty safe subset at K={candidate_k}, step={step}"
            )
        selected_certificate = certificates[int(selected_mask)]
        if not selected_certificate.passes:
            raise RuntimeError("Selected exact VCCC set does not pass its cached all-order certificate.")
        selected_indices = mask_indices(int(selected_mask), len(candidate_positions))
        selected_positions = [int(candidate_positions[index]) for index in selected_indices]
        selected_tokens = [int(candidate_token_ids[index]) for index in selected_indices]
        for position, token_id in zip(selected_positions, selected_tokens, strict=True):
            x[0, position] = token_id
        rows.append({
            **common,
            "remaining_generation_masks_before": len(active),
            "remaining_generation_masks_after": len(active) - len(selected_positions),
            "baseline_threshold": None,
            "candidate_requested_k": int(candidate_k),
            "effective_candidate_k": len(candidate_positions),
            "candidate_positions": list(candidate_positions),
            "candidate_top1_token_ids": list(candidate_token_ids),
            "candidate_top1_probabilities": list(candidate_probabilities),
            "candidate_top2_probabilities": list(candidate_top2_probabilities),
            "candidate_probability_margins": list(candidate_probability_margins),
            # Keep the entire <=16-position current ranking and cut-off so
            # P_K can be independently reconstructed from raw action rows.
            "ranked_active_positions": list(ranked_active_positions),
            "ranked_active_top1_probabilities": [
                float(assignments[position]["top1_probability"]) for position in ranked_active_positions
            ],
            "ranked_active_top2_probabilities": [
                float(assignments[position]["top2_probability"]) for position in ranked_active_positions
            ],
            "ranked_active_probability_margins": [
                float(assignments[position]["probability_margin"]) for position in ranked_active_positions
            ],
            "candidate_cutoff_probability_margin": (
                float(assignments[candidate_positions[-1]]["probability_margin"])
                if candidate_positions
                else None
            ),
            "next_excluded_probability_margin": (
                float(assignments[ranked_active_positions[len(candidate_positions)]]["probability_margin"])
                if len(ranked_active_positions) > len(candidate_positions)
                else None
            ),
            "full_candidate_set_certificate_pass": bool(full_certificate.passes),
            "full_candidate_set_certificate_min_logit_margin": full_certificate.certificate_margin,
            "full_candidate_set_certificate_query_count": full_certificate.query_count,
            "selected_set_certificate_pass": bool(selected_certificate.passes),
            "selected_set_certificate_min_logit_margin": selected_certificate.certificate_margin,
            "selected_set_certificate_query_count": selected_certificate.query_count,
            "selected_set_certificate_missing_query_count": selected_certificate.missing_query_count,
            "selected_positions": selected_positions,
            "selected_top1_token_ids": selected_tokens,
            "actual_commit_set_size": len(selected_positions),
            "threshold_eligible_positions": [],
            "fallback_used": False,
        })
        accounting.policy_action_steps += 1
        accounting.committed_tokens += len(selected_positions)
        del base_logits
    _sync_cuda(torch)
    elapsed = time.perf_counter() - started
    completed = not _active_generation_positions(x, seed, mask_token_id=mask_token_id)
    return RolloutResult(
        prompt_id=seed.prompt_id,
        dataset=seed.dataset,
        example_id=seed.example_id,
        policy=EXACT_POLICY,
        candidate_k=int(candidate_k),
        forward_convention=EXACT_FORWARD_CONVENTION,
        completed=completed,
        terminal_status="completed" if completed else "max_steps_reached_with_generation_masks",
        final_generation_token_ids=tuple(
            int(value) for value in x[0, seed.generation_start :].detach().cpu().tolist()
        ),
        wall_seconds=elapsed,
        accounting=accounting,
        step_rows=rows,
        context_rows=contexts,
        query_rows=queries,
    )


def _descriptive_quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
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
    values = [
        {"prompt_id": str(row["prompt_id"]), field: value}
        for row in rows
        if row.get("prompt_id") is not None and (value := _finite_float(row.get(field))) is not None
    ]
    if not values:
        return {
            "availability": "unavailable",
            "observation_count": 0,
            "micro_estimate": None,
            "prompt_macro_estimate": None,
            "prompt_clustered_ci95_low": None,
            "prompt_clustered_ci95_high": None,
            "prompt_cluster_count": 0,
        }
    bootstrap = prompt_clustered_bootstrap(
        values,
        value_field=field,
        prompt_field="prompt_id",
        iterations=int(iterations),
        seed=int(seed),
        weighting="macro",
    )
    return {
        "availability": "available",
        "observation_count": len(values),
        "micro_estimate": math.fsum(float(row[field]) for row in values) / len(values),
        "prompt_macro_estimate": bootstrap["estimate"],
        "prompt_clustered_ci95_low": bootstrap["ci95_low"],
        "prompt_clustered_ci95_high": bootstrap["ci95_high"],
        "prompt_cluster_count": bootstrap["cluster_count"],
    }


def _attach_metric(
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


def throughput_summary(
    prompt_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Policy/K summaries with full verification cost in measured timing."""

    groups: dict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        groups[(str(row["policy"]), row.get("candidate_k"))].append(row)
    iterations = int(config["reporting"]["clustered_bootstrap_replicates"])
    seed = int(config["reporting"]["bootstrap_seed"])
    result: list[dict[str, Any]] = []
    for index, ((policy, candidate_k), rows) in enumerate(
        sorted(groups.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else int(item[0][1])))
    ):
        total_seconds = math.fsum(float(row["wall_seconds"]) for row in rows)
        total_tokens = math.fsum(float(row["committed_tokens"]) for row in rows)
        summary: dict[str, Any] = {
            "policy": policy,
            "candidate_k": candidate_k,
            "prompt_count": len(rows),
            "completed_prompt_count": sum(bool(row.get("completed")) for row in rows),
            "total_committed_generated_positions": total_tokens,
            "total_wall_seconds": total_seconds,
            "aggregate_verification_aware_throughput_tokens_per_second": (
                total_tokens / total_seconds if total_seconds > 0.0 else None
            ),
            "total_action_forwards": sum(int(row["action_forwards"]) for row in rows),
            "total_subset_context_forwards": sum(int(row["subset_context_forwards"]) for row in rows),
            "total_model_forward_evaluations": sum(int(row["model_forward_evaluations"]) for row in rows),
            "total_exact_forward_evaluations": sum(
                int(value)
                for row in rows
                if (value := _finite_float(row.get("exact_forward_evaluations"))) is not None
            ),
            "total_model_batch_calls": sum(int(row["model_batch_calls"]) for row in rows),
            "total_failed_model_batch_calls": sum(int(row["failed_model_batch_calls"]) for row in rows),
            "total_subset_oom_retries": sum(int(row["subset_oom_retries"]) for row in rows),
        }
        _attach_metric(
            summary,
            "throughput",
            rows,
            field="verification_aware_throughput_tokens_per_second",
            iterations=iterations,
            seed=seed + index * 17,
        )
        _attach_metric(
            summary,
            "commits_per_step",
            rows,
            field="mean_commits_per_action_step",
            iterations=iterations,
            seed=seed + index * 17 + 1,
        )
        _attach_metric(
            summary,
            "steps",
            rows,
            field="policy_action_steps",
            iterations=iterations,
            seed=seed + index * 17 + 2,
        )
        result.append(summary)
    return result


def commit_batch_summary(step_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Distribution of actual batch sizes, separate from throughput."""

    groups: dict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for row in step_rows:
        groups[(str(row["policy"]), row.get("candidate_k"))].append(row)
    result: list[dict[str, Any]] = []
    for (policy, candidate_k), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else int(item[0][1]))
    ):
        sizes = [float(row["actual_commit_set_size"]) for row in rows]
        full_pass = [
            float(bool(row["full_candidate_set_certificate_pass"]))
            for row in rows
            if row.get("full_candidate_set_certificate_pass") is not None
        ]
        effective = [
            float(row["effective_candidate_k"])
            for row in rows
            if _finite_float(row.get("effective_candidate_k")) is not None
        ]
        result.append({
            "policy": policy,
            "candidate_k": candidate_k,
            "action_step_count": len(rows),
            "actual_commit_set_size_mean": math.fsum(sizes) / len(sizes) if sizes else None,
            "actual_commit_set_size_median": _descriptive_quantile(sizes, 0.5),
            "actual_commit_set_size_q25": _descriptive_quantile(sizes, 0.25),
            "actual_commit_set_size_q75": _descriptive_quantile(sizes, 0.75),
            "actual_commit_set_size_min": min(sizes) if sizes else None,
            "actual_commit_set_size_max": max(sizes) if sizes else None,
            "effective_candidate_k_mean": math.fsum(effective) / len(effective) if effective else None,
            "full_topk_certificate_pass_rate": math.fsum(full_pass) / len(full_pass) if full_pass else None,
        })
    return result


def agreement_rows(
    baseline_results: Sequence[RolloutResult], oracle_results: Sequence[RolloutResult]
) -> list[dict[str, Any]]:
    baseline_by_prompt = {result.prompt_id: result for result in baseline_results}
    result: list[dict[str, Any]] = []
    for oracle in oracle_results:
        baseline = baseline_by_prompt.get(oracle.prompt_id)
        if baseline is None:
            continue
        raw_agreement = final_token_agreement(
            baseline.final_generation_token_ids, oracle.final_generation_token_ids
        )
        both_completed = bool(baseline.completed and oracle.completed)
        primary_agreement = raw_agreement if both_completed else {
            **raw_agreement,
            "compared_token_count": 0,
            "matching_token_count": 0,
            "token_position_agreement": None,
            "final_sequence_exact_match": None,
        }
        result.append({
            "prompt_id": oracle.prompt_id,
            "dataset": oracle.dataset,
            "example_id": oracle.example_id,
            "baseline_policy": baseline.policy,
            "baseline_forward_convention": baseline.forward_convention,
            "candidate_k": oracle.candidate_k,
            "baseline_completed": baseline.completed,
            "oracle_completed": oracle.completed,
            "both_completed": both_completed,
            "agreement_status": (
                "included_both_completed"
                if both_completed
                else "excluded_incomplete_rollout_from_primary_agreement"
            ),
            "raw_compared_token_count": raw_agreement["compared_token_count"],
            "raw_matching_token_count": raw_agreement["matching_token_count"],
            "raw_token_position_agreement": raw_agreement["token_position_agreement"],
            "raw_final_sequence_exact_match": raw_agreement["final_sequence_exact_match"],
            **primary_agreement,
        })
    return result


def agreement_summary(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    iterations = int(config["reporting"]["clustered_bootstrap_replicates"])
    seed = int(config["reporting"]["bootstrap_seed"])
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("candidate_k") is not None:
            groups[(str(row.get("baseline_policy")), int(row["candidate_k"]))].append(row)
    output: list[dict[str, Any]] = []
    for offset, ((baseline_policy, candidate_k), members) in enumerate(sorted(groups.items())):
        eligible = [row for row in members if bool(row.get("both_completed"))]
        matching = sum(int(row["matching_token_count"]) for row in eligible)
        compared = sum(int(row["compared_token_count"]) for row in eligible)
        summary: dict[str, Any] = {
            "baseline_policy": baseline_policy,
            "candidate_k": candidate_k,
            "paired_prompt_count": len(members),
            "both_completed_prompt_count": sum(bool(row.get("both_completed")) for row in members),
            "agreement_eligible_prompt_count": len(eligible),
            "micro_token_position_agreement": matching / compared if compared else None,
        }
        _attach_metric(
            summary,
            "final_sequence_exact_match",
            members,
            field="final_sequence_exact_match",
            iterations=iterations,
            seed=seed + 101 + offset,
        )
        _attach_metric(
            summary,
            "token_position_agreement",
            members,
            field="token_position_agreement",
            iterations=iterations,
            seed=seed + 201 + offset,
        )
        _attach_metric(
            summary,
            "both_completed",
            members,
            field="both_completed",
            iterations=iterations,
            seed=seed + 301 + offset,
        )
        output.append(summary)
    return output


def _fmt_rate(value: Any) -> str:
    numeric = _finite_float(value)
    return "unavailable" if numeric is None else f"{numeric * 100.0:.2f}%"


def _fmt_value(value: Any, digits: int = 3) -> str:
    numeric = _finite_float(value)
    return "unavailable" if numeric is None else f"{numeric:.{digits}f}"


def _summary_row(
    rows: Sequence[Mapping[str, Any]], policy: str, candidate_k: int | None
) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if str(row.get("policy")) == policy and row.get("candidate_k") == candidate_k
        ),
        None,
    )


def _policy_display_label(policy: Any, candidate_k: Any) -> str:
    if str(policy) == NATIVE_FAST_POLICY:
        return "Fast tau=.8 native"
    if str(policy) == FAST_POLICY:
        return "Fast tau=.8 matched"
    if candidate_k is not None:
        return f"Exact K={candidate_k}"
    return str(policy)


def write_figures(
    output_dir: Path,
    *,
    throughput_rows: Sequence[Mapping[str, Any]],
    batch_rows: Sequence[Mapping[str, Any]],
    agreement_rows_: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Compact figures; data tables remain the authoritative numerical record."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    labels = [_policy_display_label(row.get("policy"), row.get("candidate_k")) for row in throughput_rows]
    values = [
        _finite_float(row.get("aggregate_verification_aware_throughput_tokens_per_second")) or 0.0
        for row in throughput_rows
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color=["#2563eb" if label.startswith("Fast") else "#dc2626" for label in labels])
    axis.set_ylabel("generated positions / second")
    axis.set_title("Verification-aware throughput")
    figure.tight_layout()
    path = output_dir / "verification_aware_throughput.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(str(path))

    labels = [_policy_display_label(row.get("policy"), row.get("candidate_k")) for row in batch_rows]
    values = [_finite_float(row.get("actual_commit_set_size_mean")) or 0.0 for row in batch_rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color=["#2563eb" if label.startswith("Fast") else "#7c3aed" for label in labels])
    axis.set_ylabel("mean actual commits / action step")
    axis.set_title("Actual commit-set size")
    figure.tight_layout()
    path = output_dir / "commit_set_size.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(str(path))

    labels = [f"K={row['candidate_k']}" for row in agreement_rows_]
    values = [_finite_float(row.get("token_position_agreement_prompt_macro_estimate")) or 0.0 for row in agreement_rows_]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(labels, values, color="#059669")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("prompt-macro agreement with native Fast tau=.8")
    axis.set_title("Final generated-token agreement")
    figure.tight_layout()
    path = output_dir / "final_output_agreement.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(str(path))
    return paths


def write_report(
    output_root: Path,
    *,
    metadata: Mapping[str, Any],
    throughput_rows: Sequence[Mapping[str, Any]],
    batch_rows: Sequence[Mapping[str, Any]],
    agreement_rows_: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the direct experimental answer with K strata kept separate."""

    native_baseline = _summary_row(throughput_rows, NATIVE_FAST_POLICY, None)
    matched_baseline = _summary_row(throughput_rows, FAST_POLICY, None)
    batch_by_key = {
        (str(row.get("policy")), row.get("candidate_k")): row for row in batch_rows
    }
    agreement_by_key = {
        (str(row.get("baseline_policy")), int(row["candidate_k"])): row
        for row in agreement_rows_
    }
    direct: dict[str, Any] = {
        "selected_t0_prompt_seeds": metadata.get("validated_t0_seed_count"),
        "baseline_threshold": config["fast_dllm_baseline"]["threshold"],
        "native_fast_dllm_throughput_tokens_per_second": (
            None
            if native_baseline is None
            else native_baseline.get("aggregate_verification_aware_throughput_tokens_per_second")
        ),
        "matched_fast_dllm_throughput_tokens_per_second": (
            None
            if matched_baseline is None
            else matched_baseline.get("aggregate_verification_aware_throughput_tokens_per_second")
        ),
        "by_candidate_k": {},
    }
    lines = [
        "# Experiment 3: Exact VCCC oracle checking",
        "",
        "## Direct answers",
        "",
        "1. Selected identical t=0 prompt seeds: "
        + str(metadata.get("validated_t0_seed_count", 0))
        + " (the source trajectory is used only for prompt/provenance selection).",
        "2. Native Fast-dLLM threshold=0.8 throughput (historical direct use_cache=True call, no past-key reuse): "
        + _fmt_value(
            None
            if native_baseline is None
            else native_baseline.get("aggregate_verification_aware_throughput_tokens_per_second")
        )
        + " generated positions/s; mean commits/action step="
        + _fmt_value(
            None if native_baseline is None else native_baseline.get("commits_per_step_prompt_macro_estimate")
        )
        + ".",
        "3. The separately reported matched Fast-dLLM control uses use_cache=False only for output agreement with the exact oracle: throughput="
        + _fmt_value(
            None
            if matched_baseline is None
            else matched_baseline.get("aggregate_verification_aware_throughput_tokens_per_second")
        )
        + " generated positions/s. It is not the native Fast-dLLM throughput claim.",
        "4. Conservative logical-work bound (if every action commits only one generated position): "
        + str(metadata.get("worst_case_work_bounds", {}).get("maximum_all_policy_model_forwards"))
        + " model forwards and "
        + str(metadata.get("worst_case_work_bounds", {}).get("maximum_scalar_margin_queries"))
        + " scalar certificate queries across this cohort. Actual work can be lower when a policy commits multiple positions.",
    ]
    for answer_index, candidate_k in enumerate(
        (int(value) for value in config["exact_vccc_oracle"]["candidate_sizes"]), start=5
    ):
        throughput = _summary_row(throughput_rows, EXACT_POLICY, candidate_k)
        batch = batch_by_key.get((EXACT_POLICY, candidate_k))
        native_agreement = agreement_by_key.get((NATIVE_FAST_POLICY, candidate_k))
        matched_agreement = agreement_by_key.get((FAST_POLICY, candidate_k))
        direct["by_candidate_k"][str(candidate_k)] = {
            "verification_aware_throughput_tokens_per_second": (
                None if throughput is None else throughput.get("aggregate_verification_aware_throughput_tokens_per_second")
            ),
            "mean_actual_commit_set_size": (
                None if batch is None else batch.get("actual_commit_set_size_mean")
            ),
            "full_topk_certificate_pass_rate": (
                None if batch is None else batch.get("full_topk_certificate_pass_rate")
            ),
            "final_sequence_exact_match_prompt_macro": (
                None
                if native_agreement is None
                else native_agreement.get("final_sequence_exact_match_prompt_macro_estimate")
            ),
            "token_position_agreement_prompt_macro": (
                None
                if native_agreement is None
                else native_agreement.get("token_position_agreement_prompt_macro_estimate")
            ),
            "agreement_eligible_prompt_count": (
                None if native_agreement is None else native_agreement.get("agreement_eligible_prompt_count")
            ),
            "matched_control_token_position_agreement_prompt_macro": (
                None
                if matched_agreement is None
                else matched_agreement.get("token_position_agreement_prompt_macro_estimate")
            ),
        }
        lines.append(
            f"{answer_index}. Exact VCCC K={candidate_k}: verification-aware throughput="
            + _fmt_value(
                None if throughput is None else throughput.get("aggregate_verification_aware_throughput_tokens_per_second")
            )
            + " positions/s; mean actual commit set="
            + _fmt_value(None if batch is None else batch.get("actual_commit_set_size_mean"))
            + "; full top-K certificate pass="
            + _fmt_rate(None if batch is None else batch.get("full_topk_certificate_pass_rate"))
            + "; final-sequence match with native Fast tau=.8="
            + _fmt_rate(
                None
                if native_agreement is None
                else native_agreement.get("final_sequence_exact_match_prompt_macro_estimate")
            )
            + "; generated-token agreement with native Fast="
            + _fmt_rate(
                None
                if native_agreement is None
                else native_agreement.get("token_position_agreement_prompt_macro_estimate")
            )
            + "; completed-prompt agreement denominator="
            + str(
                None
                if native_agreement is None
                else native_agreement.get("agreement_eligible_prompt_count")
            )
            + "; matched no-cache control token agreement="
            + _fmt_rate(
                None
                if matched_agreement is None
                else matched_agreement.get("token_position_agreement_prompt_macro_estimate")
            )
            + "."
        )
    lines.extend([
        "",
        "## Protocol",
        "",
        "At every Exact-VCCC state, only still-masked fixed generation slots are ranked by the fresh full-vocabulary FP32 probability margin p1-p2. P_K is the top min(K, remaining masks) positions, tied by physical position. Each candidate keeps its fresh deterministic current top-1 token.",
        "",
        "For every A subset P_K, the runner performs an independent fixed-position no-cache full-vocabulary forward. A subset S passes only when every i in S retains its fixed top-1 under every A subset S minus {i}; gamma=0 is inclusive, but deterministic argmax must still match. The actual action is C_K, the maximum-cardinality passing S; ties use summed p1-p2 then sorted physical positions. Thus top-K constrains the exponential search but is not blindly committed.",
        "",
        "Fast-dLLM uses the normal p1>=0.8 action on its own trajectory and commits the first highest-confidence masked generation position only when no position reaches threshold. Its native throughput uses the historical direct `model(x, use_cache=True)` convention without past-key reuse. A separate matched Fast rollout uses fixed-position no-cache exact forwards so the final outputs can be compared fairly with Exact VCCC. Exact-VCCC throughput is committed generated positions divided by CUDA-synchronized rollout seconds and includes every subset verification forward; it is therefore an offline verification-aware measurement, not a deployable speed claim.",
        "",
        "Final-output agreement compares terminal generated token IDs from the same prompt seed between Exact VCCC and native Fast-dLLM; both full-sequence exact match and position-wise Hamming agreement are reported. The matched no-cache Fast-control agreement is retained separately as a forward-convention diagnostic. Incomplete rollouts are explicitly retained in raw provenance but excluded from the primary agreement denominator. Intermediate action overlap is intentionally not interpreted after trajectories diverge.",
        "",
        "## Artifacts",
        "",
        "- raw/prompt_selection.jsonl: deterministic source prompt/t=0 seed screening.",
        "- raw/policy_steps.jsonl: every actual Fast or Exact VCCC action, including candidate margins and committed set size.",
        "- raw/exact_subset_contexts.jsonl and raw/exact_margin_queries.jsonl: every exact top-K subset context and scalar all-order query, appended after each prompt for interruption resilience.",
        "- tables/throughput_summary.csv, tables/commit_batch_summary.csv, and tables/final_output_agreement.csv: requested throughput, actual batch sizes, and same-prompt final-output comparison.",
        "",
        "All confidence intervals are prompt-clustered. K=2, K=4, and K=8 are separate strata and are never pooled. `progress.json` records durable prompt-level progress while the audit is running.",
    ])
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return direct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--smoke", action="store_true", help="Run one deterministic prompt seed.")
    return parser.parse_args()


def make_output_root(base: Path, run_id: str | None, *, prefix: str) -> Path:
    token = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = base / f"{prefix}{token}"
    root.mkdir(parents=True, exist_ok=False)
    for child in ("raw", "tables", "figures"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def worst_case_work_bounds(
    generation_length: int, candidate_sizes: Sequence[int], prompt_count: int
) -> dict[str, Any]:
    """Logical bound when each policy action commits only one position.

    This is deliberately a work-count bound, not a wall-clock prediction:
    model batch size, sequence length, and adaptive OOM splitting determine
    elapsed time. It makes the exponential K=8 cost visible before a full run.
    """

    per_k: dict[str, dict[str, int]] = {}
    for requested_k in candidate_sizes:
        contexts = 0
        queries = 0
        for remaining in range(int(generation_length), 0, -1):
            effective_k = min(int(requested_k), remaining)
            contexts += 1 << effective_k  # base + every nonempty subset
            queries += effective_k * (1 << (effective_k - 1))
        per_k[str(int(requested_k))] = {
            "maximum_exact_forwards_per_prompt": contexts,
            "maximum_scalar_margin_queries_per_prompt": queries,
        }
    fast_actions_per_prompt = int(generation_length)
    exact_forwards_per_prompt = sum(
        row["maximum_exact_forwards_per_prompt"] for row in per_k.values()
    )
    return {
        "assumption": "one actual commit per action step until all generated positions are revealed",
        "prompt_count": int(prompt_count),
        "native_fast_action_forwards_per_prompt": fast_actions_per_prompt,
        "matched_fast_action_forwards_per_prompt": fast_actions_per_prompt,
        "exact_by_candidate_k": per_k,
        "maximum_exact_forwards_per_prompt": exact_forwards_per_prompt,
        "maximum_all_policy_model_forwards_per_prompt": exact_forwards_per_prompt + 2 * fast_actions_per_prompt,
        "maximum_all_policy_model_forwards": (
            exact_forwards_per_prompt + 2 * fast_actions_per_prompt
        ) * int(prompt_count),
        "maximum_scalar_margin_queries": sum(
            row["maximum_scalar_margin_queries_per_prompt"] for row in per_k.values()
        ) * int(prompt_count),
    }


def main() -> None:
    args = parse_args()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    runner_started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Exact VCCC oracle configuration must be a mapping.")
    candidate_sizes = tuple(int(value) for value in config["exact_vccc_oracle"]["candidate_sizes"])
    if candidate_sizes != tuple(sorted(set(candidate_sizes))) or any(value < 1 for value in candidate_sizes):
        raise ValueError("exact_vccc_oracle.candidate_sizes must be ascending, unique positive K values.")
    if int(config["decoding"]["max_steps"]) < int(config["decoding"]["generation_length"]):
        raise ValueError("decoding.max_steps must allow every generation position to make progress.")
    probe_root = args.probe_root.resolve()
    source_root = args.source_run.resolve()
    source_manifest_path = source_root / "run_manifest.json"
    trajectories_path = source_root / "raw" / "trajectories.jsonl"
    if not source_manifest_path.exists() or not trajectories_path.exists():
        raise FileNotFoundError("Source run must contain run_manifest.json and raw/trajectories.jsonl.")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, Mapping):
        raise ValueError("Source run_manifest.json must be an object.")
    _check_source_manifest(source_manifest, config)
    output_base = Path(args.output_root) if args.output_root else probe_root / str(config["storage"]["output_root"])
    output_root = make_output_root(output_base, args.run_id, prefix=str(config["storage"]["run_prefix"]))
    shutil.copy2(args.config, output_root / "config.yaml")
    write_json(output_root / "source_manifest.json", source_manifest)

    trajectories = read_jsonl(trajectories_path)
    seeds, screening_rows, source_selection_counts = select_prompt_seeds(
        trajectories, config, smoke=bool(args.smoke)
    )
    if not seeds:
        write_jsonl(output_root / "raw" / "prompt_selection.jsonl", screening_rows)
        raise RuntimeError("No deterministic primary t=0 prompt seeds were available.")

    from scripts.collect_states import mask_token_id
    from scripts.run_top1_dynamics_audit import _model_snapshot_metadata, load_model
    import torch

    model, tokenizer, _dtype = load_model(config, probe_root)
    source_snapshot = source_manifest["model_snapshot"]
    current_snapshot = _model_snapshot_metadata(model, tokenizer, config)
    source_hint = source_snapshot.get("resolved_hf_commit_hint")
    current_hint = current_snapshot.get("resolved_hf_commit_hint")
    if source_hint and current_hint and str(source_hint) != str(current_hint):
        raise RuntimeError(
            "Current Hugging Face snapshot differs from source t=0 seeds: "
            f"source={source_hint!r}, current={current_hint!r}"
        )
    snapshot_verification = {
        "source": source_snapshot,
        "current": current_snapshot,
        "status": (
            "resolved_commit_match"
            if source_hint and current_hint and str(source_hint) == str(current_hint)
            else "requested_revision_match_resolved_commit_hint_unavailable"
        ),
    }
    mask_id = mask_token_id(model, tokenizer)
    validated_seeds: list[PromptSeed] = []
    for seed in seeds:
        error = _validate_t0_masks(seed, mask_token_id=mask_id)
        if error is not None:
            screening_rows.append({
                "prompt_id": seed.prompt_id,
                "dataset": seed.dataset,
                "example_id": seed.example_id,
                "state_key": seed.source_t0_state_key,
                "selection_stage": "loaded_model_t0_mask_validation",
                "status": "excluded_after_model_load",
                "exclusion_reason": error,
            })
            continue
        validated_seeds.append(seed)
        screening_rows.append({
            "prompt_id": seed.prompt_id,
            "dataset": seed.dataset,
            "example_id": seed.example_id,
            "state_key": seed.source_t0_state_key,
            "selection_stage": "loaded_model_t0_mask_validation",
            "status": "retained_t0_seed",
            "exclusion_reason": None,
        })
    if not validated_seeds:
        write_jsonl(output_root / "raw" / "prompt_selection.jsonl", screening_rows)
        raise RuntimeError("No selected t=0 seed retained the fixed generation-mask invariant.")
    work_bounds = worst_case_work_bounds(
        int(config["decoding"]["generation_length"]), candidate_sizes, len(validated_seeds)
    )

    # Exclude each forward convention's first lazy allocation/compile cost
    # from policy timing. Both are read-only t=0 calls and are retained in
    # provenance rather than silently folded into a throughput result.
    warmup_input = torch.tensor(
        [validated_seeds[0].token_sequence],
        device=next(model.parameters()).device,
        dtype=torch.long,
    )
    _sync_cuda(torch)
    exact_warmup_started = time.perf_counter()
    warmup_logits = exact_logits_batched(model, warmup_input)
    _sync_cuda(torch)
    exact_warmup_seconds = time.perf_counter() - exact_warmup_started
    warmup_assignments = _base_assignments(
        warmup_logits,
        validated_seeds[0].generation_positions,
        tie_tolerance=float(config["exact_vccc_oracle"]["tie_tolerance"]),
    )
    warmup_candidates = select_top_probability_margin_positions(
        warmup_assignments,
        min(max(candidate_sizes), len(warmup_assignments)),
    )
    reducer_validation = validate_vectorized_margin_reducer(
        warmup_logits,
        warmup_candidates,
        tuple(int(warmup_assignments[position]["token_id"]) for position in warmup_candidates),
        tie_tolerance=float(config["exact_vccc_oracle"]["tie_tolerance"]),
    )
    del warmup_logits
    _sync_cuda(torch)
    native_warmup_started = time.perf_counter()
    warmup_logits = _native_fast_logits(model, warmup_input)
    _sync_cuda(torch)
    native_warmup_seconds = time.perf_counter() - native_warmup_started
    del warmup_logits, warmup_input
    if bool(torch.cuda.is_available()):
        torch.cuda.reset_peak_memory_stats()

    # The large raw certificate artifacts are committed prompt-by-prompt. A
    # pod/session interruption therefore leaves inspectable evidence and a
    # precise progress record instead of a single end-of-run serialization.
    context_path = output_root / "raw" / "exact_subset_contexts.jsonl"
    query_path = output_root / "raw" / "exact_margin_queries.jsonl"
    policy_steps_path = output_root / "raw" / "policy_steps.jsonl"
    prompt_rollouts_path = output_root / "raw" / "prompt_rollouts.jsonl"
    agreement_path = output_root / "raw" / "final_output_agreement.jsonl"
    for path in (context_path, query_path, policy_steps_path, prompt_rollouts_path, agreement_path):
        append_jsonl(path, ())
    raw_counts = {
        "exact_subset_context_rows": 0,
        "exact_margin_query_rows": 0,
        "policy_step_rows": 0,
        "prompt_rollout_rows": 0,
        "final_output_agreement_rows": 0,
    }
    progress: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": started_at_utc,
        "total_prompts": len(validated_seeds),
        "completed_prompts": 0,
        "worst_case_work_bounds": work_bounds,
        "raw_row_counts": raw_counts,
    }
    write_json_atomic(output_root / "progress.json", progress)
    write_jsonl(output_root / "raw" / "prompt_selection.jsonl", screening_rows)

    all_steps: list[dict[str, Any]] = []
    native_baseline_results: list[RolloutResult] = []
    matched_baseline_results: list[RolloutResult] = []
    oracle_results: list[RolloutResult] = []
    agreements: list[dict[str, Any]] = []
    tie_tolerance = float(config["exact_vccc_oracle"]["tie_tolerance"])
    gamma = float(config["exact_vccc_oracle"]["margin_threshold"])
    for ordinal, seed in enumerate(validated_seeds, 1):
        native_baseline = _fast_rollout(
            model,
            seed,
            mask_token_id=mask_id,
            threshold=float(config["fast_dllm_baseline"]["threshold"]),
            tie_tolerance=tie_tolerance,
            policy=NATIVE_FAST_POLICY,
            forward_convention=NATIVE_FAST_FORWARD_CONVENTION,
            native_use_cache=True,
            validate_source_t0=False,
        )
        native_baseline_results.append(native_baseline)
        all_steps.extend(native_baseline.step_rows)
        matched_baseline = _fast_rollout(
            model,
            seed,
            mask_token_id=mask_id,
            threshold=float(config["fast_dllm_baseline"]["threshold"]),
            tie_tolerance=tie_tolerance,
            policy=FAST_POLICY,
            forward_convention=MATCHED_FAST_FORWARD_CONVENTION,
            native_use_cache=False,
            validate_source_t0=True,
        )
        matched_baseline_results.append(matched_baseline)
        all_steps.extend(matched_baseline.step_rows)
        prompt_oracles: list[RolloutResult] = []
        for candidate_k in candidate_sizes:
            oracle = _exact_oracle_rollout(
                model,
                seed,
                mask_token_id=mask_id,
                candidate_k=candidate_k,
                subset_batch_size=int(config["execution"]["subset_batch_size"]),
                gamma=gamma,
                tie_tolerance=tie_tolerance,
            )
            oracle_results.append(oracle)
            prompt_oracles.append(oracle)
            all_steps.extend(oracle.step_rows)

        # Flush every artifact needed to reconstruct this prompt's candidate
        # pool, selected C_K, and terminal outcomes before marking it done.
        raw_counts["exact_subset_context_rows"] += append_jsonl(
            context_path, (row for oracle in prompt_oracles for row in oracle.context_rows)
        )
        raw_counts["exact_margin_query_rows"] += append_jsonl(
            query_path, (row for oracle in prompt_oracles for row in oracle.query_rows)
        )
        prompt_results = [native_baseline, matched_baseline, *prompt_oracles]
        raw_counts["policy_step_rows"] += append_jsonl(
            policy_steps_path, (row for result in prompt_results for row in result.step_rows)
        )
        prompt_rollout_rows = [result.prompt_row() for result in prompt_results]
        raw_counts["prompt_rollout_rows"] += append_jsonl(prompt_rollouts_path, prompt_rollout_rows)
        prompt_agreements = [
            *agreement_rows([native_baseline], prompt_oracles),
            *agreement_rows([matched_baseline], prompt_oracles),
        ]
        agreements.extend(prompt_agreements)
        raw_counts["final_output_agreement_rows"] += append_jsonl(agreement_path, prompt_agreements)
        for oracle in prompt_oracles:
            # These rows have just been flushed; do not retain every prompt's
            # 2^K evidence in the Python process until the final report.
            oracle.context_rows.clear()
            oracle.query_rows.clear()
        results_so_far = [*native_baseline_results, *matched_baseline_results, *oracle_results]
        progress.update({
            "completed_prompts": ordinal,
            "last_completed_prompt_id": seed.prompt_id,
            "raw_row_counts": dict(raw_counts),
            "model_forward_evaluations_so_far": sum(
                result.accounting.model_forward_evaluations for result in results_so_far
            ),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json_atomic(output_root / "progress.json", progress)
        print(json.dumps({
            "stage": "policy_rollout",
            "prompt": ordinal,
            "total_prompts": len(validated_seeds),
            "prompt_id": seed.prompt_id,
            "native_fast_steps": native_baseline.accounting.policy_action_steps,
            "matched_fast_steps": matched_baseline.accounting.policy_action_steps,
            "model_forwards_so_far": sum(
                result.accounting.model_forward_evaluations for result in results_so_far
            ),
            "raw_row_counts": raw_counts,
        }, ensure_ascii=False))

    all_results = [*native_baseline_results, *matched_baseline_results, *oracle_results]
    prompt_rows = [result.prompt_row() for result in all_results]
    throughput_rows = throughput_summary(prompt_rows, config)
    batch_rows = commit_batch_summary(all_steps)
    agreement_summary_rows = agreement_summary(agreements, config)
    if bool(torch.cuda.is_available()):
        _sync_cuda(torch)
        peak_vram_mib: float | None = float(torch.cuda.max_memory_allocated() / 2**20)
    else:
        peak_vram_mib = None
    runtime_seconds = time.perf_counter() - runner_started
    total_accounting = {
        "action_forwards": sum(result.accounting.action_forwards for result in all_results),
        "subset_context_forwards": sum(
            result.accounting.subset_context_forwards for result in all_results
        ),
        "model_batch_calls": sum(result.accounting.model_batch_calls for result in all_results),
        "failed_model_batch_calls": sum(result.accounting.failed_model_batch_calls for result in all_results),
        "subset_oom_retries": sum(result.accounting.subset_oom_retries for result in all_results),
        "committed_tokens": sum(result.accounting.committed_tokens for result in all_results),
    }
    total_accounting["model_forward_evaluations"] = (
        int(total_accounting["action_forwards"]) + int(total_accounting["subset_context_forwards"])
    )
    exact_results = [
        result for result in all_results if "use_cache_false" in result.forward_convention
    ]
    total_accounting["exact_forward_evaluations"] = sum(
        result.accounting.model_forward_evaluations for result in exact_results
    )
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_smoke" if args.smoke else "completed",
        "audit_name": "exact_top1_vccc_oracle_checking",
        "started_at_utc": started_at_utc,
        "source_run_root": str(source_root),
        "source_run_status": source_manifest.get("status"),
        "source_commit": source_manifest.get("source_git_commit"),
        "frozen_model": config["model"]["name"],
        "model_snapshot_replay_verification": snapshot_verification,
        "source_selection": source_selection_counts,
        "validated_t0_seed_count": len(validated_seeds),
        "worst_case_work_bounds": work_bounds,
        "t0_exact_replay_validation": {
            "status": "passed_for_every_matched_fast_control",
            "checked_prompt_count": len(matched_baseline_results),
            "comparison": "fresh_fixed_position_use_cache_false_top1_equals_archived_exact_t0_top1_for_every_generation_position",
        },
        "unmeasured_warmup": {
            "exact_no_cache": {
                "model_forward_evaluations": 1,
                "model_batch_calls": 1,
                "wall_seconds": exact_warmup_seconds,
            },
            "native_fast": {
                "model_forward_evaluations": 1,
                "model_batch_calls": 1,
                "wall_seconds": native_warmup_seconds,
            },
            "excluded_from_policy_throughput": True,
        },
        "vectorized_margin_reducer_validation": reducer_validation,
        "fast_dllm_baseline": {
            "threshold": config["fast_dllm_baseline"]["threshold"],
            "fallback_rule": config["fast_dllm_baseline"]["fallback_rule"],
            "native_throughput_forward_convention": NATIVE_FAST_FORWARD_CONVENTION,
            "matched_agreement_forward_convention": MATCHED_FAST_FORWARD_CONVENTION,
        },
        "exact_vccc_oracle": {
            "candidate_sizes": list(candidate_sizes),
            "candidate_rank": config["exact_vccc_oracle"]["candidate_rank"],
            "selected_set_rule": config["exact_vccc_oracle"]["selected_set_rule"],
            "gamma": gamma,
            "tie_policy": "inclusive margin threshold plus deterministic argmax assignment match",
            "forward_convention": EXACT_FORWARD_CONVENTION,
        },
        "forward_accounting": total_accounting,
        "raw_artifact_row_counts": dict(raw_counts),
        "runtime_seconds": runtime_seconds,
        "peak_vram_mib": peak_vram_mib,
    }
    write_jsonl(output_root / "raw" / "prompt_selection.jsonl", screening_rows)
    tables: dict[str, Sequence[Mapping[str, Any]]] = {
        "prompt_selection": screening_rows,
        "policy_steps": all_steps,
        "prompt_rollouts": prompt_rows,
        "throughput_summary": throughput_rows,
        "commit_batch_summary": batch_rows,
        "final_output_agreement": agreements,
        "final_output_agreement_summary": agreement_summary_rows,
    }
    for name, rows in tables.items():
        write_csv(output_root / "tables" / f"{name}.csv", rows)
    figure_paths = write_figures(
        output_root / "figures",
        throughput_rows=throughput_rows,
        batch_rows=batch_rows,
        agreement_rows_=[
            row for row in agreement_summary_rows if row.get("baseline_policy") == NATIVE_FAST_POLICY
        ],
    )
    metadata["figure_paths"] = figure_paths
    direct = write_report(
        output_root,
        metadata=metadata,
        throughput_rows=throughput_rows,
        batch_rows=batch_rows,
        agreement_rows_=agreement_summary_rows,
        config=config,
    )
    metadata["direct_answers"] = direct
    write_json(output_root / "run_metadata.json", metadata)
    write_json(
        output_root / "summary.json",
        {"status": metadata["status"], "metadata": metadata, "direct_answers": direct},
    )
    progress.update({
        "status": metadata["status"],
        "completed_prompts": len(validated_seeds),
        "raw_row_counts": dict(raw_counts),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json_atomic(output_root / "progress.json", progress)
    print(json.dumps({
        "status": metadata["status"],
        "output_root": str(output_root),
        "model_forwards": total_accounting["model_forward_evaluations"],
        "exact_forwards": total_accounting["exact_forward_evaluations"],
        "runtime_seconds": runtime_seconds,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
