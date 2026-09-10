#!/usr/bin/env python3
"""Safe Dependency Headroom / Terminal Token Agreement audit (schema v2).

The runner has no decoder loop, selector replay, oracle, or surrogate.  It
calls only the pinned upstream Fast-dLLM and DAPD functions.  Setup applies
small default-off observer hooks to those exact sources; they copy primitive
values after native selection and cannot modify a decoder decision.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
from importlib import metadata as importlib_metadata
import inspect
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "safe_dependency_headroom/v2"
ACTIVE_BASELINES = {"fast_dllm", "dapd"}
AGREEMENT_LABELS = {0: "0/2 match", 1: "1/2 match", 2: "2/2 match"}
COMMON_OUTPUT_FIELDS = [
    "run_id", "prompt_id", "seed", "baseline", "decode_step",
    "masked_position_count", "selected_positions", "final_output_token_ids",
    "decoded_text", "fast_output_token_ids", "fast_decoded_text",
]
RAW_CSV_SCHEMAS = {
    "prompt_cohort": ["prompt_id", "prompt"],
    "prompt_runs": [
        "run_id", "prompt_id", "seed", "prompt", "prompt_token_ids", "prompt_length",
        "initial_masked_position_count", "initial_state_sha256",
        "fast_final_output_token_ids", "fast_decoded_text",
        "dapd_final_output_token_ids", "dapd_decoded_text",
        "dapd_actual_rejection_events", "dapd_headroom_eligible_rejection_events",
    ],
    "fast_steps": COMMON_OUTPUT_FIELDS + [
        "block_index", "mask_positions", "selected_token_ids",
    ],
    "dapd_steps": COMMON_OUTPUT_FIELDS + [
        "mask_positions", "graph_selected_positions", "direct_selected_positions",
        "staged_added_positions", "edge_threshold", "algorithm",
        "selector_decision_count", "selector_rejection_count",
    ],
    "dapd_decisions": COMMON_OUTPUT_FIELDS + [
        "batch_index", "candidate_position", "candidate_rank", "selected_positions_before",
        "accepted", "decisive_blocker_position", "normalized_dependency_score",
        "raw_attention_dependency_score", "edge_threshold", "combined_selection_score",
        "graph_selected_positions", "direct_selected_positions", "staged_added_positions",
        "selection_stage", "selected_in_graph_selector",
        "selected_in_step_after_algorithm", "eligible_for_headroom",
    ],
    "dapd_events": COMMON_OUTPUT_FIELDS + [
        "event_id", "event_kind", "position_i", "position_j",
        "position_i_generation", "position_j_generation", "attention_dependency_score",
        "raw_attention_dependency_score", "normalized_dependency_score", "edge_threshold",
        "exclusion_reason", "selection_stage", "selected_positions_before",
        "batch_index", "candidate_position", "candidate_rank", "combined_selection_score",
        "selected_in_graph_selector", "selected_in_step_after_algorithm",
        "eligible_for_headroom", "dapd_token_i", "dapd_token_j", "fast_token_i",
        "fast_token_j", "match_i", "match_j", "match_both", "agreement_count",
        "agreement_category",
    ],
    "dapd_pairs": COMMON_OUTPUT_FIELDS + [
        "event_id", "pair_row_index", "pair_role", "position_i", "position_j",
        "position_i_generation", "position_j_generation", "attention_dependency_score",
        "raw_attention_dependency_score", "normalized_dependency_score", "edge_threshold",
        "batch_index", "candidate_position", "candidate_rank", "selected_positions_before",
        "combined_selection_score", "exclusion_reason", "selection_stage",
        "dapd_token_i", "dapd_token_j", "fast_token_i", "fast_token_j", "match_i",
        "match_j", "match_both", "agreement_count", "agreement_category",
    ],
    "outputs": [
        "run_id", "prompt_id", "seed", "baseline", "final_output_token_ids", "decoded_text",
        "fast_output_token_ids", "fast_decoded_text",
    ],
    "equivalence": [
        "run_id", "prompt_id", "seed", "baseline", "observer_disabled_vs_enabled",
        "terminal_output_equal", "native_forward_passes_equal", "native_selection_trace_equal",
        "native_stats_equal", "native_forward_passes", "trace_reconstructs_terminal_output",
    ],
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    rows = list(records)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]], fixed_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    observed = {key for row in rows for key in row}
    fields = list(fixed_fields or [])
    fields.extend(sorted(observed - set(fields)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            cooked = {
                key: json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
            writer.writerow(cooked)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not resolve source commit at {path}: {exc}") from exc


def _under_project(value: str, label: str) -> Path:
    result = (ROOT / value).resolve()
    try:
        result.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay below project root: {value}") from exc
    return result


def _set_seed(seed: int) -> None:
    """Use the same strict deterministic policy for every native decoder call."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _as_int_list(value: Any, field: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"Observer field {field!r} must be a list/tuple, got {type(value).__name__}.")
    return [int(item) for item in value]


def _decode_text(tokenizer: Any, ids: list[int]) -> str:
    try:
        return str(tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    except TypeError:
        return str(tokenizer.decode(ids, skip_special_tokens=False))


def _initial_input(prompt_ids: torch.Tensor, generation_length: int, mask_id: int) -> torch.Tensor:
    return torch.cat(
        [
            prompt_ids.clone(),
            torch.full(
                (prompt_ids.shape[0], generation_length),
                int(mask_id),
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            ),
        ],
        dim=1,
    )


def _validate_config(config: dict[str, Any]) -> None:
    _require(config.get("schema_version") == SCHEMA_VERSION, f"Expected schema_version={SCHEMA_VERSION!r}.")
    for section in ("model", "decoding", "dapd", "analysis", "baselines", "storage"):
        _require(isinstance(config.get(section), dict), f"Missing YAML mapping {section!r}.")
    model, decoding, baselines = config["model"], config["decoding"], config["baselines"]
    for key in ("hf_revision", "fast_dllm_commit", "dapd_commit"):
        _require(
            re.fullmatch(r"[0-9a-fA-F]{7,64}", str(model.get(key, ""))) is not None,
            f"model.{key} must be an immutable Git SHA.",
        )
    _require(float(decoding.get("temperature", float("nan"))) == 0.0, "Only deterministic greedy temperature=0.0 is supported.")
    _require(decoding.get("top_p") is None, "top_p must be null for deterministic greedy decoding.")
    _require(int(decoding["generation_length"]) == int(decoding["block_length"]), "generation_length must equal block_length; multiblock policy is unvalidated.")
    _require(int(decoding["steps"]) == int(decoding["generation_length"]), "steps must equal generation_length for the shared fixed-length policy.")
    _require(decoding.get("use_cache_policy") == "native_source_default", "Runner must preserve native source cache behavior.")
    _require(
        decoding.get("prompt_encoding_policy") == "project_chat_template_then_default_tokenizer_call",
        "prompt_encoding_policy must name the existing project tokenizer path used for every baseline.",
    )
    _require(decoding.get("eos_policy") == "fixed_generation_length_no_early_stop", "All baselines need fixed-length no-early-EOS policy.")
    _require(decoding.get("position_indexing") == "absolute_prompt_plus_generation_offset", "All baselines need the shared absolute position policy.")
    _require(decoding.get("factor") is None, "Fast dynamic factor selection is out of scope; factor must be null/absent.")
    active = baselines.get("active")
    _require(isinstance(active, list) and len(active) == len(set(active)), "baselines.active must be a duplicate-free list.")
    _require(ACTIVE_BASELINES.issubset(set(active)), "baselines.active must include fast_dllm and dapd.")
    _require(set(active).issubset(ACTIVE_BASELINES | {"demask"}), "Unknown active baseline requested.")
    demask = baselines.get("demask")
    _require(
        isinstance(demask, dict) and str(demask.get("status", "")).startswith("blocked"),
        "DEMASK must be explicitly marked blocked until official assets are available.",
    )
    _require(bool(demask.get("reason") or demask.get("required_assets")), "DEMASK blocker must state missing official assets/interface.")
    bins = config["analysis"].get("dependency_bins")
    _require(
        isinstance(bins, list)
        and len(bins) >= 2
        and all(float(left) < float(right) for left, right in zip(bins, bins[1:])),
        "analysis.dependency_bins must be strictly increasing.",
    )
    _require(int(config["analysis"].get("bootstrap_replicates", 0)) > 0, "bootstrap_replicates must be positive.")
    _require(int(config.get("integrity", {}).get("verify_prompt_limit", 1)) > 0, "integrity.verify_prompt_limit must be positive.")
    for key in ("raw_root", "summary_root"):
        _require(bool(config["storage"].get(key)), f"storage.{key} is required.")


def _load_prompts(config: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []

    def add(item: Any) -> None:
        ordinal = len(result)
        if isinstance(item, str):
            prompt, prompt_id = item, f"p{ordinal:05d}"
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompt, prompt_id = item["prompt"], str(item.get("prompt_id", f"p{ordinal:05d}"))
        else:
            raise RuntimeError("Each prompt must be a string or object containing a string prompt field.")
        _require(bool(prompt.strip()), f"Prompt {prompt_id!r} is empty.")
        result.append({"prompt_id": prompt_id, "prompt": prompt})

    for item in config.get("prompts", []) or []:
        add(item)
    if config.get("prompt_file"):
        prompt_file = _under_project(str(config["prompt_file"]), "prompt_file")
        _require(prompt_file.is_file(), f"prompt_file unavailable: {prompt_file}")
        expected_hash = config.get("prompt_file_sha256")
        if expected_hash is not None:
            _require(
                re.fullmatch(r"[0-9a-fA-F]{64}", str(expected_hash)) is not None
                and _sha256_file(prompt_file).lower() == str(expected_hash).lower(),
                "prompt_file SHA-256 differs from the reproducible cohort declared in config.",
            )
        for line in prompt_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                add(json.loads(line))
    _require(bool(result), "No prompts configured.")
    ids = [row["prompt_id"] for row in result]
    _require(len(set(ids)) == len(ids), "prompt_id values must be unique for prompt-cluster bootstrap.")
    return result


def _import_observed_decoders(config: dict[str, Any]) -> dict[str, Any]:
    """Import the real pinned modules and reject a vendored DAPD Fast copy."""

    fast_root = ROOT / "vendor" / "Fast-dLLM"
    fast_llada = fast_root / "v1" / "llada"
    fast_file = fast_llada / "generate.py"
    dapd_root = ROOT / "vendor" / "DAPD"
    dapd_core_file = dapd_root / "dapd" / "core.py"
    dapd_file = dapd_root / "dapd" / "generation.py"
    _require(fast_file.is_file() and dapd_core_file.is_file() and dapd_file.is_file(), "Pinned Fast-dLLM/DAPD sources are missing; run scripts/setup.sh.")
    _require(_git_head(fast_root) == config["model"]["fast_dllm_commit"], "Fast-dLLM vendor commit mismatch; rerun setup.")
    _require(_git_head(dapd_root) == config["model"]["dapd_commit"], "DAPD vendor commit mismatch; rerun setup.")

    sys.path.insert(0, str(fast_llada))
    module_name = "safe_dependency_headroom_pinned_fast_generate"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, fast_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned Fast-dLLM source {fast_file}.")
    fast_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = fast_module
    spec.loader.exec_module(fast_module)
    sys.path.insert(0, str(dapd_root))
    from dapd.generation import generate_dapd  # noqa: PLC0415

    fast_generate = fast_module.generate
    _require(
        Path(inspect.getsourcefile(fast_generate) or "").resolve() == fast_file.resolve(),
        "Fast decoder did not resolve to vendor/Fast-dLLM/v1/llada/generate.py.",
    )
    _require(
        Path(inspect.getsourcefile(generate_dapd) or "").resolve() == dapd_file.resolve(),
        "DAPD decoder did not resolve to vendor/DAPD/dapd/generation.py.",
    )
    _require("step_observer" in inspect.signature(fast_generate).parameters, "Fast-dLLM source lacks required default-off step_observer; rerun setup.")
    dapd_parameters = inspect.signature(generate_dapd).parameters
    _require(
        "selection_observer" in dapd_parameters and "step_observer" in dapd_parameters,
        "DAPD source lacks required default-off observation hooks; rerun setup.",
    )
    return {
        "fast_generate": fast_generate,
        "fast_module": fast_module,
        "generate_dapd": generate_dapd,
        "fast_file": fast_file,
        "dapd_file": dapd_file,
        "dapd_core_file": dapd_core_file,
        "fast_root": fast_root,
        "dapd_root": dapd_root,
    }


def _load_frozen_model(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Reuse this project's pinned Fast-dLLM LLaDA loader."""

    sys.path.insert(0, str(ROOT))
    from scripts.run_counterfactual_probe import load_model  # noqa: PLC0415

    return load_model(config, ROOT)


def _installed_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Required installed package is missing: {distribution}") from exc


def _frozen_versions(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            result[name.lower()] = version
    return result


def _verify_runtime_patch_provenance(config: dict[str, Any], api: dict[str, Any]) -> dict[str, Any]:
    """Reject a vendor edit made after setup's exact patched-source check."""

    setup_manifest_path = ROOT / "outputs" / "safe_dependency_headroom_source_manifest.json"
    _require(
        setup_manifest_path.is_file(),
        "Missing setup source manifest; rerun scripts/setup.sh so patched vendor files are hash-verified.",
    )
    try:
        setup_manifest = json.loads(setup_manifest_path.read_text(encoding="utf-8"))
        setup_sources = setup_manifest["sources"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Malformed setup source manifest: {setup_manifest_path}") from exc
    torch_install = setup_manifest.get("torch_install")
    _require(isinstance(torch_install, dict) and isinstance(torch_install.get("requested_version"), str), "Setup source manifest lacks pinned CUDA torch provenance; rerun setup.")
    _require(
        torch.__version__ == torch_install["requested_version"],
        f"Runtime torch {torch.__version__} differs from setup-pinned {torch_install['requested_version']}; rerun setup.",
    )
    freeze_path = ROOT / "outputs" / "dependency_versions.txt"
    _require(freeze_path.is_file(), "Missing dependency_versions.txt from setup; rerun scripts/setup.sh.")
    frozen = _frozen_versions(freeze_path)
    installed_core = {
        "torch": torch.__version__,
        "transformers": _installed_version("transformers"),
    }
    for name, version in installed_core.items():
        _require(frozen.get(name) == version, f"Runtime {name}={version} differs from setup dependency freeze; rerun setup.")

    checks = (
        (
            "fast_dllm",
            config["model"]["fast_dllm_commit"],
            ROOT / "patches" / "fast_dllm_trace_hooks.patch",
            {"v1/llada/generate.py": api["fast_file"]},
        ),
        (
            "dapd",
            config["model"]["dapd_commit"],
            ROOT / "patches" / "dapd_trace_hooks.patch",
            {
                "dapd/core.py": api["dapd_core_file"],
                "dapd/generation.py": api["dapd_file"],
            },
        ),
    )
    actual: dict[str, Any] = {}
    for label, expected_commit, patch_path, source_files in checks:
        source = setup_sources.get(label)
        _require(isinstance(source, dict), f"Setup source manifest lacks {label} provenance.")
        _require(source.get("expected_commit") == expected_commit == source.get("resolved_commit"), f"Setup manifest {label} commit does not match current config pin.")
        _require(source.get("observation_patch_sha256") == _sha256_file(patch_path), f"Setup manifest {label} patch hash differs from tracked patch; rerun setup.")
        expected_file_hashes = source.get("patched_files_sha256")
        _require(isinstance(expected_file_hashes, dict), f"Setup manifest {label} lacks patched file hashes; rerun setup.")
        actual[label] = {}
        for relative_name, current_path in source_files.items():
            expected_hash = expected_file_hashes.get(relative_name)
            current_hash = _sha256_file(current_path)
            _require(
                isinstance(expected_hash, str) and current_hash == expected_hash,
                f"{label} patched file changed after setup verification: {relative_name}; rerun setup from a clean pinned checkout.",
            )
            actual[label][relative_name] = current_hash
    return {
        "setup_manifest": str(setup_manifest_path),
        "setup_manifest_sha256": _sha256_file(setup_manifest_path),
        "dependency_versions_file": str(freeze_path),
        "dependency_versions_sha256": _sha256_file(freeze_path),
        "verified_installed_core_versions": installed_core,
        "verified_patched_file_sha256": actual,
    }


def _run_fast_decoder(
    api: dict[str, Any],
    model: Any,
    prompt_ids: torch.Tensor,
    decoding: dict[str, Any],
    observer: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[torch.Tensor, int]:
    kwargs: dict[str, Any] = {
        "steps": int(decoding["steps"]),
        "gen_length": int(decoding["generation_length"]),
        "block_length": int(decoding["block_length"]),
        "temperature": float(decoding["temperature"]),
        "remasking": str(decoding["remasking"]),
        "mask_id": int(decoding["mask_id"]),
        "threshold": float(decoding["fast_threshold"]),
        "factor": None,
    }
    if observer is not None:
        kwargs["step_observer"] = observer
    return api["fast_generate"](model, prompt_ids.clone(), **kwargs)


def _run_dapd_decoder(
    api: dict[str, Any],
    model: Any,
    prompt_ids: torch.Tensor,
    config: dict[str, Any],
    selection_observer: Callable[[dict[str, Any]], None] | None = None,
    step_observer: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    decoding, dapd = config["decoding"], config["dapd"]
    kwargs: dict[str, Any] = {
        "attention_mask": torch.ones_like(prompt_ids),
        "gen_length": int(decoding["generation_length"]),
        "mask_id": int(decoding["mask_id"]),
        "layer_ratio": float(dapd["layer_ratio"]),
        "tau_min": float(dapd["tau_min"]),
        "tau_max": float(dapd["tau_max"]),
        "alg": str(dapd["algorithm"]),
        "collect_step_history": True,
    }
    if selection_observer is not None:
        kwargs["selection_observer"] = selection_observer
    if step_observer is not None:
        kwargs["step_observer"] = step_observer
    return api["generate_dapd"](model, prompt_ids.clone(), **kwargs)


class _FastTraceCollector:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any]) -> None:
        row = _json_safe(dict(payload))
        row["decode_step"] = int(row["decode_step"])
        row["block_index"] = int(row["block_index"])
        row["masked_position_count"] = int(row["masked_position_count"])
        row["mask_positions"] = _as_int_list(row.get("mask_positions"), "mask_positions")
        row["selected_positions"] = _as_int_list(row.get("selected_positions"), "selected_positions")
        row["selected_token_ids"] = _as_int_list(row.get("selected_token_ids"), "selected_token_ids")
        _require(len(row["selected_positions"]) == len(row["selected_token_ids"]), "Fast observer reported mismatched selected positions/tokens.")
        self.rows.append(row)


class _DapdTraceCollector:
    """Join actual in-selector decisions to the following native DAPD step."""

    def __init__(self, run_id: str, prompt_id: str) -> None:
        self.run_id = run_id
        self.prompt_id = prompt_id
        self.pending: list[dict[str, Any]] = []
        self.step_rows: list[dict[str, Any]] = []
        self.decision_rows: list[dict[str, Any]] = []
        self.event_rows: list[dict[str, Any]] = []
        self.pair_rows: list[dict[str, Any]] = []

    def on_decision(self, payload: dict[str, Any]) -> None:
        row = _json_safe(dict(payload))
        for name in ("batch_index", "candidate_position", "candidate_rank"):
            row[name] = int(row[name])
        row["accepted"] = bool(row["accepted"])
        row["selected_positions_before"] = _as_int_list(row.get("selected_positions_before"), "selected_positions_before")
        if row.get("decisive_blocker_position") is not None:
            row["decisive_blocker_position"] = int(row["decisive_blocker_position"])
        self.pending.append(row)

    def on_step(self, payload: dict[str, Any]) -> None:
        row = _json_safe(dict(payload))
        row["decode_step"] = int(row["decode_step"])
        row["masked_position_count"] = int(row["masked_position_count"])
        for name in (
            "mask_positions",
            "graph_selected_positions",
            "direct_selected_positions",
            "staged_added_positions",
            "selected_positions",
        ):
            row[name] = _as_int_list(row.get(name), name)
        row["edge_threshold"] = float(row["edge_threshold"])
        row["selector_decision_count"] = len(self.pending)
        row["selector_rejection_count"] = sum(not bool(decision["accepted"]) for decision in self.pending)
        graph_selected, final_selected = set(row["graph_selected_positions"]), set(row["selected_positions"])

        for decision in self.pending:
            candidate = int(decision["candidate_position"])
            selected_in_graph = candidate in graph_selected
            selected_in_step = candidate in final_selected
            _require(
                selected_in_graph == bool(decision["accepted"]),
                "DAPD callback does not match native graph selector output.",
            )
            decision_row = {
                **decision,
                "decode_step": row["decode_step"],
                "masked_position_count": row["masked_position_count"],
                "selected_positions": row["selected_positions"],
                "graph_selected_positions": row["graph_selected_positions"],
                "direct_selected_positions": row["direct_selected_positions"],
                "staged_added_positions": row["staged_added_positions"],
                "selection_stage": "greedy_independent_set",
                "selected_in_graph_selector": selected_in_graph,
                "selected_in_step_after_algorithm": selected_in_step,
                # A later step may still commit this position.  Only a same
                # step direct/staged/fallback addition removes this exact
                # co-commit exclusion from the primary headroom denominator.
                "eligible_for_headroom": not selected_in_step,
            }
            self.decision_rows.append(decision_row)
            if bool(decision["accepted"]):
                continue
            blocker, normalized = decision.get("decisive_blocker_position"), decision.get("normalized_dependency_score")
            _require(blocker is not None and normalized is not None, "A rejected DAPD candidate lacks the actual first blocking edge.")
            _require(int(blocker) in decision["selected_positions_before"], "DAPD's decisive blocker is not in its actual selected-before set.")
            _require(float(normalized) > float(decision["edge_threshold"]), "DAPD rejection does not satisfy native dependency > tau.")
            event = {
                **decision_row,
                "event_id": f"{self.run_id}:{self.prompt_id}:dapd:{row['decode_step']}:{candidate}:{int(decision['candidate_rank'])}",
                "event_kind": "candidate_rejection",
                "position_i": candidate,
                "position_j": int(blocker),
                "attention_dependency_score": decision.get("raw_attention_dependency_score"),
                "raw_attention_dependency_score": decision.get("raw_attention_dependency_score"),
                "normalized_dependency_score": float(normalized),
                "edge_threshold": float(decision["edge_threshold"]),
                "exclusion_reason": "first_selected_dependency_gt_tau",
                "selection_stage": "greedy_independent_set",
            }
            self.event_rows.append(event)
            self.pair_rows.append({**event, "pair_row_index": 0, "pair_role": "actual_first_blocking_edge"})
        self.pending.clear()
        self.step_rows.append(row)


def _capture_fast_native_selection_without_step_observer(
    api: dict[str, Any],
    model: Any,
    prompt_ids: torch.Tensor,
    decoding: dict[str, Any],
) -> tuple[torch.Tensor, int, _FastTraceCollector]:
    """Observe the native transfer helper while the source callback is off.

    Fast-dLLM has no step-history return value.  This short-lived wrapper calls
    its *actual* ``get_transfer_index`` once, snapshots its unchanged return,
    returns it byte-for-byte, and restores the module global in ``finally``.
    It does not reproduce either decoder loop or selector; it only lets the
    equivalence control expose the selection already made by upstream code.
    """

    fast_module = api["fast_module"]
    original = fast_module.get_transfer_index
    trace = _FastTraceCollector()
    call_index = 0

    def observing_transfer(
        logits: torch.Tensor,
        temperature: float,
        remasking: str,
        mask_index: torch.Tensor,
        x: torch.Tensor,
        num_transfer_tokens: Any,
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal call_index
        x0, transfer_index = original(
            logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold
        )
        trace({
            "decode_step": call_index,
            "block_index": 0,  # Config validation permits exactly one block.
            "masked_position_count": int(mask_index.sum().item()),
            "mask_positions": tuple(
                int(value) for value in mask_index[0].nonzero(as_tuple=True)[0].detach().cpu().tolist()
            ),
            "selected_positions": tuple(
                int(value) for value in transfer_index[0].nonzero(as_tuple=True)[0].detach().cpu().tolist()
            ),
            "selected_token_ids": tuple(
                int(value) for value in x0[0, transfer_index[0]].detach().cpu().tolist()
            ),
        })
        call_index += 1
        return x0, transfer_index

    fast_module.get_transfer_index = observing_transfer
    try:
        output, nfe = _run_fast_decoder(api, model, prompt_ids, decoding)
    finally:
        fast_module.get_transfer_index = original
    _require(fast_module.get_transfer_index is original, "Fast native transfer helper was not restored after no-observer trace.")
    return output, int(nfe), trace


def _assert_no_masks(output: torch.Tensor, mask_id: int, baseline: str) -> None:
    _require(not bool(output.eq(mask_id).any().item()), f"{baseline} returned unresolved mask tokens.")


def _assert_fast_trace(initial: torch.Tensor, output: torch.Tensor, prompt_length: int, collector: _FastTraceCollector) -> None:
    reconstructed, seen = initial.clone(), set()
    for row in collector.rows:
        _require(row["masked_position_count"] == len(row["mask_positions"]), "Fast mask count differs from traced mask positions.")
        for position, token_id in zip(row["selected_positions"], row["selected_token_ids"]):
            _require(position >= prompt_length, "Fast trace selected a prompt token.")
            _require(position not in seen, "Fast trace selected a generated position twice.")
            reconstructed[0, position] = token_id
            seen.add(position)
    _require(torch.equal(reconstructed, output), "Fast observer trace cannot reconstruct native terminal output.")


def _assert_dapd_trace(
    initial: torch.Tensor,
    output: torch.Tensor,
    prompt_length: int,
    collector: _DapdTraceCollector,
    stats: dict[str, Any],
) -> None:
    _require(not collector.pending, "DAPD emitted decisions without a corresponding step observation.")
    history = stats.get("step_positions")
    _require(isinstance(history, list), "DAPD did not return native step history.")
    expected = [sorted(int(position) for batch, position in step if int(batch) == 0) for step in history]
    observed = [sorted(row["selected_positions"]) for row in collector.step_rows]
    _require(expected == observed, "DAPD observer differs from native step history.")
    reconstructed, seen = initial.clone(), set()
    for row in collector.step_rows:
        _require(row["masked_position_count"] == len(row["mask_positions"]), "DAPD mask count differs from traced mask positions.")
        for position in row["selected_positions"]:
            _require(position >= prompt_length, "DAPD trace selected a prompt token.")
            _require(position not in seen, "DAPD trace selected a generated position twice.")
            reconstructed[0, position] = output[0, position]
            seen.add(position)
    _require(torch.equal(reconstructed, output), "DAPD observer trace cannot reconstruct native terminal output.")


def _dapd_step_positions(stats: dict[str, Any]) -> list[list[int]]:
    history = stats.get("step_positions")
    _require(isinstance(history, list), "DAPD did not return native step history.")
    return [sorted(int(position) for batch, position in step if int(batch) == 0) for step in history]


def _run_equivalence_check(
    api: dict[str, Any],
    model: Any,
    prompt_ids: torch.Tensor,
    config: dict[str, Any],
    prompt_seed: int,
    run_id: str,
    prompt_id: str,
) -> tuple[torch.Tensor, _FastTraceCollector, torch.Tensor, dict[str, Any], _DapdTraceCollector, list[dict[str, Any]]]:
    """Fail closed if adding hooks changes an upstream decoder result."""

    decoding = config["decoding"]
    initial = _initial_input(prompt_ids, int(decoding["generation_length"]), int(decoding["mask_id"]))
    records: list[dict[str, Any]] = []

    _set_seed(prompt_seed)
    fast_pristine, fast_pristine_nfe, fast_native_trace = _capture_fast_native_selection_without_step_observer(
        api, model, prompt_ids, decoding
    )
    _set_seed(prompt_seed)
    fast_trace = _FastTraceCollector()
    fast_observed, fast_observed_nfe = _run_fast_decoder(api, model, prompt_ids, decoding, fast_trace)
    _require(torch.equal(fast_pristine, fast_observed), "Fast-dLLM observer changed terminal output.")
    _require(int(fast_pristine_nfe) == int(fast_observed_nfe), "Fast-dLLM observer changed forward-pass count.")
    _require(_canonical(fast_native_trace.rows) == _canonical(fast_trace.rows), "Fast-dLLM source observer changed native selected positions/tokens.")
    _assert_fast_trace(initial, fast_pristine, int(prompt_ids.shape[1]), fast_native_trace)
    _assert_fast_trace(initial, fast_observed, int(prompt_ids.shape[1]), fast_trace)
    records.append({
        "run_id": run_id,
        "prompt_id": prompt_id,
        "seed": prompt_seed,
        "baseline": "fast_dllm",
        "observer_disabled_vs_enabled": "exact_match",
        "terminal_output_equal": True,
        "native_forward_passes_equal": True,
        "native_selection_trace_equal": True,
        "native_forward_passes": int(fast_observed_nfe),
        "trace_reconstructs_terminal_output": True,
    })

    _set_seed(prompt_seed)
    dapd_pristine, dapd_pristine_stats = _run_dapd_decoder(api, model, prompt_ids, config)
    _set_seed(prompt_seed)
    dapd_trace = _DapdTraceCollector(run_id, prompt_id)
    dapd_observed, dapd_observed_stats = _run_dapd_decoder(
        api, model, prompt_ids, config, dapd_trace.on_decision, dapd_trace.on_step
    )
    _require(torch.equal(dapd_pristine, dapd_observed), "DAPD observer changed terminal output.")
    _require(_canonical(dapd_pristine_stats) == _canonical(dapd_observed_stats), "DAPD observer changed decoder stats.")
    _require(
        _dapd_step_positions(dapd_pristine_stats) == _dapd_step_positions(dapd_observed_stats),
        "DAPD observer changed native selected positions.",
    )
    _assert_dapd_trace(initial, dapd_observed, int(prompt_ids.shape[1]), dapd_trace, dapd_observed_stats)
    records.append({
        "run_id": run_id,
        "prompt_id": prompt_id,
        "seed": prompt_seed,
        "baseline": "dapd",
        "observer_disabled_vs_enabled": "exact_match",
        "terminal_output_equal": True,
        "native_stats_equal": True,
        "native_selection_trace_equal": True,
        "native_forward_passes": int(dapd_observed_stats["total_forward_passes"]),
        "trace_reconstructs_terminal_output": True,
    })
    return fast_observed, fast_trace, dapd_observed, dapd_observed_stats, dapd_trace, records


def _run_observed_prompt(
    api: dict[str, Any],
    model: Any,
    prompt_ids: torch.Tensor,
    config: dict[str, Any],
    prompt_seed: int,
    run_id: str,
    prompt_id: str,
) -> tuple[torch.Tensor, _FastTraceCollector, torch.Tensor, dict[str, Any], _DapdTraceCollector]:
    decoding = config["decoding"]
    _set_seed(prompt_seed)
    fast_trace = _FastTraceCollector()
    fast_output, _ = _run_fast_decoder(api, model, prompt_ids, decoding, fast_trace)
    _set_seed(prompt_seed)
    dapd_trace = _DapdTraceCollector(run_id, prompt_id)
    dapd_output, dapd_stats = _run_dapd_decoder(api, model, prompt_ids, config, dapd_trace.on_decision, dapd_trace.on_step)
    initial = _initial_input(prompt_ids, int(decoding["generation_length"]), int(decoding["mask_id"]))
    _assert_fast_trace(initial, fast_output, int(prompt_ids.shape[1]), fast_trace)
    _assert_dapd_trace(initial, dapd_output, int(prompt_ids.shape[1]), dapd_trace, dapd_stats)
    return fast_output, fast_trace, dapd_output, dapd_stats, dapd_trace


def _output_common(
    run_id: str,
    prompt_id: str,
    seed: int,
    baseline: str,
    final_ids: list[int],
    decoded_text: str,
    fast_ids: list[int],
    fast_text: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "seed": seed,
        "baseline": baseline,
        "final_output_token_ids": final_ids,
        "decoded_text": decoded_text,
        "fast_output_token_ids": fast_ids,
        "fast_decoded_text": fast_text,
    }


def _enrich_dapd_events(
    rows: list[dict[str, Any]],
    common: dict[str, Any],
    prompt_length: int,
    dapd_ids: list[int],
    fast_ids: list[int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        record = {**common, **row, "baseline": "dapd"}
        i, j = int(record["position_i"]) - prompt_length, int(record["position_j"]) - prompt_length
        _require(0 <= i < len(dapd_ids) and 0 <= j < len(dapd_ids), "DAPD blocker pair is outside generation indexing.")
        match_i, match_j = bool(dapd_ids[i] == fast_ids[i]), bool(dapd_ids[j] == fast_ids[j])
        agreement_count = int(match_i) + int(match_j)
        record.update({
            "position_i_generation": i,
            "position_j_generation": j,
            "dapd_token_i": int(dapd_ids[i]),
            "dapd_token_j": int(dapd_ids[j]),
            "fast_token_i": int(fast_ids[i]),
            "fast_token_j": int(fast_ids[j]),
            "match_i": match_i,
            "match_j": match_j,
            "match_both": bool(match_i and match_j),
            "agreement_count": agreement_count,
            "agreement_category": AGREEMENT_LABELS[agreement_count],
        })
        result.append(record)
    return result


def _rates(events: list[dict[str, Any]]) -> dict[str, float | None]:
    if not events:
        return {AGREEMENT_LABELS[count]: None for count in (0, 1, 2)}
    counts = Counter(int(event["agreement_count"]) for event in events)
    return {AGREEMENT_LABELS[count]: float(counts[count] / len(events)) for count in (0, 1, 2)}


def _bootstrap_prompt_clusters(
    events: list[dict[str, Any]],
    prompt_ids: list[str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Resample every input prompt, including the zero-event clusters."""

    groups: dict[str, list[int]] = {prompt_id: [] for prompt_id in prompt_ids}
    for event in events:
        groups[str(event["prompt_id"])].append(int(event["agreement_count"]))
    rng, indexes = np.random.default_rng(seed), np.arange(len(prompt_ids))
    draws = {0: [], 1: [], 2: []}
    undefined = 0
    for _ in range(replicates):
        chosen = rng.choice(indexes, size=len(indexes), replace=True)
        values: list[int] = []
        for index in chosen:
            values.extend(groups[prompt_ids[int(index)]])
        if not values:
            undefined += 1
            continue
        values_array = np.asarray(values)
        for category in (0, 1, 2):
            draws[category].append(float(np.mean(values_array == category)))
    return {
        "unit": "prompt_cluster",
        "replicates_requested": int(replicates),
        "valid_replicates": int(replicates - undefined),
        "undefined_zero_event_replicates": int(undefined),
        "95_ci": {
            AGREEMENT_LABELS[category]: [float(item) for item in np.quantile(draws[category], [0.025, 0.975])]
            if draws[category]
            else None
            for category in (0, 1, 2)
        },
    }


def _summarize_dapd(events: list[dict[str, Any]], prompt_ids: list[str], config: dict[str, Any]) -> dict[str, Any]:
    eligible = [event for event in events if bool(event["eligible_for_headroom"])]
    unique_all = {(event["prompt_id"], event["position_i_generation"], event["position_j_generation"]) for event in events}
    unique_eligible = {(event["prompt_id"], event["position_i_generation"], event["position_j_generation"]) for event in eligible}
    first_pair: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in sorted(eligible, key=lambda item: (str(item["prompt_id"]), int(item["decode_step"]), int(item["candidate_rank"]))):
        first_pair.setdefault((event["prompt_id"], event["position_i_generation"], event["position_j_generation"]), event)
    primary_rates, all_rates = _rates(eligible), _rates(events)
    prompts_with_events = {event["prompt_id"] for event in events}
    prompts_with_eligible = {event["prompt_id"] for event in eligible}
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": "dapd",
        "status": "completed",
        "primary_metric": "event_weighted_terminal_agreement_for_same_step_headroom_eligible_actual_first_blocker_edges",
        "input_prompt_count": len(prompt_ids),
        "unique_prompts": len(prompt_ids),
        "prompts_with_rejection_events": len(prompts_with_events),
        "prompts_with_headroom_eligible_events": len(prompts_with_eligible),
        "zero_rejection_prompt_count": len(prompt_ids) - len(prompts_with_events),
        "rejection_events": len(events),
        "headroom_eligible_rejection_events": len(eligible),
        "same_step_selected_after_algorithm_events": len(events) - len(eligible),
        "unique_position_pairs": len(unique_all),
        "unique_headroom_eligible_position_pairs": len(unique_eligible),
        "agreement_rates": primary_rates,
        "agreement_0_rate": primary_rates["0/2 match"],
        "agreement_1_rate": primary_rates["1/2 match"],
        "agreement_2_rate": primary_rates["2/2 match"],
        "all_actual_rejection_event_agreement_rates": all_rates,
        "first_occurrence_unique_pair_agreement_rates": _rates(list(first_pair.values())),
        "prompt_bootstrap_95_ci": _bootstrap_prompt_clusters(
            eligible,
            prompt_ids,
            int(config["analysis"]["bootstrap_replicates"]),
            int(config["analysis"]["bootstrap_seed"]),
        ),
    }


def _plot_dapd_bins(events: list[dict[str, Any]], bins: list[float], png_path: Path, csv_path: Path) -> dict[str, Any]:
    eligible = [
        event
        for event in events
        if bool(event["eligible_for_headroom"]) and event.get("normalized_dependency_score") is not None
    ]
    rows: list[dict[str, Any]] = []
    for index, (low, high) in enumerate(zip(bins, bins[1:])):
        last = index == len(bins) - 2
        members = [
            event
            for event in eligible
            if float(event["normalized_dependency_score"]) >= low
            and (float(event["normalized_dependency_score"]) <= high if last else float(event["normalized_dependency_score"]) < high)
        ]
        rates = _rates(members)
        rows.append({
            "bin_index": index,
            "bin_low": low,
            "bin_high": high,
            "bin_label": f"[{low:g}, {high:g}{']' if last else ')'}",
            "event_count": len(members),
            "agreement_0_rate": rates["0/2 match"],
            "agreement_1_rate": rates["1/2 match"],
            "agreement_2_rate": rates["2/2 match"],
        })
    _write_csv(csv_path, rows)
    if not eligible:
        return {"status": "not_generated_no_headroom_eligible_dapd_events", "path": None, "bin_table": str(csv_path)}
    import matplotlib.pyplot as plt  # noqa: PLC0415

    zero = np.asarray([row["agreement_0_rate"] or 0.0 for row in rows])
    one = np.asarray([row["agreement_1_rate"] or 0.0 for row in rows])
    two = np.asarray([row["agreement_2_rate"] or 0.0 for row in rows])
    figure, axis = plt.subplots(figsize=(9, 4.8))
    labels = [row["bin_label"] for row in rows]
    axis.bar(labels, zero, label="0/2 match", color="#c44e52")
    axis.bar(labels, one, bottom=zero, label="1/2 match", color="#dd8452")
    axis.bar(labels, two, bottom=zero + one, label="2/2 match", color="#55a868")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Event-weighted terminal agreement")
    axis.set_xlabel("DAPD normalized dependency score (actual first blocking edge)")
    axis.legend(loc="upper right")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return {"status": "generated", "path": str(png_path), "bin_table": str(csv_path)}


def _demask_blocker(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    demask = config["baselines"]["demask"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "baseline": "demask",
        "status": str(demask["status"]),
        "reason": demask.get("reason") or "Official DEMASK predictor/checkpoint/interface is not present in this checkout.",
        "required_assets": demask.get("required_assets", []),
        "required_interface": demask.get("required_interface"),
        "surrogate_used": False,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _source_manifest(
    config: dict[str, Any],
    api: dict[str, Any],
    patch_provenance: dict[str, Any],
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, str]],
    run_id: str,
) -> dict[str, Any]:
    requested = str(config["model"]["hf_revision"])
    model_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") if hasattr(tokenizer, "init_kwargs") else None
    for name, commit in (("model", model_commit), ("tokenizer", tokenizer_commit)):
        if commit is not None:
            _require(str(commit).lower().startswith(requested.lower()), f"Resolved {name} revision {commit} disagrees with requested immutable revision {requested}.")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_sha256": _sha256_text(_canonical(config)),
        "prompt_set_sha256": _sha256_text(_canonical(prompts)),
        "model": {
            "name": config["model"]["name"],
            "requested_immutable_revision": requested,
            "resolved_model_commit_hash": model_commit,
            "resolved_tokenizer_commit_hash": tokenizer_commit,
            "dtype": config["model"]["dtype"],
        },
        "sources": {
            "fast_dllm": {
                "expected_commit": config["model"]["fast_dllm_commit"],
                "resolved_commit": _git_head(api["fast_root"]),
                "imported_file": str(api["fast_file"]),
                "patch_sha256": _sha256_file(ROOT / "patches" / "fast_dllm_trace_hooks.patch"),
            },
            "dapd": {
                "expected_commit": config["model"]["dapd_commit"],
                "resolved_commit": _git_head(api["dapd_root"]),
                "imported_file": str(api["dapd_file"]),
                "patch_sha256": _sha256_file(ROOT / "patches" / "dapd_trace_hooks.patch"),
            },
        },
        "runtime_patch_verification": patch_provenance,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "hf_home": os.environ.get("HF_HOME"),
        },
        "decoding": config["decoding"],
        "dapd": config["dapd"],
        "demask": config["baselines"]["demask"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    _require(config_path.is_file(), f"Config does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "Config must be a YAML mapping.")
    _validate_config(config)
    prompts = _load_prompts(config)
    prompt_ids = [prompt["prompt_id"] for prompt in prompts]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{config.get('run', {}).get('id_prefix', 'sdh-v2')}-{stamp}-{uuid.uuid4().hex[:8]}"
    _require(re.fullmatch(r"[A-Za-z0-9._-]+", run_id) is not None, "run_id has unsupported characters.")
    raw_dir = _under_project(str(config["storage"]["raw_root"]), "storage.raw_root") / run_id
    summary_dir = _under_project(str(config["storage"]["summary_root"]), "storage.summary_root") / run_id
    _require(not raw_dir.exists() and not summary_dir.exists(), f"Refusing to overwrite existing run {run_id}.")
    raw_dir.mkdir(parents=True, exist_ok=False)
    summary_dir.mkdir(parents=True, exist_ok=False)
    _write_json(summary_dir / "config.resolved.json", config)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "initialized",
        "raw_dir": str(raw_dir),
        "summary_dir": str(summary_dir),
        "config_path": str(config_path),
    }
    _write_json(summary_dir / "run_manifest.json", manifest)
    demask_blocker = _demask_blocker(config, run_id)
    _write_json(summary_dir / "demask_blocker.json", demask_blocker)
    prompt_cohort_path = raw_dir / "prompt_cohort.jsonl"
    _append_jsonl(prompt_cohort_path, prompts)

    try:
        if "demask" in config["baselines"]["active"]:
            raise RuntimeError(
                "DEMASK was explicitly activated, but this checkout has no official compatible predictor/checkpoint/interface; "
                f"see {summary_dir / 'demask_blocker.json'}. No surrogate will be used."
            )
        hf_home = ROOT / "cache" / "huggingface"
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        _require(torch.cuda.is_available(), "CUDA unavailable: this frozen 8B experiment cannot fall back to CPU or quantization.")
        _set_seed(int(config["decoding"]["seed"]))
        api = _import_observed_decoders(config)
        patch_provenance = _verify_runtime_patch_provenance(config, api)
        model, tokenizer, _ = _load_frozen_model(config)
        from scripts.collect_states import mask_token_id, tokenized_prompt  # noqa: PLC0415

        _write_json(
            summary_dir / "source_manifest.json",
            _source_manifest(config, api, patch_provenance, model, tokenizer, prompts, run_id),
        )
        manifest.update({"status": "running", "source_manifest": str(summary_dir / "source_manifest.json")})
        _write_json(summary_dir / "run_manifest.json", manifest)
        raw_files = {
            "prompt_runs": raw_dir / "prompt_runs.jsonl",
            "fast_steps": raw_dir / "fast_dllm_steps.jsonl",
            "dapd_steps": raw_dir / "dapd_steps.jsonl",
            "dapd_decisions": raw_dir / "dapd_selector_decisions.jsonl",
            "dapd_events": raw_dir / "dapd_rejection_events.jsonl",
            "dapd_pairs": raw_dir / "dapd_rejection_pairs.jsonl",
            "outputs": raw_dir / "final_outputs.jsonl",
            "equivalence": raw_dir / "instrumentation_equivalence.jsonl",
        }
        # Materialize empty JSONL/CSV artifacts too. A valid zero-rejection
        # run must remain machine-readable rather than omitting its schemas.
        for path in raw_files.values():
            path.touch(exist_ok=False)
        all_events: list[dict[str, Any]] = []
        all_equivalence: list[dict[str, Any]] = []
        integrity = config.get("integrity", {})
        verify_limit = min(int(integrity.get("verify_prompt_limit", len(prompts))), len(prompts))
        base_seed, mask_id = int(config["decoding"]["seed"]), int(config["decoding"]["mask_id"])
        _require(int(mask_token_id(model, tokenizer)) == mask_id, "Configured mask_id disagrees with loaded tokenizer/model.")

        for index, prompt_record in enumerate(prompts):
            prompt_id, prompt = prompt_record["prompt_id"], prompt_record["prompt"]
            prompt_seed = base_seed + index
            prompt_tensor = tokenized_prompt(tokenizer, prompt, next(model.parameters()).device)
            _require(prompt_tensor.shape[0] == 1, "This audit requires batch size one.")
            initial = _initial_input(prompt_tensor, int(config["decoding"]["generation_length"]), mask_id)
            start = time.perf_counter()
            if index < verify_limit:
                fast_output, fast_trace, dapd_output, dapd_stats, dapd_trace, equivalence = _run_equivalence_check(
                    api, model, prompt_tensor, config, prompt_seed, run_id, prompt_id
                )
                _append_jsonl(raw_files["equivalence"], equivalence)
                all_equivalence.extend(equivalence)
            else:
                fast_output, fast_trace, dapd_output, dapd_stats, dapd_trace = _run_observed_prompt(
                    api, model, prompt_tensor, config, prompt_seed, run_id, prompt_id
                )
            _assert_no_masks(fast_output, mask_id, "Fast-dLLM")
            _assert_no_masks(dapd_output, mask_id, "DAPD")
            prompt_length = int(prompt_tensor.shape[1])
            fast_ids = [int(value) for value in fast_output[0, prompt_length:].detach().cpu().tolist()]
            dapd_ids = [int(value) for value in dapd_output[0, prompt_length:].detach().cpu().tolist()]
            _require(len(fast_ids) == len(dapd_ids) == int(config["decoding"]["generation_length"]), "Baselines did not emit common fixed generation length.")
            fast_text, dapd_text = _decode_text(tokenizer, fast_ids), _decode_text(tokenizer, dapd_ids)
            fast_common = _output_common(run_id, prompt_id, prompt_seed, "fast_dllm", fast_ids, fast_text, fast_ids, fast_text)
            dapd_common = _output_common(run_id, prompt_id, prompt_seed, "dapd", dapd_ids, dapd_text, fast_ids, fast_text)
            fast_steps = [{**fast_common, **row, "baseline": "fast_dllm"} for row in fast_trace.rows]
            dapd_steps = [{**dapd_common, **row, "baseline": "dapd"} for row in dapd_trace.step_rows]
            dapd_decisions = [{**dapd_common, **row, "baseline": "dapd"} for row in dapd_trace.decision_rows]
            dapd_events = _enrich_dapd_events(dapd_trace.event_rows, dapd_common, prompt_length, dapd_ids, fast_ids)
            dapd_pairs = _enrich_dapd_events(dapd_trace.pair_rows, dapd_common, prompt_length, dapd_ids, fast_ids)
            prompt_run = {
                "run_id": run_id,
                "prompt_id": prompt_id,
                "seed": prompt_seed,
                "prompt": prompt,
                "prompt_token_ids": [int(value) for value in prompt_tensor[0].detach().cpu().tolist()],
                "prompt_length": prompt_length,
                "initial_masked_position_count": int(config["decoding"]["generation_length"]),
                "initial_state_sha256": _sha256_text(_canonical(initial)),
                "fast_final_output_token_ids": fast_ids,
                "fast_decoded_text": fast_text,
                "dapd_final_output_token_ids": dapd_ids,
                "dapd_decoded_text": dapd_text,
                "fast_native_step_count": len(fast_trace.rows),
                "dapd_native_step_count": len(dapd_trace.step_rows),
                "dapd_native_forward_passes": int(dapd_stats["total_forward_passes"]),
                "dapd_actual_rejection_events": len(dapd_events),
                "dapd_headroom_eligible_rejection_events": sum(bool(row["eligible_for_headroom"]) for row in dapd_events),
                "elapsed_seconds_including_optional_equivalence": time.perf_counter() - start,
            }
            # JSONL is append-only and fsync'd after each prompt. CSV is
            # materialized after completion only as a convenience view.
            _append_jsonl(raw_files["prompt_runs"], [prompt_run])
            _append_jsonl(raw_files["fast_steps"], fast_steps)
            _append_jsonl(raw_files["dapd_steps"], dapd_steps)
            _append_jsonl(raw_files["dapd_decisions"], dapd_decisions)
            _append_jsonl(raw_files["dapd_events"], dapd_events)
            _append_jsonl(raw_files["dapd_pairs"], dapd_pairs)
            _append_jsonl(raw_files["outputs"], [fast_common, dapd_common])
            all_events.extend(dapd_events)

        _write_csv(prompt_cohort_path.with_suffix(".csv"), _read_jsonl(prompt_cohort_path), RAW_CSV_SCHEMAS["prompt_cohort"])
        for name, path in raw_files.items():
            _write_csv(path.with_suffix(".csv"), _read_jsonl(path), RAW_CSV_SCHEMAS[name])
        dapd_summary = _summarize_dapd(all_events, prompt_ids, config)
        fast_summary = {
            "schema_version": SCHEMA_VERSION,
            "baseline": "fast_dllm",
            "status": "completed",
            "input_prompt_count": len(prompt_ids),
            "unique_prompts": len(prompt_ids),
            "rejection_events": 0,
            "unique_position_pairs": 0,
            "agreement_rates": None,
            "prompt_bootstrap_95_ci": None,
            "note": "Fast-dLLM is the terminal-token comparator, not a dependency-rejection baseline.",
        }
        demask_summary = {
            "schema_version": SCHEMA_VERSION,
            "baseline": "demask",
            "status": demask_blocker["status"],
            "input_prompt_count": len(prompt_ids),
            "unique_prompts": len(prompt_ids),
            "rejection_events": None,
            "unique_position_pairs": None,
            "agreement_rates": None,
            "prompt_bootstrap_95_ci": None,
            "reason": demask_blocker["reason"],
            "blocker_artifact": str(summary_dir / "demask_blocker.json"),
        }
        summaries = [fast_summary, dapd_summary, demask_summary]
        _write_json(summary_dir / "baseline_summary.json", {"schema_version": SCHEMA_VERSION, "run_id": run_id, "baselines": summaries})
        _write_csv(summary_dir / "baseline_summary.csv", summaries)
        for summary in summaries:
            _write_json(summary_dir / f"{summary['baseline']}_summary.json", summary)
        plot_status = _plot_dapd_bins(
            all_events,
            [float(value) for value in config["analysis"]["dependency_bins"]],
            summary_dir / "dapd_dependency_agreement.png",
            summary_dir / "dapd_dependency_bins.csv",
        )
        _write_json(summary_dir / "demask_dependency_agreement_plot_status.json", {
            "baseline": "demask",
            "status": "not_generated_blocked_missing_official_assets",
            "blocker_artifact": str(summary_dir / "demask_blocker.json"),
        })
        manifest.update({
            "status": "completed",
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_prompt_count": len(prompts),
            "dapd_rejection_events": len(all_events),
            "dapd_headroom_eligible_rejection_events": dapd_summary["headroom_eligible_rejection_events"],
            "instrumentation_equivalence_records": len(all_equivalence),
            "dapd_dependency_plot": plot_status,
        })
        _write_json(summary_dir / "run_manifest.json", manifest)
        print(json.dumps({
            "run_id": run_id,
            "raw_dir": str(raw_dir),
            "summary_dir": str(summary_dir),
            "dapd_rejection_events": len(all_events),
            "dapd_headroom_eligible_rejection_events": dapd_summary["headroom_eligible_rejection_events"],
            "demask_status": demask_blocker["status"],
        }, ensure_ascii=False, indent=2))
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        _write_json(summary_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
