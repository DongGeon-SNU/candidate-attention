#!/usr/bin/env python3
"""Fail closed before installing, downloading, or using shared-GPU resources."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MIN_DOWNLOAD_FREE_GIB = 40.0
# Approximate BF16 weights (~16 GiB) plus runtime and activations; no quantization.
MIN_UNQUANTIZED_MODEL_FREE_MIB = 20_000


def nvidia_rows() -> list[list[str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=20)
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []
    return [[value.strip() for value in line.split(",")] for line in completed.stdout.splitlines() if line.strip()]


def selected_visible_gpu(rows: list[list[str]]) -> tuple[list[str] | None, str | None]:
    """Choose the physical GPU that PyTorch logical ``cuda:0`` will use.

    On a shared host, preflight may not bless the least-busy physical GPU and
    then let the runner use another one.  The wrappers require an explicit
    ``CUDA_VISIBLE_DEVICES`` selection when more than one GPU is exposed; this
    helper verifies that its first selector maps to a visible nvidia-smi row.
    """

    selector_text = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not selector_text:
        if len(rows) == 1:
            return rows[0], None
        return None, "multiple GPUs are visible; set CUDA_VISIBLE_DEVICES to the H100 assigned to this run"
    selector = selector_text.split(",", 1)[0].strip()
    selected = next((row for row in rows if selector in {row[0], row[1]}), None)
    if selected is None:
        return None, (
            f"CUDA_VISIBLE_DEVICES first selector {selector!r} could not be mapped to nvidia-smi index/UUID; "
            "use one explicit GPU index or UUID"
        )
    return selected, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path, required=True)
    parser.add_argument("--stage", choices=("setup", "smoke", "pilot", "audit"), required=True)
    args = parser.parse_args()

    environment = json.loads(args.environment_json.read_text(encoding="utf-8"))
    failures: list[str] = []
    persistence = environment["persistence"]["status"]
    if persistence not in {"likely_persistent", "confirmed_by_operator"}:
        failures.append("persistent storage is not confirmed")
    free_gib = shutil.disk_usage(args.probe_root).free / 2**30
    if free_gib < MIN_DOWNLOAD_FREE_GIB:
        failures.append(f"only {free_gib:.2f} GiB free; at least {MIN_DOWNLOAD_FREE_GIB:.0f} GiB is required before download")
    warnings: list[str] = []
    if environment["hf_authentication"] != "authenticated":
        warnings.append("HF_TOKEN is not detected; attempting the public model download without authentication")
    rows = nvidia_rows()
    if not rows:
        failures.append("nvidia-smi did not expose a usable GPU")
    else:
        selected, selection_error = selected_visible_gpu(rows)
        if selection_error:
            failures.append(selection_error)
        assert selected is not None or selection_error is not None
        if selected is None:
            selected = rows[0]
        largest_free_mib = int(selected[5])
        if largest_free_mib < MIN_UNQUANTIZED_MODEL_FREE_MIB:
            failures.append(
                f"selected CUDA-visible GPU has {largest_free_mib} MiB free; "
                f"the unquantized BF16 LLaDA smoke test requires about {MIN_UNQUANTIZED_MODEL_FREE_MIB} MiB free"
            )
        if int(selected[4]) > int(selected[3]) // 2:
            failures.append(
                f"selected CUDA-visible GPU already has {selected[4]} MiB allocated; do not automatically add work to a heavily shared GPU"
            )
    if failures:
        print("PRECHECK BLOCKED: " + "; ".join(failures), file=sys.stderr)
        raise SystemExit(3)
    print(
        f"PRECHECK PASSED for {args.stage}: persistence, disk, and selected CUDA-visible GPU capacity are sufficient."
    )
    for warning in warnings:
        print("PRECHECK NOTE: " + warning)


if __name__ == "__main__":
    main()
