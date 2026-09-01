#!/usr/bin/env python3
"""Fail closed before installing, downloading, or using shared-GPU resources."""

from __future__ import annotations

import argparse
import json
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
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=20)
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []
    return [[value.strip() for value in line.split(",")] for line in completed.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path, required=True)
    parser.add_argument("--stage", choices=("setup", "smoke", "pilot"), required=True)
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
        largest_free_mib = max(int(row[3]) for row in rows)
        if largest_free_mib < MIN_UNQUANTIZED_MODEL_FREE_MIB:
            failures.append(
                f"largest available GPU memory is {largest_free_mib} MiB; "
                f"the unquantized BF16 LLaDA smoke test requires about {MIN_UNQUANTIZED_MODEL_FREE_MIB} MiB free"
            )
    if failures:
        print("PRECHECK BLOCKED: " + "; ".join(failures), file=sys.stderr)
        raise SystemExit(3)
    print(f"PRECHECK PASSED for {args.stage}: persistence, disk, and GPU capacity are sufficient.")
    for warning in warnings:
        print("PRECHECK NOTE: " + warning)


if __name__ == "__main__":
    main()
