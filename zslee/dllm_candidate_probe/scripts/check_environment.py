#!/usr/bin/env python3
"""Write a secret-safe target-environment report before a model download."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(command: list[str]) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return None, str(error)
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def mounted_filesystem(path: Path) -> dict[str, str] | None:
    """Find the most-specific Linux mount containing ``path`` without a shell."""

    mounts = Path("/proc/mounts")
    if not mounts.exists():
        return None
    resolved = str(path.resolve())
    candidates: list[tuple[str, str, str]] = []
    for line in mounts.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 3:
            source, mountpoint, filesystem = fields[:3]
            if resolved == mountpoint or resolved.startswith(mountpoint.rstrip("/") + "/"):
                candidates.append((mountpoint, source, filesystem))
    if not candidates:
        return None
    mountpoint, source, filesystem = max(candidates, key=lambda row: len(row[0]))
    return {"mountpoint": mountpoint, "source": source, "filesystem": filesystem}


def persistence_assessment(probe_root: Path) -> dict[str, Any]:
    mount = mounted_filesystem(probe_root)
    if mount is None:
        return {
            "status": "unconfirmed",
            "reason": "No /proc/mounts data is available; ask the scheduler owner to confirm the volume.",
            "mount": None,
        }
    override = os.environ.get("PERSISTENCE_CONFIRMED") == "1"
    ephemeral_types = {"overlay", "tmpfs", "ramfs", "aufs"}
    likely_persistent = mount["mountpoint"] != "/" and mount["filesystem"] not in ephemeral_types
    if override:
        status, reason = "confirmed_by_operator", "PERSISTENCE_CONFIRMED=1 was supplied by the job owner."
    elif likely_persistent:
        status, reason = "likely_persistent", "A non-overlay mount distinct from container root contains the project."
    else:
        status, reason = "unconfirmed", "The project is on container root or an ephemeral filesystem."
    return {"status": status, "reason": reason, "mount": mount}


def gpu_summary() -> dict[str, Any]:
    code, output = run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]
    )
    process_code, process_output = run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    )
    version_code, version_output = run(["nvidia-smi"])
    gpus = [line.strip() for line in output.splitlines() if line.strip()] if code == 0 else []
    processes = [line.strip() for line in process_output.splitlines() if line.strip()] if process_code == 0 else []
    driver_line = next((line.strip() for line in version_output.splitlines() if "Driver Version" in line), "unavailable")
    return {
        "gpus": gpus,
        "compute_process_count": len(processes),
        "compute_processes": processes,
        "driver_cuda_line": driver_line if version_code == 0 else "nvidia-smi unavailable",
    }


def torch_summary() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False, "version": None, "cuda_runtime": None, "bf16_supported": None}
    available = bool(torch.cuda.is_available())
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": available,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if available else False,
    }


def execution_kind() -> str:
    if os.environ.get("KUBERNETES_SERVICE_HOST") or Path("/var/run/secrets/kubernetes.io").exists():
        return "Kubernetes pod/job (detected)"
    if any(os.environ.get(key) for key in ("JUPYTERHUB_USER", "JPY_SESSION_NAME", "COLAB_GPU")):
        return "Jupyter container/session (detected)"
    if any(os.environ.get(key) for key in ("SLURM_JOB_ID", "PBS_JOBID", "JOB_NAME")):
        return "batch job (detected)"
    return "terminal session (best-effort inference)"


def report(probe_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(probe_root)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_path": str(Path.cwd()),
        "probe_root": str(probe_root),
        "user": os.environ.get("USER") or os.environ.get("USERNAME") or "unavailable",
        "hostname": socket.gethostname(),
        "pwd": str(Path.cwd()),
        "id": os.environ.get("UID", "unavailable; run `id` in a POSIX target if needed"),
        "os": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "gpu": gpu_summary(),
        "disk": {
            "filesystem_path_checked": str(probe_root),
            "free_gib": round(disk.free / 2**30, 2),
            "total_gib": round(disk.total / 2**30, 2),
            "df_h": run(["df", "-h", str(probe_root)])[1],
        },
        "persistence": persistence_assessment(probe_root),
        "torch": torch_summary(),
        "hf_authentication": "authenticated" if os.environ.get("HF_TOKEN") else "not_detected",
        "scheduler_cli": {"kubectl": shutil.which("kubectl") is not None, "squeue": shutil.which("squeue") is not None},
        "execution_environment": execution_kind(),
        "note": "No environment-variable dump was collected; no secret values are included.",
    }


def markdown(data: dict[str, Any]) -> str:
    gpu, persistence, torch = data["gpu"], data["persistence"], data["torch"]
    lines = [
        "# Shared-GPU environment report",
        "",
        f"- Generated (UTC): `{data['generated_at_utc']}`",
        f"- Current path / pwd: `{data['current_path']}`",
        f"- Probe root: `{data['probe_root']}`",
        f"- User: `{data['user']}`",
        f"- Hostname: `{data['hostname']}`",
        f"- id: `{data['id']}`",
        f"- OS: `{data['os']}`",
        f"- Python: `{data['python']}`",
        f"- Execution environment: {data['execution_environment']}",
        "",
        "## GPU",
        "",
        f"- Driver/runtime: `{gpu['driver_cuda_line']}`",
        f"- GPUs (`name, total MiB, used MiB, free MiB`): {gpu['gpus'] or 'unavailable'}",
        f"- Visible compute processes: {gpu['compute_process_count']} ({gpu['compute_processes'] or 'none'})",
        f"- PyTorch: installed={torch['installed']}, version={torch['version']}, CUDA={torch['cuda_runtime']}, BF16={torch['bf16_supported']}",
        "",
        "## Disk and persistence",
        "",
        f"- Free / total at probe root: {data['disk']['free_gib']} GiB / {data['disk']['total_gib']} GiB",
        f"- `df -h`: `{data['disk']['df_h'] or 'unavailable'}`",
        f"- Persistence: **{persistence['status']}** — {persistence['reason']}",
        f"- Mount: `{json.dumps(persistence['mount'], ensure_ascii=False)}`",
        "",
        "## Access and authentication",
        "",
        f"- Hugging Face: {'인증 확인됨' if data['hf_authentication'] == 'authenticated' else 'not detected; set HF_TOKEN as a Portainer/Kubernetes secret environment variable if access is denied.'}",
        f"- Scheduler CLI: kubectl={data['scheduler_cli']['kubectl']}, squeue={data['scheduler_cli']['squeue']}",
        "",
        "No secret values or full environment-variable dumps are stored in this report.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.probe_root.mkdir(parents=True, exist_ok=True)
    data = report(args.probe_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
