#!/usr/bin/env bash
# H100 preflight and one small exact top-1-dynamics audit run.
set -euo pipefail

PROBE_ROOT="${DLLM_PROBE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -z "${PERSISTENT_ROOT:-}" ]]; then
  echo "Set PERSISTENT_ROOT to the mounted persistent-volume root." >&2
  exit 2
fi
EXPECTED_ROOT="${PERSISTENT_ROOT%/}/zslee/dllm_candidate_probe"
if [[ "${PROBE_ROOT}" != "${EXPECTED_ROOT}" ]]; then
  echo "Project must be located at ${EXPECTED_ROOT}; current path is ${PROBE_ROOT}." >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${PROBE_ROOT}/logs" "${PROBE_ROOT}/outputs/top1_dynamics_audit"
LOG_FILE="${PROBE_ROOT}/logs/top1_dynamics_smoke_${STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export HF_HOME="${PROBE_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
PYTHON_BIN="${PROBE_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project virtual environment. Run scripts/setup.sh first." >&2
  exit 2
fi
# The runner loads PyTorch logical cuda:0.  On a multi-GPU host require the
# scheduler/operator to map that logical device to the intended H100 rather
# than letting preflight inspect one physical GPU and execution use another.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  VISIBLE_GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if [[ "${VISIBLE_GPU_COUNT}" -gt 1 ]]; then
    echo "Multiple GPUs are visible; export CUDA_VISIBLE_DEVICES=<assigned-H100-index> before running." >&2
    exit 2
  fi
fi

"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/check_environment.py" \
  --probe-root "${PROBE_ROOT}" \
  --output "${PROBE_ROOT}/outputs/environment_report.md"
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/preflight.py" \
  --probe-root "${PROBE_ROOT}" \
  --environment-json "${PROBE_ROOT}/outputs/environment_report.json" \
  --stage smoke
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/run_top1_dynamics_audit.py" \
  --config "${PROBE_ROOT}/configs/top1_dynamics_audit.yaml" \
  --mode smoke \
  --probe-root "${PROBE_ROOT}" \
  --allow-remote-datasets \
  --allow-fallback-smoke-data \
  "$@"
echo "top-1 dynamics smoke finished; inspect run_manifest.json and log: ${LOG_FILE}"
