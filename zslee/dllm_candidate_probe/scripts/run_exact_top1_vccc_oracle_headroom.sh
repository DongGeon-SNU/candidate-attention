#!/usr/bin/env bash
# Exact all-order current-top1 VCCC headroom audit.  This reads one completed
# top1 trajectory bundle and writes a new child under outputs/vccc_oracle_audit.
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
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_exact_top1_vccc_oracle_headroom.sh <completed-top1-run-directory> [runner arguments]" >&2
  exit 2
fi
SOURCE_RUN="$1"
shift
if [[ ! -f "${SOURCE_RUN}/run_manifest.json" || ! -f "${SOURCE_RUN}/raw/trajectories.jsonl" || ! -f "${SOURCE_RUN}/raw/state_positions.jsonl" ]]; then
  echo "Source run must contain run_manifest.json, raw/trajectories.jsonl, and raw/state_positions.jsonl: ${SOURCE_RUN}" >&2
  exit 2
fi
PYTHON_BIN="${PROBE_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project virtual environment. Run scripts/setup.sh first." >&2
  exit 2
fi
# The audit never stops or changes another job.  If the only process is this
# exact job after startup, the owner can explicitly acknowledge it.
if command -v nvidia-smi >/dev/null 2>&1 && [[ "${EXACT_TOP1_VCCC_HEADROOM_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  BUSY="$({ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true; } | tr -d '[:space:]')"
  if [[ -n "${BUSY}" ]]; then
    echo "GPU has active compute process(es): ${BUSY}. It was not modified. Wait for it, or set EXACT_TOP1_VCCC_HEADROOM_ALLOW_BUSY_GPU=1 only if it is this job." >&2
    exit 3
  fi
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${PROBE_ROOT}/logs" "${PROBE_ROOT}/outputs/vccc_oracle_audit"
LOG_FILE="${PROBE_ROOT}/logs/exact_top1_vccc_oracle_headroom_${STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
export HF_HOME="${PROBE_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/check_environment.py" \
  --probe-root "${PROBE_ROOT}" \
  --output "${PROBE_ROOT}/outputs/environment_report.md"
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/preflight.py" \
  --probe-root "${PROBE_ROOT}" \
  --environment-json "${PROBE_ROOT}/outputs/environment_report.json" \
  --stage audit
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/run_exact_top1_vccc_oracle_headroom.py" \
  --config "${PROBE_ROOT}/configs/exact_top1_vccc_oracle_headroom.yaml" \
  --probe-root "${PROBE_ROOT}" \
  --source-run "${SOURCE_RUN}" \
  "$@"
echo "Exact top-1 VCCC headroom audit finished; inspect report.md under outputs/vccc_oracle_audit/exact_top1_vccc_oracle_headroom_*. Log: ${LOG_FILE}"
