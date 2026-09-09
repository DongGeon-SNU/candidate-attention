#!/usr/bin/env bash
# Offline candidate-conditioned normal-decoder rollout audit.
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
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_vccc_rollout_candidate_audit.sh <completed-top1-run> <completed-vccc-run> [runner arguments]" >&2
  exit 2
fi
SOURCE_RUN="$1"
VCCC_RUN="$2"
shift 2
if [[ ! -f "${SOURCE_RUN}/run_manifest.json" || ! -f "${SOURCE_RUN}/raw/trajectories.jsonl" ]]; then
  echo "Source run must contain run_manifest.json and raw/trajectories.jsonl: ${SOURCE_RUN}" >&2
  exit 2
fi
if [[ ! -f "${VCCC_RUN}/run_metadata.json" || ! -f "${VCCC_RUN}/raw/candidate_polarity_assignments.jsonl" || ! -f "${VCCC_RUN}/raw/selected_states.jsonl" ]]; then
  echo "VCCC run must contain run_metadata.json, raw/candidate_polarity_assignments.jsonl, and raw/selected_states.jsonl: ${VCCC_RUN}" >&2
  exit 2
fi
PYTHON_BIN="${PROBE_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project virtual environment. Run scripts/setup.sh first." >&2
  exit 2
fi
if command -v nvidia-smi >/dev/null 2>&1 && [[ "${VCCC_ROLLOUT_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  BUSY="$({ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true; } | tr -d '[:space:]')"
  if [[ -n "${BUSY}" ]]; then
    echo "GPU has active compute process(es): ${BUSY}. It was not modified. Wait for it, or set VCCC_ROLLOUT_ALLOW_BUSY_GPU=1 only if it is this job." >&2
    exit 3
  fi
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${PROBE_ROOT}/logs" "${PROBE_ROOT}/outputs/vccc_oracle_audit"
LOG_FILE="${PROBE_ROOT}/logs/vccc_rollout_candidate_audit_${STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
export HF_HOME="${PROBE_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/check_environment.py" --probe-root "${PROBE_ROOT}" --output "${PROBE_ROOT}/outputs/environment_report.md"
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/preflight.py" --probe-root "${PROBE_ROOT}" --environment-json "${PROBE_ROOT}/outputs/environment_report.json" --stage audit
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/run_vccc_rollout_candidate_audit.py" \
  --config "${PROBE_ROOT}/configs/vccc_rollout_candidate_audit.yaml" \
  --probe-root "${PROBE_ROOT}" \
  --source-run "${SOURCE_RUN}" \
  --vccc-run "${VCCC_RUN}" \
  "$@"
echo "VCCC rollout candidate audit finished; inspect report.md under outputs/vccc_oracle_audit/. Log: ${LOG_FILE}"
