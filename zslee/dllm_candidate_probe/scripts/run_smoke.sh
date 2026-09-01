#!/usr/bin/env bash
# One prompt, one decoding state, batch/microbatch one. No background processes.
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
mkdir -p "${PROBE_ROOT}/logs" "${PROBE_ROOT}/outputs"
LOG_FILE="${PROBE_ROOT}/logs/smoke_${STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export HF_HOME="${PROBE_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
PYTHON_BIN="${PROBE_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project virtual environment. Run scripts/setup.sh on the shared-GPU PVC first." >&2
  exit 2
fi

"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/check_environment.py" \
  --probe-root "${PROBE_ROOT}" \
  --output "${PROBE_ROOT}/outputs/environment_report.md"
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/preflight.py" \
  --probe-root "${PROBE_ROOT}" \
  --environment-json "${PROBE_ROOT}/outputs/environment_report.json" \
  --stage smoke
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/run_counterfactual_probe.py" \
  --config "${PROBE_ROOT}/configs/pilot.yaml" \
  --mode smoke \
  --probe-root "${PROBE_ROOT}"
echo "smoke completed; log: ${LOG_FILE}"
