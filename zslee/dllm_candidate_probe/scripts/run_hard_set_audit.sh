#!/usr/bin/env bash
# The Python entrypoint estimates all planned LOO work and refuses over 30 min.
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
mkdir -p "${PROBE_ROOT}/logs" "${PROBE_ROOT}/outputs/hard_set"
LOG_FILE="${PROBE_ROOT}/logs/hard_set_audit_${STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export HF_HOME="${PROBE_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
PYTHON_BIN="${PROBE_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project virtual environment. Run scripts/setup.sh first." >&2
  exit 2
fi
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/check_environment.py" --probe-root "${PROBE_ROOT}" --output "${PROBE_ROOT}/outputs/environment_report.md"
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/preflight.py" --probe-root "${PROBE_ROOT}" --environment-json "${PROBE_ROOT}/outputs/environment_report.json" --stage audit
"${PYTHON_BIN}" "${PROBE_ROOT}/scripts/run_hard_set_audit.py" --config "${PROBE_ROOT}/configs/hard_set_audit.yaml" --mode audit --probe-root "${PROBE_ROOT}"
echo "hard-set audit finished or was safely budget-blocked; log: ${LOG_FILE}"
