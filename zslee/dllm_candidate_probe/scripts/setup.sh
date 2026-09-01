#!/usr/bin/env bash
# Shared-GPU setup only. Run after copying this project to a persistent volume.
set -euo pipefail

PROBE_ROOT="${DLLM_PROBE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -z "${PERSISTENT_ROOT:-}" ]]; then
  echo "Set PERSISTENT_ROOT to the mounted persistent-volume root before setup." >&2
  exit 2
fi
EXPECTED_ROOT="${PERSISTENT_ROOT%/}/zslee/dllm_candidate_probe"
if [[ "${PROBE_ROOT}" != "${EXPECTED_ROOT}" ]]; then
  echo "Project must be located at ${EXPECTED_ROOT}; current path is ${PROBE_ROOT}." >&2
  exit 2
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${PROBE_ROOT}/logs" "${PROBE_ROOT}/outputs"
LOG_FILE="${PROBE_ROOT}/logs/setup_${STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export HF_HOME="${PROBE_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PIP_CACHE_DIR="${PROBE_ROOT}/cache/pip"

echo "setup root: ${PROBE_ROOT}"
python3 "${PROBE_ROOT}/scripts/check_environment.py" \
  --probe-root "${PROBE_ROOT}" \
  --output "${PROBE_ROOT}/outputs/environment_report.md"

# This gate intentionally runs before any model/dependency download. It records
# the exact reason if a PVC, 40 GiB disk headroom, HF_TOKEN, or GPU capacity is absent.
python3 "${PROBE_ROOT}/scripts/preflight.py" \
  --probe-root "${PROBE_ROOT}" \
  --environment-json "${PROBE_ROOT}/outputs/environment_report.json" \
  --stage setup

mkdir -p "${HF_HOME}" "${PROBE_ROOT}/cache/pip" "${PROBE_ROOT}/vendor" \
  "${PROBE_ROOT}/outputs/raw" "${PROBE_ROOT}/outputs/tables" "${PROBE_ROOT}/outputs/figures"

if [[ ! -x "${PROBE_ROOT}/.venv/bin/python" ]]; then
  python3 -m venv "${PROBE_ROOT}/.venv"
fi
PYTHON_BIN="${PROBE_ROOT}/.venv/bin/python"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install torch
"${PYTHON_BIN}" -m pip install -r "${PROBE_ROOT}/requirements.txt"

FAST_DIR="${PROBE_ROOT}/vendor/Fast-dLLM"
FAST_REPO="https://github.com/NVlabs/Fast-dLLM.git"
FAST_COMMIT="a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b"
if [[ ! -d "${FAST_DIR}/.git" ]]; then
  git clone "${FAST_REPO}" "${FAST_DIR}"
fi
git -C "${FAST_DIR}" fetch --tags --force
git -C "${FAST_DIR}" checkout --detach "${FAST_COMMIT}"
git -C "${FAST_DIR}" rev-parse HEAD | tee "${PROBE_ROOT}/outputs/fast_dllm_commit.txt"
"${PYTHON_BIN}" -m pip install -r "${FAST_DIR}/v1/requirements.txt"
"${PYTHON_BIN}" -m unittest discover -s "${PROBE_ROOT}/tests" -p "test_*.py"
"${PYTHON_BIN}" -m pip freeze | sort > "${PROBE_ROOT}/outputs/dependency_versions.txt"
echo "setup completed; log: ${LOG_FILE}"
