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
# the exact reason if a PVC, 40 GiB disk headroom, or GPU capacity is absent.
# HF_TOKEN is optional for this public model and is only needed if access fails.
python3 "${PROBE_ROOT}/scripts/preflight.py" \
  --probe-root "${PROBE_ROOT}" \
  --environment-json "${PROBE_ROOT}/outputs/environment_report.json" \
  --stage setup

mkdir -p "${HF_HOME}" "${PROBE_ROOT}/cache/pip" "${PROBE_ROOT}/vendor" \
  "${PROBE_ROOT}/outputs/raw" "${PROBE_ROOT}/outputs/tables" "${PROBE_ROOT}/outputs/figures" \
  "${PROBE_ROOT}/results/raw" "${PROBE_ROOT}/results/summary"

if [[ ! -x "${PROBE_ROOT}/.venv/bin/python" ]]; then
  python3 -m venv "${PROBE_ROOT}/.venv"
fi
PYTHON_BIN="${PROBE_ROOT}/.venv/bin/python"
"${PYTHON_BIN}" -m pip install --upgrade pip
# H100 reproducibility gate.  A bare `pip install torch` may select a CPU
# wheel or change CUDA/kernel behavior between reruns.  This pinned CUDA wheel
# is intentionally overrideable only as an explicit, recorded setup choice.
TORCH_VERSION="${SAFE_DEPENDENCY_TORCH_VERSION:-2.5.1+cu121}"
TORCH_INDEX_URL="${SAFE_DEPENDENCY_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
"${PYTHON_BIN}" -m pip install --upgrade --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
export TORCH_VERSION TORCH_INDEX_URL
"${PYTHON_BIN}" - <<'PY'
import os
import torch

expected = os.environ["TORCH_VERSION"]
if torch.__version__ != expected:
    raise SystemExit(f"Pinned torch mismatch: expected {expected}, got {torch.__version__}")
if torch.version.cuda is None or not torch.cuda.is_available():
    raise SystemExit(
        f"Pinned torch must expose CUDA on this H100 job; torch={torch.__version__}, cuda={torch.version.cuda}"
    )
print(f"verified torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY
"${PYTHON_BIN}" -m pip install -r "${PROBE_ROOT}/requirements.txt"

PATCH_DIR="${PROBE_ROOT}/patches"
FAST_DIR="${PROBE_ROOT}/vendor/Fast-dLLM"
FAST_REPO="https://github.com/NVlabs/Fast-dLLM.git"
FAST_COMMIT="a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b"
DAPD_DIR="${PROBE_ROOT}/vendor/DAPD"
DAPD_REPO="https://github.com/quasar529/DAPD.git"
DAPD_COMMIT="05727b08da4cb4008a275123d7d9885dd5714f7c"

# We deliberately keep the observation hooks as small, versioned patches
# against the two upstream releases.  The experiment runner refuses to start
# without them; setup verifies both the upstream commit and the exact patch
# state.  A rerun recognizes an already-applied patch, but never overwrites an
# unrelated dirty vendor checkout.
ensure_pinned_checkout() {
  local label="$1"
  local repo_url="$2"
  local repo_dir="$3"
  local expected_commit="$4"

  if [[ ! -d "${repo_dir}/.git" ]]; then
    git clone "${repo_url}" "${repo_dir}"
  fi
  git -C "${repo_dir}" fetch --tags --force

  local current_commit
  current_commit="$(git -C "${repo_dir}" rev-parse HEAD)"
  if [[ "${current_commit}" != "${expected_commit}" ]]; then
    if ! git -C "${repo_dir}" diff --quiet || ! git -C "${repo_dir}" diff --cached --quiet; then
      echo "${label} is dirty at ${current_commit}; refusing to overwrite it. Restore or reclone ${repo_dir}." >&2
      exit 2
    fi
    git -C "${repo_dir}" checkout --detach "${expected_commit}"
  fi

  current_commit="$(git -C "${repo_dir}" rev-parse HEAD)"
  if [[ "${current_commit}" != "${expected_commit}" ]]; then
    echo "${label} pin mismatch: expected ${expected_commit}, found ${current_commit}." >&2
    exit 2
  fi
}

apply_observation_patch() {
  local label="$1"
  local repo_dir="$2"
  local patch_file="$3"

  if [[ ! -f "${patch_file}" ]]; then
    echo "Missing required ${label} observation patch: ${patch_file}" >&2
    exit 2
  fi

  # Reverse-check first: it is the only safe idempotence test because a
  # patched vendor tree is intentionally dirty relative to its upstream pin.
  if git -C "${repo_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo "${label} observation patch already applied"
  elif git -C "${repo_dir}" apply --check "${patch_file}" >/dev/null 2>&1; then
    git -C "${repo_dir}" apply --whitespace=nowarn "${patch_file}"
    echo "${label} observation patch applied"
  else
    echo "${label} checkout does not match the pinned source or observation patch; refusing to continue." >&2
    exit 2
  fi
}

verify_exact_patch_state() {
  local label="$1"
  local repo_dir="$2"
  local patch_file="$3"
  local expected_commit="$4"
  local scratch
  scratch="$(mktemp -d)"

  # First reject extra tracked changes. Then reconstruct exactly the expected
  # patched files from the pinned commit in a disposable tree and byte-compare
  # them with the vendor checkout. This prevents a partially compatible local
  # edit from being mistaken for a verified observation-only source patch.
  mapfile -t expected_paths < <(sed -n 's#^+++ b/##p' "${patch_file}" | sort)
  mapfile -t actual_paths < <(git -C "${repo_dir}" diff --name-only | sort)
  if [[ "${#expected_paths[@]}" -eq 0 ]] || \
     [[ "${#expected_paths[@]}" -ne "${#actual_paths[@]}" ]] || \
     ! cmp -s <(printf '%s\n' "${expected_paths[@]}") <(printf '%s\n' "${actual_paths[@]}"); then
    rm -rf -- "${scratch}"
    echo "${label} has tracked changes beyond the required observation patch." >&2
    exit 2
  fi

  if ! git -C "${repo_dir}" archive "${expected_commit}" | tar -x -C "${scratch}" || \
     ! git -C "${scratch}" init -q || \
     ! git -C "${scratch}" apply --check "${patch_file}" || \
     ! git -C "${scratch}" apply --whitespace=nowarn "${patch_file}"; then
    rm -rf -- "${scratch}"
    echo "Could not reconstruct the expected ${label} patched source tree." >&2
    exit 2
  fi

  for expected_path in "${expected_paths[@]}"; do
    if ! cmp -s "${repo_dir}/${expected_path}" "${scratch}/${expected_path}"; then
      rm -rf -- "${scratch}"
      echo "${label} patched file differs from the pinned patch result: ${expected_path}" >&2
      exit 2
    fi
  done
  rm -rf -- "${scratch}"
}

ensure_pinned_checkout "Fast-dLLM" "${FAST_REPO}" "${FAST_DIR}" "${FAST_COMMIT}"
ensure_pinned_checkout "DAPD" "${DAPD_REPO}" "${DAPD_DIR}" "${DAPD_COMMIT}"
apply_observation_patch "Fast-dLLM" "${FAST_DIR}" "${PATCH_DIR}/fast_dllm_trace_hooks.patch"
apply_observation_patch "DAPD" "${DAPD_DIR}" "${PATCH_DIR}/dapd_trace_hooks.patch"
verify_exact_patch_state "Fast-dLLM" "${FAST_DIR}" "${PATCH_DIR}/fast_dllm_trace_hooks.patch" "${FAST_COMMIT}"
verify_exact_patch_state "DAPD" "${DAPD_DIR}" "${PATCH_DIR}/dapd_trace_hooks.patch" "${DAPD_COMMIT}"

# A patch that merely applies is not sufficient evidence that the runner's
# callbacks are available. Check the exact public hook names before installing
# or downloading a model for the experiment.
if ! grep -Fq "step_observer" "${FAST_DIR}/v1/llada/generate.py"; then
  echo "Fast-dLLM observation hook missing after patch application." >&2
  exit 2
fi
if ! grep -Fq "selection_observer" "${DAPD_DIR}/dapd/generation.py" || \
   ! grep -Fq "step_observer" "${DAPD_DIR}/dapd/generation.py" || \
   ! grep -Fq "decision_observer" "${DAPD_DIR}/dapd/core.py"; then
  echo "DAPD observation hooks missing after patch application." >&2
  exit 2
fi

FAST_HEAD="$(git -C "${FAST_DIR}" rev-parse HEAD)"
DAPD_HEAD="$(git -C "${DAPD_DIR}" rev-parse HEAD)"
printf '%s\n' "${FAST_HEAD}" > "${PROBE_ROOT}/outputs/fast_dllm_commit.txt"
printf '%s\n' "${DAPD_HEAD}" > "${PROBE_ROOT}/outputs/dapd_commit.txt"

"${PYTHON_BIN}" -m py_compile \
  "${FAST_DIR}/v1/llada/generate.py" \
  "${DAPD_DIR}/dapd/core.py" \
  "${DAPD_DIR}/dapd/generation.py"
"${PYTHON_BIN}" -m pip install -r "${FAST_DIR}/v1/requirements.txt"

FAST_PATCH_SHA256="$(sha256sum "${PATCH_DIR}/fast_dllm_trace_hooks.patch" | awk '{print $1}')"
DAPD_PATCH_SHA256="$(sha256sum "${PATCH_DIR}/dapd_trace_hooks.patch" | awk '{print $1}')"
FAST_GENERATE_SHA256="$(sha256sum "${FAST_DIR}/v1/llada/generate.py" | awk '{print $1}')"
DAPD_CORE_SHA256="$(sha256sum "${DAPD_DIR}/dapd/core.py" | awk '{print $1}')"
DAPD_GENERATION_SHA256="$(sha256sum "${DAPD_DIR}/dapd/generation.py" | awk '{print $1}')"
export FAST_HEAD DAPD_HEAD FAST_COMMIT DAPD_COMMIT FAST_PATCH_SHA256 DAPD_PATCH_SHA256 \
  FAST_GENERATE_SHA256 DAPD_CORE_SHA256 DAPD_GENERATION_SHA256
"${PYTHON_BIN}" - "${PROBE_ROOT}/outputs/safe_dependency_headroom_source_manifest.json" <<'PY'
import json
import os
import platform
import sys
from datetime import datetime, timezone

destination = sys.argv[1]
payload = {
    "schema_version": "safe_dependency_headroom/source-manifest/v2",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "torch_install": {
        "requested_version": os.environ["TORCH_VERSION"],
        "requested_index_url": os.environ["TORCH_INDEX_URL"],
    },
    "sources": {
        "fast_dllm": {
            "expected_commit": os.environ["FAST_COMMIT"],
            "resolved_commit": os.environ["FAST_HEAD"],
            "observation_patch_sha256": os.environ["FAST_PATCH_SHA256"],
            "patched_files_sha256": {
                "v1/llada/generate.py": os.environ["FAST_GENERATE_SHA256"],
            },
        },
        "dapd": {
            "expected_commit": os.environ["DAPD_COMMIT"],
            "resolved_commit": os.environ["DAPD_HEAD"],
            "observation_patch_sha256": os.environ["DAPD_PATCH_SHA256"],
            "patched_files_sha256": {
                "dapd/core.py": os.environ["DAPD_CORE_SHA256"],
                "dapd/generation.py": os.environ["DAPD_GENERATION_SHA256"],
            },
        },
    },
}
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

# DAPD is independently pinned because this audit must use its published
# attention graph and independent-set rule, not an in-project approximation.
"${PYTHON_BIN}" -m unittest discover -s "${PROBE_ROOT}/tests" -p "test_*.py"
"${PYTHON_BIN}" -m pip freeze | sort > "${PROBE_ROOT}/outputs/dependency_versions.txt"
echo "setup completed; log: ${LOG_FILE}"
