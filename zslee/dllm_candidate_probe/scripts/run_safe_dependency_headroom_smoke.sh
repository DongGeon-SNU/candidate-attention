#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/run_safe_dependency_headroom.py --config configs/safe_dependency_headroom_smoke.yaml --allow-missing-demask
