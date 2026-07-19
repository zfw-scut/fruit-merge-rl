#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${1:-}" == --* || "$#" -eq 0 ]]; then
  CONFIG_PATH="configs/train_dqn_fast30_parallel.toml"
else
  CONFIG_PATH="$1"
  shift
fi

CONDA_ENV="${CONDA_ENV:-python-torch}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p runs/launcher_logs
LOG_FILE="runs/launcher_logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "config=${CONFIG_PATH}"
echo "conda_env=${CONDA_ENV}"
echo "log_file=${LOG_FILE}"
echo

conda run --no-capture-output -n "${CONDA_ENV}" \
  python -u -m daxigua_rl.scripts.train_dqn \
  --config "${CONFIG_PATH}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
