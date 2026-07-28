#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${1:-}" == --* || "$#" -eq 0 ]]; then
  CONFIG_PATH="configs/train_dqn_causal_250k.toml"
else
  CONFIG_PATH="$1"
  shift
fi

CONDA_ENV="${CONDA_ENV:-python-torch}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# 正式训练会同时使用 rollout、反事实和 Shapley 多进程队列。即使所有动态资源都
# 正确释放，Linux 常见的 1024 soft limit 也过于接近固定句柄基线。默认把 soft
# limit 提升到当前进程允许的 65535；调用方仍可用 TRAIN_NOFILE_LIMIT 显式覆盖。
REQUESTED_NOFILE="${TRAIN_NOFILE_LIMIT:-65535}"
if ! [[ "${REQUESTED_NOFILE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_NOFILE_LIMIT must be a positive integer" >&2
  exit 2
fi
HARD_NOFILE="$(ulimit -Hn)"
APPLIED_NOFILE="${REQUESTED_NOFILE}"
if [[ "${HARD_NOFILE}" != "unlimited" ]] \
    && (( APPLIED_NOFILE > HARD_NOFILE )); then
  APPLIED_NOFILE="${HARD_NOFILE}"
fi
ulimit -Sn "${APPLIED_NOFILE}"

mkdir -p runs/launcher_logs
LOG_FILE="runs/launcher_logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "config=${CONFIG_PATH}"
echo "conda_env=${CONDA_ENV}"
echo "nofile_soft=$(ulimit -Sn)"
echo "nofile_hard=${HARD_NOFILE}"
echo "log_file=${LOG_FILE}"
echo

conda run --no-capture-output -n "${CONDA_ENV}" \
  python -u -m daxigua_rl.scripts.train_dqn \
  --config "${CONFIG_PATH}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
