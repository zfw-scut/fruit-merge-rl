#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir/src"

python tools/preflight_training.py \
  --config configs/gnn_dqn_reward_v2_1.toml \
  --output runs/preflight/gnn_dqn.json

python tools/benchmark_training_pipeline.py \
  --output runs/autotune/training_pipeline.json

python tools/compare_reward_throughput.py \
  --pipeline-report runs/autotune/training_pipeline.json \
  --output runs/autotune/reward_overhead.json

python tools/run_autotuned_training.py \
  --config configs/gnn_dqn_reward_v2_1.toml \
  --autotune-report runs/autotune/training_pipeline.json \
  "$@"
