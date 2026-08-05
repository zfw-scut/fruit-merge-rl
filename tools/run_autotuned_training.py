"""读取性能标定结果并以选中配置启动正式训练。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daxigua.rl.config import TrainingConfig
from daxigua.rl.trainer import BaselineTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=Path,
        default=PROJECT_ROOT / 'configs' / 'gnn_dqn_baseline.toml',
    )
    parser.add_argument(
        '--autotune-report', type=Path,
        default=PROJECT_ROOT / 'runs' / 'autotune' / 'training_pipeline.json',
    )
    parser.add_argument('--run-dir')
    parser.add_argument('--max-wall-hours', type=float)
    parser.add_argument('--total-transitions', type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    report = json.loads(args.autotune_report.read_text(encoding='utf-8'))
    base = TrainingConfig.from_toml(args.config)
    active_envs = int(report['selected_num_envs'])
    maximum_envs = int(report['maximum_successful_envs'])
    maximum_envs = max(active_envs, maximum_envs)
    replay_capacity = min(
        2_097_152,
        max(1_048_576, 256 * maximum_envs),
    )
    warmup = min(replay_capacity // 4, 524_288)
    config = replace(
        base,
        run_dir=args.run_dir or base.run_dir,
        max_envs=maximum_envs,
        active_envs=active_envs,
        total_transitions=(
            args.total_transitions
            if args.total_transitions is not None
            else base.total_transitions
        ),
        max_wall_seconds=(
            args.max_wall_hours * 3600.0
            if args.max_wall_hours is not None
            else base.max_wall_seconds
        ),
        replay=replace(
            base.replay,
            capacity=replay_capacity,
            batch_size=int(report['selected_batch_size']),
            warmup_transitions=warmup,
        ),
        dqn=replace(
            base.dqn,
            compile_model=bool(report['selected_compile_model']),
            use_bfloat16=bool(report['selected_use_bfloat16']),
        ),
    )
    trainer = BaselineTrainer(config, project_root=PROJECT_ROOT)

    def stop_handler(signum, _frame):
        trainer.request_stop(f'signal_{signum}')

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, stop_handler)
    result = trainer.run(final_evaluation=True)
    print(result)


if __name__ == '__main__':
    main()
