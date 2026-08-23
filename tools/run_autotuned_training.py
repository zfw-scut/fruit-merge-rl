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
from daxigua.rl.checkpoint import sha256_file
from daxigua.rl.trainer import BaselineTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=Path,
        default=PROJECT_ROOT / 'configs' / 'gnn_dqn_reward_v2_1.toml',
    )
    parser.add_argument(
        '--autotune-report', type=Path,
        default=PROJECT_ROOT / 'runs' / 'autotune' / 'training_pipeline.json',
    )
    parser.add_argument('--run-dir')
    parser.add_argument('--max-wall-hours', type=float)
    parser.add_argument('--total-transitions', type=int)
    parser.add_argument('--reward-scale', type=float)
    parser.add_argument('--dashboard-port', type=int)
    parser.add_argument('--disable-dashboard', action='store_true')
    parser.add_argument('--curve-snapshot-interval', type=float)
    parser.add_argument('--disable-curve-snapshots', action='store_true')
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--init-checkpoint', type=Path)
    initialization.add_argument(
        '--prewarm-checkpoint',
        type=Path,
        help='只用来源策略生成预热数据，新训练模型保持随机初始化',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = json.loads(args.autotune_report.read_text(encoding='utf-8'))
    base = TrainingConfig.from_toml(args.config)
    if not report.get('complete', False):
        raise ValueError('throughput calibration report is incomplete')
    report_config = Path(report['config']).resolve()
    if report_config != args.config.resolve():
        raise ValueError('throughput report belongs to a different config')
    if args.prewarm_checkpoint is not None:
        expected_teacher_hash = report.get('prewarm_checkpoint_sha256')
        if (
                expected_teacher_hash
                and sha256_file(args.prewarm_checkpoint)
                != expected_teacher_hash):
            raise ValueError(
                'prewarm checkpoint hash differs from calibration input'
            )
    active_envs = int(report['selected_num_envs'])
    maximum_envs = (
        max(active_envs, int(report['maximum_successful_envs']))
        if base.autoscale.enabled else active_envs
    )
    replay_capacity = int(
        report.get('selected_replay_capacity', base.replay.capacity)
    )
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
        ),
        dqn=replace(
            base.dqn,
            compile_model=bool(report['selected_compile_model']),
            use_bfloat16=bool(report['selected_use_bfloat16']),
        ),
        reward=replace(
            base.reward,
            reward_scale=(
                args.reward_scale
                if args.reward_scale is not None
                else base.reward.reward_scale
            ),
        ),
        dashboard=replace(
            base.dashboard,
            enabled=(base.dashboard.enabled and not args.disable_dashboard),
            port=(
                args.dashboard_port
                if args.dashboard_port is not None
                else base.dashboard.port
            ),
            curve_snapshot_enabled=(
                base.dashboard.curve_snapshot_enabled
                and not args.disable_curve_snapshots
            ),
            curve_snapshot_interval_seconds=(
                args.curve_snapshot_interval
                if args.curve_snapshot_interval is not None
                else base.dashboard.curve_snapshot_interval_seconds
            ),
        ),
    )
    trainer = BaselineTrainer(config, project_root=PROJECT_ROOT)
    if args.init_checkpoint is not None:
        trainer.initialize_from_checkpoint(args.init_checkpoint)
    elif args.prewarm_checkpoint is not None:
        trainer.load_stage_pilot_checkpoint(args.prewarm_checkpoint)
    elif config.stage_pilot_policy_epsilon is not None:
        raise ValueError(
            'this config requires --prewarm-checkpoint; the teacher must not '
            'be loaded through --init-checkpoint'
        )

    def stop_handler(signum, _frame):
        trainer.request_stop(f'signal_{signum}')

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, stop_handler)
    result = trainer.run(final_evaluation=True)
    print(result)


if __name__ == '__main__':
    main()
