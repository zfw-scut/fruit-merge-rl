"""第一版 GNN-DQN 的正式训练入口。"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    parser.add_argument('--run-dir')
    parser.add_argument('--device')
    parser.add_argument('--total-transitions', type=int)
    parser.add_argument('--max-wall-hours', type=float)
    parser.add_argument('--max-envs', type=int)
    parser.add_argument('--active-envs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--replay-capacity', type=int)
    parser.add_argument('--warmup-transitions', type=int)
    parser.add_argument('--dashboard-port', type=int)
    parser.add_argument('--disable-dashboard', action='store_true')
    parser.add_argument('--curve-snapshot-interval', type=float)
    parser.add_argument('--disable-curve-snapshots', action='store_true')
    parser.add_argument('--disable-autoscale', action='store_true')
    parser.add_argument('--compile-model', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--skip-final-evaluation', action='store_true')
    parser.add_argument(
        '--smoke', action='store_true',
        help='使用极小 CUDA 配置验证完整训练、面板、评估和保存链路',
    )
    return parser.parse_args()


def resolve_config(args):
    config = TrainingConfig.from_toml(args.config)
    replay = replace(
        config.replay,
        capacity=(
            args.replay_capacity
            if args.replay_capacity is not None
            else config.replay.capacity
        ),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else config.replay.batch_size
        ),
        warmup_transitions=(
            args.warmup_transitions
            if args.warmup_transitions is not None
            else config.replay.warmup_transitions
        ),
    )
    dashboard = replace(
        config.dashboard,
        enabled=(config.dashboard.enabled and not args.disable_dashboard),
        port=(
            args.dashboard_port
            if args.dashboard_port is not None
            else config.dashboard.port
        ),
        curve_snapshot_enabled=(
            config.dashboard.curve_snapshot_enabled
            and not args.disable_curve_snapshots
        ),
        curve_snapshot_interval_seconds=(
            args.curve_snapshot_interval
            if args.curve_snapshot_interval is not None
            else config.dashboard.curve_snapshot_interval_seconds
        ),
    )
    autoscale = replace(
        config.autoscale,
        enabled=(config.autoscale.enabled and not args.disable_autoscale),
    )
    dqn = replace(
        config.dqn,
        compile_model=(config.dqn.compile_model or args.compile_model),
    )
    resolved = replace(
        config,
        run_dir=args.run_dir or config.run_dir,
        device=args.device or config.device,
        total_transitions=(
            args.total_transitions
            if args.total_transitions is not None
            else config.total_transitions
        ),
        max_wall_seconds=(
            args.max_wall_hours * 3600.0
            if args.max_wall_hours is not None
            else config.max_wall_seconds
        ),
        max_envs=args.max_envs or config.max_envs,
        active_envs=args.active_envs or config.active_envs,
        replay=replay,
        dqn=dqn,
        dashboard=dashboard,
        autoscale=autoscale,
    )
    if args.smoke:
        resolved = replace(
            resolved,
            max_envs=8,
            active_envs=8,
            total_transitions=24,
            max_wall_seconds=0.0,
            finalization_reserve_seconds=0.0,
            log_interval_seconds=0.1,
            checkpoint_interval_seconds=3600.0,
            max_episode_drops=8,
            replay=replace(
                resolved.replay,
                capacity=64,
                batch_size=4,
                warmup_transitions=16,
                warmup_stage_ratios=(1.0, 0.0, 0.0, 0.0),
            ),
            evaluation=replace(
                resolved.evaluation,
                fast_interval_transitions=1_000_000,
                accurate_milestones=(),
                periodic_episodes=2,
                final_episodes=2,
                parallel_envs=2,
                max_episode_drops=2,
            ),
            analysis=replace(
                resolved.analysis,
                transition_sample_size=16,
                transition_chunk_size=4,
                trajectory_episodes=2,
            ),
            autoscale=replace(resolved.autoscale, enabled=False),
        )
    return resolved


def main():
    args = parse_args()
    config = resolve_config(args)
    trainer = BaselineTrainer(config, project_root=PROJECT_ROOT)

    def stop_handler(signum, _frame):
        trainer.request_stop(f'signal_{signum}')

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, stop_handler)
    if args.resume is not None:
        trainer.resume(args.resume)
    result = trainer.run(
        final_evaluation=not args.skip_final_evaluation
    )
    print(result)


if __name__ == '__main__':
    main()
