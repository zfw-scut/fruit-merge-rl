"""串行运行两组等训练量的 5 层 GNN epsilon 对照实验。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_status(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    os.replace(temporary, path)


def _run_one(name, config, run_dir, log_path, args, status_path, started_at):
    command = [
        sys.executable,
        str(PROJECT_ROOT / 'tools' / 'train_gnn_dqn.py'),
        '--config', str(config),
        '--run-dir', str(run_dir),
        '--max-envs', str(args.envs),
        '--active-envs', str(args.envs),
        '--batch-size', str(args.batch_size),
        '--dashboard-port', str(args.dashboard_port),
    ]
    _write_status(status_path, {
        'state': 'running',
        'current_experiment': name,
        'started_at': started_at,
        'experiment_started_at': time.time(),
        'run_dir': str(run_dir),
        'command': command,
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', buffering=1) as log:
        log.write(f'\n[{time.strftime("%Y-%m-%d %H:%M:%S")}] 启动 {name}\n')
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f'{name} 训练失败，退出码 {result.returncode}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-root', type=Path,
        default=PROJECT_ROOT / 'runs' / 'gnn_l5_epsilon_ablation_24m',
    )
    parser.add_argument('--envs', type=int, default=1792)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--dashboard-port', type=int, default=8765)
    parser.add_argument('--no-final-dashboard', action='store_true')
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / 'ablation_status.json'
    started_at = time.time()
    experiments = (
        (
            'fast_epsilon',
            PROJECT_ROOT / 'configs' / 'gnn_dqn_baseline_l5_fast_24m.toml',
            output_root / 'fast_epsilon',
        ),
        (
            'slow_epsilon',
            PROJECT_ROOT / 'configs' / 'gnn_dqn_baseline_l5_slow_24m.toml',
            output_root / 'slow_epsilon',
        ),
    )
    completed = []
    latest_run = None
    try:
        for name, config, run_dir in experiments:
            latest_run = run_dir
            _run_one(
                name,
                config,
                run_dir,
                output_root / f'{name}.log',
                args,
                status_path,
                started_at,
            )
            completed.append(name)
        _write_status(status_path, {
            'state': 'completed',
            'completed_experiments': completed,
            'started_at': started_at,
            'completed_at': time.time(),
            'fast_run_dir': str(experiments[0][2]),
            'slow_run_dir': str(experiments[1][2]),
            'dashboard_port': args.dashboard_port,
        })
    except BaseException as error:
        _write_status(status_path, {
            'state': 'failed',
            'completed_experiments': completed,
            'started_at': started_at,
            'failed_at': time.time(),
            'error_type': type(error).__name__,
            'message': str(error),
            'latest_run_dir': str(latest_run) if latest_run else None,
        })
        raise
    if not args.no_final_dashboard:
        subprocess.run([
            sys.executable,
            str(PROJECT_ROOT / 'tools' / 'serve_training_dashboard.py'),
            '--run-dir', str(experiments[-1][2]),
            '--port', str(args.dashboard_port),
        ], cwd=PROJECT_ROOT, check=False)


if __name__ == '__main__':
    main()
