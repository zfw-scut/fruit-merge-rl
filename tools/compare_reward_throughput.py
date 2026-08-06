"""在选定训练规模上成对测量Reward V2相对分数奖励的吞吐损耗。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_TOOL = PROJECT_ROOT / 'tools' / 'benchmark_training_pipeline.py'


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pipeline-report',
        type=Path,
        default=PROJECT_ROOT / 'runs' / 'autotune' / 'training_pipeline.json',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'runs' / 'autotune' / 'reward_overhead.json',
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--measured-steps', type=int, default=5)
    parser.add_argument('--pre-roll-steps', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260806)
    parser.add_argument('--reward-scale', type=float, default=1.0)
    parser.add_argument('--warning-loss', type=float, default=0.05)
    parser.add_argument('--maximum-loss', type=float, default=0.08)
    return parser.parse_args(argv)


def _run_candidate(args, selected, reward_kind, repeat_index):
    name = f'{reward_kind}_{repeat_index + 1:02d}'
    output = args.output.parent / f'reward_overhead_{name}.json'
    log_path = args.output.parent / f'reward_overhead_{name}.log'
    command = [
        sys.executable,
        str(BENCHMARK_TOOL),
        '--single-envs', str(selected['num_envs']),
        '--single-batch-size', str(selected['batch_size']),
        '--measured-steps', str(args.measured_steps),
        '--pre-roll-steps', str(args.pre_roll_steps),
        '--device', args.device,
        '--reward-kind', reward_kind,
        '--reward-scale', str(args.reward_scale),
        '--seed', str(args.seed + repeat_index),
        '--output', str(output),
    ]
    if selected['compile_model']:
        command.append('--compile-model')
    if not selected['use_bfloat16']:
        command.append('--fp32')
    with log_path.open('w', encoding='utf-8') as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f'{reward_kind}第{repeat_index + 1}次性能测量失败：{log_path}'
        )
    result = json.loads(output.read_text(encoding='utf-8'))
    result['log_path'] = str(log_path)
    return result


def _status(overhead_fraction, warning_loss, maximum_loss):
    if overhead_fraction > maximum_loss:
        return 'blocked'
    if overhead_fraction > warning_loss:
        return 'warning'
    return 'passed'


def main(argv=None):
    args = parse_args(argv)
    if args.repeats <= 0 or args.measured_steps <= 0:
        raise SystemExit('repeats and measured-steps must be positive')
    if not 0.0 <= args.warning_loss <= args.maximum_loss < 1.0:
        raise SystemExit('loss thresholds must satisfy 0 <= warning <= maximum < 1')
    pipeline = json.loads(args.pipeline_report.read_text(encoding='utf-8'))
    selected = {
        'num_envs': int(pipeline['selected_num_envs']),
        'batch_size': int(pipeline['selected_batch_size']),
        'compile_model': bool(pipeline['selected_compile_model']),
        'use_bfloat16': bool(pipeline['selected_use_bfloat16']),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = {'score_v1': [], 'spatial_v2': []}
    # 每个repeat使用同一随机种子配对，顺序交替以减小温度等系统漂移。
    for repeat_index in range(args.repeats):
        order = (
            ('score_v1', 'spatial_v2')
            if repeat_index % 2 == 0
            else ('spatial_v2', 'score_v1')
        )
        for reward_kind in order:
            results[reward_kind].append(_run_candidate(
                args, selected, reward_kind, repeat_index
            ))

    score_speed = statistics.median(
        item['end_to_end_env_steps_per_second']
        for item in results['score_v1']
    )
    spatial_speed = statistics.median(
        item['end_to_end_env_steps_per_second']
        for item in results['spatial_v2']
    )
    throughput_ratio = spatial_speed / max(score_speed, 1e-9)
    overhead_fraction = 1.0 - throughput_ratio
    status = _status(
        overhead_fraction, args.warning_loss, args.maximum_loss
    )
    report = {
        'created_at': time.time(),
        'status': status,
        'selected': selected,
        'device': args.device,
        'repeats': args.repeats,
        'measured_steps': args.measured_steps,
        'warning_loss_fraction': args.warning_loss,
        'maximum_loss_fraction': args.maximum_loss,
        'score_v1_median_env_steps_per_second': score_speed,
        'spatial_v2_median_env_steps_per_second': spatial_speed,
        'spatial_to_score_throughput_ratio': throughput_ratio,
        'reward_overhead_fraction': overhead_fraction,
        'results': results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status == 'blocked':
        raise SystemExit(
            f'Reward V2吞吐损失超过{args.maximum_loss:.1%}门禁，'
            '禁止开始正式训练'
        )


if __name__ == '__main__':
    main()
