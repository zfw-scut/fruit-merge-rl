"""连续采样GPU状态并标定堵塞风险事件采集环境数。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='从指定环境数起短跑标定风险事件生成吞吐。'
    )
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--start-envs', type=int, default=2000)
    parser.add_argument('--step-envs', type=int, default=500)
    parser.add_argument('--max-envs', type=int, default=4000)
    parser.add_argument('--trial-seconds', type=float, default=180.0)
    parser.add_argument('--episodes', type=int, default=200_000)
    parser.add_argument('--max-drops', type=int, default=300)
    parser.add_argument('--seed-base', type=int, default=93_000_000)
    parser.add_argument('--gpu-sample-seconds', type=float, default=1.0)
    return parser.parse_args(argv)


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    temporary.replace(path)


def _number(value):
    return float(value.strip().split()[0])


def _gpu_sample():
    output = subprocess.run(
        (
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits',
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
    utilization, memory, power = output.split(',')
    return {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'gpu_utilization_percent': _number(utilization),
        'memory_used_mib': _number(memory),
        'power_watts': _number(power),
    }


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = round((len(ordered) - 1) * float(fraction))
    return ordered[index]


def _gpu_summary(samples):
    if not samples:
        return {'samples': 0}
    result = {'samples': len(samples)}
    for key in (
            'gpu_utilization_percent', 'memory_used_mib', 'power_watts'):
        values = [float(sample[key]) for sample in samples]
        result[key] = {
            'mean': statistics.fmean(values),
            'median': statistics.median(values),
            'p10': _percentile(values, 0.1),
            'p90': _percentile(values, 0.9),
            'minimum': min(values),
            'maximum': max(values),
        }
    return result


def _load_manifest(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _candidate_command(args, envs, candidate_dir, seed_base):
    return (
        sys.executable,
        str(PROJECT_ROOT / 'tools' / 'generate_pair_risk_dataset.py'),
        'collect',
        '--output-dir', str(candidate_dir),
        '--device', str(args.device),
        '--parallel-envs', str(envs),
        '--episodes', str(args.episodes),
        '--target-confirmed-events', '1000000000',
        '--seed-base', str(seed_base),
        '--max-drops', str(args.max_drops),
        '--max-wall-seconds', str(args.trial_seconds),
        '--compile-model',
        '--no-auto-finalize',
        '--progress-interval-seconds', '10',
    )


def run_candidate(args, envs, index, root):
    candidate_dir = root / f'envs_{envs}'
    if candidate_dir.exists():
        raise FileExistsError(f'candidate output already exists: {candidate_dir}')
    log_path = root / f'envs_{envs}.log'
    samples_path = root / f'envs_{envs}_gpu.jsonl'
    seed_base = int(args.seed_base) + index * 10_000_019
    command = _candidate_command(
        args, envs, candidate_dir, seed_base
    )
    samples = []
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        manifest_path = candidate_dir / 'manifest.json'
        while process.poll() is None:
            manifest = _load_manifest(manifest_path)
            if manifest is not None and manifest.get('status') == 'running':
                try:
                    sample = _gpu_sample()
                except (OSError, subprocess.SubprocessError, ValueError):
                    sample = None
                if sample is not None:
                    samples.append(sample)
                    with samples_path.open('a', encoding='utf-8') as output:
                        output.write(json.dumps(sample) + '\n')
            time.sleep(float(args.gpu_sample_seconds))
        return_code = process.wait()
    manifest = _load_manifest(candidate_dir / 'manifest.json') or {}
    elapsed = float(manifest.get('elapsed_seconds') or 0.0)
    confirmed = int(manifest.get('confirmed_events') or 0)
    result = {
        'parallel_envs': int(envs),
        'status': manifest.get('status') or 'process_failed',
        'return_code': int(return_code),
        'seed_base': seed_base,
        'transitions': int(manifest.get('transitions') or 0),
        'confirmed_events': confirmed,
        'confirmed_events_by_level': (
            manifest.get('confirmed_events_by_level') or {}
        ),
        'elapsed_seconds': elapsed,
        'env_steps_per_second': float(
            manifest.get('env_steps_per_second') or 0.0
        ),
        'confirmed_events_per_second': (
            confirmed / elapsed if elapsed > 0.0 else 0.0
        ),
        'completed_episodes': int(manifest.get('completed_episodes') or 0),
        'peak_cuda_allocated_bytes': manifest.get(
            'peak_cuda_allocated_bytes'
        ),
        'gpu': _gpu_summary(samples),
        'candidate_dir': str(candidate_dir),
        'log': str(log_path),
    }
    return result


def benchmark(args):
    if args.start_envs <= 0 or args.step_envs <= 0:
        raise ValueError('environment counts must be positive')
    if args.max_envs < args.start_envs:
        raise ValueError('max-envs must not be below start-envs')
    if args.trial_seconds <= 0.0 or args.gpu_sample_seconds <= 0.0:
        raise ValueError('benchmark durations must be positive')
    root = Path(args.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'benchmark output is not empty: {root}')
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    report_path = root / 'report.json'
    for index, envs in enumerate(range(
            int(args.start_envs),
            int(args.max_envs) + 1,
            int(args.step_envs))):
        row = run_candidate(args, envs, index, root)
        rows.append(row)
        valid = [
            item for item in rows
            if item['return_code'] == 0 and item['confirmed_events'] > 0
        ]
        selected = (
            max(valid, key=lambda item: item['confirmed_events_per_second'])
            if valid else None
        )
        report = {
            'format_version': 1,
            'purpose': 'pair_risk_collection_env_calibration',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'parameters': {
                'start_envs': args.start_envs,
                'step_envs': args.step_envs,
                'max_envs': args.max_envs,
                'trial_seconds': args.trial_seconds,
                'max_drops': args.max_drops,
                'gpu_sample_seconds': args.gpu_sample_seconds,
            },
            'results': rows,
            'selected_parallel_envs': (
                selected['parallel_envs'] if selected is not None else None
            ),
            'selection_metric': 'confirmed_events_per_second',
        }
        _atomic_json(report_path, report)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None):
    benchmark(parse_args(argv))


if __name__ == '__main__':
    main()
