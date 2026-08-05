"""在独立子进程中测量环境数、batch 和完整训练管线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daxigua.rl.benchmark import benchmark_training_candidate


def _integer_list(value):
    return tuple(int(item.strip()) for item in value.split(',') if item.strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--candidate-envs', type=_integer_list,
        default=(1024, 2048, 4096, 8192, 16384, 32768),
    )
    parser.add_argument(
        '--batch-sizes', type=_integer_list, default=(256, 512, 1024, 2048)
    )
    parser.add_argument('--initial-batch-size', type=int, default=512)
    parser.add_argument('--measured-steps', type=int, default=3)
    parser.add_argument('--pre-roll-steps', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--compile-model', action='store_true')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--skip-execution-variants', action='store_true')
    parser.add_argument('--skip-profile', action='store_true')
    parser.add_argument('--profiler-output', type=Path)
    parser.add_argument(
        '--output', type=Path,
        default=PROJECT_ROOT / 'runs' / 'autotune' / 'training_pipeline.json',
    )
    parser.add_argument('--single-envs', type=int)
    parser.add_argument('--single-batch-size', type=int)
    return parser.parse_args()


def _run_single(args):
    result = benchmark_training_candidate(
        num_envs=args.single_envs,
        batch_size=args.single_batch_size,
        device=args.device,
        measured_steps=args.measured_steps,
        pre_roll_steps=args.pre_roll_steps,
        use_bfloat16=not args.fp32,
        compile_model=args.compile_model,
        profiler_output=args.profiler_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def _subprocess_result(
        args,
        envs,
        batch_size,
        index,
        *,
        compile_model=None,
        fp32=None,
        profiler_output=None,
        measured_steps=None):
    single_output = args.output.parent / f'candidate_{index:02d}.json'
    candidate_log = args.output.parent / f'candidate_{index:02d}.log'
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        '--single-envs', str(envs),
        '--single-batch-size', str(batch_size),
        '--measured-steps', str(
            args.measured_steps if measured_steps is None else measured_steps
        ),
        '--pre-roll-steps', str(args.pre_roll_steps),
        '--device', args.device,
        '--output', str(single_output),
    )
    compile_model = args.compile_model if compile_model is None else compile_model
    fp32 = args.fp32 if fp32 is None else fp32
    if compile_model:
        command += ('--compile-model',)
    if fp32:
        command += ('--fp32',)
    if profiler_output is not None:
        command += ('--profiler-output', str(profiler_output))
    with candidate_log.open('w', encoding='utf-8') as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        return {
            'num_envs': envs,
            'batch_size': batch_size,
            'status': 'failed',
            'returncode': completed.returncode,
            'log_path': str(candidate_log),
        }
    result = json.loads(single_output.read_text(encoding='utf-8'))
    result['status'] = 'ok'
    result['log_path'] = str(candidate_log)
    return result


def main():
    args = parse_args()
    if args.single_envs is not None:
        if args.single_batch_size is None:
            raise SystemExit('--single-batch-size is required with --single-envs')
        _run_single(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    index = 0
    previous_success = None
    consecutive_low_env_gains = 0
    for envs in args.candidate_envs:
        result = _subprocess_result(
            args, envs, args.initial_batch_size, index
        )
        results.append(result)
        index += 1
        if result['status'] != 'ok':
            break
        if (
            result.get('total_memory_mb')
            and result.get(
                'projected_peak_memory_mb',
                result['peak_memory_reserved_mb'],
            ) / result['total_memory_mb'] >= 0.85
        ):
            break
        if previous_success is not None:
            previous_speed = previous_success[
                'end_to_end_env_steps_per_second'
            ]
            gain = (
                result['end_to_end_env_steps_per_second']
                / max(previous_speed, 1e-9)
                - 1.0
            )
            result['gain_over_previous_env_candidate'] = gain
            consecutive_low_env_gains = (
                consecutive_low_env_gains + 1 if gain < 0.10 else 0
            )
        previous_success = result
        if consecutive_low_env_gains >= 2:
            break
    successful = [
        item for item in results
        if item['status'] == 'ok'
        and (
            not item.get('total_memory_mb')
            or item.get(
                'projected_peak_memory_mb',
                item['peak_memory_reserved_mb'],
            ) / item['total_memory_mb'] <= 0.85
        )
    ]
    if not successful:
        raise SystemExit('every environment candidate failed')
    best_environment = max(
        successful,
        key=lambda item: item['end_to_end_env_steps_per_second'],
    )['num_envs']
    for batch_size in args.batch_sizes:
        if batch_size == args.initial_batch_size:
            continue
        results.append(_subprocess_result(
            args, best_environment, batch_size, index
        ))
        index += 1
    initial_for_best_env = next(
        item for item in results
        if item['status'] == 'ok'
        and item['num_envs'] == best_environment
        and item['batch_size'] == args.initial_batch_size
    )
    initial_batch_speed = initial_for_best_env[
        'end_to_end_env_steps_per_second'
    ]
    eager_candidates = [
        item for item in results
        if item['status'] == 'ok'
        and item['num_envs'] == best_environment
        and not item.get('compile_model', False)
        and item.get('use_bfloat16', True)
        and (
            item['batch_size'] <= args.initial_batch_size
            or item['end_to_end_env_steps_per_second']
            >= initial_batch_speed * 1.15
        )
    ]
    eager_best = max(
        eager_candidates,
        key=lambda item: item['end_to_end_env_steps_per_second'],
    )
    if not args.skip_execution_variants and args.device.startswith('cuda'):
        results.append(_subprocess_result(
            args,
            eager_best['num_envs'],
            eager_best['batch_size'],
            index,
            compile_model=True,
            fp32=False,
        ))
        index += 1
        results.append(_subprocess_result(
            args,
            eager_best['num_envs'],
            eager_best['batch_size'],
            index,
            compile_model=False,
            fp32=True,
        ))
        index += 1
    eligible = [
        item for item in results
        if item['status'] == 'ok'
        and (
            not item.get('total_memory_mb')
            or item.get(
                'projected_peak_memory_mb',
                item['peak_memory_reserved_mb'],
            )
            / item['total_memory_mb'] <= 0.85
        )
        and not (
            item['num_envs'] == best_environment
            and item['batch_size'] > args.initial_batch_size
            and item.get('use_bfloat16', True)
            and not item.get('compile_model', False)
            and item['end_to_end_env_steps_per_second']
            < initial_batch_speed * 1.15
        )
    ]
    eager_reference_speed = eager_best['end_to_end_env_steps_per_second']
    eligible = [
        item for item in eligible
        if not item.get('compile_model', False)
        or item['end_to_end_env_steps_per_second']
        >= eager_reference_speed * 1.10
    ]
    if not eligible:
        raise SystemExit('all successful candidates exceeded memory headroom')
    best = max(
        eligible,
        key=lambda item: item['end_to_end_env_steps_per_second'],
    )
    profile_result = None
    if not args.skip_profile and args.device.startswith('cuda'):
        profile_result = _subprocess_result(
            args,
            best['num_envs'],
            best['batch_size'],
            index,
            compile_model=best.get('compile_model', False),
            fp32=not best.get('use_bfloat16', True),
            profiler_output=args.output.parent / 'selected_pipeline_trace.json',
            measured_steps=1,
        )
        index += 1
    report = {
        'created_at': time.time(),
        'selection_metric': 'end_to_end_env_steps_per_second_at_utd_1',
        'selected_num_envs': best['num_envs'],
        'selected_batch_size': best['batch_size'],
        'selected_compile_model': best.get('compile_model', False),
        'selected_use_bfloat16': best.get('use_bfloat16', True),
        'maximum_successful_envs': max(
            item['num_envs']
            for item in eligible
        ),
        'profile_result': profile_result,
        'results': results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
