"""用真实短训练自动标定正式训练的环境数与 learner batch。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from daxigua.rl.config import TrainingConfig
from daxigua.rl.trainer import BaselineTrainer


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / 'configs'
    / 'sab-full-fall-edge18-128m-b16m-r1.toml'
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / 'runs' / 'autotune' / 'real_training_calibration.json'
)
STAGE_METRICS = (
    'actor_seconds',
    'physics_seconds',
    'reward_seconds',
    'learner_seconds',
    'decision_data_seconds',
    'branch_seconds',
)


def _round_multiple(value, multiple):
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def environment_bracket(initial, minimum, maximum, *, multiple=128):
    """生成三个便宜的初始括区点，不假定吞吐严格单调。"""

    minimum = _round_multiple(minimum, multiple)
    maximum = _round_multiple(maximum, multiple)
    initial = min(maximum, max(minimum, _round_multiple(initial, multiple)))
    lower = min(maximum, max(
        minimum, _round_multiple(initial * 0.75, multiple)
    ))
    higher = min(maximum, max(
        minimum, _round_multiple(initial * 1.25, multiple)
    ))
    return tuple(dict.fromkeys((initial, lower, higher)))


def _branch_rate(config):
    branch = config.branch_learning
    horizon = config.total_transitions - branch.start_transition
    if not branch.enabled or horizon <= 0:
        raise ValueError('calibration requires an enabled branch phase')
    return branch.transition_budget / horizon


def build_trial_config(
        base,
        *,
        run_dir,
        envs,
        batch_size,
        parent_steps,
        branch_steps,
        compile_model=None):
    """保留正式结构和容量，仅缩短预算并显式覆盖训练阶段。"""

    envs = int(envs)
    batch_size = int(batch_size)
    warmup = max(4096, batch_size * 4)
    warmup = min(base.replay.capacity, warmup)
    if warmup < batch_size:
        raise ValueError('calibration warmup cannot fit learner batch')
    parent_transitions = envs * int(parent_steps)
    branch_parent_transitions = envs * int(branch_steps)
    branch_start = warmup + parent_transitions
    simulator_batch = base.branch_learning.simulator_batch_size
    branch_budget = int(
        branch_parent_transitions * _branch_rate(base)
        // simulator_batch
        * simulator_batch
    )
    branch_budget = max(simulator_batch, branch_budget)
    total_transitions = branch_start + branch_parent_transitions
    forced_epsilon = base.dqn.epsilon_end
    dqn = replace(
        base.dqn,
        epsilon_start=forced_epsilon,
        epsilon_end=forced_epsilon,
        epsilon_schedule=(
            (0, forced_epsilon),
            (total_transitions, forced_epsilon),
        ),
        compile_model=(
            base.dqn.compile_model
            if compile_model is None else bool(compile_model)
        ),
    )
    return replace(
        base,
        run_dir=str(run_dir),
        max_envs=envs,
        active_envs=envs,
        total_transitions=total_transitions,
        max_wall_seconds=0.0,
        finalization_reserve_seconds=0.0,
        log_interval_seconds=2.0,
        checkpoint_interval_seconds=1_000_000_000.0,
        stage_pilot_envs=min(base.stage_pilot_envs, envs),
        stage_pilot_max_drops=min(300, base.stage_pilot_max_drops),
        stage_pilot_policy_epsilon=forced_epsilon,
        dqn=dqn,
        replay=replace(
            base.replay,
            batch_size=batch_size,
            warmup_transitions=warmup,
        ),
        branch_learning=replace(
            base.branch_learning,
            transition_budget=branch_budget,
            start_transition=branch_start,
        ),
        evaluation=replace(
            base.evaluation,
            fast_interval_transitions=total_transitions * 2,
            accurate_milestones=(),
        ),
        analysis=replace(
            base.analysis,
            transition_sample_size=0,
            trajectory_episodes=0,
            critical_event_episodes=0,
        ),
        decision_data=replace(base.decision_data, enabled=False),
        dashboard=replace(
            base.dashboard,
            enabled=False,
            curve_snapshot_enabled=False,
        ),
        autoscale=replace(base.autoscale, enabled=False),
    )


def _read_jsonl(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _stable_rows(rows):
    # 首个窗口包含编译或新联合图首次执行，后续窗口才代表稳态。
    return rows[1:] if len(rows) > 1 else rows


def summarize_phase(rows):
    rows = _stable_rows(rows)
    if not rows:
        raise RuntimeError('training phase produced no measurable window')
    speed = statistics.median(
        float(row['env_steps_per_second']) for row in rows
    )
    durations = [float(row['metric_window_seconds']) for row in rows]
    elapsed = max(sum(durations), 1e-9)
    stage_fractions = {
        name.removesuffix('_seconds') + '_fraction': (
            sum(float(row.get(name, 0.0)) for row in rows) / elapsed
        )
        for name in STAGE_METRICS
    }
    return {
        'window_count': len(rows),
        'parent_transitions_per_second': speed,
        'updates_per_second': statistics.median(
            float(row.get('updates_per_second', 0.0)) for row in rows
        ),
        'learner_samples_per_second': statistics.median(
            float(row.get('learner_samples_per_second', 0.0))
            for row in rows
        ),
        **stage_fractions,
    }


def summarize_trial_metrics(rows, branch_start):
    parent_rows = [
        row for row in rows
        if int(row.get('transitions', 0)) <= int(branch_start)
        and int(row.get('branch_transitions', 0)) == 0
    ]
    branch_rows = [
        row for row in rows
        if int(row.get('transitions', 0)) > int(branch_start)
        and float(row.get('branch_steps_per_second', 0.0)) > 0.0
        and float(row.get('branch_sample_fraction', 0.0)) > 0.0
    ]
    return {
        'parent_only': summarize_phase(parent_rows),
        'branch_active': summarize_phase(branch_rows),
    }


def _memory_fraction(result):
    values = []
    total = float(result.get('total_memory_mb') or 0.0)
    if total > 0.0:
        values.append(
            float(result.get('peak_memory_reserved_mb') or 0.0) / total
        )
        values.append(
            float(result.get('external_peak_memory_used_mb') or 0.0) / total
        )
    return max(values, default=0.0)


def eligible_trials(results, *, memory_limit=0.85):
    return [
        result for result in results
        if result.get('status') == 'ok'
        and result.get('phases', {}).get('branch_active')
        and _memory_fraction(result) <= float(memory_limit)
    ]


def select_recommendation(
        results, *, memory_limit=0.85, throughput_tolerance=0.20):
    eligible = eligible_trials(results, memory_limit=memory_limit)
    if not eligible:
        raise RuntimeError('no successful trial stayed inside VRAM headroom')
    fastest = max(
        result['phases']['branch_active'][
            'parent_transitions_per_second'
        ]
        for result in eligible
    )
    near_fastest = [
        result for result in eligible
        if result['phases']['branch_active'][
            'parent_transitions_per_second'
        ] >= fastest * (1.0 - float(throughput_tolerance))
    ]
    selected = min(
        near_fastest,
        key=lambda result: (
            _memory_fraction(result),
            int(result['envs']),
            int(result['batch_size']),
            -result['phases']['branch_active'][
                'parent_transitions_per_second'
            ],
        ),
    )
    return selected, fastest


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _run_worker(args):
    started = time.perf_counter()
    base = TrainingConfig.from_toml(args.config)
    config = build_trial_config(
        base,
        run_dir=args.run_dir,
        envs=args.envs,
        batch_size=args.batch_size,
        parent_steps=args.parent_steps,
        branch_steps=args.branch_steps,
        compile_model=not args.worker_disable_compile,
    )
    device = torch.device(args.device)
    config = replace(config, device=str(device))
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('real training calibration requires CUDA')
    torch.cuda.reset_peak_memory_stats(device)
    trainer = BaselineTrainer(config, project_root=PROJECT_ROOT)
    teacher = trainer.load_stage_pilot_checkpoint(args.prewarm_checkpoint)

    prewarm_started = time.perf_counter()
    trainer.estimate_stage_thresholds()
    trainer.stagger_initial_states(
        donor_envs=min(args.donor_envs, config.active_envs)
    )
    trainer.release_stage_pilot_model()
    trainer.fill_warmup_replay()
    prewarm_seconds = time.perf_counter() - prewarm_started
    # 标定只缩短旁路Replay的启动等待；实际分支batch、loss和容量均不变。
    trainer.branch_replay_training_threshold = (
        config.branch_learning.learner_batch_size
    )
    training_started = time.perf_counter()
    progress = trainer.run(
        final_evaluation=False,
        finalize_artifacts=False,
    )
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    phases = summarize_trial_metrics(
        _read_jsonl(Path(config.run_dir) / 'metrics.jsonl'),
        config.branch_learning.start_transition,
    )
    properties = torch.cuda.get_device_properties(device)
    return {
        'status': 'ok',
        'envs': config.active_envs,
        'batch_size': config.replay.batch_size,
        'compile_model': config.dqn.compile_model,
        'use_bfloat16': config.dqn.use_bfloat16,
        'forced_epsilon': config.dqn.epsilon_end,
        'parent_steps': args.parent_steps,
        'branch_steps': args.branch_steps,
        'branch_rate': _branch_rate(base),
        'branch_budget': config.branch_learning.transition_budget,
        'branch_replay_training_threshold': (
            trainer.branch_replay_training_threshold
        ),
        'formal_branch_replay_warmup': (
            base.branch_learning.replay_warmup
        ),
        'stage_pilot_max_drops': config.stage_pilot_max_drops,
        'stage_pilot_policy_epsilon': (
            config.stage_pilot_policy_epsilon
        ),
        'donor_envs': min(args.donor_envs, config.active_envs),
        'prewarm_seconds': prewarm_seconds,
        'training_seconds': training_seconds,
        'wall_seconds': time.perf_counter() - started,
        'peak_memory_allocated_mb': (
            torch.cuda.max_memory_allocated(device) / 1024 ** 2
        ),
        'peak_memory_reserved_mb': (
            torch.cuda.max_memory_reserved(device) / 1024 ** 2
        ),
        'total_memory_mb': properties.total_memory / 1024 ** 2,
        'device_name': torch.cuda.get_device_name(device),
        'replay_capacity': config.replay.capacity,
        'branch_replay_capacity': config.branch_learning.replay_capacity,
        'phases': phases,
        'progress': progress,
        'stage_pilot_teacher': teacher,
    }


def _nvidia_sample():
    try:
        completed = subprocess.run(
            (
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits',
            ),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        first = completed.stdout.strip().splitlines()[0]
        utilization, used, total = (
            float(value.strip()) for value in first.split(',')
        )
        return {
            'timestamp': time.time(),
            'gpu_utilization': utilization,
            'gpu_memory_used_mb': used,
            'gpu_memory_total_mb': total,
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _trial_command(args, *, envs, batch_size, index, compile_model):
    trial_root = (
        args.output.parent / 'real_training_trials' / args.session_id
    )
    run_dir = trial_root / f'candidate_{index:02d}_e{envs}_b{batch_size}'
    result_path = run_dir / 'candidate_result.json'
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        '--worker',
        '--config', str(args.config),
        '--prewarm-checkpoint', str(args.prewarm_checkpoint),
        '--device', args.device,
        '--run-dir', str(run_dir),
        '--worker-output', str(result_path),
        '--envs', str(envs),
        '--batch-size', str(batch_size),
        '--parent-steps', str(args.parent_steps),
        '--branch-steps', str(args.branch_steps),
        '--donor-envs', str(args.donor_envs),
    ]
    if not compile_model:
        command.append('--worker-disable-compile')
    return command, run_dir, result_path


def _execute_trial(args, *, envs, batch_size, index, compile_model):
    command, run_dir, result_path = _trial_command(
        args,
        envs=envs,
        batch_size=batch_size,
        index=index,
        compile_model=compile_model,
    )
    print(
        f'[calibration] start candidate {index}: '
        f'envs={envs}, batch={batch_size}, compile={compile_model}',
        flush=True,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / 'candidate.log'
    samples = []
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + args.candidate_timeout
        timed_out = False
        while process.poll() is None:
            if time.monotonic() >= deadline:
                timed_out = True
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=20)
                break
            sample = _nvidia_sample()
            if sample is not None:
                samples.append(sample)
            time.sleep(args.telemetry_interval)
    if timed_out:
        result = {
            'status': 'failed',
            'error_type': 'CandidateTimeout',
            'message': (
                f'candidate exceeded {args.candidate_timeout:.0f} seconds'
            ),
        }
    elif result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
    else:
        result = {
            'status': 'failed',
            'error_type': 'WorkerExitedWithoutReport',
            'message': f'worker exit code {process.returncode}',
        }
    result.update({
        'envs': envs,
        'batch_size': batch_size,
        'compile_model': compile_model,
        'returncode': process.returncode,
        'run_dir': str(run_dir),
        'log_path': str(log_path),
        'external_telemetry_samples': len(samples),
        'external_gpu_utilization_median': (
            statistics.median(
                sample['gpu_utilization'] for sample in samples
            ) if samples else None
        ),
        'external_peak_memory_used_mb': (
            max(sample['gpu_memory_used_mb'] for sample in samples)
            if samples else None
        ),
    })
    print(
        f'[calibration] finish candidate {index}: '
        f'status={result.get("status")}, '
        f'branch_speed='
        f'{result.get("phases", {}).get("branch_active", {}).get("parent_transitions_per_second")}',
        flush=True,
    )
    return result


def _worker_main(args):
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = _run_worker(args)
    except BaseException as error:
        result = {
            'status': 'failed',
            'error_type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
        }
    args.worker_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result['status'] != 'ok':
        raise SystemExit(1)


def _compile_failure(result):
    message = (
        str(result.get('message', ''))
        + ' '
        + str(result.get('traceback', ''))
    ).lower()
    return any(word in message for word in ('compile', 'inductor', 'triton'))


def _write_report(args, results, *, complete, compile_model, selected=None):
    successful = [item for item in results if item.get('status') == 'ok']
    report = {
        'format_version': 1,
        'created_at': time.time(),
        'complete': bool(complete),
        'selection_metric': (
            'branch_active_parent_transitions_per_second'
        ),
        'throughput_tolerance': args.throughput_tolerance,
        'memory_limit': args.memory_limit,
        'config': str(Path(args.config).resolve()),
        'prewarm_checkpoint': str(Path(args.prewarm_checkpoint).resolve()),
        'prewarm_checkpoint_sha256': args.prewarm_checkpoint_sha256,
        'execution_compile_model': bool(compile_model),
        'candidate_count': len(results),
        'results': results,
    }
    if selected is not None:
        report.update({
            'selected_num_envs': int(selected['envs']),
            'selected_batch_size': int(selected['batch_size']),
            'selected_compile_model': bool(selected['compile_model']),
            'selected_use_bfloat16': bool(selected['use_bfloat16']),
            'selected_replay_capacity': int(selected['replay_capacity']),
            'maximum_successful_envs': max(
                int(item['envs']) for item in successful
            ),
            'selected_memory_fraction': _memory_fraction(selected),
            'selected_parent_only_speed': selected['phases'][
                'parent_only'
            ]['parent_transitions_per_second'],
            'selected_branch_active_speed': selected['phases'][
                'branch_active'
            ]['parent_transitions_per_second'],
            'recommended_launch_argv': [
                sys.executable,
                str(PROJECT_ROOT / 'tools' / 'run_autotuned_training.py'),
                '--config', str(Path(args.config).resolve()),
                '--autotune-report', str(Path(args.output).resolve()),
                '--prewarm-checkpoint',
                str(Path(args.prewarm_checkpoint).resolve()),
            ],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return report


def _orchestrator_main(args):
    base = TrainingConfig.from_toml(args.config)
    if not base.branch_learning.enabled:
        raise SystemExit('the selected formal config must enable branch learning')
    if not args.prewarm_checkpoint.is_file():
        raise SystemExit(f'checkpoint not found: {args.prewarm_checkpoint}')
    args.prewarm_checkpoint_sha256 = _sha256(args.prewarm_checkpoint)
    args.session_id = time.strftime('%Y%m%d-%H%M%S')
    initial = args.initial_envs or base.active_envs
    minimum = args.min_envs or max(256, initial // 2)
    maximum = args.max_envs or initial * 2
    compile_model = base.dqn.compile_model
    env_candidates = list(environment_bracket(initial, minimum, maximum))
    results = []
    tested = set()

    def run(envs, batch_size):
        nonlocal compile_model
        key = (int(envs), int(batch_size), bool(compile_model))
        if key in tested:
            return None
        tested.add(key)
        result = _execute_trial(
            args,
            envs=int(envs),
            batch_size=int(batch_size),
            index=len(results),
            compile_model=compile_model,
        )
        results.append(result)
        if (
                len(results) == 1
                and result.get('status') != 'ok'
                and compile_model
                and _compile_failure(result)):
            compile_model = False
            return run(envs, batch_size)
        _write_report(
            args, results, complete=False, compile_model=compile_model
        )
        return result

    for envs in env_candidates:
        run(envs, base.replay.batch_size)
    eligible = eligible_trials(results, memory_limit=args.memory_limit)
    if not eligible:
        _write_report(
            args, results, complete=True, compile_model=compile_model
        )
        raise SystemExit('no environment candidate completed inside VRAM limit')
    fastest_env = max(
        eligible,
        key=lambda item: item['phases']['branch_active'][
            'parent_transitions_per_second'
        ],
    )['envs']
    low = min(env_candidates)
    high = max(env_candidates)
    extra_env = None
    if fastest_env == high and high < maximum:
        extra_env = min(maximum, _round_multiple(initial * 1.5, 128))
    elif fastest_env == low and low > minimum:
        extra_env = max(minimum, _round_multiple(initial * 0.5, 128))
    if extra_env is not None:
        run(extra_env, base.replay.batch_size)

    selected_env_trial, _ = select_recommendation(
        results,
        memory_limit=args.memory_limit,
        throughput_tolerance=args.throughput_tolerance,
    )
    alternate_batch = _round_multiple(base.replay.batch_size * 1.5, 64)
    if alternate_batch != base.replay.batch_size:
        run(selected_env_trial['envs'], alternate_batch)
    selected, fastest = select_recommendation(
        results,
        memory_limit=args.memory_limit,
        throughput_tolerance=args.throughput_tolerance,
    )
    report = _write_report(
        args,
        results,
        complete=True,
        compile_model=compile_model,
        selected=selected,
    )
    report['fastest_eligible_branch_speed'] = fastest
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps({
        'output': str(args.output),
        'selected_num_envs': report['selected_num_envs'],
        'selected_batch_size': report['selected_batch_size'],
        'selected_branch_active_speed': (
            report['selected_branch_active_speed']
        ),
        'selected_memory_fraction': report['selected_memory_fraction'],
        'candidate_count': report['candidate_count'],
    }, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--prewarm-checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--initial-envs', type=int)
    parser.add_argument('--min-envs', type=int)
    parser.add_argument('--max-envs', type=int)
    parser.add_argument('--parent-steps', type=int, default=32)
    parser.add_argument('--branch-steps', type=int, default=48)
    parser.add_argument('--donor-envs', type=int, default=256)
    parser.add_argument('--memory-limit', type=float, default=0.85)
    parser.add_argument('--throughput-tolerance', type=float, default=0.20)
    parser.add_argument('--telemetry-interval', type=float, default=2.0)
    parser.add_argument('--candidate-timeout', type=float, default=300.0)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--envs', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--batch-size', type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        '--worker-disable-compile', action='store_true', help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    for name in ('parent_steps', 'branch_steps', 'donor_envs'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if not 0.0 < args.memory_limit <= 1.0:
        parser.error('--memory-limit must be in (0, 1]')
    if not 0.0 <= args.throughput_tolerance < 1.0:
        parser.error('--throughput-tolerance must be in [0, 1)')
    if args.telemetry_interval <= 0.0:
        parser.error('--telemetry-interval must be positive')
    if args.candidate_timeout <= 0.0:
        parser.error('--candidate-timeout must be positive')
    if args.worker:
        for name in ('worker_output', 'run_dir', 'envs', 'batch_size'):
            if getattr(args, name) is None:
                parser.error(f'worker mode requires --{name.replace("_", "-")}')
    return args


def main():
    args = parse_args()
    if args.worker:
        _worker_main(args)
    else:
        _orchestrator_main(args)


if __name__ == '__main__':
    main()
