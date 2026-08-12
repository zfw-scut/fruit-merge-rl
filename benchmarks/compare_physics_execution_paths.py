"""固定种子和动作序列，对比当前训练与场景实验室执行路径。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from daxigua.simulator import (
    PHYSICS_IDENTITY,
    SimulatorConfig,
    TensorVectorSimulator,
)


STATE_FIELDS = (
    'positions',
    'velocities',
    'angles',
    'angular_velocities',
    'levels',
    'fruit_ids',
    'active',
    'fruit_queue',
    'score',
    'step_count',
    'physics_frame',
    'fail_frames',
    'next_fruit_id',
    'rng_state',
    'needs_reset',
)
DISCRETE_TRAJECTORY_FIELDS = (
    'levels',
    'fruit_ids',
    'active',
    'fruit_queue',
    'score',
    'step_count',
    'next_fruit_id',
    'rng_state',
    'needs_reset',
)
FLOAT_FIELDS = (
    'positions',
    'velocities',
    'angles',
    'angular_velocities',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260813)
    parser.add_argument('--seed-count', type=int, default=8)
    parser.add_argument('--seed-stride', type=int, default=1009)
    parser.add_argument('--max-drops', type=int, default=200)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--paths',
        choices=('all', 'same-fps-only', 'current-defaults-only'),
        default='all',
    )
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def make_config(physics_fps, *, device):
    factory = (
        SimulatorConfig.training_fast
        if physics_fps == 30
        else SimulatorConfig.high_fidelity_fast
    )
    return factory(
        max_fruits=64,
        action_count=21,
        queue_length=4,
        use_cuda_extension=torch.device(device).type == 'cuda',
        track_action_effects=False,
    )


def snapshot(simulator):
    return {
        name: getattr(simulator, name).detach().cpu().clone()
        for name in STATE_FIELDS
    }


def run_episodes(
        *, seeds, actions, config, device, execution, record_trajectory=True):
    num_envs = len(seeds)
    simulator = TensorVectorSimulator(num_envs, config=config, device=device)
    simulator.reset(
        seeds=torch.tensor(seeds, dtype=torch.int64, device=device)
    )
    steps = [[] for _ in seeds]
    total_fast_forwarded_frames = [0 for _ in seeds]
    total_frames = [0 for _ in seeds]
    settle_timeouts = [0 for _ in seeds]
    enabled = torch.ones(num_envs, dtype=torch.bool, device=device)

    for drop_index in range(len(actions[0])):
        action_tensor = torch.tensor(
            [row[drop_index] for row in actions],
            dtype=torch.int64,
            device=simulator.device,
        )
        score_before = simulator.score.detach().cpu().clone()
        if execution == 'full_step':
            if num_envs == 1:
                physics = simulator.step(action_tensor).physics
            else:
                physics = simulator.step_masked(action_tensor, enabled).physics
            frames = physics.frames_simulated.detach().cpu()
            fast_forwarded = physics.fast_forwarded_frames.detach().cpu()
            stable = physics.stable.detach().cpu()
            done = physics.done.detach().cpu()
            settle_timeout = physics.settle_timeout.detach().cpu()
        elif execution == 'incremental_until_stable':
            safe_actions = torch.where(enabled, action_tensor, 0)
            if device.type == 'cuda' and config.use_cuda_extension:
                simulator._step_cuda_extension(
                    safe_actions,
                    enabled_mask=enabled,
                    perform_drop=True,
                    physics_frame_budget=0,
                )
            else:
                if not bool(enabled.all().item()):
                    raise RuntimeError(
                        'batched masked incremental audit requires CUDA'
                    )
                simulator.begin_incremental_action(safe_actions)
            frames = torch.zeros(
                num_envs, dtype=torch.int64, device=device
            )
            stable = torch.zeros(
                num_envs, dtype=torch.bool, device=device
            )
            done = torch.zeros_like(stable)
            running = enabled.clone()
            for _ in range(config.max_physics_frames):
                if not bool(running.any().item()):
                    break
                if device.type == 'cuda' and config.use_cuda_extension:
                    physics = simulator._step_cuda_extension(
                        torch.zeros_like(action_tensor),
                        enabled_mask=running,
                        perform_drop=False,
                        physics_frame_budget=1,
                    ).physics
                else:
                    physics = simulator.advance_incremental_frame()
                frames += running.to(torch.int64)
                stable |= physics.stable & running
                done |= physics.done & running
                running &= ~stable & ~done
            settle_timeout = enabled & ~stable & ~done & (
                frames == config.max_physics_frames
            )
            frames = frames.detach().cpu()
            fast_forwarded = torch.zeros_like(frames)
            stable = stable.detach().cpu()
            done = done.detach().cpu()
            settle_timeout = settle_timeout.detach().cpu()
        else:
            raise ValueError(f'unknown execution path: {execution}')

        batch_state = snapshot(simulator) if record_trajectory else None
        score_after = simulator.score.detach().cpu()
        needs_reset = simulator.needs_reset.detach().cpu()
        for env_index in range(num_envs):
            if not bool(enabled[env_index].item()):
                continue
            env_frames = int(frames[env_index].item())
            env_fast_forwarded = int(fast_forwarded[env_index].item())
            env_settle_timeout = bool(settle_timeout[env_index].item())
            total_frames[env_index] += env_frames
            total_fast_forwarded_frames[env_index] += env_fast_forwarded
            settle_timeouts[env_index] += int(env_settle_timeout)
            if record_trajectory:
                steps[env_index].append({
                    'action': int(actions[env_index][drop_index]),
                    'frames': env_frames,
                    'fast_forwarded_frames': env_fast_forwarded,
                    'stable': bool(stable[env_index].item()),
                    'done': bool(done[env_index].item()),
                    'settle_timeout': env_settle_timeout,
                    'score_delta': int(
                        score_after[env_index].item()
                        - score_before[env_index].item()
                    ),
                    'state': {
                        name: value[env_index:env_index + 1].clone()
                        for name, value in batch_state.items()
                    },
                })
            if bool(needs_reset[env_index].item()):
                enabled[env_index] = False
        if not bool(enabled.any().item()):
            break

    episodes = []
    for env_index in range(num_envs):
        active_levels = simulator.levels[env_index][
            simulator.active[env_index]
        ]
        episodes.append({
            'steps': steps[env_index],
            'summary': {
                'score': int(simulator.score[env_index].item()),
                'drops': int(simulator.step_count[env_index].item()),
                'done': bool(simulator.needs_reset[env_index].item()),
                'fruit_count': int(
                    simulator.active[env_index].sum().item()
                ),
                'max_level': (
                    int(active_levels.max().item())
                    if int(active_levels.numel())
                    else 0
                ),
                'physics_frames': total_frames[env_index],
                'fast_forwarded_frames': (
                    total_fast_forwarded_frames[env_index]
                ),
                'settle_timeouts': settle_timeouts[env_index],
            },
        })
    return episodes


def max_abs_difference(left, right, field):
    left_value = left[field]
    right_value = right[field]
    if left_value.shape != right_value.shape:
        return float('inf')
    if not left_value.numel():
        return 0.0
    return float((left_value - right_value).abs().max().item())


def compare_episodes(left, right, *, float_tolerance=1e-4):
    common_steps = min(len(left['steps']), len(right['steps']))
    first_discrete_divergence = None
    first_float_divergence = None
    first_float_divergence_detail = None
    max_float_differences = {field: 0.0 for field in FLOAT_FIELDS}
    exact_state_steps = 0

    for step_index in range(common_steps):
        left_state = left['steps'][step_index]['state']
        right_state = right['steps'][step_index]['state']
        discrete_equal = all(
            torch.equal(left_state[field], right_state[field])
            for field in DISCRETE_TRAJECTORY_FIELDS
        )
        if not discrete_equal and first_discrete_divergence is None:
            first_discrete_divergence = step_index + 1

        step_float_differences = {
            field: max_abs_difference(left_state, right_state, field)
            for field in FLOAT_FIELDS
        }
        for field, difference in step_float_differences.items():
            max_float_differences[field] = max(
                max_float_differences[field], difference
            )
        floats_equal = all(
            difference <= float_tolerance
            for difference in step_float_differences.values()
        )
        if not floats_equal and first_float_divergence is None:
            first_float_divergence = step_index + 1
            first_float_divergence_detail = {
                'action': left['steps'][step_index]['action'],
                'frames_left': left['steps'][step_index]['frames'],
                'frames_right': right['steps'][step_index]['frames'],
                'fast_forwarded_frames_left': (
                    left['steps'][step_index]['fast_forwarded_frames']
                ),
                'fast_forwarded_frames_right': (
                    right['steps'][step_index]['fast_forwarded_frames']
                ),
                'max_abs_difference': step_float_differences,
            }
        if discrete_equal and all(
                difference == 0.0
                for difference in step_float_differences.values()):
            exact_state_steps += 1

    left_summary = left['summary']
    right_summary = right['summary']
    return {
        'common_steps': common_steps,
        'first_discrete_divergence_drop': first_discrete_divergence,
        'first_float_divergence_drop': first_float_divergence,
        'first_float_divergence_detail': first_float_divergence_detail,
        'exact_state_steps': exact_state_steps,
        'max_abs_difference': max_float_differences,
        'final_score_left': left_summary['score'],
        'final_score_right': right_summary['score'],
        'final_score_difference_right_minus_left': (
            right_summary['score'] - left_summary['score']
        ),
        'final_drops_left': left_summary['drops'],
        'final_drops_right': right_summary['drops'],
        'same_final_score': left_summary['score'] == right_summary['score'],
        'same_episode_length': left_summary['drops'] == right_summary['drops'],
    }


def aggregate_pair(seed_reports, pair_name):
    comparisons = [report['comparisons'][pair_name] for report in seed_reports]
    score_differences = [
        item['final_score_difference_right_minus_left'] for item in comparisons
    ]
    left_scores = [item['final_score_left'] for item in comparisons]
    right_scores = [item['final_score_right'] for item in comparisons]
    discrete_divergences = [
        item['first_discrete_divergence_drop']
        for item in comparisons
        if item['first_discrete_divergence_drop'] is not None
    ]
    float_divergences = [
        item['first_float_divergence_drop']
        for item in comparisons
        if item['first_float_divergence_drop'] is not None
    ]
    return {
        'seed_count': len(comparisons),
        'same_final_score_count': sum(
            item['same_final_score'] for item in comparisons
        ),
        'same_episode_length_count': sum(
            item['same_episode_length'] for item in comparisons
        ),
        'seeds_with_discrete_divergence': len(discrete_divergences),
        'seeds_with_float_divergence': len(float_divergences),
        'median_first_discrete_divergence_drop': (
            statistics.median(discrete_divergences)
            if discrete_divergences else None
        ),
        'median_first_float_divergence_drop': (
            statistics.median(float_divergences)
            if float_divergences else None
        ),
        'mean_score_difference_right_minus_left': statistics.mean(
            score_differences
        ),
        'mean_score_left': statistics.mean(left_scores),
        'mean_score_right': statistics.mean(right_scores),
        'mean_score_difference_percent_of_left': (
            100.0 * statistics.mean(score_differences)
            / statistics.mean(left_scores)
            if statistics.mean(left_scores) else None
        ),
        'mean_absolute_score_difference': statistics.mean(
            abs(value) for value in score_differences
        ),
        'max_absolute_score_difference': max(
            abs(value) for value in score_differences
        ),
    }


def aggregate_path(seed_reports, path_name):
    summaries = [report['paths'][path_name] for report in seed_reports]
    return {
        'seed_count': len(summaries),
        'mean_score': statistics.mean(item['score'] for item in summaries),
        'median_score': statistics.median(item['score'] for item in summaries),
        'min_score': min(item['score'] for item in summaries),
        'max_score': max(item['score'] for item in summaries),
        'mean_drops': statistics.mean(item['drops'] for item in summaries),
        'done_count': sum(item['done'] for item in summaries),
        'mean_fast_forwarded_frames': statistics.mean(
            item['fast_forwarded_frames'] for item in summaries
        ),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    if args.seed_count <= 0:
        raise ValueError('seed-count must be positive')

    configs = {
        'training_30_full_fall': make_config(30, device=device),
        'lab_30_incremental': make_config(30, device=device),
        'lab_120_incremental': make_config(120, device=device),
    }
    execution = {
        'training_30_full_fall': 'full_step',
        'lab_30_incremental': 'incremental_until_stable',
        'lab_120_incremental': 'incremental_until_stable',
    }
    if args.paths == 'same-fps-only':
        configs = {
            name: config for name, config in configs.items()
            if name in ('training_30_full_fall', 'lab_30_incremental')
        }
        execution = {
            name: value for name, value in execution.items()
            if name in configs
        }
    elif args.paths == 'current-defaults-only':
        configs = {
            name: config for name, config in configs.items()
            if name in ('training_30_full_fall', 'lab_120_incremental')
        }
        execution = {
            name: value for name, value in execution.items()
            if name in configs
        }
    pair_definitions = {
        # right - left 的符号统一表示后一个路径相对前一个路径的得分变化。
        'same_fps_full_step_vs_lab_incremental': (
            'training_30_full_fall', 'lab_30_incremental'
        ),
        'lab_30_vs_training_30_current': (
            'lab_30_incremental', 'training_30_full_fall'
        ),
        'current_defaults_lab_120_vs_training_30': (
            'lab_120_incremental', 'training_30_full_fall'
        ),
    }
    pair_definitions = {
        name: pair for name, pair in pair_definitions.items()
        if all(path in configs for path in pair)
    }

    seeds = [
        args.seed + seed_index * args.seed_stride
        for seed_index in range(args.seed_count)
    ]
    action_sequences = []
    for seed_index in range(args.seed_count):
        seed = seeds[seed_index]
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed ^ 0x5EED5EED)
        action_sequences.append(torch.randint(
            0, 21, (args.max_drops,), generator=generator
        ).tolist())
    batched_episodes = {
        name: run_episodes(
                seeds=seeds,
                actions=action_sequences,
                config=config,
                device=device,
                execution=execution[name],
                record_trajectory=not args.summary_only,
            )
        for name, config in configs.items()
    }
    seed_reports = []
    for seed_index, seed in enumerate(seeds):
        episodes = {
            name: paths[seed_index]
            for name, paths in batched_episodes.items()
        }
        seed_reports.append({
            'seed': seed,
            'actions': action_sequences[seed_index],
            'paths': {
                name: episode['summary'] for name, episode in episodes.items()
            },
            'comparisons': (
                {}
                if args.summary_only
                else {
                    pair_name: compare_episodes(
                        episodes[left_name], episodes[right_name]
                    )
                    for pair_name, (left_name, right_name)
                    in pair_definitions.items()
                }
            ),
        })

    report = {
        'physics_identity': PHYSICS_IDENTITY,
        'device': str(device),
        'seed': args.seed,
        'seed_count': args.seed_count,
        'seed_stride': args.seed_stride,
        'max_drops': args.max_drops,
        'action_policy': 'fixed torch randint sequence per seed',
        'path_configs': {
            name: asdict(config) for name, config in configs.items()
        },
        'pair_definitions': pair_definitions,
        'seeds': seed_reports,
        'aggregate': {
            'paths': {
                path_name: aggregate_path(seed_reports, path_name)
                for path_name in configs
            },
            'comparisons': (
                {}
                if args.summary_only
                else {
                    pair_name: aggregate_pair(seed_reports, pair_name)
                    for pair_name in pair_definitions
                }
            ),
        },
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)


if __name__ == '__main__':
    main()
