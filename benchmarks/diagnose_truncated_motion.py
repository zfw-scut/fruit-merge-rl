"""逐帧抽取 CUDA 截断环境，定位稳定窗口无法满足的运动来源。"""

import argparse
import json
import math
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from daxigua.simulator import (
    BatchSimulationTrace,
    save_trace_archive,
    SimulatorConfig,
    TensorVectorSimulator,
    write_replay_html,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--max-drops', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260804)
    parser.add_argument('--tail-frames', type=int, default=120)
    parser.add_argument('--solver-iterations', type=int, default=4)
    parser.add_argument(
        '--env-indices',
        help='逗号分隔的指定环境索引；提供后不再随机抽样。',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'recordings' / 'truncation-diagnostics',
    )
    return parser.parse_args()


def select_trace_row(trace, row):
    count = int(trace.env_indices.numel())
    values = {}
    for field_name in trace.__dataclass_fields__:
        value = getattr(trace, field_name)
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == count:
            values[field_name] = value[row:row + 1].clone()
        else:
            values[field_name] = value
    return BatchSimulationTrace(**values)


def longest_true_run(values):
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def classify_motion(history, config):
    if not history:
        return 'no_active_motion'
    floor_y = config.board_height - config.wall_width
    linear_frames = sum(item['linear_bad'] for item in history)
    angular_frames = sum(item['angular_bad'] for item in history)
    floor_fraction = sum(
        abs(floor_y - (item['y'] + item['radius'])) <= 1.0
        for item in history
    ) / len(history)
    left = config.wall_width
    right = config.board_width - config.wall_width
    wall_fraction = sum(
        min(
            abs(item['x'] - item['radius'] - left),
            abs(item['x'] + item['radius'] - right),
        ) <= 1.0
        for item in history
    ) / len(history)
    x_range = max(item['x'] for item in history) - min(
        item['x'] for item in history
    )
    y_range = max(item['y'] for item in history) - min(
        item['y'] for item in history
    )
    vertical_sign_changes = sum(
        history[index - 1]['vy'] * history[index]['vy'] < 0
        for index in range(1, len(history))
    )
    if angular_frames > linear_frames and angular_frames:
        return 'angular_velocity_not_decaying'
    if linear_frames and max(x_range, y_range) <= 0.1:
        return 'stationary_contact_velocity_residual'
    if floor_fraction >= 0.7 and vertical_sign_changes >= 4 and y_range > 0.5:
        return 'micro_bounce_on_floor_or_stack'
    rolling_error = sum(
        abs(abs(item['vx']) - abs(item['omega']) * item['radius'])
        for item in history
    ) / len(history)
    if floor_fraction >= 0.7 and rolling_error <= 2.0:
        return 'persistent_floor_rolling'
    if floor_fraction >= 0.7 and x_range > 1.0:
        return 'persistent_floor_sliding'
    if wall_fraction >= 0.7 and y_range > 1.0:
        return 'persistent_wall_sliding'
    if linear_frames:
        return 'persistent_linear_motion_in_stack'
    if angular_frames:
        return 'angular_velocity_not_decaying'
    return 'intermittent_stability_window_reset'


def analyze_trace(trace, config, tail_frames):
    trace = trace.cpu()
    record_count = int(trace.record_counts[0])
    start = max(1, record_count - int(tail_frames))
    stable_by_record = []
    any_linear_by_record = []
    any_angular_by_record = []
    histories = {}
    for record_index in range(1, record_count):
        active = trace.active[0, record_index]
        speed = torch.linalg.vector_norm(
            trace.velocities[0, record_index], dim=-1
        )
        angular = trace.angular_velocities[0, record_index].abs()
        linear_bad = active & (speed > config.stable_velocity_epsilon)
        angular_bad = active & (
            angular > config.stable_angular_velocity_epsilon
        )
        stable_by_record.append(
            not bool((linear_bad | angular_bad).any().item())
        )
        any_linear_by_record.append(bool(linear_bad.any().item()))
        any_angular_by_record.append(bool(angular_bad.any().item()))
        if record_index < start:
            continue
        for slot in torch.nonzero(active, as_tuple=False).flatten().tolist():
            fruit_id = int(trace.fruit_ids[0, record_index, slot])
            vx = float(trace.velocities[0, record_index, slot, 0])
            vy = float(trace.velocities[0, record_index, slot, 1])
            omega = float(trace.angular_velocities[0, record_index, slot])
            histories.setdefault(fruit_id, []).append({
                'level': int(trace.levels[0, record_index, slot]),
                'x': float(trace.positions[0, record_index, slot, 0]),
                'y': float(trace.positions[0, record_index, slot, 1]),
                'radius': float(trace.physics_radii[0, record_index, slot]),
                'vx': vx,
                'vy': vy,
                'speed': math.hypot(vx, vy),
                'omega': omega,
                'linear_bad': math.hypot(vx, vy)
                > config.stable_velocity_epsilon,
                'angular_bad': abs(omega)
                > config.stable_angular_velocity_epsilon,
            })

    offenders = []
    for fruit_id, history in histories.items():
        linear_bad_frames = sum(item['linear_bad'] for item in history)
        angular_bad_frames = sum(item['angular_bad'] for item in history)
        if not linear_bad_frames and not angular_bad_frames:
            continue
        final = history[-1]
        duration = max(1.0 / config.physics_fps, (
            len(history) - 1
        ) / config.physics_fps)
        net_displacement = math.hypot(
            final['x'] - history[0]['x'],
            final['y'] - history[0]['y'],
        )
        offenders.append({
            'fruit_id': fruit_id,
            'level': final['level'],
            'classification': classify_motion(history, config),
            'observed_tail_frames': len(history),
            'linear_bad_frames': linear_bad_frames,
            'angular_bad_frames': angular_bad_frames,
            'max_speed': max(item['speed'] for item in history),
            'max_abs_angular_velocity': max(
                abs(item['omega']) for item in history
            ),
            'x_range': max(item['x'] for item in history)
            - min(item['x'] for item in history),
            'y_range': max(item['y'] for item in history)
            - min(item['y'] for item in history),
            'net_displacement': net_displacement,
            'net_displacement_speed': net_displacement / duration,
            'final': {
                key: final[key]
                for key in ('x', 'y', 'vx', 'vy', 'speed', 'omega')
            },
        })
    offenders.sort(
        key=lambda item: (
            item['linear_bad_frames'] + item['angular_bad_frames'],
            item['max_speed'],
        ),
        reverse=True,
    )
    tail_stable = stable_by_record[-tail_frames:]
    return {
        'env_index': int(trace.env_indices[0]),
        'action': int(trace.actions[0]),
        'step_count': None,
        'physics_frames': int(
            trace.frame_numbers[0, record_count - 1]
        ),
        'record_count': record_count,
        'fruit_count': int(trace.active[0, record_count - 1].sum()),
        'score': int(trace.scores[0, record_count - 1]),
        'merge_count': int(trace.merge_counts[0, record_count - 1]),
        'stable_required_frames': config.stable_frames,
        'longest_stable_run_entire_step': longest_true_run(stable_by_record),
        'longest_stable_run_tail': longest_true_run(tail_stable),
        'tail_records': len(tail_stable),
        'tail_frames_with_linear_instability': sum(
            any_linear_by_record[-tail_frames:]
        ),
        'tail_frames_with_angular_instability': sum(
            any_angular_by_record[-tail_frames:]
        ),
        'offenders': offenders[:8],
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    if args.env_indices:
        sampled_values = [
            int(value.strip())
            for value in args.env_indices.split(',')
            if value.strip()
        ]
        if not sampled_values or len(set(sampled_values)) != len(sampled_values):
            raise ValueError('env-indices must contain unique indices')
        if any(value < 0 or value >= args.num_envs for value in sampled_values):
            raise ValueError('env-indices contains an out-of-range index')
    else:
        if args.samples <= 0 or args.samples > args.num_envs:
            raise ValueError('samples must be in [1, num_envs]')
        sampled_values = None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulatorConfig(solver_iterations=args.solver_iterations)
    simulator = TensorVectorSimulator(
        args.num_envs, config=config, device='cuda'
    )
    simulator.reset(seeds=args.seed)
    action_generator = torch.Generator(device=simulator.device)
    action_generator.manual_seed(args.seed)
    sample_generator = torch.Generator(device='cpu')
    sample_generator.manual_seed(args.seed + 1)
    sampled_envs = (
        torch.tensor(
            sampled_values, dtype=torch.int64, device=simulator.device
        )
        if sampled_values is not None
        else torch.randperm(
            args.num_envs, generator=sample_generator
        )[:args.samples].to(device=simulator.device)
    )

    running = torch.ones(
        args.num_envs, dtype=torch.bool, device=simulator.device
    )
    diagnostics = []
    captured = set()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for drop_index in range(args.max_drops):
        pending = sampled_envs[
            running[sampled_envs]
            & torch.tensor(
                [int(value) not in captured for value in sampled_envs.tolist()],
                dtype=torch.bool,
                device=simulator.device,
            )
        ]
        if pending.numel() == 0:
            break
        actions = torch.randint(
            config.action_count,
            (args.num_envs,),
            dtype=torch.int64,
            device=simulator.device,
            generator=action_generator,
        )
        result, trace = simulator.step_masked_with_trace(
            actions,
            running,
            pending,
            frame_stride=1,
        )
        stopped_boundary = running & (
            result.physics.done | result.physics.truncated
        )
        newly_captured = running & (
            result.physics.done
            | result.physics.truncated
            | result.physics.settle_timeout
        )
        pending_list = pending.tolist()
        for row, env_index in enumerate(pending_list):
            if not bool(newly_captured[env_index].item()):
                continue
            single_trace = select_trace_row(trace, row).cpu()
            analysis = analyze_trace(single_trace, config, args.tail_frames)
            analysis['step_count'] = int(simulator.step_count[env_index].item())
            analysis['stop_kind'] = (
                'terminated'
                if bool(result.physics.done[env_index].item())
                else (
                    'truncated'
                    if bool(result.physics.truncated[env_index].item())
                    else 'settle_timeout'
                )
            )
            html_path = args.output_dir / f'env-{env_index}-final-step.html'
            tensor_path = args.output_dir / f'env-{env_index}-final-step.pt.gz'
            write_replay_html(
                html_path,
                single_trace,
                config,
                title=f'等待超时环境 {env_index} 当前投放',
            )
            save_trace_archive(tensor_path, single_trace)
            analysis['replay'] = str(html_path.resolve())
            analysis['trace'] = str(tensor_path.resolve())
            diagnostics.append(analysis)
            captured.add(env_index)
        captured_this_step = torch.zeros_like(running)
        if pending_list:
            captured_this_step[pending] = newly_captured[pending]
        running &= ~(stopped_boundary | captured_this_step)
        if (drop_index + 1) % 20 == 0:
            print(
                json.dumps({
                    'drop_iteration': drop_index + 1,
                    'sampled_remaining': int(pending.numel()),
                    'captured': len(captured),
                    'elapsed_seconds': time.perf_counter() - started,
                }, ensure_ascii=False),
                flush=True,
            )

    torch.cuda.synchronize()
    report = {
        'seed': args.seed,
        'num_envs': args.num_envs,
        'sampled_envs': sampled_envs.tolist(),
        'captured': len(diagnostics),
        'elapsed_seconds': time.perf_counter() - started,
        'config': {
            'stable_velocity_epsilon': config.stable_velocity_epsilon,
            'stable_angular_velocity_epsilon': (
                config.stable_angular_velocity_epsilon
            ),
            'stable_frames': config.stable_frames,
            'max_physics_frames': config.max_physics_frames,
            'solver_iterations': config.solver_iterations,
        },
        'diagnostics': diagnostics,
    }
    report_path = args.output_dir / 'report.json'
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
