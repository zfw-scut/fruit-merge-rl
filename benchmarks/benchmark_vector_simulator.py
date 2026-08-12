"""测量 CUDA 多环境模拟器的稳态吞吐。"""

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from daxigua.simulator import SimulatorConfig, TensorVectorSimulator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--max-fruits', type=int, default=64)
    parser.add_argument('--physics-fps', type=int, default=120)
    parser.add_argument('--max-physics-frames', type=int, default=720)
    parser.add_argument('--stable-frames', type=int, default=15)
    parser.add_argument('--solver-iterations', type=int, default=4)
    parser.add_argument('--adaptive-collision-substeps', action='store_true')
    parser.add_argument('--max-collision-substeps', type=int, default=4)
    parser.add_argument(
        '--collision-substep-motion-fraction', type=float, default=0.25
    )
    parser.add_argument(
        '--collision-substep-penetration-threshold', type=float, default=1.0
    )
    parser.add_argument(
        '--restitution-velocity-threshold', type=float, default=35.0
    )
    parser.add_argument('--position-correction', type=float, default=0.75)
    parser.add_argument('--seed', type=int, default=20260804)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def run_steps(simulator, steps, generator):
    total_frames = torch.zeros((), dtype=torch.int64, device=simulator.device)
    total_fast_forwarded = torch.zeros_like(total_frames)
    total_collision_substeps = torch.zeros_like(total_frames)
    total_merges = torch.zeros_like(total_frames)
    total_stable = torch.zeros_like(total_frames)
    total_settle_timeouts = torch.zeros_like(total_frames)
    total_done = torch.zeros_like(total_frames)
    reset_count = 0
    for _ in range(steps):
        actions = torch.randint(
            0,
            simulator.config.action_count,
            (simulator.num_envs,),
            device=simulator.device,
            generator=generator,
        )
        result = simulator.step(actions)
        total_frames += result.physics.frames_simulated.sum()
        if result.physics.fast_forwarded_frames is not None:
            total_fast_forwarded += (
                result.physics.fast_forwarded_frames.sum()
            )
        if result.physics.collision_substeps is not None:
            total_collision_substeps += result.physics.collision_substeps.sum()
        total_merges += result.physics.merge_events.count.sum()
        total_stable += result.physics.stable.sum()
        if result.physics.settle_timeout is not None:
            total_settle_timeouts += result.physics.settle_timeout.sum()
        total_done += result.physics.done.sum()
        reset_mask = result.physics.done | result.physics.truncated
        if bool(reset_mask.any().item()):
            reset_count += int(reset_mask.sum().item())
            simulator.reset(reset_mask)
    return {
        'physics_frames': total_frames,
        'fast_forwarded_frames': total_fast_forwarded,
        'collision_substeps': total_collision_substeps,
        'merge_events': total_merges,
        'stable_intervals': total_stable,
        'settle_timeout_intervals': total_settle_timeouts,
        'done_intervals': total_done,
        'reset_count': reset_count,
    }


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')

    config = SimulatorConfig(
        max_fruits=args.max_fruits,
        physics_fps=args.physics_fps,
        max_physics_frames=args.max_physics_frames,
        stable_frames=args.stable_frames,
        solver_iterations=args.solver_iterations,
        adaptive_collision_substeps=args.adaptive_collision_substeps,
        max_collision_substeps=args.max_collision_substeps,
        collision_substep_motion_fraction=(
            args.collision_substep_motion_fraction
        ),
        collision_substep_penetration_threshold=(
            args.collision_substep_penetration_threshold
        ),
        restitution_velocity_threshold=args.restitution_velocity_threshold,
        position_correction=args.position_correction,
    )
    simulator = TensorVectorSimulator(
        args.num_envs,
        config=config,
        device=args.device,
    )
    generator = torch.Generator(device=simulator.device)
    generator.manual_seed(args.seed)

    # 首次执行包含扩展编译/加载和 CUDA context 暖机，不计入稳态。
    run_steps(simulator, args.warmup_steps, generator)
    simulator.reset(seeds=args.seed)
    if simulator.device.type == 'cuda':
        torch.cuda.synchronize(simulator.device)
        torch.cuda.reset_peak_memory_stats(simulator.device)

    started = time.perf_counter()
    totals = run_steps(
        simulator, args.steps, generator
    )
    if simulator.device.type == 'cuda':
        torch.cuda.synchronize(simulator.device)
    elapsed = time.perf_counter() - started
    transitions = args.num_envs * args.steps
    total_frames = int(totals['physics_frames'].item())
    fast_forwarded_frames = int(
        totals['fast_forwarded_frames'].item()
    )
    executed_frames = total_frames - fast_forwarded_frames
    collision_substeps = int(totals['collision_substeps'].item())
    fruit_counts = simulator.active.sum(dim=1)
    observation = simulator.observe()
    finite_state = bool(
        torch.isfinite(simulator.positions).all().item()
        and torch.isfinite(simulator.velocities).all().item()
    )
    report = {
        'device': str(simulator.device),
        'num_envs': args.num_envs,
        'steps': args.steps,
        'transitions': transitions,
        'elapsed_seconds': elapsed,
        'env_steps_per_second': transitions / elapsed,
        'semantic_physics_frames': total_frames,
        'executed_physics_frames': executed_frames,
        'fast_forwarded_frames': fast_forwarded_frames,
        'collision_substeps': collision_substeps,
        'collision_substeps_per_executed_frame': (
            collision_substeps / executed_frames if executed_frames else 0.0
        ),
        'extra_collision_substep_ratio': (
            (collision_substeps - executed_frames) / executed_frames
            if executed_frames else 0.0
        ),
        'fast_forward_ratio': (
            fast_forwarded_frames / total_frames if total_frames else 0.0
        ),
        'semantic_physics_frames_per_second': total_frames / elapsed,
        'physics_frames_per_second': total_frames / elapsed,
        'executed_physics_frames_per_second': executed_frames / elapsed,
        'semantic_frames_per_transition': total_frames / transitions,
        'executed_frames_per_transition': executed_frames / transitions,
        'merge_events': int(totals['merge_events'].item()),
        'merge_events_per_transition': (
            int(totals['merge_events'].item()) / transitions
        ),
        'stable_interval_count': int(totals['stable_intervals'].item()),
        'settle_timeout_interval_count': int(
            totals['settle_timeout_intervals'].item()
        ),
        'done_interval_count': int(totals['done_intervals'].item()),
        'resets': totals['reset_count'],
        'mean_live_fruits': float(fruit_counts.float().mean().item()),
        'max_live_fruits': int(fruit_counts.max().item()),
        'mean_score': float(simulator.score.float().mean().item()),
        'mean_max_height': float(observation.max_height.mean().item()),
        'finite_state': finite_state,
        'peak_cuda_memory_mib': (
            torch.cuda.max_memory_allocated(simulator.device) / 1024 ** 2
            if simulator.device.type == 'cuda'
            else 0.0
        ),
        'config': {
            'max_fruits': config.max_fruits,
            'solver_iterations': config.solver_iterations,
            'drop_fast_forward': config.drop_fast_forward,
            'adaptive_collision_substeps': config.adaptive_collision_substeps,
            'max_collision_substeps': config.max_collision_substeps,
            'restitution_velocity_threshold': (
                config.restitution_velocity_threshold
            ),
            'position_correction': config.position_correction,
            'collision_substep_motion_fraction': (
                config.collision_substep_motion_fraction
            ),
            'collision_substep_penetration_threshold': (
                config.collision_substep_penetration_threshold
            ),
            'max_physics_frames': config.max_physics_frames,
            'stable_frames': config.stable_frames,
            'physics_fps': config.physics_fps,
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered)
    if not finite_state:
        raise RuntimeError('benchmark detected non-finite simulator state')


if __name__ == '__main__':
    main()
