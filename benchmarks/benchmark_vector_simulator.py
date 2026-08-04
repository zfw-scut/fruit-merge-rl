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
    parser.add_argument('--solver-iterations', type=int, default=4)
    parser.add_argument('--kinematic-rest-frames', type=int, default=4)
    parser.add_argument('--kinematic-rest-epsilon', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=20260804)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def run_steps(simulator, steps, generator):
    total_frames = torch.zeros((), dtype=torch.int64, device=simulator.device)
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
        reset_mask = result.physics.done | result.physics.truncated
        if bool(reset_mask.any().item()):
            reset_count += int(reset_mask.sum().item())
            simulator.reset(reset_mask)
    return total_frames, reset_count


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')

    config = SimulatorConfig(
        max_fruits=args.max_fruits,
        solver_iterations=args.solver_iterations,
        kinematic_rest_frames=args.kinematic_rest_frames,
        kinematic_rest_displacement_epsilon=args.kinematic_rest_epsilon,
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
    total_frames, reset_count = run_steps(
        simulator, args.steps, generator
    )
    if simulator.device.type == 'cuda':
        torch.cuda.synchronize(simulator.device)
    elapsed = time.perf_counter() - started
    transitions = args.num_envs * args.steps
    fruit_counts = simulator.active.sum(dim=1)
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
        'physics_frames_per_second': int(total_frames.item()) / elapsed,
        'resets': reset_count,
        'mean_live_fruits': float(fruit_counts.float().mean().item()),
        'max_live_fruits': int(fruit_counts.max().item()),
        'mean_score': float(simulator.score.float().mean().item()),
        'finite_state': finite_state,
        'peak_cuda_memory_mib': (
            torch.cuda.max_memory_allocated(simulator.device) / 1024 ** 2
            if simulator.device.type == 'cuda'
            else 0.0
        ),
        'config': {
            'max_fruits': config.max_fruits,
            'solver_iterations': config.solver_iterations,
            'kinematic_rest_frames': config.kinematic_rest_frames,
            'kinematic_rest_displacement_epsilon': (
                config.kinematic_rest_displacement_epsilon
            ),
            'max_physics_frames': config.max_physics_frames,
            'stable_frames': config.stable_frames,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not finite_state:
        raise RuntimeError('benchmark detected non-finite simulator state')


if __name__ == '__main__':
    main()
