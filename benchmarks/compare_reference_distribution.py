"""Pymunk 与 CUDA 在同一动作/当前水果流上的分布门禁。"""

import argparse
import json
from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from daxigua.simulator import SimulatorConfig, TensorVectorSimulator
from daxigua.simulator.reference import PymunkReferenceGame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20260804)
    parser.add_argument('--relative-tolerance', type=float, default=0.15)
    parser.add_argument('--solver-iterations', type=int, default=4)
    parser.add_argument('--kinematic-rest-frames', type=int, default=4)
    parser.add_argument('--kinematic-rest-epsilon', type=float, default=0.1)
    return parser.parse_args()


def relative_error(actual, reference):
    return abs(actual - reference) / max(1.0, abs(reference))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    config = SimulatorConfig(
        solver_iterations=args.solver_iterations,
        kinematic_rest_frames=args.kinematic_rest_frames,
        kinematic_rest_displacement_epsilon=args.kinematic_rest_epsilon,
    )
    rng = random.Random(args.seed)
    action_rows = [
        [rng.randrange(config.action_count) for _ in range(args.num_envs)]
        for _ in range(args.steps)
    ]
    # 强制两个后端每步使用相同 q0，隔离 Python Random 和设备
    # LCG 序列差异，只比较物理、合成和终止分布。
    level_rows = [
        [rng.randrange(1, 6) for _ in range(args.num_envs)]
        for _ in range(args.steps)
    ]

    reference_games = [
        PymunkReferenceGame(config, seed=args.seed + env_index)
        for env_index in range(args.num_envs)
    ]
    reference = {
        'score': 0,
        'merges': 0,
        'physics_frames': 0,
        'terminated': 0,
        'truncated': 0,
        'settle_timeout': 0,
    }
    for actions, levels in zip(action_rows, level_rows):
        for env_index, action in enumerate(actions):
            game = reference_games[env_index]
            game.fruit_queue[0] = levels[env_index]
            _state, _drop, physics = game.step(action)
            reference['score'] += physics.score_delta
            reference['merges'] += len(physics.merge_events)
            reference['physics_frames'] += physics.frames_simulated
            reference['terminated'] += int(physics.done)
            reference['truncated'] += int(physics.truncated)
            reference['settle_timeout'] += int(physics.settle_timeout)
            if physics.done or physics.truncated:
                game.reset()

    simulator = TensorVectorSimulator(
        args.num_envs, config=config, device='cuda'
    )
    cuda = {key: 0 for key in reference}
    for actions, levels in zip(action_rows, level_rows):
        simulator.fruit_queue[:, 0] = torch.tensor(
            levels, dtype=torch.int64, device='cuda'
        )
        result = simulator.step(
            torch.tensor(actions, dtype=torch.int64, device='cuda')
        )
        cuda['score'] += int(result.physics.score_delta.sum().item())
        cuda['merges'] += int(result.physics.merge_events.count.sum().item())
        cuda['physics_frames'] += int(
            result.physics.frames_simulated.sum().item()
        )
        cuda['terminated'] += int(result.physics.done.sum().item())
        cuda['truncated'] += int(result.physics.truncated.sum().item())
        cuda['settle_timeout'] += int(
            result.physics.settle_timeout.sum().item()
        )
        reset_mask = result.physics.done | result.physics.truncated
        if bool(reset_mask.any().item()):
            simulator.reset(reset_mask)

    relative_errors = {
        key: relative_error(cuda[key], reference[key])
        for key in ('score', 'merges', 'physics_frames')
    }
    boundary_difference = {
        key: abs(cuda[key] - reference[key])
        for key in ('terminated', 'truncated', 'settle_timeout')
    }
    report = {
        'transitions': args.num_envs * args.steps,
        'reference': reference,
        'cuda': cuda,
        'relative_errors': relative_errors,
        'boundary_difference': boundary_difference,
        'relative_tolerance': args.relative_tolerance,
        'solver_iterations': config.solver_iterations,
        'kinematic_rest_frames': config.kinematic_rest_frames,
        'kinematic_rest_displacement_epsilon': (
            config.kinematic_rest_displacement_epsilon
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    failed_metrics = [
        key
        for key, error in relative_errors.items()
        if error > args.relative_tolerance
    ]
    boundary_limit = max(2, int(0.01 * args.num_envs * args.steps))
    failed_boundaries = [
        key
        for key, difference in boundary_difference.items()
        if difference > boundary_limit
    ]
    if failed_metrics or failed_boundaries:
        raise SystemExit(
            'distribution gate failed: '
            f'metrics={failed_metrics}, boundaries={failed_boundaries}'
        )


if __name__ == '__main__':
    main()
