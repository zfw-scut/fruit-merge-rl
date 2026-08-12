"""抽样测量批量模拟器在决策边界上的真实碰撞圆穿透。"""

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
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--sample-envs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260804)
    parser.add_argument('--max-fruits', type=int, default=128)
    parser.add_argument('--physics-fps', type=int, default=30)
    parser.add_argument('--max-physics-frames', type=int, default=180)
    parser.add_argument('--stable-frames', type=int, default=4)
    parser.add_argument('--solver-iterations', type=int, default=4)
    parser.add_argument('--adaptive-collision-substeps', action='store_true')
    parser.add_argument('--max-collision-substeps', type=int, default=2)
    parser.add_argument('--position-correction', type=float, default=0.75)
    parser.add_argument(
        '--restitution-velocity-threshold', type=float, default=35.0
    )
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def new_group():
    return {
        'frame_count': 0,
        'penetrations': [],
        'normalized_penetrations': [],
        'frame_max_penetrations': [],
        'frame_max_linear_speeds': [],
        'frame_max_angular_speeds': [],
    }


def quantile_summary(values):
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float32)
    quantiles = torch.quantile(
        tensor, torch.tensor([0.0, 0.5, 0.9, 0.95, 1.0])
    ).tolist()
    return dict(zip(('min', 'p50', 'p90', 'p95', 'max'), quantiles))


def finish_group(group):
    return {
        'frame_count': group['frame_count'],
        'contact_count': len(group['penetrations']),
        'penetration_px': quantile_summary(group['penetrations']),
        'penetration_over_min_radius': quantile_summary(
            group['normalized_penetrations']
        ),
        'frame_max_penetration_px': quantile_summary(
            group['frame_max_penetrations']
        ),
        'frame_max_linear_speed': quantile_summary(
            group['frame_max_linear_speeds']
        ),
        'frame_max_angular_speed': quantile_summary(
            group['frame_max_angular_speeds']
        ),
    }


def sample_group(state, env_index, group):
    active = state['active'][env_index]
    positions = state['positions'][env_index, active]
    radii = state['physics_radii'][env_index, active]
    velocities = state['velocities'][env_index, active]
    angular = state['angular_velocities'][env_index, active].abs()
    group['frame_count'] += 1
    group['frame_max_linear_speeds'].append(
        float(velocities.norm(dim=-1).max().item()) if radii.numel() else 0.0
    )
    group['frame_max_angular_speeds'].append(
        float(angular.max().item()) if radii.numel() else 0.0
    )
    if radii.numel() < 2:
        return
    distances = torch.cdist(positions, positions)
    penetrations = radii[:, None] + radii[None, :] - distances
    mask = (
        torch.triu(torch.ones_like(penetrations, dtype=torch.bool), 1)
        & (penetrations > 0)
    )
    values = penetrations[mask]
    if not values.numel():
        return
    minimum_radii = torch.minimum(radii[:, None], radii[None, :])[mask]
    group['penetrations'].extend(values.cpu().tolist())
    group['normalized_penetrations'].extend(
        (values / minimum_radii).cpu().tolist()
    )
    group['frame_max_penetrations'].append(float(values.max().item()))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    if not 0 < args.sample_envs <= args.num_envs:
        raise ValueError('sample-envs must be in [1, num-envs]')

    config = SimulatorConfig(
        max_fruits=args.max_fruits,
        physics_fps=args.physics_fps,
        max_physics_frames=args.max_physics_frames,
        stable_frames=args.stable_frames,
        solver_iterations=args.solver_iterations,
        adaptive_collision_substeps=args.adaptive_collision_substeps,
        max_collision_substeps=args.max_collision_substeps,
        position_correction=args.position_correction,
        restitution_velocity_threshold=args.restitution_velocity_threshold,
    )
    simulator = TensorVectorSimulator(
        args.num_envs, config=config, device='cuda'
    )
    simulator.reset(seeds=args.seed)
    generator = torch.Generator(device=simulator.device)
    generator.manual_seed(args.seed)
    groups = {name: new_group() for name in ('all', 'stable', 'timeout')}

    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.steps):
        actions = torch.randint(
            config.action_count,
            (args.num_envs,),
            device=simulator.device,
            generator=generator,
        )
        result = simulator.step(actions)
        state = {
            'active': simulator.active[:args.sample_envs].cpu(),
            'positions': simulator.positions[:args.sample_envs].cpu(),
            'physics_radii': simulator.physics_radii[:args.sample_envs].cpu(),
            'velocities': simulator.velocities[:args.sample_envs].cpu(),
            'angular_velocities': (
                simulator.angular_velocities[:args.sample_envs].cpu()
            ),
        }
        timeout = result.physics.settle_timeout[:args.sample_envs].cpu()
        for env_index in range(args.sample_envs):
            sample_group(state, env_index, groups['all'])
            target = (
                'timeout'
                if bool(timeout[env_index])
                else 'stable'
            )
            sample_group(state, env_index, groups[target])
        reset_mask = result.physics.done | result.physics.truncated
        if bool(reset_mask.any().item()):
            simulator.reset(reset_mask)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    report = {
        'device': str(simulator.device),
        'num_envs': args.num_envs,
        'sample_envs': args.sample_envs,
        'steps': args.steps,
        'elapsed_seconds_including_diagnostics': elapsed,
        'groups': {name: finish_group(group) for name, group in groups.items()},
        'config': {
            name: getattr(config, name)
            for name in (
                'max_fruits',
                'physics_fps',
                'max_physics_frames',
                'stable_frames',
                'solver_iterations',
                'drop_fast_forward',
                'adaptive_collision_substeps',
                'max_collision_substeps',
                'position_correction',
                'restitution_velocity_threshold',
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(rendered, encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
