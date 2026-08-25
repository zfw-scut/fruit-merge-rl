"""用策略/随机混合决策并行生成可复用的复杂稳定场景数据集。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from daxigua.rl.observations import TensorState  # noqa: E402
from daxigua.rl.viewer import (  # noqa: E402
    load_viewer_model,
    viewer_simulator_config,
)
from daxigua.simulator import (  # noqa: E402
    TensorVectorSimulator,
    save_static_scene_dataset,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / 'runs'
    / 'cloud_rtx5090_auxiliary_action_structured_branch_seed20260811_128m'
    / 'checkpoints'
    / 'final.pt'
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / 'runs' / 'diagnostics'
    / 'mergeability_complex_scenes_2000_20260825'
)
SEED_STRIDE = 1_000_003


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='并行生成策略/随机混合的复杂稳定场景快照。'
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-envs', type=int, default=2000)
    parser.add_argument('--target-drops', type=int, default=300)
    parser.add_argument('--policy-fraction', type=float, default=0.5)
    parser.add_argument('--seed-base', type=int, default=202_608_250_000)
    parser.add_argument('--action-seed', type=int, default=202_608_251)
    parser.add_argument('--progress-interval', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _validate_args(args):
    if args.num_envs <= 0:
        raise ValueError('num-envs must be positive')
    if args.target_drops <= 0:
        raise ValueError('target-drops must be positive')
    if not 0.0 <= args.policy_fraction <= 1.0:
        raise ValueError('policy-fraction must be in [0, 1]')
    if args.progress_interval <= 0:
        raise ValueError('progress-interval must be positive')


def _mode_mask(num_envs, fraction, seed, device):
    policy_count = round(num_envs * float(fraction))
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    order = torch.randperm(num_envs, generator=generator)
    mask = torch.zeros(num_envs, dtype=torch.bool)
    mask[order[:policy_count]] = True
    return mask.to(device)


def _action_generator(device, seed):
    generator = torch.Generator(device=torch.device(device).type)
    generator.manual_seed(int(seed))
    return generator


def _summary(observation, policy_mask):
    rows = {}
    for name, mask in (
            ('all', torch.ones_like(policy_mask)),
            ('policy', policy_mask),
            ('random', ~policy_mask)):
        count = int(mask.sum().item())
        if count == 0:
            rows[name] = {'scenes': 0}
            continue
        fruit_count = observation.fruit_count[mask].to(torch.float32)
        step_count = observation.step_count[mask].to(torch.float32)
        score = observation.score[mask].to(torch.float32)
        done = observation.done[mask].to(torch.float32)
        rows[name] = {
            'scenes': count,
            'terminal_rate': float(done.mean().item()),
            'fruit_count_mean': float(fruit_count.mean().item()),
            'fruit_count_min': int(fruit_count.min().item()),
            'fruit_count_max': int(fruit_count.max().item()),
            'drops_mean': float(step_count.mean().item()),
            'drops_min': int(step_count.min().item()),
            'drops_max': int(step_count.max().item()),
            'score_mean': float(score.mean().item()),
            'score_max': int(score.max().item()),
        }
    return rows


@torch.inference_mode()
def collect(args):
    _validate_args(args)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_viewer_model(args.checkpoint, device=args.device)
    config = viewer_simulator_config(30, loaded.model_config, loaded.device)
    simulator = TensorVectorSimulator(
        args.num_envs, config=config, device=loaded.device
    )
    env_indices = torch.arange(
        args.num_envs, dtype=torch.int64, device=loaded.device
    )
    seeds = env_indices * SEED_STRIDE + int(args.seed_base)
    simulator.reset(seeds=seeds)
    policy_mask = _mode_mask(
        args.num_envs, args.policy_fraction, args.action_seed, loaded.device
    )
    enabled = torch.ones(
        args.num_envs, dtype=torch.bool, device=loaded.device
    )
    action_generator = _action_generator(loaded.device, args.action_seed)

    if loaded.device.type == 'cuda':
        torch.cuda.synchronize(loaded.device)
    started = time.perf_counter()
    for drop in range(1, args.target_drops + 1):
        observation = simulator.observe()
        actions = torch.randint(
            loaded.model_config.action_count,
            (args.num_envs,),
            dtype=torch.int64,
            device=loaded.device,
            generator=action_generator,
        )
        policy_rows = torch.nonzero(
            enabled & policy_mask, as_tuple=False
        ).flatten()
        if int(policy_rows.numel()) > 0:
            state = TensorState.from_observation(
                observation,
                physics_fps=config.physics_fps,
                rows=policy_rows,
            )
            actions[policy_rows] = loaded.model(state).argmax(dim=1)

        if loaded.device.type == 'cuda':
            step = simulator.step_masked(actions, enabled)
        else:
            if not bool(enabled.all().item()):
                raise RuntimeError(
                    'CPU smoke collection cannot continue after an early terminal'
                )
            step = simulator.step(actions)
        enabled &= ~(step.physics.done | step.physics.truncated)
        if (
                drop % args.progress_interval == 0
                or drop == args.target_drops
                or not bool(enabled.any().item())):
            current = simulator.observe()
            elapsed = time.perf_counter() - started
            print(json.dumps({
                'phase': 'collect',
                'drop': drop,
                'target_drops': args.target_drops,
                'active_envs': int(enabled.sum().item()),
                'terminal_envs': int((~enabled).sum().item()),
                'fruit_count_mean': float(
                    current.fruit_count.to(torch.float32).mean().item()
                ),
                'elapsed_seconds': elapsed,
            }, ensure_ascii=False), flush=True)
        if not bool(enabled.any().item()):
            break

    if loaded.device.type == 'cuda':
        torch.cuda.synchronize(loaded.device)
    elapsed = time.perf_counter() - started
    observation = simulator.observe().clone()
    summary = _summary(observation, policy_mask)
    manifest = {
        'purpose': 'reusable_complex_stable_scene_dataset',
        'scene_count': args.num_envs,
        'target_drops': args.target_drops,
        'policy_fraction': args.policy_fraction,
        'policy_semantics': {'1': 'SAB-128 greedy', '0': 'uniform random'},
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'device': str(loaded.device),
        'physics_fps': config.physics_fps,
        'simulator_config': asdict(config),
        'seed_base': args.seed_base,
        'seed_stride': SEED_STRIDE,
        'action_seed': args.action_seed,
        'elapsed_seconds': elapsed,
        'summary': summary,
    }
    metadata = {
        'seed': seeds,
        'policy_mode': policy_mask.to(torch.int8),
        'capture_at_target': enabled.to(torch.int8),
    }
    dataset_path = output_dir / 'scene_states.pt'
    save_static_scene_dataset(
        dataset_path,
        observation,
        metadata=metadata,
        manifest=manifest,
    )
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps({
        'phase': 'complete',
        'output_dir': str(output_dir),
        'dataset': str(dataset_path),
        'manifest': str(manifest_path),
        'elapsed_seconds': elapsed,
        'summary': summary,
    }, ensure_ascii=False, indent=2), flush=True)
    return dataset_path


def main(argv=None):
    collect(parse_args(argv))


if __name__ == '__main__':
    main()
