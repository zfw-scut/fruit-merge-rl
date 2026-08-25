"""用SAB-128长期CUDA rollout采集场景可合成性及逐投放变化量。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from daxigua.rl.merge_potential_stats import (  # noqa: E402
    DeviceTableAccumulator,
    ShardedTensorWriter,
)
from daxigua.rl.mergeability import (  # noqa: E402
    MergeabilityCalculator,
    MergeabilityConfig,
)
from daxigua.rl.mergeability_rollout import (  # noqa: E402
    SCENE_VALUE_DTYPES,
    scene_mergeability_delta,
    scene_mergeability_values,
)
from daxigua.rl.observations import TensorState  # noqa: E402
from daxigua.rl.viewer import (  # noqa: E402
    load_viewer_model,
    viewer_simulator_config,
)
from daxigua.simulator import (  # noqa: E402
    PHYSICS_IDENTITY,
    TensorVectorSimulator,
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
    / 'mergeability_rollout_stats_20260825'
)
SEED_STRIDE = 1_000_003


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='并行长期rollout并采集场景可合成性变化量。'
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-envs', type=int, default=2000)
    parser.add_argument('--decision-steps', type=int, default=2000)
    parser.add_argument('--seed-base', type=int, default=202_608_260_000)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--transfer-interval', type=int, default=16)
    parser.add_argument('--shard-rows', type=int, default=1_000_000)
    parser.add_argument('--writer-queue-depth', type=int, default=2)
    parser.add_argument('--foreground-writer', action='store_true')
    parser.add_argument('--progress-interval', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _validate_args(args):
    for name in (
            'num_envs', 'decision_steps', 'transfer_interval', 'shard_rows',
            'writer_queue_depth', 'progress_interval'):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f'{name.replace("_", "-")} must be positive')
    if args.warmup_steps < 0:
        raise ValueError('warmup-steps must be non-negative')


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git_revision():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _write_manifest(path, base, **updates):
    payload = dict(base)
    payload.update(updates)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    os.replace(temporary, path)


def _reset_finished(
        simulator,
        finished,
        episode_ids,
        episode_seeds,
        *,
        next_episode_id,
        seed_base):
    finished_rows = torch.nonzero(finished, as_tuple=False).flatten()
    count = int(finished_rows.numel())
    if count == 0:
        return next_episode_id
    new_ids = torch.arange(
        next_episode_id,
        next_episode_id + count,
        dtype=torch.int64,
        device=simulator.device,
    )
    new_seeds = new_ids * SEED_STRIDE + int(seed_base)
    episode_ids[finished_rows] = new_ids
    episode_seeds[finished_rows] = new_seeds
    simulator.reset(mask=finished, seeds=new_seeds)
    return next_episode_id + count


@torch.inference_mode()
def collect(args):
    _validate_args(args)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / 'raw'
    manifest_path = output_dir / 'manifest.json'

    loaded = load_viewer_model(args.checkpoint, device=args.device)
    config = viewer_simulator_config(30, loaded.model_config, loaded.device)
    if config.drop_fast_forward:
        raise RuntimeError('collection requires no-fast-forward physics')
    simulator = TensorVectorSimulator(
        args.num_envs, config=config, device=loaded.device
    )
    env_ids = torch.arange(
        args.num_envs, dtype=torch.int64, device=loaded.device
    )
    episode_ids = env_ids.clone()
    episode_seeds = episode_ids * SEED_STRIDE + int(args.seed_base)
    simulator.reset(seeds=episode_seeds)
    calculator = MergeabilityCalculator(
        MergeabilityConfig.from_simulator_config(config)
    ).to(loaded.device)

    warmup_started = time.perf_counter()
    for _ in range(args.warmup_steps):
        observation = simulator.observe()
        state = TensorState.from_observation(
            observation, physics_fps=config.physics_fps
        )
        actions = loaded.model(state).argmax(dim=1)
        step = simulator.step(actions)
        current_state = TensorState.from_observation(
            step.observation, physics_fps=config.physics_fps
        )
        calculator(current_state)
        finished = step.physics.done | step.physics.truncated
        if bool(finished.any().item()):
            simulator.reset(mask=finished, seeds=episode_seeds[finished])
    simulator.reset(seeds=episode_seeds)
    if loaded.device.type == 'cuda':
        torch.cuda.synchronize(loaded.device)
        torch.cuda.reset_peak_memory_stats(loaded.device)
    warmup_seconds = time.perf_counter() - warmup_started

    writer = ShardedTensorWriter(
        raw_dir,
        'scene_values',
        SCENE_VALUE_DTYPES,
        shard_rows=args.shard_rows,
        background=not args.foreground_writer,
        queue_depth=args.writer_queue_depth,
    )
    accumulator = DeviceTableAccumulator(
        writer, transfer_interval=args.transfer_interval
    )
    base_manifest = {
        'format_version': 1,
        'purpose': 'mergeability_scene_value_rollout',
        'created_at_utc': _utc_now(),
        'status': 'running',
        'git_revision': _git_revision(),
        'physics_identity': PHYSICS_IDENTITY,
        'simulator_config': asdict(config),
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'checkpoint_progress': loaded.progress,
        'policy': 'SAB-128 greedy',
        'scene_value_definition': 'sum(mergeability * pi * physics_radius^2)',
        'device': str(loaded.device),
        'cuda_device_name': (
            torch.cuda.get_device_name(loaded.device)
            if loaded.device.type == 'cuda' else None
        ),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'warmup_seconds': warmup_seconds,
        'parameters': {
            'num_envs': args.num_envs,
            'decision_steps': args.decision_steps,
            'seed_base': args.seed_base,
            'seed_stride': SEED_STRIDE,
            'warmup_steps': args.warmup_steps,
            'transfer_interval': args.transfer_interval,
            'shard_rows': args.shard_rows,
            'background_writer': not args.foreground_writer,
            'writer_queue_depth': args.writer_queue_depth,
        },
        'table_schema': {
            name: str(dtype) for name, dtype in SCENE_VALUE_DTYPES.items()
        },
    }
    _write_manifest(manifest_path, base_manifest)

    previous_value = torch.zeros(
        args.num_envs, dtype=torch.float32, device=loaded.device
    )
    previous_valid = torch.zeros(
        args.num_envs, dtype=torch.bool, device=loaded.device
    )
    next_episode_id = args.num_envs
    completed_episodes = 0
    started = time.perf_counter()
    status = 'complete'
    failure = None
    try:
        for decision_step in range(1, args.decision_steps + 1):
            observation = simulator.observe()
            state = TensorState.from_observation(
                observation, physics_fps=config.physics_fps
            )
            actions = loaded.model(state).argmax(dim=1)
            step = simulator.step(actions)
            current = step.observation
            current_state = TensorState.from_observation(
                current, physics_fps=config.physics_fps
            )
            result = calculator(current_state)
            scene_value, occupied_area, weighted_mean = (
                scene_mergeability_values(current_state, result)
            )
            delta, delta_valid = scene_mergeability_delta(
                scene_value, previous_value, previous_valid
            )
            finished = step.physics.done | step.physics.truncated
            accumulator.append({
                'environment_id': env_ids,
                'episode_id': episode_ids,
                'episode_seed': episode_seeds,
                'decision_step': torch.full_like(env_ids, decision_step),
                'episode_drop': current.step_count,
                'score': current.score,
                'fruit_count': current.fruit_count,
                'max_level': current.max_level,
                'scene_mergeability': scene_value,
                'occupied_area': occupied_area,
                'area_weighted_mean': weighted_mean,
                'delta': delta,
                'delta_valid': delta_valid,
                'done': finished,
            })
            accumulator.advance()
            previous_value.copy_(scene_value)
            previous_valid.fill_(True)

            finished_count = int(finished.sum().item())
            if finished_count:
                completed_episodes += finished_count
                next_episode_id = _reset_finished(
                    simulator,
                    finished,
                    episode_ids,
                    episode_seeds,
                    next_episode_id=next_episode_id,
                    seed_base=args.seed_base,
                )
                previous_valid[finished] = False

            if (
                    decision_step % args.progress_interval == 0
                    or decision_step == args.decision_steps):
                elapsed = time.perf_counter() - started
                print(json.dumps({
                    'phase': 'collect',
                    'decision_step': decision_step,
                    'decision_steps': args.decision_steps,
                    'transitions': decision_step * args.num_envs,
                    'completed_episodes': completed_episodes,
                    'scene_value_mean': float(scene_value.mean().item()),
                    'valid_delta_mean': float(
                        delta[delta_valid].mean().item()
                        if bool(delta_valid.any().item()) else 0.0
                    ),
                    'elapsed_seconds': elapsed,
                    'env_steps_per_second': (
                        decision_step * args.num_envs / max(elapsed, 1e-9)
                    ),
                }, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        status = 'interrupted'
    except Exception as error:
        status = 'failed'
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        close_error = None
        try:
            accumulator.close()
        except Exception as error:  # pragma: no cover - I/O failure path
            close_error = error
        if loaded.device.type == 'cuda':
            torch.cuda.synchronize(loaded.device)
        elapsed = time.perf_counter() - started
        _write_manifest(
            manifest_path,
            base_manifest,
            status=status if close_error is None else 'failed',
            updated_at_utc=_utc_now(),
            decision_steps=(
                writer.total_rows // args.num_envs
                if args.num_envs else 0
            ),
            transitions=writer.total_rows,
            completed_episodes=completed_episodes,
            unique_episodes_started=next_episode_id,
            elapsed_seconds=elapsed,
            env_steps_per_second=writer.total_rows / max(elapsed, 1e-9),
            table_rows=writer.total_rows,
            table_shards=writer.shard_count,
            peak_cuda_allocated_bytes=(
                int(torch.cuda.max_memory_allocated(loaded.device))
                if loaded.device.type == 'cuda' else None
            ),
            peak_cuda_reserved_bytes=(
                int(torch.cuda.max_memory_reserved(loaded.device))
                if loaded.device.type == 'cuda' else None
            ),
            failure=failure or (str(close_error) if close_error else None),
        )
        if close_error is not None:
            raise close_error
    result = json.loads(manifest_path.read_text(encoding='utf-8'))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    collect(parse_args(argv))


if __name__ == '__main__':
    main()
