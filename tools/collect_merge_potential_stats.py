"""用SAB-128批量采集水果未来合成时间，并生成统计表。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
    END_COLLECTOR_STOP,
    END_DROP_LIMIT,
    END_NATURAL,
    END_SIMULATOR_TRUNCATED,
    EPISODE_DTYPES,
    FruitSnapshotSampler,
    MERGE_SOURCE_DTYPES,
    SNAPSHOT_DTYPES,
    ShardedTensorWriter,
    extract_episode_rows,
    extract_merge_sources,
    extract_snapshot_features,
    summarize_dataset,
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


SAB128_SHA256 = (
    'fc40b9019c65ecba8502f4334d1418b4f93c0e54e984d42ccc4d0b477bddca07'
)
DEFAULT_SAB128_CHECKPOINT = (
    PROJECT_ROOT
    / 'runs'
    / 'cloud_rtx5090_auxiliary_action_structured_branch_seed20260811_128m'
    / 'checkpoints'
    / 'final.pt'
)
SEED_STRIDE = 1_000_003


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    os.replace(temporary, path)


def _git_revision():
    try:
        return subprocess.run(
            ('git', 'rev-parse', 'HEAD'),
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _parse_horizons(value):
    horizons = tuple(
        int(item.strip()) for item in str(value).split(',') if item.strip()
    )
    if not horizons or any(item <= 0 for item in horizons):
        raise argparse.ArgumentTypeError(
            'horizons must be comma-separated positive integers'
        )
    return horizons


def _add_collect_arguments(parser):
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--checkpoint', type=Path, default=DEFAULT_SAB128_CHECKPOINT
    )
    parser.add_argument(
        '--expected-checkpoint-sha256', default=SAB128_SHA256,
        help='不匹配时拒绝采集；传空字符串可显式关闭检查。',
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--physics-fps', type=int, choices=(30, 120), default=30)
    parser.add_argument('--episodes', type=int, default=20_000)
    parser.add_argument('--parallel-envs', type=int, default=512)
    parser.add_argument('--seed-base', type=int, default=53_000_000)
    parser.add_argument(
        '--max-drops', type=int, default=0,
        help='0表示不设投放上限；正数只作为诊断截断并标记为未知。',
    )
    parser.add_argument(
        '--max-wall-seconds', type=float, default=0.0,
        help='0表示不限采集墙钟；正数到达后把在途对局标记为截断未知。',
    )
    parser.add_argument('--snapshot-stride', type=int, default=8)
    parser.add_argument('--max-snapshots-per-fruit', type=int, default=32)
    parser.add_argument('--snapshots-per-scale', type=int, default=4)
    parser.add_argument('--feature-env-chunk', type=int, default=256)
    parser.add_argument('--transfer-interval', type=int, default=8)
    parser.add_argument('--shard-rows', type=int, default=1_000_000)
    parser.add_argument('--writer-queue-depth', type=int, default=2)
    parser.add_argument(
        '--background-writer',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--compile-model',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--compile-mode',
        choices=('default', 'reduce-overhead', 'max-autotune'),
        default='reduce-overhead',
    )
    parser.add_argument(
        '--autocast-bfloat16',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='可能改变少量动作排序，正式使用前需与FP32做同seed核验。',
    )
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--progress-interval-seconds', type=float, default=10.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='采集并分析水果距离下一次合成的投放次数。'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    collect = subparsers.add_parser(
        'collect', help='在CUDA批量环境中采集原始张量表。'
    )
    _add_collect_arguments(collect)
    summarize = subparsers.add_parser(
        'summarize', help='关联原始事件并生成标签和CSV统计表。'
    )
    summarize.add_argument('dataset_dir', type=Path)
    summarize.add_argument('--output-dir', type=Path)
    summarize.add_argument(
        '--horizons', type=_parse_horizons,
        default=_parse_horizons('1,2,4,8,16,32,64,128,256,512,1024'),
    )
    summarize.add_argument('--factor-bins', type=int, default=10)
    summarize.add_argument('--interaction-bins', type=int, default=5)
    summarize.add_argument('--peer-count-cap', type=int, default=8)
    summarize.add_argument('--labeled-shard-rows', type=int, default=1_000_000)
    return parser.parse_args(argv)


def _validate_collect_args(args):
    for name in (
            'episodes', 'parallel_envs', 'snapshot_stride',
            'max_snapshots_per_fruit', 'feature_env_chunk',
            'snapshots_per_scale', 'transfer_interval', 'shard_rows',
            'writer_queue_depth'):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f'{name} must be positive')
    if args.max_drops < 0:
        raise ValueError('max_drops cannot be negative')
    if args.max_wall_seconds < 0:
        raise ValueError('max_wall_seconds cannot be negative')
    if args.warmup_steps < 0:
        raise ValueError('warmup_steps cannot be negative')
    if args.progress_interval_seconds <= 0:
        raise ValueError('progress_interval_seconds must be positive')


def _prepare_output(output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f'output directory is not empty: {output_dir}'
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'raw').mkdir(exist_ok=True)
    return output_dir


def _episode_seeds(episodes, seed_base):
    return (
        torch.arange(int(episodes), dtype=torch.int64)
        .mul_(SEED_STRIDE)
        .add_(int(seed_base))
    )


def _autocast_context(args, device):
    if not args.autocast_bfloat16:
        return nullcontext()
    if device.type != 'cuda':
        raise RuntimeError('bfloat16 autocast is only enabled for CUDA collection')
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)


def _policy_actions(model, observation, args, physics_fps):
    state = TensorState.from_observation(
        observation, physics_fps=physics_fps
    )
    with _autocast_context(args, observation.positions.device):
        return model(state).argmax(dim=1)


def _write_manifest(path, base, **updates):
    payload = dict(base)
    payload.update(updates)
    _atomic_json(path, payload)


@torch.inference_mode()
def collect(args):
    _validate_collect_args(args)
    loaded = load_viewer_model(args.checkpoint, device=args.device)
    expected_sha = str(args.expected_checkpoint_sha256).strip().lower()
    if expected_sha and loaded.checkpoint_sha256.lower() != expected_sha:
        raise ValueError(
            'checkpoint SHA-256 mismatch: '
            f'{loaded.checkpoint_sha256} != {expected_sha}'
        )
    config = viewer_simulator_config(
        args.physics_fps, loaded.model_config, loaded.device
    )
    if config.drop_fast_forward:
        raise RuntimeError('collection requires the current no-fast-forward physics')
    output_dir = _prepare_output(args.output_dir)
    manifest_path = output_dir / 'manifest.json'
    batch_size = min(int(args.parallel_envs), int(args.episodes))
    simulator = TensorVectorSimulator(
        batch_size, config=config, device=loaded.device
    )
    all_seeds = _episode_seeds(args.episodes, args.seed_base)
    episode_ids = torch.arange(
        batch_size, dtype=torch.int64, device=loaded.device
    )
    slot_seeds = all_seeds[:batch_size].to(loaded.device)
    simulator.reset(seeds=slot_seeds)

    policy_model = loaded.model
    if args.compile_model:
        if not hasattr(torch, 'compile'):
            raise RuntimeError('torch.compile is unavailable')
        policy_model = torch.compile(
            policy_model, mode=args.compile_mode, dynamic=False
        )
    warmup_started = time.perf_counter()
    for _ in range(int(args.warmup_steps)):
        warmup_actions = _policy_actions(
            policy_model, simulator.observe(), args, config.physics_fps
        )
        simulator.step(warmup_actions)
    simulator.reset(seeds=slot_seeds)
    if loaded.device.type == 'cuda':
        torch.cuda.synchronize(loaded.device)
        torch.cuda.reset_peak_memory_stats(loaded.device)
    warmup_seconds = time.perf_counter() - warmup_started

    tracker = FruitSnapshotSampler(
        batch_size,
        config.max_fruits,
        device=loaded.device,
        snapshot_stride=args.snapshot_stride,
        max_snapshots_per_fruit=args.max_snapshots_per_fruit,
        snapshots_per_scale=args.snapshots_per_scale,
    )
    raw_dir = output_dir / 'raw'
    writers = {
        'snapshots': ShardedTensorWriter(
            raw_dir, 'snapshots', SNAPSHOT_DTYPES,
            shard_rows=args.shard_rows,
            background=args.background_writer,
            queue_depth=args.writer_queue_depth,
        ),
        'merge_sources': ShardedTensorWriter(
            raw_dir, 'merge_sources', MERGE_SOURCE_DTYPES,
            shard_rows=args.shard_rows,
            background=args.background_writer,
            queue_depth=args.writer_queue_depth,
        ),
        'episodes': ShardedTensorWriter(
            raw_dir, 'episodes', EPISODE_DTYPES,
            shard_rows=max(1, min(args.shard_rows, args.episodes)),
            background=args.background_writer,
            queue_depth=args.writer_queue_depth,
        ),
    }
    accumulators = {
        name: DeviceTableAccumulator(
            writer, transfer_interval=args.transfer_interval
        )
        for name, writer in writers.items()
    }
    base_manifest = {
        'format_version': 1,
        'purpose': 'merge_potential_t_merge_collection',
        'created_at_utc': _utc_now(),
        'status': 'running',
        'git_revision': _git_revision(),
        'physics_identity': PHYSICS_IDENTITY,
        'simulator_config': asdict(config),
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'checkpoint_progress': loaded.progress,
        'policy': 'greedy',
        'device': str(loaded.device),
        'cuda_device_name': (
            torch.cuda.get_device_name(loaded.device)
            if loaded.device.type == 'cuda' else None
        ),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'warmup_seconds': warmup_seconds,
        'parameters': {
            'episodes': int(args.episodes),
            'parallel_envs': batch_size,
            'seed_base': int(args.seed_base),
            'seed_stride': SEED_STRIDE,
            'max_drops': int(args.max_drops),
            'max_wall_seconds': float(args.max_wall_seconds),
            'snapshot_stride': int(args.snapshot_stride),
            'max_snapshots_per_fruit': int(args.max_snapshots_per_fruit),
            'snapshots_per_scale': int(args.snapshots_per_scale),
            'feature_env_chunk': int(args.feature_env_chunk),
            'transfer_interval': int(args.transfer_interval),
            'shard_rows': int(args.shard_rows),
            'background_writer': bool(args.background_writer),
            'writer_queue_depth': int(args.writer_queue_depth),
            'progress_interval_seconds': float(
                args.progress_interval_seconds
            ),
            'compile_model': bool(args.compile_model),
            'compile_mode': args.compile_mode,
            'autocast_bfloat16': bool(args.autocast_bfloat16),
            'warmup_steps': int(args.warmup_steps),
        },
        'table_schemas': {
            'snapshots': list(SNAPSHOT_DTYPES),
            'merge_sources': list(MERGE_SOURCE_DTYPES),
            'episodes': list(EPISODE_DTYPES),
        },
    }
    _write_manifest(manifest_path, base_manifest)

    completed = 0
    next_episode_id = batch_size
    transitions = 0
    decision_steps = 0
    status = 'complete'
    failure = None
    started = time.perf_counter()
    last_progress = started

    try:
        while completed < int(args.episodes):
            observation = simulator.observe()
            enabled = episode_ids >= 0
            sample_kinds = tracker.select(observation, enabled)
            accumulators['snapshots'].append(extract_snapshot_features(
                observation,
                sample_kinds,
                episode_ids,
                config,
                env_chunk_size=args.feature_env_chunk,
            ))
            actions = _policy_actions(
                policy_model, observation, args, config.physics_fps
            )
            result = simulator.step_masked(actions, enabled)
            transitions += int(enabled.sum().item())
            decision_steps += 1
            after = result.observation
            accumulators['merge_sources'].append(extract_merge_sources(
                result.physics.merge_events,
                episode_ids,
                after.step_count,
            ))
            capped = torch.zeros_like(enabled)
            if args.max_drops > 0:
                capped = after.step_count >= int(args.max_drops)
            finished = enabled & (
                result.physics.done | result.physics.truncated | capped
            )
            finished_rows = torch.nonzero(
                finished, as_tuple=False
            ).flatten()
            if finished_rows.numel() > 0:
                selected_done = result.physics.done[finished_rows]
                selected_truncated = result.physics.truncated[finished_rows]
                end_kinds = torch.full(
                    finished_rows.shape,
                    END_DROP_LIMIT,
                    dtype=torch.int8,
                    device=loaded.device,
                )
                end_kinds[selected_truncated] = END_SIMULATOR_TRUNCATED
                end_kinds[selected_done] = END_NATURAL
                accumulators['episodes'].append(extract_episode_rows(
                    after,
                    finished_rows,
                    episode_ids,
                    slot_seeds,
                    end_kinds,
                ))
                finished_count = int(finished_rows.numel())
                completed += finished_count
                if completed < int(args.episodes):
                    assign_count = min(
                        finished_count,
                        int(args.episodes) - next_episode_id,
                    )
                    replacements = torch.full(
                        (finished_count,),
                        -1,
                        dtype=torch.int64,
                        device=loaded.device,
                    )
                    reset_seeds = torch.arange(
                        finished_count,
                        dtype=torch.int64,
                        device=loaded.device,
                    ).add_(int(args.seed_base) + 9_000_000_000)
                    if assign_count > 0:
                        replacements[:assign_count] = torch.arange(
                            next_episode_id,
                            next_episode_id + assign_count,
                            dtype=torch.int64,
                            device=loaded.device,
                        )
                        reset_seeds[:assign_count] = all_seeds[
                            next_episode_id:next_episode_id + assign_count
                        ].to(loaded.device)
                        next_episode_id += assign_count
                    simulator.reset(finished, seeds=reset_seeds)
                    episode_ids[finished_rows] = replacements
                    slot_seeds[finished_rows] = reset_seeds
                    tracker.reset(finished_rows)
            for accumulator in accumulators.values():
                accumulator.advance()

            now = time.perf_counter()
            if now - last_progress >= args.progress_interval_seconds:
                elapsed = now - started
                rows = {
                    name: writer.total_rows
                    for name, writer in writers.items()
                }
                _write_manifest(
                    manifest_path,
                    base_manifest,
                    status='running',
                    updated_at_utc=_utc_now(),
                    completed_episodes=completed,
                    transitions=transitions,
                    decision_steps=decision_steps,
                    elapsed_seconds=elapsed,
                    env_steps_per_second=transitions / max(elapsed, 1e-9),
                    peak_cuda_allocated_bytes=(
                        int(torch.cuda.max_memory_allocated(loaded.device))
                        if loaded.device.type == 'cuda' else None
                    ),
                    peak_cuda_reserved_bytes=(
                        int(torch.cuda.max_memory_reserved(loaded.device))
                        if loaded.device.type == 'cuda' else None
                    ),
                    flushed_rows=rows,
                )
                print(json.dumps({
                    'completed_episodes': completed,
                    'target_episodes': int(args.episodes),
                    'transitions': transitions,
                    'env_steps_per_second': transitions / max(elapsed, 1e-9),
                    'flushed_rows': rows,
                }, ensure_ascii=False), flush=True)
                last_progress = now
            if (
                    args.max_wall_seconds > 0
                    and now - started >= args.max_wall_seconds):
                status = 'wall_time_reached'
                current = simulator.observe()
                active_rows = torch.nonzero(
                    episode_ids >= 0, as_tuple=False
                ).flatten()
                if active_rows.numel() > 0:
                    end_kinds = torch.full(
                        active_rows.shape,
                        END_COLLECTOR_STOP,
                        dtype=torch.int8,
                        device=loaded.device,
                    )
                    accumulators['episodes'].append(extract_episode_rows(
                        current,
                        active_rows,
                        episode_ids,
                        slot_seeds,
                        end_kinds,
                    ))
                break
    except KeyboardInterrupt:
        status = 'interrupted'
        observation = simulator.observe()
        active_rows = torch.nonzero(
            episode_ids >= 0, as_tuple=False
        ).flatten()
        if active_rows.numel() > 0:
            end_kinds = torch.full(
                active_rows.shape,
                END_COLLECTOR_STOP,
                dtype=torch.int8,
                device=loaded.device,
            )
            accumulators['episodes'].append(extract_episode_rows(
                observation,
                active_rows,
                episode_ids,
                slot_seeds,
                end_kinds,
            ))
    except Exception as error:
        status = 'failed'
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        close_error = None
        for accumulator in accumulators.values():
            try:
                accumulator.close()
            except Exception as error:  # pragma: no cover - I/O failure path
                close_error = close_error or error
        elapsed = time.perf_counter() - started
        _write_manifest(
            manifest_path,
            base_manifest,
            status=status if close_error is None else 'failed',
            updated_at_utc=_utc_now(),
            completed_episodes=completed,
            transitions=transitions,
            decision_steps=decision_steps,
            elapsed_seconds=elapsed,
            env_steps_per_second=transitions / max(elapsed, 1e-9),
            peak_cuda_allocated_bytes=(
                int(torch.cuda.max_memory_allocated(loaded.device))
                if loaded.device.type == 'cuda' else None
            ),
            peak_cuda_reserved_bytes=(
                int(torch.cuda.max_memory_reserved(loaded.device))
                if loaded.device.type == 'cuda' else None
            ),
            table_rows={
                name: writer.total_rows for name, writer in writers.items()
            },
            table_shards={
                name: writer.shard_count for name, writer in writers.items()
            },
            failure=failure or (str(close_error) if close_error else None),
        )
        if close_error is not None:
            raise close_error

    result = json.loads(manifest_path.read_text(encoding='utf-8'))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'collect':
        collect(args)
        return
    result = summarize_dataset(
        args.dataset_dir,
        output_dir=args.output_dir,
        horizons=args.horizons,
        factor_bins=args.factor_bins,
        interaction_bins=args.interaction_bins,
        peer_count_cap=args.peer_count_cap,
        labeled_shard_rows=args.labeled_shard_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
