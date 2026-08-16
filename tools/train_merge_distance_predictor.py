"""采集、标注、训练和评估逐水果合成步距预测器。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from daxigua.rl.merge_distance import (  # noqa: E402
    DEFAULT_MERGE_HORIZONS,
    MergeDistanceConfig,
    MergeDistancePredictor,
    cumulative_merge_probabilities,
    merge_distance_loss,
    merge_distance_targets,
)
from daxigua.rl.merge_distance_data import (  # noqa: E402
    SCENE_DTYPES,
    SceneSnapshotSampler,
    extract_scene_rows,
    label_scene_dataset,
    labeled_scene_paths,
    load_labeled_scene_shard,
    scene_columns_to_state,
    split_mask,
)
from daxigua.rl.merge_potential_stats import (  # noqa: E402
    DeviceTableAccumulator,
    END_COLLECTOR_STOP,
    END_DROP_LIMIT,
    END_NATURAL,
    END_SIMULATOR_TRUNCATED,
    EPISODE_DTYPES,
    MERGE_SOURCE_DTYPES,
    ShardedTensorWriter,
    extract_episode_rows,
    extract_merge_sources,
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


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
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


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _parse_horizons(value):
    horizons = tuple(
        int(item.strip()) for item in str(value).split(',') if item.strip()
    )
    if not horizons or any(item <= 0 for item in horizons):
        raise argparse.ArgumentTypeError(
            'horizons must be comma-separated positive integers'
        )
    if any(left >= right for left, right in zip(horizons, horizons[1:])):
        raise argparse.ArgumentTypeError('horizons must be strictly increasing')
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
    parser.add_argument('--seed-base', type=int, default=71_000_000)
    parser.add_argument(
        '--max-drops', type=int, default=0,
        help='0表示不设投放上限；正数边界会标记为截断未知。',
    )
    parser.add_argument('--max-wall-seconds', type=float, default=0.0)
    parser.add_argument('--scene-stride', type=int, default=8)
    parser.add_argument('--max-scenes-per-episode', type=int, default=1024)
    parser.add_argument('--transfer-interval', type=int, default=8)
    parser.add_argument('--scene-shard-rows', type=int, default=32_768)
    parser.add_argument('--event-shard-rows', type=int, default=1_000_000)
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
        help='正式采集默认保持FP32动作排序。',
    )
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--progress-interval-seconds', type=float, default=10.0)


def _add_train_arguments(parser):
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--grad-clip-norm', type=float, default=5.0)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--edge-hidden-dim', type=int, default=64)
    parser.add_argument('--message-layers', type=int, default=2)
    parser.add_argument('--queue-hidden-dim', type=int, default=32)
    parser.add_argument('--level-embedding-dim', type=int, default=12)
    parser.add_argument('--balance-power', type=float, default=0.5)
    parser.add_argument('--max-balance-weight', type=float, default=8.0)
    parser.add_argument('--seed', type=int, default=20260817)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-train-batches', type=int, default=0)
    parser.add_argument('--max-eval-batches', type=int, default=0)
    parser.add_argument('--early-stopping-patience', type=int, default=4)
    parser.add_argument('--telemetry-interval-seconds', type=float, default=10.0)
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
        default=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='SAB-128策略条件下的逐水果合成步距预测器。'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    collect_parser = subparsers.add_parser(
        'collect', help='采集完整稳定场景与未来事件。'
    )
    _add_collect_arguments(collect_parser)
    label_parser = subparsers.add_parser(
        'label', help='把场景与未来合成事件关联为逐槽位标签。'
    )
    label_parser.add_argument('dataset_dir', type=Path)
    label_parser.add_argument('--output-dir', type=Path)
    label_parser.add_argument(
        '--horizons', type=_parse_horizons,
        default=DEFAULT_MERGE_HORIZONS,
    )
    label_parser.add_argument('--shard-rows', type=int, default=32_768)
    train_parser = subparsers.add_parser(
        'train', help='训练轻量合成步距图网络。'
    )
    _add_train_arguments(train_parser)
    evaluate_parser = subparsers.add_parser(
        'evaluate', help='在固定数据划分上评估已保存预测器。'
    )
    evaluate_parser.add_argument('checkpoint', type=Path)
    evaluate_parser.add_argument('dataset_dir', type=Path)
    evaluate_parser.add_argument(
        '--split', choices=('validation', 'test'), default='test'
    )
    evaluate_parser.add_argument('--device', default='cuda')
    evaluate_parser.add_argument('--batch-size', type=int, default=512)
    evaluate_parser.add_argument('--max-eval-batches', type=int, default=0)
    evaluate_parser.add_argument(
        '--autocast-bfloat16',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _validate_collect_args(args):
    positive = (
        'episodes', 'parallel_envs', 'scene_stride',
        'max_scenes_per_episode', 'transfer_interval', 'scene_shard_rows',
        'event_shard_rows', 'writer_queue_depth',
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f'{name} must be positive')
    if args.max_drops < 0 or args.max_wall_seconds < 0:
        raise ValueError('collection limits cannot be negative')
    if args.warmup_steps < 0 or args.progress_interval_seconds <= 0:
        raise ValueError('warmup and progress values are invalid')


def _prepare_empty_output(output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'raw').mkdir(exist_ok=True)
    return output_dir


def _episode_seeds(episodes, seed_base):
    return (
        torch.arange(int(episodes), dtype=torch.int64)
        .mul_(SEED_STRIDE)
        .add_(int(seed_base))
    )


def _autocast(enabled, device):
    if not enabled:
        return nullcontext()
    if device.type != 'cuda':
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)


def _policy_actions(model, observation, args, physics_fps):
    state = TensorState.from_observation(
        observation, physics_fps=physics_fps
    )
    with _autocast(args.autocast_bfloat16, observation.positions.device):
        return model(state).argmax(dim=1)


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
    simulator_config = viewer_simulator_config(
        args.physics_fps, loaded.model_config, loaded.device
    )
    if simulator_config.drop_fast_forward:
        raise RuntimeError('collection requires no-fast-forward physics')
    output_dir = _prepare_empty_output(args.output_dir)
    manifest_path = output_dir / 'manifest.json'
    batch_size = min(int(args.parallel_envs), int(args.episodes))
    simulator = TensorVectorSimulator(
        batch_size, config=simulator_config, device=loaded.device
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
        actions = _policy_actions(
            policy_model, simulator.observe(), args, simulator_config.physics_fps
        )
        simulator.step(actions)
    simulator.reset(seeds=slot_seeds)
    if loaded.device.type == 'cuda':
        torch.cuda.synchronize(loaded.device)
        torch.cuda.reset_peak_memory_stats(loaded.device)
    warmup_seconds = time.perf_counter() - warmup_started

    sampler = SceneSnapshotSampler(
        batch_size,
        device=loaded.device,
        scene_stride=args.scene_stride,
        max_scenes_per_episode=args.max_scenes_per_episode,
    )
    raw_dir = output_dir / 'raw'
    writers = {
        'scenes': ShardedTensorWriter(
            raw_dir,
            'scenes',
            SCENE_DTYPES,
            shard_rows=args.scene_shard_rows,
            background=args.background_writer,
            queue_depth=args.writer_queue_depth,
        ),
        'merge_sources': ShardedTensorWriter(
            raw_dir,
            'merge_sources',
            MERGE_SOURCE_DTYPES,
            shard_rows=args.event_shard_rows,
            background=args.background_writer,
            queue_depth=args.writer_queue_depth,
        ),
        'episodes': ShardedTensorWriter(
            raw_dir,
            'episodes',
            EPISODE_DTYPES,
            shard_rows=max(1, min(args.event_shard_rows, args.episodes)),
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
        'purpose': 'merge_distance_predictor_collection',
        'created_at_utc': _utc_now(),
        'status': 'running',
        'git_revision': _git_revision(),
        'physics_identity': PHYSICS_IDENTITY,
        'simulator_config': asdict(simulator_config),
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
            'scene_stride': int(args.scene_stride),
            'max_scenes_per_episode': int(args.max_scenes_per_episode),
            'transfer_interval': int(args.transfer_interval),
            'scene_shard_rows': int(args.scene_shard_rows),
            'event_shard_rows': int(args.event_shard_rows),
            'background_writer': bool(args.background_writer),
            'compile_model': bool(args.compile_model),
            'compile_mode': args.compile_mode,
            'autocast_bfloat16': bool(args.autocast_bfloat16),
        },
        'table_schemas': {
            'scenes': list(SCENE_DTYPES),
            'merge_sources': list(MERGE_SOURCE_DTYPES),
            'episodes': list(EPISODE_DTYPES),
        },
    }
    _atomic_json(manifest_path, base_manifest)

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
            selected = sampler.select(observation, enabled)
            accumulators['scenes'].append(extract_scene_rows(
                observation, selected, episode_ids
            ))
            actions = _policy_actions(
                policy_model,
                observation,
                args,
                simulator_config.physics_fps,
            )
            result = simulator.step_masked(actions, enabled)
            transitions += int(enabled.sum().item())
            decision_steps += 1
            after = result.observation
            accumulators['merge_sources'].append(extract_merge_sources(
                result.physics.merge_events, episode_ids, after.step_count
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
                    sampler.reset(finished_rows)
            for accumulator in accumulators.values():
                accumulator.advance()

            now = time.perf_counter()
            if now - last_progress >= args.progress_interval_seconds:
                elapsed = now - started
                progress = dict(
                    status='running',
                    updated_at_utc=_utc_now(),
                    completed_episodes=completed,
                    transitions=transitions,
                    decision_steps=decision_steps,
                    elapsed_seconds=elapsed,
                    env_steps_per_second=transitions / max(elapsed, 1e-9),
                    flushed_rows={
                        name: writer.total_rows
                        for name, writer in writers.items()
                    },
                )
                _atomic_json(manifest_path, {**base_manifest, **progress})
                print(json.dumps(progress, ensure_ascii=False), flush=True)
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
                    accumulators['episodes'].append(extract_episode_rows(
                        current,
                        active_rows,
                        episode_ids,
                        slot_seeds,
                        torch.full(
                            active_rows.shape,
                            END_COLLECTOR_STOP,
                            dtype=torch.int8,
                            device=loaded.device,
                        ),
                    ))
                break
    except KeyboardInterrupt:
        status = 'interrupted'
        observation = simulator.observe()
        active_rows = torch.nonzero(
            episode_ids >= 0, as_tuple=False
        ).flatten()
        if active_rows.numel() > 0:
            accumulators['episodes'].append(extract_episode_rows(
                observation,
                active_rows,
                episode_ids,
                slot_seeds,
                torch.full(
                    active_rows.shape,
                    END_COLLECTOR_STOP,
                    dtype=torch.int8,
                    device=loaded.device,
                ),
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
            except Exception as error:  # pragma: no cover - I/O failure
                close_error = close_error or error
        elapsed = time.perf_counter() - started
        final = {
            **base_manifest,
            'status': status if close_error is None else 'failed',
            'updated_at_utc': _utc_now(),
            'completed_episodes': completed,
            'transitions': transitions,
            'decision_steps': decision_steps,
            'elapsed_seconds': elapsed,
            'env_steps_per_second': transitions / max(elapsed, 1e-9),
            'peak_cuda_allocated_bytes': (
                int(torch.cuda.max_memory_allocated(loaded.device))
                if loaded.device.type == 'cuda' else None
            ),
            'peak_cuda_reserved_bytes': (
                int(torch.cuda.max_memory_reserved(loaded.device))
                if loaded.device.type == 'cuda' else None
            ),
            'table_rows': {
                name: writer.total_rows for name, writer in writers.items()
            },
            'table_shards': {
                name: writer.shard_count for name, writer in writers.items()
            },
            'failure': failure or (str(close_error) if close_error else None),
        }
        _atomic_json(manifest_path, final)
        if close_error is not None:
            raise close_error
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return final


def _load_dataset_manifest(dataset_dir):
    path = Path(dataset_dir) / 'dataset_manifest.json'
    if not path.exists():
        raise FileNotFoundError(f'missing predictor dataset manifest: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('purpose') != 'merge_distance_predictor_dataset':
        raise ValueError('dataset manifest has the wrong purpose')
    if payload.get('status') not in (None, 'complete'):
        raise ValueError('predictor dataset labeling is not complete')
    return payload


def _geometry_from_manifest(manifest):
    simulator = manifest.get('source_identity', {}).get(
        'simulator_config', {}
    ) or {}
    return {
        'board_width': float(simulator.get('board_width', 560)),
        'board_height': float(simulator.get('board_height', 1120)),
        'spawn_y': float(simulator.get('spawn_y', 252)),
        'wall_width': float(simulator.get('wall_width', 20)),
        'gravity_y': float(simulator.get('gravity_y', 1800.0)),
    }


def _physics_fps(manifest):
    return float(
        manifest.get('source_identity', {}).get('physics_fps', 30)
    )


def _class_balance_table(manifest, config, power, maximum):
    counts = torch.as_tensor(
        manifest['class_counts_by_level']['train'], dtype=torch.float64
    )
    if tuple(counts.shape) != (12, config.merge_class_count):
        raise ValueError('dataset class counts do not match model classes')
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].pow(-float(power))
    normalization = (
        (weights * counts).sum() / counts.sum().clamp_min(1.0)
    )
    weights = weights / normalization.clamp_min(1e-12)
    return weights.clamp_max(float(maximum)).to(torch.float32)


def _batch_indices(columns, split, batch_size, *, shuffle, generator):
    eligible = torch.nonzero(
        split_mask(columns['episode_id'], split), as_tuple=False
    ).flatten()
    if shuffle and eligible.numel() > 1:
        eligible = eligible[torch.randperm(
            eligible.numel(), generator=generator
        )]
    for start in range(0, int(eligible.numel()), int(batch_size)):
        yield eligible[start:start + int(batch_size)]


class _MetricAccumulator:
    def __init__(self, horizons):
        self.horizons = tuple(horizons)
        self.rows = 0
        self.nll_sum = 0.0
        self.exact = 0
        self.adjacent = 0
        self.terminal_brier = 0.0
        self.horizon_brier = torch.zeros(len(self.horizons), dtype=torch.float64)
        self.weight = 0.0
        self.weighted_nll = 0.0
        self.level_rows = torch.zeros(12, dtype=torch.int64)
        self.level_nll = torch.zeros(12, dtype=torch.float64)
        self.level_exact = torch.zeros(12, dtype=torch.int64)

    @torch.no_grad()
    def add(self, probabilities, columns, rows, active):
        outcomes = columns['outcome'].index_select(0, rows).to(
            probabilities.device
        )
        t_merge = columns['t_merge'].index_select(0, rows).to(
            probabilities.device
        )
        levels = columns['levels'].index_select(0, rows).to(
            probabilities.device
        )
        weights = columns['fruit_weight'].index_select(0, rows).to(
            probabilities.device
        )
        targets, resolved = merge_distance_targets(
            outcomes, t_merge, self.horizons
        )
        valid = active & resolved
        if not bool(valid.any().item()):
            return
        selected_probabilities = probabilities[valid].float()
        selected_targets = targets[valid]
        selected_levels = levels[valid].to(torch.int64)
        selected_weights = weights[valid].float()
        nll = -torch.log(selected_probabilities.gather(
            1, selected_targets[:, None]
        ).squeeze(1).clamp_min(1e-9))
        predicted = selected_probabilities.argmax(dim=1)
        terminal_class = len(self.horizons) + 1
        time_true = selected_targets != terminal_class
        time_predicted = predicted != terminal_class
        adjacent = (
            time_true
            & time_predicted
            & ((predicted - selected_targets).abs() <= 1)
        ) | ((~time_true) & (~time_predicted))
        self.rows += int(selected_targets.numel())
        self.nll_sum += float(nll.sum().item())
        self.exact += int((predicted == selected_targets).sum().item())
        self.adjacent += int(adjacent.sum().item())
        terminal_truth = (selected_targets == terminal_class).float()
        self.terminal_brier += float((
            selected_probabilities[:, terminal_class] - terminal_truth
        ).square().sum().item())
        cdf = cumulative_merge_probabilities(
            selected_probabilities, self.horizons
        )
        horizon_indices = torch.arange(
            len(self.horizons), device=probabilities.device
        )
        truth = (
            selected_targets[:, None] <= horizon_indices[None, :]
        ).float()
        self.horizon_brier += (cdf - truth).square().sum(dim=0).cpu()
        self.weight += float(selected_weights.sum().item())
        self.weighted_nll += float((nll * selected_weights).sum().item())
        for level in range(1, 12):
            chosen = selected_levels == level
            if not bool(chosen.any().item()):
                continue
            self.level_rows[level] += int(chosen.sum().item())
            self.level_nll[level] += float(nll[chosen].sum().item())
            self.level_exact[level] += int(
                (predicted[chosen] == selected_targets[chosen]).sum().item()
            )

    def result(self):
        denominator = max(1, self.rows)
        per_level = {}
        for level in range(1, 12):
            rows = int(self.level_rows[level].item())
            per_level[str(level)] = {
                'samples': rows,
                'nll': (
                    float(self.level_nll[level].item()) / rows
                    if rows else None
                ),
                'exact_bin_accuracy': (
                    float(self.level_exact[level].item()) / rows
                    if rows else None
                ),
            }
        return {
            'resolved_fruit_samples': self.rows,
            'nll': self.nll_sum / denominator,
            'lifecycle_weighted_nll': (
                self.weighted_nll / max(1e-9, self.weight)
            ),
            'exact_bin_accuracy': self.exact / denominator,
            'adjacent_bin_accuracy': self.adjacent / denominator,
            'terminal_unmerged_brier': self.terminal_brier / denominator,
            'horizon_brier': {
                str(horizon): float(value.item()) / denominator
                for horizon, value in zip(self.horizons, self.horizon_brier)
            },
            'by_level': per_level,
        }


def _evaluate_model(
        model,
        paths,
        manifest,
        *,
        split,
        device,
        batch_size,
        autocast_bfloat16,
        max_batches=0):
    model.eval()
    metrics = _MetricAccumulator(model.config.horizons)
    batches = 0
    with torch.inference_mode():
        for path in paths:
            columns = load_labeled_scene_shard(path)
            generator = torch.Generator().manual_seed(0)
            for rows in _batch_indices(
                    columns,
                    split,
                    batch_size,
                    shuffle=False,
                    generator=generator):
                state = scene_columns_to_state(
                    columns,
                    rows,
                    device=device,
                    physics_fps=_physics_fps(manifest),
                    non_blocking=True,
                )
                with _autocast(autocast_bfloat16, device):
                    output = model(state)
                metrics.add(
                    output.probabilities,
                    columns,
                    rows,
                    state.active,
                )
                batches += 1
                if max_batches > 0 and batches >= int(max_batches):
                    return metrics.result()
    return metrics.result()


def _checkpoint_payload(model, manifest, args, epoch, metrics):
    return {
        'format_version': 1,
        'purpose': 'merge_distance_predictor',
        'created_at_utc': _utc_now(),
        'git_revision': _git_revision(),
        'model_config': model.config.to_dict(),
        'geometry_config': model.geometry_config,
        'model_state': model.state_dict(),
        'dataset_identity': {
            'source_dataset': manifest.get('source_dataset'),
            'source_identity': manifest.get('source_identity'),
            'horizons': manifest.get('horizons'),
        },
        'training': {
            'epoch': int(epoch),
            'batch_size': int(args.batch_size),
            'learning_rate': float(args.learning_rate),
            'weight_decay': float(args.weight_decay),
            'balance_power': float(args.balance_power),
            'max_balance_weight': float(args.max_balance_weight),
            'seed': int(args.seed),
        },
        'metrics': metrics,
    }


def train(args):
    if args.num_workers != 0:
        raise ValueError(
            'shard streaming currently keeps num_workers=0 to avoid '
            'duplicating large scene shards in host memory'
        )
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError('epochs and batch_size must be positive')
    if args.telemetry_interval_seconds <= 0:
        raise ValueError('telemetry_interval_seconds must be positive')
    manifest = _load_dataset_manifest(args.dataset_dir)
    horizons = tuple(manifest['horizons'])
    config = MergeDistanceConfig(
        hidden_dim=args.hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        message_layers=args.message_layers,
        queue_hidden_dim=args.queue_hidden_dim,
        level_embedding_dim=args.level_embedding_dim,
        horizons=horizons,
    )
    output_dir = _prepare_empty_output(args.output_dir)
    paths = labeled_scene_paths(args.dataset_dir)
    if not paths:
        raise ValueError('predictor dataset has no labeled scene shards')
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    model = MergeDistancePredictor(
        config, **_geometry_from_manifest(manifest)
    ).to(device)
    class_balance = _class_balance_table(
        manifest,
        config,
        args.balance_power,
        args.max_balance_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=(device.type == 'cuda'),
    )
    train_model = model
    if args.compile_model:
        if not hasattr(torch, 'compile'):
            raise RuntimeError('torch.compile is unavailable')
        train_model = torch.compile(
            model, mode=args.compile_mode, dynamic=False
        )

    history_path = output_dir / 'metrics.jsonl'
    best_nll = float('inf')
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_manifest = {
        'format_version': 1,
        'purpose': 'merge_distance_predictor_training',
        'status': 'running',
        'phase': 'initializing',
        'created_at_utc': _utc_now(),
        'git_revision': _git_revision(),
        'dataset_dir': str(Path(args.dataset_dir).resolve()),
        'model_config': config.to_dict(),
        'geometry_config': model.geometry_config,
        'parameter_count': parameter_count,
        'device': str(device),
        'cuda_device_name': (
            torch.cuda.get_device_name(device)
            if device.type == 'cuda' else None
        ),
        'arguments': vars(args) | {
            'dataset_dir': str(args.dataset_dir),
            'output_dir': str(args.output_dir),
        },
        'total_epochs': int(args.epochs),
        'current_epoch': 0,
        'completed_epochs': 0,
        'progress_fraction': 0.0,
    }
    _atomic_json(output_dir / 'manifest.json', run_manifest)

    generator = torch.Generator().manual_seed(args.seed)
    last_completed_epoch = 0
    last_telemetry = time.perf_counter()
    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            shuffled_paths = list(paths)
            random.Random(args.seed + epoch).shuffle(shuffled_paths)
            loss_sum = 0.0
            batches = 0
            fruit_samples = 0
            epoch_started = time.perf_counter()
            for path in shuffled_paths:
                columns = load_labeled_scene_shard(path)
                for rows in _batch_indices(
                        columns,
                        'train',
                        args.batch_size,
                        shuffle=True,
                        generator=generator):
                    state = scene_columns_to_state(
                        columns,
                        rows,
                        device=device,
                        physics_fps=_physics_fps(manifest),
                        non_blocking=True,
                    )
                    outcomes = columns['outcome'].index_select(0, rows).to(
                        device, non_blocking=True
                    )
                    t_merge = columns['t_merge'].index_select(0, rows).to(
                        device, non_blocking=True
                    )
                    lifecycle_weight = columns['fruit_weight'].index_select(
                        0, rows
                    ).to(device, non_blocking=True)
                    targets, resolved = merge_distance_targets(
                        outcomes, t_merge, horizons
                    )
                    valid = state.active & resolved
                    balance = torch.zeros_like(
                        lifecycle_weight, dtype=torch.float32
                    )
                    if bool(valid.any().item()):
                        levels = state.levels.to(torch.long).clamp(0, 11)
                        balance[valid] = class_balance[
                            levels[valid], targets[valid]
                        ]
                    sample_weights = lifecycle_weight * balance
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(args.autocast_bfloat16, device):
                        output = train_model(state)
                        loss = merge_distance_loss(
                            output.logits,
                            state.active,
                            outcomes,
                            t_merge,
                            horizons,
                            sample_weights=sample_weights,
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm
                    )
                    optimizer.step()
                    loss_sum += float(loss.detach().item())
                    batches += 1
                    fruit_samples += int(valid.sum().item())
                    now = time.perf_counter()
                    if (
                            now - last_telemetry
                            >= float(args.telemetry_interval_seconds)):
                        _atomic_json(output_dir / 'manifest.json', {
                            **run_manifest,
                            'phase': 'training',
                            'updated_at_utc': _utc_now(),
                            'elapsed_seconds': now - started,
                            'current_epoch': epoch,
                            'completed_epochs': epoch - 1,
                            'progress_fraction': (
                                (epoch - 1) / int(args.epochs)
                            ),
                            'epoch_batch': batches,
                            'train_loss': loss_sum / max(1, batches),
                            'train_resolved_fruit_samples': fruit_samples,
                        })
                        last_telemetry = now
                    if (
                            args.max_train_batches > 0
                            and batches >= int(args.max_train_batches)):
                        break
                if (
                        args.max_train_batches > 0
                        and batches >= int(args.max_train_batches)):
                    break
            _atomic_json(output_dir / 'manifest.json', {
                **run_manifest,
                'phase': 'validation',
                'updated_at_utc': _utc_now(),
                'elapsed_seconds': time.perf_counter() - started,
                'current_epoch': epoch,
                'completed_epochs': epoch - 1,
                'progress_fraction': (epoch - 1) / int(args.epochs),
                'epoch_batch': batches,
                'train_loss': loss_sum / max(1, batches),
                'train_resolved_fruit_samples': fruit_samples,
            })
            validation = _evaluate_model(
                model,
                paths,
                manifest,
                split='validation',
                device=device,
                batch_size=args.batch_size,
                autocast_bfloat16=args.autocast_bfloat16,
                max_batches=args.max_eval_batches,
            )
            epoch_record = {
                'epoch': epoch,
                'train_loss': loss_sum / max(1, batches),
                'train_batches': batches,
                'train_resolved_fruit_samples': fruit_samples,
                'epoch_seconds': time.perf_counter() - epoch_started,
                'validation': validation,
            }
            with history_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(epoch_record, ensure_ascii=False) + '\n')
            _atomic_torch_save(
                output_dir / 'checkpoints' / 'latest.pt',
                _checkpoint_payload(
                    model, manifest, args, epoch, validation
                ),
            )
            validation_nll = float(validation['nll'])
            if validation_nll < best_nll:
                best_nll = validation_nll
                best_epoch = epoch
                stale_epochs = 0
                _atomic_torch_save(
                    output_dir / 'checkpoints' / 'best.pt',
                    _checkpoint_payload(
                        model, manifest, args, epoch, validation
                    ),
                )
            else:
                stale_epochs += 1
            last_completed_epoch = epoch
            _atomic_json(output_dir / 'manifest.json', {
                **run_manifest,
                'phase': 'training',
                'updated_at_utc': _utc_now(),
                'elapsed_seconds': time.perf_counter() - started,
                'current_epoch': epoch,
                'completed_epochs': epoch,
                'progress_fraction': epoch / int(args.epochs),
                'epoch_batch': batches,
                'train_loss': epoch_record['train_loss'],
                'train_resolved_fruit_samples': fruit_samples,
                'latest_validation': validation,
                'best_epoch': best_epoch,
                'best_validation_nll': best_nll,
            })
            print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
            if stale_epochs >= int(args.early_stopping_patience):
                break
    except Exception as error:
        _atomic_json(output_dir / 'manifest.json', {
            **run_manifest,
            'status': 'failed',
            'phase': 'failed',
            'updated_at_utc': _utc_now(),
            'failure': f'{type(error).__name__}: {error}',
        })
        raise

    best_path = output_dir / 'checkpoints' / 'best.pt'
    best = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(best['model_state'], strict=True)
    _atomic_json(output_dir / 'manifest.json', {
        **run_manifest,
        'phase': 'evaluation',
        'updated_at_utc': _utc_now(),
        'elapsed_seconds': time.perf_counter() - started,
        'current_epoch': last_completed_epoch,
        'completed_epochs': last_completed_epoch,
        'progress_fraction': 1.0,
        'best_epoch': best_epoch,
        'best_validation_nll': best_nll,
    })
    test_metrics = _evaluate_model(
        model,
        paths,
        manifest,
        split='test',
        device=device,
        batch_size=args.batch_size,
        autocast_bfloat16=args.autocast_bfloat16,
        max_batches=args.max_eval_batches,
    )
    final_path = output_dir / 'checkpoints' / 'final.pt'
    _atomic_torch_save(
        final_path,
        _checkpoint_payload(
            model,
            manifest,
            args,
            best_epoch,
            {'validation': best['metrics'], 'test': test_metrics},
        ),
    )
    final_manifest = {
        **run_manifest,
        'status': 'complete',
        'phase': 'completed',
        'updated_at_utc': _utc_now(),
        'elapsed_seconds': time.perf_counter() - started,
        'best_epoch': best_epoch,
        'best_validation_nll': best_nll,
        'current_epoch': last_completed_epoch,
        'completed_epochs': last_completed_epoch,
        'progress_fraction': 1.0,
        'test_metrics': test_metrics,
        'checkpoint': str(final_path.resolve()),
        'checkpoint_sha256': _sha256(final_path),
    }
    _atomic_json(output_dir / 'manifest.json', final_manifest)
    print(json.dumps(final_manifest, ensure_ascii=False, indent=2))
    return final_manifest


def load_predictor_checkpoint(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('purpose') != 'merge_distance_predictor':
        raise ValueError('checkpoint is not a merge distance predictor')
    config = MergeDistanceConfig.from_dict(payload['model_config'])
    model = MergeDistancePredictor(
        config, **payload['geometry_config']
    )
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device).eval(), payload


def evaluate(args):
    manifest = _load_dataset_manifest(args.dataset_dir)
    device = torch.device(args.device)
    model, checkpoint = load_predictor_checkpoint(args.checkpoint, device)
    if tuple(manifest['horizons']) != tuple(model.config.horizons):
        raise ValueError('checkpoint horizons do not match dataset')
    metrics = _evaluate_model(
        model,
        labeled_scene_paths(args.dataset_dir),
        manifest,
        split=args.split,
        device=device,
        batch_size=args.batch_size,
        autocast_bfloat16=args.autocast_bfloat16,
        max_batches=args.max_eval_batches,
    )
    result = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': _sha256(args.checkpoint),
        'checkpoint_metrics': checkpoint.get('metrics'),
        'dataset_dir': str(Path(args.dataset_dir).resolve()),
        'split': args.split,
        'metrics': metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'collect':
        collect(args)
        return
    if args.command == 'label':
        result = label_scene_dataset(
            args.dataset_dir,
            output_dir=args.output_dir,
            horizons=args.horizons,
            shard_rows=args.shard_rows,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == 'train':
        train(args)
        return
    evaluate(args)


if __name__ == '__main__':
    main()
