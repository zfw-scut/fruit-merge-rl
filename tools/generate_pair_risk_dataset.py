"""用SAB-128真实后续轨迹生成单帧堵塞风险监督数据。"""

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
    ShardedTensorWriter,
)
from daxigua.rl.observations import TensorState  # noqa: E402
from daxigua.rl.pair_failure import (  # noqa: E402
    PairFailureConfig,
    PairFailureTracker,
)
from daxigua.rl.pair_risk import (  # noqa: E402
    END_COLLECTOR_STOP,
    END_DROP_LIMIT,
    END_NATURAL,
    END_SIMULATOR_TRUNCATED,
    EPISODE_DTYPES,
    EVENT_DTYPES,
    EXPOSURE_DTYPES,
    extract_episode_rows,
    extract_pair_events,
    extract_pair_exposures,
    finalize_pair_risk_dataset,
)
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='生成并标注同级高等级水果对单帧堵塞风险数据。'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    collect = subparsers.add_parser(
        'collect', help='运行SAB-128对局并自动生成训练标签。'
    )
    collect.add_argument('--output-dir', type=Path, required=True)
    collect.add_argument(
        '--checkpoint', type=Path, default=DEFAULT_SAB128_CHECKPOINT
    )
    collect.add_argument(
        '--expected-checkpoint-sha256', default=SAB128_SHA256,
        help='传空字符串可显式关闭登记模型校验。',
    )
    collect.add_argument('--device', default='cuda')
    collect.add_argument('--parallel-envs', type=int, default=512)
    collect.add_argument('--episodes', type=int, default=200_000)
    collect.add_argument('--target-confirmed-events', type=int, default=1_000)
    collect.add_argument('--seed-base', type=int, default=91_000_000)
    collect.add_argument(
        '--max-drops', type=int, default=0,
        help='0表示不设单局投放上限；正数仅用于诊断。',
    )
    collect.add_argument(
        '--max-wall-seconds', type=float, default=0.0,
        help='0表示不限墙钟；停止时未完成未来观察的样本会被截断。',
    )
    collect.add_argument('--exposure-stride', type=int, default=4)
    collect.add_argument('--forecast-horizon', type=int, default=24)
    collect.add_argument('--motion-window-drops', type=int, default=4)
    collect.add_argument('--confirmation-drops', type=int, default=24)
    collect.add_argument('--max-net-displacement-ratio', type=float, default=0.12)
    collect.add_argument('--adjacent-surface-gap-ratio', type=float, default=1.25)
    collect.add_argument('--transfer-interval', type=int, default=8)
    collect.add_argument('--shard-rows', type=int, default=32_768)
    collect.add_argument('--writer-queue-depth', type=int, default=2)
    collect.add_argument(
        '--background-writer',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    collect.add_argument(
        '--compile-model',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    collect.add_argument(
        '--compile-mode',
        choices=('default', 'reduce-overhead', 'max-autotune'),
        default='reduce-overhead',
    )
    collect.add_argument(
        '--autocast-bfloat16',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='默认关闭以保持SAB-128动作与FP32评估一致。',
    )
    collect.add_argument('--warmup-steps', type=int, default=2)
    collect.add_argument('--progress-interval-seconds', type=float, default=10.0)
    collect.add_argument(
        '--auto-finalize',
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    finalize = subparsers.add_parser(
        'finalize', help='仅对已经采集完成的原始表重新生成标签。'
    )
    finalize.add_argument('dataset_dir', type=Path)
    finalize.add_argument('--forecast-horizon', type=int, default=24)
    finalize.add_argument('--confirmation-drops', type=int, default=24)
    finalize.add_argument('--shard-rows', type=int, default=65_536)
    return parser.parse_args(argv)


def _validate_collect_args(args):
    for name in (
            'parallel_envs', 'episodes', 'target_confirmed_events',
            'exposure_stride', 'forecast_horizon', 'motion_window_drops',
            'confirmation_drops', 'transfer_interval', 'shard_rows',
            'writer_queue_depth'):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f'{name} must be positive')
    if args.parallel_envs > args.episodes:
        raise ValueError('parallel-envs cannot exceed episodes')
    if args.confirmation_drops <= args.motion_window_drops:
        raise ValueError('confirmation window must exceed motion window')
    if args.max_drops < 0 or args.max_wall_seconds < 0.0:
        raise ValueError('collection limits cannot be negative')
    if args.warmup_steps < 0 or args.progress_interval_seconds <= 0.0:
        raise ValueError('warmup/progress values are invalid')


def _prepare_output(output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'raw').mkdir()
    return output_dir


def _episode_seeds(episodes, seed_base):
    return (
        torch.arange(int(episodes), dtype=torch.int64)
        .mul_(SEED_STRIDE)
        .add_(int(seed_base))
    )


def _autocast_context(enabled, device):
    if not enabled:
        return nullcontext()
    if torch.device(device).type != 'cuda':
        raise RuntimeError('bfloat16 collection requires CUDA')
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)


def _policy_actions(model, observation, *, physics_fps, bfloat16):
    state = TensorState.from_observation(
        observation, physics_fps=physics_fps
    )
    with _autocast_context(bfloat16, observation.positions.device):
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
        raise ValueError('checkpoint does not match the registered SAB-128')
    config = viewer_simulator_config(30, loaded.model_config, loaded.device)
    if config.drop_fast_forward:
        raise RuntimeError('pair-risk data requires no-fast-forward physics')
    output_dir = _prepare_output(args.output_dir)
    manifest_path = output_dir / 'manifest.json'
    batch_size = int(args.parallel_envs)
    all_seeds = _episode_seeds(args.episodes, args.seed_base)
    episode_ids = torch.arange(
        batch_size, dtype=torch.int64, device=loaded.device
    )
    slot_seeds = all_seeds[:batch_size].to(loaded.device)
    simulator = TensorVectorSimulator(
        batch_size, config=config, device=loaded.device
    )
    simulator.reset(seeds=slot_seeds)
    policy_model = loaded.model
    if args.compile_model:
        if not hasattr(torch, 'compile'):
            raise RuntimeError('torch.compile is unavailable')
        policy_model = torch.compile(
            policy_model, mode=args.compile_mode, dynamic=False
        )
    for _ in range(int(args.warmup_steps)):
        actions = _policy_actions(
            policy_model,
            simulator.observe(),
            physics_fps=config.physics_fps,
            bfloat16=args.autocast_bfloat16,
        )
        simulator.step(actions)
    simulator.reset(seeds=slot_seeds)

    detector_config = PairFailureConfig(
        motion_window_drops=args.motion_window_drops,
        confirmation_drops=args.confirmation_drops,
        max_net_displacement_ratio=args.max_net_displacement_ratio,
        adjacent_surface_gap_ratio=args.adjacent_surface_gap_ratio,
    )
    tracker = PairFailureTracker(
        batch_size,
        config.max_fruits,
        device=loaded.device,
        config=detector_config,
    )
    raw_dir = output_dir / 'raw'
    writers = {
        'pair_risk_exposures': ShardedTensorWriter(
            raw_dir, 'pair_risk_exposures', EXPOSURE_DTYPES,
            shard_rows=args.shard_rows,
            background=args.background_writer,
            queue_depth=args.writer_queue_depth,
        ),
        'pair_risk_events': ShardedTensorWriter(
            raw_dir, 'pair_risk_events', EVENT_DTYPES,
            shard_rows=max(1024, min(args.shard_rows, 16_384)),
            background=args.background_writer,
            queue_depth=args.writer_queue_depth,
        ),
        'pair_risk_episodes': ShardedTensorWriter(
            raw_dir, 'pair_risk_episodes', EPISODE_DTYPES,
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
        'purpose': 'pair_failure_single_frame_risk_collection',
        'status': 'running',
        'created_at_utc': _utc_now(),
        'git_revision': _git_revision(),
        'physics_identity': PHYSICS_IDENTITY,
        'simulator_config': asdict(config),
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'checkpoint_progress': loaded.progress,
        'policy': 'greedy',
        'device': str(loaded.device),
        'parameters': {
            name: getattr(args, name)
            for name in (
                'parallel_envs', 'episodes', 'target_confirmed_events',
                'seed_base', 'max_drops', 'max_wall_seconds',
                'exposure_stride', 'forecast_horizon',
                'motion_window_drops', 'confirmation_drops',
                'max_net_displacement_ratio',
                'adjacent_surface_gap_ratio', 'transfer_interval',
                'shard_rows', 'compile_model', 'compile_mode',
                'autocast_bfloat16',
            )
        },
        'table_schemas': {
            'pair_risk_exposures': list(EXPOSURE_DTYPES),
            'pair_risk_events': list(EVENT_DTYPES),
            'pair_risk_episodes': list(EPISODE_DTYPES),
        },
    }
    _write_manifest(manifest_path, base_manifest)

    completed = 0
    next_episode_id = batch_size
    transitions = 0
    decision_steps = 0
    confirmed_counter = torch.zeros(
        (), dtype=torch.int64, device=loaded.device
    )
    confirmed_total = 0
    status = 'complete'
    failure = None
    started = time.perf_counter()
    last_progress = started

    def append_active_episodes(end_kind):
        observation = simulator.observe()
        rows = torch.nonzero(episode_ids >= 0, as_tuple=False).flatten()
        if rows.numel() == 0:
            return
        kinds = torch.full(
            rows.shape, int(end_kind), dtype=torch.int8, device=loaded.device
        )
        accumulators['pair_risk_episodes'].append(extract_episode_rows(
            observation, rows, episode_ids, slot_seeds, kinds
        ))

    try:
        while completed < int(args.episodes):
            enabled = episode_ids >= 0
            observation = simulator.observe()
            actions = _policy_actions(
                policy_model,
                observation,
                physics_fps=config.physics_fps,
                bfloat16=args.autocast_bfloat16,
            )
            result = (
                simulator.step_masked(actions, enabled)
                if loaded.device.type == 'cuda'
                else simulator.step(actions)
            )
            transitions += int(enabled.sum().item())
            decision_steps += 1
            after = result.observation
            update = tracker.update_observation(after)
            confirmed_counter.add_(
                (update.confirmed & enabled[:, None]).sum()
            )
            accumulators['pair_risk_exposures'].append(
                extract_pair_exposures(
                    after,
                    episode_ids,
                    tracker.pair_i,
                    tracker.pair_j,
                    exposure_stride=args.exposure_stride,
                )
            )
            accumulators['pair_risk_events'].append(
                extract_pair_events(update, episode_ids, after.step_count)
            )

            capped = torch.zeros_like(enabled)
            if args.max_drops > 0:
                capped = after.step_count >= int(args.max_drops)
            finished = enabled & (
                result.physics.done | result.physics.truncated | capped
            )
            finished_rows = torch.nonzero(finished, as_tuple=False).flatten()
            if finished_rows.numel() > 0:
                done = result.physics.done[finished_rows]
                truncated = result.physics.truncated[finished_rows]
                end_kinds = torch.full(
                    finished_rows.shape,
                    END_DROP_LIMIT,
                    dtype=torch.int8,
                    device=loaded.device,
                )
                end_kinds[truncated] = END_SIMULATOR_TRUNCATED
                end_kinds[done] = END_NATURAL
                accumulators['pair_risk_episodes'].append(
                    extract_episode_rows(
                        after,
                        finished_rows,
                        episode_ids,
                        slot_seeds,
                        end_kinds,
                    )
                )
                finished_count = int(finished_rows.numel())
                completed += finished_count
                assign_count = min(
                    finished_count,
                    max(0, int(args.episodes) - next_episode_id),
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
                tracker.reset(finished)
                episode_ids[finished_rows] = replacements
                slot_seeds[finished_rows] = reset_seeds

            for accumulator in accumulators.values():
                accumulator.advance()
            now = time.perf_counter()
            if now - last_progress >= args.progress_interval_seconds:
                confirmed_total = int(confirmed_counter.item())
                elapsed = now - started
                _write_manifest(
                    manifest_path,
                    base_manifest,
                    status='running',
                    updated_at_utc=_utc_now(),
                    completed_episodes=completed,
                    transitions=transitions,
                    decision_steps=decision_steps,
                    confirmed_events=confirmed_total,
                    elapsed_seconds=elapsed,
                    env_steps_per_second=transitions / max(elapsed, 1e-9),
                    table_rows={
                        name: writer.total_rows
                        for name, writer in writers.items()
                    },
                )
                print(json.dumps({
                    'completed_episodes': completed,
                    'transitions': transitions,
                    'confirmed_events': confirmed_total,
                    'target_confirmed_events': args.target_confirmed_events,
                    'env_steps_per_second': transitions / max(elapsed, 1e-9),
                    'exposure_rows': writers[
                        'pair_risk_exposures'
                    ].total_rows,
                }, ensure_ascii=False), flush=True)
                last_progress = now
                if confirmed_total >= int(args.target_confirmed_events):
                    status = 'target_reached'
                    append_active_episodes(END_COLLECTOR_STOP)
                    break
            if (
                    args.max_wall_seconds > 0.0
                    and now - started >= args.max_wall_seconds):
                status = 'wall_time_reached'
                append_active_episodes(END_COLLECTOR_STOP)
                break
    except KeyboardInterrupt:
        status = 'interrupted'
        append_active_episodes(END_COLLECTOR_STOP)
    except Exception as error:
        status = 'failed'
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        close_error = None
        for accumulator in accumulators.values():
            try:
                accumulator.close()
            except Exception as error:  # pragma: no cover
                close_error = close_error or error
        confirmed_total = int(confirmed_counter.item())
        elapsed = time.perf_counter() - started
        _write_manifest(
            manifest_path,
            base_manifest,
            status=status if close_error is None else 'failed',
            updated_at_utc=_utc_now(),
            completed_episodes=completed,
            transitions=transitions,
            decision_steps=decision_steps,
            confirmed_events=confirmed_total,
            elapsed_seconds=elapsed,
            env_steps_per_second=transitions / max(elapsed, 1e-9),
            table_rows={name: writer.total_rows for name, writer in writers.items()},
            table_shards={name: writer.shard_count for name, writer in writers.items()},
            peak_cuda_allocated_bytes=(
                int(torch.cuda.max_memory_allocated(loaded.device))
                if loaded.device.type == 'cuda' else None
            ),
            failure=failure or (str(close_error) if close_error else None),
        )
        if close_error is not None:
            raise close_error

    result = json.loads(manifest_path.read_text(encoding='utf-8'))
    if args.auto_finalize and result['status'] != 'failed':
        result['labeled'] = finalize_pair_risk_dataset(
            output_dir,
            forecast_horizon=args.forecast_horizon,
            confirmation_drops=args.confirmation_drops,
            shard_rows=max(args.shard_rows, 65_536),
        )
        _atomic_json(manifest_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'collect':
        collect(args)
        return
    result = finalize_pair_risk_dataset(
        args.dataset_dir,
        forecast_horizon=args.forecast_horizon,
        confirmation_drops=args.confirmation_drops,
        shard_rows=args.shard_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
