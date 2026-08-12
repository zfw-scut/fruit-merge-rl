"""让一批 CUDA 环境随机投放，直到失败、技术截断或投放上限。"""

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

from daxigua.simulator import (
    BatchSimulationTrace,
    PHYSICS_IDENTITY,
    save_trace_archive,
    SimulatorConfig,
    TensorVectorSimulator,
    write_replay_catalog,
    write_replay_html,
)
from daxigua.simulator.cuda_backend import load_cuda_extension


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--max-drops', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260804)
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
    parser.add_argument('--progress-every', type=int, default=50)
    parser.add_argument('--replay-samples', type=int, default=0)
    parser.add_argument(
        '--replay-selection',
        choices=('uniform', 'most-timeouts', 'timeout-rate-stratified'),
        default='uniform',
    )
    parser.add_argument('--replay-frame-stride', type=int, default=2)
    parser.add_argument('--replay-tail-drops', type=int, default=5)
    parser.add_argument(
        '--replay-full-episodes', action='store_true'
    )
    parser.add_argument(
        '--replay-output-dir',
        type=Path,
        default=(
            PROJECT_ROOT / 'recordings' / 'random-4096-final-replays'
        ),
    )
    parser.add_argument(
        '--replay-from-plan',
        type=Path,
        help='跳过正式运行，按先前保存的 replay-plan.json 复跑完整局',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'recordings' / 'random-4096-until-stop.json',
    )
    return parser.parse_args()


def select_replay_environments(
        selection, sample_count, seed, settle_timeout_counts, step_counts):
    """按指定策略选择正式运行中的环境编号。"""

    timeout_values = settle_timeout_counts.detach().cpu().tolist()
    step_values = step_counts.detach().cpu().tolist()
    environment_count = len(timeout_values)
    if sample_count < 0 or sample_count > environment_count:
        raise ValueError('sample_count is outside the environment range')
    if sample_count == 0:
        return []
    if selection == 'uniform':
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed ^ 0x5EED5EED)
        return torch.randperm(
            environment_count, generator=generator
        )[:sample_count].tolist()

    timeout_rates = [
        count / max(1, steps)
        for count, steps in zip(timeout_values, step_values)
    ]
    if selection == 'most-timeouts':
        return sorted(
            range(environment_count),
            key=lambda index: (
                timeout_values[index],
                timeout_rates[index],
                step_values[index],
                -index,
            ),
            reverse=True,
        )[:sample_count]
    if selection == 'timeout-rate-stratified':
        ordered = sorted(
            range(environment_count),
            key=lambda index: (
                timeout_rates[index], timeout_values[index], index
            ),
        )
        selected = []
        for sample_index in range(sample_count):
            # 取每个等宽分层的上边界；既避开完全无超时样本，也覆盖到最大值。
            rank = (
                (sample_index + 1) * environment_count + sample_count - 1
            ) // sample_count - 1
            selected.append(ordered[min(rank, environment_count - 1)])
        return list(reversed(selected))
    raise ValueError(f'unsupported replay selection: {selection}')


def select_trace_row(
        trace, row, original_env_index, *, record_count=None):
    """提取单行追踪，并把复跑局部索引恢复成正式运行环境索引。"""

    row_count = int(trace.env_indices.numel())
    trace_capacity = int(trace.frame_numbers.shape[1])
    if record_count is None:
        record_count = int(trace.record_counts[row].item())
    values = {}
    for field_name in trace.__dataclass_fields__:
        value = getattr(trace, field_name)
        if (
                isinstance(value, torch.Tensor)
                and value.ndim
                and value.shape[0] == row_count):
            selected = value[row:row + 1]
            if selected.ndim >= 2 and selected.shape[1] == trace_capacity:
                selected = selected[:, :record_count]
            values[field_name] = selected.detach().cpu().clone()
        else:
            values[field_name] = value
    values['env_indices'] = torch.tensor(
        [original_env_index], dtype=torch.int64
    )
    return BatchSimulationTrace(**values)


def trim_trace_frames(trace, record_capacity):
    """在复制到 CPU 前移除逐帧缓冲中未使用的尾部容量。"""

    source_capacity = int(trace.frame_numbers.shape[1])
    values = {}
    for field_name in trace.__dataclass_fields__:
        value = getattr(trace, field_name)
        if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and value.shape[1] == source_capacity):
            values[field_name] = value[:, :record_capacity]
        else:
            values[field_name] = value
    return BatchSimulationTrace(**values).cpu()


def combine_trace_rows(traces):
    """把不同终局步的一行追踪组合成回放页的并列样本。"""

    if not traces:
        raise ValueError('traces must not be empty')
    values = {}
    for field_name in traces[0].__dataclass_fields__:
        first = getattr(traces[0], field_name)
        if isinstance(first, torch.Tensor) and first.ndim:
            values[field_name] = torch.cat(
                [getattr(trace, field_name) for trace in traces], dim=0
            )
        else:
            values[field_name] = first
    return BatchSimulationTrace(**values)


def replay_final_steps(args, config, sampled_envs, terminal_metadata):
    """以正式运行的 RNG/动作流确定性复现抽样环境的最后一次投放。"""

    sample_count = len(sampled_envs)
    device = torch.device('cuda')
    local_simulator = TensorVectorSimulator(
        sample_count, config=config, device=device
    )
    original_env_tensor = torch.tensor(
        sampled_envs, dtype=torch.int64, device=device
    )
    seed_values = (
        args.seed
        + original_env_tensor * local_simulator._SEED_STRIDE
    ) & local_simulator._RNG_MASK
    local_simulator.reset(seeds=seed_values)

    target_steps = torch.tensor(
        [terminal_metadata[index]['step_count'] for index in sampled_envs],
        dtype=torch.int64,
        device=device,
    )
    replay_running = torch.ones(
        sample_count, dtype=torch.bool, device=device
    )
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(args.seed)
    tail_drops = min(
        args.replay_tail_drops, int(target_steps.min().item())
    )
    traces_by_original_env = {
        original_index: [] for original_index in sampled_envs
    }

    for drop_index in range(int(target_steps.max().item())):
        original_actions = torch.randint(
            config.action_count,
            (args.num_envs,),
            dtype=torch.int64,
            device=device,
            generator=action_generator,
        )
        local_actions = original_actions[original_env_tensor]
        current_drop = drop_index + 1
        capture_indices = torch.nonzero(
            replay_running
            & (current_drop > target_steps - tail_drops)
            & (current_drop <= target_steps),
            as_tuple=False,
        ).flatten()
        if capture_indices.numel():
            result, trace = local_simulator.step_masked_with_trace(
                local_actions,
                replay_running,
                capture_indices,
                frame_stride=args.replay_frame_stride,
            )
            record_counts = trace.record_counts.cpu().tolist()
            trace = trim_trace_frames(trace, max(record_counts))
            for trace_row, local_index in enumerate(
                    capture_indices.cpu().tolist()):
                original_index = sampled_envs[local_index]
                traces_by_original_env[original_index].append(
                    select_trace_row(
                        trace,
                        trace_row,
                        original_index,
                        record_count=record_counts[trace_row],
                    )
                )
        else:
            result = local_simulator.step_masked(
                local_actions, replay_running
            )

        reached_target = replay_running & (
            local_simulator.step_count >= target_steps
        )
        replay_running &= ~reached_target

    replay_step_counts = local_simulator.step_count.cpu().tolist()
    replay_scores = local_simulator.score.cpu().tolist()
    replay_done = local_simulator.terminated.cpu().tolist()
    replay_needs_reset = local_simulator.needs_reset.cpu().tolist()
    for local_index, original_index in enumerate(sampled_envs):
        expected = terminal_metadata[original_index]
        actual_kind = (
            'terminated'
            if replay_done[local_index]
            else (
                'truncated'
                if replay_needs_reset[local_index]
                else 'capped'
            )
        )
        if replay_step_counts[local_index] != expected['step_count']:
            raise RuntimeError(
                f'replay step mismatch for environment {original_index}'
            )
        if replay_scores[local_index] != expected['score']:
            raise RuntimeError(
                f'replay score mismatch for environment {original_index}'
            )
        if actual_kind != expected['end_kind']:
            raise RuntimeError(
                f'replay boundary mismatch for environment {original_index}'
            )
        if len(traces_by_original_env[original_index]) != tail_drops:
            raise RuntimeError(
                f'missing tail traces for environment {original_index}'
            )

    args.replay_output_dir.mkdir(parents=True, exist_ok=True)
    ordered_trace_sequences = [
        traces_by_original_env[index] for index in sampled_envs
    ]
    combined_trace_sequence = tuple(
        combine_trace_rows([
            traces_by_original_env[index][tail_index]
            for index in sampled_envs
        ])
        for tail_index in range(tail_drops)
    )
    combined_name = f'final-steps-{sample_count}'
    combined_path = write_replay_html(
        args.replay_output_dir / f'{combined_name}.html',
        combined_trace_sequence,
        config,
        title=(
            f'{args.num_envs} 环境随机长局：{sample_count} 个终局前 '
            f'{tail_drops} 次投放回放'
        ),
    )
    combined_trace_path = save_trace_archive(
        args.replay_output_dir / f'{combined_name}.pt.gz',
        combined_trace_sequence,
    )

    replay_entries = []
    for original_index, traces in zip(
            sampled_envs, ordered_trace_sequences):
        metadata = terminal_metadata[original_index]
        html_path = write_replay_html(
            args.replay_output_dir / f'env-{original_index}-final-step.html',
            tuple(traces),
            config,
            title=(
                f'环境 {original_index} 失败前 {tail_drops} 次投放'
            ),
        )
        replay_frames = sum(
            int(trace.frame_numbers[
                0, int(trace.record_counts[0]) - 1
            ])
            for trace in traces
        )
        replay_entries.append({
            'env_index': original_index,
            **metadata,
            'drops_in_replay': tail_drops,
            'frames_in_replay': replay_frames,
            'replay': str(html_path.resolve()),
        })

    manifest = {
        'seed': args.seed,
        'source_num_envs': args.num_envs,
        'sample_count': sample_count,
        'frame_stride': args.replay_frame_stride,
        'selection': args.replay_selection,
        'scope': (
            f'final {tail_drops} decision intervals before termination '
            'or drop cap'
        ),
        'combined_replay': str(combined_path.resolve()),
        'combined_trace': str(combined_trace_path.resolve()),
        'replays': replay_entries,
    }
    manifest_path = args.replay_output_dir / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return manifest, manifest_path


def write_full_replay_index(path, entries, source_num_envs):
    """生成按需加载的单页目录，避免打开大量标签页。"""

    return write_replay_catalog(
        path,
        entries,
        title=f'{source_num_envs} 环境随机测试：{len(entries)} 条完整局回放',
        description=(
            '每条均从第 1 次投放记录到失败或投放上限。'
            '目录显示等待超时次数和比例；播放器可用 T / Shift+T '
            '跳到下/上一次等待超时。选择左侧条目后只加载当前一局。'
        ),
    )


def replay_full_episodes(args, config, sampled_envs, terminal_metadata):
    """确定性复跑并分别保存抽样环境从首次投放到结束的完整一局。"""

    sample_count = len(sampled_envs)
    device = torch.device('cuda')
    local_simulator = TensorVectorSimulator(
        sample_count, config=config, device=device
    )
    original_env_tensor = torch.tensor(
        sampled_envs, dtype=torch.int64, device=device
    )
    seed_values = (
        args.seed
        + original_env_tensor * local_simulator._SEED_STRIDE
    ) & local_simulator._RNG_MASK
    local_simulator.reset(seeds=seed_values)
    target_steps = torch.tensor(
        [terminal_metadata[index]['step_count'] for index in sampled_envs],
        dtype=torch.int64,
        device=device,
    )
    replay_running = torch.ones(
        sample_count, dtype=torch.bool, device=device
    )
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(args.seed)
    trace_sequences = {index: [] for index in sampled_envs}
    entries_by_env = {}
    args.replay_output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for drop_index in range(int(target_steps.max().item())):
        original_actions = torch.randint(
            config.action_count,
            (args.num_envs,),
            dtype=torch.int64,
            device=device,
            generator=action_generator,
        )
        local_actions = original_actions[original_env_tensor]
        capture_indices = torch.nonzero(
            replay_running, as_tuple=False
        ).flatten()
        result, trace = local_simulator.step_masked_with_trace(
            local_actions,
            replay_running,
            capture_indices,
            frame_stride=args.replay_frame_stride,
        )
        record_counts = trace.record_counts.cpu().tolist()
        trace = trim_trace_frames(trace, max(record_counts))
        capture_list = capture_indices.cpu().tolist()
        for trace_row, local_index in enumerate(capture_list):
            original_index = sampled_envs[local_index]
            trace_sequences[original_index].append(
                select_trace_row(
                    trace,
                    trace_row,
                    original_index,
                    record_count=record_counts[trace_row],
                )
            )

        reached_target = replay_running & (
            local_simulator.step_count >= target_steps
        )
        reached_list = torch.nonzero(
            reached_target, as_tuple=False
        ).flatten().cpu().tolist()
        for local_index in reached_list:
            original_index = sampled_envs[local_index]
            expected = terminal_metadata[original_index]
            actual_score = int(local_simulator.score[local_index].item())
            actual_done = bool(
                local_simulator.terminated[local_index].item()
            )
            actual_needs_reset = bool(
                local_simulator.needs_reset[local_index].item()
            )
            actual_kind = (
                'terminated'
                if actual_done
                else ('truncated' if actual_needs_reset else 'capped')
            )
            if actual_score != expected['score']:
                raise RuntimeError(
                    f'replay score mismatch for environment {original_index}'
                )
            if actual_kind != expected['end_kind']:
                raise RuntimeError(
                    f'replay boundary mismatch for environment {original_index}'
                )

            sequence = tuple(trace_sequences[original_index])
            if len(sequence) != expected['step_count']:
                raise RuntimeError(
                    f'replay length mismatch for environment {original_index}'
                )
            actual_timeout_count = sum(
                bool(step_trace.settle_timeout[0].item())
                for step_trace in sequence
            )
            if actual_timeout_count != expected['settle_timeout_count']:
                raise RuntimeError(
                    f'replay timeout mismatch for environment {original_index}'
                )
            html_path = args.replay_output_dir / (
                f'env-{original_index}-full-episode.html'
            )
            trace_path = args.replay_output_dir / (
                f'env-{original_index}-full-episode.pt.gz'
            )
            write_replay_html(
                html_path,
                sequence,
                config,
                title=(
                    f'环境 {original_index} 完整一局：'
                    f'{expected["step_count"]} 次投放'
                ),
            )
            save_trace_archive(trace_path, sequence)
            physics_frames = sum(
                int(step_trace.frame_numbers[
                    0, int(step_trace.record_counts[0]) - 1
                ])
                for step_trace in sequence
            )
            entries_by_env[original_index] = {
                'env_index': original_index,
                **expected,
                'drops_in_replay': len(sequence),
                'physics_frames_in_replay': physics_frames,
                'replay': str(html_path.resolve()),
                'trace': str(trace_path.resolve()),
            }
            trace_sequences[original_index] = []

        replay_running &= ~reached_target
        if (drop_index + 1) % 25 == 0:
            print(json.dumps({
                'replay_drop_iteration': drop_index + 1,
                'replay_running': int(replay_running.sum().item()),
                'replays_written': len(entries_by_env),
                'elapsed_seconds': time.perf_counter() - started,
            }, ensure_ascii=False), flush=True)

    entries = [entries_by_env[index] for index in sampled_envs]
    index_path = write_full_replay_index(
        args.replay_output_dir / 'full-episodes-index.html',
        entries,
        args.num_envs,
    )
    manifest = {
        'seed': args.seed,
        'source_num_envs': args.num_envs,
        'sample_count': sample_count,
        'frame_stride': args.replay_frame_stride,
        'selection': args.replay_selection,
        'scope': 'complete episode from first drop through termination or cap',
        'elapsed_seconds': time.perf_counter() - started,
        'index': str(index_path.resolve()),
        'replays': entries,
    }
    manifest_path = args.replay_output_dir / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return manifest, manifest_path


def tensor_summary(values):
    if values.numel() == 0:
        return None
    values = values.to(torch.float32)
    quantiles = torch.quantile(
        values,
        torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
            device=values.device,
        ),
    ).cpu().tolist()
    return {
        'count': int(values.numel()),
        'mean': float(values.mean().item()),
        'min': quantiles[0],
        'p25': quantiles[1],
        'median': quantiles[2],
        'p75': quantiles[3],
        'p90': quantiles[4],
        'p95': quantiles[5],
        'max': quantiles[6],
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    if args.num_envs <= 0 or args.max_drops <= 0:
        raise ValueError('num-envs and max-drops must be positive')
    if args.replay_samples < 0 or args.replay_samples > args.num_envs:
        raise ValueError('replay-samples must be in [0, num-envs]')
    if args.replay_frame_stride <= 0:
        raise ValueError('replay-frame-stride must be positive')
    if args.replay_tail_drops <= 0:
        raise ValueError('replay-tail-drops must be positive')

    build_started = time.perf_counter()
    load_cuda_extension()
    torch.cuda.synchronize()
    extension_load_seconds = time.perf_counter() - build_started

    if args.replay_from_plan is not None:
        plan = json.loads(args.replay_from_plan.read_text(encoding='utf-8'))
        plan_identity = plan.get('physics_identity')
        if plan_identity != PHYSICS_IDENTITY:
            rendered = plan_identity or 'legacy_unspecified'
            raise ValueError(
                'replay plan physics identity does not match current '
                f'simulator: {rendered!r} != {PHYSICS_IDENTITY!r}'
            )
        args.seed = int(plan['seed'])
        args.num_envs = int(plan['source_num_envs'])
        args.replay_frame_stride = int(plan['frame_stride'])
        args.replay_selection = str(plan['selection'])
        args.replay_output_dir = Path(plan['replay_output_dir'])
        sampled_envs = [int(value) for value in plan['sampled_envs']]
        terminal_metadata = {
            int(index): metadata
            for index, metadata in plan['terminal_metadata'].items()
        }
        config = SimulatorConfig(**plan['config'])
        manifest, manifest_path = replay_full_episodes(
            args, config, sampled_envs, terminal_metadata
        )
        print(json.dumps({
            'replay_manifest': str(manifest_path.resolve()),
            'full_episode_index': manifest['index'],
            'sampled_replay_envs': sampled_envs,
        }, ensure_ascii=False, indent=2))
        return

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
        args.num_envs, config=config, device='cuda'
    )
    simulator.reset(seeds=args.seed)
    generator = torch.Generator(device=simulator.device)
    generator.manual_seed(args.seed)
    running = torch.ones(
        args.num_envs, dtype=torch.bool, device=simulator.device
    )
    terminated = torch.zeros_like(running)
    truncated = torch.zeros_like(running)
    environments_with_settle_timeout = torch.zeros_like(running)
    settle_timeout_counts = torch.zeros(
        args.num_envs, dtype=torch.int64, device=simulator.device
    )
    settle_timeout_intervals = torch.zeros(
        (), dtype=torch.int64, device=simulator.device
    )
    total_physics_frames = torch.zeros(
        (), dtype=torch.int64, device=simulator.device
    )
    total_fast_forwarded_frames = torch.zeros_like(total_physics_frames)
    total_collision_substeps = torch.zeros_like(total_physics_frames)
    total_merge_events = torch.zeros_like(total_physics_frames)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(simulator.device)
    started = time.perf_counter()
    iterations = 0
    for drop_index in range(args.max_drops):
        if drop_index and not bool(running.any().item()):
            break
        actions = torch.randint(
            config.action_count,
            (args.num_envs,),
            dtype=torch.int64,
            device=simulator.device,
            generator=generator,
        )
        result = simulator.step_masked(actions, running)
        total_physics_frames += result.physics.frames_simulated.sum()
        total_fast_forwarded_frames += (
            result.physics.fast_forwarded_frames.sum()
        )
        total_collision_substeps += result.physics.collision_substeps.sum()
        total_merge_events += result.physics.merge_events.count.sum()
        active_settle_timeout = running & result.physics.settle_timeout
        environments_with_settle_timeout |= active_settle_timeout
        settle_timeout_counts += active_settle_timeout.to(torch.int64)
        settle_timeout_intervals += active_settle_timeout.sum()
        newly_terminated = running & result.physics.done
        newly_truncated = running & result.physics.truncated
        terminated |= newly_terminated
        truncated |= newly_truncated
        running &= ~(newly_terminated | newly_truncated)
        iterations = drop_index + 1
        if (
                args.progress_every > 0
                and iterations % args.progress_every == 0):
            torch.cuda.synchronize()
            print(
                json.dumps({
                    'drop_iteration': iterations,
                    'running': int(running.sum().item()),
                    'terminated': int(terminated.sum().item()),
                    'truncated': int(truncated.sum().item()),
                    'settle_timeout_intervals': int(
                        settle_timeout_intervals.item()
                    ),
                    'elapsed_seconds': time.perf_counter() - started,
                }, ensure_ascii=False),
                flush=True,
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    transitions = int(simulator.step_count.sum().item())
    physics_frames = int(total_physics_frames.item())
    fast_forwarded_frames = int(total_fast_forwarded_frames.item())
    executed_physics_frames = physics_frames - fast_forwarded_frames
    collision_substeps = int(total_collision_substeps.item())
    merge_events = int(total_merge_events.item())
    capped = running
    step_counts = simulator.step_count
    scores = simulator.score
    terminated_cpu = terminated.cpu()
    truncated_cpu = truncated.cpu()
    step_counts_cpu = step_counts.cpu()
    scores_cpu = scores.cpu()
    settle_timeout_counts_cpu = settle_timeout_counts.cpu()
    settle_timeout_rates = (
        settle_timeout_counts.to(torch.float32)
        / step_counts.clamp_min(1).to(torch.float32)
    )
    report = {
        'physics_identity': PHYSICS_IDENTITY,
        'device': str(simulator.device),
        'num_envs': args.num_envs,
        'max_drops': args.max_drops,
        'batch_iterations': iterations,
        'extension_load_seconds': extension_load_seconds,
        'simulation_seconds': elapsed,
        'total_wall_seconds': extension_load_seconds + elapsed,
        'transitions': transitions,
        'physics_frames': physics_frames,
        'semantic_physics_frames': physics_frames,
        'executed_physics_frames': executed_physics_frames,
        'fast_forwarded_frames': fast_forwarded_frames,
        'collision_substeps': collision_substeps,
        'collision_substeps_per_executed_frame': (
            collision_substeps / executed_physics_frames
            if executed_physics_frames else 0.0
        ),
        'extra_collision_substep_ratio': (
            (collision_substeps - executed_physics_frames)
            / executed_physics_frames
            if executed_physics_frames else 0.0
        ),
        'merge_events': merge_events,
        'merge_events_per_transition': merge_events / transitions,
        'fast_forward_ratio': (
            fast_forwarded_frames / physics_frames if physics_frames else 0.0
        ),
        'env_steps_per_second': transitions / elapsed,
        'physics_frames_per_second': physics_frames / elapsed,
        'semantic_physics_frames_per_second': physics_frames / elapsed,
        'executed_physics_frames_per_second': (
            executed_physics_frames / elapsed
        ),
        'semantic_frames_per_transition': physics_frames / transitions,
        'executed_frames_per_transition': (
            executed_physics_frames / transitions
        ),
        'terminated_count': int(terminated.sum().item()),
        'truncated_count': int(truncated.sum().item()),
        'settle_timeout_interval_count': int(
            settle_timeout_intervals.item()
        ),
        'environments_with_settle_timeout_count': int(
            environments_with_settle_timeout.sum().item()
        ),
        'settle_timeout_counts': tensor_summary(settle_timeout_counts),
        'settle_timeout_rates': tensor_summary(settle_timeout_rates),
        'capped_count': int(capped.sum().item()),
        'all_step_counts': tensor_summary(step_counts),
        'terminated_step_counts': tensor_summary(step_counts[terminated]),
        'truncated_step_counts': tensor_summary(step_counts[truncated]),
        'capped_step_counts': tensor_summary(step_counts[capped]),
        'all_scores': tensor_summary(scores),
        'terminated_scores': tensor_summary(scores[terminated]),
        'capped_scores': tensor_summary(scores[capped]),
        'peak_cuda_memory_mib': (
            torch.cuda.max_memory_allocated(simulator.device) / 1024 ** 2
        ),
        'config': {
            'max_fruits': config.max_fruits,
            'physics_fps': config.physics_fps,
            'max_physics_frames': config.max_physics_frames,
            'stable_frames': config.stable_frames,
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
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.replay_samples:
        sampled_envs = select_replay_environments(
            args.replay_selection,
            args.replay_samples,
            args.seed,
            settle_timeout_counts_cpu,
            step_counts_cpu,
        )
        terminal_metadata = {}
        for env_index in sampled_envs:
            end_kind = (
                'terminated'
                if bool(terminated_cpu[env_index])
                else (
                    'truncated'
                    if bool(truncated_cpu[env_index])
                    else 'capped'
                )
            )
            terminal_metadata[env_index] = {
                'end_kind': end_kind,
                'step_count': int(step_counts_cpu[env_index]),
                'score': int(scores_cpu[env_index]),
                'settle_timeout_count': int(
                    settle_timeout_counts_cpu[env_index]
                ),
                'settle_timeout_rate': (
                    int(settle_timeout_counts_cpu[env_index])
                    / max(1, int(step_counts_cpu[env_index]))
                ),
            }
        args.replay_output_dir.mkdir(parents=True, exist_ok=True)
        replay_plan = {
            'physics_identity': PHYSICS_IDENTITY,
            'seed': args.seed,
            'source_num_envs': args.num_envs,
            'frame_stride': args.replay_frame_stride,
            'selection': args.replay_selection,
            'replay_output_dir': str(args.replay_output_dir.resolve()),
            'sampled_envs': sampled_envs,
            'terminal_metadata': terminal_metadata,
            'config': {
                'max_fruits': config.max_fruits,
                'physics_fps': config.physics_fps,
                'max_physics_frames': config.max_physics_frames,
                'stable_frames': config.stable_frames,
                'solver_iterations': config.solver_iterations,
                'drop_fast_forward': config.drop_fast_forward,
                'adaptive_collision_substeps': (
                    config.adaptive_collision_substeps
                ),
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
            },
        }
        replay_plan_path = args.replay_output_dir / 'replay-plan.json'
        replay_plan_path.write_text(
            json.dumps(replay_plan, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        report['replay_plan'] = str(replay_plan_path.resolve())
        report['sampled_replay_envs'] = sampled_envs
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        replay_function = (
            replay_full_episodes
            if args.replay_full_episodes
            else replay_final_steps
        )
        replay_manifest, manifest_path = replay_function(
            args, config, sampled_envs, terminal_metadata
        )
        report['replay_manifest'] = str(manifest_path.resolve())
        report['sampled_replay_envs'] = sampled_envs
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(json.dumps({
            'replay_manifest': str(manifest_path.resolve()),
            'combined_replay': replay_manifest['combined_replay'],
            'sampled_replay_envs': sampled_envs,
        } if not args.replay_full_episodes else {
            'replay_manifest': str(manifest_path.resolve()),
            'full_episode_index': replay_manifest['index'],
            'sampled_replay_envs': sampled_envs,
        }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
