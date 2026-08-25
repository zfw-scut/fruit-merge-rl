"""并行采集不同程度负可合成性变化的投放前后场景。"""

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

from daxigua.rl.mergeability import (  # noqa: E402
    MergeabilityCalculator,
    MergeabilityConfig,
)
from daxigua.rl.mergeability_rollout import (  # noqa: E402
    scene_mergeability_delta,
    scene_mergeability_values,
)
from daxigua.rl.mergeability_transitions import (  # noqa: E402
    DEFAULT_NEGATIVE_SEVERITY_BANDS,
    PriorityReservoir,
    capture_compact_scene_rows,
    clone_compact_scene_batch,
    compact_scene_row,
    negative_severity_codes,
    select_compact_scene_rows,
    severity_band_manifest,
    update_compact_scene_batch,
)
from daxigua.rl.observations import TensorState  # noqa: E402
from daxigua.rl.viewer import (  # noqa: E402
    load_viewer_model,
    viewer_simulator_config,
)
from daxigua.simulator import PHYSICS_IDENTITY, TensorVectorSimulator  # noqa: E402


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / 'runs'
    / 'cloud_rtx5090_auxiliary_action_structured_branch_seed20260811_128m'
    / 'checkpoints'
    / 'final.pt'
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / 'runs' / 'diagnostics'
    / 'mergeability_negative_transitions_20260825'
)
SEED_STRIDE = 1_000_003
DATASET_FORMAT = 'daxigua_mergeability_negative_transition_dataset'
DATASET_VERSION = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='用SAB-128采集分层负变化的一步前后场景。'
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-envs', type=int, default=2000)
    parser.add_argument('--decision-steps', type=int, default=1000)
    parser.add_argument('--quota-per-band', type=int, default=64)
    parser.add_argument('--seed-base', type=int, default=202_608_270_000)
    parser.add_argument('--priority-seed', type=int, default=202_608_271)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--progress-interval', type=int, default=25)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _validate_args(args):
    for name in (
            'num_envs', 'decision_steps', 'quota_per_band',
            'progress_interval'):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f'{name.replace("_", "-")} must be positive')
    if int(args.warmup_steps) < 0:
        raise ValueError('warmup-steps must be non-negative')


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git_revision():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    os.replace(temporary, path)


def _safe_print(payload, *, indent=None):
    try:
        print(
            json.dumps(payload, ensure_ascii=False, indent=indent), flush=True
        )
    except BrokenPipeError:  # detached SSH progress output must not kill rollout
        pass


def _reset_finished(
        simulator,
        finished,
        episode_ids,
        episode_seeds,
        *,
        next_episode_id,
        seed_base):
    rows = torch.nonzero(finished, as_tuple=False).flatten()
    count = int(rows.numel())
    if count == 0:
        return next_episode_id
    new_ids = torch.arange(
        next_episode_id,
        next_episode_id + count,
        dtype=torch.int64,
        device=simulator.device,
    )
    new_seeds = new_ids * SEED_STRIDE + int(seed_base)
    episode_ids[rows] = new_ids
    episode_seeds[rows] = new_seeds
    simulator.reset(mask=finished, seeds=new_seeds)
    return next_episode_id + count


def _selected_cpu(value, rows):
    return value.index_select(0, rows).detach().cpu().clone()


def _candidate_batch(
        *,
        rows,
        priorities,
        before_scene,
        after_observation,
        after_mergeability,
        previous_value,
        previous_area,
        previous_mean,
        after_value,
        after_area,
        after_mean,
        delta,
        actions,
        step,
        finished,
        env_ids,
        episode_ids,
        episode_seeds,
        decision_step):
    before = select_compact_scene_rows(before_scene, rows)
    after = capture_compact_scene_rows(
        after_observation, after_mergeability.score, rows
    )
    merge_events = step.physics.merge_events
    metadata = {
        'environment_id': _selected_cpu(env_ids, rows),
        'episode_id': _selected_cpu(episode_ids, rows),
        'episode_seed': _selected_cpu(episode_seeds, rows),
        'episode_drop': _selected_cpu(after_observation.step_count, rows),
        'action_index': _selected_cpu(actions, rows),
        'drop_x': _selected_cpu(step.drop.drop_x, rows),
        'dropped_level': _selected_cpu(step.drop.dropped_levels, rows),
        'dropped_fruit_id': _selected_cpu(step.drop.fruit_ids, rows),
        'score_delta': _selected_cpu(step.physics.score_delta, rows),
        'merge_count': _selected_cpu(merge_events.count, rows),
        'terminal': _selected_cpu(finished, rows),
        'before_scene_value': _selected_cpu(previous_value, rows),
        'after_scene_value': _selected_cpu(after_value, rows),
        'before_occupied_area': _selected_cpu(previous_area, rows),
        'after_occupied_area': _selected_cpu(after_area, rows),
        'before_weighted_mean': _selected_cpu(previous_mean, rows),
        'after_weighted_mean': _selected_cpu(after_mean, rows),
        'delta': _selected_cpu(delta, rows),
        'priority': priorities.detach().cpu().clone(),
    }
    event_values = {
        'source_levels': _selected_cpu(merge_events.source_levels, rows),
        'new_levels': _selected_cpu(merge_events.new_levels, rows),
        'positions': _selected_cpu(merge_events.positions, rows),
        'score_deltas': _selected_cpu(merge_events.score_deltas, rows),
        'source_ids': _selected_cpu(merge_events.source_ids, rows),
        'new_fruit_ids': _selected_cpu(merge_events.new_fruit_ids, rows),
    }
    samples = []
    for row_index in range(int(rows.numel())):
        sample = {
            name: (
                bool(values[row_index].item())
                if name == 'terminal'
                else float(values[row_index].item())
                if name in {
                    'drop_x', 'before_scene_value', 'after_scene_value',
                    'before_occupied_area', 'after_occupied_area',
                    'before_weighted_mean', 'after_weighted_mean', 'delta',
                    'priority',
                }
                else int(values[row_index].item())
            )
            for name, values in metadata.items()
        }
        sample['decision_step'] = int(decision_step)
        sample['before'] = compact_scene_row(before, row_index)
        sample['after'] = compact_scene_row(after, row_index)
        sample['merge_events'] = {
            name: values[row_index].clone()
            for name, values in event_values.items()
        }
        samples.append(sample)
    return samples


@torch.inference_mode()
def collect(args):
    _validate_args(args)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.json'
    dataset_path = output_dir / 'negative_transitions.pt'

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

    bands = DEFAULT_NEGATIVE_SEVERITY_BANDS
    reservoir = PriorityReservoir(len(bands), args.quota_per_band)
    priority_generator = torch.Generator(device=loaded.device)
    priority_generator.manual_seed(int(args.priority_seed))
    band_counts = torch.zeros(
        len(bands), dtype=torch.int64, device=loaded.device
    )
    previous_value = torch.zeros(
        args.num_envs, dtype=torch.float32, device=loaded.device
    )
    previous_area = torch.zeros_like(previous_value)
    previous_mean = torch.zeros_like(previous_value)
    previous_valid = torch.zeros(
        args.num_envs, dtype=torch.bool, device=loaded.device
    )
    previous_scene = None
    next_episode_id = args.num_envs
    completed_episodes = 0
    started = time.perf_counter()
    base_manifest = {
        'format_version': 1,
        'purpose': 'mergeability_negative_transition_gallery_source',
        'created_at_utc': _utc_now(),
        'status': 'running',
        'git_revision': _git_revision(),
        'physics_identity': PHYSICS_IDENTITY,
        'simulator_config': asdict(config),
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'policy': 'SAB-128 greedy',
        'scene_value_definition': 'sum(mergeability * pi * physics_radius^2)',
        'selection': 'uniform random priority reservoir within severity band',
        'severity_bands': severity_band_manifest(bands),
        'parameters': {
            'num_envs': int(args.num_envs),
            'decision_steps': int(args.decision_steps),
            'quota_per_band': int(args.quota_per_band),
            'seed_base': int(args.seed_base),
            'seed_stride': SEED_STRIDE,
            'priority_seed': int(args.priority_seed),
            'warmup_steps': int(args.warmup_steps),
        },
        'device': str(loaded.device),
        'cuda_device_name': (
            torch.cuda.get_device_name(loaded.device)
            if loaded.device.type == 'cuda' else None
        ),
    }
    _atomic_json(manifest_path, base_manifest)

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
            codes = negative_severity_codes(delta, delta_valid, bands)
            encoded = (codes.to(torch.int64) + 1).clamp_min(0)
            band_counts.add_(torch.bincount(
                encoded, minlength=len(bands) + 1
            )[1:])

            if previous_scene is not None:
                priorities = torch.rand(
                    args.num_envs,
                    dtype=torch.float32,
                    device=loaded.device,
                    generator=priority_generator,
                )
                top_k = min(args.quota_per_band, args.num_envs)
                finished = step.physics.done | step.physics.truncated
                for band in bands:
                    minimum = reservoir.minimum_priority(band.code)
                    eligible = (
                        (codes == band.code) & (priorities > minimum)
                    )
                    ranked = priorities.masked_fill(~eligible, -1.0)
                    values, rows = torch.topk(ranked, k=top_k)
                    keep = values >= 0.0
                    if not bool(keep.any().item()):
                        continue
                    rows = rows[keep]
                    values = values[keep]
                    candidates = _candidate_batch(
                        rows=rows,
                        priorities=values,
                        before_scene=previous_scene,
                        after_observation=current,
                        after_mergeability=result,
                        previous_value=previous_value,
                        previous_area=previous_area,
                        previous_mean=previous_mean,
                        after_value=scene_value,
                        after_area=occupied_area,
                        after_mean=weighted_mean,
                        delta=delta,
                        actions=actions,
                        step=step,
                        finished=finished,
                        env_ids=env_ids,
                        episode_ids=episode_ids,
                        episode_seeds=episode_seeds,
                        decision_step=decision_step,
                    )
                    for candidate in candidates:
                        candidate['severity_code'] = int(band.code)
                        candidate['severity_key'] = band.key
                        reservoir.add(
                            band.code, candidate['priority'], candidate
                        )
            else:
                finished = step.physics.done | step.physics.truncated

            if previous_scene is None:
                previous_scene = clone_compact_scene_batch(
                    current, result.score
                )
            else:
                update_compact_scene_batch(
                    previous_scene, current, result.score
                )
            previous_value.copy_(scene_value)
            previous_area.copy_(occupied_area)
            previous_mean.copy_(weighted_mean)
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
                _safe_print({
                    'phase': 'collect',
                    'decision_step': decision_step,
                    'decision_steps': args.decision_steps,
                    'transitions': decision_step * args.num_envs,
                    'completed_episodes': completed_episodes,
                    'selected_per_band': reservoir.selected_counts(),
                    'elapsed_seconds': elapsed,
                    'env_steps_per_second': (
                        decision_step * args.num_envs / max(elapsed, 1e-9)
                    ),
                })
    except Exception as error:
        _atomic_json(manifest_path, {
            **base_manifest,
            'status': 'failed',
            'updated_at_utc': _utc_now(),
            'failure': f'{type(error).__name__}: {error}',
        })
        raise

    if loaded.device.type == 'cuda':
        torch.cuda.synchronize(loaded.device)
    elapsed = time.perf_counter() - started
    selected_counts = reservoir.selected_counts()
    complete = all(count == args.quota_per_band for count in selected_counts)
    samples = []
    for band in bands:
        selected = reservoir.samples(band.code)
        selected.sort(key=lambda item: abs(float(item['delta'])))
        samples.extend(selected)
    observed_counts = tuple(int(value) for value in band_counts.cpu().tolist())
    final_manifest = {
        **base_manifest,
        'status': 'complete' if complete else 'incomplete',
        'updated_at_utc': _utc_now(),
        'decision_steps': int(args.decision_steps),
        'transitions': int(args.decision_steps * args.num_envs),
        'completed_episodes': int(completed_episodes),
        'unique_episodes_started': int(next_episode_id),
        'observed_per_band': observed_counts,
        'selected_per_band': selected_counts,
        'selected_total': len(samples),
        'elapsed_seconds': elapsed,
        'env_steps_per_second': (
            args.decision_steps * args.num_envs / max(elapsed, 1e-9)
        ),
        'peak_cuda_allocated_bytes': (
            int(torch.cuda.max_memory_allocated(loaded.device))
            if loaded.device.type == 'cuda' else None
        ),
        'peak_cuda_reserved_bytes': (
            int(torch.cuda.max_memory_reserved(loaded.device))
            if loaded.device.type == 'cuda' else None
        ),
        'dataset': str(dataset_path),
        'failure': None,
    }
    torch.save({
        'format': DATASET_FORMAT,
        'format_version': DATASET_VERSION,
        'manifest': final_manifest,
        'severity_bands': severity_band_manifest(bands),
        'samples': samples,
    }, dataset_path)
    _atomic_json(manifest_path, final_manifest)
    _safe_print(final_manifest, indent=2)
    if not complete and not args.allow_incomplete:
        raise RuntimeError(
            f'negative transition quotas were not filled: {selected_counts}'
        )
    return dataset_path


def main(argv=None):
    collect(parse_args(argv))


if __name__ == '__main__':
    main()
