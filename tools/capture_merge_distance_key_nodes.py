"""批量捕捉高等级水果合成步距预测的稳定跳变，并保存成对快照。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
import torch  # noqa: E402

from daxigua.rl.merge_distance import (  # noqa: E402
    MergeDistanceConfig,
    MergeDistancePredictor,
)
from daxigua.rl.merge_distance_key_nodes import (  # noqa: E402
    StableSwitchDetector,
    merge_difficulty,
)
from daxigua.rl.observations import TensorState  # noqa: E402
from daxigua.rl.viewer import (  # noqa: E402
    load_viewer_model,
    viewer_simulator_config,
)
from daxigua.simulator import TensorVectorSimulator  # noqa: E402


DEFAULT_POLICY_CHECKPOINT = (
    PROJECT_ROOT / 'runs'
    / 'cloud_rtx5090_auxiliary_action_structured_branch_seed20260811_128m'
    / 'checkpoints' / 'final.pt'
)
DEFAULT_PREDICTOR_CHECKPOINT = (
    PROJECT_ROOT / 'runs' / 'merge_distance'
    / 'sab128_merge_distance_20k_20260817' / 'checkpoints' / 'final.pt'
)
POLICY_SHA256 = (
    'fc40b9019c65ecba8502f4334d1418b4f93c0e54e984d42ccc4d0b477bddca07'
)
PREDICTOR_SHA256 = (
    '8f96212a8eff99ef5e2fd01911512df23a043565c473b9d6c684cf22fcff17e6'
)
SEED_STRIDE = 1_000_003
COLORS = (
    '#7d3c98', '#d35400', '#e67e22', '#f4d03f', '#73c66b', '#d94b45',
    '#f1948a', '#d4a017', '#8d6e63', '#239b56', '#117864',
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_predictor(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('purpose') != 'merge_distance_predictor':
        raise ValueError('checkpoint is not a merge distance predictor')
    config = MergeDistanceConfig.from_dict(payload['model_config'])
    model = MergeDistancePredictor(config, **payload['geometry_config'])
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device).eval(), payload


def _class_label(class_index, horizons):
    class_index = int(class_index)
    if class_index == 0:
        return f'T<= {horizons[0]}'
    if class_index < len(horizons):
        return f'{horizons[class_index - 1]} < T <= {horizons[class_index]}'
    if class_index == len(horizons):
        return f'T > {horizons[-1]}'
    return 'terminal-unmerged'


def _capture_snapshot(observation, row, target_id, prediction):
    row = int(row)
    return {
        'positions': observation.positions[row].detach().cpu(),
        'radii': observation.physics_radii[row].detach().cpu(),
        'levels': observation.levels[row].detach().cpu(),
        'fruit_ids': observation.fruit_ids[row].detach().cpu(),
        'active': observation.active[row].detach().cpu(),
        'score': int(observation.score[row].item()),
        'drops': int(observation.step_count[row].item()),
        'target_id': int(target_id),
        'difficulty': float(prediction['difficulty']),
        'predicted_class': int(prediction['predicted_class']),
        'confidence': float(prediction['confidence']),
        'eventual_merge_probability': float(
            prediction['eventual_merge_probability']
        ),
    }


def _draw_snapshot(axis, snapshot, config, horizons, title):
    axis.set_facecolor('#f7f1e5')
    axis.add_patch(Rectangle(
        (config.wall_width, 0),
        config.board_width - 2 * config.wall_width,
        config.board_height - config.wall_width,
        fill=False,
        edgecolor='#4b3b2a',
        linewidth=2.0,
    ))
    axis.axhline(
        config.spawn_y,
        color='#c0392b',
        linewidth=1.0,
        linestyle='--',
        alpha=0.75,
    )
    for slot in torch.nonzero(snapshot['active'], as_tuple=False).flatten():
        slot = int(slot.item())
        level = int(snapshot['levels'][slot].item())
        fruit_id = int(snapshot['fruit_ids'][slot].item())
        x, y = (float(value) for value in snapshot['positions'][slot].tolist())
        radius = float(snapshot['radii'][slot].item())
        target = fruit_id == snapshot['target_id']
        axis.add_patch(Circle(
            (x, y),
            radius,
            facecolor=COLORS[level - 1],
            edgecolor='#111827' if target else 'white',
            linewidth=4.0 if target else 1.0,
            alpha=0.92,
        ))
        axis.text(
            x,
            y,
            f'L{level}',
            ha='center',
            va='center',
            fontsize=7,
            color='white',
            weight='bold',
        )
    label = _class_label(snapshot['predicted_class'], horizons)
    axis.set_title(
        f"{title} · drop {snapshot['drops']} · score {snapshot['score']}\n"
        f"difficulty {snapshot['difficulty']:.2f} · {label} · "
        f"p={snapshot['confidence']:.2f}",
        fontsize=9,
    )
    axis.set_xlim(0, config.board_width)
    axis.set_ylim(config.board_height, 0)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])


def render_pair(start, end, switch, config, horizons, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 7.0), squeeze=False)
    _draw_snapshot(axes[0, 0], start, config, horizons, 'transition start')
    _draw_snapshot(axes[0, 1], end, config, horizons, 'new stable state')
    figure.suptitle(
        f"L{end['target_level']} fruit #{end['target_id']} · "
        f"{switch.direction} · stable-platform delta {switch.delta:+.2f}",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def _autocast(enabled, device):
    if not enabled or device.type != 'cuda':
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='捕捉L7-L11水果合成步距预测的稳定区间跳变。'
    )
    parser.add_argument('--policy-checkpoint', type=Path, default=DEFAULT_POLICY_CHECKPOINT)
    parser.add_argument('--predictor-checkpoint', type=Path, default=DEFAULT_PREDICTOR_CHECKPOINT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--physics-fps', type=int, choices=(30, 120), default=30)
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--parallel-envs', type=int, default=256)
    parser.add_argument('--seed-base', type=int, default=83_000_000)
    parser.add_argument('--max-drops', type=int, default=2000)
    parser.add_argument('--max-wall-seconds', type=float, default=0.0)
    parser.add_argument('--min-level', type=int, default=7)
    parser.add_argument('--max-level', type=int, default=11)
    parser.add_argument('--prediction-stride', type=int, default=1)
    parser.add_argument('--stable-window', type=int, default=4)
    parser.add_argument('--stable-range', type=float, default=0.4)
    parser.add_argument('--jump-threshold', type=float, default=0.75)
    parser.add_argument('--transition-timeout', type=int, default=32)
    parser.add_argument('--max-pairs', type=int, default=80)
    parser.add_argument('--progress-seconds', type=float, default=10.0)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument(
        '--autocast-bfloat16',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args(argv)


def _validate_args(args):
    for name in ('episodes', 'parallel_envs', 'prediction_stride', 'max_pairs'):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f'{name} must be positive')
    if args.max_drops <= 0:
        raise ValueError('max_drops must be positive for this diagnostic')
    if not 1 <= args.min_level <= args.max_level <= 11:
        raise ValueError('target level range must be inside L1-L11')


@torch.inference_mode()
def run(args):
    _validate_args(args)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(tempfile.mkdtemp(prefix='daxigua-merge-key-nodes-')).resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = output_dir / 'pairs'
    pairs_dir.mkdir(exist_ok=True)

    loaded = load_viewer_model(args.policy_checkpoint, device=args.device)
    if loaded.checkpoint_sha256 != POLICY_SHA256:
        raise ValueError('policy checkpoint does not match SAB-128')
    predictor_sha = _sha256(args.predictor_checkpoint)
    if predictor_sha != PREDICTOR_SHA256:
        raise ValueError('predictor checkpoint SHA-256 mismatch')
    predictor, predictor_payload = _load_predictor(
        args.predictor_checkpoint, loaded.device
    )
    horizons = tuple(predictor.config.horizons)
    config = viewer_simulator_config(
        args.physics_fps, loaded.model_config, loaded.device
    )
    batch_size = min(int(args.episodes), int(args.parallel_envs))
    simulator = TensorVectorSimulator(
        batch_size, config=config, device=loaded.device
    )
    all_seeds = (
        torch.arange(args.episodes, dtype=torch.int64)
        .mul_(SEED_STRIDE)
        .add_(args.seed_base)
    )
    episode_ids = torch.arange(
        batch_size, dtype=torch.int64, device=loaded.device
    )
    slot_seeds = all_seeds[:batch_size].to(loaded.device)
    simulator.reset(seeds=slot_seeds)

    detectors = {}
    start_snapshots = {}
    events = []
    completed = 0
    next_episode = batch_size
    decisions = 0
    transitions = 0
    started = time.perf_counter()
    last_progress = started
    stop_reason = 'episodes_complete'

    while bool((episode_ids >= 0).any().item()):
        observation = simulator.observe()
        enabled = episode_ids >= 0
        predict_now = (observation.step_count % args.prediction_stride) == 0
        target_mask = (
            observation.active
            & (observation.levels >= args.min_level)
            & (observation.levels <= args.max_level)
            & enabled.unsqueeze(1)
            & predict_now.unsqueeze(1)
        )
        target_rows = torch.nonzero(
            target_mask.any(dim=1), as_tuple=False
        ).flatten()
        current_keys = set()
        if target_rows.numel() > 0:
            state = TensorState.from_observation(
                observation,
                physics_fps=config.physics_fps,
                rows=target_rows,
            )
            with _autocast(args.autocast_bfloat16, loaded.device):
                probabilities = predictor(state).probabilities.float()
            local_indices = torch.nonzero(
                target_mask[target_rows], as_tuple=False
            )
            global_rows = target_rows[local_indices[:, 0]]
            slots = local_indices[:, 1]
            selected = probabilities[local_indices[:, 0], slots]
            difficulties = merge_difficulty(selected)
            classes = selected.argmax(dim=-1)
            confidences = selected.gather(1, classes[:, None]).squeeze(1)
            eventual = 1.0 - selected[:, predictor.config.terminal_unmerged_class]
            integer_rows = torch.stack((
                global_rows,
                slots,
                observation.fruit_ids[global_rows, slots],
                observation.levels[global_rows, slots],
                observation.step_count[global_rows],
                episode_ids[global_rows],
            ), dim=1).detach().cpu().tolist()
            float_rows = torch.stack((
                difficulties, confidences, eventual
            ), dim=1).detach().cpu().tolist()
            class_rows = classes.detach().cpu().tolist()

            for integers, floats, predicted_class in zip(
                    integer_rows, float_rows, class_rows):
                row, slot, fruit_id, level, drop, episode_id = integers
                key = (int(row), int(fruit_id))
                current_keys.add(key)
                detector = detectors.get(key)
                if detector is None:
                    detector = StableSwitchDetector(
                        window_size=args.stable_window,
                        stable_range=args.stable_range,
                        jump_threshold=args.jump_threshold,
                        transition_timeout=args.transition_timeout,
                    )
                    detectors[key] = detector
                prediction = {
                    'difficulty': floats[0],
                    'confidence': floats[1],
                    'eventual_merge_probability': floats[2],
                    'predicted_class': predicted_class,
                }
                update = detector.update(floats[0], drop)
                if update.transition_started:
                    start_snapshots[key] = _capture_snapshot(
                        observation, row, fruit_id, prediction
                    )
                if update.transition_cancelled:
                    start_snapshots.pop(key, None)
                if update.switch is None:
                    continue
                start_snapshot = start_snapshots.pop(key, None)
                if start_snapshot is None:
                    continue
                end_snapshot = _capture_snapshot(
                    observation, row, fruit_id, prediction
                )
                end_snapshot['target_level'] = int(level)
                event_index = len(events) + 1
                filename = (
                    f'{event_index:04d}_episode_{episode_id:04d}_'
                    f'L{level}_fruit_{fruit_id}_drops_'
                    f'{update.switch.start_step}_{update.switch.confirmed_step}.png'
                )
                image_path = pairs_dir / filename
                render_pair(
                    start_snapshot,
                    end_snapshot,
                    update.switch,
                    config,
                    horizons,
                    image_path,
                )
                event = {
                    'event_index': event_index,
                    'episode_id': int(episode_id),
                    'seed': int(all_seeds[episode_id].item()),
                    'env_row': int(row),
                    'fruit_id': int(fruit_id),
                    'level': int(level),
                    'direction': update.switch.direction,
                    'previous_difficulty': update.switch.previous_value,
                    'new_difficulty': update.switch.new_value,
                    'delta': update.switch.delta,
                    'pair_endpoint_delta': (
                        end_snapshot['difficulty']
                        - start_snapshot['difficulty']
                    ),
                    'transition_start_drop': update.switch.start_step,
                    'new_stable_start_drop': update.switch.settled_start_step,
                    'confirmation_drop': update.switch.confirmed_step,
                    'start_prediction': {
                        key: value for key, value in start_snapshot.items()
                        if key in (
                            'difficulty', 'predicted_class', 'confidence',
                            'eventual_merge_probability',
                        )
                    },
                    'end_prediction': prediction,
                    'image': str(image_path),
                }
                events.append(event)
                with (output_dir / 'events.jsonl').open(
                        'a', encoding='utf-8') as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + '\n')
                if len(events) >= args.max_pairs:
                    stop_reason = 'max_pairs_reached'
                    break

        if predict_now.any():
            for key in tuple(detectors):
                row = key[0]
                if bool(predict_now[row].item()) and key not in current_keys:
                    detectors.pop(key, None)
                    start_snapshots.pop(key, None)
        if len(events) >= args.max_pairs:
            break

        with _autocast(args.autocast_bfloat16, loaded.device):
            state = TensorState.from_observation(
                observation, physics_fps=config.physics_fps
            )
            actions = loaded.model(state).argmax(dim=1)
        result = simulator.step_masked(actions, enabled)
        transitions += int(enabled.sum().item())
        decisions += 1
        capped = result.observation.step_count >= args.max_drops
        finished = enabled & (
            result.physics.done | result.physics.truncated | capped
        )
        finished_rows = torch.nonzero(
            finished, as_tuple=False
        ).flatten()
        if finished_rows.numel() > 0:
            completed += int(finished_rows.numel())
            for row in finished_rows.detach().cpu().tolist():
                for key in tuple(detectors):
                    if key[0] == row:
                        detectors.pop(key, None)
                        start_snapshots.pop(key, None)
            replacement_count = min(
                int(finished_rows.numel()), args.episodes - next_episode
            )
            reset_seeds = torch.arange(
                int(finished_rows.numel()),
                dtype=torch.int64,
                device=loaded.device,
            ).add_(args.seed_base + 9_000_000_000)
            replacements = torch.full(
                (int(finished_rows.numel()),),
                -1,
                dtype=torch.int64,
                device=loaded.device,
            )
            if replacement_count > 0:
                replacements[:replacement_count] = torch.arange(
                    next_episode,
                    next_episode + replacement_count,
                    dtype=torch.int64,
                    device=loaded.device,
                )
                reset_seeds[:replacement_count] = all_seeds[
                    next_episode:next_episode + replacement_count
                ].to(loaded.device)
                next_episode += replacement_count
            simulator.reset(finished, seeds=reset_seeds)
            episode_ids[finished_rows] = replacements
            slot_seeds[finished_rows] = reset_seeds

        now = time.perf_counter()
        if now - last_progress >= args.progress_seconds:
            elapsed = now - started
            print(json.dumps({
                'completed_episodes': completed,
                'active_environments': int((episode_ids >= 0).sum().item()),
                'pairs': len(events),
                'tracked_fruits': len(detectors),
                'transitions_per_second': transitions / max(elapsed, 1e-9),
            }, ensure_ascii=False), flush=True)
            last_progress = now
        if args.max_wall_seconds > 0 and now - started >= args.max_wall_seconds:
            stop_reason = 'wall_time_reached'
            break

    elapsed = time.perf_counter() - started
    manifest = {
        'purpose': 'merge_distance_key_node_diagnostic',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': stop_reason,
        'policy_checkpoint': str(loaded.checkpoint_path),
        'policy_checkpoint_sha256': loaded.checkpoint_sha256,
        'predictor_checkpoint': str(args.predictor_checkpoint.resolve()),
        'predictor_checkpoint_sha256': predictor_sha,
        'predictor_metrics': predictor_payload.get('metrics'),
        'physics_fps': config.physics_fps,
        'parameters': {
            key: getattr(args, key)
            for key in (
                'episodes', 'parallel_envs', 'seed_base', 'max_drops',
                'min_level', 'max_level', 'prediction_stride',
                'stable_window', 'stable_range', 'jump_threshold',
                'transition_timeout', 'max_pairs',
            )
        },
        'completed_episodes': completed,
        'decision_batches': decisions,
        'transitions': transitions,
        'elapsed_seconds': elapsed,
        'transitions_per_second': transitions / max(elapsed, 1e-9),
        'event_count': len(events),
        'improved_count': sum(
            event['direction'] == 'improved' for event in events
        ),
        'worsened_count': sum(
            event['direction'] == 'worsened' for event in events
        ),
        'events': events,
    }
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps({
        'output_dir': str(output_dir),
        'manifest': str(manifest_path),
        'pairs': len(events),
        'status': stop_reason,
        'elapsed_seconds': elapsed,
    }, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
