"""用 SAB-128 在 CPU 并行场景中采集同级水果对长期停滞验证图。"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
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

from daxigua.rl.observations import TensorState  # noqa: E402
from daxigua.rl.pair_failure import (  # noqa: E402
    PairFailureConfig,
    PairFailureTracker,
)
from daxigua.rl.viewer import (  # noqa: E402
    load_viewer_model,
    viewer_simulator_config,
)
from daxigua.simulator import TensorVectorSimulator  # noqa: E402


SAB128_SHA256 = (
    'fc40b9019c65ecba8502f4334d1418b4f93c0e54e984d42ccc4d0b477bddca07'
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / 'runs'
    / 'cloud_rtx5090_auxiliary_action_structured_branch_seed20260811_128m'
    / 'checkpoints'
    / 'final.pt'
)
SEED_STRIDE = 1_000_003
COLORS = (
    '#7d3c98', '#d35400', '#e67e22', '#f4d03f', '#73c66b', '#d94b45',
    '#f1948a', '#d4a017', '#8d6e63', '#239b56', '#117864',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='CPU 并行采集 L7～L11 同级水果对长期停滞验证图。'
    )
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--seed-base', type=int, default=116_000_000)
    parser.add_argument('--max-drops', type=int, default=1000)
    parser.add_argument('--max-images', type=int, default=16)
    parser.add_argument('--motion-window-drops', type=int, default=4)
    parser.add_argument('--confirmation-drops', type=int, default=24)
    parser.add_argument('--max-net-displacement-ratio', type=float, default=0.12)
    parser.add_argument('--adjacent-surface-gap-ratio', type=float, default=1.25)
    parser.add_argument('--progress-seconds', type=float, default=10.0)
    parser.add_argument(
        '--output-dir', type=Path,
        help='省略时在系统临时目录创建输出目录。',
    )
    return parser.parse_args(argv)


def _scene_snapshot(observation, row, *, episode, seed):
    return {
        'episode': int(episode),
        'seed': int(seed),
        'step': int(observation.step_count[row].item()),
        'score': int(observation.score[row].item()),
        'positions': observation.positions[row].detach().cpu().clone(),
        'radii': observation.physics_radii[row].detach().cpu().clone(),
        'levels': observation.levels[row].detach().cpu().clone(),
        'fruit_ids': observation.fruit_ids[row].detach().cpu().clone(),
        'active': observation.active[row].detach().cpu().clone(),
    }


def _snapshot_at(history, step):
    for snapshot in reversed(history):
        if snapshot['step'] == int(step):
            return snapshot
    return None


def _draw_scene(axis, snapshot, config, pair_ids, title):
    axis.set_facecolor('#f7f1e5')
    axis.add_patch(Rectangle(
        (config.wall_width, 0),
        config.board_width - 2 * config.wall_width,
        config.board_height - config.wall_width,
        fill=False,
        edgecolor='#4b3b2a',
        linewidth=2.2,
    ))
    axis.axhline(
        config.spawn_y,
        color='#c0392b',
        linewidth=1.0,
        linestyle='--',
        alpha=0.8,
    )
    pair_positions = []
    active_slots = torch.nonzero(
        snapshot['active'], as_tuple=False
    ).flatten().tolist()
    for slot in active_slots:
        level = int(snapshot['levels'][slot].item())
        x, y = (
            float(value) for value in snapshot['positions'][slot].tolist()
        )
        radius = float(snapshot['radii'][slot].item())
        fruit_id = int(snapshot['fruit_ids'][slot].item())
        selected = fruit_id in pair_ids
        axis.add_patch(Circle(
            (x, y),
            radius,
            facecolor=COLORS[level - 1],
            edgecolor='#e63946' if selected else 'white',
            linewidth=4.0 if selected else 1.0,
            alpha=0.92,
            zorder=3 if selected else 2,
        ))
        axis.text(
            x,
            y,
            f'L{level}\n#{fruit_id}',
            ha='center',
            va='center',
            fontsize=8,
            color='white',
            weight='bold',
            zorder=4,
        )
        if selected:
            pair_positions.append((x, y))
    if len(pair_positions) == 2:
        axis.plot(
            [pair_positions[0][0], pair_positions[1][0]],
            [pair_positions[0][1], pair_positions[1][1]],
            color='#e63946',
            linewidth=1.8,
            linestyle=':',
            zorder=5,
        )
    axis.set_xlim(0, config.board_width)
    axis.set_ylim(config.board_height, 0)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(
        f'{title}\ndrop={snapshot["step"]} · score={snapshot["score"]}',
        fontsize=10,
    )


def render_event(pre_onset, onset, confirmation, event, config, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 7.2))
    pair_ids = (event['fruit_id_i'], event['fruit_id_j'])
    _draw_scene(
        axes[0], pre_onset, config, pair_ids,
        'Pre-onset reference · condition not yet met',
    )
    _draw_scene(axes[1], onset, config, pair_ids, 'Candidate onset')
    _draw_scene(
        axes[2], confirmation, config, pair_ids, 'Failure confirmed'
    )
    band = 'high L9-L11' if event['level'] >= 9 else 'medium L7-L8'
    figure.suptitle(
        f"Pair stagnation · {band} · L{event['level']} · "
        f"{event['duration_drops']} drops\n"
        f"normalized net displacement="
        f"({event['net_displacement_ratio_i']:.3f}, "
        f"{event['net_displacement_ratio_j']:.3f}) · "
        f"surface gap={event['surface_gap_ratio']:.3f}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def render_gallery(image_paths, output_path):
    if not image_paths:
        return None
    columns = 2
    rows = math.ceil(len(image_paths) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(8.5 * columns, 6.8 * rows), squeeze=False
    )
    for index, axis in enumerate(axes.flat):
        axis.axis('off')
        if index < len(image_paths):
            axis.imshow(plt.imread(image_paths[index]))
            axis.set_title(image_paths[index].stem, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(figure)
    return output_path


def _new_episode_seed(seed_base, episode_index):
    return int(seed_base + episode_index * SEED_STRIDE)


@torch.inference_mode()
def run(args):
    if args.num_envs <= 0 or args.episodes <= 0:
        raise ValueError('num-envs and episodes must be positive')
    if args.num_envs > args.episodes:
        raise ValueError('num-envs cannot exceed episodes')
    if args.max_drops <= 0 or args.max_images < 0:
        raise ValueError('max-drops must be positive and max-images non-negative')

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(tempfile.mkdtemp(prefix='daxigua-pair-failure-')).resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / 'images'
    images_dir.mkdir()

    detector_config = PairFailureConfig(
        motion_window_drops=args.motion_window_drops,
        confirmation_drops=args.confirmation_drops,
        max_net_displacement_ratio=args.max_net_displacement_ratio,
        adjacent_surface_gap_ratio=args.adjacent_surface_gap_ratio,
    )
    loaded = load_viewer_model(args.checkpoint, device='cpu')
    if loaded.checkpoint_sha256.lower() != SAB128_SHA256:
        raise ValueError('checkpoint does not match the registered SAB-128')
    config = viewer_simulator_config(30, loaded.model_config, loaded.device)
    simulator = TensorVectorSimulator(
        args.num_envs, config=config, device=loaded.device
    )
    tracker = PairFailureTracker(
        args.num_envs,
        loaded.model_config.max_fruits,
        device=loaded.device,
        config=detector_config,
    )

    lane_episodes = list(range(args.num_envs))
    next_episode = args.num_envs
    lane_seeds = [
        _new_episode_seed(args.seed_base, episode)
        for episode in lane_episodes
    ]
    simulator.reset(seeds=torch.tensor(lane_seeds, dtype=torch.int64))
    initial = simulator.observe()
    tracker.update_observation(initial)
    scene_history = [
        # 检测器在短窗口结束时才回溯报告 onset，因此额外保留一帧，
        # 才能在人工验证图中恢复 onset 前一次投放的真实场景。
        deque(maxlen=detector_config.motion_window_drops + 2)
        for _ in range(args.num_envs)
    ]
    for lane in range(args.num_envs):
        scene_history[lane].append(_scene_snapshot(
            initial,
            lane,
            episode=lane_episodes[lane],
            seed=lane_seeds[lane],
        ))

    pending = {}
    event_rows = []
    episode_rows = []
    image_paths = []
    completed = 0
    transitions = 0
    candidate_starts = 0
    candidate_cancellations = 0
    confirmed_total = 0
    detector_seconds = 0.0
    started_at = time.perf_counter()
    last_progress = started_at

    while completed < args.episodes:
        observation = simulator.observe()
        state = TensorState.from_observation(
            observation, physics_fps=config.physics_fps
        )
        actions = loaded.model(state).argmax(dim=1)
        result = simulator.step(actions)
        transitions += args.num_envs
        after = result.observation

        for lane in range(args.num_envs):
            scene_history[lane].append(_scene_snapshot(
                after,
                lane,
                episode=lane_episodes[lane],
                seed=lane_seeds[lane],
            ))
        detector_started = time.perf_counter()
        update = tracker.update_observation(after)
        detector_seconds += time.perf_counter() - detector_started

        for lane, pair_index in torch.nonzero(
                update.started, as_tuple=False).tolist():
            if lane_episodes[lane] < 0:
                continue
            candidate_starts += 1
            onset_step = int(update.onset_steps[lane, pair_index].item())
            onset = _snapshot_at(scene_history[lane], onset_step)
            pre_onset = _snapshot_at(scene_history[lane], onset_step - 1)
            if onset is not None and pre_onset is not None:
                pending[(lane, pair_index)] = {
                    'pre_onset': pre_onset,
                    'onset': onset,
                    'episode': lane_episodes[lane],
                    'seed': lane_seeds[lane],
                    'slot_i': int(tracker.pair_i[pair_index].item()),
                    'slot_j': int(tracker.pair_j[pair_index].item()),
                }

        for lane, pair_index in torch.nonzero(
                update.confirmed, as_tuple=False).tolist():
            if lane_episodes[lane] < 0:
                continue
            confirmed_total += 1
            key = (lane, pair_index)
            record = pending.get(key)
            if record is None:
                continue
            confirmation = scene_history[lane][-1]
            event = {
                'event': confirmed_total,
                'episode': record['episode'] + 1,
                'seed': record['seed'],
                'level': int(update.levels[lane, pair_index].item()),
                'fruit_id_i': int(
                    update.fruit_id_i[lane, pair_index].item()
                ),
                'fruit_id_j': int(
                    update.fruit_id_j[lane, pair_index].item()
                ),
                'slot_i': record['slot_i'],
                'slot_j': record['slot_j'],
                'pre_onset_drop': record['pre_onset']['step'],
                'onset_drop': record['onset']['step'],
                'confirmed_drop': confirmation['step'],
                'duration_drops': int(
                    update.duration_drops[lane, pair_index].item()
                ),
                'net_displacement_ratio_i': float(
                    update.net_displacement_ratio_i[
                        lane, pair_index
                    ].item()
                ),
                'net_displacement_ratio_j': float(
                    update.net_displacement_ratio_j[
                        lane, pair_index
                    ].item()
                ),
                'surface_gap_ratio': float(
                    update.surface_gap_ratio[lane, pair_index].item()
                ),
            }
            if len(image_paths) < args.max_images:
                image_path = images_dir / (
                    f"event_{len(image_paths) + 1:03d}_L{event['level']}_"
                    f"episode_{event['episode']:03d}_"
                    f"drop_{event['onset_drop']:04d}.png"
                )
                render_event(
                    record['pre_onset'], record['onset'], confirmation,
                    event, config, image_path
                )
                event['image'] = image_path.name
                image_paths.append(image_path)
            else:
                event['image'] = None
            event_rows.append(event)

        for lane, pair_index in torch.nonzero(
                update.ended, as_tuple=False).tolist():
            if lane_episodes[lane] < 0:
                pending.pop((lane, pair_index), None)
                continue
            if not bool(update.ended_after_confirmation[lane, pair_index]):
                candidate_cancellations += 1
            pending.pop((lane, pair_index), None)

        capped = after.step_count >= args.max_drops
        finished = result.physics.done | result.physics.truncated | capped
        finished_lanes = torch.nonzero(
            finished, as_tuple=False
        ).flatten().tolist()
        if finished_lanes:
            reset_mask = torch.zeros(args.num_envs, dtype=torch.bool)
            reset_seeds = []
            for lane in finished_lanes:
                episode = lane_episodes[lane]
                if 0 <= episode < args.episodes:
                    episode_rows.append({
                        'episode': episode + 1,
                        'seed': lane_seeds[lane],
                        'drops': int(after.step_count[lane].item()),
                        'score': int(after.score[lane].item()),
                        'end_kind': (
                            'failed'
                            if bool(result.physics.done[lane].item())
                            else 'drop_limit'
                        ),
                    })
                    completed += 1
                for key in [key for key in pending if key[0] == lane]:
                    pending.pop(key, None)
                scene_history[lane].clear()
                if next_episode < args.episodes:
                    lane_episodes[lane] = next_episode
                    lane_seeds[lane] = _new_episode_seed(
                        args.seed_base, next_episode
                    )
                    next_episode += 1
                else:
                    lane_episodes[lane] = -1
                    lane_seeds[lane] = _new_episode_seed(
                        args.seed_base + 9_000_000_000, lane
                    )
                reset_mask[lane] = True
                reset_seeds.append(lane_seeds[lane])
            simulator.reset(
                reset_mask,
                seeds=torch.tensor(reset_seeds, dtype=torch.int64),
            )
            tracker.reset(reset_mask)
            reset_observation = simulator.observe()
            for lane in finished_lanes:
                scene_history[lane].append(_scene_snapshot(
                    reset_observation,
                    lane,
                    episode=lane_episodes[lane],
                    seed=lane_seeds[lane],
                ))

        now = time.perf_counter()
        if now - last_progress >= args.progress_seconds:
            print(json.dumps({
                'phase': 'collecting',
                'completed_episodes': completed,
                'target_episodes': args.episodes,
                'transitions': transitions,
                'transitions_per_second': (
                    transitions / max(now - started_at, 1e-9)
                ),
                'candidate_starts': candidate_starts,
                'confirmed_events': confirmed_total,
                'rendered_events': len(image_paths),
            }, ensure_ascii=False), flush=True)
            last_progress = now

    gallery_path = render_gallery(
        image_paths, output_dir / 'pair_failure_gallery.png'
    )
    elapsed = time.perf_counter() - started_at
    report = {
        'purpose': 'pair_failure_threshold_validation',
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'device': 'cpu',
        'physics_fps': config.physics_fps,
        'num_envs': args.num_envs,
        'episodes': args.episodes,
        'max_drops': args.max_drops,
        'detector_config': {
            'motion_window_drops': detector_config.motion_window_drops,
            'confirmation_drops': detector_config.confirmation_drops,
            'max_net_displacement_ratio': (
                detector_config.max_net_displacement_ratio
            ),
            'adjacent_surface_gap_ratio': (
                detector_config.adjacent_surface_gap_ratio
            ),
        },
        'elapsed_seconds': elapsed,
        'transitions': transitions,
        'transitions_per_second': transitions / max(elapsed, 1e-9),
        'detector_seconds': detector_seconds,
        'detector_wall_fraction': detector_seconds / max(elapsed, 1e-9),
        'candidate_starts': candidate_starts,
        'candidate_cancellations_before_confirmation': (
            candidate_cancellations
        ),
        'confirmed_events': confirmed_total,
        'rendered_events': len(image_paths),
        'gallery': None if gallery_path is None else gallery_path.name,
        'episode_results': sorted(
            episode_rows, key=lambda row: row['episode']
        ),
        'events': event_rows,
    }
    report_path = output_dir / 'index.json'
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps({
        'output_dir': str(output_dir),
        'gallery': None if gallery_path is None else str(gallery_path),
        'report': str(report_path),
        'confirmed_events': confirmed_total,
        'rendered_events': len(image_paths),
    }, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
