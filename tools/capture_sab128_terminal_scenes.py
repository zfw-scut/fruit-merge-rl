"""用 SAB-128 并行运行对局，并把每局结束场景保存为独立图片。"""

from __future__ import annotations

import argparse
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

from daxigua.core import fruit_name  # noqa: E402
from daxigua.rl.observations import TensorState  # noqa: E402
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
        description='运行 SAB-128 对局并逐局保存终局静态图片。'
    )
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        '--device',
        default='cuda',
        help='cuda 使用批量并行；cpu 使用逐局诊断路径，建议减少 episodes。',
    )
    parser.add_argument('--seed-base', type=int, default=92_000_000)
    parser.add_argument(
        '--max-drops',
        type=int,
        default=5000,
        help='诊断安全边界；达到边界的图片会明确标为 drop_limit。',
    )
    parser.add_argument('--progress-seconds', type=float, default=10.0)
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='省略时自动创建系统临时目录。',
    )
    return parser.parse_args(argv)


def _snapshot(observation, row, seed, end_kind):
    return {
        'positions': observation.positions[row].detach().cpu(),
        'radii': observation.physics_radii[row].detach().cpu(),
        'levels': observation.levels[row].detach().cpu(),
        'active': observation.active[row].detach().cpu(),
        'seed': int(seed),
        'end_kind': end_kind,
        'drops': int(observation.step_count[row].item()),
        'score': int(observation.score[row].item()),
        'max_level': int(observation.max_level[row].item()),
        'fruit_count': int(observation.fruit_count[row].item()),
    }


def render_terminal_scene(snapshot, config, output_path):
    figure, axis = plt.subplots(figsize=(4.2, 7.2))
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
        linewidth=1.1,
        linestyle='--',
        alpha=0.8,
    )
    active_slots = torch.nonzero(
        snapshot['active'], as_tuple=False
    ).flatten().tolist()
    for slot in active_slots:
        level = int(snapshot['levels'][slot].item())
        x, y = (
            float(value) for value in snapshot['positions'][slot].tolist()
        )
        radius = float(snapshot['radii'][slot].item())
        axis.add_patch(Circle(
            (x, y),
            radius,
            facecolor=COLORS[level - 1],
            edgecolor='white',
            linewidth=1.2,
            alpha=0.92,
        ))
        axis.text(
            x,
            y,
            f'L{level}',
            ha='center',
            va='center',
            fontsize=8,
            color='white',
            weight='bold',
        )
    axis.set_xlim(0, config.board_width)
    axis.set_ylim(config.board_height, 0)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(
        f"SAB-128 · {snapshot['end_kind']}\n"
        f"seed={snapshot['seed']} · drops={snapshot['drops']} · "
        f"score={snapshot['score']} · fruits={snapshot['fruit_count']}",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)


def _report_row(index, snapshot, image_path):
    level_counts = {}
    for level in snapshot['levels'][snapshot['active']].tolist():
        key = f'L{level}_{fruit_name(level)}'
        level_counts[key] = level_counts.get(key, 0) + 1
    return {
        'episode': index + 1,
        'seed': snapshot['seed'],
        'end_kind': snapshot['end_kind'],
        'drops': snapshot['drops'],
        'score': snapshot['score'],
        'max_level': snapshot['max_level'],
        'fruit_count': snapshot['fruit_count'],
        'level_counts': level_counts,
        'image': image_path.name,
    }


def _end_kind(result, capped, row=0):
    if bool(result.physics.done[row].item()):
        return 'failed'
    if bool(result.physics.truncated[row].item()):
        return 'truncated'
    if bool(capped[row].item()):
        return 'drop_limit'
    raise RuntimeError('requested end kind for an unfinished episode')


def _collect_cuda(args, loaded, config, seeds):
    simulator = TensorVectorSimulator(
        args.episodes, config=config, device=loaded.device
    )
    simulator.reset(seeds=seeds)
    active = torch.ones(
        args.episodes, dtype=torch.bool, device=loaded.device
    )
    snapshots = [None] * args.episodes
    transitions = 0
    started = time.perf_counter()
    last_progress = started

    while bool(active.any().item()):
        observation = simulator.observe()
        active_rows = torch.nonzero(active, as_tuple=False).flatten()
        state = TensorState.from_observation(
            observation,
            physics_fps=config.physics_fps,
            rows=active_rows,
        )
        selected_actions = loaded.model(state).argmax(dim=1)
        actions = torch.zeros(
            args.episodes, dtype=torch.int64, device=loaded.device
        )
        actions[active_rows] = selected_actions
        result = simulator.step_masked(actions, active)
        transitions += int(active_rows.numel())
        after = result.observation
        capped = after.step_count >= args.max_drops
        finished = active & (
            result.physics.done | result.physics.truncated | capped
        )
        finished_rows = torch.nonzero(
            finished, as_tuple=False
        ).flatten().detach().cpu().tolist()
        for row in finished_rows:
            snapshots[row] = _snapshot(
                after, row, seeds[row].item(), _end_kind(result, capped, row)
            )
        active &= ~finished

        now = time.perf_counter()
        if now - last_progress >= args.progress_seconds:
            elapsed = now - started
            completed = args.episodes - int(active.sum().item())
            print(json.dumps({
                'phase': 'playing',
                'device': 'cuda',
                'completed': completed,
                'episodes': args.episodes,
                'active': int(active.sum().item()),
                'transitions_per_second': transitions / max(elapsed, 1e-9),
                'longest_active_drops': (
                    int(after.step_count[active].max().item())
                    if bool(active.any().item()) else 0
                ),
            }, ensure_ascii=False), flush=True)
            last_progress = now
    return snapshots, time.perf_counter() - started


def _collect_cpu(args, loaded, config, seeds):
    """CPU 诊断路径逐局运行，避免依赖仅 CUDA 可用的掩码步进。"""

    simulator = TensorVectorSimulator(1, config=config, device=loaded.device)
    snapshots = [None] * args.episodes
    transitions = 0
    started = time.perf_counter()
    last_progress = started
    for index in range(args.episodes):
        seed = int(seeds[index].item())
        simulator.reset(seeds=seed)
        while True:
            observation = simulator.observe()
            state = TensorState.from_observation(
                observation, physics_fps=config.physics_fps
            )
            action = loaded.model(state).argmax(dim=1)
            result = simulator.step(action)
            transitions += 1
            after = result.observation
            capped = after.step_count >= args.max_drops
            finished = (
                result.physics.done | result.physics.truncated | capped
            )
            if bool(finished[0].item()):
                snapshots[index] = _snapshot(
                    after, 0, seed, _end_kind(result, capped)
                )
                break

            now = time.perf_counter()
            if now - last_progress >= args.progress_seconds:
                elapsed = now - started
                print(json.dumps({
                    'phase': 'playing',
                    'device': 'cpu',
                    'completed': index,
                    'episodes': args.episodes,
                    'active': 1,
                    'transitions_per_second': (
                        transitions / max(elapsed, 1e-9)
                    ),
                    'current_episode_drops': int(after.step_count[0].item()),
                }, ensure_ascii=False), flush=True)
                last_progress = now
    return snapshots, time.perf_counter() - started


@torch.inference_mode()
def run(args):
    if args.episodes <= 0:
        raise ValueError('episodes must be positive')
    if args.max_drops <= 0:
        raise ValueError('max-drops must be positive')
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(tempfile.mkdtemp(prefix='daxigua-sab128-terminal-')).resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / 'images'
    images_dir.mkdir()

    loaded = load_viewer_model(args.checkpoint, device=args.device)
    if loaded.checkpoint_sha256.lower() != SAB128_SHA256:
        raise ValueError('checkpoint does not match the registered SAB-128')
    config = viewer_simulator_config(
        30, loaded.model_config, loaded.device
    )
    seeds = (
        torch.arange(args.episodes, dtype=torch.int64)
        .mul_(SEED_STRIDE)
        .add_(args.seed_base)
        .to(loaded.device)
    )
    if loaded.device.type == 'cuda':
        snapshots, collection_seconds = _collect_cuda(
            args, loaded, config, seeds
        )
    else:
        snapshots, collection_seconds = _collect_cpu(
            args, loaded, config, seeds
        )
    report_rows = []
    for index, snapshot in enumerate(snapshots):
        image_name = (
            f"episode_{index + 1:04d}_score_{snapshot['score']:05d}_"
            f"drops_{snapshot['drops']:04d}_{snapshot['end_kind']}.png"
        )
        image_path = images_dir / image_name
        render_terminal_scene(snapshot, config, image_path)
        report_rows.append(_report_row(index, snapshot, image_path))
        if (index + 1) % 32 == 0 or index + 1 == args.episodes:
            print(json.dumps({
                'phase': 'rendering',
                'rendered': index + 1,
                'episodes': args.episodes,
            }, ensure_ascii=False), flush=True)

    report = {
        'purpose': 'sab128_terminal_scene_gallery',
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'physics_fps': config.physics_fps,
        'episodes': args.episodes,
        'max_drops': args.max_drops,
        'collection_seconds': collection_seconds,
        'natural_end_count': sum(
            row['end_kind'] == 'failed' for row in report_rows
        ),
        'truncated_count': sum(
            row['end_kind'] != 'failed' for row in report_rows
        ),
        'items': report_rows,
    }
    report_path = output_dir / 'index.json'
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps({
        'output_dir': str(output_dir),
        'images_dir': str(images_dir),
        'report': str(report_path),
        'natural_end_count': report['natural_end_count'],
        'truncated_count': report['truncated_count'],
    }, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
