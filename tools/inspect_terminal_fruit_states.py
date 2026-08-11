"""并行运行若干局 greedy 游戏，并绘制每局最后一次投放后的水果分布。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
import torch  # noqa: E402

from daxigua.core import fruit_name, fruit_radius  # noqa: E402
from daxigua.rl.observations import TensorState  # noqa: E402
from daxigua.rl.viewer import (  # noqa: E402
    load_viewer_model,
    viewer_simulator_config,
)
from daxigua.simulator import TensorVectorSimulator  # noqa: E402


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / 'runs'
    / 'cloud_rtx5090_auxiliary_action_structured_128m_to_120fps_seed20260812_16m'
    / 'checkpoints'
    / 'final.pt'
)
SEED_STRIDE = 1_000_003
COLORS = (
    '#7d3c98', '#d35400', '#e67e22', '#f4d03f', '#73c66b', '#d94b45',
    '#f1948a', '#d4a017', '#8d6e63', '#239b56', '#117864',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='并行执行 K 局迁移模型游戏并绘制各局终局水果分布。'
    )
    parser.add_argument('episodes', type=int, help='游戏局数 K（同时并行执行）')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--physics-fps', type=int, choices=(30, 120), default=120)
    parser.add_argument('--seed', type=int, default=42_000_000)
    parser.add_argument('--max-drops', type=int, default=1000)
    parser.add_argument(
        '--output-dir', type=Path,
        help='输出目录；省略时创建系统临时目录。',
    )
    return parser.parse_args()


@torch.inference_mode()
def run_parallel(loaded, *, episodes, physics_fps, seed_base, max_drops):
    config = viewer_simulator_config(
        physics_fps, loaded.model_config, loaded.device
    )
    simulator = TensorVectorSimulator(episodes, config=config, device=loaded.device)
    seeds = (
        torch.arange(episodes, dtype=torch.int64)
        .mul_(SEED_STRIDE)
        .add_(seed_base)
        .to(loaded.device)
    )
    simulator.reset(seeds=seeds)

    # 首次推理同时完成 CUDA/模型的惰性初始化。
    loaded.model(TensorState.from_observation(
        simulator.observe(), physics_fps=config.physics_fps
    ))
    active = torch.ones(episodes, dtype=torch.bool, device=loaded.device)
    snapshots = [None] * episodes
    end_kinds = [None] * episodes

    while bool(active.any().item()):
        observation = simulator.observe()
        state = TensorState.from_observation(
            observation, physics_fps=config.physics_fps
        )
        actions = loaded.model(state).argmax(dim=1)
        result = simulator.step_masked(actions, active)
        observation = result.observation
        capped = observation.step_count >= int(max_drops)
        finished = active & (result.physics.done | result.physics.truncated | capped)
        for row in torch.nonzero(finished, as_tuple=False).flatten().tolist():
            snapshots[row] = {
                'positions': observation.positions[row].detach().cpu(),
                'levels': observation.levels[row].detach().cpu(),
                'active': observation.active[row].detach().cpu(),
                'score': int(observation.score[row].item()),
                'drops': int(observation.step_count[row].item()),
                'max_level': int(observation.max_level[row].item()),
                'fruit_count': int(observation.fruit_count[row].item()),
                'seed': int(seeds[row].item()),
            }
            if bool(result.physics.done[row].item()):
                end_kinds[row] = 'failed'
            elif bool(result.physics.truncated[row].item()):
                end_kinds[row] = 'truncated'
            else:
                end_kinds[row] = 'drop_limit'
        active &= ~finished

    return snapshots, end_kinds, config


def render(snapshots, end_kinds, config, output_path):
    columns = min(4, len(snapshots))
    rows = math.ceil(len(snapshots) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 7.0 * rows), squeeze=False
    )
    for index, axis in enumerate(axes.flat):
        if index >= len(snapshots):
            axis.set_visible(False)
            continue
        snapshot = snapshots[index]
        axis.set_facecolor('#f7f1e5')
        axis.add_patch(Rectangle(
            (config.wall_width, 0),
            config.board_width - 2 * config.wall_width,
            config.board_height - config.wall_width,
            fill=False, edgecolor='#4b3b2a', linewidth=2.0,
        ))
        axis.axhline(
            config.spawn_y, color='#c0392b', linewidth=1.0, linestyle='--',
            alpha=0.75,
        )
        for position, level in zip(
                snapshot['positions'][snapshot['active']],
                snapshot['levels'][snapshot['active']]):
            level = int(level.item())
            x, y = (float(value) for value in position.tolist())
            radius = float(fruit_radius(level))
            axis.add_patch(Circle(
                (x, y), radius, facecolor=COLORS[level - 1],
                edgecolor='white', linewidth=1.0, alpha=0.9,
            ))
            axis.text(
                x, y, str(level), ha='center', va='center', fontsize=7,
                color='white', weight='bold',
            )
        axis.set_xlim(0, config.board_width)
        axis.set_ylim(config.board_height, 0)
        axis.set_aspect('equal')
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"#{index + 1}  seed={snapshot['seed']}\n"
            f"{end_kinds[index]} · drops={snapshot['drops']} · "
            f"score={snapshot['score']} · fruits={snapshot['fruit_count']}",
            fontsize=10,
        )
    figure.suptitle(
        f'Terminal fruit states ({config.physics_fps} FPS)', fontsize=14
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def report_rows(snapshots, end_kinds):
    rows = []
    for index, (snapshot, end_kind) in enumerate(zip(snapshots, end_kinds), 1):
        counts = {}
        for level in snapshot['levels'][snapshot['active']].tolist():
            key = f'L{level}_{fruit_name(level)}'
            counts[key] = counts.get(key, 0) + 1
        rows.append({
            'episode': index,
            'seed': snapshot['seed'],
            'end_kind': end_kind,
            'drops': snapshot['drops'],
            'score': snapshot['score'],
            'max_level': snapshot['max_level'],
            'fruit_count': snapshot['fruit_count'],
            'level_counts': counts,
        })
    return rows


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError('episodes must be positive')
    if args.max_drops <= 0:
        raise ValueError('max-drops must be positive')
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(tempfile.mkdtemp(prefix='daxigua-terminal-states-')).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_viewer_model(args.checkpoint, device=args.device)
    snapshots, end_kinds, config = run_parallel(
        loaded,
        episodes=args.episodes,
        physics_fps=args.physics_fps,
        seed_base=args.seed,
        max_drops=args.max_drops,
    )
    image_path = output_dir / 'terminal_fruit_states.png'
    report_path = output_dir / 'terminal_fruit_states.json'
    render(snapshots, end_kinds, config, image_path)
    payload = {
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'device': str(loaded.device),
        'physics_fps': config.physics_fps,
        'max_drops': args.max_drops,
        'image': str(image_path),
        'episodes': report_rows(snapshots, end_kinds),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps({
        'image': str(image_path),
        'report': str(report_path),
        'episodes': len(snapshots),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
