"""运行单局 CPU 对局，并可视化 Fruit Graph 边上的配对最低结果圆。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
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

from daxigua.rl.observations import TensorState  # noqa: E402
from daxigua.rl.viewer import (  # noqa: E402
    load_viewer_model,
    viewer_simulator_config,
)
from daxigua.simulator import TensorVectorSimulator  # noqa: E402


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / 'runs'
    / 'cloud_rtx5090_auxiliary_action_structured_branch_seed20260811_128m'
    / 'checkpoints'
    / 'final.pt'
)
FRUIT_COLORS = (
    '#7d3c98', '#d35400', '#e67e22', '#f4d03f', '#73c66b', '#d94b45',
    '#f1948a', '#d4a017', '#8d6e63', '#239b56', '#117864',
)
EDGE_COLORS = (
    '#e63946', '#2a9d8f', '#4361ee', '#f77f00', '#8338ec', '#0081a7',
    '#ef476f', '#6a994e', '#9c6644', '#3a0ca3', '#ff006e', '#0077b6',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='CPU 运行 SAB-128 并绘制随机 Fruit Graph 配对结果圆。'
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--drops', type=int, default=100)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260823)
    parser.add_argument('--pair-seed', type=int, default=823)
    parser.add_argument('--output', type=Path)
    return parser.parse_args(argv)


def _selected_pairs(model, state, count, seed):
    neighbor_indices, edge_features, edge_mask = model._physical_graph(state)
    active = state.active[0]
    candidates = {}
    for source in torch.nonzero(active, as_tuple=False).flatten().tolist():
        for edge_slot in torch.nonzero(
                edge_mask[0, source], as_tuple=False).flatten().tolist():
            target = int(neighbor_indices[0, source, edge_slot].item())
            if not bool(active[target].item()) or source == target:
                continue
            pair = tuple(sorted((source, target)))
            candidates.setdefault(
                pair, edge_features[0, source, edge_slot].detach().cpu()
            )
    pairs = sorted(candidates)
    selected = random.Random(seed).sample(pairs, min(count, len(pairs)))
    return [(pair, candidates[pair]) for pair in selected], len(pairs)


def _render(state, observation, config, selected, candidate_count, output):
    positions = state.positions[0].detach().cpu()
    radii = state.physics_radii[0].detach().cpu()
    levels = state.levels[0].detach().cpu()
    active = state.active[0].detach().cpu()

    figure, (axis, notes) = plt.subplots(
        1, 2, figsize=(11.5, 8.2), gridspec_kw={'width_ratios': (1.0, 0.9)}
    )
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
        config.spawn_y, color='#c0392b', linewidth=1.1,
        linestyle='--', alpha=0.8,
    )
    for slot in torch.nonzero(active, as_tuple=False).flatten().tolist():
        level = int(levels[slot].item())
        x, y = (float(value) for value in positions[slot].tolist())
        radius = float(radii[slot].item())
        axis.add_patch(Circle(
            (x, y), radius,
            facecolor=FRUIT_COLORS[level - 1],
            edgecolor='#342a20', linewidth=0.8, alpha=0.82,
        ))
        axis.text(
            x, y, f'{slot}\nL{level}', ha='center', va='center',
            fontsize=6.5, color='white', weight='bold',
        )

    descriptions = []
    for index, ((left, right), edge) in enumerate(selected, start=1):
        color = EDGE_COLORS[(index - 1) % len(EDGE_COLORS)]
        left_position = positions[left]
        right_position = positions[right]
        center_x = (float(edge[14].item()) + 1.0) * 0.5 * config.board_width
        center_y = (float(edge[15].item()) + 1.0) * 0.5 * config.board_height
        result_radius = float(edge[16].item()) * config.board_width
        result_exists = bool(edge[17].item() >= 0.5)
        axis.plot(
            [float(left_position[0]), float(right_position[0])],
            [float(left_position[1]), float(right_position[1])],
            color=color, linewidth=1.4, alpha=0.9,
        )
        if result_exists:
            axis.add_patch(Circle(
                (center_x, center_y), result_radius,
                fill=False, edgecolor=color, linewidth=2.0,
                linestyle='--', alpha=0.95,
            ))
        axis.scatter(
            [center_x], [center_y], s=42, color=color,
            edgecolors='white', linewidths=0.8, zorder=10,
        )
        axis.text(
            center_x, center_y - 8.0, str(index), color=color,
            fontsize=9, weight='bold', ha='center', va='bottom', zorder=11,
        )
        left_level = int(levels[left].item())
        right_level = int(levels[right].item())
        result_text = (
            f'L{max(left_level, right_level) + 1}, r={result_radius:.1f}'
            if result_exists else 'vanish (no result circle)'
        )
        descriptions.append(
            f'{index}. slot {left} L{left_level} + slot {right} L{right_level}\n'
            f'   center=({center_x:.1f}, {center_y:.1f}) -> {result_text}'
        )

    axis.set_xlim(0, config.board_width)
    axis.set_ylim(config.board_height, 0)
    axis.set_aspect('equal')
    axis.set_title(
        f'Pair minimum-result circles after '
        f'{int(observation.step_count[0].item())} drops\n'
        f'score={int(observation.score[0].item())}, '
        f'fruits={int(observation.fruit_count[0].item())}',
        fontsize=12,
    )
    axis.set_xlabel('x')
    axis.set_ylabel('y (positive downward)')

    notes.axis('off')
    notes.text(
        0.0, 1.0,
        f'{len(selected)} random pairs from {candidate_count} retained '
        f'undirected Fruit Graph edges\n\n' + '\n\n'.join(descriptions),
        ha='left', va='top', fontsize=9.5, linespacing=1.35,
        family='monospace',
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(figure)


@torch.inference_mode()
def run(args):
    if args.drops <= 0 or args.pairs <= 0:
        raise ValueError('drops and pairs must be positive')
    loaded = load_viewer_model(args.checkpoint, device=args.device)
    if loaded.device.type != 'cpu':
        raise ValueError('this local diagnostic script requires --device cpu')
    config = viewer_simulator_config(30, loaded.model_config, loaded.device)
    simulator = TensorVectorSimulator(1, config=config, device=loaded.device)
    simulator.reset(seeds=args.seed)

    actual_drops = 0
    for _ in range(args.drops):
        observation = simulator.observe()
        state = TensorState.from_observation(
            observation, physics_fps=config.physics_fps
        )
        action = loaded.model(state).argmax(dim=1)
        result = simulator.step(action)
        actual_drops += 1
        if bool((result.physics.done | result.physics.truncated)[0].item()):
            break

    final_observation = simulator.observe()
    state = TensorState.from_observation(
        final_observation, physics_fps=config.physics_fps
    )
    selected, candidate_count = _selected_pairs(
        loaded.model, state, args.pairs, args.pair_seed
    )
    if not selected:
        raise RuntimeError('the final scene has no retained fruit graph edges')
    output = (
        args.output.resolve()
        if args.output is not None
        else Path(tempfile.mkdtemp(prefix='daxigua-pair-result-')).resolve()
        / 'pair_result_circles.png'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _render(
        state, final_observation, config, selected, candidate_count, output
    )
    result = {
        'output': str(output),
        'requested_drops': args.drops,
        'actual_drops': actual_drops,
        'score': int(final_observation.score[0].item()),
        'fruit_count': int(final_observation.fruit_count[0].item()),
        'candidate_edges': candidate_count,
        'selected_edges': len(selected),
        'terminated_early': actual_drops < args.drops,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
