"""用 SAB-128 CPU 对局生成多场景单水果可合成性静态画廊。"""

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
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
import torch  # noqa: E402

from daxigua.core import fruit_name  # noqa: E402
from daxigua.rl.mergeability import (  # noqa: E402
    MERGEABILITY_SOURCE_EXTERNAL,
    MERGEABILITY_SOURCE_INTERNAL,
    MergeabilityCalculator,
    MergeabilityConfig,
)
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
SEED_STRIDE = 1_000_003
SOURCE_NAMES = {
    0: 'none',
    MERGEABILITY_SOURCE_INTERNAL: 'internal',
    MERGEABILITY_SOURCE_EXTERNAL: 'external',
}
SOURCE_COLORS = {
    0: '#64748b',
    MERGEABILITY_SOURCE_INTERNAL: '#2563eb',
    MERGEABILITY_SOURCE_EXTERNAL: '#7c3aed',
}


def _parse_drop_points(value):
    points = sorted({int(item.strip()) for item in value.split(',') if item.strip()})
    if not points or points[0] <= 0:
        raise argparse.ArgumentTypeError('drop points must be positive integers')
    return tuple(points)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='运行少量 SAB-128 CPU 对局并生成可合成性多场景画廊。'
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--seed-base', type=int, default=202_608_250)
    parser.add_argument(
        '--drop-points', type=_parse_drop_points, default=(60, 120, 180)
    )
    parser.add_argument('--output-dir', type=Path)
    return parser.parse_args(argv)


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _snapshot(state, result, *, seed, drops, score, end_kind):
    active_slots = torch.nonzero(
        state.active[0], as_tuple=False
    ).flatten().tolist()
    fruits = []
    for slot in active_slots:
        level = int(state.levels[0, slot].item())
        source = int(result.source[0, slot].item())
        fruits.append({
            'slot': slot,
            'level': level,
            'name': fruit_name(level),
            'x': float(state.positions[0, slot, 0].item()),
            'y': float(state.positions[0, slot, 1].item()),
            'radius': float(state.physics_radii[0, slot].item()),
            'score': float(result.score[0, slot].item()),
            'difficulty': _finite_number(result.difficulty[0, slot].item()),
            'internal_score': float(result.internal_score[0, slot].item()),
            'external_score': float(result.external_score[0, slot].item()),
            'source': SOURCE_NAMES[source],
            'source_code': source,
            'dependency': int(
                result.primary_dependency_slot[0, slot].item()
            ),
            'capacity_radius': float(
                result.external_capacity_radius[0, slot].item()
            ),
            'capacity_level': int(
                result.external_capacity_level[0, slot].item()
            ),
            'entry_level': int(
                result.external_entry_level[0, slot].item()
            ),
        })
    return {
        'seed': int(seed),
        'drops': int(drops),
        'game_score': int(score),
        'end_kind': end_kind,
        'fruits': fruits,
    }


@torch.inference_mode()
def collect_scenes(args, loaded, config, calculator):
    simulator = TensorVectorSimulator(1, config=config, device=loaded.device)
    scenes = []
    for seed_index in range(args.seeds):
        seed = args.seed_base + seed_index * SEED_STRIDE
        simulator.reset(seeds=seed)
        point_index = 0
        max_drops = args.drop_points[-1]
        for drops in range(1, max_drops + 1):
            observation = simulator.observe()
            state = TensorState.from_observation(
                observation, physics_fps=config.physics_fps
            )
            action = loaded.model(state).argmax(dim=1)
            step = simulator.step(action)
            after = step.observation
            finished = bool(
                (step.physics.done | step.physics.truncated)[0].item()
            )
            requested = (
                point_index < len(args.drop_points)
                and drops == args.drop_points[point_index]
            )
            if requested or finished:
                state = TensorState.from_observation(
                    after, physics_fps=config.physics_fps
                )
                result = calculator(state)
                scenes.append(_snapshot(
                    state,
                    result,
                    seed=seed,
                    drops=drops,
                    score=after.score[0].item(),
                    end_kind=(
                        'terminal' if finished else 'requested_drop'
                    ),
                ))
                if requested:
                    point_index += 1
            if finished:
                break
        print(json.dumps({
            'phase': 'collect',
            'seed_index': seed_index + 1,
            'seeds': args.seeds,
            'seed': seed,
            'captured': len(scenes),
        }, ensure_ascii=False), flush=True)
    return scenes


def render_scene(scene, config, output_path):
    figure, (axis, notes) = plt.subplots(
        1, 2, figsize=(11.0, 8.2), gridspec_kw={'width_ratios': (1.0, 1.0)}
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
        config.spawn_y,
        color='#c0392b',
        linewidth=1.1,
        linestyle='--',
        alpha=0.8,
    )
    fruit_by_slot = {fruit['slot']: fruit for fruit in scene['fruits']}
    color_map = plt.get_cmap('RdYlGn')

    for fruit in scene['fruits']:
        if fruit['source_code'] != MERGEABILITY_SOURCE_INTERNAL:
            continue
        dependency = fruit_by_slot.get(fruit['dependency'])
        if dependency is None:
            continue
        axis.annotate(
            '',
            xy=(dependency['x'], dependency['y']),
            xytext=(fruit['x'], fruit['y']),
            arrowprops={
                'arrowstyle': '->',
                'color': SOURCE_COLORS[MERGEABILITY_SOURCE_INTERNAL],
                'linewidth': 1.6,
                'linestyle': '--',
                'alpha': 0.8,
            },
            zorder=2,
        )

    for fruit in scene['fruits']:
        fill = color_map(max(0.0, min(1.0, fruit['score'])))
        edge = SOURCE_COLORS[fruit['source_code']]
        axis.add_patch(Circle(
            (fruit['x'], fruit['y']),
            fruit['radius'],
            facecolor=fill,
            edgecolor=edge,
            linewidth=2.4,
            alpha=0.90,
            zorder=3,
        ))
        axis.text(
            fruit['x'],
            fruit['y'],
            f"L{fruit['level']}\n{fruit['score']:.2f}",
            ha='center',
            va='center',
            fontsize=7.0,
            color='#172033',
            weight='bold',
            zorder=4,
        )

    axis.set_xlim(0, config.board_width)
    axis.set_ylim(config.board_height, 0)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(
        f"Mergeability V1 · seed={scene['seed']} · drop={scene['drops']}\n"
        f"game score={scene['game_score']} · fruits={len(scene['fruits'])}",
        fontsize=11,
    )
    color_bar = figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=color_map),
        ax=axis,
        orientation='horizontal',
        fraction=0.035,
        pad=0.02,
    )
    color_bar.set_label('raw mergeability: low → high', fontsize=8)

    notes.axis('off')
    header = (
        'slot  L   M     Int   Ext   source    dep  cap/entry\n'
        '---- --- ----- ----- ----- --------- ---- ---------'
    )
    ordered = sorted(
        scene['fruits'], key=lambda item: (-item['level'], item['score'])
    )
    rows = []
    for fruit in ordered:
        dependency = (
            str(fruit['dependency']) if fruit['dependency'] >= 0 else '-'
        )
        rows.append(
            f"{fruit['slot']:>4} {fruit['level']:>3} "
            f"{fruit['score']:>5.2f} {fruit['internal_score']:>5.2f} "
            f"{fruit['external_score']:>5.2f} "
            f"{fruit['source']:>9} {dependency:>4} "
            f"L{fruit['capacity_level']}/L{fruit['entry_level']}"
        )
    notes.text(
        0.0, 1.0,
        header + '\n' + '\n'.join(rows),
        ha='left',
        va='top',
        fontsize=7.2,
        family='monospace',
        linespacing=1.25,
    )
    notes.text(
        0.0, 0.01,
        'fill: mergeability  |  blue edge/arrow: internal path\n'
        'purple edge: external path  |  cap/entry: corridor / usable input level\n'
        'V1 uses straight vertical probes; curved rolling paths are not modeled.',
        ha='left',
        va='bottom',
        fontsize=8.0,
        color='#475569',
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches='tight')
    plt.close(figure)


def render_overview(image_paths, output_path):
    columns = 2
    rows = math.ceil(len(image_paths) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 6.2 * rows))
    axes = list(getattr(axes, 'flat', (axes,)))
    for axis, image_path in zip(axes, image_paths):
        axis.imshow(plt.imread(image_path))
        axis.axis('off')
    for axis in axes[len(image_paths):]:
        axis.axis('off')
    figure.suptitle(
        'Mergeability V1 · SAB-128 real-scene gallery',
        fontsize=16,
        weight='bold',
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(figure)


@torch.inference_mode()
def run(args):
    if args.seeds <= 0:
        raise ValueError('seeds must be positive')
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(tempfile.mkdtemp(prefix='daxigua-mergeability-v1-')).resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_viewer_model(args.checkpoint, device=args.device)
    config = viewer_simulator_config(30, loaded.model_config, loaded.device)
    calculator = MergeabilityCalculator(
        MergeabilityConfig.from_simulator_config(config)
    ).to(loaded.device)
    scenes = collect_scenes(args, loaded, config, calculator)
    image_paths = []
    for index, scene in enumerate(scenes, start=1):
        image_path = output_dir / (
            f"scene_{index:02d}_seed_{scene['seed']}_drop_{scene['drops']:04d}.png"
        )
        render_scene(scene, config, image_path)
        image_paths.append(image_path)
    overview_path = output_dir / 'overview.png'
    render_overview(image_paths, overview_path)
    report = {
        'purpose': 'mergeability_v1_real_scene_gallery',
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'device': str(loaded.device),
        'physics_fps': config.physics_fps,
        'algorithm': {
            'neighborhood_scale': calculator.config.neighborhood_scale,
            'top_k': calculator.config.top_k,
            'score_cost_scale': calculator.config.score_cost_scale,
            'probe_offsets': calculator.config.probe_offsets,
            'external_supply': 'vertical_corridor_v1',
        },
        'overview': overview_path.name,
        'scenes': scenes,
    }
    report_path = output_dir / 'report.json'
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps({
        'output_dir': str(output_dir),
        'overview': str(overview_path),
        'report': str(report_path),
        'scene_count': len(scenes),
    }, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
