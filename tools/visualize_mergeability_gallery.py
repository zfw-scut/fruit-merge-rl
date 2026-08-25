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
    resolve_viewer_device,
    viewer_simulator_config,
)
from daxigua.simulator import (  # noqa: E402
    SimulatorConfig,
    TensorVectorSimulator,
    load_static_scene_dataset,
)


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
    parser.add_argument('--overview-page-size', type=int, default=16)
    parser.add_argument(
        '--dataset',
        type=Path,
        help='读取可复用静态场景数据集；设置后不再运行模型采集。',
    )
    parser.add_argument('--gallery-count', type=int, default=256)
    parser.add_argument('--calculation-batch-size', type=int, default=512)
    parser.add_argument('--output-dir', type=Path)
    return parser.parse_args(argv)


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _snapshot(
        state,
        result,
        *,
        seed,
        drops,
        score,
        end_kind,
        policy_mode=None,
        row=0):
    active_slots = torch.nonzero(
        state.active[row], as_tuple=False
    ).flatten().tolist()
    fruits = []
    for slot in active_slots:
        level = int(state.levels[row, slot].item())
        source = int(result.source[row, slot].item())
        fruits.append({
            'slot': slot,
            'level': level,
            'name': fruit_name(level),
            'x': float(state.positions[row, slot, 0].item()),
            'y': float(state.positions[row, slot, 1].item()),
            'radius': float(state.physics_radii[row, slot].item()),
            'score': float(result.score[row, slot].item()),
            'difficulty': _finite_number(result.difficulty[row, slot].item()),
            'internal_score': float(result.internal_score[row, slot].item()),
            'external_score': float(result.external_score[row, slot].item()),
            'source': SOURCE_NAMES[source],
            'source_code': source,
            'dependency': int(
                result.primary_dependency_slot[row, slot].item()
            ),
            'capacity_radius': float(
                result.external_capacity_radius[row, slot].item()
            ),
            'capacity_level': int(
                result.external_capacity_level[row, slot].item()
            ),
            'entry_level': int(
                result.external_entry_level[row, slot].item()
            ),
        })
    return {
        'seed': int(seed),
        'drops': int(drops),
        'game_score': int(score),
        'end_kind': end_kind,
        'policy_mode': policy_mode,
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


def _evenly_spaced_rows(rows, count):
    count = min(int(count), int(rows.numel()))
    if count <= 0:
        return rows[:0]
    if count == int(rows.numel()):
        return rows
    positions = torch.linspace(0, rows.numel() - 1, count).round().to(torch.long)
    return rows.index_select(0, positions)


def _gallery_rows(dataset, count):
    count = min(int(count), dataset.batch_size)
    if count <= 0:
        raise ValueError('gallery-count must be positive')
    modes = dataset.metadata.get('policy_mode')
    if modes is None or not bool((modes == 0).any().item()) \
            or not bool((modes == 1).any().item()):
        return _evenly_spaced_rows(torch.arange(dataset.batch_size), count)
    policy_rows = torch.nonzero(modes == 1, as_tuple=False).flatten().cpu()
    random_rows = torch.nonzero(modes == 0, as_tuple=False).flatten().cpu()
    policy_count = min((count + 1) // 2, int(policy_rows.numel()))
    random_count = min(count - policy_count, int(random_rows.numel()))
    missing = count - policy_count - random_count
    if missing > 0:
        policy_extra = min(missing, int(policy_rows.numel()) - policy_count)
        policy_count += policy_extra
        random_count += missing - policy_extra
    selected = torch.cat((
        _evenly_spaced_rows(policy_rows, policy_count),
        _evenly_spaced_rows(random_rows, random_count),
    ))
    return selected.sort().values


@torch.inference_mode()
def collect_dataset_scenes(args, output_dir):
    if args.calculation_batch_size <= 0:
        raise ValueError('calculation-batch-size must be positive')
    device = resolve_viewer_device(args.device)
    dataset = load_static_scene_dataset(args.dataset, device=device)
    simulator_values = dataset.manifest.get('simulator_config')
    if not isinstance(simulator_values, dict):
        raise ValueError('static scene dataset misses simulator_config')
    config = SimulatorConfig(**simulator_values)
    calculator = MergeabilityCalculator(
        MergeabilityConfig.from_simulator_config(config)
    ).to(device)
    state = TensorState.from_observation(
        dataset.observation, physics_fps=config.physics_fps
    )
    result_chunks = []
    for start in range(0, state.batch_size, args.calculation_batch_size):
        stop = min(start + args.calculation_batch_size, state.batch_size)
        rows = torch.arange(start, stop, device=device)
        result_chunks.append(calculator(state.index_select(rows)))
        print(json.dumps({
            'phase': 'calculate_dataset',
            'calculated': stop,
            'scenes': state.batch_size,
        }, ensure_ascii=False), flush=True)
    result_type = type(result_chunks[0])
    result = result_type(*(
        torch.cat(tuple(getattr(chunk, name) for chunk in result_chunks), dim=0)
        for name in result_type._fields
    ))
    result_path = output_dir / 'mergeability_v1.pt'
    torch.save({
        'format': 'daxigua_mergeability_result',
        'format_version': 1,
        'source_dataset': str(Path(args.dataset).resolve()),
        'algorithm': {
            'neighborhood_scale': calculator.config.neighborhood_scale,
            'top_k': calculator.config.top_k,
            'score_cost_scale': calculator.config.score_cost_scale,
            'probe_offsets': calculator.config.probe_offsets,
            'external_supply': 'vertical_corridor_v1',
        },
        'result': {
            name: getattr(result, name).detach().cpu()
            for name in result_type._fields
        },
    }, result_path)

    gallery_rows = _gallery_rows(dataset, args.gallery_count)
    device_rows = gallery_rows.to(device)
    selected_state = state.index_select(device_rows).to('cpu')
    selected_result = result_type(*(
        getattr(result, name).index_select(0, device_rows).detach().cpu()
        for name in result_type._fields
    ))
    observation = dataset.observation
    metadata = dataset.metadata
    scenes = []
    for selected_row, dataset_row in enumerate(gallery_rows.tolist()):
        policy_value = metadata.get('policy_mode')
        policy_mode = (
            'SAB-128 greedy'
            if policy_value is not None
            and int(policy_value[dataset_row].item()) == 1
            else 'uniform random'
        )
        seed_value = metadata.get('seed')
        scenes.append(_snapshot(
            selected_state,
            selected_result,
            seed=(
                int(seed_value[dataset_row].item())
                if seed_value is not None else dataset_row
            ),
            drops=int(observation.step_count[dataset_row].item()),
            score=int(observation.score[dataset_row].item()),
            end_kind=(
                'terminal'
                if bool(observation.done[dataset_row].item())
                else 'requested_drop'
            ),
            policy_mode=policy_mode,
            row=selected_row,
        ))
    source = result.source
    active = state.active
    analysis = {
        'all_scene_count': dataset.batch_size,
        'gallery_scene_count': len(scenes),
        'active_fruits': int(active.sum().item()),
        'source_none': int(((source == 0) & active).sum().item()),
        'source_internal': int(
            ((source == MERGEABILITY_SOURCE_INTERNAL) & active).sum().item()
        ),
        'source_external': int(
            ((source == MERGEABILITY_SOURCE_EXTERNAL) & active).sum().item()
        ),
        'score_mean': float(result.score[active].mean().item()),
    }
    return scenes, config, calculator, dataset, result_path, analysis


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
        f"Mergeability V1 · {scene.get('policy_mode') or 'SAB-128 greedy'} · "
        f"seed={scene['seed']} · drop={scene['drops']}\n"
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


def render_overviews(image_paths, output_dir, *, page_size):
    if page_size <= 0:
        raise ValueError('overview page size must be positive')
    columns = min(4, page_size)
    outputs = []
    for page_start in range(0, len(image_paths), page_size):
        page_paths = image_paths[page_start:page_start + page_size]
        rows = math.ceil(len(page_paths) / columns)
        figure, axes = plt.subplots(
            rows, columns, figsize=(5.2 * columns, 4.1 * rows)
        )
        axes = list(getattr(axes, 'flat', (axes,)))
        for axis, image_path in zip(axes, page_paths):
            axis.imshow(plt.imread(image_path))
            axis.axis('off')
        for axis in axes[len(page_paths):]:
            axis.axis('off')
        page_number = len(outputs) + 1
        figure.suptitle(
            'Mergeability V1 · SAB-128 real-scene gallery · '
            f'page {page_number}',
            fontsize=16,
            weight='bold',
        )
        figure.tight_layout()
        output_path = output_dir / f'overview_{page_number:02d}.png'
        figure.savefig(output_path, dpi=105, bbox_inches='tight')
        plt.close(figure)
        outputs.append(output_path)
    return outputs


@torch.inference_mode()
def run(args):
    if args.dataset is None and args.seeds <= 0:
        raise ValueError('seeds must be positive')
    if args.overview_page_size <= 0:
        raise ValueError('overview-page-size must be positive')
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(tempfile.mkdtemp(prefix='daxigua-mergeability-v1-')).resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = None
    result_path = None
    dataset_analysis = None
    if args.dataset is not None:
        (
            scenes,
            config,
            calculator,
            dataset,
            result_path,
            dataset_analysis,
        ) = collect_dataset_scenes(args, output_dir)
        checkpoint_path = dataset.manifest.get('checkpoint')
        checkpoint_sha256 = dataset.manifest.get('checkpoint_sha256')
        report_device = str(resolve_viewer_device(args.device))
    else:
        loaded = load_viewer_model(args.checkpoint, device=args.device)
        config = viewer_simulator_config(30, loaded.model_config, loaded.device)
        calculator = MergeabilityCalculator(
            MergeabilityConfig.from_simulator_config(config)
        ).to(loaded.device)
        scenes = collect_scenes(args, loaded, config, calculator)
        checkpoint_path = str(loaded.checkpoint_path)
        checkpoint_sha256 = loaded.checkpoint_sha256
        report_device = str(loaded.device)
    image_paths = []
    for index, scene in enumerate(scenes, start=1):
        image_path = output_dir / (
            f"scene_{index:02d}_seed_{scene['seed']}_drop_{scene['drops']:04d}.png"
        )
        render_scene(scene, config, image_path)
        image_paths.append(image_path)
        if index % 16 == 0 or index == len(scenes):
            print(json.dumps({
                'phase': 'render',
                'rendered': index,
                'scenes': len(scenes),
            }, ensure_ascii=False), flush=True)
    overview_paths = render_overviews(
        image_paths, output_dir, page_size=args.overview_page_size
    )
    report = {
        'purpose': 'mergeability_v1_real_scene_gallery',
        'checkpoint': checkpoint_path,
        'checkpoint_sha256': checkpoint_sha256,
        'device': report_device,
        'physics_fps': config.physics_fps,
        'source_dataset': (
            str(Path(args.dataset).resolve()) if args.dataset is not None else None
        ),
        'full_result': result_path.name if result_path is not None else None,
        'dataset_analysis': dataset_analysis,
        'algorithm': {
            'neighborhood_scale': calculator.config.neighborhood_scale,
            'top_k': calculator.config.top_k,
            'score_cost_scale': calculator.config.score_cost_scale,
            'probe_offsets': calculator.config.probe_offsets,
            'external_supply': 'vertical_corridor_v1',
        },
        'overviews': [path.name for path in overview_paths],
        'scenes': scenes,
    }
    report_path = output_dir / 'report.json'
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps({
        'output_dir': str(output_dir),
        'overviews': [str(path) for path in overview_paths],
        'report': str(report_path),
        'scene_count': len(scenes),
    }, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
