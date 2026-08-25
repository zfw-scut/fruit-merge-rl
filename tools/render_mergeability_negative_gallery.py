"""把负可合成性变化快照渲染为一步投放前后对照画廊。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


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


DATASET_FORMAT = 'daxigua_mergeability_negative_transition_dataset'
DATASET_VERSION = 1
SEVERITY_COLORS = {
    'slight': '#ca8a04',
    'moderate': '#ea580c',
    'severe': '#dc2626',
    'extreme': '#991b1b',
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='渲染可合成性负变化的一步投放前后画廊。'
    )
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dpi', type=int, default=145)
    parser.add_argument('--progress-interval', type=int, default=16)
    return parser.parse_args(argv)


def _load_dataset(path):
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility.
        payload = torch.load(path, map_location='cpu')
    if not isinstance(payload, dict):
        raise ValueError('negative transition dataset must be a dict')
    if payload.get('format') != DATASET_FORMAT:
        raise ValueError('unsupported negative transition dataset format')
    if int(payload.get('format_version', -1)) != DATASET_VERSION:
        raise ValueError('unsupported negative transition dataset version')
    if not isinstance(payload.get('samples'), list):
        raise ValueError('negative transition samples are missing')
    return payload


def _active_fruits(scene):
    rows = torch.nonzero(scene['active'], as_tuple=False).flatten().tolist()
    fruits = []
    for slot in rows:
        fruits.append({
            'slot': int(slot),
            'id': int(scene['fruit_ids'][slot].item()),
            'level': int(scene['levels'][slot].item()),
            'x': float(scene['positions'][slot, 0].item()),
            'y': float(scene['positions'][slot, 1].item()),
            'radius': float(scene['physics_radii'][slot].item()),
            'mergeability': float(scene['mergeability_score'][slot].item()),
        })
    return fruits


def _contribution_map(scene):
    return {
        fruit['id']: (
            fruit['mergeability'] * math.pi * fruit['radius'] ** 2
        )
        for fruit in _active_fruits(scene)
    }


def _transition_analysis(sample):
    before = _contribution_map(sample['before'])
    after = _contribution_map(sample['after'])
    before_ids = set(before)
    after_ids = set(after)
    common = before_ids & after_ids
    disappeared = before_ids - after_ids
    created = after_ids - before_ids
    persisted_delta = sum(after[key] - before[key] for key in common)
    disappeared_delta = -sum(before[key] for key in disappeared)
    created_delta = sum(after[key] for key in created)
    common_losses = sorted(
        (
            (after[key] - before[key], key)
            for key in common if after[key] < before[key]
        ),
        key=lambda item: item[0],
    )
    return {
        'before_ids': before_ids,
        'after_ids': after_ids,
        'disappeared_ids': disappeared,
        'created_ids': created,
        'top_loss_ids': {key for _, key in common_losses[:5]},
        'persisted_delta': persisted_delta,
        'disappeared_delta': disappeared_delta,
        'created_delta': created_delta,
        'common_losses': common_losses,
    }


def _draw_board(
        axis,
        scene,
        config,
        *,
        title,
        role,
        analysis,
        drop_x=None,
        dropped_level=None,
        dropped_id=None):
    width = float(config['board_width'])
    height = float(config['board_height'])
    wall = float(config['wall_width'])
    spawn_y = float(config['spawn_y'])
    axis.set_facecolor('#f7f1e5')
    axis.add_patch(Rectangle(
        (wall, 0), width - 2 * wall, height - wall,
        fill=False, edgecolor='#4b3b2a', linewidth=2.0,
    ))
    axis.axhline(
        spawn_y, color='#c0392b', linewidth=1.0, linestyle='--', alpha=0.75
    )
    if role == 'before' and drop_x is not None:
        axis.plot(
            [drop_x, drop_x], [0.0, spawn_y],
            color='#2563eb', linewidth=1.8, linestyle=':', zorder=5,
        )
        axis.scatter(
            [drop_x], [spawn_y], marker='v', s=36,
            color='#2563eb', zorder=6,
        )
        axis.text(
            drop_x, max(12.0, spawn_y - 20.0),
            f'drop L{int(dropped_level)} #{int(dropped_id)}',
            ha='center', va='bottom', fontsize=7.0, color='#1d4ed8',
            weight='bold', zorder=6,
        )

    color_map = plt.get_cmap('RdYlGn')
    for fruit in _active_fruits(scene):
        fruit_id = fruit['id']
        if role == 'before' and fruit_id in analysis['disappeared_ids']:
            edge = '#dc2626'
            linewidth = 3.0
        elif role == 'after' and fruit_id in analysis['created_ids']:
            edge = '#16a34a'
            linewidth = 3.0
        elif fruit_id in analysis['top_loss_ids']:
            edge = '#f97316'
            linewidth = 2.6
        else:
            edge = '#ffffff'
            linewidth = 0.9
        fill = color_map(max(0.0, min(1.0, fruit['mergeability'])))
        axis.add_patch(Circle(
            (fruit['x'], fruit['y']), fruit['radius'],
            facecolor=fill, edgecolor=edge, linewidth=linewidth,
            alpha=0.92, zorder=3,
        ))
        axis.text(
            fruit['x'], fruit['y'],
            f"L{fruit['level']} #{fruit_id}\nM {fruit['mergeability']:.2f}",
            ha='center', va='center', fontsize=5.8,
            color='#172033', weight='bold', zorder=4,
        )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=10.5, weight='bold', pad=5)


def _short_ids(values, limit=8):
    ordered = sorted(int(value) for value in values)
    text = ', '.join(f'#{value}' for value in ordered[:limit])
    if len(ordered) > limit:
        text += f', +{len(ordered) - limit}'
    return text or '-'


def _merge_lines(sample):
    count = int(sample['merge_count'])
    events = sample['merge_events']
    rows = []
    for index in range(count):
        source = int(events['source_levels'][index].item())
        new = int(events['new_levels'][index].item())
        rows.append(f'L{source}+L{source}->' + (f'L{new}' if new else 'clear'))
    return ', '.join(rows[:5]) + (f', +{len(rows) - 5}' if len(rows) > 5 else '')


def _draw_info(axis, sample, band, analysis):
    axis.axis('off')
    accent = SEVERITY_COLORS.get(band['key'], '#991b1b')
    axis.text(
        0.0, 0.99,
        f"{band['key'].upper()} NEGATIVE CHANGE\n"
        f"scene delta = {float(sample['delta']):,.2f}",
        ha='left', va='top', transform=axis.transAxes,
        fontsize=15, color=accent, weight='bold', linespacing=1.35,
    )
    before = sample['before']
    after = sample['after']
    lines = [
        f"Episode: {int(sample['episode_id'])}",
        f"Environment: {int(sample['environment_id'])}",
        f"Episode drop: {int(sample['episode_drop'])}",
        f"Batch decision step: {int(sample['decision_step'])}",
        '',
        f"Action: {int(sample['action_index'])}",
        f"Dropped: L{int(sample['dropped_level'])} "
        f"#{int(sample['dropped_fruit_id'])}",
        f"Drop x: {float(sample['drop_x']):.2f}",
        f"Score delta: {int(sample['score_delta']):+,d}",
        f"Merges: {int(sample['merge_count'])}",
        f"Terminal after drop: {bool(sample['terminal'])}",
        '',
        'Scene mergeability',
        f"  {float(sample['before_scene_value']):,.2f}",
        f"→ {float(sample['after_scene_value']):,.2f}",
        f"  Δ {float(sample['delta']):+,.2f}",
        '',
        'Contribution decomposition',
        f"  persisted {analysis['persisted_delta']:+,.2f}",
        f"  disappeared {analysis['disappeared_delta']:+,.2f}",
        f"  created {analysis['created_delta']:+,.2f}",
        '',
        f"Fruits: {int(before['fruit_count'])} → {int(after['fruit_count'])}",
        f"Disappeared IDs: {_short_ids(analysis['disappeared_ids'])}",
        f"Created IDs: {_short_ids(analysis['created_ids'])}",
        f"Merge chain: {_merge_lines(sample) or '-'}",
    ]
    axis.text(
        0.0, 0.78, '\n'.join(lines),
        ha='left', va='top', transform=axis.transAxes,
        fontsize=8.6, family='monospace', color='#312a25',
        linespacing=1.32,
    )
    axis.text(
        0.0, 0.015,
        'fill: raw mergeability\n'
        'red before edge: disappeared ID\n'
        'green after edge: created ID\n'
        'orange edge: largest persisted losses',
        ha='left', va='bottom', transform=axis.transAxes,
        fontsize=8.2, color='#64748b', linespacing=1.35,
    )


def render_sample(sample, band, config, output_path, *, dpi):
    analysis = _transition_analysis(sample)
    figure, axes = plt.subplots(
        1, 3, figsize=(11.2, 8.6), squeeze=False,
        gridspec_kw={'width_ratios': (1.0, 1.0, 0.86)},
    )
    _draw_board(
        axes[0][0], sample['before'], config,
        title=(
            f"Before · total={float(sample['before_scene_value']):,.0f} · "
            f"fruits={int(sample['before']['fruit_count'])}"
        ),
        role='before', analysis=analysis,
        drop_x=float(sample['drop_x']),
        dropped_level=int(sample['dropped_level']),
        dropped_id=int(sample['dropped_fruit_id']),
    )
    _draw_board(
        axes[0][1], sample['after'], config,
        title=(
            f"After stable · total={float(sample['after_scene_value']):,.0f} · "
            f"fruits={int(sample['after']['fruit_count'])}"
        ),
        role='after', analysis=analysis,
    )
    _draw_info(axes[0][2], sample, band, analysis)
    color_bar = figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=plt.get_cmap('RdYlGn')),
        ax=axes[0][:2].tolist(), orientation='horizontal',
        fraction=0.025, pad=0.025,
    )
    color_bar.set_label('raw mergeability: low → high', fontsize=8)
    figure.suptitle(
        'One-drop negative mergeability transition',
        fontsize=13, weight='bold', y=0.995,
    )
    figure.subplots_adjust(left=0.035, right=0.98, top=0.95, bottom=0.06, wspace=0.13)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches='tight')
    plt.close(figure)
    return analysis


def _draw_overview(payload, samples_by_band, output_path):
    config = payload['manifest']['simulator_config']
    bands = payload['severity_bands']
    figure, axes = plt.subplots(
        len(bands), 2, figsize=(8.2, 7.0 * len(bands)), squeeze=False
    )
    for row, band in enumerate(bands):
        samples = samples_by_band[int(band['code'])]
        sample = samples[len(samples) // 2]
        analysis = _transition_analysis(sample)
        _draw_board(
            axes[row][0], sample['before'], config,
            title=(
                f"{band['key'].upper()} before · "
                f"Δ={float(sample['delta']):,.0f}"
            ),
            role='before', analysis=analysis,
            drop_x=float(sample['drop_x']),
            dropped_level=int(sample['dropped_level']),
            dropped_id=int(sample['dropped_fruit_id']),
        )
        _draw_board(
            axes[row][1], sample['after'], config,
            title=f"{band['key'].upper()} after stable",
            role='after', analysis=analysis,
        )
    figure.suptitle(
        'Negative mergeability transition severity overview',
        fontsize=15, weight='bold', y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.992))
    figure.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(figure)


def render(dataset_path, output_dir, *, dpi=145, progress_interval=16):
    payload = _load_dataset(dataset_path)
    samples = payload['samples']
    bands = payload['severity_bands']
    config = payload['manifest'].get('simulator_config')
    if not isinstance(config, dict):
        raise ValueError('dataset misses simulator_config')
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'gallery output is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    band_by_code = {int(band['code']): band for band in bands}
    samples_by_band = {code: [] for code in band_by_code}
    for sample in samples:
        samples_by_band[int(sample['severity_code'])].append(sample)
    for values in samples_by_band.values():
        values.sort(key=lambda item: abs(float(item['delta'])))

    rows = []
    rendered = 0
    for code in sorted(samples_by_band):
        band = band_by_code[code]
        directory = output_dir / f"{code + 1:02d}_{band['key']}"
        directory.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples_by_band[code], 1):
            magnitude = round(abs(float(sample['delta'])))
            filename = f'sample_{index:03d}_neg_{magnitude:07d}.png'
            path = directory / filename
            analysis = render_sample(sample, band, config, path, dpi=dpi)
            relative = path.relative_to(output_dir).as_posix()
            rows.append({
                'file': relative,
                'severity_code': code,
                'severity_key': band['key'],
                'severity_label': band['label'],
                'delta': float(sample['delta']),
                'before_scene_value': float(sample['before_scene_value']),
                'after_scene_value': float(sample['after_scene_value']),
                'episode_id': int(sample['episode_id']),
                'episode_seed': int(sample['episode_seed']),
                'episode_drop': int(sample['episode_drop']),
                'action_index': int(sample['action_index']),
                'dropped_level': int(sample['dropped_level']),
                'dropped_fruit_id': int(sample['dropped_fruit_id']),
                'drop_x': float(sample['drop_x']),
                'merge_count': int(sample['merge_count']),
                'score_delta': int(sample['score_delta']),
                'terminal': bool(sample['terminal']),
                'persisted_delta': float(analysis['persisted_delta']),
                'disappeared_delta': float(analysis['disappeared_delta']),
                'created_delta': float(analysis['created_delta']),
                'disappeared_count': len(analysis['disappeared_ids']),
                'created_count': len(analysis['created_ids']),
            })
            rendered += 1
            if rendered % int(progress_interval) == 0:
                print(json.dumps({
                    'phase': 'render', 'rendered': rendered,
                    'total': len(samples),
                }), flush=True)

    overview_path = output_dir / 'overview.png'
    _draw_overview(payload, samples_by_band, overview_path)
    fieldnames = list(rows[0]) if rows else []
    with (output_dir / 'index.csv').open(
            'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        'format_version': 1,
        'purpose': 'mergeability_negative_transition_human_gallery',
        'source_dataset': str(Path(dataset_path).resolve()),
        'scene_value_definition': payload['manifest'].get(
            'scene_value_definition'
        ),
        'comparison_semantics': 'one real drop: stable before -> stable after',
        'severity_bands': bands,
        'image_count': len(rows),
        'images_per_band': {
            band_by_code[code]['key']: len(samples_by_band[code])
            for code in sorted(samples_by_band)
        },
        'overview': overview_path.name,
        'images': rows,
    }
    (output_dir / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps({
        'output_dir': str(output_dir),
        'image_count': len(rows),
        'overview': str(overview_path),
        'report': str(output_dir / 'report.json'),
    }, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None):
    args = parse_args(argv)
    if int(args.dpi) <= 0 or int(args.progress_interval) <= 0:
        raise ValueError('dpi and progress-interval must be positive')
    output_dir = args.output_dir or args.dataset.parent / 'gallery'
    render(
        args.dataset, output_dir,
        dpi=args.dpi, progress_interval=args.progress_interval,
    )


if __name__ == '__main__':
    main()
